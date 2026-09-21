# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import functools
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from numpy import typing as npt

from xrex.models.layers import get_parameter


@functools.lru_cache(maxsize=2)
def _load_sid_decoder(
    path: str,
) -> tuple[npt.NDArray, tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    import safetensors.numpy

    tensors = safetensors.numpy.load_file(path)
    return tensors["stages"], (
        tensors["dec_w0"],
        tensors["dec_b0"],
        tensors["dec_w1"],
        tensors["dec_b1"],
    )


def reconstruct_entity_sid(
    sids: jnp.ndarray,
    target_dim: int,
    decoder_path: str,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    fprop_dtype: jnp.dtype,
    name_prefix: str,
) -> jnp.ndarray:
    stages_np, (w0_np, b0_np, w1_np, b1_np) = _load_sid_decoder(decoder_path)
    num_levels, codebook_size, _ = stages_np.shape
    assert sids.shape[-1] == num_levels, (
        f"post_sids have {sids.shape[-1]} levels but the decoder at {decoder_path!r} "
        f"expects {num_levels}; set sid_num_levels to match the artifact."
    )

    stages = jnp.asarray(stages_np)
    w0, b0 = jnp.asarray(w0_np), jnp.asarray(b0_np)
    w1, b1 = jnp.asarray(w1_np), jnp.asarray(b1_np)

    codes = sids.astype(jnp.int32) - 1
    missing = sids[..., 0] == 0
    codes = jnp.clip(codes, 0, codebook_size - 1)
    quant = stages[jnp.arange(num_levels), codes].sum(-2)
    h = jax.nn.relu(quant @ w0 + b0)
    recon = h @ w1 + b1
    recon = recon * jax.lax.rsqrt((recon**2).sum(-1, keepdims=True) + 1e-12)
    recon = jnp.where(missing[..., None], 0.0, recon)
    recon = jax.lax.stop_gradient(recon)

    recon_dim = recon.shape[-1]
    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
    proj = get_parameter(
        f"{name_prefix}_sid_recon_proj",
        [recon_dim, target_dim],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(recon_dim),
    )
    return jnp.dot(recon, proj).astype(fprop_dtype)
