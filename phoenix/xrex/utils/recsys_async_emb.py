# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import shard_map
from jax.sharding import PartitionSpec as P

if TYPE_CHECKING:
    from xrex.cuda.async_emb import (
        async_emb,
    )


def lookup_start(
    context_handle: async_emb.AsyncEmbContextHandle,
    token_ids: jax.Array,
    table: jax.Array,
    gate: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    from xrex.cuda.async_emb import (
        async_emb,
    )

    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(P(context_handle.data_axis, None), P(None, context_handle.table_axis), P()),
        out_specs=(P(None, context_handle.table_axis), P(context_handle.data_axis, None)),
        check_vma=False,
    )
    def start(
        token_ids: jax.Array, table: jax.Array, gate: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        table_out, lookup_pin = async_emb.lookup_start(token_ids, table, gate, context_handle)
        return table_out, lookup_pin[None, :]

    return start(token_ids, table, gate)


def lookup_done(
    context_handle: async_emb.AsyncEmbContextHandle, lookup_pin: jax.Array
) -> jax.Array:
    from xrex.cuda.async_emb import (
        async_emb,
    )

    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(P(context_handle.data_axis, None),),
        out_specs=P(context_handle.data_axis, None),
        check_vma=False,
    )
    def done(lookup_pin: jax.Array) -> jax.Array:
        return async_emb.lookup_done(lookup_pin, context_handle)

    return done(lookup_pin)


def depend(
    context_handle: async_emb.AsyncEmbContextHandle, x: jax.Array, pin: jax.Array
) -> jax.Array:
    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(P(context_handle.data_axis), P()),
        out_specs=P(context_handle.data_axis),
        check_vma=False,
    )
    def add_pin(x: jax.Array, pin: jax.Array) -> jax.Array:
        return x.at[(0,) * x.ndim].add(pin.astype(x.dtype)[0])

    return add_pin(x, pin)


def stage_update(
    context_handle: async_emb.AsyncEmbContextHandle,
    grads: jax.Array,
    segment_ids: jax.Array,
    unique_tokens: jax.Array,
    pending: jax.Array,
    gate: jax.Array,
) -> jax.Array:
    from xrex.cuda.async_emb import (
        async_emb,
    )

    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(
            P(context_handle.data_axis, None),
            P(context_handle.data_axis),
            P(),
            P(),
            P(),
        ),
        out_specs=P(context_handle.data_axis, None),
        check_vma=False,
    )
    def stage(
        grads: jax.Array,
        segment_ids: jax.Array,
        unique_tokens: jax.Array,
        pending: jax.Array,
        gate: jax.Array,
    ) -> jax.Array:
        pin = async_emb.stage_update(
            grads, segment_ids, unique_tokens, pending, gate, context_handle
        )
        return pin[None, :]

    return stage(grads, segment_ids, unique_tokens, pending, gate)
