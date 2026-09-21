# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from xrex.data.recsys.ads_late_window import (
    ADS_LATE_WINDOW_TWIN_HEAD_INDICES,
    ADS_LATE_WINDOW_TWIN_SUPPRESSOR,
)
from xrex.data.recsys.constants import (
    CLICK_ACTION_INDEX,
    CLICK_CONDITIONED_ACTION_INDICES,
    VIEW_THROUGH_ACTION_INDICES,
)

FRESH_STREAM_ID = 0
EARLY_RELABEL_STREAM_ID = 1
LATE_SPLIT_STREAM_ID = 2


def _head_mask(indices: Sequence[int], num_actions: int) -> jax.Array:
    return jnp.zeros(num_actions).at[jnp.array(indices)].set(1.0)


def ads_head_masking_factor(
    targets: jax.Array, source_id: jax.Array, num_actions: int
) -> jax.Array:
    ct = _head_mask(CLICK_CONDITIONED_ACTION_INDICES, num_actions)
    vt = _head_mask(VIEW_THROUGH_ACTION_INDICES, num_actions)
    twins = _head_mask(ADS_LATE_WINDOW_TWIN_HEAD_INDICES, num_actions)
    engagement = 1 - ct - vt - twins

    s = source_id[:, :, None]
    fresh = (s == FRESH_STREAM_ID).astype(jnp.float32)
    late = (s == LATE_SPLIT_STREAM_ID).astype(jnp.float32)
    early = 1 - fresh - late
    click = targets[:, :, CLICK_ACTION_INDEX][:, :, None].astype(jnp.float32)

    suppressor_index = list(range(num_actions))
    for twin, suppressor in ADS_LATE_WINDOW_TWIN_SUPPRESSOR.items():
        suppressor_index[twin] = suppressor
    no_early_counterpart = 1 - targets[:, :, jnp.array(suppressor_index)].astype(jnp.float32)

    return (
        engagement * fresh
        + ct * early * click
        + vt * early * (1 - click)
        + twins * late * click * no_early_counterpart
    )


def ads_late_window_no_early_slice(
    mask: jax.Array,
    raw_targets: jax.Array,
    click_mask: jax.Array,
    sample_source: jax.Array | None,
) -> jax.Array:
    late = (
        jnp.zeros_like(mask)
        if sample_source is None
        else (sample_source == LATE_SPLIT_STREAM_ID).astype(mask.dtype)
    )
    suppressors = jnp.array(sorted(set(ADS_LATE_WINDOW_TWIN_SUPPRESSOR.values())))
    no_early = jnp.prod(1 - raw_targets[:, :, suppressors].astype(mask.dtype), axis=-1)
    return mask * late * click_mask * no_early
