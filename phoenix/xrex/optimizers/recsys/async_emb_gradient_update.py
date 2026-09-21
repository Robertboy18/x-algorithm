# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

import jax

if TYPE_CHECKING:
    from xrex.cuda.async_emb import (
        async_emb,
    )


class AsyncEmbGradientUpdate(NamedTuple):
    unique_tokens: jax.Array
    grads: jax.Array
    segment_ids: jax.Array
    pending: jax.Array


@runtime_checkable
class AsyncEmbOptimizer(Protocol):
    def gradient_update_start(
        self,
        context: async_emb.AsyncEmbContextHandle,
        update: AsyncEmbGradientUpdate,
        table: jax.Array,
        state: Any,
        gate: jax.Array,
    ) -> tuple[tuple[jax.Array, ...], jax.Array, Any, dict[str, jax.Array]]:
        raise NotImplementedError(
            f"use_async_emb requires an embedding optimizer implementing "
            f"AsyncEmbOptimizer, and {type(self).__name__} has no fused table update"
        )

    def gradient_update_done(
        self, context: async_emb.AsyncEmbContextHandle, gate: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError(
            f"use_async_emb requires an embedding optimizer implementing "
            f"AsyncEmbOptimizer, and {type(self).__name__} has no fused table update"
        )
