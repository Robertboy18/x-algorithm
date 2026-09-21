# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
"""CUDA kernels used by ``xrex``.

``xrex`` needs three generic array primitives that xAI maintains as CUDA
kernels: an Adler-32 checksum (checkpoint integrity), a 1-D int32 dedupe
(the training embedding compressor) and a batched top-k (the retrieval
serving path). This directory is the single implementation of all three.

One more kernel lives here with a different contract: ``async_emb``
(pipelined embedding communication) has NO reference path — it raises a loud,
named ImportError when its compiled extension is absent, and its callers
treat that as "the feature is off" (``use_async_emb``).

Each of the three primitive kernels is a subpackage with the same two-layer
shape:

* a **reference implementation in pure JAX (or zlib) that always works** — no
  compiler, no CUDA, no extension module. This is the path that runs unless a
  compiled extension is present, and it is what an installation without
  ``nvcc`` gets;
* the **CUDA/C++ source** under ``<kernel>/src/``, with the shared XLA-FFI
  helpers under ``xla_utils/``. Each subpackage tries to import a nanobind
  extension named after the file (``adler32_api``, ``unique_api``,
  ``top_k_by_key_api``) at import time. When the import succeeds the kernel
  is registered as an XLA FFI target and used instead of the reference; when
  it fails (``ModuleNotFoundError``) the reference stands.

  All three are built: each ``<kernel>/src/BUILD`` feeds the
  ``xrex-cuda-kernels`` wheel and the loader imports
  ``xrex_cuda_kernels.<api>``, so an environment with the package runs the
  compiled kernel and one without it runs the reference — and logs a warning
  saying so, because a venv quietly on the reference path is the failure mode
  this layout exists to prevent. ``wheel_inventory_test.py`` fails if a kernel
  is added here without its build, its wheel entry and its loader import.

The FFI target names are prefixed ``xrex_``.

Kernels with a compiled-only contract (``async_emb``, ``fa3``) import their
extension as ``xrex_cuda_kernels.<kernel>_api`` at module import time. Only a
missing extension becomes that kernel's named ``ImportError``; a
present-but-broken one raises its own error. The ``xrex-cuda-kernels`` package
is defined by ``xrex/cuda/wheel/BUILD``, a bazel wheel of those extensions.
Installing the ``xrex/cuda/wheel`` directory with ``pip``/``uv`` runs that
bazel build through the PEP-517 backend in ``wheel/backend.py`` (needs
bazelisk and git; the CUDA toolchain is fetched hermetically). An index may
also carry prebuilt wheels of the same target.
"""
