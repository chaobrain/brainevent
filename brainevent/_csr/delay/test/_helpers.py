# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared backend selection and direct calls for delay operator tests."""

import jax
import pytest

from brainevent._csr.delay.delay import HybridWorkspace, WeightedHybridWorkspace, csr_delay_deliver_layout_p

GPU = pytest.mark.skipif(jax.default_backend() != "gpu", reason="CUDA required")
BACKENDS = [None, "jax_raw", pytest.param("cuda_raw", marks=GPU)]


def cuda_step(
    weights, indices, indptr, events, delay_count, delay_off, row_task_count,
    head, buffer, workspace, *, weighted=False, check_status=False,
):
    outputs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (buffer, *workspace))
    result = csr_delay_deliver_layout_p(
        buffer, events, weights, indices, indptr, delay_count, delay_off,
        head, row_task_count, *workspace, outs=outputs, weighted=weighted,
        check_status=check_status, backend="cuda_raw",
    )
    workspace_type = WeightedHybridWorkspace if weighted else HybridWorkspace
    return result[0], workspace_type(*result[1:])
