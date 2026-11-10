# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Scope explicit int64 metadata and float64 kernel tests to each test."""

import pytest

from brainevent._test_util import jax_x64_enabled


@pytest.fixture(autouse=True)
def _enable_x64_for_kernel_contracts():
    """Preserve int64 offsets and float64 values without leaking JAX state."""
    with jax_x64_enabled():
        yield
