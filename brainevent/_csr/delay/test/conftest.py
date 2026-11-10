# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Keep delay dtype coverage scoped to each test and large allocations opt-in."""

import pytest

from brainevent._test_util import jax_x64_enabled


def pytest_addoption(parser):
    parser.addoption(
        "--run-large-delay-tests", action="store_true", default=False,
        help="run delay tests with more than 2**31 edges (about 20-30 GB GPU memory)",
    )


@pytest.fixture(autouse=True)
def _delay_x64(_jax_x64_state_is_restored):
    with jax_x64_enabled():
        yield
