# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test staged TCSR construction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSC as PlainCSC
from brainevent._csr.main import CSR as PlainCSR
from brainevent._tcsr import main
from brainevent._tcsr.main import TCSR


def test_default_constructor_requires_sorted_plain_csr() -> None:
    """Keep dense conversion and sorting outside the default constructor."""
    with pytest.raises(TypeError, match="sorted plain CSR"):
        TCSR(jnp.eye(2, dtype=jnp.float32))


def test_default_constructor_rejects_removed_trusted_keyword() -> None:
    """Require the factory name, rather than a flag, to express trust."""
    source = PlainCSR(
        (
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        ),
        shape=(1, 1),
    )

    with jax.enable_x64(), pytest.raises(TypeError, match="trusted"):
        TCSR(source, trusted=True)


def test_from_sorted_csr_enters_default_constructor_without_reordering() -> None:
    """Preserve canonical order when the caller establishes sortedness."""
    source = PlainCSR(
        (
            jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32),
            jnp.asarray([0, 2, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 3], dtype=jnp.int32),
        ),
        shape=(2, 3),
    )

    with jax.enable_x64():
        matrix = TCSR.from_sorted_csr(source, binary_backend="jax")

    np.testing.assert_array_equal(matrix.indices, [0, 2, 1])
    np.testing.assert_array_equal(matrix.data, [10.0, 20.0, 30.0])
    assert matrix.has_tcsc_mirror is False


def test_sorted_entry_points_do_not_call_sort_csr(monkeypatch) -> None:
    """Keep sorting completely outside trusted construction paths."""
    source = PlainCSR(
        (
            jnp.asarray([10.0, 20.0], dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2], dtype=jnp.int32),
        ),
        shape=(1, 2),
    )

    def fail_if_called(source: PlainCSR) -> PlainCSR:
        raise AssertionError(f"unexpected sort of {source!r}")

    monkeypatch.setattr(main, "sort_csr", fail_if_called)
    with jax.enable_x64():
        direct = TCSR(source)
        forwarded = TCSR.from_sorted_csr(source)

    np.testing.assert_array_equal(direct.data, source.data)
    np.testing.assert_array_equal(forwarded.data, source.data)


@pytest.mark.parametrize("factory", ["fromcsr", "fromdense", "fromcsc"])
def test_checked_factories_call_sort_csr_once(monkeypatch, factory: str) -> None:
    """Route every checked source through the single sorting stage once."""
    source = PlainCSR(
        (
            jnp.asarray([20.0, 10.0], dtype=jnp.float32),
            jnp.asarray([1, 0], dtype=jnp.int32),
            jnp.asarray([0, 2], dtype=jnp.int32),
        ),
        shape=(1, 2),
    )
    original_sort = main.sort_csr
    calls = 0

    def counted_sort(csr: PlainCSR) -> PlainCSR:
        nonlocal calls
        calls += 1
        return original_sort(csr)

    monkeypatch.setattr(main, "sort_csr", counted_sort)
    checked_source = {
        "fromcsr": source,
        "fromdense": source.todense(),
        "fromcsc": source.tocsc(),
    }[factory]

    with jax.enable_x64():
        matrix = getattr(TCSR, factory)(checked_source)

    assert calls == 1
    np.testing.assert_array_equal(matrix.indices, [0, 1])


def test_fromcsr_sorts_indices_and_corresponding_data() -> None:
    """Build checked TCSR storage with aligned canonical value order."""
    source = PlainCSR(
        (
            jnp.asarray([30.0, 10.0, 20.0], dtype=jnp.float32),
            jnp.asarray([2, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 3], dtype=jnp.int32),
        ),
        shape=(2, 3),
    )

    with jax.enable_x64():
        matrix = TCSR.fromcsr(source)
        dense = matrix.todense()

    np.testing.assert_array_equal(matrix.indices, [0, 2, 1])
    np.testing.assert_array_equal(matrix.data, [10.0, 30.0, 20.0])
    np.testing.assert_array_equal(
        dense,
        [[10.0, 0.0, 30.0], [0.0, 20.0, 0.0]],
    )


def test_fromdense_uses_checked_csr_construction() -> None:
    """Convert dense input through the checked CSR factory."""
    source = jnp.asarray(
        [[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]],
        dtype=jnp.float32,
    )

    with jax.enable_x64():
        matrix = TCSR.fromdense(source)
        dense = matrix.todense()

    np.testing.assert_array_equal(dense, source)


def test_fromcsc_uses_checked_csr_construction() -> None:
    """Convert CSC input through the same checked CSR factory."""
    expected = jnp.asarray(
        [[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]],
        dtype=jnp.float32,
    )

    with jax.enable_x64():
        source = PlainCSC.fromdense(expected)
        matrix = TCSR.fromcsc(source)
        dense = matrix.todense()

    np.testing.assert_array_equal(dense, expected)
