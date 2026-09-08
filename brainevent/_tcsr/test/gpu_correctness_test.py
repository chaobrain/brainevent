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

"""Compare GPU CSR and TCSR event-driven products."""

from __future__ import annotations

from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR
from brainevent._event import BinaryArray
from brainevent._tcsr.main import TCSR


pytestmark = pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="requires a JAX GPU backend",
)

_SIZE = 10_000
_DENSITY = 1e-2
_BATCH_SIZE = 4
_CONNECTIONS_PER_ROW = int(_SIZE * _DENSITY)
_NNZ = _SIZE * _CONNECTIONS_PER_ROW
_RTOL = 1e-5
_ATOL = 1e-5
_SMALL_SHAPE = (37, 53)
_BINARY_BACKEND_NAMES = (
    "CSR cusparse",
    "CSR cuda_raw",
    "TCSR cuda_raw",
    "TCSR jax",
    "CSR jax_raw",
)


def _make_case() -> tuple[jax.Array, ...]:
    """Create one deterministic, exactly one-percent-dense CSR workload."""
    rows = np.arange(_SIZE, dtype=np.int64)[:, None]
    offsets = np.arange(_CONNECTIONS_PER_ROW, dtype=np.int64)[None, :]
    indices = np.sort((rows * 131 + offsets * 97) % _SIZE, axis=1)
    indptr = np.arange(0, _NNZ + 1, _CONNECTIONS_PER_ROW, dtype=np.int32)

    rng = np.random.default_rng(20260908)
    data = rng.uniform(-1.0, 1.0, size=_NNZ).astype(np.float32)
    vector = rng.random(_SIZE) < 0.1
    matrix = rng.random((_BATCH_SIZE, _SIZE)) < 0.1
    return (
        jnp.asarray(data),
        jnp.asarray(indices.reshape(-1), dtype=jnp.int32),
        jnp.asarray(indptr),
        jnp.asarray(vector, dtype=jnp.bool_),
        jnp.asarray(matrix, dtype=jnp.bool_),
    )


def _assert_matches(
    actual: Any,
    expected: Any,
    *,
    backend: str,
    operation: str,
) -> None:
    """Compare every result element and identify the failing execution path."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{backend} {operation} differs from CSR cuSPARSE",
    )


def _expanded_weights(data: jax.Array, nnz: int) -> np.ndarray:
    """Expand compact homogeneous weights for independent references."""
    weights = np.asarray(data)
    if weights.size == 1:
        return np.full(nnz, weights.item(), dtype=weights.dtype)
    return weights


def _csr_reference(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    operand: jax.Array,
    *,
    shape: tuple[int, int],
    reverse: bool,
    binary: bool,
) -> np.ndarray:
    """Evaluate sparse products independently with NumPy scatter-adds."""
    host_indices = np.asarray(indices)
    host_indptr = np.asarray(indptr)
    host_operand = np.asarray(operand)
    weights = _expanded_weights(data, host_indices.size)
    row_ids = np.repeat(
        np.arange(shape[0], dtype=np.int32),
        np.diff(host_indptr),
    )
    values = host_operand > 0 if binary else host_operand

    if reverse:
        if values.ndim == 1:
            result = np.zeros(shape[1], dtype=weights.dtype)
            np.add.at(result, host_indices, weights * values[row_ids])
            return result
        result = np.zeros((values.shape[0], shape[1]), dtype=weights.dtype)
        for batch_index in range(values.shape[0]):
            np.add.at(
                result[batch_index],
                host_indices,
                weights * values[batch_index, row_ids],
            )
        return result

    if values.ndim == 1:
        result = np.zeros(shape[0], dtype=weights.dtype)
        np.add.at(result, row_ids, weights * values[host_indices])
        return result
    result = np.zeros((shape[0], values.shape[1]), dtype=weights.dtype)
    for batch_index in range(values.shape[1]):
        np.add.at(
            result[:, batch_index],
            row_ids,
            weights * values[host_indices, batch_index],
        )
    return result


def _make_csr_case(
    shape: tuple[int, int],
    row_lengths: np.ndarray,
    *,
    homogeneous: bool,
    seed: int,
    value_dtype: Any = np.float32,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create deterministic sorted CSR arrays with requested row lengths."""
    if row_lengths.shape != (shape[0],):
        raise ValueError("row_lengths must contain one value per matrix row")
    rng = np.random.default_rng(seed)
    host_indices = np.concatenate(
        [
            np.sort(rng.choice(shape[1], int(length), replace=False))
            for length in row_lengths
        ]
    ).astype(np.int32)
    host_indptr = np.concatenate(
        [np.zeros(1, dtype=np.int32), np.cumsum(row_lengths, dtype=np.int32)]
    )
    weight_size = 1 if homogeneous else host_indices.size
    host_data = rng.uniform(-1.0, 1.0, size=weight_size).astype(value_dtype)
    return (
        jnp.asarray(host_data),
        jnp.asarray(host_indices),
        jnp.asarray(host_indptr),
    )


def _make_events(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    seed: int,
) -> jax.Array:
    """Create deterministic events including inactive nonpositive values."""
    rng = np.random.default_rng(seed)
    if jnp.dtype(dtype) == jnp.dtype(jnp.bool_):
        values = rng.random(shape) < 0.3
    elif jnp.dtype(dtype) == jnp.dtype(jnp.int8):
        values = rng.integers(-1, 2, size=shape, dtype=np.int8)
    else:
        values = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return jnp.asarray(values, dtype=dtype)


def _binary_matrices(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Construct all five binary sparse execution paths."""
    source = CSR((data, indices, indptr), shape=shape)
    return {
        "CSR cusparse": CSR(
            (data, indices, indptr), shape=shape, backend="cusparse"
        ),
        "CSR cuda_raw": CSR(
            (data, indices, indptr), shape=shape, backend="cuda_raw"
        ),
        "TCSR cuda_raw": TCSR.from_sorted_csr(
            source, binary_backend="cuda_raw"
        ),
        "TCSR jax": TCSR.from_sorted_csr(source, binary_backend="jax"),
        "CSR jax_raw": CSR(
            (data, indices, indptr), shape=shape, backend="jax_raw"
        ),
    }


def _binary_product(sparse: Any, operand: jax.Array, *, reverse: bool) -> Any:
    """Apply a binary operand on the selected side of a sparse matrix."""
    events = BinaryArray(operand)
    return events @ sparse if reverse else sparse @ events


def _float_matrix(
    data: Any,
    indices: jax.Array,
    indptr: jax.Array,
    shape: tuple[int, int],
    *,
    backend: str,
) -> TCSR:
    """Construct a TCSR matrix for one floating-point backend."""
    source = CSR((data, indices, indptr), shape=shape)
    return TCSR.from_sorted_csr(source, backend=backend)


def _float_product(sparse: TCSR, operand: jax.Array, *, reverse: bool) -> Any:
    """Apply a floating operand on the selected side of a TCSR matrix."""
    return operand @ sparse if reverse else sparse @ operand


def _jax_dense_from_csr(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    shape: tuple[int, int],
) -> jax.Array:
    """Build a differentiable dense reference for small CSR cases."""
    row_ids = jnp.repeat(
        jnp.arange(shape[0], dtype=indptr.dtype),
        jnp.diff(indptr),
        total_repeat_length=indices.size,
    )
    weights = (
        jnp.broadcast_to(data[0], indices.shape)
        if data.size == 1
        else data
    )
    return jnp.zeros(shape, dtype=data.dtype).at[row_ids, indices].add(weights)


def test_gpu_csr_and_tcsr_mv_mm_match_cusparse() -> None:
    """Match CSR and TCSR MV/MM results against cuSPARSE on GPU."""
    data, indices, indptr, vector, matrix = _make_case()
    assert data.size == _NNZ
    assert data.size / (_SIZE * _SIZE) == _DENSITY
    reference_mv = _csr_reference(
        data,
        indices,
        indptr,
        vector,
        shape=(_SIZE, _SIZE),
        reverse=True,
        binary=True,
    )
    reference_mm = _csr_reference(
        data,
        indices,
        indptr,
        matrix,
        shape=(_SIZE, _SIZE),
        reverse=True,
        binary=True,
    )

    with jax.enable_x64():
        csr_cusparse = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
            backend="cusparse",
        )
        csr_cuda = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
            backend="cuda_raw",
        )
        csr_jax = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
            backend="jax_raw",
        )
        tcsr_source = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
        )
        tcsr_cuda = TCSR.from_sorted_csr(
            tcsr_source,
            binary_backend="cuda_raw",
        )
        tcsr_jax = TCSR.from_sorted_csr(
            tcsr_source,
            binary_backend="jax",
        )

        vector_events = BinaryArray(vector)
        matrix_events = BinaryArray(matrix)
        expected_mv = vector_events @ csr_cusparse
        expected_mm = matrix_events @ csr_cusparse
        results = {
            "CSR cuda_raw": (
                vector_events @ csr_cuda,
                matrix_events @ csr_cuda,
            ),
            "TCSR cuda_raw": (
                vector_events @ tcsr_cuda,
                matrix_events @ tcsr_cuda,
            ),
            "TCSR jax": (
                vector_events @ tcsr_jax,
                matrix_events @ tcsr_jax,
            ),
            "CSR jax_raw": (
                vector_events @ csr_jax,
                matrix_events @ csr_jax,
            ),
        }
        jax.block_until_ready((expected_mv, expected_mm, results))

    assert expected_mv.shape == (_SIZE,)
    assert expected_mm.shape == (_BATCH_SIZE, _SIZE)
    _assert_matches(
        expected_mv,
        reference_mv,
        backend="CSR cusparse",
        operation="MV",
    )
    _assert_matches(
        expected_mm,
        reference_mm,
        backend="CSR cusparse",
        operation="MM",
    )
    for backend, (actual_mv, actual_mm) in results.items():
        assert actual_mv.shape == expected_mv.shape
        assert actual_mm.shape == expected_mm.shape
        _assert_matches(
            actual_mv,
            expected_mv,
            backend=backend,
            operation="MV",
        )
        _assert_matches(
            actual_mm,
            expected_mm,
            backend=backend,
            operation="MM",
        )


@pytest.mark.parametrize(
    ("reverse", "rank", "homogeneous", "event_dtype", "value_dtype"),
    [
        (True, 1, False, jnp.bool_, np.float32),
        (False, 1, True, jnp.float32, np.float32),
        (True, 2, True, jnp.int8, np.float32),
        (False, 2, False, jnp.bool_, np.float64),
    ],
    ids=(
        "left-mv-heterogeneous-bool",
        "right-mv-homogeneous-float",
        "left-mm-homogeneous-int8",
        "right-mm-heterogeneous-bool",
    ),
)
@pytest.mark.parametrize(
    "backend",
    _BINARY_BACKEND_NAMES,
    ids=("csr-cusparse", "csr-cuda", "tcsr-cuda", "tcsr-jax", "csr-jax"),
)
def test_gpu_binary_five_backend_orthogonal_cases(
    backend: str,
    reverse: bool,
    rank: int,
    homogeneous: bool,
    event_dtype: Any,
    value_dtype: Any,
) -> None:
    """Match representative directions, ranks, weights, and event dtypes."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 7) % 54
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            _SMALL_SHAPE,
            row_lengths,
            homogeneous=homogeneous,
            seed=101 + rank + int(reverse),
            value_dtype=value_dtype,
        )
    operand_size = _SMALL_SHAPE[0] if reverse else _SMALL_SHAPE[1]
    operand_shape = (
        (operand_size,)
        if rank == 1
        else ((_BATCH_SIZE, operand_size) if reverse else (operand_size, _BATCH_SIZE))
    )
    operand = _make_events(operand_shape, event_dtype, seed=211 + rank)
    expected = _csr_reference(
        data,
        indices,
        indptr,
        operand,
        shape=_SMALL_SHAPE,
        reverse=reverse,
        binary=True,
    )

    with jax.enable_x64():
        matrices = _binary_matrices(data, indices, indptr, _SMALL_SHAPE)
        actual = _binary_product(matrices[backend], operand, reverse=reverse)
        jax.block_until_ready(actual)

    _assert_matches(
        actual,
        expected,
        backend=backend,
        operation="MV" if rank == 1 else "MM",
    )


def test_gpu_tcsr_transposed_view_matches_reference_in_four_quadrants() -> None:
    """Match both operand sides and ranks for transposed TCSR GPU views."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 11) % 54
    data, indices, indptr = _make_csr_case(
        _SMALL_SHAPE,
        row_lengths,
        homogeneous=False,
        seed=307,
    )

    with jax.enable_x64():
        source = CSR((data, indices, indptr), shape=_SMALL_SHAPE)
        matrices = {
            "TCSR cuda_raw transpose": TCSR.from_sorted_csr(
                source, binary_backend="cuda_raw"
            ).T,
            "TCSR jax transpose": TCSR.from_sorted_csr(
                source, binary_backend="jax"
            ).T,
        }
        results: list[tuple[str, str, Any, np.ndarray]] = []
        for reverse in (True, False):
            for rank in (1, 2):
                view_shape = _SMALL_SHAPE[::-1]
                operand_size = view_shape[0] if reverse else view_shape[1]
                operand_shape = (
                    (operand_size,)
                    if rank == 1
                    else (
                        (_BATCH_SIZE, operand_size)
                        if reverse
                        else (operand_size, _BATCH_SIZE)
                    )
                )
                operand = _make_events(
                    operand_shape,
                    jnp.bool_,
                    seed=401 + rank + int(reverse),
                )
                reference_operand = operand if rank == 1 else operand.T
                expected = _csr_reference(
                    data,
                    indices,
                    indptr,
                    reference_operand,
                    shape=_SMALL_SHAPE,
                    reverse=not reverse,
                    binary=True,
                )
                if rank == 2:
                    expected = expected.T
                for backend, sparse in matrices.items():
                    actual = _binary_product(sparse, operand, reverse=reverse)
                    results.append(
                        (
                            backend,
                            f"{'left' if reverse else 'right'}-{'MV' if rank == 1 else 'MM'}",
                            actual,
                            expected,
                        )
                    )
        jax.block_until_ready([actual for _, _, actual, _ in results])

    for backend, operation, actual, expected in results:
        _assert_matches(
            actual,
            expected,
            backend=backend,
            operation=operation,
        )


def test_gpu_binary_ragged_rows_and_dispatch_boundaries() -> None:
    """Match empty, ragged, and CUDA dispatch-boundary rows."""
    shape = (13, 1024)
    row_lengths = np.asarray(
        [0, 1, 15, 16, 31, 32, 33, 511, 512, 513, 7, 64, 1024],
        dtype=np.int32,
    )
    data, indices, indptr = _make_csr_case(
        shape,
        row_lengths,
        homogeneous=False,
        seed=503,
    )

    cases: list[tuple[str, jax.Array, bool]] = []
    for reverse in (True, False):
        operand_size = shape[0] if reverse else shape[1]
        cases.extend(
            [
                (
                    f"{'left' if reverse else 'right'}-MV-ragged",
                    _make_events((operand_size,), jnp.bool_, seed=601),
                    reverse,
                ),
                (
                    f"{'left' if reverse else 'right'}-MM-ragged",
                    _make_events(
                        (_BATCH_SIZE, operand_size)
                        if reverse
                        else (operand_size, _BATCH_SIZE),
                        jnp.bool_,
                        seed=607,
                    ),
                    reverse,
                ),
            ]
        )

    with jax.enable_x64():
        matrices = _binary_matrices(data, indices, indptr, shape)
        results = []
        for operation, operand, reverse in cases:
            expected = _csr_reference(
                data,
                indices,
                indptr,
                operand,
                shape=shape,
                reverse=reverse,
                binary=True,
            )
            for backend, sparse in matrices.items():
                actual = _binary_product(sparse, operand, reverse=reverse)
                results.append((backend, operation, actual, expected))
        jax.block_until_ready([actual for _, _, actual, _ in results])

    for backend, operation, actual, expected in results:
        _assert_matches(
            actual,
            expected,
            backend=backend,
            operation=operation,
        )


def test_gpu_binary_object_jit_and_vmap_match_direct_mm() -> None:
    """Match compiled object calls and mapped MV against direct MM."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 5) % 54
    data, indices, indptr = _make_csr_case(
        _SMALL_SHAPE,
        row_lengths,
        homogeneous=False,
        seed=701,
    )
    events = _make_events(
        (_BATCH_SIZE, _SMALL_SHAPE[0]),
        jnp.bool_,
        seed=709,
    )
    expected = _csr_reference(
        data,
        indices,
        indptr,
        events,
        shape=_SMALL_SHAPE,
        reverse=True,
        binary=True,
    )

    with jax.enable_x64():
        matrices = _binary_matrices(data, indices, indptr, _SMALL_SHAPE)
        results = []
        for backend, sparse in matrices.items():
            direct = BinaryArray(events) @ sparse
            compiled = jax.jit(
                lambda values, matrix=sparse: BinaryArray(values) @ matrix
            )(events)
            mapped = jax.jit(
                jax.vmap(
                    lambda values, matrix=sparse: BinaryArray(values) @ matrix
                )
            )(events)
            results.append((backend, compiled, mapped, direct))
        jax.block_until_ready(
            [
                (compiled, mapped, direct)
                for _, compiled, mapped, direct in results
            ]
        )

    for backend, compiled, mapped, direct in results:
        _assert_matches(
            compiled,
            expected,
            backend=backend,
            operation="jitted MM",
        )
        _assert_matches(
            mapped,
            direct,
            backend=backend,
            operation="vmapped MV",
        )


@pytest.mark.parametrize(
    ("backend", "reverse", "rank", "homogeneous", "value_dtype"),
    [
        ("cuda_raw", True, 1, False, np.float32),
        ("cuda_raw", False, 2, True, np.float64),
        ("jax_raw", True, 2, True, np.float64),
        ("jax_raw", False, 1, False, np.float32),
    ],
    ids=(
        "cuda-left-mv-heterogeneous-f32",
        "cuda-right-mm-homogeneous-f64",
        "jax-left-mm-homogeneous-f64",
        "jax-right-mv-heterogeneous-f32",
    ),
)
def test_gpu_tcsr_float_forward_orthogonal_cases(
    backend: str,
    reverse: bool,
    rank: int,
    homogeneous: bool,
    value_dtype: Any,
) -> None:
    """Match TCSR floating MV/MM branches against an independent reference."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 13) % 54
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            _SMALL_SHAPE,
            row_lengths,
            homogeneous=homogeneous,
            seed=809 + rank,
            value_dtype=value_dtype,
        )
        operand_size = _SMALL_SHAPE[0] if reverse else _SMALL_SHAPE[1]
        operand_shape = (
            (operand_size,)
            if rank == 1
            else (
                (_BATCH_SIZE, operand_size)
                if reverse
                else (operand_size, _BATCH_SIZE)
            )
        )
        rng = np.random.default_rng(821 + rank)
        operand = jnp.asarray(
            rng.uniform(-1.0, 1.0, size=operand_shape).astype(value_dtype)
        )
        expected = _csr_reference(
            data,
            indices,
            indptr,
            operand,
            shape=_SMALL_SHAPE,
            reverse=reverse,
            binary=False,
        )
        sparse = _float_matrix(
            data,
            indices,
            indptr,
            _SMALL_SHAPE,
            backend=backend,
        )
        actual = jax.jit(
            lambda values: _float_product(sparse, values, reverse=reverse)
        )(operand)
        jax.block_until_ready(actual)

    _assert_matches(
        actual,
        expected,
        backend=f"TCSR {backend}",
        operation="float MV" if rank == 1 else "float MM",
    )


@pytest.mark.parametrize(
    ("backend", "reverse", "rank", "homogeneous"),
    [
        ("cuda_raw", True, 1, False),
        ("cuda_raw", False, 2, True),
        ("jax_raw", True, 2, True),
        ("jax_raw", False, 1, False),
    ],
    ids=(
        "cuda-left-mv-heterogeneous",
        "cuda-right-mm-homogeneous",
        "jax-left-mm-homogeneous",
        "jax-right-mv-heterogeneous",
    ),
)
def test_gpu_tcsr_float_jvp_and_vjp_match_dense(
    backend: str,
    reverse: bool,
    rank: int,
    homogeneous: bool,
) -> None:
    """Match representative TCSR floating JVP and VJP results."""
    shape = (11, 17)
    row_lengths = (np.arange(shape[0], dtype=np.int32) * 3) % 10
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            shape,
            row_lengths,
            homogeneous=homogeneous,
            seed=907 + rank,
        )
        operand_size = shape[0] if reverse else shape[1]
        operand_shape = (
            (operand_size,)
            if rank == 1
            else (
                (_BATCH_SIZE, operand_size)
                if reverse
                else (operand_size, _BATCH_SIZE)
            )
        )
        operand = jnp.asarray(
            np.random.default_rng(919 + rank)
            .uniform(-1.0, 1.0, size=operand_shape)
            .astype(np.float32)
        )
        template = _float_matrix(
            data,
            indices,
            indptr,
            shape,
            backend=backend,
        )

        def sparse_function(weights, values):
            matrix = template.with_data(weights)
            return _float_product(matrix, values, reverse=reverse)

        def dense_function(weights, values):
            matrix = _jax_dense_from_csr(weights, indices, indptr, shape)
            lhs, rhs = (values, matrix) if reverse else (matrix, values)
            return jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)

        tangents = (jnp.full_like(data, 0.25), jnp.full_like(operand, -0.5))
        actual_primal, actual_tangent = jax.jit(
            lambda weights, values: jax.jvp(
                sparse_function,
                (weights, values),
                tangents,
            )
        )(data, operand)
        expected_primal, expected_tangent = jax.jit(
            lambda weights, values: jax.jvp(
                dense_function,
                (weights, values),
                tangents,
            )
        )(data, operand)
        cotangent = jnp.full_like(expected_primal, 0.75)

        def sparse_vjp(weights, values, cotangent_value):
            primal, pullback = jax.vjp(sparse_function, weights, values)
            return primal, *pullback(cotangent_value)

        def dense_vjp(weights, values, cotangent_value):
            primal, pullback = jax.vjp(dense_function, weights, values)
            return primal, *pullback(cotangent_value)

        actual_vjp = jax.jit(sparse_vjp)(data, operand, cotangent)
        expected_vjp = jax.jit(dense_vjp)(data, operand, cotangent)
        jax.block_until_ready(
            (
                actual_primal,
                actual_tangent,
                expected_primal,
                expected_tangent,
                actual_vjp,
                expected_vjp,
            )
        )

    _assert_matches(
        actual_primal,
        expected_primal,
        backend=f"TCSR {backend}",
        operation="float JVP primal",
    )
    _assert_matches(
        actual_tangent,
        expected_tangent,
        backend=f"TCSR {backend}",
        operation="float JVP tangent",
    )
    for name, actual, expected in zip(
        ("primal", "weight cotangent", "operand cotangent"),
        actual_vjp,
        expected_vjp,
    ):
        _assert_matches(
            actual,
            expected,
            backend=f"TCSR {backend}",
            operation=f"float VJP {name}",
        )


@pytest.mark.parametrize("backend", ("cuda_raw", "jax_raw"))
def test_gpu_tcsr_float_mv_mm_preserve_weight_units(backend: str) -> None:
    """Preserve weight units through TCSR floating MV and MM products."""
    shape = (7, 9)
    row_lengths = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            shape,
            row_lengths,
            homogeneous=False,
            seed=1009,
        )
        sparse = _float_matrix(
            data * u.mV,
            indices,
            indptr,
            shape,
            backend=backend,
        )
        vector = jnp.linspace(-1.0, 1.0, shape[0], dtype=jnp.float32)
        matrix = jnp.broadcast_to(vector, (_BATCH_SIZE, shape[0]))
        actual_mv = vector @ sparse
        actual_mm = matrix @ sparse
        jax.block_until_ready((actual_mv, actual_mm))

    expected_mv = _csr_reference(
        data,
        indices,
        indptr,
        vector,
        shape=shape,
        reverse=True,
        binary=False,
    )
    expected_mm = _csr_reference(
        data,
        indices,
        indptr,
        matrix,
        shape=shape,
        reverse=True,
        binary=False,
    )
    assert u.get_unit(actual_mv) == u.mV
    assert u.get_unit(actual_mm) == u.mV
    _assert_matches(
        u.get_mantissa(actual_mv),
        expected_mv,
        backend=f"TCSR {backend}",
        operation="unitful float MV",
    )
    _assert_matches(
        u.get_mantissa(actual_mm),
        expected_mm,
        backend=f"TCSR {backend}",
        operation="unitful float MM",
    )
