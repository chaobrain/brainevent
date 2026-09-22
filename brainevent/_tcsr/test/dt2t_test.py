# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test TCSR per-slot diagonal expansion services."""

from contextlib import contextmanager

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR as PlainCSR
from brainevent._test_util import requires_gpu
from brainevent._tcsr import dt2t as dt2t_module
from brainevent._tcsr.dt2t import csrmv_dt2t, csrmm_dt2t
from brainevent._tcsr.main import TCSR


def _shape_of(dtype, shape=(2,)):
    return jax.ShapeDtypeStruct(shape, dtype)


def _recording_ffi_call(calls):
    def ffi_call(name, out_info, **ffi_kwargs):
        def call(*args, **kwargs):
            calls.append((name, out_info, ffi_kwargs, args, kwargs))
            return tuple(jnp.zeros(info.shape, info.dtype) for info in out_info)

        return call

    return ffi_call


@contextmanager
def _explicit_int64_allowed():
    previous_explicit = jax.config.jax_explicit_x64_dtypes
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_explicit_x64_dtypes", previous_explicit)


def _matrix(*, backend="jax_raw"):
    source = PlainCSR(
        (
            jnp.asarray([2.0, 3.0, 5.0, 7.0, 11.0], dtype=jnp.float32),
            jnp.asarray([1, 3, 0, 2, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 4, 5], dtype=jnp.int32),
        ),
        shape=(3, 4),
        backend=backend,
    )
    return TCSR.from_sorted_csr(source, backend=backend)


def _expected(y, w, indices, indptr, *, transpose):
    if transpose:
        return w * y[..., indices]
    rows = jnp.repeat(
        jnp.arange(indptr.size - 1, dtype=indptr.dtype),
        jnp.diff(indptr),
        total_repeat_length=indices.size,
    )
    return w * y[..., rows]


@pytest.mark.parametrize("transpose", [False, True])
def test_mv_matches_csr_shape_semantics_and_slot_order(transpose):
    with _explicit_int64_allowed():
        matrix = _matrix()
        y = jnp.arange(
            matrix._base_shape[1 if transpose else 0], dtype=jnp.float32
        ) + 1.0
        w = jnp.asarray([13.0, 17.0, 19.0, 23.0, 29.0], dtype=jnp.float32)

        result = csrmv_dt2t(
            y,
            w,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=transpose,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )
        expected = _expected(
            y,
            w,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            transpose=transpose,
        )

        assert result.shape == w.shape
        np.testing.assert_allclose(result, expected)
        assert matrix.has_tcsc_mirror is transpose
        if transpose:
            permutation = matrix._tcs_buffers.tcsc.permutation
            assert not np.array_equal(permutation, jnp.arange(w.size))


@pytest.mark.parametrize("transpose", [False, True])
def test_mm_keeps_bn_layout_and_matches_stacked_mv(transpose):
    with _explicit_int64_allowed():
        matrix = _matrix()
        batch = 3
        neuron_count = matrix._base_shape[1 if transpose else 0]
        y = jnp.arange(batch * neuron_count, dtype=jnp.float32).reshape(
            batch, neuron_count
        ) + 1.0
        w = jnp.arange(batch * matrix.nse, dtype=jnp.float32).reshape(
            batch, matrix.nse
        ) + 2.0

        result = csrmm_dt2t(
            y,
            w,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=transpose,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )
        expected = _expected(
            y,
            w,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            transpose=transpose,
        )
        stacked = jnp.stack(
            [
                csrmv_dt2t(
                    y[b],
                    w[b],
                    matrix._tcsr_indices,
                    matrix._tcsr_indptr,
                    shape=matrix._base_shape,
                    transpose=transpose,
                    buffers=matrix._tcs_buffers,
                    backend="jax_raw",
                )
                for b in range(batch)
            ]
        )

        assert result.shape == (batch, matrix.nse)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, stacked)


@pytest.mark.parametrize("transpose", [False, True])
def test_mv_batching_normalizes_mapped_axis_to_zero(transpose):
    with _explicit_int64_allowed():
        matrix = _matrix()
        outer = 3
        neuron_count = matrix._base_shape[1 if transpose else 0]
        y = jnp.arange(neuron_count * outer, dtype=jnp.float32).reshape(
            neuron_count, outer
        ) + 1.0
        w = jnp.arange(matrix.nse * outer, dtype=jnp.float32).reshape(
            matrix.nse, outer
        ) + 2.0

        call = lambda y_, w_: csrmv_dt2t(
            y_,
            w_,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=transpose,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )
        result = jax.vmap(call, in_axes=(1, 1), out_axes=0)(y, w)
        expected = jnp.stack([call(y[:, i], w[:, i]) for i in range(outer)])

        assert result.shape == (outer, matrix.nse)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("mapped_operand", ["both", "y", "w"])
def test_mv_batching_broadcasts_unmapped_operand(mapped_operand):
    with _explicit_int64_allowed():
        matrix = _matrix()
        outer = 2
        y = jnp.arange(outer * 3, dtype=jnp.float32).reshape(outer, 3) + 1.0
        w = jnp.arange(outer * matrix.nse, dtype=jnp.float32).reshape(
            outer, matrix.nse
        ) + 2.0
        y_arg = y if mapped_operand != "w" else y[0]
        w_arg = w if mapped_operand != "y" else w[0]
        in_axes = (
            0 if mapped_operand != "w" else None,
            0 if mapped_operand != "y" else None,
        )

        call = lambda y_, w_: csrmv_dt2t(
            y_,
            w_,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=False,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )
        result = jax.vmap(call, in_axes=in_axes, out_axes=0)(y_arg, w_arg)

        assert result.shape == (outer, matrix.nse)


@pytest.mark.parametrize("transpose", [False, True])
def test_mm_nested_batching_moves_mapped_axis_to_zero(transpose):
    with _explicit_int64_allowed():
        matrix = _matrix()
        outer, batch = 2, 3
        neuron_count = matrix._base_shape[1 if transpose else 0]
        y = jnp.arange(
            batch * outer * neuron_count, dtype=jnp.float32
        ).reshape(batch, outer, neuron_count) + 1.0
        w = jnp.arange(batch * outer * matrix.nse, dtype=jnp.float32).reshape(
            batch, outer, matrix.nse
        ) + 2.0

        call = lambda y_, w_: csrmm_dt2t(
            y_,
            w_,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=transpose,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )
        result = jax.vmap(call, in_axes=(1, 1), out_axes=0)(y, w)
        expected = jnp.stack([call(y[:, i], w[:, i]) for i in range(outer)])

        assert result.shape == (outer, batch, matrix.nse)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("service", [csrmv_dt2t, csrmm_dt2t])
@pytest.mark.parametrize("transpose", [False, True])
def test_jvp_matches_product_rule(service, transpose):
    with _explicit_int64_allowed():
        matrix = _matrix()
        neuron_count = matrix._base_shape[1 if transpose else 0]
        y = jnp.arange(neuron_count, dtype=jnp.float32) + 1.0
        w = jnp.arange(matrix.nse, dtype=jnp.float32) + 2.0
        if service is csrmm_dt2t:
            y = jnp.stack((y, y + 3.0))
            w = jnp.stack((w, w + 5.0))
        y_dot = jnp.full_like(y, 2.0)
        w_dot = jnp.full_like(w, 3.0)
        call = lambda y_, w_: service(
            y_,
            w_,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=transpose,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )

        result, tangent = jax.jvp(call, (y, w), (y_dot, w_dot))
        expanded_y = _expected(
            y, jnp.ones_like(w), matrix._tcsr_indices,
            matrix._tcsr_indptr, transpose=transpose,
        )
        expanded_y_dot = _expected(
            y_dot, jnp.ones_like(w), matrix._tcsr_indices,
            matrix._tcsr_indptr, transpose=transpose,
        )

        np.testing.assert_allclose(result, w * expanded_y)
        np.testing.assert_allclose(
            tangent, w_dot * expanded_y + w * expanded_y_dot
        )


@pytest.mark.parametrize("method,transpose", [("dt2t", False), ("dt2t_transposed", True)])
def test_tcsr_object_methods_match_current_view_csr_semantics(method, transpose):
    with _explicit_int64_allowed():
        matrix = _matrix()
        for view in (matrix, matrix.T):
            y = jnp.arange(view.shape[1 if transpose else 0], dtype=jnp.float32) + 1.0
            w = jnp.arange(view.nse, dtype=jnp.float32) + 2.0
            result = getattr(view, method)(y, w)
            expected = _expected(y, w, view.indices, view.indptr, transpose=transpose)

            assert result.shape == w.shape
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("service,rank", [(csrmv_dt2t, 1), (csrmm_dt2t, 2)])
def test_services_keep_weight_unit_and_drop_y_unit(service, rank):
    with _explicit_int64_allowed():
        matrix = _matrix()
        y = jnp.ones((3,), dtype=jnp.float32)
        w = jnp.ones((matrix.nse,), dtype=jnp.float32)
        if rank == 2:
            y = y[None, :]
            w = w[None, :]
        result = service(
            y * u.mV,
            w * u.siemens,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=False,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )

        assert u.get_unit(result) == u.siemens


def test_validation_fails_before_cuda_loading(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        dt2t_module, "load_cuda_file", lambda *args, **kwargs: loaded.append(args)
    )
    with _explicit_int64_allowed():
        matrix = _matrix()
        with pytest.raises(AssertionError, match="same dtype"):
            csrmv_dt2t(
                jnp.ones(3, dtype=jnp.float16),
                jnp.ones(matrix.nse, dtype=jnp.float32),
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                shape=matrix._base_shape,
                backend="cuda_raw",
            )
        with pytest.raises(AssertionError, match="Batch mismatch"):
            csrmm_dt2t(
                jnp.ones((2, 3), dtype=jnp.float32),
                jnp.ones((3, matrix.nse), dtype=jnp.float32),
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                shape=matrix._base_shape,
                backend="cuda_raw",
            )
    assert loaded == []


def test_validation_rejects_invalid_rank_shape_and_structure_dtype():
    with _explicit_int64_allowed():
        matrix = _matrix()
        with pytest.raises(AssertionError, match="both be 1D"):
            csrmv_dt2t(
                jnp.ones((1, 3), dtype=jnp.float32),
                jnp.ones(matrix.nse, dtype=jnp.float32),
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                shape=matrix._base_shape,
                backend="jax_raw",
            )
        with pytest.raises(AssertionError, match="neuron dimension"):
            csrmm_dt2t(
                jnp.ones((2, 4), dtype=jnp.float32),
                jnp.ones((2, matrix.nse), dtype=jnp.float32),
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                shape=matrix._base_shape,
                backend="jax_raw",
            )
        with pytest.raises(TypeError, match="indices must use int32"):
            csrmv_dt2t(
                jnp.ones(3, dtype=jnp.float32),
                jnp.ones(matrix.nse, dtype=jnp.float32),
                matrix._tcsr_indices.astype(jnp.int64),
                matrix._tcsr_indptr,
                shape=matrix._base_shape,
                backend="jax_raw",
            )
        with pytest.raises(TypeError, match="indptr must use int64"):
            csrmv_dt2t(
                jnp.ones(3, dtype=jnp.float32),
                jnp.ones(matrix.nse, dtype=jnp.float32),
                matrix._tcsr_indices,
                matrix._tcsr_indptr.astype(jnp.int32),
                shape=matrix._base_shape,
                backend="jax_raw",
            )


def test_transpose_service_without_owner_builds_invocation_local_mirror():
    with _explicit_int64_allowed():
        matrix = _matrix()
        y = jnp.arange(4, dtype=jnp.float32) + 1.0
        w = jnp.arange(matrix.nse, dtype=jnp.float32) + 2.0
        result = csrmv_dt2t(
            y,
            w,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=True,
            backend="jax_raw",
        )

        np.testing.assert_allclose(
            result,
            _expected(
                y, w, matrix._tcsr_indices, matrix._tcsr_indptr,
                transpose=True,
            ),
        )
        assert matrix.has_tcsc_mirror is False


def test_jit_transpose_route_materializes_and_reuses_owner_mirror():
    with _explicit_int64_allowed():
        matrix = _matrix()
        y = jnp.arange(4, dtype=jnp.float32) + 1.0
        w = jnp.arange(matrix.nse, dtype=jnp.float32) + 2.0

        call = jax.jit(
            lambda y_, w_: csrmv_dt2t(
                y_,
                w_,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                shape=matrix._base_shape,
                transpose=True,
                buffers=matrix._tcs_buffers,
                backend="jax_raw",
            )
        )
        result = call(y, w)
        mirror = matrix._tcs_buffers.tcsc
        repeated = call(y + 1.0, w)

        assert mirror is not None
        assert matrix._tcs_buffers.tcsc is mirror
        np.testing.assert_allclose(
            result,
            _expected(
                y,
                w,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                transpose=True,
            ),
        )
        np.testing.assert_allclose(
            repeated,
            _expected(
                y + 1.0,
                w,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                transpose=True,
            ),
        )


def test_jit_dynamic_transpose_structure_requires_prepared_mirror():
    with _explicit_int64_allowed():
        matrix = _matrix()
        y = jnp.arange(4, dtype=jnp.float32) + 1.0
        w = jnp.arange(matrix.nse, dtype=jnp.float32) + 2.0
        call = jax.jit(
            lambda indices, indptr: csrmv_dt2t(
                y,
                w,
                indices,
                indptr,
                shape=matrix._base_shape,
                transpose=True,
                backend="jax_raw",
            )
        )

        with pytest.raises(RuntimeError, match="must be materialized"):
            call(matrix._tcsr_indices, matrix._tcsr_indptr)


@pytest.mark.parametrize(
    "factory,prefix,rank",
    [
        (dt2t_module._csrmv_dt2t_cuda_kernel, "csrmv_dt2t", 1),
        (dt2t_module._csrmm_dt2t_cuda_kernel, "csrmm_dt2t", 2),
    ],
)
@pytest.mark.parametrize(
    "transpose,direction",
    [(False, "nt"), (True, "t_indexed")],
)
def test_cuda_generator_selects_exact_direction_target(
    monkeypatch, factory, prefix, rank, transpose, direction
):
    ffi_calls = []
    load_calls = []
    monkeypatch.setattr(
        dt2t_module,
        "load_cuda_file",
        lambda path, name: load_calls.append((path, name)),
    )
    monkeypatch.setattr(
        dt2t_module.jax.ffi, "ffi_call", _recording_ffi_call(ffi_calls)
    )
    nse = 5
    rows = 4 if transpose else 3
    data_shape = (2, nse) if rank == 2 else (nse,)
    y_shape = (2, rows) if rank == 2 else (rows,)
    permutation_shape = (nse,) if transpose else (0,)
    with _explicit_int64_allowed():
        kernel = factory(
            transpose=transpose,
            mirror_enabled=transpose,
            w_info=_shape_of(jnp.float32, data_shape),
            outs=[_shape_of(jnp.float32, data_shape)],
            indices_info=_shape_of(jnp.int32, (nse,)),
            indptr_info=_shape_of(jnp.int64, (rows + 1,)),
            permutation_info=_shape_of(jnp.int64, permutation_shape),
        )
        kernel(
            jnp.ones(y_shape, dtype=jnp.float32),
            jnp.ones(data_shape, dtype=jnp.float32),
            jnp.zeros((nse,), dtype=jnp.int32),
            jnp.zeros((rows + 1,), dtype=jnp.int64),
            jnp.zeros(permutation_shape, dtype=jnp.int64),
        )

    assert [name for _, name in load_calls] == ["tcsr_dt2t"]
    assert [call[0] for call in ffi_calls] == [
        f"tcsr_dt2t.{prefix}_{direction}_f32"
    ]


def test_cuda_generator_rejects_unprepared_transpose_route():
    with pytest.raises(ValueError, match="prepared mirror"):
        dt2t_module._csrmv_dt2t_cuda_kernel(
            transpose=True,
            mirror_enabled=False,
            w_info=_shape_of(jnp.float32, (2,)),
            outs=[_shape_of(jnp.float32, (2,))],
            indices_info=_shape_of(jnp.int32, (2,)),
            indptr_info=_shape_of(jnp.int64, (3,)),
            permutation_info=_shape_of(jnp.int64, (0,)),
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@requires_gpu
def test_cuda_supports_csr_float_dtypes(dtype):
    with _explicit_int64_allowed():
        matrix = _matrix(backend="cuda_raw")
        y = jnp.arange(4, dtype=dtype) + 1
        w = jnp.arange(matrix.nse, dtype=dtype) + 2
        kwargs = dict(
            shape=matrix._base_shape,
            transpose=True,
            buffers=matrix._tcs_buffers,
        )
        expected = csrmv_dt2t(
            y, w, matrix._tcsr_indices, matrix._tcsr_indptr,
            backend="jax_raw", **kwargs,
        )
        result = csrmv_dt2t(
            y, w, matrix._tcsr_indices, matrix._tcsr_indptr,
            backend="cuda_raw", **kwargs,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("service", [csrmv_dt2t, csrmm_dt2t])
@requires_gpu
def test_cuda_matches_jax_reference(service, transpose):
    with _explicit_int64_allowed():
        matrix = _matrix(backend="cuda_raw")
        neuron_count = matrix._base_shape[1 if transpose else 0]
        y = jnp.arange(neuron_count, dtype=jnp.float32) + 1.0
        w = jnp.arange(matrix.nse, dtype=jnp.float32) + 2.0
        if service is csrmm_dt2t:
            y = jnp.stack((y, y + 1.0))
            w = jnp.stack((w, w + 2.0))
        kwargs = dict(
            shape=matrix._base_shape,
            transpose=transpose,
            buffers=matrix._tcs_buffers,
        )

        expected = service(
            y, w, matrix._tcsr_indices, matrix._tcsr_indptr,
            backend="jax_raw", **kwargs,
        )
        result = service(
            y, w, matrix._tcsr_indices, matrix._tcsr_indptr,
            backend="cuda_raw", **kwargs,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("transpose", [False, True])
def test_empty_rows_and_duplicate_columns_preserve_slot_order(transpose):
    with _explicit_int64_allowed():
        source = PlainCSR(
            (
                jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float32),
                jnp.asarray([2, 2, 0], dtype=jnp.int32),
                jnp.asarray([0, 2, 2, 3, 3], dtype=jnp.int32),
            ),
            shape=(4, 3),
            backend="jax_raw",
        )
        matrix = TCSR.from_sorted_csr(source, backend="jax_raw")
        neuron_count = matrix._base_shape[1 if transpose else 0]
        y = jnp.arange(neuron_count, dtype=jnp.float32) + 1.0
        w = jnp.asarray([7.0, 11.0, 13.0], dtype=jnp.float32)

        result = csrmv_dt2t(
            y,
            w,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            shape=matrix._base_shape,
            transpose=transpose,
            buffers=matrix._tcs_buffers,
            backend="jax_raw",
        )

        np.testing.assert_allclose(
            result,
            _expected(
                y,
                w,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                transpose=transpose,
            ),
        )


@pytest.mark.parametrize("transpose", [False, True])
@requires_gpu
def test_empty_structure_and_zero_batch_work_on_jax_and_cuda(transpose):
    with _explicit_int64_allowed():
        source = PlainCSR(
            (
                jnp.empty((0,), dtype=jnp.float32),
                jnp.empty((0,), dtype=jnp.int32),
                jnp.asarray([0, 0, 0], dtype=jnp.int32),
            ),
            shape=(2, 3),
            backend="cuda_raw",
        )
        matrix = TCSR.from_sorted_csr(source, backend="cuda_raw")
        neuron_count = matrix._base_shape[1 if transpose else 0]
        y = jnp.ones((neuron_count,), dtype=jnp.float32)
        mv_w = jnp.empty((0,), dtype=jnp.float32)
        mm_y = jnp.empty((0, neuron_count), dtype=jnp.float32)
        mm_w = jnp.empty((0, 0), dtype=jnp.float32)

        for backend in ("jax_raw", "cuda_raw"):
            kwargs = dict(
                shape=matrix._base_shape,
                transpose=transpose,
                buffers=matrix._tcs_buffers,
                backend=backend,
            )
            mv_result = csrmv_dt2t(
                y,
                mv_w,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                **kwargs,
            )
            mm_result = csrmm_dt2t(
                mm_y,
                mm_w,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                **kwargs,
            )

            assert mv_result.shape == (0,)
            assert mm_result.shape == (0, 0)
