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

import re
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import brainevent._csr.binary as binary_mod
import brainevent._csr.binary_indexed as binary_indexed_mod
import brainevent._csr.float as float_mod
import brainevent._csr.plasticity_binary as plasticity_mod
import brainevent._csr.slice as slice_mod
import brainevent._csr.DT2T as DT2T_mod
from brainevent._csr.binary import binary_csrmm, binary_csrmv
from brainevent._csr.binary_indexed import binary_csrmm_indexed, binary_csrmv_indexed
from brainevent._csr.float import csrmm, csrmv
from brainevent._csr.main import _make_binary_task_workspace
from brainevent._csr.plasticity_binary import update_csr_on_binary_post, update_csr_on_binary_pre
from brainevent._csr.slice import csr_slice_rows, csr_slice_rows_grad
from brainevent._csr.DT2T import csrmv_DT2T


requires_gpu = pytest.mark.skipif(
    jax.default_backend() != 'gpu',
    reason='CUDA int64 indptr tests require a GPU backend',
)


def _structure(indptr_dtype):
    weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4], dtype=indptr_dtype)
    return weights, indices, indptr


def _shape(dtype, shape=(2,)):
    return jax.ShapeDtypeStruct(shape, dtype)


@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _cuda_kwargs(indices_dtype=jnp.int64, indptr_dtype=jnp.int64, nse=2):
    return {
        'outs': [_shape(jnp.float32)],
        'shape': (1, 2),
        'indices_info': _shape(indices_dtype, (nse,)),
        'indptr_info': _shape(indptr_dtype, (2,)),
    }


def _recording_ffi_call(calls):
    def ffi_call(name, out_info, **ffi_kwargs):
        def call(*args, **kwargs):
            calls.append((name, out_info, ffi_kwargs, args, kwargs))
            if isinstance(out_info, (tuple, list)):
                return tuple(jnp.zeros(info.shape, info.dtype) for info in out_info)
            return jnp.zeros(out_info.shape, out_info.dtype)

        return call

    return ffi_call


@pytest.mark.parametrize(
    'factory,args,kwargs',
    [
        (
            float_mod._csrmv_cuda_kernel,
            (_shape(jnp.float32), False),
            {'outs': [_shape(jnp.float32)]},
        ),
        (
            float_mod._csrmm_cuda_kernel,
            (_shape(jnp.float32), False),
            {'outs': [_shape(jnp.float32, (1, 1))]},
        ),
        (
            binary_mod._binary_csrmv_cuda_kernel,
            (_shape(jnp.float32), _shape(jnp.bool_), False),
            {'outs': [_shape(jnp.float32)]},
        ),
        (
            binary_mod._binary_csrmm_cuda_kernel,
            (_shape(jnp.float32), _shape(jnp.bool_, (2, 1)), False),
            {'outs': [_shape(jnp.float32, (1, 1))]},
        ),
        (
            binary_indexed_mod._binary_csrmv_indexed_cuda_kernel,
            (_shape(jnp.float32), _shape(jnp.bool_), False),
            {'outs': [_shape(jnp.float32)], 'perm_info': _shape(jnp.int32)},
        ),
        (
            binary_indexed_mod._binary_csrmm_indexed_cuda_kernel,
            (_shape(jnp.float32), _shape(jnp.bool_, (2, 1)), False),
            {'outs': [_shape(jnp.float32, (1, 1))], 'perm_info': _shape(jnp.int32)},
        ),
        (
            slice_mod._csr_slice_rows_cuda_kernel_generator,
            (),
            {
                'outs': [_shape(jnp.float32, (1, 2))],
                'data_info': _shape(jnp.float32),
                'row_indices_info': _shape(jnp.int32, (1,)),
            },
        ),
        (
            slice_mod._csr_slice_rows_grad_cuda_kernel_generator,
            (),
            {
                'outs': [_shape(jnp.float32)],
                'ct_info': _shape(jnp.float32, (1, 2)),
                'row_indices_info': _shape(jnp.int32, (1,)),
            },
        ),
        (
            DT2T_mod._csrmv_DT2T_cuda_kernel,
            (False, _shape(jnp.float32)),
            {'outs': [_shape(jnp.float32)]},
        ),
    ],
)
def test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda(factory, args, kwargs):
    call_kwargs = _cuda_kwargs()
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match="indices with dtype int32"):
        factory(*args, **call_kwargs)


def test_plasticity_pre_cuda_rejects_int64_indices_before_loading_cuda():
    with pytest.raises(TypeError, match="indices with dtype int32"):
        plasticity_mod._csr_on_pre_cuda_kernel(
            _shape(jnp.float32),
            _shape(jnp.bool_),
            _shape(jnp.int64),
            outs=[_shape(jnp.float32)],
            indptr_info=_shape(jnp.int64),
        )


def test_plasticity_post_cuda_rejects_int64_indices_before_loading_cuda():
    with pytest.raises(TypeError, match="indices with dtype int32"):
        plasticity_mod._csr2csc_on_post_cuda_kernel(
            _shape(jnp.float32),
            _shape(jnp.bool_),
            _shape(jnp.int64),
            outs=[_shape(jnp.float32)],
            indptr_info=_shape(jnp.int64),
            weight_indices_info=_shape(jnp.int32),
        )


def test_plasticity_post_cuda_rejects_int64_weight_indices_before_loading_cuda():
    kwargs = {
        'outs': [_shape(jnp.float32)],
        'indptr_info': _shape(jnp.int64),
        'weight_indices_info': _shape(jnp.int64),
    }

    with pytest.raises(TypeError, match="weight_indices with dtype int32"):
        plasticity_mod._csr2csc_on_post_cuda_kernel(
            _shape(jnp.float32),
            _shape(jnp.bool_),
            _shape(jnp.int32),
            **kwargs,
        )


def test_float_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(float_mod, "load_cuda_file", lambda path, name: load_calls.append((path, name)))
    monkeypatch.setattr(float_mod.jax.ffi, "ffi_call", _recording_ffi_call(ffi_calls))

    with _jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)

        mv_kernel = float_mod._csrmv_cuda_kernel(
            _shape(jnp.float32, (1,)),
            False,
            **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
        )
        mv_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([1.0, 3.0], dtype=jnp.float64),
        )

        mm_kernel = float_mod._csrmm_cuda_kernel(
            _shape(jnp.float32, (1,)),
            True,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': [_shape(jnp.float32, (2, 1))],
            },
        )
        mm_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([[1.0], [3.0]], dtype=jnp.float32),
        )

    assert [name for _, name in load_calls] == ['csr_float_csrmv', 'csr_float_csrmm']
    assert [call[0] for call in ffi_calls] == [
        'csr_float_csrmv.csrmv_nt_auto_f32',
        'csr_float_csrmm.csrmm_t_warp_homo_f32',
    ]


def test_binary_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(binary_mod, "load_cuda_file", lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))
    monkeypatch.setattr(binary_mod.jax.ffi, "ffi_call", _recording_ffi_call(ffi_calls))

    with _jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)
        workspace = _make_binary_task_workspace(indptr)
        task_kwargs = {
            'task_begin_info': _shape(workspace.task_begin.dtype, workspace.task_begin.shape),
            'task_end_info': _shape(workspace.task_end.dtype, workspace.task_end.shape),
            'status_info': _shape(workspace.status.dtype, workspace.status.shape),
            'task_capacity': workspace.task_capacity,
        }
        mv_task_outs = (
            _shape(jnp.float32),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_nt_task_outs = (
            _shape(jnp.float32, (1, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_t_task_outs = (
            _shape(jnp.float32, (2, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )

        mv_kernel = binary_mod._binary_csrmv_cuda_kernel(
            _shape(jnp.float32, (1,)),
            _shape(jnp.bool_, (2,)),
            False,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([True, False]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mv_t_kernel = binary_mod._binary_csrmv_cuda_kernel(
            _shape(jnp.float32, (1,)),
            _shape(jnp.bool_, (1,)),
            True,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_t_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([True]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_nt_kernel = binary_mod._binary_csrmm_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.bool_, (2, 1)),
            False,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_nt_task_outs,
                **task_kwargs,
            },
        )
        mm_nt_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([[True], [False]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_t_kernel = binary_mod._binary_csrmm_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.float32, (2, 1)),
            True,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_t_task_outs,
                **task_kwargs,
            },
        )
        mm_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            jnp.array([[1.0], [0.0]], dtype=jnp.float32),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

    assert [name for _, name, _ in load_calls] == [
        'csr_binary_csrmv',
        'csr_binary_csrmv_hybrid',
        'csr_binary_csrmm',
        'csr_binary_csrmm_hybrid',
    ]
    assert [call[0] for call in ffi_calls] == [
        'csr_binary_csrmv.binary_csrmv_nt_auto_homo_f32_bool',
        'csr_binary_csrmv_hybrid.binary_csrmv_wat_hybrid_homo_f32_bool',
        'csr_binary_csrmm.binary_csrmm_nt_auto_hetero_f32_bool',
        'csr_binary_csrmm_hybrid.binary_csrmm_sraw_hybrid_hetero_f32_float',
    ]


def test_binary_indexed_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(binary_indexed_mod, "load_cuda_file", lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))
    monkeypatch.setattr(binary_indexed_mod.jax.ffi, "ffi_call", _recording_ffi_call(ffi_calls))

    with _jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)
        perm = jnp.array([1, 0], dtype=jnp.int32)
        workspace = _make_binary_task_workspace(indptr)
        task_kwargs = {
            'task_begin_info': _shape(workspace.task_begin.dtype, workspace.task_begin.shape),
            'task_end_info': _shape(workspace.task_end.dtype, workspace.task_end.shape),
            'status_info': _shape(workspace.status.dtype, workspace.status.shape),
            'task_capacity': workspace.task_capacity,
        }
        mv_task_outs = (
            _shape(jnp.float32),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_nt_task_outs = (
            _shape(jnp.float32, (1, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_t_task_outs = (
            _shape(jnp.float32, (2, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )

        mv_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.bool_, (2,)),
            False,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'perm_info': _shape(jnp.int32, (2,)),
                **task_kwargs,
            },
        )
        mv_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm,
            jnp.array([True, False]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mv_t_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.bool_, (1,)),
            True,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'perm_info': _shape(jnp.int64, (2,)),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([True]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mv_t_homo_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            _shape(jnp.float32, (1,)),
            _shape(jnp.bool_, (1,)),
            True,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'perm_info': _shape(jnp.int64, (2,)),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([True]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_nt_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.bool_, (2, 1)),
            False,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_nt_task_outs,
                'perm_info': _shape(jnp.int32, (2,)),
                **task_kwargs,
            },
        )
        mm_nt_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm,
            jnp.array([[True], [False]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_t_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.bool_, (1, 1)),
            True,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_t_task_outs,
                'perm_info': _shape(jnp.int64, (2,)),
                **task_kwargs,
            },
        )
        mm_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([[True]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_t_homo_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            _shape(jnp.float32, (1,)),
            _shape(jnp.bool_, (1, 1)),
            True,
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_t_task_outs,
                'perm_info': _shape(jnp.int64, (2,)),
                **task_kwargs,
            },
        )
        mm_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([[True]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

    assert [name for _, name, _ in load_calls] == [
        'csr_binary_indexed_csrmv',
        'csr_binary_indexed_csrmv_hybrid',
        'csr_binary_csrmv_hybrid',
        'csr_binary_indexed_csrmm',
        'csr_binary_indexed_csrmm_hybrid',
        'csr_binary_csrmm_hybrid',
    ]
    assert [call[0] for call in ffi_calls] == [
        'csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero_f32_bool',
        'csr_binary_indexed_csrmv_hybrid.binary_indexed_csrmv_wat_hybrid_hetero_f32_bool',
        'csr_binary_csrmv_hybrid.binary_csrmv_wat_hybrid_homo_f32_bool',
        'csr_binary_indexed_csrmm.binary_csrmm_nt_auto_perm_hetero_f32_bool',
        'csr_binary_indexed_csrmm_hybrid.binary_indexed_csrmm_sraw_hybrid_hetero_f32_bool',
        'csr_binary_csrmm_hybrid.binary_csrmm_sraw_hybrid_homo_f32_bool',
    ]


def test_slice_DT2T_and_plasticity_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(slice_mod, "load_cuda_file", lambda path, name: load_calls.append((path, name)))
    monkeypatch.setattr(slice_mod.jax.ffi, "ffi_call", _recording_ffi_call(ffi_calls))
    monkeypatch.setattr(DT2T_mod, "load_cuda_file", lambda path, name: load_calls.append((path, name)))
    monkeypatch.setattr(DT2T_mod.jax.ffi, "ffi_call", _recording_ffi_call(ffi_calls))
    monkeypatch.setattr(plasticity_mod, "load_cuda_file", lambda path, name: load_calls.append((path, name)))
    monkeypatch.setattr(plasticity_mod.jax.ffi, "ffi_call", _recording_ffi_call(ffi_calls))

    with _jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)

        slice_kernel = slice_mod._csr_slice_rows_cuda_kernel_generator(
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': [_shape(jnp.float32, (1, 2))],
                'data_info': _shape(jnp.float32, (2,)),
                'row_indices_info': _shape(jnp.int32, (1,)),
            }
        )
        slice_kernel(jnp.array([1.0, 2.0]), indices, indptr, jnp.array([0], dtype=jnp.int32))

        slice_grad_kernel = slice_mod._csr_slice_rows_grad_cuda_kernel_generator(
            **{
                **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': [_shape(jnp.float32, (2,))],
                'ct_info': _shape(jnp.float32, (1, 2)),
                'row_indices_info': _shape(jnp.int32, (1,)),
            }
        )
        slice_grad_kernel(jnp.array([[1.0, 2.0]]), indices, indptr, jnp.array([0], dtype=jnp.int32))

        DT2T_kernel = DT2T_mod._csrmv_DT2T_cuda_kernel(
            False,
            _shape(jnp.float32, (2,)),
            **_cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
        )
        DT2T_kernel(jnp.array([1.0]), jnp.array([2.0, 3.0]), indices, indptr)

        pre_kernel = plasticity_mod._csr_on_pre_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.bool_, (1,)),
            _shape(jnp.int32, (2,)),
            outs=[_shape(jnp.float32, (2,))],
            indptr_info=_shape(jnp.int64, (2,)),
        )
        pre_kernel(
            jnp.array([1.0, 2.0]),
            indices,
            indptr,
            jnp.array([True]),
            jnp.array([0.5, 1.5]),
        )

        post_kernel = plasticity_mod._csr2csc_on_post_cuda_kernel(
            _shape(jnp.float32, (2,)),
            _shape(jnp.float32, (2,)),
            _shape(jnp.int32, (2,)),
            outs=[_shape(jnp.float32, (2,))],
            indptr_info=_shape(jnp.int64, (2,)),
            weight_indices_info=_shape(jnp.int32, (2,)),
        )
        post_kernel(
            jnp.array([1.0, 2.0]),
            indices,
            indptr,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([0.5]),
            jnp.array([1.0, -1.0]),
        )

    assert [name for _, name in load_calls] == [
        'csr_slice_rows',
        'csr_slice_rows',
        'csrmv_DT2T',
        'csr_plasticity_binary_pre',
        'csr_plasticity_binary_post',
    ]
    assert [call[0] for call in ffi_calls] == [
        'csr_slice_rows.csr_slice_rows_fwd_hetero_auto_f32',
        'csr_slice_rows.csr_slice_rows_grad_auto_f32',
        'csrmv_DT2T.csrmv_DT2T_nt_auto_f32',
        'csr_plasticity_binary_pre.update_csr_on_pre_f32_bool',
        'csr_plasticity_binary_post.update_csr_on_post_f32_float',
    ]


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_float_csrmv_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    vector = jnp.array([1.0, 2.0], dtype=jnp.float32) if transpose else jnp.array([1.0, 2.0, 3.0])

    got = csrmv(data, indices, indptr64, vector, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = csrmv(data, indices, indptr32, vector, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_float_csrmm_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    matrix = (
        jnp.array([[1.0, 0.5], [2.0, 1.5]], dtype=jnp.float32)
        if transpose else
        jnp.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]], dtype=jnp.float32)
    )

    got = csrmm(data, indices, indptr64, matrix, shape=(2, 3), transpose=transpose, backend='cuda_raw')
    expected = csrmm(data, indices, indptr32, matrix, shape=(2, 3), transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmv_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmv(data, indices, indptr64, vector, shape=(2, 3), transpose=transpose,
                       backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmv(data, indices, indptr32, vector, shape=(2, 3), transpose=transpose,
                            backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_binary_csrmm_cuda_accepts_int64_indptr(transpose, homo):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    data = weights if not homo else jnp.array([2.0], dtype=jnp.float32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmm(data, indices, indptr64, matrix, shape=(2, 3), transpose=transpose,
                       backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmm(data, indices, indptr32, matrix, shape=(2, 3), transpose=transpose,
                            backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
def test_binary_indexed_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmv_indexed(weights, indices, indptr64, perm, vector, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmv_indexed(weights, indices, indptr32, perm, vector, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize('transpose', [False, True])
def test_binary_indexed_csrmm_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmm_indexed(weights, indices, indptr64, perm, matrix, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmm_indexed(weights, indices, indptr32, perm, matrix, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_slice_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    rows = jnp.array([1, 0], dtype=jnp.int32)

    got = csr_slice_rows(
        weights, indices, indptr64, rows, shape=(2, 3), backend='cuda_raw'
    )
    expected = jnp.array([[0.0, 3.0, 4.0], [1.0, 0.0, 2.0]], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_slice_grad_cuda_accepts_int64_indptr():
    _, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    rows = jnp.array([1, 0], dtype=jnp.int32)
    ct = jnp.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)

    got = csr_slice_rows_grad(ct, indices, indptr64, rows, shape=(2, 3), backend='cuda_raw')
    expected = jnp.array([1.0, 3.0, 20.0, 30.0], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_DT_to_T_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    y = jnp.array([1.0, 2.0], dtype=jnp.float32)

    got = csrmv_DT2T(y, weights, indices, indptr64, shape=(2, 3), backend='cuda_raw')
    expected = csrmv_DT2T(y, weights, indices, indptr32, shape=(2, 3), backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_plasticity_pre_cuda_accepts_int64_indptr():
    weights, indices, indptr32 = _structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    pre_spike = jnp.array([True, False])
    post_trace = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32)

    got = update_csr_on_binary_pre(
        weights, indices, indptr64, pre_spike, post_trace, shape=(2, 3), backend='cuda_raw'
    )
    expected = jnp.array([1.5, 4.5, 3.0, 4.0], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_plasticity_post_cuda_accepts_int64_indptr():
    weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
    weight_indices = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
    pre_trace = jnp.array([0.5, 1.5], dtype=jnp.float32)
    post_spike = jnp.array([False, True])

    got = update_csr_on_binary_post(
        weights, indices, indptr, weight_indices, pre_trace, post_spike, shape=(2, 2), backend='cuda_raw'
    )
    expected = jnp.array([1.0, 2.5, 3.0, 5.5], dtype=jnp.float32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_csr_cuda_sources_do_not_cast_indptr_to_int32():
    csr_dir = Path(__file__).parent
    for path in csr_dir.glob('*.cu'):
        text = path.read_text()
        assert 'static_cast<const int32_t*>(indptr.data_ptr())' not in text, path.name
        assert 'const int32_t*  __restrict__ indptr' not in text, path.name
        assert 'const int32_t*   __restrict__ indptr' not in text, path.name


def test_binary_hybrid_cuda_exports_use_csr_abi_names():
    csr_dir = Path(__file__).parent
    weight_suffixes = ('f32', 'f64', 'f16', 'bf16')
    event_suffixes = ('bool', 'float')
    expected_symbols = {
        'binary_csrmv_hybrid.cu': [
            f'binary_csrmv_wat_hybrid_{mode}_{weight}_{event}'
            for mode in ('hetero', 'homo')
            for weight in weight_suffixes
            for event in event_suffixes
        ],
        'binary_csrmm_hybrid.cu': [
            f'binary_csrmm_sraw_hybrid_{mode}_{weight}_{event}'
            for mode in ('hetero', 'homo')
            for weight in weight_suffixes
            for event in event_suffixes
        ],
        'binary_indexed_csrmv_hybrid.cu': [
            f'binary_indexed_csrmv_wat_hybrid_hetero_{weight}_{event}'
            for weight in weight_suffixes
            for event in event_suffixes
        ],
        'binary_indexed_csrmm_hybrid.cu': [
            f'binary_indexed_csrmm_sraw_hybrid_hetero_{weight}_{event}'
            for weight in weight_suffixes
            for event in event_suffixes
        ],
    }

    for filename, symbols in expected_symbols.items():
        text = (csr_dir / filename).read_text()
        exported = set(re.findall(r'^// @BE ([A-Za-z0-9_]+)$', text, re.MULTILINE))

        for symbol in symbols:
            assert symbol in exported
            suffix = '_' + '_'.join(symbol.split('_')[-2:])
            assert re.search(
                rf'^// @BE {symbol}\nDEFINE_[A-Z0-9_]+\(\\?{suffix},',
                text,
                re.MULTILINE,
            )

        binary_names = {name for name in exported if name.startswith('binary')}
        assert not {name for name in binary_names if 'fcnmm' in name or 'const_block' in name}
