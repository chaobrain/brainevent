from pathlib import Path
from typing import Optional, Union, Tuple, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._op._pipeline import load_cuda_file

#from brainevent._fcn import fcnmv_p

def new_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_fcnmv_op.cu'),
        name='fcn_float_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    _dtype_sfx = {
        np.dtype('float16'): '_f16',
        np.dtype('float32'): '_f32',
        np.dtype('float64'): '_f64',
        np.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(np.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    transpose_sfx = '_scatter' if transpose else '_gather'


    kernel_name = f'fcn_float_mv.fcnmv{transpose_sfx}{mode_sfx}_auto{sfx}'


    def kernel(weights, indices, vector):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, vector)

    return kernel


def binary_fcnmv_1t1r(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    

    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_thread{spike_sfx}{sfx}'
        #kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_4{spike_sfx}{sfx}'
        #binary_fcnmv_gather_hetero_thread_bool_f32
        '''
        if n_conn <= 20:
            kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_thread_unroll4{spike_sfx}{sfx}'
        else:
            kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_4{spike_sfx}{sfx}'
        '''
    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel

def binary_fcnmv_1t1r_unroll4(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    

    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_thread_unroll4{spike_sfx}{sfx}'
        #kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_4{spike_sfx}{sfx}'
        #binary_fcnmv_gather_hetero_thread_unroll4_bool_f32
        '''
        if n_conn <= 20:
            kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_thread_unroll4{spike_sfx}{sfx}'
        else:
            kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_4{spike_sfx}{sfx}'
        '''
    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel


def binary_1t1r_pipeline(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    
    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_thread_pipeline{spike_sfx}{sfx}'

    # =========================================================================
    # 状态变更：前端数据对齐注入点
    # =========================================================================
    def kernel(weights, indices, spikes):
        # 1. 计算对齐到 4 的倍数所需的 Padding 长度
        pad_len = (4 - (n_conn % 4)) % 4
        
        if pad_len > 0:
            # 2. 对 indices 进行补齐，常数填充 0 确保内存安全
            # indices_info.shape 为 (n_pre, n_conn)，pad 最后维度
            indices = jnp.pad(indices, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
            
            # 3. 对 weights 进行补齐 (仅 hetero 模式有二维 weights)
            if not homo:
                # JAX 的 jnp.pad 会自动将 0 转换为权重的原始 dtype (如 f16/f32)
                weights = jnp.pad(weights, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
        
        # 注意：out_info (输出张量的维度) 是 (n_pre,)，不受 n_conn 填充的影响，无需修改
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel


def binary_fcnmv_128_4(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    

    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_4{spike_sfx}{sfx}'
        #// @BE binary_fcnmv_gather_homo_128_4_float_f32

    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel


def binary_fcnmv_256_8(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    
    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        #binary_fcnmv_gather_homo_256_8_float_f16
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_256_8{spike_sfx}{sfx}'
        #// @BE binary_fcnmv_gather_homo_128_4_float_f32
    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel

def binary_fcnmv_256_4(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    
    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_256_4{spike_sfx}{sfx}'
        #// @BE binary_fcnmv_gather_homo_128_4_float_f32
    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel

def binary_fcnmv_128_2(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    
    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_2{spike_sfx}{sfx}'
        #// @BE binary_fcnmv_gather_homo_128_4_float_f32
    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel



def binary_fcnmv_1t1r_unroll2(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_mv_op',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'
    

    if transpose:
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_thread_unroll2{spike_sfx}{sfx}'
        #kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_4{spike_sfx}{sfx}'
        #binary_fcnmv_gather_hetero_thread_unroll4_bool_f32

        #binary_fcnmv_gather_homo_thread_unroll2_bool_f32
        '''
        if n_conn <= 20:
            kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_thread_unroll4{spike_sfx}{sfx}'
        else:
            kernel_name = f'fcn_binary_mv_op.binary_fcnmv{transpose_sfx}{mode_sfx}_128_4{spike_sfx}{sfx}'
        '''
    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel



def raw_cuda_unbranch(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mtsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_stsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'

    if transpose:
        # Scatter mode: if is_active(spikes[i]) → output[indices[i,k]] += weights[i,k]
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        # Gather mode: y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
        # binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_f32
        #binary_fcnmv_gather_hetero_unbranch_basic_bool_f64
        kernel_name = (
            f'fcn_binary_mtsr.binary_fcnmv{transpose_sfx}{mode_sfx}_basic_raw_unbranch{spike_sfx}{sfx}'
        )

    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel

def raw_cuda_template(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mtsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_stsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'

    if transpose:
        # Scatter mode: if is_active(spikes[i]) → output[indices[i,k]] += weights[i,k]
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        # Gather mode: y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
        # binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_f32
        # binary_fcnmv_gather_homo_basic_bg_mr_kern_template_bool_f64
        kernel_name = (
            f'fcn_binary_mtsr.binary_fcnmv{transpose_sfx}{mode_sfx}_basic_bg_mr_kern_template{spike_sfx}{sfx}'
        )

    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel

def raw_cuda_l2(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mtsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_stsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'

    if transpose:
        # Scatter mode: if is_active(spikes[i]) → output[indices[i,k]] += weights[i,k]
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        # Gather mode: y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
        # binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_f32
        #binary_fcnmv_gather_hetero_untail_basic_bool_f64
        kernel_name = (
            f'fcn_binary_mtsr.binary_fcnmv{transpose_sfx}{mode_sfx}_L2_basic{spike_sfx}{sfx}'
        )

    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel

def raw_cuda_bit(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mtsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_stsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'

    if transpose:
        # Scatter mode: if is_active(spikes[i]) → output[indices[i,k]] += weights[i,k]
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        #// @BE binary_fcnmv_gather_homo_basic_bit_bool_bf16
        # binary_fcnmv_gather_homo_basic_bit_bool_f32
        kernel_name = (
            f'fcn_binary_mtsr.binary_fcnmv{transpose_sfx}{mode_sfx}_basic_bit{spike_sfx}{sfx}'
        )

    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel


def raw_cuda_untail(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_mtsr.cu'),
        name='fcn_binary_mtsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_stsr.cu'),
        name='fcn_binary_stsr',
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_spike = (spike_info.dtype == jnp.bool_)
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_spike else '_float'
    transpose_sfx = '_scatter' if transpose else '_gather'

    if transpose:
        # Scatter mode: if is_active(spikes[i]) → output[indices[i,k]] += weights[i,k]
        kernel_name = (
            f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )
    else:
        # Gather mode: y[i] = sum_k weights[i,k] * is_active(spikes[indices[i,k]])
        # binary_fcnmv_gather_hetero_basic_raw_unbranch_bool_f32
        #binary_fcnmv_gather_hetero_untail_basic_bool_f64
        kernel_name = (
            f'fcn_binary_mtsr.binary_fcnmv{transpose_sfx}{mode_sfx}_untail_basic{spike_sfx}{sfx}'
        )

    def kernel(weights, indices, spikes):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel