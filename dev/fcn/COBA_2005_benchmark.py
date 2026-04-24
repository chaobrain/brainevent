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


from typing import Union

import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import brainevent
from brainevent._misc import fixed_conn_num_to_csc

conn_num_base = 80
            
def _build_col_major_fcn(weights, indices: np.ndarray, shape, dtype):
    """Build a compact CSC mirror from row-major FCN data."""
    weight_value, weight_unit = u.split_mantissa_unit(weights)
    weight_value = np.asarray(weight_value)
    col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(
        weight_value,
        indices,
        shape=shape,
    )
    col_weights = u.maybe_decimal(u.math.asarray(col_weights, dtype=dtype) * weight_unit)
    return (
        col_weights,
        u.math.asarray(col_indices, dtype=np.int32),
        u.math.asarray(col_indptr, dtype=np.int32),
    )


def _use_compact_col_scatter_path(*, x_ndim: int, transpose: bool, mv_layout: str) -> bool:
    return x_ndim == 1 and (not transpose) and mv_layout == 'col_scatter'


def _use_compact_full_compaction(*, x_ndim: int, transpose: bool, mv_layout: str) -> bool:
    if x_ndim != 1:
        return False
    if transpose:
        return True
    return mv_layout == 'col_scatter'


class FixedNumConn(brainstate.nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        conn_num: Union[int, float],
        efferent_target: str = 'post',  # 'pre' or 'post'
        data_type: str = 'binary',
        homo: bool = True,
        conn_weight_base: u.Quantity = 0.6 * u.mS,
        allow_multi_conn: bool = True,
        mv_layout: str = 'row_gather',
    ):
        super().__init__()

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.efferent_target = efferent_target
        self.data_type = data_type
        if data_type not in ('binary', 'float', 'bitpack', 'bitpack_a0', 'bitpack_a1', 'compact'):
            raise ValueError('data_type must be one of "binary", "float", "bitpack", "bitpack_a0", "bitpack_a1", "compact".')
        if efferent_target not in ('pre', 'post'):
            raise ValueError('The target of the connection must be either "pre" or "post".')
        if isinstance(conn_num, float):
            assert 0. <= conn_num <= 1., 'Connection probability must be in [0, 1].'
            conn_num = (
                int(self.out_size[-1] * conn_num)
                if efferent_target == 'post' else
                int(self.in_size[-1] * conn_num)
            )
        assert isinstance(conn_num, int), 'Connection number must be an integer.'
        self.conn_num = conn_num
        self.allow_multi_conn = allow_multi_conn
        self.mv_layout = mv_layout
        if mv_layout not in ('col_scatter', 'row_gather', 'auto'):
            raise ValueError("`mv_layout` must be one of 'col_scatter', 'row_gather', or 'auto'.")

        # connections
        if self.efferent_target == 'post':
            n_post = self.out_size[-1]
            n_pre = self.in_size[-1]
        else:
            n_post = self.in_size[-1]
            n_pre = self.out_size[-1]

        with jax.ensure_compile_time_eval():
            assert allow_multi_conn
            indices_np = np.random.randint(0, n_post, size=(n_pre, self.conn_num)).astype(np.int32, copy=False)
            conn_weight = conn_weight_base * conn_num_base / self.conn_num
            if not homo:
                conn_weight = u.math.full((n_pre, self.conn_num), conn_weight)
            row_weight = u.math.asarray(conn_weight, dtype=brainstate.environ.dftype())
            self.weight = brainstate.ParamState(row_weight)
            self.indices = u.math.asarray(indices_np, dtype=np.int32)
            self.shape = (n_pre, n_post)
            self.col_weight = None
            self.col_indices = None
            self.col_indptr = None
            if data_type in ('binary', 'compact') and mv_layout == 'col_scatter':
                self.col_weight, self.col_indices, self.col_indptr = _build_col_major_fcn(
                    row_weight,
                    indices_np,
                    self.shape,
                    brainstate.environ.dftype(),
                )

    def update(self, x) -> Union[jax.Array, u.Quantity]:
        assert x.ndim in [1, 2], 'Input must be 1D or 2D.'
        transpose = (self.efferent_target == 'post')
        if self.data_type == 'compact':
            # Compact MV has three semantic routes:
            # 1) post update      -> transpose=True row-scatter -> full compaction
            # 2) pre row_gather   -> transpose=False gather     -> light compaction
            # 3) pre col_scatter  -> transpose=False CSC path   -> full compaction
            use_compact_col_scatter = _use_compact_col_scatter_path(
                x_ndim=x.ndim,
                transpose=transpose,
                mv_layout=self.mv_layout,
            )
            use_compact_full = _use_compact_full_compaction(
                x_ndim=x.ndim,
                transpose=transpose,
                mv_layout=self.mv_layout,
            )
            cb = (
                brainevent.CompactBinary.from_array(x)
                if use_compact_full else
                brainevent.CompactBinary.from_array_light(x)
            )
            if x.ndim == 1:
                return brainevent.compact_binary_fcnmv(
                    self.weight.value, self.indices,
                    cb.packed, cb.active_ids, cb.n_active, cb.value,
                    shape=self.shape, transpose=transpose,
                    col_weights=self.col_weight if use_compact_col_scatter else None,
                    col_indices=self.col_indices if use_compact_col_scatter else None,
                    col_indptr=self.col_indptr if use_compact_col_scatter else None,
                )
            else:
                return brainevent.compact_binary_fcnmm(
                    self.weight.value, self.indices,
                    cb.packed, cb.active_ids, cb.n_active, cb.value,
                    shape=self.shape, transpose=transpose, pack_axis=1,
                )
        if self.data_type in ('bitpack', 'bitpack_a0', 'bitpack_a1'):
            bp = brainevent.BitPackedBinary(x)
            if x.ndim == 1:
                # 1D: batching rule promotes MV→MM automatically
                return brainevent.bitpack_binary_fcnmv(
                    self.weight.value, self.indices, bp.packed[0], bp.value,
                    shape=self.shape, transpose=transpose,
                )
            else:
                pack_axis = 1 if self.data_type == 'bitpack_a1' else 0
                return brainevent.bitpack_binary_fcnmm(
                    self.weight.value, self.indices, bp.packed[pack_axis], bp.value,
                    shape=self.shape, transpose=transpose, pack_axis=pack_axis,
                )
        if self.data_type == 'binary':
            if x.ndim == 1:
                binary_mv_kwargs = {}
                if (not transpose) and self.mv_layout == 'col_scatter':
                    binary_mv_kwargs = dict(
                        col_weights=self.col_weight,
                        col_indices=self.col_indices,
                        col_indptr=self.col_indptr,
                    )
                return brainevent.binary_fcnmv(
                    self.weight.value,
                    self.indices,
                    x,
                    shape=self.shape,
                    transpose=transpose,
                    **binary_mv_kwargs,
                )
            else:
                fn = brainevent.binary_fcnmm
        elif self.data_type == 'float':
            fn = brainevent.fcnmv if x.ndim == 1 else brainevent.fcnmm
        else:
            raise ValueError(f'Unsupported data_type for direct update path: {self.data_type!r}.')
        return fn(self.weight.value, self.indices, x, shape=self.shape, transpose=transpose)


class EINet(brainstate.nn.Module):
    def __init__(
        self,
        scale: float,
        data_type: str,
        efferent_target: str,
        conn_num: int = 80,
        homo: bool = True,
        mv_layout: str = 'row_gather',
    ):
        super().__init__()
        self.n_exc = int(3200 * scale)
        self.n_inh = int(800 * scale)
        self.num = self.n_exc + self.n_inh
        self.N = brainpy.state.LIFRef(
            self.num, V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55., 2., unit=u.mV)
        )
        self.E = brainpy.state.AlignPostProj(
            comm=FixedNumConn(
                self.n_exc, self.num, conn_num=conn_num, homo=homo,
                data_type=data_type, efferent_target=efferent_target,
                conn_weight_base=0.6 * u.mS,
                mv_layout=mv_layout,
            ),
            syn=brainpy.state.Expon.desc(self.num, tau=5. * u.ms),
            out=brainpy.state.COBA.desc(E=0. * u.mV),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=FixedNumConn(
                self.n_inh, self.num, conn_num=conn_num, homo=homo,
                data_type=data_type, efferent_target=efferent_target,
                conn_weight_base=6.7 * u.mS,
                mv_layout=mv_layout,
            ),
            syn=brainpy.state.Expon.desc(self.num, tau=10. * u.ms),
            out=brainpy.state.COBA.desc(E=-80. * u.mV),
            post=self.N
        )

    def init_state(self, *args, **kwargs):
        self.rate = brainstate.ShortTermState(u.math.zeros(self.num))

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike()
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            self.rate.value += self.N.get_spike()


def make_simulation_run(
    scale: float,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    conn_num: int = 80,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    @brainstate.transform.jit
    def run():
        # Build the network inside JIT so the benchmark does not keep a large
        # connectivity object alive via the jitted function closure.
        net = EINet(
            scale,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
        )
        net.init_all_states()

        def fn(t):
            return net.update(t, 20* u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second)

    return run


def make_training_run(
    scale: float,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    def loss_fn():
        net = EINet(
            scale,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=homo,
            mv_layout=mv_layout,
        )
        net.init_all_states()

        def fn(t):
            return net.update(t, 20. * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        firing_rates = net.rate.value.sum() / net.num / duration.to_decimal(u.second)
        return jax.numpy.mean((firing_rates - 10.) ** 2)

    @brainstate.transform.jit
    def run():
        grads = brainstate.transform.grad(loss_fn)()
        return grads

    return run


def make_simulation_batch_run(
    scale: float,
    batch_size: int = 16,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    conn_num: int = 80,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    @brainstate.transform.jit
    def run():
        net = EINet(
            scale,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
        )
        mapper = brainstate.nn.Map(net, init_map_size=batch_size)
        mapper.init_all_states()

        def fn(t):
            ts = jax.numpy.ones(batch_size) * t
            return mapper.map('update', in_axes=(0, None))(ts, 20. * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second) / batch_size

    return run


def make_training_batch_run(
    scale: float,
    batch_size: int = 16,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    conn_num: int = 80,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    def loss_fn():
        net = EINet(
            scale,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
        )
        mapper = brainstate.nn.Map(net, init_map_size=batch_size)
        mapper.init_all_states()

        def fn(t):
            ts = jax.numpy.ones(batch_size) * t
            return mapper.map('update', in_axes=(0, None))(ts, 20. * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        firing_rates = net.rate.value.sum() / net.num / duration.to_decimal(u.second)
        return jax.numpy.mean((firing_rates - 10.) ** 2)

    @brainstate.transform.jit
    def run():
        grads = brainstate.transform.grad(loss_fn)()
        return grads

    return run
