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

conn_num_base = 80


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
    ):
        super().__init__()

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.efferent_target = efferent_target
        self.data_type = data_type
        if data_type not in ('binary', 'sparse_float', 'float'):
            raise ValueError('data_type must be either "binary" or "sparse_float" or "float".')
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

        # connections
        if self.efferent_target == 'post':
            n_post = self.out_size[-1]
            n_pre = self.in_size[-1]
        else:
            n_post = self.in_size[-1]
            n_pre = self.out_size[-1]

        with jax.ensure_compile_time_eval():
            assert allow_multi_conn
            indices = brainstate.random.randint(0, n_post, size=(n_pre, self.conn_num))
            conn_weight = conn_weight_base / conn_num_base * self.conn_num
            if not homo:
                conn_weight = u.math.full((n_pre, self.conn_num), conn_weight)
            self.weight = brainstate.ParamState(
                u.math.asarray(conn_weight, dtype=brainstate.environ.dftype())
            )
            self.indices = u.math.asarray(indices, dtype=np.int32)
            self.shape = (n_pre, n_post)
            # csr = (
            #     brainevent.FixedPostNumConn((conn_weight, indices), shape=(n_pre, n_post))
            #     if self.efferent_target == 'post' else
            #     brainevent.FixedPreNumConn((conn_weight, indices), shape=(n_pre, n_post))
            # )
            # self.conn = csr

    def update(self, x) -> Union[jax.Array, u.Quantity]:
        assert x.ndim in [1, 2], 'Input must be 1D or 2D.'
        if self.data_type == 'binary':
            fn = brainevent.binary_fcnmv if x.ndim == 1 else brainevent.binary_fcnmm
        elif self.data_type == 'sparse_float':
            fn = brainevent.sparse_float_fcnmv if x.ndim == 1 else brainevent.sparse_float_fcnmm
        else:
            fn = brainevent.float_fcnmv if x.ndim == 1 else brainevent.float_fcnmm
        transpose = (self.efferent_target == 'post')
        return fn(self.weight.value, self.indices, x, shape=self.shape, transpose=transpose)


class EINet(brainstate.nn.Module):
    def __init__(
        self,
        scale: float,
        data_type: str,
        efferent_target: str,
        conn_num: int = 80,
        homo: bool = True,
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
                data_type=data_type, efferent_target=efferent_target, conn_weight_base=0.6 * u.mS,
            ),
            syn=brainpy.state.Expon.desc(self.num, tau=5. * u.ms),
            out=brainpy.state.COBA.desc(E=0. * u.mV),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=FixedNumConn(
                self.n_inh, self.num, conn_num=conn_num, homo=homo,
                data_type=data_type, efferent_target=efferent_target, conn_weight_base=6.7 * u.mS,
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
):
    @brainstate.transform.jit
    def run():
        net = EINet(scale, data_type=data_type, efferent_target=efferent_target)
        net.init_all_states()

        def fn(t):
            return net.update(t, 20. * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        return net.rate.value.sum() / net.num / duration.to_decimal(u.second)

    return run


def make_training_run(
    scale: float,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
):
    def loss_fn():
        net = EINet(scale, data_type=data_type, efferent_target=efferent_target)
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
):
    @brainstate.transform.jit
    def run():
        net = EINet(scale, data_type=data_type, efferent_target=efferent_target)
        mapper = brainstate.nn.Map(net, init_map_size=batch_size)
        mapper.init_all_states()

        def fn(t):
            ts = jax.numpy.ones(batch_size) * t
            return mapper.map('update', in_axes=(0, None))(ts, 20. * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        return net.rate.value.sum() / net.num / duration.to_decimal(u.second) / batch_size

    return run


def make_training_batch_run(
    scale: float,
    batch_size: int = 16,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
):
    def loss_fn():
        net = EINet(scale, data_type=data_type, efferent_target=efferent_target)
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
