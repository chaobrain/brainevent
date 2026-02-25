# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

#
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007),
#   Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#


import time

import brainpy
import brainstate
import braintools
import brainunit as u
import jax

import brainevent
from model import FixedNumConn

brainevent.config.set_backend('gpu', 'tvmffi')

batch_size = 16
data_type = 'binary'
efferent_target = 'post'


class EINet(brainstate.nn.Module):
    def __init__(self, scale):
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
                self.n_exc, self.num, conn_num=80 / self.num, conn_weight=0.6 * u.mS,
                data_type=data_type, efferent_target=efferent_target
            ),
            syn=brainpy.state.Expon.desc(self.num, tau=5. * u.ms),
            out=brainpy.state.COBA.desc(E=0. * u.mV),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=FixedNumConn(
                self.n_inh, self.num, conn_num=80 / self.num, conn_weight=6.7 * u.mS,
                data_type=data_type, efferent_target=efferent_target
            ),
            syn=brainpy.state.Expon.desc(self.num, tau=10. * u.ms),
            out=brainpy.state.COBA.desc(E=-80. * u.mV),
            post=self.N
        )

    def init_state(self, *args, **kwargs):
        self.rate = brainstate.ShortTermState(u.math.zeros(self.num))

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            self.rate.value += self.N.get_spike()


@brainstate.transform.jit(static_argnums=0)
def run(scale: float):
    # network
    net = EINet(scale)
    mapper = brainstate.nn.Map(net, init_map_size=batch_size)
    mapper.init_all_states()

    def fn(t):
        ts = jax.numpy.ones(batch_size) * t
        return mapper.map('update', in_axes=(0, None))(ts, 20. * u.mA)

    # simulation
    duration = 1e4 * u.ms
    with brainstate.environ.context(dt=0.1 * u.ms):
        times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
        brainstate.transform.for_loop(fn, times)

    return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second) / batch_size


# for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
for s in [1, ]:
    jax.block_until_ready(run(s))

    t0 = time.time()
    n, rate = jax.block_until_ready(run(s))
    t1 = time.time()
    print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')

# --------------------
# 2026/02/04, NVIDIA GeForce RTX 3090, brainevent 0.0.6, Warp 1.11.1+cu13, jax 0.9.0, Ubuntu 24.04
#
# scale=1, size=4000, time = 1.7868633270263672 s, firing rate = 59.56795120239258 Hz
# scale=2, size=8000, time = 1.8208961486816406 s, firing rate = 59.56516647338867 Hz
# scale=4, size=16000, time = 1.9305737018585205 s, firing rate = 59.5695915222168 Hz
# scale=6, size=24000, time = 2.1917836666107178 s, firing rate = 59.56669998168945 Hz
# scale=8, size=32000, time = 2.235652208328247 s, firing rate = 59.57065200805664 Hz
# scale=10, size=40000, time = 2.2676308155059814 s, firing rate = 59.57038497924805 Hz
# scale=20, size=80000, time = 2.3961074352264404 s, firing rate = 59.570884704589844 Hz
# scale=40, size=160000, time = 2.8628697395324707 s, firing rate = 59.571327209472656 Hz
# scale=60, size=240000, time = 3.671980857849121 s, firing rate = 59.57017135620117 Hz
# scale=80, size=320000, time = 4.697195053100586 s, firing rate = 59.56940841674805 Hz
# scale=100, size=400000, time = 5.40840220451355 s, firing rate = 59.569679260253906 Hz
