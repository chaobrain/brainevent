# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import brainstate
import brainunit as u
import jax


class EINet(brainstate.nn.DynamicsGroup):
    def __init__(self, scale=1.0):
        super().__init__()
        self.n_exc = int(3200 * scale)
        self.n_inh = int(800 * scale)
        self.num = self.n_exc + self.n_inh
        self.N = brainstate.nn.LIFRef(
            self.num, V_rest=-49. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=brainstate.init.Normal(-55., 2., unit=u.mV)
        )
        self.E = brainstate.nn.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num, conn_num=80 / self.num, conn_weight=1.62 * u.mS),
            syn=brainstate.nn.Expon.desc(self.num, tau=5. * u.ms),
            out=brainstate.nn.CUBA.desc(scale=u.volt),
            post=self.N
        )
        self.I = brainstate.nn.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num, conn_num=80 / self.num, conn_weight=-9.0 * u.mS),
            syn=brainstate.nn.Expon.desc(self.num, tau=10. * u.ms),
            out=brainstate.nn.CUBA.desc(scale=u.volt),
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


@brainstate.compile.jit(static_argnums=0)
def run(scale: float):
    # network
    net = EINet(scale)
    brainstate.nn.init_all_states(net)

    duration = 1e4 * u.ms
    # simulation
    with brainstate.environ.context(dt=0.1 * u.ms):
        times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
        brainstate.compile.for_loop(lambda t: net.update(t, 20. * u.mA), times,
                                    # pbar=brainstate.compile.ProgressBar(100)
                                    )

    return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second)


for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
    jax.block_until_ready(run(s))

    t0 = time.time()
    n, rate = jax.block_until_ready(run(s))
    t1 = time.time()
    print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')

# A6000 NVIDIA GPU

# scale=1, size=4000, time = 2.6354849338531494 s, firing rate = 24.982027053833008 Hz
# scale=2, size=8000, time = 2.6781561374664307 s, firing rate = 23.719463348388672 Hz
# scale=4, size=16000, time = 2.7448785305023193 s, firing rate = 24.592931747436523 Hz
# scale=6, size=24000, time = 2.8237478733062744 s, firing rate = 24.159996032714844 Hz
# scale=8, size=32000, time = 2.9344418048858643 s, firing rate = 24.956790924072266 Hz
# scale=10, size=40000, time = 3.042517900466919 s, firing rate = 23.644424438476562 Hz
# scale=20, size=80000, time = 3.6727631092071533 s, firing rate = 24.226743698120117 Hz
# scale=40, size=160000, time = 4.857396602630615 s, firing rate = 24.329742431640625 Hz
# scale=60, size=240000, time = 6.812030792236328 s, firing rate = 24.370006561279297 Hz
# scale=80, size=320000, time = 9.227966547012329 s, firing rate = 24.41067886352539 Hz
# scale=100, size=400000, time = 11.405697584152222 s, firing rate = 24.32524871826172 Hz


# AMD Ryzen 7 7840HS

# scale=1, size=4000, time = 1.1661601066589355 s, firing rate = 22.438201904296875 Hz
# scale=2, size=8000, time = 3.3255884647369385 s, firing rate = 23.868364334106445 Hz
# scale=4, size=16000, time = 6.950139999389648 s, firing rate = 24.21693229675293 Hz
# scale=6, size=24000, time = 10.011993169784546 s, firing rate = 24.240270614624023 Hz
# scale=8, size=32000, time = 13.027734518051147 s, firing rate = 24.753198623657227 Hz
# scale=10, size=40000, time = 16.449942350387573 s, firing rate = 24.7176570892334 Hz
# scale=20, size=80000, time = 30.754598140716553 s, firing rate = 24.119956970214844 Hz
# scale=40, size=160000, time = 63.6387836933136 s, firing rate = 24.72784996032715 Hz
# scale=60, size=240000, time = 78.58532166481018 s, firing rate = 24.402742385864258 Hz
# scale=80, size=320000, time = 102.4250214099884 s, firing rate = 24.59092140197754 Hz
# scale=100, size=400000, time = 145.35173273086548 s, firing rate = 24.33751106262207 Hz


#
# scale=1, size=4000, time = 6.877645969390869 s, firing rate = 49.86395263671875 Hz
# scale=2, size=8000, time = 7.334841966629028 s, firing rate = 51.66765213012695 Hz
# scale=4, size=16000, time = 7.954581260681152 s, firing rate = 50.640995025634766 Hz
# scale=6, size=24000, time = 7.8790388107299805 s, firing rate = 50.05788803100586 Hz
# scale=8, size=32000, time = 7.738804817199707 s, firing rate = 50.67148208618164 Hz
# scale=10, size=40000, time = 8.00656533241272 s, firing rate = 51.362403869628906 Hz
# scale=20, size=80000, time = 5.608523607254028 s, firing rate = 51.10150909423828 Hz
# scale=40, size=160000, time = 7.927639007568359 s, firing rate = 51.04258728027344 Hz
# scale=60, size=240000, time = 10.51952600479126 s, firing rate = 51.01384735107422 Hz
# scale=80, size=320000, time = 13.65773606300354 s, firing rate = 50.80935287475586 Hz
# scale=100, size=400000, time = 16.7642924785614 s, firing rate = 50.967491149902344 Hz


# scale=1, size=4000, time = 5.818003177642822 s, firing rate = 47.783626556396484 Hz
# scale=2, size=8000, time = 5.346902370452881 s, firing rate = 50.735164642333984 Hz
# scale=4, size=16000, time = 5.5788414478302 s, firing rate = 50.86026382446289 Hz
# scale=6, size=24000, time = 5.770918607711792 s, firing rate = 51.01063537597656 Hz
# scale=8, size=32000, time = 5.189192056655884 s, firing rate = 50.55289077758789 Hz
# scale=10, size=40000, time = 7.3012917041778564 s, firing rate = 51.468936920166016 Hz
# scale=20, size=80000, time = 7.324301481246948 s, firing rate = 50.845970153808594 Hz
# scale=40, size=160000, time = 8.84085488319397 s, firing rate = 50.70090866088867 Hz
# scale=60, size=240000, time = 8.829878807067871 s, firing rate = 50.923377990722656 Hz
# scale=80, size=320000, time = 10.720089197158813 s, firing rate = 50.86113357543945 Hz
# scale=100, size=400000, time = 12.893761396408081 s, firing rate = 51.1276741027832 Hz
