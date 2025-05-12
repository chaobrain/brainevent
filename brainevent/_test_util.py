# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import brainstate
import jax.numpy as jnp

import brainevent


def allclose(x, y, rtol=1e-4, atol=1e-4):
    x = x.data if isinstance(x, brainevent.EventArray) else x
    y = y.data if isinstance(y, brainevent.EventArray) else y
    return jnp.allclose(x, y, rtol=rtol, atol=atol)


def gen_events(shape, prob=0.5, asbool=True):
    events = brainstate.random.random(shape) < prob
    if not asbool:
        events = jnp.asarray(events, dtype=float)
    return brainevent.EventArray(events)
