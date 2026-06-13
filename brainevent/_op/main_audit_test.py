# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Audit tests for ``brainevent/_op/main.py`` (L3, M13).

* L3 - ``apply_primitive`` is resolved through the ``_compatible_import`` shim
  and is identical to the function the legacy ``jax.interpreters.xla`` path
  exposed.
* M13 - a global backend that is not registered for a platform no longer fails
  silently: the selection emits a (deduplicated) warning and falls back to the
  primitive default, still producing the correct result.
"""

import importlib.util
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.interpreters import xla

from brainevent._compatible_import import apply_primitive
from brainevent._op.main import XLACustomKernel
from brainevent.config import set_backend, clear_backends

numba_installed = importlib.util.find_spec('numba') is not None


def test_l3_apply_primitive_shim_is_xla_apply_primitive():
    """The shim resolves to exactly the same callable as the legacy path."""
    assert xla.apply_primitive is apply_primitive


def test_m13_warn_backend_once_deduplicates():
    """``_warn_backend_once`` emits each distinct message at most once."""
    k = XLACustomKernel('m13_dedup_probe')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        k._warn_backend_once('message-A')
        k._warn_backend_once('message-A')  # duplicate -> suppressed
        k._warn_backend_once('message-B')
    messages = [str(w.message) for w in caught]
    assert messages == ['message-A', 'message-B']


@pytest.mark.skipif(not numba_installed, reason='Numba not installed')
def test_m13_unregistered_global_backend_warns_and_falls_back():
    """A global backend absent for this platform warns, then uses the default."""
    import numba
    from brainevent import numba_kernel

    def gen(**kwargs):
        @numba.njit
        def add_one(x, out):
            for i in range(x.size):
                out[i] = x[i] + 1.0

        def kernel(x):
            return numba_kernel(add_one, outs=kwargs['outs'])(x)

        return kernel

    prim = XLACustomKernel('m13_global_mismatch_probe')
    prim.def_numba_kernel(gen)  # 'numba' becomes the cpu default automatically

    cpu = jax.devices('cpu')[0]
    x = jax.device_put(jnp.arange(8, dtype=jnp.float32), cpu)

    # Request a global cpu backend that was never registered for this primitive.
    set_backend('cpu', 'warp')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = prim(x, outs=[jax.ShapeDtypeStruct((8,), jnp.float32)])
            jax.block_until_ready(out)
        text = ' '.join(str(w.message) for w in caught)
        assert 'warp' in text and 'ignoring' in text.lower(), text
        # ...and the fallback still produced the correct numbers.
        np.testing.assert_allclose(np.asarray(out[0]), np.arange(8) + 1.0)
    finally:
        clear_backends()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
