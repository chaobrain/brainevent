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

__all__ = [
    'Primitive',
    'Tracer',
    'apply_primitive',
    'init_zero',
    'pallas_triton_params',
    'pallas_mosaic_tpu_params',
]

import jax
from jax.interpreters import ad

# ``apply_primitive`` is the eager (impl) evaluator for a primitive.  It has
# lived at ``jax.interpreters.xla.apply_primitive`` for a long time but that
# module is a thinning legacy shim; newer jax exposes the same function from
# ``jax._src.dispatch``.  Resolve it once here so a jax version bump only needs
# a change in this file rather than at every ``def_impl`` call site (L3).
try:
    from jax.interpreters.xla import apply_primitive
except (ImportError, AttributeError):  # pragma: no cover - version-dependent path
    from jax._src.dispatch import apply_primitive


def pallas_triton_params():
    if jax.__version_info__ < (0, 9, 1):
        return dict(backend='triton')
    else:
        from jax.experimental.pallas import triton as pltriton
        return dict(compiler_params=pltriton.CompilerParams())


def pallas_mosaic_tpu_params():
    if jax.__version_info__ < (0, 9, 1):
        return dict(backend='mosaic_tpu')
    else:
        from jax.experimental.pallas import tpu as pltpu
        return dict(compiler_params=pltpu.TPUCompilerParams())

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Primitive
else:
    from jax.extend.core import Primitive

from jax.core import Tracer


def init_zero(a):
    if jax.__version_info__ < (0, 4, 34):
        return ad.Zero.from_value(a)
    elif jax.__version_info__ < (0, 9, 1):
        return ad.Zero.from_primal_value(a)
    else:
        return ad.Zero(a.aval.to_tangent_aval())


