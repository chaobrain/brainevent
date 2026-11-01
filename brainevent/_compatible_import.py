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
    'varying_axes',
    'with_varying_axes',
    'pvary',
    'match_varying_axes',
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


# Varying manual axes (``vma``) under ``jax.shard_map`` (#245).
#
# Inside ``shard_map`` with ``check_vma=True``, every abstract value records the
# manual mesh axes it varies over. Only the spelling differs between versions, and
# both spellings are public: jax 0.9 has ``aval.vma`` (a frozenset), while jax 0.10
# and later have ``aval.manual_axis_type.varying``. The constructor keyword follows
# the same rename (``vma=`` / ``manual_axis_type=``, as in ``jax.ShapeDtypeStruct``).
# The helpers below hold the whole difference.

def varying_axes(aval) -> frozenset:
    """The manual mesh axes that ``aval`` varies over (its ``vma``).

    Parameters
    ----------
    aval : jax.core.AbstractValue
        An abstract value, such as ``jax.typeof(x)``.

    Returns
    -------
    frozenset
        The varying axis names. Empty outside ``shard_map``, under
        ``check_vma=False``, and for abstract values without the field.
    """
    mat = getattr(aval, 'manual_axis_type', None)
    if mat is not None:  # jax >= 0.10
        return mat.varying
    return getattr(aval, 'vma', None) or frozenset()  # jax 0.9


def with_varying_axes(aval, vma: frozenset, mesh):
    """``aval`` typed as varying over the manual axes ``vma`` of ``mesh``.

    Parameters
    ----------
    aval : jax.core.ShapedArray
        The abstract value to re-type; its shape and dtype are kept.
    vma : frozenset
        Non-empty set of manual axis names of ``mesh``.
    mesh : jax.sharding.AbstractMesh
        The mesh the axes belong to, e.g. ``varying_input.sharding.mesh``.

    Returns
    -------
    jax.core.ShapedArray
        The same shape and dtype, replicated over ``mesh`` and varying over ``vma``.
    """
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if hasattr(aval, 'manual_axis_type'):  # jax >= 0.10
        return aval.update(sharding=sharding, manual_axis_type=jax.sharding.ManualAxisType(varying=vma))
    return aval.update(sharding=sharding, vma=vma)  # jax 0.9


def pvary(x, axes):
    """Mark ``x`` as varying over the manual axes ``axes``.

    The value is unchanged. In reverse mode the cotangent is summed over ``axes``
    (``psum``), which is the gradient of an operand replicated over them.

    Parameters
    ----------
    x : jax.Array
        A value inside ``shard_map``.
    axes : tuple
        Manual axis names to add to the varying set of ``x``.

    Returns
    -------
    jax.Array
        ``x`` typed as varying over ``axes`` as well.
    """
    return jax.lax.pcast(x, tuple(axes), to='varying')


def match_varying_axes(x, *operands):
    """``x`` marked varying over every manual axis that one of ``operands`` varies over.

    For the results of jax operations that type their output as replicated under
    ``shard_map`` although it is computed per shard from varying operands (such as
    ``jax.experimental.sparse.csr_matvec``, ``csr_matmat`` and
    ``bcoo_dot_general_sampled``).

    Parameters
    ----------
    x : jax.Array
        The result to re-type; its value is unchanged.
    *operands : jax.Array
        The operands ``x`` was computed from.

    Returns
    -------
    jax.Array
        ``x``, varying over the union of the operands' varying axes as well.
    """
    vma = frozenset().union(*(varying_axes(jax.typeof(o)) for o in operands))
    missing = vma - varying_axes(jax.typeof(x))
    return pvary(x, sorted(missing, key=str)) if missing else x
