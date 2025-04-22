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

import functools

import jax
from jax import tree_util
from jax.interpreters import ad

from ._compatible_import import Primitive

__all__ = [
    'defjvp',
]


def defjvp(primitive, *jvp_rules):
    """
    Define JVP rules for any JAX primitive.

    This function is similar to ``jax.interpreters.ad.defjvp``.
    However, the JAX one only supports primitive with ``multiple_results=False``.
    ``brainevent.defjvp`` enables to define the independent JVP rule for
    each input parameter no matter ``multiple_results=False/True``.

    For examples, please see ``test_ad_support.py``.

    Args:
      primitive: Primitive, XLACustomOp.
      *jvp_rules: The JVP translation rule for each primal.
    """
    from ._xla_custom_op import XLACustomKernel

    if isinstance(primitive, XLACustomKernel):
        primitive = primitive.primitive
    assert isinstance(primitive, Primitive), f'The primitive should be a JAX primitive. But we got {primitive}'
    if primitive.multiple_results:
        ad.primitive_jvps[primitive] = functools.partial(_standard_jvp, jvp_rules, primitive)
    else:
        ad.primitive_jvps[primitive] = functools.partial(ad.standard_jvp, jvp_rules, primitive)


def _standard_jvp(jvp_rules, primitive: Primitive, primals, tangents, **params):
    assert primitive.multiple_results
    val_out = tuple(primitive.bind(*primals, **params))
    tree = tree_util.tree_structure(val_out)
    tangents_out = []
    for rule, t in zip(jvp_rules, tangents):
        if rule is not None and type(t) is not ad.Zero:
            r = tuple(rule(t, *primals, **params))
            tangents_out.append(r)
            assert tree_util.tree_structure(r) == tree
    r = functools.reduce(
        _add_tangents,
        tangents_out,
        tree_util.tree_map(
            # compatible with JAX 0.4.34
            lambda a: (
                ad.Zero.from_primal_value(a)
                if jax.__version_info__ >= (0, 4, 34) else
                ad.Zero.from_value(a)
            ),
            val_out
        )
    )
    return val_out, r


def _add_tangents(xs, ys):
    return tree_util.tree_map(ad.add_tangents, xs, ys, is_leaf=lambda a: isinstance(a, ad.Zero))
