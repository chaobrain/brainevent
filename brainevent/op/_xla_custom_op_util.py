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

# -*- coding: utf-8 -*-

import functools

import jax
from jax import tree_util
from jax.interpreters import ad

from brainevent._compatible_import import Primitive

__all__ = [
    'defjvp',
    'general_batching_rule',
]


def defjvp(primitive, *jvp_rules):
    """
    Define JVP rules for any JAX primitive.

    This function allows defining Jacobian-vector product (JVP) rules for JAX primitives,
    extending the functionality of ``jax.interpreters.ad.defjvp``. While the standard
    JAX function primarily supports primitives that return a single result
    (``multiple_results=False``), this implementation supports defining independent
    JVP rules for each input parameter regardless of whether the primitive returns
    single or multiple results.

    This is particularly useful for custom operations or primitives where different
    inputs might have different differentiation rules or where the primitive naturally
    produces multiple outputs that need distinct handling in automatic differentiation.

    For concrete usage examples, refer to the test file ``test_ad_support.py``.

    Args:
        primitive: The JAX ``Primitive`` object or an ``XLACustomKernel`` instance
            for which the JVP rule is being defined. If an ``XLACustomKernel`` is
            provided, its underlying ``Primitive`` is extracted.
        *jvp_rules: A variable number of functions, each representing the JVP rule
            corresponding to a primal input argument of the primitive. Each rule
            function should accept the tangent vector for its corresponding primal input,
            followed by all the primal inputs, and any keyword arguments passed to the
            primitive. It should return the tangent vector(s) corresponding to the
            primitive's output(s). If a rule is ``None``, it implies the JVP for that
            input is zero.
    """
    # Import XLACustomKernel locally to avoid circular dependencies.
    from ._xla_custom_op import XLACustomKernel

    # If the input is an XLACustomKernel, extract the underlying JAX primitive.
    if isinstance(primitive, XLACustomKernel):
        primitive = primitive.primitive
    # Ensure that the 'primitive' argument is indeed a JAX Primitive object.
    assert isinstance(primitive, Primitive), f'The primitive should be a JAX primitive. But we got {primitive}'

    # Check if the primitive returns multiple results.
    if primitive.multiple_results:
        # If yes, use the custom _standard_jvp function designed to handle multiple results.
        # ad.primitive_jvps is the JAX registry for JVP rules.
        # functools.partial pre-fills the jvp_rules and primitive arguments for _standard_jvp.
        ad.primitive_jvps[primitive] = functools.partial(_standard_jvp, jvp_rules, primitive)
    else:
        # If no (single result), use the standard JAX JVP handler (ad.standard_jvp).
        # This maintains compatibility with standard JAX behavior for single-result primitives.
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


def general_batching_rule(prim, args, axes, **kwargs):
    """
    Implements a general batching rule for JAX primitive operations.

    This function handles batching for JAX primitives by separating batched and non-batched
    arguments, then applying the primitive to each element in the batch using jax.lax.scan.

    Args:
        prim: The JAX primitive operation to be batched.
        args: Sequence of input arguments to the primitive.
        axes: Sequence of axis indices indicating the batch dimension for each argument.
              None indicates that the corresponding argument is not batched.
        **kwargs: Additional keyword arguments to pass to the primitive.

    Returns:
        tuple: A tuple containing:
            - outs: The batched outputs from applying the primitive.
            - out_dim: A pytree with the same structure as outs, indicating
              the batch dimensions of each output.

    Note:
        This function moves all batch dimensions to the leading axis (0) before
        applying scan, then processes each slice of the batched inputs.
    """
    batch_axes, batch_args, non_batch_args = [], {}, {}
    for ax_i, ax in enumerate(axes):
        if ax is None:
            non_batch_args[f'ax{ax_i}'] = args[ax_i]
        else:
            batch_args[f'ax{ax_i}'] = args[ax_i] if ax == 0 else jax.numpy.moveaxis(args[ax_i], ax, 0)
            batch_axes.append(ax_i)

    def f(_, x):
        """
        Internal function for jax.lax.scan that applies the primitive to a single batch element.

        Args:
            _: Carry value (unused).
            x: Dictionary containing the current batch slice for each batched argument.

        Returns:
            tuple: (carry value, primitive output)
        """
        pars = tuple(
            [(x[f'ax{i}'] if i in batch_axes else non_batch_args[f'ax{i}'])
             for i in range(len(axes))]
        )
        return 0, prim.bind(*pars, **kwargs)

    _, outs = jax.lax.scan(f, 0, batch_args)
    out_vals, out_tree = jax.tree.flatten(outs)
    out_dim = jax.tree.unflatten(out_tree, (0,) * len(out_vals))
    return outs, out_dim
