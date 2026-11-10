# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Head validation and JAX-compatible device error checks for region delay."""

from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from brainevent._compatible_import import match_varying_axes


_T = TypeVar("_T")


def _head_out_of_bounds(head: Any, slots: int) -> Any:
    """Compare in the input's original integer width, before any narrowing."""
    return (head < 0) | (head >= slots)


def _checked_head(value: Any, slots: int) -> Any:
    """Validate a scalar slot and return it without changing its dtype."""
    invalid = _head_out_of_bounds(value, slots)
    if isinstance(invalid, jax.core.Tracer):
        return _checked_result(value, jnp.where(invalid, jnp.int32(2), jnp.int32(0)))
    if invalid:
        raise ValueError(f"head must satisfy 0 <= head < L={slots}; got {int(value)}")
    return value


def _raise_device_error(code: Any) -> None:
    code = int(code)
    messages = {1: "task capacity/reservation error", 2: "head must satisfy 0 <= head < L",
                3: "inconsistent delay/task metadata"}
    if code:
        raise ValueError("region_delay: " + messages.get(code, f"device error {code}"))


def _checked_result(value: _T, code: jax.Array, *, head_only: bool = False) -> _T:
    """Tie an error to every leaf through one callback, without ordered effects."""
    if head_only:
        code = jnp.where(code == 2, code, 0)
    return _check_error(value, code)


@jax.custom_jvp
def _check_error(value: _T, code: jax.Array) -> _T:
    if isinstance(code, jax.core.Tracer):
        def fail(error):
            # vmap may evaluate both branches of a conditional. Good batch
            # elements must return their value even on this callback path.
            def check(result, status):
                _raise_device_error(status)
                return result

            outputs = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), value)
            result = jax.pure_callback(check, outputs, value, error, vmap_method="sequential")
            # pure_callback constructs its result type from shape/dtype only.
            # It returns this shard's value, so preserve that value's manual
            # axes before joining the two cond branches. The custom JVP below
            # keeps the derivative an identity without transposing this cast.
            return jax.tree.map(match_varying_axes, result, value)

        return jax.lax.cond(code != 0, fail, lambda _: value, code)
    else:
        _raise_device_error(code)
    return value


@_check_error.defjvp
def _check_error_jvp(primals, tangents):
    value, code = primals
    return _check_error(value, code), tangents[0]
