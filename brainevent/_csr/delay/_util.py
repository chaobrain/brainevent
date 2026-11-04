"""Head validation and JAX-compatible device error checks for region delay."""

from typing import Any

import jax
import jax.numpy as jnp

from brainevent._compatible_import import match_varying_axes


def _raise_if_bad_head(head: Any, slots: int) -> None:
    if not 0 <= int(head) < slots:
        raise ValueError(f"head must satisfy 0 <= head < L={slots}; got {int(head)}")


def _raise_device_error(code: Any) -> None:
    code = int(code)
    messages = {1: "task capacity/reservation error", 2: "head must satisfy 0 <= head < L",
                3: "inconsistent delay/task metadata"}
    if code:
        raise ValueError("region_delay: " + messages.get(code, f"device error {code}"))


def _check_projection(ring, events, weights, indices, indptr, head, workspace, *, weighted):
    """Validate mutable operands without binding or wrapping the delivery primitive."""
    if ring.ndim != 2 or ring.shape[0] == 0:
        raise ValueError("buffer must have shape (positive L, N)")
    if head.shape != (1,) or head.dtype != jnp.int32:
        raise TypeError("head must be int32[1], with 0 <= head < L")
    rows = indptr.size - 1
    if events.shape != (rows,):
        raise ValueError(f"events must have shape ({rows},)")
    if ring.dtype not in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64) or weights.dtype != ring.dtype:
        raise TypeError("require float16/bfloat16/float32/float64 weights and ring of one dtype")
    if weighted:
        if events.dtype != ring.dtype:
            raise TypeError("weighted events must share the weights/ring dtype")
    elif events.dtype not in (jnp.bool_, jnp.float32):
        raise TypeError("binary events must be bool or float32")
    if weighted:
        if len(workspace) != 5:
            raise TypeError("weighted projection requires WeightedHybridWorkspace arrays")
        if workspace[3].shape != workspace[0].shape or workspace[3].dtype != ring.dtype:
            raise TypeError("task_value must match task capacity and weights/ring dtype")
    elif len(workspace) != 4:
        raise TypeError("binary projection requires HybridWorkspace arrays")
    if indptr.dtype == jnp.int32 and indices.size > 2**31 - 1:
        raise ValueError("int32 indptr cannot address more than INT32_MAX edges; use int64")


def _checked_result(value: jax.Array, code: jax.Array, *, head_only: bool = False) -> jax.Array:
    """Tie an error to this result, without poisoning later ordered effects."""
    if head_only:
        code = jnp.where(code == 2, code, 0)
    return _check_error(value, code)


@jax.custom_jvp
def _check_error(value: jax.Array, code: jax.Array) -> jax.Array:
    if isinstance(code, jax.core.Tracer):
        def fail(error):
            # vmap may evaluate both branches of a conditional. Good batch
            # elements must return their value even on this callback path.
            def check(result, status):
                _raise_device_error(status)
                return result

            result = jax.pure_callback(
                check, jax.ShapeDtypeStruct(value.shape, value.dtype), value, error,
                vmap_method="sequential")
            # pure_callback constructs its result type from shape/dtype only.
            # It returns this shard's value, so preserve that value's manual
            # axes before joining the two cond branches. The custom JVP below
            # keeps the derivative an identity without transposing this cast.
            return match_varying_axes(result, value)

        return jax.lax.cond(code != 0, fail, lambda _: value, code)
    else:
        _raise_device_error(code)
    return value


@_check_error.defjvp
def _check_error_jvp(primals, tangents):
    value, code = primals
    return _check_error(value, code), tangents[0]
