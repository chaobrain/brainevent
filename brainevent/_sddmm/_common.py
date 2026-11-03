"""Backend metadata shared by sampled-product leaf implementations."""

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from brainevent._op import XLACustomKernel

Mode = Literal['float', 'binary', 'weighted']
VALUE_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)
# Both native leaves expose the same value/event ABI pairs.
VALUE_SUFFIXES = dict(zip(map(jnp.dtype, VALUE_DTYPES), ('f16', 'bf16', 'f32', 'f64')))
BINARY_TARGET_SUFFIXES = {
    (value_dtype, event_dtype): f'{suffix}_{event_suffix}'
    for value_dtype, suffix in VALUE_SUFFIXES.items()
    for event_dtype, event_suffix in [(jnp.dtype(jnp.bool_), 'bool'), (jnp.dtype(jnp.float32), 'float')]
}
BINARY_TARGET_SUFFIXES.update({
    (jnp.dtype(jnp.float64), jnp.dtype(jnp.int8)): 'f64_int8',
    (jnp.dtype(jnp.float64), jnp.dtype(jnp.float64)): 'f64_double',
})


@dataclass(frozen=True)
class _CSRBackend:
    """Carry the forward CSR backend selection through gradient dispatch."""

    primitive: 'XLACustomKernel'
    requested: str | None


def check_jax_backend(backend: str | None | _CSRBackend) -> None:
    """Validate a leaf's portable backend selection."""
    if isinstance(backend, _CSRBackend):
        return
    if backend not in (None, 'auto', 'jax_raw', 'cuda_raw'):
        raise ValueError(f'unsupported sampled-product backend: {backend!r}')
