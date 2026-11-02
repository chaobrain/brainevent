"""Sparse pattern helpers shared by sampled-product leaf implementations."""

from typing import Literal

import jax
import jax.numpy as jnp

from brainevent._misc import _csr_to_coo

Mode = Literal['float', 'binary', 'weighted']
VALUE_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)


def coordinates(indices: jax.Array, indptr: jax.Array | None) -> tuple[jax.Array, jax.Array]:
    """Read physical-slot coordinates without reordering or coalescing."""
    if indptr is None:
        return indices[:, 0], indices[:, 1]
    return _csr_to_coo(indices, indptr)


def check_jax_backend(backend: str | None) -> None:
    """Validate a leaf's portable backend selection."""
    if backend not in (None, 'auto', 'jax_raw', 'cuda_raw'):
        raise ValueError(f'unsupported sampled-product backend: {backend!r}')
