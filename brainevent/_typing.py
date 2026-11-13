# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""
This module defines several type aliases using the `typing` module.

Type Aliases:
    MatrixShape: A tuple representing the shape of a matrix, with two integers.
    Data: A union type that can be a `jax.Array`, `numpy.ndarray`, `brainunit.Quantity`, or scalar `numbers.Number`.
    ArrayData: Like `Data` but restricted to array-like values (no bare Python scalars).
    Index: A union type that can be either a `jax.Array` or `numpy.ndarray`.
    Row: Alias for Index, representing a row index.
    Col: Alias for Index, representing a column index.
    Indptr: Alias for Index, representing an index pointer.
    Kernel: A callable type, representing a function or method.
"""

import numbers
from typing import Union, Tuple, Callable, Literal

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

# A tuple representing the shape of a matrix, with two integers.
MatrixShape = Tuple[int, int]

# A union type that can be a jax.Array, numpy.ndarray, brainunit.Quantity, or scalar number.
Data = Union[jax.Array, np.ndarray, u.Quantity, numbers.Number]

# Array-like data only (stored matrix values after ``asarray`` coercion): unlike
# ``Data``, bare Python scalars are excluded, so attribute access such as
# ``.shape`` / ``.dtype`` / ``.reshape`` is always valid.
ArrayData = Union[jax.Array, np.ndarray, u.Quantity]

# A union type that can be either a jax.Array or numpy.ndarray.
Index = Union[jax.Array, np.ndarray]

# Alias for Index, representing a row index.
Row = Index

# Alias for Index, representing a column index.
Col = Index

# Alias for Index, representing an index pointer.
Indptr = Index

# A callable type, representing a function or method.
Kernel = Callable

# kernel function generator
KernelGenerator = Callable[..., Kernel]

# Represents a scalar weight value, which can be a number, NumPy array, JAX array, or BrainUnit quantity.
WeightScalar = Union[numbers.Number, np.ndarray, jax.Array, u.Quantity]

# Represents a probability value, which can be a float, NumPy array, or JAX array.
Prob = Union[float, np.ndarray, jax.Array]

# Represents a seed value for random number generation, which can be an integer, NumPy array, or JAX array.
Seed = Union[int, np.ndarray, jax.Array]

# Represents a random key for Pallas, which is a tuple of four uint32 values.
PallasRandomKey = Tuple[jnp.uint32, jnp.uint32, jnp.uint32, jnp.uint32]  # uint32, 4
