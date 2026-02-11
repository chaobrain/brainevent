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

import abc

import jax
import jax.numpy as jnp
import numpy as np

from ._typing import PallasRandomKey
from .config import get_lfsr_algorithm

__all__ = [
    'PallasLFSR88RNG',
    'PallasLFSR113RNG',
    'PallasLFSR128RNG',
    'get_pallas_lfsr_rng_class',
    'PallasLFSRRNG',
]


class LFSRBase(abc.ABC):
    """Abstract base class for Linear Feedback Shift Register random number generators.

    This class defines the common interface and functionality for LFSR-based
    random number generators such as LFSR88, LFSR113, and LFSR128.  It handles
    the basic operations for managing the generator state and defines abstract
    methods that concrete implementations must provide.

    Parameters
    ----------
    seed : int
        An integer used to initialize the random state.  The concrete
        subclass determines how the seed is expanded into the full
        internal state.

    See Also
    --------
    PallasLFSR88RNG : Combined LFSR88 generator with period ~2^88.
    PallasLFSR113RNG : Combined LFSR113 generator with period ~2^113.
    PallasLFSR128RNG : Combined LFSR128 generator with period ~2^128.

    Notes
    -----
    LFSR (Linear Feedback Shift Register) algorithms are efficient
    pseudorandom number generators based on bitwise shift and XOR
    operations.  A single LFSR advances its state by shifting bits and
    feeding back a linear combination (XOR) of selected bit positions.
    The "combined" variants used here (LFSR88, LFSR113, LFSR128) run
    multiple independent LFSRs in parallel and XOR their outputs to
    produce the final random value.  This improves the statistical
    quality and extends the period well beyond what a single LFSR can
    achieve.

    These generators are particularly well-suited for GPU execution via
    JAX Pallas kernels because:

    - The state is compact (4 x ``uint32`` values).
    - Each step requires only bitwise shifts, masks, and XOR -- all of
      which map directly to single GPU instructions.
    - No division, modulo, or memory-indirect operations are needed,
      making them ideal for high-throughput parallel random number
      generation inside Pallas grid kernels.

    The internal state is stored as a tuple of four ``jnp.uint32``
    scalars.  The class is registered as a JAX pytree node (in concrete
    subclasses) so that instances can be passed through ``jax.jit``,
    ``jax.vmap``, and other JAX transformations.

    Examples
    --------
    .. code-block:: python

        >>> # Create a concrete LFSR implementation
        >>> rng = PallasLFSR113RNG(seed=42)
        >>> random_float = rng.rand()
        >>> random_int = rng.randint()
    """

    def __init__(self, seed: int):
        """Initialize the random number generator with a seed.

        Parameters
        ----------
        seed : int
            An integer used to initialize the random state.
        """
        self._key = self.generate_key(seed)

    @property
    def key(self) -> PallasRandomKey:
        """Get the current random state key.

        Returns
        -------
        PallasRandomKey
            The current state of the random number generator, as a
            tuple of four ``jnp.uint32`` scalars.

        See Also
        --------
        generate_key : Create an initial key from a seed.
        generate_next_key : Advance the state by one step.
        """
        return self._key

    @key.setter
    def key(self, value: PallasRandomKey):
        """Set the random state key.

        Validates that the provided key is a tuple of 4 ``jax.Array``
        elements, each with dtype ``uint32``, before setting it as the
        current state.

        Parameters
        ----------
        value : PallasRandomKey
            The new state to set for the random number generator.  Must
            be a tuple of exactly 4 ``jax.Array`` or ``numpy.ndarray``
            elements with dtype ``jnp.uint32``.

        Raises
        ------
        TypeError
            If *value* is not a tuple of length 4, or if any element is
            not a ``jax.Array`` or ``numpy.ndarray``.
        ValueError
            If any element of *value* does not have dtype ``jnp.uint32``.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG(seed=0)
            >>> new_key = tuple(jnp.asarray(i, dtype=jnp.uint32) for i in [10, 20, 30, 40])
            >>> rng.key = new_key
        """
        if not isinstance(value, tuple) or len(value) != 4:
            raise TypeError("Key must be a tuple of length 4")
        for i, val in enumerate(value):
            if not isinstance(val, (jax.Array, np.ndarray)):
                raise TypeError(f"Key element {i} must be a jnp.ndarray")
            if val.dtype != jnp.uint32:
                raise ValueError(f"Key element {i} must be of type jnp.uint32")
        self._key = value

    @abc.abstractmethod
    def generate_key(self, seed: int) -> PallasRandomKey:
        """Initialize the random key from a seed value.

        This method must be implemented by concrete subclasses to create
        the initial state from a seed value.

        Parameters
        ----------
        seed : int
            An integer used to initialize the random state.

        Returns
        -------
        PallasRandomKey
            The initial state of the random number generator, as a
            tuple of four ``jnp.uint32`` scalars.

        See Also
        --------
        generate_next_key : Advance the state by one step.
        """
        pass

    @abc.abstractmethod
    def generate_next_key(self) -> PallasRandomKey:
        """Generate the next random key and update the internal state.

        This method must be implemented by concrete subclasses to advance
        the random state by one iteration according to the specific LFSR
        algorithm.

        Returns
        -------
        PallasRandomKey
            The new state of the random number generator after one
            iteration.

        See Also
        --------
        generate_key : Create an initial key from a seed.

        Notes
        -----
        This method mutates the internal ``_key`` attribute in place.
        """
        pass

    @abc.abstractmethod
    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float in [0, 1).

        Returns
        -------
        jax.Array
            A scalar ``float32`` value in the range [0, 1).

        See Also
        --------
        randint : Generate a random 32-bit unsigned integer.
        uniform : Generate a random float in an arbitrary range.
        """
        pass

    @abc.abstractmethod
    def randint(self) -> jax.Array:
        """Generate a uniformly distributed random 32-bit unsigned integer.

        Returns
        -------
        jax.Array
            A scalar ``uint32`` value in the range [0, 2^32 - 1].

        See Also
        --------
        rand : Generate a random float in [0, 1).
        random_integers : Generate a random integer in a specified range.
        """
        pass

    @abc.abstractmethod
    def randn(self, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number from the standard normal distribution N(0, 1).

        Parameters
        ----------
        epsilon : float, optional
            A small positive value used to clamp the uniform input away
            from zero before applying ``log``, preventing ``-inf``.
            Defaults to ``1e-10``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value sampled from N(0, 1).

        See Also
        --------
        normal : Generate a value from N(mu, sigma).

        Notes
        -----
        Concrete implementations use the Box-Muller transform to convert
        two uniform samples into a normally distributed value.
        """
        pass

    def uniform(self, low: float, high: float) -> jax.Array:
        """Generate a uniformly distributed random float in [low, high).

        Maps a random value from :meth:`rand` (in [0, 1)) to the
        specified range via ``rand() * (high - low) + low``.

        Parameters
        ----------
        low : float
            The lower bound of the range (inclusive).
        high : float
            The upper bound of the range (exclusive).

        Returns
        -------
        jax.Array
            A scalar ``float32`` value in the range [low, high).

        See Also
        --------
        rand : Generate a random float in [0, 1).
        normal : Generate a value from a normal distribution.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG(seed=42)
            >>> value = rng.uniform(10.0, 20.0)  # Random value between 10 and 20
        """
        r = self.rand()
        return r * (high - low) + low

    def normal(self, mu: float, sigma: float, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number from the normal distribution N(mu, sigma).

        Uses the :meth:`randn` method to generate a standard normal value
        and then scales and shifts it to the desired mean and standard
        deviation.

        Parameters
        ----------
        mu : float
            The mean of the normal distribution.
        sigma : float
            The standard deviation of the normal distribution.
        epsilon : float, optional
            A small positive value forwarded to :meth:`randn` to avoid
            numerical issues.  Defaults to ``1e-10``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value sampled from N(mu, sigma^2).

        See Also
        --------
        randn : Generate a standard normal value.
        uniform : Generate a uniform random value.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR113RNG(seed=42)
            >>> value = rng.normal(0.0, 1.0)  # Standard normal
            >>> value = rng.normal(5.0, 2.0)  # N(5, 4)
        """
        r = self.randn(epsilon)
        return mu + sigma * r

    def random_integers(self, low: int, high: int) -> jax.Array:
        """Generate a uniformly distributed random integer in [low, high].

        Parameters
        ----------
        low : int
            The lower bound of the range (inclusive).
        high : int
            The upper bound of the range (inclusive).

        Returns
        -------
        jax.Array
            A scalar integer value in the range [low, high].

        See Also
        --------
        randint : Generate a raw 32-bit unsigned integer.
        uniform : Generate a uniform float in a range.

        Notes
        -----
        The mapping uses modular arithmetic:
        ``randint() % (high + 1 - low) + low``.  This introduces a
        slight bias when ``(high - low + 1)`` does not evenly divide
        ``2^32``, but the bias is negligible for typical ranges.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG(seed=42)
            >>> dice_roll = rng.random_integers(1, 6)  # Random integer from 1 to 6
            >>> coin_flip = rng.random_integers(0, 1)  # 0 or 1
        """
        val = self.randint()
        return val % (high + 1 - low) + low

    def tree_flatten(self):
        """Flatten the RNG object for JAX pytree utilities.

        Returns
        -------
        children : tuple
            A single-element tuple containing the RNG key.
        aux_data : tuple
            An empty tuple (no auxiliary data is needed).

        See Also
        --------
        tree_unflatten : Reconstruct an RNG from flattened data.

        Notes
        -----
        This method, together with :meth:`tree_unflatten`, allows
        LFSR RNG instances to be passed through ``jax.jit``,
        ``jax.vmap``, and other JAX transformations that require pytree
        support.
        """
        return (self.key,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct an RNG object from flattened pytree data.

        Parameters
        ----------
        aux_data : tuple
            An empty tuple (no auxiliary data is needed).
        children : tuple
            A single-element tuple containing the RNG key.

        Returns
        -------
        LFSRBase
            A new RNG instance with the given key as its state.

        See Also
        --------
        tree_flatten : Flatten an RNG into pytree leaves.
        """
        obj = object.__new__(cls)
        key, = children
        obj.key = key
        return obj


#############################################
# Random Number Generator: LFSR88 algorithm #
#############################################

@jax.tree_util.register_pytree_node_class
class PallasLFSR88RNG(LFSRBase):
    """Combined LFSR random number generator by L'Ecuyer (LFSR88).

    Implements the LFSR88 algorithm, a combined Linear Feedback Shift
    Register random number generator developed by Pierre L'Ecuyer.  The
    algorithm combines three independent LFSRs to produce high-quality
    pseudorandom numbers with a period of approximately 2^88.

    Parameters
    ----------
    seed : int
        An integer used to initialize the three-component state.  The
        seed is offset by ``+2``, ``+8``, and ``+16`` for the three
        components to satisfy the minimum-seed constraints of the
        algorithm.

    See Also
    --------
    PallasLFSR113RNG : Four-component variant with period ~2^113.
    PallasLFSR128RNG : Four-component variant with period ~2^128.
    LFSRBase : Abstract base class defining the LFSR interface.

    Notes
    -----
    The LFSR88 algorithm combines three Tausworthe generators with
    the following parameters:

    - Component 1: shift (13, 19), mask ``0xFFFFFFFE``, left-shift 12
    - Component 2: shift (2, 25), mask ``0xFFFFFFF8``, left-shift 4
    - Component 3: shift (3, 11), mask ``0xFFFFFFF0``, left-shift 17

    Each component advances independently, and the final output is the
    XOR of all three component values.  The fourth element of the
    internal key tuple is unused by the generation algorithm and is set
    to zero.

    The implementation is based on L'Ecuyer's original C code:
    https://github.com/cmcqueen/simplerandom/blob/main/c/lecuyer/lfsr88.c

    Examples
    --------
    .. code-block:: python

        >>> rng = PallasLFSR88RNG(seed=42)
        >>> rand_float = rng.rand()        # Random float in [0, 1)
        >>> rand_int = rng.randint()        # Random 32-bit unsigned integer
        >>> norm_val = rng.normal(0, 1)     # Value from N(0, 1)
        >>> unif_val = rng.uniform(5.0, 10.0)  # Float in [5, 10)
        >>> dice = rng.random_integers(1, 6)   # Integer from 1 to 6
    """
    __module__ = 'brainevent'

    def generate_key(self, seed: int) -> PallasRandomKey:
        """Initialize the random key of the LFSR88 algorithm.

        Creates a 4-element state tuple from the given seed, ensuring
        that each element meets the minimum required value to guarantee
        proper algorithm function.

        Parameters
        ----------
        seed : int
            An integer seed value used to initialize the generator
            state.

        Returns
        -------
        PallasRandomKey
            A tuple of four ``jnp.uint32`` scalars containing the
            initial state.  The fourth element is set to ``0`` as it is
            not used by the LFSR88 algorithm.

        Notes
        -----
        The LFSR88 algorithm requires that the initial seeds are at
        least 2, 8, and 16 for the three components respectively.
        This method adds these offsets to the provided seed to ensure
        the constraint is always satisfied.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG.__new__(PallasLFSR88RNG)
            >>> key = rng.generate_key(42)
            >>> len(key)
            4
        """
        return (
            jnp.asarray(seed + 2, dtype=jnp.uint32),
            jnp.asarray(seed + 8, dtype=jnp.uint32),
            jnp.asarray(seed + 16, dtype=jnp.uint32),
            jnp.asarray(0, dtype=jnp.uint32)
        )

    def generate_next_key(self) -> PallasRandomKey:
        """Generate the next random key and update the internal state.

        Computes the next state of the LFSR88 generator by applying the
        three-component LFSR transformations to the current state.

        Returns
        -------
        PallasRandomKey
            A tuple of four ``jnp.uint32`` scalars containing the new
            state after one iteration.

        Notes
        -----
        This method mutates the internal ``_key`` attribute.  The fourth
        element stores the last intermediate value ``b`` from the third
        component, though this is not part of the original algorithm's
        state.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG(seed=42)
            >>> new_key = rng.generate_next_key()
            >>> len(new_key)
            4
        """
        key = self.key
        b = jnp.asarray(((key[0] << 13) ^ key[0]) >> 19, dtype=jnp.uint32)
        s1 = ((key[0] & jnp.asarray(4294967294, dtype=jnp.uint32)) << 12) ^ b
        b = ((key[1] << 2) ^ key[1]) >> 25
        s2 = ((key[1] & jnp.asarray(4294967288, dtype=jnp.uint32)) << 4) ^ b
        b = ((key[2] << 3) ^ key[2]) >> 11
        s3 = ((key[2] & jnp.asarray(4294967280, dtype=jnp.uint32)) << 17) ^ b
        # The original C code doesn't use the 4th element for generation,
        # but we store 'b' there for potential future use or consistency.
        new_key = (
            jnp.asarray(s1, dtype=jnp.uint32),
            jnp.asarray(s2, dtype=jnp.uint32),
            jnp.asarray(s3, dtype=jnp.uint32),
            jnp.asarray(b, dtype=jnp.uint32)
        )
        self.key = new_key
        return new_key

    def randn(self, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number from the standard normal distribution N(0, 1).

        Uses the Box-Muller transform to convert two uniform random
        numbers into a normally distributed value.

        Parameters
        ----------
        epsilon : float, optional
            A small positive value used to clamp the first uniform
            sample away from zero, preventing ``log(0)``.  Defaults to
            ``1e-10``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value sampled from N(0, 1).

        See Also
        --------
        normal : Generate a value from N(mu, sigma).
        rand : Generate a uniform float in [0, 1).

        Notes
        -----
        The Box-Muller transform generates two independent standard
        normal values from two independent uniform values.  This
        implementation returns only the sine component.

        References: Box-Muller transform,
        https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG(seed=42)
            >>> value = rng.randn()  # Random value from standard normal
        """
        u1 = self.rand()
        u2 = self.rand()

        # Ensure u1 is not zero to avoid log(0)
        u1 = jnp.maximum(u1, epsilon)

        # Box-Muller transform
        mag = jnp.sqrt(-2.0 * jnp.log(u1))
        z2 = mag * jnp.sin(2 * jnp.pi * u2)  # Using sin component

        return z2

    def randint(self) -> jax.Array:
        """Generate a uniformly distributed random 32-bit unsigned integer.

        Advances the generator state and returns the XOR of the three
        state components.

        Returns
        -------
        jax.Array
            A scalar ``uint32`` value in the range [0, 2^32 - 1].

        See Also
        --------
        rand : Generate a random float in [0, 1).
        random_integers : Generate a random integer in a specified range.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG(seed=42)
            >>> value = rng.randint()
        """
        key = self.generate_next_key()
        return jnp.asarray(key[0] ^ key[1] ^ key[2], dtype=jnp.uint32)

    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float in [0, 1).

        Advances the generator state and converts the resulting integer
        to a floating-point number by multiplying with ``1 / (2^32 - 1)``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value in the range [0, 1).

        See Also
        --------
        randint : Generate a raw 32-bit unsigned integer.
        uniform : Generate a float in an arbitrary range.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR88RNG(seed=42)
            >>> value = rng.rand()  # e.g. 0.27183515
        """
        key = self.generate_next_key()
        # 2.3283064365386963e-10 is 1 / (2^32 - 1) approx
        return (key[0] ^ key[1] ^ key[2]) * 2.3283064365386963e-10


##############################################
# Random Number Generator: LFSR113 algorithm #
##############################################

@jax.tree_util.register_pytree_node_class
class PallasLFSR113RNG(LFSRBase):
    """Combined LFSR random number generator by L'Ecuyer (LFSR113).

    Implements the LFSR113 algorithm, a combined Linear Feedback Shift
    Register random number generator developed by Pierre L'Ecuyer.  The
    algorithm combines four independent LFSRs to produce high-quality
    pseudorandom numbers with a period of approximately 2^113.

    Parameters
    ----------
    seed : int
        An integer used to initialize the four-component state.  The
        seed is offset by ``+2``, ``+8``, ``+16``, and ``+128`` for
        the four components to satisfy the minimum-seed constraints.

    See Also
    --------
    PallasLFSR88RNG : Three-component variant with period ~2^88.
    PallasLFSR128RNG : Four-component variant with period ~2^128.
    LFSRBase : Abstract base class defining the LFSR interface.

    Notes
    -----
    The LFSR113 algorithm combines four Tausworthe generators with
    the following parameters:

    - Component 1: shift (6, 13), mask ``0xFFFFFFFE``, left-shift 18
    - Component 2: shift (2, 27), mask ``0xFFFFFFF8``, left-shift 2
    - Component 3: shift (13, 21), mask ``0xFFFFFFF0``, left-shift 7
    - Component 4: shift (3, 12), mask ``0xFFFFFF80``, left-shift 13

    Each component advances independently, and the final output is the
    XOR of all four component values.

    The implementation is based on L'Ecuyer's original C code:
    https://github.com/cmcqueen/simplerandom/blob/main/c/lecuyer/lfsr113.c

    Examples
    --------
    .. code-block:: python

        >>> rng = PallasLFSR113RNG(seed=42)
        >>> rand_float = rng.rand()        # Random float in [0, 1)
        >>> rand_int = rng.randint()        # Random 32-bit unsigned integer
        >>> norm_val = rng.normal(0, 1)     # Value from N(0, 1)
        >>> unif_val = rng.uniform(5.0, 10.0)  # Float in [5, 10)
        >>> dice = rng.random_integers(1, 6)   # Integer from 1 to 6
    """
    __module__ = 'brainevent'

    def generate_key(self, seed: int) -> PallasRandomKey:
        """Initialize the random key of the LFSR113 algorithm.

        Creates a 4-element state tuple from the given seed, ensuring
        that each element meets the minimum required value to guarantee
        proper algorithm function.

        Parameters
        ----------
        seed : int
            An integer seed value used to initialize the generator
            state.

        Returns
        -------
        PallasRandomKey
            A tuple of four ``jnp.uint32`` scalars containing the
            initial state.

        Notes
        -----
        The LFSR113 algorithm requires that the initial seeds are at
        least 2, 8, 16, and 128 for the four components respectively.
        This method adds these offsets to the provided seed to ensure
        the constraint is always satisfied.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR113RNG.__new__(PallasLFSR113RNG)
            >>> key = rng.generate_key(42)
            >>> len(key)
            4
        """
        return (
            jnp.asarray(seed + 2, dtype=jnp.uint32),
            jnp.asarray(seed + 8, dtype=jnp.uint32),
            jnp.asarray(seed + 16, dtype=jnp.uint32),
            jnp.asarray(seed + 128, dtype=jnp.uint32)
        )

    def generate_next_key(self) -> PallasRandomKey:
        """Generate the next random key and update the internal state.

        Computes the next state of the LFSR113 generator by applying the
        four-component LFSR transformations to the current state.

        Returns
        -------
        PallasRandomKey
            A tuple of four ``jnp.uint32`` scalars containing the new
            state after one iteration.

        Notes
        -----
        This method mutates the internal ``_key`` attribute.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR113RNG(seed=42)
            >>> new_key = rng.generate_next_key()
            >>> len(new_key)
            4
        """
        key = self.key
        z1 = key[0]
        z2 = key[1]
        z3 = key[2]
        z4 = key[3]
        b1 = ((z1 << 6) ^ z1) >> 13
        z1 = jnp.asarray(((z1 & jnp.asarray(4294967294, dtype=jnp.uint32)) << 18) ^ b1, dtype=jnp.uint32)
        b2 = ((z2 << 2) ^ z2) >> 27
        z2 = jnp.asarray(((z2 & jnp.asarray(4294967288, dtype=jnp.uint32)) << 2) ^ b2, dtype=jnp.uint32)
        b3 = ((z3 << 13) ^ z3) >> 21
        z3 = jnp.asarray(((z3 & jnp.asarray(4294967280, dtype=jnp.uint32)) << 7) ^ b3, dtype=jnp.uint32)
        b4 = ((z4 << 3) ^ z4) >> 12
        z4 = jnp.asarray(((z4 & jnp.asarray(4294967168, dtype=jnp.uint32)) << 13) ^ b4, dtype=jnp.uint32)
        new_key = (
            jnp.asarray(z1, dtype=jnp.uint32),
            jnp.asarray(z2, dtype=jnp.uint32),
            jnp.asarray(z3, dtype=jnp.uint32),
            jnp.asarray(z4, dtype=jnp.uint32)
        )
        self.key = new_key
        return new_key

    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float in [0, 1).

        Advances the generator state and converts the resulting integer
        to a floating-point number by multiplying with ``1 / (2^32 - 1)``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value in the range [0, 1).

        See Also
        --------
        randint : Generate a raw 32-bit unsigned integer.
        uniform : Generate a float in an arbitrary range.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR113RNG(seed=42)
            >>> value = rng.rand()
        """
        key = self.generate_next_key()
        # 2.3283064365386963e-10 is 1 / (2^32 - 1) approx
        return (key[0] ^ key[1] ^ key[2] ^ key[3]) * 2.3283064365386963e-10

    def randint(self) -> jax.Array:
        """Generate a uniformly distributed random 32-bit unsigned integer.

        Advances the generator state and returns the XOR of all four
        state components.

        Returns
        -------
        jax.Array
            A scalar ``uint32`` value in the range [0, 2^32 - 1].

        See Also
        --------
        rand : Generate a random float in [0, 1).
        random_integers : Generate a random integer in a specified range.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR113RNG(seed=42)
            >>> value = rng.randint()
        """
        key = self.generate_next_key()
        return jnp.asarray(key[0] ^ key[1] ^ key[2] ^ key[3], dtype=jnp.uint32)

    def randn(self, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number from the standard normal distribution N(0, 1).

        Uses the Box-Muller transform to convert two uniform random
        numbers into a normally distributed value.

        Parameters
        ----------
        epsilon : float, optional
            A small positive value used to clamp the first uniform
            sample away from zero, preventing ``log(0)``.  Defaults to
            ``1e-10``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value sampled from N(0, 1).

        See Also
        --------
        normal : Generate a value from N(mu, sigma).
        rand : Generate a uniform float in [0, 1).

        Notes
        -----
        The Box-Muller transform generates two independent standard
        normal values from two independent uniform values.  This
        implementation returns only the sine component.

        References: Box-Muller transform,
        https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR113RNG(seed=42)
            >>> value = rng.randn()
        """
        u1 = self.rand()
        u2 = self.rand()

        # Ensure u1 is not zero to avoid log(0)
        u1 = jnp.maximum(u1, epsilon)

        # Box-Muller transform
        mag = jnp.sqrt(-2.0 * jnp.log(u1))
        z2 = mag * jnp.sin(2 * jnp.pi * u2)  # Using sin component

        return z2


##############################################
# Random Number Generator: LFSR128 algorithm #
##############################################

@jax.tree_util.register_pytree_node_class
class PallasLFSR128RNG(LFSRBase):
    """Combined LFSR random number generator (LFSR128).

    Implements the LFSR128 algorithm, an extension of the LFSR family of
    Linear Feedback Shift Register random number generators.  The
    algorithm combines four independent LFSRs with expanded state to
    produce high-quality pseudorandom numbers with a very long period of
    approximately 2^128.

    Parameters
    ----------
    seed : int
        An integer used to initialize the four-component state.  The
        seed is diversified using additive constants and bitwise
        transformations to produce distinct starting values for each
        component.

    See Also
    --------
    PallasLFSR88RNG : Three-component variant with period ~2^88.
    PallasLFSR113RNG : Four-component variant with period ~2^113.
    LFSRBase : Abstract base class defining the LFSR interface.

    Notes
    -----
    The LFSR128 algorithm uses four Tausworthe generators with
    customized shift and mask parameters:

    - Component 1: shift (7, 9), mask ``0xFFFFFFFE``, left-shift 15
    - Component 2: shift (5, 23), mask ``0xFFFFFFF0``, left-shift 6
    - Component 3: shift (11, 17), mask ``0xFFFFFF80``, left-shift 8
    - Component 4: shift (13, 7), mask ``0xFFFFFFE0``, left-shift 10

    Each component advances independently, and the final output is the
    XOR of all four component values.

    The seed initialization uses different bitwise transformations
    (addition, XOR, shift, complement) with distinct constants for each
    component to ensure diverse starting points even for sequential
    seed values.

    Examples
    --------
    .. code-block:: python

        >>> rng = PallasLFSR128RNG(seed=42)
        >>> rand_float = rng.rand()        # Random float in [0, 1)
        >>> rand_int = rng.randint()        # Random 32-bit unsigned integer
        >>> norm_val = rng.normal(0, 1)     # Value from N(0, 1)
        >>> unif_val = rng.uniform(5.0, 10.0)  # Float in [5, 10)
    """
    __module__ = 'brainevent'

    def generate_key(self, seed: int) -> PallasRandomKey:
        """Initialize the random key of the LFSR128 algorithm.

        Creates a 4-element state tuple from the given seed using
        different bitwise transformations for each component to ensure
        diverse starting points.

        Parameters
        ----------
        seed : int
            An integer seed value used to initialize the generator
            state.

        Returns
        -------
        PallasRandomKey
            A tuple of four ``jnp.uint32`` scalars containing the
            initial state.

        Notes
        -----
        The four components are derived from the seed as follows:

        - ``s1 = seed + 123``
        - ``s2 = seed ^ 0xFEDC7890``
        - ``s3 = (seed << 3) + 0x1A2B3C4D``
        - ``s4 = ~(seed + 0x5F6E7D8C)``

        All arithmetic is performed in ``uint32`` to prevent overflow.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR128RNG.__new__(PallasLFSR128RNG)
            >>> key = rng.generate_key(42)
            >>> len(key)
            4
        """
        # Use different transformations for each component to ensure diversity
        # Cast to uint32 first to avoid overflow with large constants
        seed = jnp.asarray(seed, dtype=jnp.uint32)
        _c1 = jnp.asarray(123, dtype=jnp.uint32)
        _c2 = jnp.asarray(0xfedc7890, dtype=jnp.uint32)
        _c3 = jnp.asarray(0x1a2b3c4d, dtype=jnp.uint32)
        _c4 = jnp.asarray(0x5f6e7d8c, dtype=jnp.uint32)
        s1 = seed + _c1
        s2 = seed ^ _c2
        s3 = (seed << 3) + _c3
        s4 = ~(seed + _c4)
        return (s1, s2, s3, s4)

    def generate_next_key(self) -> PallasRandomKey:
        """Generate the next random key and update the internal state.

        Computes the next state of the LFSR128 generator by applying
        customized LFSR transformations to each of the four components
        of the state.

        Returns
        -------
        PallasRandomKey
            A tuple of four ``jnp.uint32`` scalars containing the new
            state after one iteration.

        Notes
        -----
        This method mutates the internal ``_key`` attribute.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR128RNG(seed=42)
            >>> new_key = rng.generate_next_key()
            >>> len(new_key)
            4
        """
        key = self.key
        z1 = key[0]
        z2 = key[1]
        z3 = key[2]
        z4 = key[3]

        # Apply different LFSR transformations to each component
        b1 = ((z1 << 7) ^ z1) >> 9
        z1 = jnp.asarray(((z1 & jnp.asarray(4294967294, dtype=jnp.uint32)) << 15) ^ b1, dtype=jnp.uint32)

        b2 = ((z2 << 5) ^ z2) >> 23
        z2 = jnp.asarray(((z2 & jnp.asarray(4294967280, dtype=jnp.uint32)) << 6) ^ b2, dtype=jnp.uint32)

        b3 = ((z3 << 11) ^ z3) >> 17
        z3 = jnp.asarray(((z3 & jnp.asarray(4294967168, dtype=jnp.uint32)) << 8) ^ b3, dtype=jnp.uint32)

        b4 = ((z4 << 13) ^ z4) >> 7
        z4 = jnp.asarray(((z4 & jnp.asarray(4294967264, dtype=jnp.uint32)) << 10) ^ b4, dtype=jnp.uint32)

        new_key = (
            jnp.asarray(z1, dtype=jnp.uint32),
            jnp.asarray(z2, dtype=jnp.uint32),
            jnp.asarray(z3, dtype=jnp.uint32),
            jnp.asarray(z4, dtype=jnp.uint32)
        )
        self.key = new_key
        return new_key

    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float in [0, 1).

        Advances the generator state and converts the resulting integer
        to a floating-point number by multiplying with ``1 / (2^32 - 1)``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value in the range [0, 1).

        See Also
        --------
        randint : Generate a raw 32-bit unsigned integer.
        uniform : Generate a float in an arbitrary range.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR128RNG(seed=42)
            >>> value = rng.rand()
        """
        key = self.generate_next_key()
        # Use all components with rotation for better mixing
        result = key[0] ^ key[1] ^ key[2] ^ key[3]
        return result * 2.3283064365386963e-10  # 1/(2^32-1)

    def randint(self) -> jax.Array:
        """Generate a uniformly distributed random 32-bit unsigned integer.

        Advances the generator state and returns a mixed result of all
        four components via XOR.

        Returns
        -------
        jax.Array
            A scalar ``uint32`` value in the range [0, 2^32 - 1].

        See Also
        --------
        rand : Generate a random float in [0, 1).
        random_integers : Generate a random integer in a specified range.

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR128RNG(seed=42)
            >>> value = rng.randint()
        """
        key = self.generate_next_key()
        return jnp.asarray(key[0] ^ key[1] ^ key[2] ^ key[3], dtype=jnp.uint32)

    def randn(self, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number from the standard normal distribution N(0, 1).

        Uses the Box-Muller transform to convert two uniform random
        numbers into a normally distributed value.

        Parameters
        ----------
        epsilon : float, optional
            A small positive value used to clamp the first uniform
            sample away from zero, preventing ``log(0)``.  Defaults to
            ``1e-10``.

        Returns
        -------
        jax.Array
            A scalar ``float32`` value sampled from N(0, 1).

        See Also
        --------
        normal : Generate a value from N(mu, sigma).
        rand : Generate a uniform float in [0, 1).

        Notes
        -----
        The Box-Muller transform generates two independent standard
        normal values from two independent uniform values.  This
        implementation returns only the sine component.

        References: Box-Muller transform,
        https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

        Examples
        --------
        .. code-block:: python

            >>> rng = PallasLFSR128RNG(seed=42)
            >>> value = rng.randn()
        """
        u1 = self.rand()
        u2 = self.rand()

        # Ensure u1 is not zero to avoid log(0)
        u1 = jnp.maximum(u1, epsilon)

        # Box-Muller transform
        mag = jnp.sqrt(-2.0 * jnp.log(u1))
        z = mag * jnp.sin(2 * jnp.pi * u2)

        return z


# ──────────────────────────────────────────────────────────────────────
#  Dispatch helpers
# ──────────────────────────────────────────────────────────────────────

_PALLAS_LFSR_CLASSES = {
    'lfsr88': PallasLFSR88RNG,
    'lfsr113': PallasLFSR113RNG,
    'lfsr128': PallasLFSR128RNG,
}


def get_pallas_lfsr_rng_class():
    """Return the Pallas RNG class for the current global LFSR algorithm."""
    return _PALLAS_LFSR_CLASSES[get_lfsr_algorithm()]


def PallasLFSRRNG(seed):
    """Factory: create a Pallas RNG instance using the globally configured algorithm.

    Parameters
    ----------
    seed : int
        Integer seed for the RNG.

    Returns
    -------
    LFSRBase
        An instance of the appropriate Pallas LFSR RNG class.
    """
    return get_pallas_lfsr_rng_class()(seed)
