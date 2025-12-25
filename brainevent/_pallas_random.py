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

__all__ = [
    'LFSR88RNG',
    'LFSR113RNG',
    'LFSR128RNG',
]


class LFSRBase(abc.ABC):
    """Abstract base class for Linear Feedback Shift Register random number generators.

    This class defines the common interface and functionality for LFSR-based
    random number generators such as LFSR88 and LFSR113. It handles the basic
    operations for managing the generator state and defines abstract methods
    that concrete implementations must provide.

    The LFSR (Linear Feedback Shift Register) algorithms are efficient
    pseudorandom number generators based on bitwise operations, ideal for
    applications requiring fast random number generation with good statistical
    properties.

    Attributes:
        _key (PallasRandomKey): The current state of the random number generator,
            represented as a jax.Array of shape (4,) and type uint32.

    Example:
        # Create a concrete LFSR implementation
        >>> rng = LFSR113RNG(seed=42)
        >>> random_float = rng.rand()
        >>> random_int = rng.randint()
    """

    def __init__(self, seed: int):
        """Initialize the random number generator with a seed.

        Args:
            seed: An integer used to initialize the random state.
        """
        self._key = self.generate_key(seed)

    @property
    def key(self) -> PallasRandomKey:
        """Get the current random state key.

        Returns:
            PallasRandomKey: The current state of the random number generator.
        """
        return self._key

    @key.setter
    def key(self, value: PallasRandomKey):
        """Set the random state key.

        Validates that the provided key is a tuple of 4 jax.Array elements, each with
        the correct type before setting it as the current state.

        Args:
            value: The new state to set for the random number generator.

        Raises:
            TypeError: If the key is not a tuple of 4 jax.Arrays.
            ValueError: If any element of the key isn't of type uint32.
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

        Args:
            seed: An integer used to initialize the random state.

        Returns:
            PallasRandomKey: The initial state of the random number generator.
        """
        pass

    @abc.abstractmethod
    def generate_next_key(self) -> PallasRandomKey:
        """Generate the next random key and update the internal state.

        This method must be implemented by concrete subclasses to advance
        the random state by one iteration according to the specific LFSR algorithm.

        Returns:
            PallasRandomKey: The new state of the random number generator.
        """
        pass

    @abc.abstractmethod
    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float between 0 and 1.

        Returns:
            jax.Array: A random float in the range [0, 1).
        """
        pass

    @abc.abstractmethod
    def randint(self) -> jax.Array:
        """Generate a uniformly distributed random 32-bit integer.

        Returns:
            jax.Array: A random integer in the range [0, 2^32-1].
        """
        pass

    @abc.abstractmethod
    def randn(self, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number with the standard normal distribution.

        Args:
            epsilon: A small positive value to avoid numerical issues in transformations.

        Returns:
            jax.Array: A random value from the standard normal distribution N(0, 1).
        """
        pass

    def uniform(self, low: float, high: float) -> jax.Array:
        """Generate a uniformly distributed random float between low and high.

        Maps a random value in [0, 1) to the specified range [low, high).

        Args:
            low: The lower bound of the range (inclusive).
            high: The upper bound of the range (exclusive).

        Returns:
            jax.Array: A floating-point value in the range [low, high).

        Example:
            >>> rng = LFSR88RNG(seed=42)
            >>> value = rng.uniform(10.0, 20.0)  # Random value between 10 and 20
        """
        r = self.rand()
        return r * (high - low) + low

    def normal(self, mu: float, sigma: float, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number from the normal distribution N(mu, sigma).

        Uses the randn method to generate a standard normal value and then
        scales and shifts it to the desired mean and standard deviation.

        Args:
            mu: The mean of the normal distribution.
            sigma: The standard deviation of the normal distribution.
            epsilon: A small positive value to avoid numerical issues.

        Returns:
            jax.Array: A random value from the normal distribution N(mu, sigma).

        Example:
            >>> rng = LFSR113RNG(seed=42)
            >>> value = rng.normal(0.0, 1.0)  # Standard normal
            >>> value = rng.normal(5.0, 2.0)  # N(5, 4)
        """
        r = self.randn(epsilon)
        return mu + sigma * r

    def random_integers(self, low: int, high: int) -> jax.Array:
        """Generate a uniformly distributed random integer between low and high (inclusive).

        Args:
            low: The lower bound of the range (inclusive).
            high: The upper bound of the range (inclusive).

        Returns:
            jax.Array: A random integer in the range [low, high].

        Example:
            >>> rng = LFSR88RNG(seed=42)
            >>> dice_roll = rng.random_integers(1, 6)  # Random integer from 1 to 6
            >>> coin_flip = rng.random_integers(0, 1)  # 0 or 1
        """
        val = self.randint()
        return val % (high + 1 - low) + low

    def tree_flatten(self):
        """
        Flatten the CSR matrix for JAX's tree utilities.

        This method is used by JAX's tree utilities to flatten the CSR matrix
        into a form suitable for transformation and reconstruction.

        Returns
        --------
        tuple
            A tuple containing two elements:
            - A tuple with the CSR matrix's data as the only element.
            - A tuple with the CSR matrix's indices, indptr, and shape.
        """
        return (self.key,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a CSR matrix from flattened data.

        This class method is used by JAX's tree utilities to reconstruct
        a CSR matrix from its flattened representation.

        Parameters
        -----------
        aux_data : tuple
            A tuple containing the CSR matrix's indices, indptr, and shape.
        children : tuple
            A tuple containing the CSR matrix's data as its only element.

        Returns
        --------
        CSR
            A new CSR matrix instance reconstructed from the flattened data.
        """
        obj = object.__new__(cls)
        key, = children
        obj.key = key
        return obj


#############################################
# Random Number Generator: LFSR88 algorithm #
#############################################

@jax.tree_util.register_pytree_node_class
class LFSR88RNG(LFSRBase):
    """Combined LFSR random number generator by L'Ecuyer (LFSR88).

    This class implements the LFSR88 algorithm, a combined Linear Feedback Shift Register
    random number generator developed by Pierre L'Ecuyer. The algorithm combines three
    different LFSRs to produce high-quality random numbers with a long period
    (approximately 2^88).

    The implementation is based on L'Ecuyer's original C code with adaptations for JAX.

    Attributes:
        key (PallasRandomKey): The current state of the random number generator,
            represented as an array of 4 unsigned 32-bit integers (though only
            the first 3 are used in calculations).

    Example:
        >>> rng = LFSR88RNG(seed=42)
        >>> rand_float = rng.rand()  # Generate a random float between 0 and 1
        >>> rand_int = rng.randint()  # Generate a random 32-bit integer
        >>> norm_val = rng.normal(0, 1)  # Generate a random value from N(0,1)
        >>> unif_val = rng.uniform(5.0, 10.0)  # Generate a random float between 5 and 10
        >>> rand_int_range = rng.random_integers(1, 6)  # Random integer from 1 to 6

    Source:
        https://github.com/cmcqueen/simplerandom/blob/main/c/lecuyer/lfsr88.c
    """
    __module__ = 'brainevent'

    def generate_key(self, seed: int) -> PallasRandomKey:
        """
        Initialize the random key of the LFSR88 algorithm.

        Creates a 4-element state vector from the given seed, ensuring that each
        element meets the minimum required value to guarantee proper algorithm function.

        Args:
            seed: An integer seed value used to initialize the generator state.

        Returns:
            PallasRandomKey: A jax.Array of shape (4,) containing the initial state
            of the generator.

        Note:
            The initial seeds MUST be larger than 1, 7, and 15 respectively.
            This method adds these values to the provided seed to ensure validity.
            The 4th element is set to 0 as it's not used in the original algorithm.
        """
        return (
            jnp.asarray(seed + 1, dtype=jnp.uint32),
            jnp.asarray(seed + 7, dtype=jnp.uint32),
            jnp.asarray(seed + 15, dtype=jnp.uint32),
            jnp.asarray(0, dtype=jnp.uint32)
        )

    def generate_next_key(self) -> PallasRandomKey:
        """
        Generate the next random key and update the internal state.

        Computes the next state of the LFSR88 generator by applying the LFSR
        transformations to each of the three components of the state vector.

        Returns:
            PallasRandomKey: A jax.Array of shape (4,) containing the new state
            of the generator after one iteration.

        Note:
            This method modifies the internal state (_key) of the generator.
            The fourth element of the key is used to store the last intermediate value 'b',
            though this isn't part of the original algorithm's state.
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
        """
        Generate a random number from the standard normal distribution N(0, 1).

        Uses the Box-Muller transform to convert uniform random numbers to normally
        distributed random numbers.

        Args:
            epsilon: A small positive value to avoid numerical issues in log(0).

        Returns:
            jax.Array: A random value from the standard normal distribution.

        Example:
            >>> rng = LFSR88RNG(seed=42)
            >>> value = rng.randn()  # Random value from standard normal distribution

        References:
            Box–Muller transform. https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
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
        """Generate a uniformly distributed random 32-bit integer.

        Advances the generator state and returns the XOR of the three state components.

        Returns:
            jax.Array: A random integer in the range [0, 2^32-1].

        Example:
            >>> rng = LFSR88RNG(seed=42)
            >>> value = rng.randint()  # Might return 2846173195, for example
        """
        key = self.generate_next_key()
        return jnp.asarray(key[0] ^ key[1] ^ key[2], dtype=jnp.uint32)

    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float between 0 and 1.

        Advances the generator state and converts the resulting integer to a
        floating-point number in [0, 1).

        Returns:
            jax.Array: A floating-point value in the range [0, 1).

        Example:
            >>> rng = LFSR88RNG(seed=42)
            >>> value = rng.rand()  # Might return 0.27183515, for example
        """
        key = self.generate_next_key()
        # 2.3283064365386963e-10 is 1 / (2^32 - 1) approx
        return (key[0] ^ key[1] ^ key[2]) * 2.3283064365386963e-10


##############################################
# Random Number Generator: LFSR113 algorithm #
##############################################

@jax.tree_util.register_pytree_node_class
class LFSR113RNG(LFSRBase):
    """Combined LFSR random number generator by L'Ecuyer (LFSR113).

    This class implements the LFSR113 algorithm, a combined Linear Feedback Shift Register
    random number generator developed by Pierre L'Ecuyer. The algorithm combines four
    different LFSRs to produce high-quality random numbers with a long period
    (approximately 2^113).

    The implementation is based on L'Ecuyer's original C code with adaptations for JAX.

    Attributes:
        key (PallasRandomKey): The current state of the random number generator,
            represented as an array of 4 unsigned 32-bit integers.

    Example:
        >>> rng = LFSR113RNG(seed=42)
        >>> rand_float = rng.rand()  # Generate a random float between 0 and 1
        >>> rand_int = rng.randint()  # Generate a random 32-bit integer
        >>> norm_val = rng.normal(0, 1)  # Generate a random value from N(0,1)
        >>> unif_val = rng.uniform(5.0, 10.0)  # Generate a random float between 5 and 10
        >>> rand_int_range = rng.random_integers(1, 6)  # Random integer from 1 to 6

    Source:
        https://github.com/cmcqueen/simplerandom/blob/main/c/lecuyer/lfsr113.c
    """
    __module__ = 'brainevent'

    def generate_key(self, seed: int) -> PallasRandomKey:
        """Initialize the random key of the LFSR113 algorithm.

        Creates a 4-element state vector from the given seed, ensuring that each
        element meets the minimum required value to guarantee proper algorithm function.

        Args:
            seed: An integer seed value used to initialize the generator state.

        Returns:
            PallasRandomKey: A jax.Array of shape (4,) containing the initial state
            of the generator.

        Note:
            The initial seeds MUST be larger than 1, 7, 15, and 127 respectively.
            This method adds these values to the provided seed to ensure validity.
        """
        return (
            jnp.asarray(seed + 1, dtype=jnp.uint32),
            jnp.asarray(seed + 7, dtype=jnp.uint32),
            jnp.asarray(seed + 15, dtype=jnp.uint32),
            jnp.asarray(seed + 127, dtype=jnp.uint32)
        )

    def generate_next_key(self) -> PallasRandomKey:
        """Generate the next random key and update the internal state.

        Computes the next state of the LFSR113 generator by applying the LFSR
        transformations to each of the four components of the state vector.

        Returns:
            PallasRandomKey: A jax.Array of shape (4,) containing the new state
            of the generator after one iteration.

        Note:
            This method modifies the internal state (_key) of the generator.
        """
        key = self.key
        z1 = key[0]
        z2 = key[1]
        z3 = key[2]
        z4 = key[3]
        b1 = ((z1 << 6) ^ z1) >> 13
        z1 = jnp.asarray(((z1 & jnp.asarray(4294967294, dtype=jnp.uint64)) << 18) ^ b1, dtype=jnp.uint32)
        b2 = ((z2 << 2) ^ z2) >> 27
        z2 = jnp.asarray(((z2 & jnp.asarray(4294967288, dtype=jnp.uint64)) << 2) ^ b2, dtype=jnp.uint32)
        b3 = ((z3 << 13) ^ z3) >> 21
        z3 = jnp.asarray(((z3 & jnp.asarray(4294967280, dtype=jnp.uint64)) << 7) ^ b3, dtype=jnp.uint32)
        b4 = ((z4 << 3) ^ z4) >> 12
        z4 = jnp.asarray(((z4 & jnp.asarray(4294967168, dtype=jnp.uint64)) << 13) ^ b4, dtype=jnp.uint32)
        new_key = (
            jnp.asarray(z1, dtype=jnp.uint32),
            jnp.asarray(z2, dtype=jnp.uint32),
            jnp.asarray(z3, dtype=jnp.uint32),
            jnp.asarray(z4, dtype=jnp.uint32)
        )
        self.key = new_key
        return new_key

    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float between 0 and 1.

        Advances the generator state and converts the resulting integer to a
        floating-point number in [0, 1).

        Returns:
            jax.Array: A floating-point value in the range [0, 1).

        Example:
            >>> rng = LFSR113RNG(seed=42)
            >>> value = rng.rand()  # Might return 0.32415783, for example
        """
        key = self.generate_next_key()
        # 2.3283064365386963e-10 is 1 / (2^32 - 1) approx
        return (key[0] ^ key[1] ^ key[2] ^ key[3]) * 2.3283064365386963e-10

    def randint(self) -> jax.Array:
        """Generate a uniformly distributed random 32-bit integer.

        Advances the generator state and returns the XOR of all four state components.

        Returns:
            jax.Array: A random integer in the range [0, 2^32-1].

        Example:
            >>> rng = LFSR113RNG(seed=42)
            >>> value = rng.randint()  # Might return 3829173452, for example
        """
        key = self.generate_next_key()
        return jnp.asarray(key[0] ^ key[1] ^ key[2] ^ key[3], dtype=jnp.uint32)

    def randn(self, epsilon: float = 1e-10) -> jax.Array:
        """Generate a random number from the standard normal distribution N(0, 1).

        Uses the Box-Muller transform to convert uniform random numbers to normally
        distributed random numbers.

        Args:
            epsilon: A small positive value to avoid numerical issues in log(0).

        Returns:
            jax.Array: A random value from the standard normal distribution.

        Example:
            >>> rng = LFSR113RNG(seed=42)
            >>> value = rng.randn()  # Random value from standard normal distribution

        References:
            Box–Muller transform. https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
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
class LFSR128RNG(LFSRBase):
    """Combined LFSR random number generator (LFSR128).

    This class implements the LFSR128 algorithm, an extension of the LFSR family of
    Linear Feedback Shift Register random number generators. The algorithm combines
    four different LFSRs with expanded state to produce high-quality random numbers
    with a very long period (approximately 2^128).

    Attributes:
        key (PallasRandomKey): The current state of the random number generator,
            represented as an array of 4 unsigned 32-bit integers.

    Example:
        >>> rng = LFSR128RNG(seed=42)
        >>> rand_float = rng.rand()  # Generate a random float between 0 and 1
        >>> rand_int = rng.randint()  # Generate a random 32-bit integer
        >>> norm_val = rng.normal(0, 1)  # Generate a random value from N(0,1)
        >>> unif_val = rng.uniform(5.0, 10.0)  # Generate a random float between 5 and 10
    """
    __module__ = 'brainevent'

    def generate_key(self, seed: int) -> PallasRandomKey:
        """Initialize the random key of the LFSR128 algorithm.

        Creates a 4-element state vector from the given seed, ensuring that each
        element meets the minimum required values to guarantee proper algorithm function.

        Args:
            seed: An integer seed value used to initialize the generator state.

        Returns:
            PallasRandomKey: A jax.Array of shape (4,) containing the initial state
            of the generator.

        Note:
            The initial seeds are derived from the provided seed to ensure diverse
            starting points for the four components of the state.
        """
        # Use different transformations for each component to ensure diversity
        s1 = seed + 123
        s2 = seed ^ 0xfedc7890
        s3 = (seed << 3) + 0x1a2b3c4d
        s4 = ~(seed + 0x5f6e7d8c)
        return (
            jnp.asarray(s1, dtype=jnp.uint32),
            jnp.asarray(s2, dtype=jnp.uint32),
            jnp.asarray(s3, dtype=jnp.uint32),
            jnp.asarray(s4, dtype=jnp.uint32)
        )

    def generate_next_key(self) -> PallasRandomKey:
        """Generate the next random key and update the internal state.

        Computes the next state of the LFSR128 generator by applying customized LFSR
        transformations to each of the four components of the state vector.

        Returns:
            PallasRandomKey: A jax.Array of shape (4,) containing the new state
            of the generator after one iteration.

        Note:
            This method modifies the internal state (_key) of the generator.
        """
        key = self.key
        z1 = key[0]
        z2 = key[1]
        z3 = key[2]
        z4 = key[3]

        # Apply different LFSR transformations to each component
        b1 = ((z1 << 7) ^ z1) >> 9
        z1 = jnp.asarray(((z1 & jnp.asarray(4294967294, dtype=jnp.uint64)) << 15) ^ b1, dtype=jnp.uint32)

        b2 = ((z2 << 5) ^ z2) >> 23
        z2 = jnp.asarray(((z2 & jnp.asarray(4294967280, dtype=jnp.uint64)) << 6) ^ b2, dtype=jnp.uint32)

        b3 = ((z3 << 11) ^ z3) >> 17
        z3 = jnp.asarray(((z3 & jnp.asarray(4294967168, dtype=jnp.uint64)) << 8) ^ b3, dtype=jnp.uint32)

        b4 = ((z4 << 13) ^ z4) >> 7
        z4 = jnp.asarray(((z4 & jnp.asarray(4294967264, dtype=jnp.uint64)) << 10) ^ b4, dtype=jnp.uint32)

        new_key = (
            jnp.asarray(z1, dtype=jnp.uint32),
            jnp.asarray(z2, dtype=jnp.uint32),
            jnp.asarray(z3, dtype=jnp.uint32),
            jnp.asarray(z4, dtype=jnp.uint32)
        )
        self.key = new_key
        return new_key

    def rand(self) -> jax.Array:
        """Generate a uniformly distributed random float between 0 and 1.

        Advances the generator state and converts the resulting integer to a
        floating-point number in [0, 1).

        Returns:
            jax.Array: A floating-point value in the range [0, 1).

        Example:
            >>> rng = LFSR128RNG(seed=42)
            >>> value = rng.rand()  # Returns a random float between 0 and 1
        """
        key = self.generate_next_key()
        # Use all components with rotation for better mixing
        result = key[0] ^ key[1] ^ key[2] ^ key[3]
        return result * 2.3283064365386963e-10  # 1/(2^32-1)

    def randint(self) -> jax.Array:
        """Generate a uniformly distributed random 32-bit integer.

        Advances the generator state and returns a mixed result of all components.

        Returns:
            jax.Array: A random integer in the range [0, 2^32-1].

        Example:
            >>> rng = LFSR128RNG(seed=42)
            >>> value = rng.randint()  # Returns a random 32-bit integer
        """
        key = self.generate_next_key()
        return jnp.asarray(key[0] ^ key[1] ^ key[2] ^ key[3], dtype=jnp.uint32)

    def randn(self, epsilon: float = 1e-10) -> jax.Array:
        """
        Generate a random number from the standard normal distribution N(0, 1).

        Uses the Box-Muller transform to convert uniform random numbers to normally
        distributed random numbers.

        Args:
            epsilon: A small positive value to avoid numerical issues in log(0).

        Returns:
            jax.Array: A random value from the standard normal distribution.

        Example:
            >>> rng = LFSR128RNG(seed=42)
            >>> value = rng.randn()  # Random value from standard normal distribution
        """
        u1 = self.rand()
        u2 = self.rand()

        # Ensure u1 is not zero to avoid log(0)
        u1 = jnp.maximum(u1, epsilon)

        # Box-Muller transform
        mag = jnp.sqrt(-2.0 * jnp.log(u1))
        z = mag * jnp.sin(2 * jnp.pi * u2)

        return z
