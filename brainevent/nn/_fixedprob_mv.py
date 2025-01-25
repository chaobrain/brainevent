# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

from typing import Union, Callable, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brainstate as bst

__all__ = [
    'FixedProb',
]

FloatScalar = Union[
    np.number,  # NumPy scalar types
    float,  # Python scalar types
]

IntScalar = Union[
    np.number,  # NumPy scalar types
    int,  # Python scalar types
]


class FixedProb(bst.nn.Module):
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    prob : float
        Probability of connection, i.e., connection probability.
    weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance, i.e., synaptic weight.
    allow_multi_conn : bool, optional
        Whether multiple connections are allowed from a single pre-synaptic neuron.
        Default is True, meaning that a value of ``a`` can be selected multiple times.
    seed: int, optional
        Random seed. Default is None. If None, the default random seed will be used.
    float_as_event : bool, optional
        Whether to treat float as event. Default is True.
    block_size : int, optional
        Block size for parallel computation. Default is 64. This is only used for GPU.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainevent'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        prob: FloatScalar,
        weight: Union[Callable, bst.typing.ArrayLike],
        allow_multi_conn: bool = True,
        seed: Optional[int] = None,
        float_as_event: bool = True,
        block_size: Optional[int] = None,
        name: Optional[str] = None,
        param_type: type = bst.ParamState,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.n_conn = int(self.out_size[-1] * prob)
        self.float_as_event = float_as_event
        self.block_size = block_size

        if self.n_conn > 1:
            # indices of post connected neurons
            with jax.ensure_compile_time_eval():
                if allow_multi_conn:
                    rng = np.random.RandomState(seed)
                    self.indices = rng.randint(0, self.out_size[-1], size=(self.in_size[-1], self.n_conn))
                else:
                    rng = bst.random.RandomState(seed)

                    @bst.augment.vmap(rngs=rng)
                    def rand_indices(key):
                        rng.set_key(key)
                        return rng.choice(self.out_size[-1], size=(self.n_conn,), replace=False)

                    self.indices = rand_indices(rng.split_key(self.in_size[-1]))
                self.indices = u.math.asarray(self.indices)

        # maximum synaptic conductance
        weight = bst.init.param(weight, (self.in_size[-1], self.n_conn), allow_none=False)
        self.weight = param_type(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        if self.n_conn > 1:
            r = event_fixed_prob(
                spk,
                self.weight.value,
                self.indices,
                n_post=self.out_size[-1],
                block_size=self.block_size,
                float_as_event=self.float_as_event
            )
        else:
            weight = self.weight.value
            unit = u.get_unit(weight)
            r = jnp.zeros(spk.shape[:-1] + (self.out_size[-1],), dtype=weight.dtype)
            r = u.maybe_decimal(u.Quantity(r, unit=unit))
        return u.math.asarray(r, dtype=bst.environ.dftype())
