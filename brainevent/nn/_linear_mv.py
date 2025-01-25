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

import brainstate as bst
from brainevent._vector_event_impl import event_liner_p_call

__all__ = [
    'Linear',
]


class Linear(bst.nn.Module):
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance.
    block_size : int, optional
        Block size for parallel computation.
    float_as_event : bool, optional
        Whether to treat float as event.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainevent'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        weight: Union[Callable, bst.typing.ArrayLike],
        float_as_event: bool = True,
        block_size: int = 64,
        name: Optional[str] = None,
        param_type: type = bst.ParamState,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.float_as_event = float_as_event
        self.block_size = block_size

        # maximum synaptic conductance
        weight = bst.init.param(weight, (self.in_size[-1], self.out_size[-1]), allow_none=False)
        self.weight = param_type(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        weight = self.weight.value
        if u.math.size(weight) == 1:
            return u.math.ones(self.out_size) * (u.math.sum(spk) * weight)

        return event_linear(spk, weight, block_size=self.block_size, float_as_event=self.float_as_event)


def event_linear(spk, weight, *, block_size, float_as_event) -> Union[jax.Array, u.Quantity]:
    """
    The event-driven linear computation.

    Parameters
    ----------
    weight : brainunit.Quantity or jax.Array
        Maximum synaptic conductance.
    spk : jax.Array
        Spike events.
    block_size : int
        Block size for parallel computation.
    float_as_event : bool
        Whether to treat float as event.

    Returns
    -------
    post_data : brainunit.Quantity or jax.Array
        Post synaptic data.
    """
    with jax.ensure_compile_time_eval():
        weight = u.math.asarray(weight)
        unit = u.get_unit(weight)
        weight = u.get_mantissa(weight)
        spk = jnp.asarray(spk)

    def mv(spk_vector):
        assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
        return event_liner_p_call(
            spk,
            weight,
            block_size=block_size,
            float_as_event=float_as_event,
        )

    assert spk.ndim >= 1, f"spk must be at least 1D. Got: {spk.ndim}"
    assert weight.ndim in [2, 0], f"weight must be 2D or 0D. Got: {weight.ndim}"

    if spk.ndim == 1:
        [post_data] = mv(spk)
    else:
        [post_data] = jax.vmap(mv)(u.math.reshape(spk, (-1, spk.shape[-1])))
        post_data = u.math.reshape(post_data, spk.shape[:-1] + post_data.shape[-1:])
    return u.maybe_decimal(u.Quantity(post_data, unit=unit))
