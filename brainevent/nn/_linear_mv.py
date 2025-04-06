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

import brainstate
import brainunit as u
import jax

from brainevent._event import EventArray
from brainevent._typing import Data

__all__ = [
    'Linear',
]


class Linear(brainstate.nn.Module):
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

    __module__ = 'brainevent.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        weight: Union[Callable, brainstate.typing.ArrayLike],
        float_as_event: bool = True,
        block_size: int = 64,
        name: Optional[str] = None,
        param_type: type = brainstate.ParamState,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.float_as_event = float_as_event
        self.block_size = block_size

        # maximum synaptic conductance
        weight = brainstate.init.param(weight, (self.in_size[-1], self.out_size[-1]), allow_none=False)
        self.weight = param_type(weight)

    def update(self, spk: jax.Array) -> Data:
        weight = self.weight.value
        if u.math.size(weight) == 1:
            return u.math.ones(self.out_size) * (u.math.sum(spk) * weight)
        return EventArray(spk) @ weight
