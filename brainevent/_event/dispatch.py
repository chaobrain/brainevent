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
# -*- coding: utf-8 -*-

from typing import Any, Optional, Tuple

from .binary import BinaryArray
from .bitpack_binary import BitPackedBinary
from .weighted import WeightedArray

__all__ = [
    'as_event_operand',
]


def as_event_operand(x: Any) -> Optional[Tuple[Any, bool]]:
    """Classify an operand of a sparse-matrix product as an event operand.

    Matrix classes call this instead of ``isinstance(x, BinaryArray)`` so that
    binary and weighted events share one dispatch path.

    Parameters
    ----------
    x : object
        Operand of ``@``.

    Returns
    -------
    result : tuple of (array, bool) or None
        ``(value, weighted)`` for :class:`BinaryArray` /
        :class:`BitPackedBinary` (``weighted=False``) and
        :class:`WeightedArray` (``weighted=True``); ``None`` for any other
        operand, which callers treat as a dense float operand.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainevent as be
        >>> from brainevent._event import as_event_operand
        >>> as_event_operand(be.WeightedArray(jnp.array([0., 2.])))[1]
        True

        >>> as_event_operand(jnp.array([0., 2.])) is None
        True
    """
    if isinstance(x, WeightedArray):
        return x.value, True
    if isinstance(x, (BinaryArray, BitPackedBinary)):
        return x.value, False
    return None
