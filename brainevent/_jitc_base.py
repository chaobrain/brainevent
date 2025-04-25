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

# -*- coding: utf-8 -*-


import brainunit as u

__all__ = [
    'JITCMatrix',
]


class JITCMatrix(u.sparse.SparseMatrix):
    """
    Just-in-time Connectivity (JITC) matrix.

    A base class for just-in-time connectivity matrices that inherits from
    the SparseMatrix class in the ``brainunit`` library. This class serves as
    an abstraction for sparse matrices that are generated or computed on demand
    rather than stored in full.

    JITC matrices are particularly useful in neural network simulations where
    connectivity patterns might be large but follow specific patterns that
    can be efficiently computed rather than explicitly stored in memory.

    Attributes:
        Inherits all attributes from ``brainunit.sparse.SparseMatrix``

    Note:
        This is a base class and should be subclassed for specific
        implementations of JITC matrices.
    """
    pass
