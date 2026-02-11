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

"""Global auto-discovery registry for all XLACustomKernel instances.

This module maintains a registry that is populated automatically when
XLACustomKernel instances are created. It avoids importing brainevent
internals to prevent circular dependencies.
"""

from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from brainevent._op.main import XLACustomKernel

__all__ = [
    'register_primitive',
    'get_registry',
    'get_primitives_by_tags',
    'get_all_primitive_names',
]

_PRIMITIVE_REGISTRY: Dict[str, 'XLACustomKernel'] = {}


def register_primitive(name: str, primitive: 'XLACustomKernel'):
    """Register a primitive in the global registry.

    Called automatically by ``XLACustomKernel.__init__``.

    Parameters
    ----------
    name : str
        The unique name of the primitive.
    primitive : XLACustomKernel
        The ``XLACustomKernel`` instance to register.
    """
    _PRIMITIVE_REGISTRY[name] = primitive


def get_registry() -> Dict[str, 'XLACustomKernel']:
    """Return a copy of the full primitive registry.

    Returns
    -------
    dict of str to XLACustomKernel
        A dictionary mapping primitive names to ``XLACustomKernel`` instances.
    """
    return dict(_PRIMITIVE_REGISTRY)


def get_primitives_by_tags(tags: Set[str]) -> Dict[str, 'XLACustomKernel']:
    """Return primitives that have all the specified tags.

    Parameters
    ----------
    tags : set of str
        A set of tag strings to filter by.  A primitive must have all
        specified tags to be included.

    Returns
    -------
    dict of str to XLACustomKernel
        A dictionary mapping primitive names to matching instances.
    """
    result = {}
    for name, prim in _PRIMITIVE_REGISTRY.items():
        if hasattr(prim, '_tags') and tags.issubset(prim._tags):
            result[name] = prim
    return result


def get_all_primitive_names() -> List[str]:
    """Return a sorted list of all registered primitive names.

    Returns
    -------
    list of str
        A sorted list of primitive name strings.
    """
    return sorted(_PRIMITIVE_REGISTRY.keys())
