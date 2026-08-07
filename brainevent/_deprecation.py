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

"""Backward-compatibility shim for public names retired between v0.0.7 and v0.1.0.

Retired names stay *resolvable* so v0.0.7 code keeps importing:

* renamed names return their replacement and emit a :class:`DeprecationWarning`;
* names whose underlying functionality was removed raise an :class:`AttributeError`
  stating the concrete migration path.

These names are intentionally **not** part of :data:`brainevent.__all__` -- they are
hidden aliases surfaced only on explicit access, via the PEP 562 module-level
``__getattr__`` hook that :mod:`brainevent` installs from :func:`resolve`.

Replacement targets are stored as *names*, not objects, and looked up in the caller's
namespace at access time. Storing the objects here would require importing them from
:mod:`brainevent`, which imports this module -- a cycle.

See Also
--------
brainevent.__getattr__ : The PEP 562 hook that delegates to :func:`resolve`.
"""

from typing import Any, Dict, List, Mapping

__all__ = ['DEPRECATED_RENAMES', 'DEPRECATED_REMOVED', 'resolve', 'public_dir']

#: old public name -> name of the replacement in the :mod:`brainevent` namespace
DEPRECATED_RENAMES: Dict[str, str] = {
    'EventArray': 'BinaryArray',
    'csr_on_pre': 'update_csr_on_binary_pre',
    'csr2csc_on_post': 'update_csr_on_binary_post',
    'dense_on_pre': 'update_dense_on_binary_pre',
    'dense_on_post': 'update_dense_on_binary_post',
    'JITCHomoR': 'JITCScalarR',
    'JITCHomoC': 'JITCScalarC',
    'FixedPostNumConn': 'FixedNumPerPre',
    'FixedPreNumConn': 'FixedNumPerPost',
}

_COO_MIGRATION = (
    'The COO sparse format was removed in brainevent 0.1.0. Use CSR / CSC '
    'instead (brainevent.CSR / brainevent.CSC); convert indices with '
    'brainevent.coo2csr or the *_index helpers (csr_to_coo_index, '
    'coo_to_csc_index, csr_to_csc_index, csc_to_csr_index).'
)
_FCN_PACK_MIGRATION = (
    'The explicit bitpack_/compact_ FCN kernels were removed in brainevent '
    '0.1.0; they were unified into fcnmv / fcnmm, which dispatch on the input '
    'event type. Wrap spikes with brainevent.BitPackedBinary or '
    'brainevent.CompactBinary and call brainevent.fcnmv / brainevent.fcnmm.'
)
_LAYOUT_MIGRATION = (
    'The fixed-number-connection layout abstraction was removed. Use '
    'FixedNumPerPost / FixedNumPerPre directly (favorable/unfavorable dispatch '
    'is now internal).'
)

#: old public name -> migration message (functionality removed, no drop-in)
DEPRECATED_REMOVED: Dict[str, str] = {}
DEPRECATED_REMOVED.update({
    name: _COO_MIGRATION for name in (
        'COO',
        'binary_coomv', 'binary_coomv_p',
        'binary_coomm', 'binary_coomm_p',
        'coomv', 'coomv_p',
        'coomm', 'coomm_p',
        'update_coo_on_binary_pre', 'update_coo_on_binary_post',
        'update_coo_on_binary_pre_p', 'update_coo_on_binary_post_p',
    )
})
DEPRECATED_REMOVED.update({
    name: _FCN_PACK_MIGRATION for name in (
        'bitpack_binary_fcnmv', 'bitpack_binary_fcnmv_p',
        'bitpack_binary_fcnmm', 'bitpack_binary_fcnmm_p',
        'compact_binary_fcnmv', 'compact_binary_fcnmv_p',
        'compact_binary_fcnmm', 'compact_binary_fcnmm_p',
    )
})
DEPRECATED_REMOVED.update({
    'EllLayout': _LAYOUT_MIGRATION,
    'CscLayout': _LAYOUT_MIGRATION,
})


def resolve(name: str, namespace: Mapping[str, Any], module: str = 'brainevent') -> Any:
    """Resolve a possibly-retired public name against ``namespace``.

    Parameters
    ----------
    name : str
        The attribute name requested from the module.
    namespace : mapping of str to object
        The module namespace to resolve replacements in, normally ``globals()``
        of :mod:`brainevent`.
    module : str, default ``'brainevent'``
        Module name used in warning and error messages.

    Returns
    -------
    object
        The replacement object, when ``name`` is a deprecated rename.

    Raises
    ------
    AttributeError
        If ``name`` was removed (message carries the migration path), or if it is
        simply unknown (plain message, matching normal attribute lookup).

    Notes
    -----
    Renames emit a :class:`DeprecationWarning` with ``stacklevel=3``: this function
    is called by the module ``__getattr__``, which is itself called by the attribute
    access, so the warning points at user code rather than at the shim.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent import _deprecation
        >>> ns = {'BinaryArray': int}
        >>> import warnings
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter('ignore')
        ...     _deprecation.resolve('EventArray', ns)
        <class 'int'>
    """
    import warnings

    if name in DEPRECATED_RENAMES:
        new_name = DEPRECATED_RENAMES[name]
        warnings.warn(
            f'{module}.{name} is deprecated and will be removed in a future '
            f'release; use {module}.{new_name} instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        return namespace[new_name]
    if name in DEPRECATED_REMOVED:
        raise AttributeError(
            f'{module}.{name} was removed in 0.1.0. {DEPRECATED_REMOVED[name]}'
        )
    raise AttributeError(f'module {module!r} has no attribute {name!r}')


def public_dir(namespace: Mapping[str, Any]) -> List[str]:
    """Return the sorted ``dir()`` listing including hidden deprecated aliases.

    Parameters
    ----------
    namespace : mapping of str to object
        The module namespace to list, normally ``globals()`` of :mod:`brainevent`.

    Returns
    -------
    list of str
        Sorted union of ``namespace`` keys and every deprecated alias, so that
        tab-completion and :func:`dir` still surface retired v0.0.7 names.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent import _deprecation
        >>> 'EventArray' in _deprecation.public_dir({'BinaryArray': None})
        True
    """
    return sorted(set(namespace) | set(DEPRECATED_RENAMES) | set(DEPRECATED_REMOVED))
