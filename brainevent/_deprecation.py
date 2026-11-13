# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

# -*- coding: utf-8 -*-

"""Backward-compatibility shim for retired public names.

Retired names (v0.0.7 to v0.1.0, and after v0.4.2) stay *resolvable*:

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
    '0.1.0. Multiply a brainevent.FixedNumPerPre / FixedNumPerPost matrix '
    'with spikes wrapped in brainevent.BinaryArray (or BitPackedBinary / '
    'CompactBinary), which dispatches on the event type, or call '
    'brainevent.binary_fcnmv / brainevent.binary_fcnmm with the raw spikes.'
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

_HYBRID_CONFIG_MIGRATION = (
    'HybridConfig and get_hybrid_config were removed after brainevent 0.4.2. '
    'Use brainevent.CSRConfig for explicit scheduler settings (pass it as '
    '`config=` to CSR / CSC / FixedNumPer* constructors) and '
    'brainevent.load_csr_config / brainevent.save_csr_config / '
    'brainevent.init_csr_config to read, persist or tune the per-device '
    'configuration.'
)
_DELAY_DELIVER_MIGRATION = (
    'The non-layout delay delivery (csr_delay_deliver / csr_delay_deliver_p / '
    'CSR.delay_deliver) was removed after brainevent 0.4.2. Build the delay '
    'layout once with brainevent.CSRDelayLayout.build(...) or '
    'CSR.delay_layout(delays, n_slots), then deliver each step with '
    'brainevent.csr_delay_deliver_layout(ring, events, layout, t).'
)
DEPRECATED_REMOVED.update({
    'HybridConfig': _HYBRID_CONFIG_MIGRATION,
    'get_hybrid_config': _HYBRID_CONFIG_MIGRATION,
    'csr_delay_deliver': _DELAY_DELIVER_MIGRATION,
    'csr_delay_deliver_p': _DELAY_DELIVER_MIGRATION,
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
