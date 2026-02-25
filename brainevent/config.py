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

"""User-level runtime configuration for brainevent.

Provides in-memory controls for:
- Numba parallel execution and thread count
- LFSR algorithm selection for JIT connectivity kernels
- Global default backend selection per platform
"""

from typing import Dict, Optional

__all__ = [
    'set_numba_parallel',
    'get_numba_parallel',
    'get_numba_num_threads',
    'set_lfsr_algorithm',
    'get_lfsr_algorithm',
    'set_backend',
    'get_backend',
    'clear_backends',
]


_numba_parallel: bool = False
_numba_num_threads: Optional[int] = None


def set_numba_parallel(parallel: bool = True, num_threads: Optional[int] = None):
    """Enable or disable Numba parallel execution and optionally set the thread count.

    Controls whether Numba-based kernels in brainevent use parallel execution
    (via ``numba.prange``). When ``num_threads`` is provided, it also calls
    ``numba.set_num_threads`` to configure the Numba thread pool size.

    Parameters
    ----------
    parallel : bool, optional
        If ``True``, enable Numba parallel mode. If ``False``, disable it.
        Defaults to ``True``.
    num_threads : int or None, optional
        Number of threads for Numba's thread pool. If ``None``, the Numba
        default is used (typically the number of CPU cores). Defaults to
        ``None``.

    See Also
    --------
    get_numba_parallel : Query whether Numba parallel mode is enabled.
    get_numba_num_threads : Query the configured Numba thread count.

    Notes
    -----
    Setting ``num_threads`` imports the ``numba`` package and immediately calls
    ``numba.set_num_threads``. This affects all subsequent Numba JIT-compiled
    functions, not just those in brainevent.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.set_numba_parallel(True, num_threads=4)
        >>> brainevent.get_numba_parallel()
        True
        >>> brainevent.get_numba_num_threads()
        4
    """
    global _numba_parallel, _numba_num_threads
    _numba_parallel = parallel
    _numba_num_threads = num_threads
    if num_threads is not None:
        import numba
        numba.set_num_threads(num_threads)


def get_numba_parallel() -> bool:
    """Return whether Numba parallel execution is currently enabled.

    Returns
    -------
    bool
        ``True`` if Numba parallel mode is enabled, ``False`` otherwise.

    See Also
    --------
    set_numba_parallel : Enable or disable Numba parallel mode.
    get_numba_num_threads : Query the configured Numba thread count.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.get_numba_parallel()
        False
        >>> brainevent.set_numba_parallel(True)
        >>> brainevent.get_numba_parallel()
        True
    """
    return _numba_parallel


def get_numba_num_threads() -> Optional[int]:
    """Return the configured Numba thread count.

    Returns
    -------
    int or None
        The number of threads configured for Numba's thread pool, or ``None``
        if no explicit thread count has been set (in which case Numba uses its
        own default, typically the number of CPU cores).

    See Also
    --------
    set_numba_parallel : Set the Numba parallel mode and thread count.
    get_numba_parallel : Query whether Numba parallel mode is enabled.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.get_numba_num_threads()  # None by default
        >>> brainevent.set_numba_parallel(True, num_threads=8)
        >>> brainevent.get_numba_num_threads()
        8
    """
    return _numba_num_threads


# ──────────────────────────────────────────────────────────────────────
#  LFSR algorithm configuration
# ──────────────────────────────────────────────────────────────────────

_VALID_LFSR_ALGORITHMS = ('lfsr88', 'lfsr113', 'lfsr128')
_lfsr_algorithm: str = 'lfsr88'


def set_lfsr_algorithm(algorithm: str):
    """Set the global LFSR algorithm used by JIT connectivity kernels.

    Parameters
    ----------
    algorithm : str
        One of ``'lfsr88'``, ``'lfsr113'``, or ``'lfsr128'``.

    Raises
    ------
    ValueError
        If *algorithm* is not one of the valid choices.

    See Also
    --------
    get_lfsr_algorithm : Query the current LFSR algorithm.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.config.set_lfsr_algorithm('lfsr113')
        >>> brainevent.config.get_lfsr_algorithm()
        'lfsr113'
    """
    global _lfsr_algorithm
    if algorithm not in _VALID_LFSR_ALGORITHMS:
        raise ValueError(
            f"Invalid LFSR algorithm {algorithm!r}. "
            f"Valid options are: {_VALID_LFSR_ALGORITHMS}"
        )
    _lfsr_algorithm = algorithm


def get_lfsr_algorithm() -> str:
    """Return the current global LFSR algorithm name.

    Returns
    -------
    str
        One of ``'lfsr88'``, ``'lfsr113'``, or ``'lfsr128'``.

    See Also
    --------
    set_lfsr_algorithm : Set the LFSR algorithm.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.config.get_lfsr_algorithm()
        'lfsr88'
    """
    return _lfsr_algorithm


# ──────────────────────────────────────────────────────────────────────
#  Global backend configuration
# ──────────────────────────────────────────────────────────────────────

_global_backends: Dict[str, str] = {}


def set_backend(platform: str, backend: Optional[str]):
    """Set the global default backend for a platform across all primitives.

    After this call, every primitive that has a kernel registered for
    *backend* on *platform* will use it by default, unless overridden by
    an explicit ``backend=`` keyword argument at call time.

    Parameters
    ----------
    platform : str
        The platform name (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).
    backend : str or None
        The backend name (e.g., ``'warp'``, ``'pallas'``, ``'numba'``).
        Pass ``None`` to clear the global default for this platform,
        reverting to per-primitive defaults.

    Raises
    ------
    ValueError
        If *backend* is an empty string.

    See Also
    --------
    get_backend : Query the current global backend for a platform.
    clear_backends : Clear all global backend defaults.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.set_backend('gpu', 'warp')
        >>> brainevent.get_backend('gpu')
        'warp'
        >>> brainevent.set_backend('gpu', None)  # clear
        >>> brainevent.get_backend('gpu') is None
        True
    """
    if isinstance(backend, str) and backend == '':
        raise ValueError("backend cannot be an empty string.")
    if backend is None:
        _global_backends.pop(platform, None)
    else:
        _global_backends[platform] = backend


def get_backend(platform: str) -> Optional[str]:
    """Get the global default backend for a platform.

    Parameters
    ----------
    platform : str
        The platform name (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).

    Returns
    -------
    str or None
        The globally configured backend name, or ``None`` if no global
        default has been set for this platform.

    See Also
    --------
    set_backend : Set the global backend for a platform.
    clear_backends : Clear all global backend defaults.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.get_backend('cpu')  # None by default
        >>> brainevent.set_backend('cpu', 'numba')
        >>> brainevent.get_backend('cpu')
        'numba'
    """
    return _global_backends.get(platform)


def clear_backends():
    """Clear all global backend defaults.

    After calling this function, all primitives revert to their
    per-primitive defaults (set via ``XLACustomKernel.set_default``
    or registration order).

    See Also
    --------
    set_backend : Set the global backend for a platform.
    get_backend : Query the current global backend for a platform.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.set_backend('gpu', 'warp')
        >>> brainevent.clear_backends()
        >>> brainevent.get_backend('gpu') is None
        True
    """
    _global_backends.clear()
