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

"""User-level configuration persistence for brainevent.

Stores per-primitive default backend selections in a JSON file at a
platform-appropriate location. Supports atomic writes, schema versioning,
and cached loading.

Config locations:
    - Linux:   ~/.config/brainevent/defaults.json
    - macOS:   ~/Library/Application Support/brainevent/defaults.json
    - Windows: %APPDATA%/brainevent/defaults.json
"""

import json
import os
import platform
import tempfile
import warnings
from typing import Any, Dict, Optional

__all__ = [
    'load_user_defaults',
    'save_user_defaults',
    'get_user_default',
    'set_user_default',
    'clear_user_defaults',
    'get_config_path',
    'invalidate_cache',
    'set_numba_parallel',
    'get_numba_parallel',
    'get_numba_num_threads',
]

_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = {1}
_cache: Optional[Dict[str, Any]] = None


def get_config_path() -> str:
    """Return the platform-appropriate path for the brainevent config file.

    Determines the configuration directory based on the current operating system
    and returns the full path to the ``defaults.json`` file used for persisting
    user-level backend preferences.

    Returns
    -------
    str
        Absolute path to the ``defaults.json`` configuration file.

    Notes
    -----
    The platform-specific base directories are:

    - **Windows**: ``%APPDATA%/brainevent/defaults.json`` (falls back to
      ``~/brainevent/defaults.json`` if ``APPDATA`` is not set).
    - **macOS**: ``~/Library/Application Support/brainevent/defaults.json``.
    - **Linux / other**: ``$XDG_CONFIG_HOME/brainevent/defaults.json`` (falls
      back to ``~/.config/brainevent/defaults.json`` if ``XDG_CONFIG_HOME`` is
      not set).

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> path = brainevent.get_config_path()  # doctest: +SKIP
        >>> print(path)  # e.g. '/home/user/.config/brainevent/defaults.json'
    """
    system = platform.system()
    if system == 'Windows':
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
    elif system == 'Darwin':
        base = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support')
    else:
        base = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config'))
    return os.path.join(base, 'brainevent', 'defaults.json')


def _read_config_file(path: str) -> Dict[str, Any]:
    """Read and validate the JSON configuration file.

    Parameters
    ----------
    path : str
        Absolute path to the configuration file.

    Returns
    -------
    dict of str to any
        The parsed configuration dictionary.  Returns an empty default
        structure (with ``schema_version``, ``defaults``, and
        ``benchmark_metadata`` keys) if the file is missing, corrupted,
        or has an unsupported schema version.
    """
    if not os.path.isfile(path):
        return {'schema_version': _SCHEMA_VERSION, 'defaults': {}, 'benchmark_metadata': {}}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(
            f"brainevent: Corrupted config file at {path}: {e}. Using built-in defaults.",
            stacklevel=3,
        )
        return {'schema_version': _SCHEMA_VERSION, 'defaults': {}, 'benchmark_metadata': {}}

    schema_ver = data.get('schema_version', 0)
    if schema_ver not in _SUPPORTED_SCHEMA_VERSIONS:
        warnings.warn(
            f"brainevent: Config file schema version {schema_ver} is not supported "
            f"(supported: {_SUPPORTED_SCHEMA_VERSIONS}). Ignoring user defaults.",
            stacklevel=3,
        )
        return {'schema_version': _SCHEMA_VERSION, 'defaults': {}, 'benchmark_metadata': {}}

    return data


def _write_config_file(path: str, data: Dict[str, Any]):
    """Atomically write the configuration dictionary to a JSON file.

    Uses a temporary file and ``os.replace`` to ensure the write is
    atomic -- the config file is never left in a partially written state.

    Parameters
    ----------
    path : str
        Destination path for the configuration file.
    data : dict of str to any
        The configuration dictionary to serialize as JSON.
    """
    config_dir = os.path.dirname(path)
    try:
        os.makedirs(config_dir, exist_ok=True)
    except OSError as e:
        warnings.warn(
            f"brainevent: Cannot create config directory {config_dir}: {e}. "
            f"Default persistence skipped.",
            stacklevel=3,
        )
        return

    try:
        fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write('\n')
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up temp file on any error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        warnings.warn(
            f"brainevent: Cannot write config file {path}: {e}. "
            f"Default persistence skipped.",
            stacklevel=3,
        )


def invalidate_cache():
    """Clear the in-memory configuration cache, forcing a re-read on next access.

    The configuration system caches the contents of the JSON config file after
    the first read to avoid repeated disk I/O. Call this function to discard
    that cached state so that the next call to :func:`load_user_defaults` or
    :func:`get_user_default` will re-read the file from disk.

    This is primarily useful after the config file has been modified externally
    (e.g., by another process or a manual edit).

    See Also
    --------
    load_user_defaults : Load (and cache) user defaults from the config file.
    clear_user_defaults : Remove all user defaults and delete the config file.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.invalidate_cache()  # Force next load to re-read from disk
    """
    global _cache
    _cache = None


def load_user_defaults() -> Dict[str, Dict[str, str]]:
    """Load user-configured backend defaults from the config file.

    Reads the JSON configuration file and returns the ``defaults`` section,
    which maps primitive names to per-platform backend selections. Results
    are cached in memory; subsequent calls return the cached copy unless
    :func:`invalidate_cache` has been called.

    Returns
    -------
    dict of str to dict of str to str
        A dictionary mapping primitive names (e.g., ``"csrmv"``) to
        dictionaries of ``{platform_name: backend_name}`` (e.g.,
        ``{"gpu": "pallas", "cpu": "numba"}``). Returns an empty dict
        if no defaults have been configured.

    See Also
    --------
    save_user_defaults : Save user defaults to the config file.
    get_user_default : Retrieve the default backend for a single primitive/platform pair.
    invalidate_cache : Clear the in-memory cache.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> defaults = brainevent.load_user_defaults()
        >>> print(defaults)  # doctest: +SKIP
        {'csrmv': {'gpu': 'pallas', 'cpu': 'numba'}}
    """
    global _cache
    if _cache is not None:
        return _cache.get('defaults', {})

    _cache = _read_config_file(get_config_path())
    return _cache.get('defaults', {})


def save_user_defaults(
    defaults: Dict[str, Dict[str, str]],
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save user-configured backend defaults to the config file.

    Merges the provided defaults with any existing ones on disk and writes
    the result atomically. The in-memory cache is updated to reflect the
    new state.

    Parameters
    ----------
    defaults : dict of str to dict of str to str
        A dictionary mapping primitive names to dictionaries of
        ``{platform_name: backend_name}``. These are merged into the
        existing defaults (new entries are added, existing entries for
        the same primitive/platform pair are overwritten).
    metadata : dict of str to any, optional
        Optional benchmark metadata to store alongside the defaults.
        When provided, this replaces the entire ``benchmark_metadata``
        section in the config file.

    See Also
    --------
    load_user_defaults : Load user defaults from the config file.
    set_user_default : Convenience function to set a single default.
    clear_user_defaults : Remove all user defaults and delete the config file.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.save_user_defaults(
        ...     {"csrmv": {"gpu": "pallas", "cpu": "numba"}}
        ... )  # doctest: +SKIP
    """
    global _cache
    path = get_config_path()
    existing = _read_config_file(path)

    # Merge defaults
    existing_defaults = existing.get('defaults', {})
    for prim_name, platform_map in defaults.items():
        if prim_name not in existing_defaults:
            existing_defaults[prim_name] = {}
        existing_defaults[prim_name].update(platform_map)
    existing['defaults'] = existing_defaults

    # Update metadata if provided
    if metadata is not None:
        existing['benchmark_metadata'] = metadata

    existing['schema_version'] = _SCHEMA_VERSION
    _write_config_file(path, existing)

    # Update cache
    _cache = existing


def get_user_default(primitive_name: str, platform_name: str) -> Optional[str]:
    """Get the user's preferred backend for a specific primitive and platform.

    Looks up the cached (or freshly loaded) user defaults and returns the
    backend name associated with the given primitive/platform combination.

    Parameters
    ----------
    primitive_name : str
        Name of the primitive (e.g., ``"csrmv"``, ``"coomv"``).
    platform_name : str
        JAX platform name (``"cpu"``, ``"gpu"``, or ``"tpu"``).

    Returns
    -------
    str or None
        The backend name string (e.g., ``"pallas"``, ``"numba"``, ``"warp"``),
        or ``None`` if no default has been set for this primitive/platform pair.

    See Also
    --------
    set_user_default : Set the default backend for a primitive/platform pair.
    load_user_defaults : Load all user defaults.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> backend = brainevent.get_user_default("csrmv", "gpu")  # doctest: +SKIP
        >>> print(backend)  # e.g. 'pallas' or None
    """
    defaults = load_user_defaults()
    return defaults.get(primitive_name, {}).get(platform_name)


def set_user_default(primitive_name: str, platform_name: str, backend: str):
    """Set and persist the user's preferred backend for a primitive/platform pair.

    This is a convenience wrapper around :func:`save_user_defaults` that sets
    a single primitive/platform/backend entry.

    Parameters
    ----------
    primitive_name : str
        Name of the primitive (e.g., ``"csrmv"``, ``"coomv"``).
    platform_name : str
        JAX platform name (``"cpu"``, ``"gpu"``, or ``"tpu"``).
    backend : str
        Backend name to set as the default (e.g., ``"pallas"``, ``"numba"``,
        ``"warp"``).

    See Also
    --------
    get_user_default : Retrieve the default backend for a primitive/platform pair.
    save_user_defaults : Save multiple defaults at once.
    clear_user_defaults : Remove all user defaults.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.set_user_default("csrmv", "gpu", "pallas")  # doctest: +SKIP
    """
    save_user_defaults({primitive_name: {platform_name: backend}})


def clear_user_defaults():
    """Remove all user defaults and delete the config file.

    Deletes the JSON configuration file from disk and clears the in-memory
    cache. After calling this function, all primitives will revert to their
    built-in default backends until new preferences are set.

    Raises
    ------
    Warning
        A ``UserWarning`` is issued if the config file cannot be deleted
        (e.g., due to permission errors). The in-memory cache is still
        cleared in that case.

    See Also
    --------
    set_user_default : Set a single backend default.
    save_user_defaults : Save multiple defaults at once.
    invalidate_cache : Clear only the in-memory cache without deleting the file.

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.clear_user_defaults()  # doctest: +SKIP
    """
    global _cache
    path = get_config_path()
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError as e:
        warnings.warn(
            f"brainevent: Cannot delete config file {path}: {e}.",
            stacklevel=3,
        )
    _cache = None


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
