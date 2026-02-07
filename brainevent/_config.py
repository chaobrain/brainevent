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
]

_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = {1}
_cache: Optional[Dict[str, Any]] = None


def get_config_path() -> str:
    """Return the platform-appropriate path for the config file.

    Returns:
        Absolute path to defaults.json.
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
    """Read and validate the config file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed config dict, or empty default if file missing/corrupted.
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
    """Atomically write the config file using temp-file + os.replace.

    Args:
        path: Destination path.
        data: Config dict to write.
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
    """Clear the in-memory config cache, forcing a re-read on next access."""
    global _cache
    _cache = None


def load_user_defaults() -> Dict[str, Dict[str, str]]:
    """Load user defaults from the config file (cached).

    Returns:
        A dict mapping primitive names to dicts of {platform: backend}.
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
    """Save user defaults to the config file.

    Merges the provided defaults with existing ones and writes atomically.

    Args:
        defaults: Dict mapping primitive names to {platform: backend}.
        metadata: Optional benchmark metadata to store alongside defaults.
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

    Args:
        primitive_name: Name of the primitive.
        platform_name: Platform name ('cpu', 'gpu', 'tpu').

    Returns:
        Backend name string, or None if not set.
    """
    defaults = load_user_defaults()
    return defaults.get(primitive_name, {}).get(platform_name)


def set_user_default(primitive_name: str, platform_name: str, backend: str):
    """Set and persist the user's preferred backend for a primitive/platform.

    Args:
        primitive_name: Name of the primitive.
        platform_name: Platform name ('cpu', 'gpu', 'tpu').
        backend: Backend name to set as default.
    """
    save_user_defaults({primitive_name: {platform_name: backend}})


def clear_user_defaults():
    """Remove all user defaults and delete the config file."""
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
