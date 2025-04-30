# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


import json
import pathlib
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union, NamedTuple

__all__ = [
    'load_config',
    'save_config',
    'set_environ',
    'set_numba_environ',
    'numba_environ_context',
]


class Config(NamedTuple):
    gpu_kernel_use_warp = True


# Config singleton
config = dict()

# Default configuration path
DEFAULT_CONFIG_PATH = pathlib.Path.home() / '.brainevent' / 'config.json'


def load_config(config_path: Optional[Union[str, pathlib.Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file in the user's home directory.

    Parameters
    ----------
    config_path : Optional[Union[str, pathlib.Path]], optional
        Path to the configuration file. If None, uses the default path.

    Returns
    -------
    Dict[str, Any]
        The loaded configuration dictionary.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = pathlib.Path(config_path)

    # Create default config directory if it doesn't exist
    config_dir = config_path.parent
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)

    # Load config if it exists, otherwise return empty dict
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def save_config(
    config_dict: Dict[str, Any],
    config_path: Optional[Union[str, pathlib.Path]] = None
) -> None:
    """
    Save configuration to a JSON file in the user's home directory.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary to save.
    config_path : Optional[Union[str, pathlib.Path]], optional
        Path to the configuration file. If None, uses the default path.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = pathlib.Path(config_path)

    # Create directory if it doesn't exist
    config_dir = config_path.parent
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def set_environ(**kwargs) -> None:
    """
    Set a global environment setting and save it to the configuration file.
    """
    global config

    # Load existing config
    config_dict = load_config()

    # Update the setting
    config_dict.update(**kwargs)

    # Save the updated config
    save_config(config_dict)


def initialize_config():
    """
    Initialize configuration from saved file if it exists.
    Should be called during package initialization.
    """
    global config

    # Load saved config
    config_dict = load_config()

    # Update global config object with saved values
    update_dict = {}
    for field in config._fields:
        if field in config_dict:
            update_dict[field] = config_dict[field]

    if update_dict:
        config = config._replace(**update_dict)


class NumbaEnvironment(threading.local):
    def __init__(self, *args, **kwargs):
        # default environment settings
        super().__init__(*args, **kwargs)
        self.parallel: bool = False
        self.setting: dict = dict(nogil=True, fastmath=True)


numba_environ = NumbaEnvironment()


@contextmanager
def numba_environ_context(
    parallel_if_possible: Union[int, bool] = None,
    **kwargs
):
    """
    Enable Numba parallel execution if possible.
    """
    old_parallel = numba_environ.parallel
    old_setting = numba_environ.setting.copy()

    try:
        numba_environ.setting.update(kwargs)
        if parallel_if_possible is not None:
            if isinstance(parallel_if_possible, bool):
                numba_environ.parallel = parallel_if_possible
            elif isinstance(parallel_if_possible, int):
                numba_environ.parallel = True
                assert parallel_if_possible > 0, 'The number of threads must be a positive integer.'
                import numba  # pylint: disable=import-outside-toplevel
                numba.set_num_threads(parallel_if_possible)
            else:
                raise ValueError('The argument `parallel_if_possible` must be a boolean or an integer.')
        yield numba_environ.setting.copy()
    finally:
        numba_environ.parallel = old_parallel
        numba_environ.setting = old_setting


@contextmanager
def set_numba_environ(
    parallel_if_possible: Union[int, bool] = None,
    **kwargs
):
    """
    Enable Numba parallel execution if possible.
    """
    numba_environ.setting.update(kwargs)
    if parallel_if_possible is not None:
        if isinstance(parallel_if_possible, bool):
            numba_environ.parallel = parallel_if_possible
        elif isinstance(parallel_if_possible, int):
            numba_environ.parallel = True
            assert parallel_if_possible > 0, 'The number of threads must be a positive integer.'
            import numba  # pylint: disable=import-outside-toplevel
            numba.set_num_threads(parallel_if_possible)
        else:
            raise ValueError('The argument `parallel_if_possible` must be a boolean or an integer.')
