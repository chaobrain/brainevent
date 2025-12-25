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


import threading
from contextlib import contextmanager
from typing import Union

__all__ = [
    'config',
]


class Config(threading.local):
    """
    A configuration class that stores settings for the brainevent package.

    This class provides a container for configuration settings that affect the behavior
    of the library, particularly regarding GPU kernel backends and Numba optimization
    settings. It implements various methods to get and set configuration values,
    and provides context managers for temporary configuration changes.

    Notes
    -----
    This class is instantiated as a singleton named 'config' to provide
    global access to configuration settings throughout the library.
    """

    def __init__(self):
        self._gpu_kernel_backend = 'default'
        self._numba_setting: dict = dict(nogil=True, fastmath=True, parallel=True)

    @property
    def gpu_kernel_backend(self) -> str:
        """
        The backend to use for GPU kernel operations.

        This property provides read access to the currently configured GPU kernel backend.
        The backend determines which GPU acceleration library will be used for computations.

        Returns
        -------
        str
            The current backend. Possible values are 'default', 'warp', or 'pallas'.

        See Also
        --------
        set_gpu_backend : Method to set the GPU backend
        """
        return self._gpu_kernel_backend

    @gpu_kernel_backend.setter
    def gpu_kernel_backend(self, value: str) -> None:
        """
        Set the backend to use for GPU kernel operations.

        This property setter provides a convenient interface to change the GPU backend.
        It delegates to the `set_gpu_backend` method which performs validation.

        Parameters
        ----------
        value : str
            The backend to set. Must be one of 'default', 'warp', or 'pallas'.

        Raises
        ------
        ValueError
            If the provided backend is not one of the supported values.

        See Also
        --------
        set_gpu_backend : Method that implements the backend setting logic
        """
        self.set_gpu_backend(backend=value)

    def set_gpu_backend(self, backend: str) -> None:
        """
        Set the GPU kernel backend.

        This method configures which GPU acceleration library will be used for
        computational operations. It validates that the provided backend is supported.

        Parameters
        ----------
        backend : str
            The backend to set. Must be one of:
            - 'default': Use the default backend
            - 'warp': Use the NVIDIA Warp framework
            - 'pallas': Use JAX Pallas for GPU operations

        Raises
        ------
        ValueError
            If the provided backend is not one of the supported values.

        Notes
        -----
        This method contains validation logic and is called by the `gpu_kernel_backend` setter.
        There appears to be a documentation inconsistency as the docstring mentions 'jax' and 'torch'
        but the implementation checks for 'warp' and 'pallas'.
        """
        if backend not in ['default', 'warp', 'pallas']:
            raise ValueError(
                f'Invalid backend: {backend}, must be one of {["default", "warp", "pallas"]}'
            )
        self._gpu_kernel_backend = backend

    def get_gpu_backend(self) -> str:
        """
        Get the current GPU kernel backend.

        This method provides the currently configured GPU backend and is
        a functional alternative to accessing the `gpu_kernel_backend` property.

        Returns
        -------
        str
            The current backend. Will be one of 'default', 'warp', or 'pallas'.

        See Also
        --------
        gpu_kernel_backend : Property that provides the same functionality
        set_gpu_backend : Method to set the GPU backend

        Examples
        --------
        >>> from brainevent import config
        >>> config.get_gpu_backend()
        'default'
        >>> config.set_gpu_backend('warp')
        >>> config.get_gpu_backend()
        'warp'
        """
        return self.gpu_kernel_backend

    def get_numba_setting(self) -> dict:
        """
        Get the current Numba compiler settings.

        This method returns a copy of the internal Numba compiler configuration
        dictionary, which controls optimization settings like parallel execution,
        fast math mode, and no GIL mode.

        Returns
        -------
        dict
            A copy of the current Numba settings dictionary containing keys like:
            - 'nogil': Boolean indicating if Numba should release the GIL when possible
            - 'fastmath': Boolean controlling whether to enable fast math optimizations
            - 'parallel': Boolean determining if Numba should parallelize operations

        See Also
        --------
        update_numba_setting : Method to update specific Numba settings
        replace_numba_setting : Method to replace all Numba settings
        numba_environ_context : Context manager for temporary Numba setting changes

        Examples
        --------
        >>> from brainevent import config
        >>> config.get_numba_setting()
        {'nogil': True, 'fastmath': True, 'parallel': False}
        """
        return self._numba_setting.copy()

    def update_numba_setting(self, **setting) -> None:
        """
        Update the current Numba compiler settings with new values.

        This method allows updating specific Numba settings while preserving
        other existing settings. It accepts keyword arguments corresponding to
        Numba optimization parameters.

        Parameters
        ----------
        **setting : any
            Keyword arguments specifying Numba settings to update. Valid keys include:
            - 'nogil': Boolean indicating if Numba should release the GIL when possible
            - 'fastmath': Boolean controlling whether to enable fast math optimizations
            - 'parallel': Boolean determining if Numba should parallelize operations

        See Also
        --------
        get_numba_setting : Method to retrieve current Numba settings
        replace_numba_setting : Method to replace all Numba settings

        Examples
        --------
        >>> from brainevent import config
        >>> config.update_numba_setting(parallel=True)
        >>> config.get_numba_setting()
        {'nogil': True, 'fastmath': True, 'parallel': True}
        """
        self._numba_setting.update(**setting)

    def replace_numba_setting(self, setting: dict) -> None:
        """
        Replace the entire Numba compiler settings dictionary.

        This method completely replaces the current Numba settings with a new
        dictionary. It creates a copy of the provided dictionary to prevent
        external modifications from affecting the internal configuration.

        Parameters
        ----------
        setting : dict
            A dictionary containing Numba compiler settings. Expected keys include:
            - 'nogil': Boolean indicating if Numba should release the GIL when possible
            - 'fastmath': Boolean controlling whether to enable fast math optimizations
            - 'parallel': Boolean determining if Numba should parallelize operations

        See Also
        --------
        get_numba_setting : Method to retrieve current Numba settings
        update_numba_setting : Method to update specific Numba settings

        Examples
        --------
        >>> from brainevent import config
        >>> config.replace_numba_setting({'nogil': False, 'fastmath': False, 'parallel': True})
        >>> config.get_numba_setting()
        {'nogil': False, 'fastmath': False, 'parallel': True}
        """
        self._numba_setting = setting.copy()

    @contextmanager
    def numba_environ_context(
        self,
        parallel_if_possible: Union[int, bool] = None,
        **kwargs
    ):
        """
        Create a context manager for temporary Numba environment configuration.

        This context manager allows for temporarily changing Numba compiler settings
        within its scope. When the context is exited, the original settings are
        restored automatically, making it useful for code sections that need
        special optimization settings without affecting the global configuration.

        Parameters
        ----------
        parallel_if_possible : bool or int, optional
            Controls parallel execution configuration:
            - If bool: Directly sets the 'parallel' setting for Numba
            - If int: Raises ValueError as thread count is not supported in this method
            - If None: Leaves the current parallel setting unchanged
        **kwargs : dict
            Additional Numba settings to temporarily apply. Valid keys include:
            - 'nogil': Boolean indicating if Numba should release the GIL
            - 'fastmath': Boolean controlling fast math optimizations
            - 'parallel': Boolean determining if Numba should parallelize operations

        Yields
        ------
        dict
            The active Numba settings dictionary during the context

        Raises
        ------
        ValueError
            If parallel_if_possible is an integer (use numba_environ_set instead)
            or if it's neither a boolean nor an integer

        See Also
        --------
        get_numba_setting : Method to retrieve current Numba settings
        update_numba_setting : Method to update specific Numba settings
        numba_environ_set : Method for permanent environment changes with thread count support

        Examples
        --------
        >>> from brainevent import config
        >>> # Temporarily enable parallel execution
        >>> with config.numba_environ_context(parallel_if_possible=True):
        ...     # Code executed with parallel Numba optimization
        ...     result = compute_intensive_function(data)
        >>> # Original settings are restored after the context ends

        Notes
        -----
        This method is safer than directly modifying settings because it ensures
        settings are properly restored even if exceptions occur within the context.
        """
        old_setting = self.get_numba_setting()

        try:
            self.update_numba_setting(**kwargs)
            if parallel_if_possible is not None:
                if isinstance(parallel_if_possible, bool):
                    self.update_numba_setting(parallel=parallel_if_possible)
                elif isinstance(parallel_if_possible, int):
                    raise ValueError(
                        'The argument `parallel_if_possible` must be a boolean when using '
                        'brainevent.config.numba_environ_context. '
                        'For setting the number of threads, use `brainevent.config.set_numba_environ` instead.'
                    )
                else:
                    raise ValueError('The argument `parallel_if_possible` must be a boolean or an integer.')
            yield self.get_numba_setting()
        finally:
            self.replace_numba_setting(old_setting)

    @contextmanager
    def numba_environ_set(
        self,
        parallel_if_possible: Union[int, bool] = None,
        **kwargs
    ):
        """
        Configure Numba environment settings and optionally set thread count.

        This context manager allows for changing Numba compiler settings and thread
        configuration. Unlike `numba_environ_context`, this method does not restore
        the original settings when the context exits - the changes remain in effect.
        It also supports direct configuration of the number of Numba threads.

        Parameters
        ----------
        parallel_if_possible : bool or int, optional
            Controls parallel execution configuration:
            - If bool: Directly sets the 'parallel' setting for Numba
            - If int: Enables parallel execution and sets the number of threads to this value
            - If None: Leaves the current parallel setting unchanged
        **kwargs : dict
            Additional Numba settings to apply. Valid keys include:
            - 'nogil': Boolean indicating if Numba should release the GIL
            - 'fastmath': Boolean controlling fast math optimizations
            - 'parallel': Boolean determining if Numba should parallelize operations

        Yields
        ------
        None
            This context manager doesn't yield any value but provides a context
            where the specified Numba settings are in effect

        Raises
        ------
        ValueError
            If parallel_if_possible is neither a boolean nor an integer
        AssertionError
            If parallel_if_possible is an integer less than or equal to zero

        See Also
        --------
        numba_environ_context : Context manager for temporary Numba setting changes
        update_numba_setting : Method to update specific Numba settings

        Examples
        --------
        >>> from brainevent import config
        >>> # Set parallel execution with 4 threads
        >>> with config.numba_environ_set(parallel_if_possible=4):
        ...     # Code executed with parallel Numba optimization using 4 threads
        ...     result = compute_intensive_function(data)
        >>> # The settings remain after the context ends

        >>> # Disable parallel execution
        >>> with config.numba_environ_set(parallel_if_possible=False, fastmath=False):
        ...     # Code executed with sequential execution and without fast math
        ...     result = compute_intensive_function(data)

        Notes
        -----
        Unlike `numba_environ_context`, this method does not restore previous settings
        when the context ends. Changes made with this method persist until explicitly changed.
        """
        self.update_numba_setting(**kwargs)
        if parallel_if_possible is not None:
            if isinstance(parallel_if_possible, bool):
                self.update_numba_setting(parallel=parallel_if_possible)
            elif isinstance(parallel_if_possible, int):
                self.update_numba_setting(parallel=True)
                assert parallel_if_possible > 0, 'The number of threads must be a positive integer.'
                import numba  # pylint: disable=import-outside-toplevel
                numba.set_num_threads(parallel_if_possible)
            else:
                raise ValueError('The argument `parallel_if_possible` must be a boolean or an integer.')


# Config singleton
config = Config()
