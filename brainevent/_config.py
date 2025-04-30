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


import threading
from contextlib import contextmanager
from typing import NamedTuple, Union


class Config(NamedTuple):
    gpu_kernel_use_warp = True


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
