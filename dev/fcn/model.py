# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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


from typing import Union, Callable, Optional

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import brainevent


class FixedNumConn(brainstate.nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        conn_num: Union[int, float],
        conn_weight: Union[Callable, brainstate.typing.ArrayLike],
        efferent_target: str = 'post',  # 'pre' or 'post'
        allow_multi_conn: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        data_type: str = 'binary'
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.efferent_target = efferent_target
        self.data_type = data_type
        if data_type not in ('binary', 'sparse_float', 'float'):
            raise ValueError('data_type must be either "binary" or "sparse_float" or "float".')
        if efferent_target not in ('pre', 'post'):
            raise ValueError('The target of the connection must be either "pre" or "post".')
        if isinstance(conn_num, float):
            assert 0. <= conn_num <= 1., 'Connection probability must be in [0, 1].'
            conn_num = (
                int(self.out_size[-1] * conn_num)
                if efferent_target == 'post' else
                int(self.in_size[-1] * conn_num)
            )
        assert isinstance(conn_num, int), 'Connection number must be an integer.'
        self.conn_num = conn_num
        self.seed = seed
        self.allow_multi_conn = allow_multi_conn

        # connections
        if self.efferent_target == 'post':
            n_post = self.out_size[-1]
            n_pre = self.in_size[-1]
        else:
            n_post = self.in_size[-1]
            n_pre = self.out_size[-1]

        with jax.ensure_compile_time_eval():
            rng = np.random if seed is None else np.random.RandomState(seed)
            indices = rng.randint(0, n_post, size=(n_pre, self.conn_num))
            conn_weight = braintools.init.param(conn_weight, (n_pre, self.conn_num))
            self.weight = u.math.asarray(conn_weight, dtype=brainstate.environ.dftype())
            self.indices = u.math.asarray(indices, dtype=np.int32)
            self.shape = (n_pre, n_post)
            csr = (
                brainevent.FixedPostNumConn((conn_weight, indices), shape=(n_pre, n_post))
                if self.efferent_target == 'post' else
                brainevent.FixedPreNumConn((conn_weight, indices), shape=(n_pre, n_post))
            )
            self.conn = csr

    def update(self, x) -> Union[jax.Array, u.Quantity]:
        assert x.ndim in [1, 2], 'Input must be 1D or 2D.'
        if self.data_type == 'binary':
            fn = brainevent.binary_fcnmv if x.ndim == 1 else brainevent.binary_fcnmm
        elif self.data_type == 'sparse_float':
            fn = brainevent.sparse_float_fcnmv if x.ndim == 1 else brainevent.sparse_float_fcnmm
        else:
            fn = brainevent.float_fcnmv if x.ndim == 1 else brainevent.float_fcnmm
        transpose = (self.efferent_target == 'post')
        return fn(self.weight, self.indices, x, shape=self.shape, transpose=transpose)
