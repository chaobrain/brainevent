# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import jax_tvm_ffi
import jax
import numpy as np


def process_tensor(args, rets, stream, a):
    # Convert to NumPy arrays for processing
    print(args)
    print(rets)
    print(stream)
    print(a)

jax_tvm_ffi.register_ffi_target(
    "process.tensor",
    process_tensor,
["args", "rets", "ctx.stream"],
    platform="gpu",
    # Enable owned tensor access so from_dlpack can be called
    pass_owned_tensor=True
)

output_shape = jax.ShapeDtypeStruct([10], jax.numpy.float32)
x = np.random.randn(10)
y = np.zeros(10)
result = jax.ffi.ffi_call("process.tensor", output_shape)(x, y, eps=1e-5)


