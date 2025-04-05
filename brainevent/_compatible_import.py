# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

__all__ = [
    'Primitive',
    'register_custom_call',
    'pallas',
    'JAXSparse',
]

import jax

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Primitive
else:
    from jax.extend.core import Primitive

if jax.__version_info__ < (0, 4, 35):
    from jax.lib import xla_client
else:
    import jax.extend as je


def register_custom_call(target_name, capsule, backend: str):
    """
    Register a custom XLA computation call target.

    This function provides JAX version compatibility, using different APIs based on
    the JAX version number to register custom calls.

    Args:
        target_name: The identifier name for the custom call.
        capsule: Python capsule object pointing to the implementation function.
        backend: str, specifies the backend type (e.g., 'cpu', 'gpu', or 'tpu').

    Notes:
        - For JAX versions before 0.4.35, uses xla_client.register_custom_call_target
        - For JAX 0.4.35 and later, uses jax.extend.ffi.register_ffi_target with api_version=0
    """
    if jax.__version_info__ < (0, 4, 35):
        xla_client.register_custom_call_target(target_name, capsule, backend)
    else:
        je.ffi.register_ffi_target(target_name, capsule, backend, api_version=0)


# import experimental module in JAX for compatibility
from jax.experimental import pallas
from jax.experimental.sparse import JAXSparse
