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

"""Central registry for all JAX primitives used in brainevent."""

from typing import List, Dict

from brainevent._coo.binary import (
    event_coomv_p,
    event_coomm_p,
)
from brainevent._coo.float import (
    coomv_p,
    coomm_p,
)
from brainevent._csr.binary import (
    binary_csrmv_p,
    binary_csrmm_p,
)
from brainevent._csr.diag_add import csr_diag_add_p
from brainevent._csr.float import (
    csrmv_p,
    csrmm_p,
    csrmv_yw2y_p,
)
from brainevent._csr.masked_float import (
    masked_float_csrmv_p,
    masked_float_csrmm_p,
)
from brainevent._dense.binary import (
    dense_mat_dot_binary_vec_p,
    binary_vec_dot_dense_mat_p,
    dense_mat_dot_binary_mat_p,
    binary_mat_dot_dense_mat_p
)
from brainevent._dense.masked_float import (
    dense_mat_dot_masked_float_vec_p,
    masked_float_vec_dot_dense_mat_p,
    dense_mat_dot_masked_float_mat_p,
    masked_float_mat_dot_dense_mat_p
)
from brainevent._fixed_conn_num.binary import (
    binary_fixed_num_mv_p,
    binary_fixed_num_mm_p,
)
from brainevent._fixed_conn_num.float import (
    fixed_num_mv_p,
    fixed_num_mm_p,
)
from brainevent._fixed_conn_num.masked_float import (
    masked_float_fixed_num_mv_p,
    masked_float_fixed_num_mm_p,
)
from brainevent._jitc_homo.binary import (
    binary_jitc_mv_homo_p,
    binary_jitc_mm_homo_p,
)
from brainevent._jitc_homo.float import (
    float_jitc_homo_matrix_p,
    float_jitc_mv_homo_p,
    float_jitc_mm_homo_p,
)
from brainevent._jitc_normal.binary import (
    binary_jitc_mv_normal_p,
    binary_jitc_mm_normal_p,
)
from brainevent._jitc_normal.float import (
    float_jitc_normal_matrix_p,
    float_jitc_mv_normal_p,
    float_jitc_mm_normal_p,
)
from brainevent._jitc_uniform.binary import (
    binary_jitc_mv_uniform_p,
    binary_jitc_mm_uniform_p,
)
from brainevent._jitc_uniform.float import (
    float_jitc_uniform_matrix_p,
    float_jitc_mv_uniform_p,
    float_jitc_mm_uniform_p,
)

__all__ = [
    'ALL_PRIMITIVES',
    'get_all_primitive_names',
    'get_primitives_by_category',
    'get_primitive_info',
]

# Organized primitive collections
COO_PRIMITIVES = {
    'coomv_p': coomv_p,
    'coomm_p': coomm_p,
    'event_coomv_p': event_coomv_p,
    'event_coomm_p': event_coomm_p,
}

CSR_PRIMITIVES = {
    'csrmv_p': csrmv_p,
    'csrmm_p': csrmm_p,
    'csrmv_yw2y_p': csrmv_yw2y_p,
    'binary_csrmv_p': binary_csrmv_p,
    'binary_csrmm_p': binary_csrmm_p,
    'masked_float_csrmv_p': masked_float_csrmv_p,
    'masked_float_csrmm_p': masked_float_csrmm_p,
    'csr_diag_add_p': csr_diag_add_p,
}

DENSE_PRIMITIVES = {
    'dense_mat_dot_binary_vec_p': dense_mat_dot_binary_vec_p,
    'binary_vec_dot_dense_mat_p': binary_vec_dot_dense_mat_p,
    'dense_mat_dot_binary_mat_p': dense_mat_dot_binary_mat_p,
    'binary_mat_dot_dense_mat_p': binary_mat_dot_dense_mat_p,
    'dense_mat_dot_masked_float_vec_p': dense_mat_dot_masked_float_vec_p,
    'masked_float_vec_dot_dense_mat_p': masked_float_vec_dot_dense_mat_p,
    'dense_mat_dot_masked_float_mat_p': dense_mat_dot_masked_float_mat_p,
    'masked_float_mat_dot_dense_mat_p': masked_float_mat_dot_dense_mat_p,
}

FIXED_CONN_PRIMITIVES = {
    'fixed_num_mv_p': fixed_num_mv_p,
    'fixed_num_mm_p': fixed_num_mm_p,
    'binary_fixed_num_mv_p': binary_fixed_num_mv_p,
    'binary_fixed_num_mm_p': binary_fixed_num_mm_p,
    'masked_float_fixed_num_mv_p': masked_float_fixed_num_mv_p,
    'masked_float_fixed_num_mm_p': masked_float_fixed_num_mm_p,
}

JITC_HOMO_PRIMITIVES = {
    'float_jitc_homo_matrix_p': float_jitc_homo_matrix_p,
    'float_jitc_mv_homo_p': float_jitc_mv_homo_p,
    'float_jitc_mm_homo_p': float_jitc_mm_homo_p,
    'binary_jitc_mv_homo_p': binary_jitc_mv_homo_p,
    'binary_jitc_mm_homo_p': binary_jitc_mm_homo_p,
}

JITC_NORMAL_PRIMITIVES = {
    'float_jitc_normal_matrix_p': float_jitc_normal_matrix_p,
    'float_jitc_mv_normal_p': float_jitc_mv_normal_p,
    'float_jitc_mm_normal_p': float_jitc_mm_normal_p,
    'binary_jitc_mv_normal_p': binary_jitc_mv_normal_p,
    'binary_jitc_mm_normal_p': binary_jitc_mm_normal_p,
}

JITC_UNIFORM_PRIMITIVES = {
    'float_jitc_uniform_matrix_p': float_jitc_uniform_matrix_p,
    'float_jitc_mv_uniform_p': float_jitc_mv_uniform_p,
    'float_jitc_mm_uniform_p': float_jitc_mm_uniform_p,
    'binary_jitc_mv_uniform_p': binary_jitc_mv_uniform_p,
    'binary_jitc_mm_uniform_p': binary_jitc_mm_uniform_p,
}

# Category mappings - centralized to avoid duplication
CATEGORY_COLLECTIONS = {
    'COO': COO_PRIMITIVES,
    'CSR': CSR_PRIMITIVES,
    'Dense': DENSE_PRIMITIVES,
    'FixedConn': FIXED_CONN_PRIMITIVES,
    'JITC_Homo': JITC_HOMO_PRIMITIVES,
    'JITC_Normal': JITC_NORMAL_PRIMITIVES,
    'JITC_Uniform': JITC_UNIFORM_PRIMITIVES,
}

# Combined collection with embedded category metadata
ALL_PRIMITIVES = {}
_PRIMITIVE_CATEGORIES = {}  # Maps kernel objects to categories

for category, collection in CATEGORY_COLLECTIONS.items():
    ALL_PRIMITIVES.update(collection)
    for kernel in collection.values():
        _PRIMITIVE_CATEGORIES[kernel] = category


def get_all_primitive_names() -> List[str]:
    """Get a list of all primitive names defined in brainevent.
    
    Returns:
        List[str]: A sorted list of all primitive names.
        
    Examples:
        >>> import brainevent
        >>> names = brainevent.get_all_primitive_names()
        >>> print(f"Total primitives: {len(names)}")
    """
    return sorted([p.primitive.name for p in ALL_PRIMITIVES.values()])


def get_primitives_by_category() -> Dict[str, List[str]]:
    """Get primitives organized by their functional categories.
    
    Returns:
        Dict[str, List[str]]: A dictionary mapping category names to lists of 
        primitive names in each category.
        
    Examples:
        >>> import brainevent
        >>> categories = brainevent.get_primitives_by_category()
        >>> for category, names in categories.items():
        ...     print(f"{category}: {len(names)} primitives")
    """
    return {
        category: sorted([p.primitive.name for p in collection.values()])
        for category, collection in CATEGORY_COLLECTIONS.items()
    }


def get_primitive_info(primitive_name: str) -> Dict:
    """Get detailed information about a specific primitive.
    
    Args:
        primitive_name: The name of the primitive to query.
        
    Returns:
        Dict containing: name, variable_name, category, kernel_object
        
    Examples:
        >>> import brainevent
        >>> info = brainevent.get_primitive_info('csrmv')
        >>> print(info['category'])
        'CSR'
    """
    for var_name, kernel in ALL_PRIMITIVES.items():
        if kernel.primitive.name == primitive_name:
            return {
                'name': primitive_name,
                'variable_name': var_name,
                'category': _PRIMITIVE_CATEGORIES[kernel],
                'kernel_object': kernel
            }

    raise ValueError(f"Primitive '{primitive_name}' not found. "
                     f"Available: {get_all_primitive_names()}")
