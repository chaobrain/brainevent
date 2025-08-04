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

"""Central registry for all JAX primitives used in brainevent."""

from typing import List, Dict

from ._coo_impl_float import coomv_p, coomm_p
from ._coo_impl_binary import event_coomv_p, event_coomm_p
from ._csr_impl_float import csrmv_p, csrmm_p, csrmv_yw2y_p
from ._csr_impl_binary import binary_csrmv_p, binary_csrmm_p
from ._csr_impl_masked_float import masked_float_csrmv_p, masked_float_csrmm_p
from ._csr_impl_diag_add import csr_diag_add_p
from ._dense_impl_binary import (
    dense_mat_dot_binary_vec_p, binary_vec_dot_dense_mat_p,
    dense_mat_dot_binary_mat_p, binary_mat_dot_dense_mat_p
)
from ._dense_impl_masked_float import (
    dense_mat_dot_masked_float_vec_p, masked_float_vec_dot_dense_mat_p,
    dense_mat_dot_masked_float_mat_p, masked_float_mat_dot_dense_mat_p
)
from ._fixed_conn_num_impl_float import fixed_num_mv_p, fixed_num_mm_p
from ._fixed_conn_num_impl_binary import binary_fixed_num_mv_p, binary_fixed_num_mm_p
from ._fixed_conn_num_impl_masked_float import masked_float_fixed_num_mv_p, masked_float_fixed_num_mm_p
from ._jitc_homo_impl_float import float_jitc_homo_matrix_p, float_jitc_mv_homo_p, float_jitc_mm_homo_p
from ._jitc_homo_impl_binary import binary_jitc_mv_homo_p, binary_jitc_mm_homo_p
from ._jitc_normal_impl_float import float_jitc_normal_matrix_p, float_jitc_mv_normal_p, float_jitc_mm_normal_p
from ._jitc_normal_impl_binary import binary_jitc_mv_normal_p, binary_jitc_mm_normal_p
from ._jitc_uniform_impl_float import float_jitc_uniform_matrix_p, float_jitc_mv_uniform_p, float_jitc_mm_uniform_p
from ._jitc_uniform_impl_binary import binary_jitc_mv_uniform_p, binary_jitc_mm_uniform_p

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

# Combined collection of all primitives
ALL_PRIMITIVES = {
    **COO_PRIMITIVES,
    **CSR_PRIMITIVES,
    **DENSE_PRIMITIVES,
    **FIXED_CONN_PRIMITIVES,
    **JITC_HOMO_PRIMITIVES,
    **JITC_NORMAL_PRIMITIVES,
    **JITC_UNIFORM_PRIMITIVES,
}


def get_all_primitive_names() -> List[str]:
    """
    Get a list of all primitive names defined in brainevent.
    
    Returns:
        List[str]: A sorted list of all primitive names.
        
    Examples:
        >>> import brainevent.primitives as primitives
        >>> names = primitives.get_all_primitive_names()
        >>> print(f"Total primitives: {len(names)}")
        >>> print("First few primitives:", names[:5])
    """
    return sorted([p.primitive.name for p in ALL_PRIMITIVES.values()])


def get_primitives_by_category() -> Dict[str, List[str]]:
    """
    Get primitives organized by their functional categories.
    
    Returns:
        Dict[str, List[str]]: A dictionary mapping category names to lists of 
        primitive names in each category.
        
    Examples:
        >>> import brainevent.primitives as primitives
        >>> categories = primitives.get_primitives_by_category()
        >>> for category, names in categories.items():
        ...     print(f"{category}: {len(names)} primitives")
    """
    return {
        'COO': sorted([p.primitive.name for p in COO_PRIMITIVES.values()]),
        'CSR': sorted([p.primitive.name for p in CSR_PRIMITIVES.values()]),
        'Dense': sorted([p.primitive.name for p in DENSE_PRIMITIVES.values()]),
        'FixedConn': sorted([p.primitive.name for p in FIXED_CONN_PRIMITIVES.values()]),
        'JITC_Homo': sorted([p.primitive.name for p in JITC_HOMO_PRIMITIVES.values()]),
        'JITC_Normal': sorted([p.primitive.name for p in JITC_NORMAL_PRIMITIVES.values()]),
        'JITC_Uniform': sorted([p.primitive.name for p in JITC_UNIFORM_PRIMITIVES.values()]),
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
    # Category mapping for efficient lookup
    categories = {
        **{id(k): 'COO' for k in COO_PRIMITIVES.values()},
        **{id(k): 'CSR' for k in CSR_PRIMITIVES.values()},
        **{id(k): 'Dense' for k in DENSE_PRIMITIVES.values()},
        **{id(k): 'FixedConn' for k in FIXED_CONN_PRIMITIVES.values()},
        **{id(k): 'JITC_Homo' for k in JITC_HOMO_PRIMITIVES.values()},
        **{id(k): 'JITC_Normal' for k in JITC_NORMAL_PRIMITIVES.values()},
        **{id(k): 'JITC_Uniform' for k in JITC_UNIFORM_PRIMITIVES.values()},
    }
    
    for var_name, kernel in ALL_PRIMITIVES.items():
        if kernel.primitive.name == primitive_name:
            return {
                'name': primitive_name,
                'variable_name': var_name,
                'category': categories[id(kernel)],
                'kernel_object': kernel
            }
    
    raise ValueError(f"Primitive '{primitive_name}' not found. "
                     f"Available: {get_all_primitive_names()}")