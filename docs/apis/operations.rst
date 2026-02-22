Matrix Operations
=================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


COO Operations
--------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_coomv
   binary_coomv_p
   binary_coomm
   binary_coomm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   coomv
   coomv_p
   coomm
   coomm_p

Plasticity operations.

.. autosummary::
   :toctree: generated/

   update_coo_on_binary_pre
   update_coo_on_binary_post
   update_coo_on_binary_pre_p
   update_coo_on_binary_post_p


CSR Operations
--------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_csrmv
   binary_csrmv_p
   binary_csrmm
   binary_csrmm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   csrmv
   csrmv_p
   csrmm
   csrmm_p
   csrmv_yw2y
   csrmv_yw2y_p

Sparse-float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   spfloat_csrmv
   spfloat_csrmv_p
   spfloat_csrmm
   spfloat_csrmm_p

Plasticity operations.

.. autosummary::
   :toctree: generated/

   update_csr_on_binary_pre
   update_csr_on_binary_pre_p
   update_csr_on_binary_post
   update_csr_on_binary_post_p

Sparse linear solver.

.. autosummary::
   :toctree: generated/

   csr_solve

Row slicing.

.. autosummary::
   :toctree: generated/

   csr_slice_rows
   csr_slice_rows_p


Dense Operations
----------------

Dense-matrix @ binary-vector / binary-matrix.

.. autosummary::
   :toctree: generated/

   binary_densemv
   binary_densemv_p
   binary_densemm
   binary_densemm_p

Indexed binary operations.

.. autosummary::
   :toctree: generated/

   indexed_binary_densemv
   indexed_binary_densemv_p
   indexed_binary_densemm
   indexed_binary_densemm_p

Dense-matrix @ sparse-float-vector / sparse-float-matrix.

.. autosummary::
   :toctree: generated/

   spfloat_densemv
   spfloat_densemv_p
   spfloat_densemm
   spfloat_densemm_p

Plasticity operations.

.. autosummary::
   :toctree: generated/

   update_dense_on_binary_pre
   update_dense_on_binary_pre_p
   update_dense_on_binary_post
   update_dense_on_binary_post_p


JITC Scalar Operations
----------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_jitsmv
   binary_jitsmv_p
   binary_jitsmm
   binary_jitsmm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   jits
   jits_p
   jitsmv
   jitsmv_p
   jitsmm
   jitsmm_p


JITC Normal Operations
----------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_jitnmv
   binary_jitnmv_p
   binary_jitnmm
   binary_jitnmm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   jitn
   jitn_p
   jitnmv
   jitnmv_p
   jitnmm
   jitnmm_p


JITC Uniform Operations
------------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_jitumv
   binary_jitumv_p
   binary_jitumm
   binary_jitumm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   jitu
   jitu_p
   jitumv
   jitumv_p
   jitumm
   jitumm_p


Fixed Connectivity Operations
-----------------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_fcnmv
   binary_fcnmv_p
   binary_fcnmm
   binary_fcnmm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   fcnmv
   fcnmv_p
   fcnmm
   fcnmm_p

Sparse-float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   spfloat_fcnmv
   spfloat_fcnmv_p
   spfloat_fcnmm
   spfloat_fcnmm_p
