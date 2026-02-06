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

   plast_coo_on_binary_pre
   plast_coo_on_binary_post
   plast_coo_on_binary_pre_p
   plast_coo_on_binary_post_p


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

   plast_csr_on_binary_pre
   plast_csr_on_binary_pre_p
   plast_csr2csc_on_binary_post
   plast_csr2csc_on_binary_post_p

Sparse linear solver.

.. autosummary::
   :toctree: generated/

   csr_solve


Dense Operations
----------------

Dense-matrix @ binary-vector / binary-matrix.

.. autosummary::
   :toctree: generated/

   dm_bv
   dm_bv_p
   bv_dm
   bv_dm_p
   dm_bm
   dm_bm_p
   bm_dm
   bm_dm_p

Indexed binary operations.

.. autosummary::
   :toctree: generated/

   indexed_bv_dm
   indexed_bv_dm_p
   indexed_dm_bv
   indexed_dm_bm
   indexed_bm_dm
   indexed_bm_dm_p

Dense-matrix @ sparse-float-vector / sparse-float-matrix.

.. autosummary::
   :toctree: generated/

   dm_sfv
   dm_sfv_p
   sfv_dm
   sfv_dm_p
   dm_sfm
   dm_sfm_p
   sfm_dm
   sfm_dm_p

Plasticity operations.

.. autosummary::
   :toctree: generated/

   plast_dense_on_binary_pre
   plast_dense_on_binary_pre_p
   plast_dense_on_binary_post
   plast_dense_on_binary_post_p


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
