Run mypy brainevent/ > mypy-report.txt || true
  mypy brainevent/ > mypy-report.txt || true
  mypy-baseline filter --allow-unsynced --baseline-path mypy-baseline.txt < mypy-report.txt
  shell: /usr/bin/bash -e {0}
  env:
    pythonLocation: /opt/hostedtoolcache/Python/3.13.13/x64
    PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.13.13/x64/lib/pkgconfig
    Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.13/x64
    Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.13/x64
    Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.13/x64
    LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.13.13/x64/lib
brainevent/_misc.py:1543: error: Unsupported target for indexed assignment ("Any | None")  [index]
Found 52 errors in 9 files (checked 78 source files)

total errors:
  fixed: 41
  new: 1
  unresolved: 61

errors by error code:
  union-attr                40  -3
  note                      26 -18
  assignment                18 -14
  arg-type                   4
  return-value               3  -3
  misc                       3  -3
  index                      3  +1
  var-annotated              2
  annotation-unchecked       2
  return                     1
  name-defined               1



def _check_csr_cuda_structure_dtypes(indices_info, indptr_info) -> None:
+     """Validate the raw CUDA CSR ABI: int32 indices and int32/int64 indptr."""
+     indices_dtype = jnp.dtype(indices_info.dtype)
+     indptr_dtype = jnp.dtype(indptr_info.dtype)
+     if indices_dtype != jnp.dtype(jnp.int32):
+         raise TypeError(
+             "CSR cuda_raw kernels require indices with dtype int32; "
+             f"got indices dtype {indices_dtype}."
+         )
+     if indptr_dtype not in _CSR_SIGNED_INDEX_DTYPES:
+         raise TypeError(
+             "CSR cuda_raw kernels require indptr with dtype int32 or int64; "
+             f"got indptr dtype {indptr_dtype}."
+         )
+ 

+ def _csr_to_csc_index_numpy(
+     csr_indptr: Union[jax.Array, np.ndarray],
+     csr_indices: Union[jax.Array, np.ndarray],
+     *,
+     shape: Tuple[int, int],
+     include_perm: bool = True,
+ ):
+     """Convert CSR indices to CSC on CPU with NumPy, then restore array type."""
+     n_post = shape[1]
+     coord_dtype = _coordinate_index_dtype(getattr(csr_indices, 'dtype', np.int32))
+     nse = getattr(csr_indices, 'size', None)
+     if nse is None:
+         nse = len(csr_indices)
+     offset_dtype = _offset_index_dtype(nse, getattr(csr_indptr, 'dtype', None))
+ 
+     csr_indptr_np = np.asarray(csr_indptr)
+     csr_indices_np = np.asarray(csr_indices)
+ 
+     counts = np.bincount(csr_indices_np, minlength=n_post).astype(offset_dtype, copy=False)
+     csc_indptr_np = np.empty(n_post + 1, dtype=offset_dtype)
+     csc_indptr_np[0] = 0
+     np.cumsum(counts, dtype=offset_dtype, out=csc_indptr_np[1:])
+ 
+     order_np = np.argsort(csr_indices_np, kind='stable')
+     order_np = np.asarray(order_np, dtype=offset_dtype)
+     csc_indices_np = np.searchsorted(csr_indptr_np, order_np, side='right') - 1
+     csc_indices_np = np.asarray(csc_indices_np, dtype=coord_dtype)
+     perm_np = order_np if include_perm else None
+ 
+     if isinstance(csr_indptr, np.ndarray) and isinstance(csr_indices, np.ndarray):
+         return csc_indptr_np, csc_indices_np, perm_np
+ 
+     old_x64 = jax.config.jax_enable_x64
+     needs_x64 = _normalize_dtype(offset_dtype) == np.dtype(np.int64)
+     if needs_x64 and not old_x64:
+         jax.config.update('jax_enable_x64', True)
+     try:
+         csc_indptr = jnp.asarray(csc_indptr_np)
+         csc_indices = jnp.asarray(csc_indices_np)
+         perm = None if perm_np is None else jnp.asarray(perm_np)
+         return csc_indptr, csc_indices, perm
+     finally:
+         if needs_x64 and not old_x64:
+             jax.config.update('jax_enable_x64', False)

+ def _load_csr_to_csc_cuda_module():
+     global _CSR_TO_CSC_CUDA_MODULE
+     if _CSR_TO_CSC_CUDA_MODULE is None:
+         from pathlib import Path
+ 
+         from ._op import load_cuda_file
+ 
+         _CSR_TO_CSC_CUDA_MODULE = load_cuda_file(
+             Path(__file__).resolve().parent / "_csr" / "csr_to_csc.cu",
+             name="csr_to_csc",
+         )
+     return _CSR_TO_CSC_CUDA_MODULE
+ 
+ 
+ def _csr_to_csc_index_gpu_column_block(
+     csr_indptr: Union[jax.Array, np.ndarray],
+     csr_indices: Union[jax.Array, np.ndarray],
+     *,
+     shape: Tuple[int, int],
+     include_perm: bool = True,
+     column_block_size: int = 4096,
+ ):
+     """Convert CSR indices to CSC using CUDA column blocks and CPU stitching."""
+     n_post = shape[1]
+     try:
+         column_block_size = int(column_block_size)
+     except (TypeError, ValueError) as exc:
+         raise ValueError("column_block_size must be a positive integer") from exc
+     if column_block_size <= 0:
+         raise ValueError("column_block_size must be a positive integer")
+ 
+     coord_dtype = _coordinate_index_dtype(getattr(csr_indices, 'dtype', np.int32))
+     nse = getattr(csr_indices, 'size', None)
+     if nse is None:
+         nse = len(csr_indices)
+     nse = int(nse)
+     offset_dtype = _offset_index_dtype(nse, getattr(csr_indptr, 'dtype', None))
+ 
+     old_x64 = jax.config.jax_enable_x64
+     needs_x64 = (
+         _normalize_dtype(offset_dtype) == np.dtype(np.int64) or
+         _normalize_dtype(coord_dtype) == np.dtype(np.int64)
+     )
+     if needs_x64 and not old_x64:
+         jax.config.update('jax_enable_x64', True)
+ 
+     try:
+         try:
+             gpu_device = jax.devices("gpu")[0]
+             _load_csr_to_csc_cuda_module()
+         except Exception:
+             return _csr_to_csc_index_numpy(
+                 csr_indptr,
+                 csr_indices,
+                 shape=shape,
+                 include_perm=include_perm,
+             )
+ 
+         csr_indices_dev = jax.device_put(
+             jnp.asarray(csr_indices, dtype=coord_dtype),
+             gpu_device,
+         )
+         csr_indptr_dev = jax.device_put(
+             jnp.asarray(csr_indptr, dtype=offset_dtype),
+             gpu_device,
+         )
+ 
+         counts_dev = jax.ffi.ffi_call(
+             "csr_to_csc.csr_to_csc_count",
+             jax.ShapeDtypeStruct((n_post,), offset_dtype),
+         )(csr_indices_dev, csr_indptr_dev)
+         counts_np = np.asarray(counts_dev, dtype=offset_dtype)
+ 
+         csc_indptr_np = np.empty(n_post + 1, dtype=offset_dtype)
+         csc_indptr_np[0] = 0
+         np.cumsum(counts_np, dtype=offset_dtype, out=csc_indptr_np[1:])
+ 
+         if int(csc_indptr_np[-1]) != nse:
+             raise RuntimeError(
+                 "CUDA CSR-to-CSC count produced an unexpected nnz total: "
+                 f"{int(csc_indptr_np[-1])} != {nse}"
+             )
+ 
+         csc_indices_np = np.empty(nse, dtype=coord_dtype)
+         perm_np = np.empty(nse, dtype=offset_dtype) if include_perm else None
+ 
+         for col_start in range(0, n_post, column_block_size):
+             col_end = min(col_start + column_block_size, n_post)
+             base = int(csc_indptr_np[col_start])
+             end = int(csc_indptr_np[col_end])
+             block_nnz = end - base
+             block_ncols = col_end - col_start
+ 
+             if block_nnz == 0:
+                 continue
+ 
+             local_indptr_np = (
+                 csc_indptr_np[col_start:col_end + 1] -
+                 csc_indptr_np[col_start]
+             ).astype(offset_dtype, copy=False)
+             initial_pos_dev = jax.device_put(
+                 jnp.asarray(local_indptr_np[:-1], dtype=offset_dtype),
+                 gpu_device,
+             )
+ 
+             scratch_info = jax.ShapeDtypeStruct((block_ncols,), offset_dtype)
+             rows_info = jax.ShapeDtypeStruct((block_nnz,), coord_dtype)
+             perm_info = jax.ShapeDtypeStruct((block_nnz,), offset_dtype)
+             _, local_rows_dev, local_perm_dev = jax.ffi.ffi_call(
+                 "csr_to_csc.csr_to_csc_fill_block",
+                 (scratch_info, rows_info, perm_info),
+             )(
+                 csr_indices_dev,
+                 csr_indptr_dev,
+                 initial_pos_dev,
+                 col_start=np.int64(col_start),
+                 col_end=np.int64(col_end),
+             )
+ 
+             csc_indices_np[base:end] = np.asarray(local_rows_dev, dtype=coord_dtype)
+             if include_perm:
+                 perm_np[base:end] = np.asarray(local_perm_dev, dtype=offset_dtype)
+ 
+         if isinstance(csr_indptr, np.ndarray) and isinstance(csr_indices, np.ndarray):
+             return csc_indptr_np, csc_indices_np, perm_np
+ 
+         csc_indptr = jax.device_put(csc_indptr_np, gpu_device)
+         csc_indices = jax.device_put(csc_indices_np, gpu_device)
+         perm = None if perm_np is None else jax.device_put(perm_np, gpu_device)
+         return csc_indptr, csc_indices, perm
+     finally:
+         if needs_x64 and not old_x64:
+             jax.config.update('jax_enable_x64', False)
+ 

+     elif method == "gpu_column_block":
+         csc_indptr, csc_indices, post_positions = _csr_to_csc_index_gpu_column_block(


-     """cuSPARSE-backed kernel for binary CSR SpMV via jax.experimental.sparse (GPU only)."""
+     """cuSPARSE-backed kernel for binary CSR SpMV via BCOO/BCSR sparse arrays."""
      import jax.experimental.sparse as jsparse
      m, k = shape
      is_homo = (weight_info.size == 1)

+ def _binary_csrmv_jax_exp_csrmv_kernel(
+     weight_info: jax.ShapeDtypeStruct,
+     vector_info: jax.ShapeDtypeStruct,
+     shape: MatrixShape,
+     transpose: bool,
+     **kwargs,
+ ):
+     """JAX experimental CSR-backed binary CSR SpMV reference kernel."""
+     import jax.experimental.sparse as jsparse
+     m, k = shape
+     is_homo = (weight_info.size == 1)
+     is_bool = (vector_info.dtype == jnp.bool_)
+     nse = kwargs['indices_info'].size
+     out_dtype = kwargs['outs'][0].dtype
+ 
+     def kernel(weights, indices, indptr, vector):
+         events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
+         indices = indices.astype(indptr.dtype) if indices.dtype != indptr.dtype else indices
+         if is_homo:
+             data = jnp.ones(nse, dtype=out_dtype)
+             mat = jsparse.CSR((data, indices, indptr), shape=(m, k))
+             return (jsparse.csr_matvec(mat, events, transpose=transpose) * weights[0].astype(out_dtype),)
+         mat = jsparse.CSR((weights.astype(out_dtype), indices, indptr), shape=(m, k))
+         return (jsparse.csr_matvec(mat, events, transpose=transpose),)
+ 
+     return kernel

+     _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
      load_cuda_file(
          Path(__file__).parent.joinpath('binary_csrmv.cu'),

+     _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
      load_cuda_file(

+     _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
      is_homo = (weight_info.size == 1)
      if is_homo:
          load_cuda_file(

      else:
          def kernel(weights, indices, indptr, vector):
+             indices = indices.astype(indptr.dtype) if indices.dtype != indptr.dtype else indices
+             vector = vector.astype(weight_info.dtype) if vector.dtype != weight_info.dtype else vector
              return csr_matvec_p.bind(weights, indices, indptr, vector, shape=kwargs['shape'], transpose=transpose),
  
      return kernel

+     _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
      is_homo = (weight_info.size == 1)
      if is_homo:
          load_cuda_file(

      else:
          def kernel(weights, indices, indptr, B):
+             indices = indices.astype(indptr.dtype) if indices.dtype != indptr.dtype else indices
+             B = B.astype(weight_info.dtype) if B.dtype != weight_info.dtype else B
              return csr_matmat_p.bind(weights, indices, indptr, B, shape=kwargs['shape'], transpose=transpose),
  
      return kernel


