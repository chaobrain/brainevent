# mmop：CSR 目录与主要函数

层级：CSR → MV / MM / 其他 → 类型或文件 → 主要函数。

树形图采用加宽缩进，并在类型之间留空行。`函数 : 文件.cu` 表示该 CUDA 文件实现对应功能；只列主要函数，省略 JVP。

```text
csr
│
├────── mv
│       │
│       ├────── binary.py
│       │       ├── binary_csrmv / binary_csrmv_p_call：接口与分发
│       │       ├── 前向 numba：_csrmv_numba_kernel
│       │       ├── 前向 jax_raw：_binary_csrmv_jax_kernel
│       │       ├── 前向 cusparse：_binary_csrmv_cusparse_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmv_cuda_kernel : binary_csrmv.cu、binary_csrmv_hybrid.cu
│       │       ├── 反向：_csrmv_transpose_rule
│       │       └── batching：_csrmv_batching（批量 MV 转 MM）
│       │
│       ├────── binary_indexed.py
│       │       ├── binary_csrmv_indexed / binary_csrmv_indexed_p_call：带 perm 的接口
│       │       ├── 前向 numba：_csrmv_indexed_numba_kernel
│       │       ├── 前向 jax_raw：_binary_csrmv_indexed_jax_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmv_indexed_cuda_kernel : binary_indexed_csrmv.cu、binary_indexed_csrmv_hybrid.cu
│       │       ├── 反向：_idx_transpose_rule
│       │       └── batching：_idx_batching
│       │
│       ├────── float.py
│       │       ├── csrmv / csrmv_p_call：普通数值 CSR × 向量
│       │       ├── 前向 numba：_csrmv_numba_kernel_generator
│       │       ├── 前向 jax_raw：_csrmv_jax_kernel
│       │       ├── 前向 cuda_raw：_csrmv_cuda_kernel : float_csrmv.cu
│       │       ├── 反向：_csrmv_transpose_rule
│       │       └── batching：_csrmv_batching
│       │
│       └────── dt2t.py
│               ├── csrmv_dt2t / cscmv_dt2t：神经元向量按连接展开并乘边权
│               ├── csrmv_dt2t_p_call：底层分发
│               ├── 前向 numba：_csrmv_dt2t_numba_kernels
│               ├── 前向 jax_raw：_csrmv_dt2t_jax_kernel
│               └── 前向 cuda_raw：_csrmv_dt2t_cuda_kernel : dt2t.cu
│
├────── mm
│       │
│       ├────── binary.py
│       │       ├── binary_csrmm / binary_csrmm_p_call：接口与分发
│       │       ├── 前向 numba：_csrmm_numba_kernel
│       │       ├── 前向 jax_raw：_binary_csrmm_jax_kernel
│       │       ├── 前向 cusparse：_binary_csrmm_cusparse_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmm_cuda_kernel : binary_csrmm.cu、binary_csrmm_hybrid.cu
│       │       ├── 反向：_csrmm_transpose_rule
│       │       └── batching：_csrmm_batching
│       │
│       ├────── binary_indexed.py
│       │       ├── binary_csrmm_indexed / binary_csrmm_indexed_p_call：带 perm 的接口
│       │       ├── 前向 numba：_csrmm_indexed_numba_kernel
│       │       ├── 前向 jax_raw：_binary_csrmm_indexed_jax_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmm_indexed_cuda_kernel : binary_indexed_csrmm.cu、binary_indexed_csrmm_hybrid.cu
│       │       ├── 反向：_csrmm_idx_transpose_rule
│       │       └── batching：_csrmm_idx_batching
│       │
│       ├────── float.py
│       │       ├── csrmm / csrmm_p_call：普通数值 CSR × 矩阵
│       │       ├── 前向 numba：_csrmm_numba_kernel_generator
│       │       ├── 前向 jax_raw：_csrmm_jax_kernel
│       │       ├── 前向 cuda_raw：_csrmm_cuda_kernel : float_csrmm.cu
│       │       ├── 反向：_csrmm_transpose_rule
│       │       └── batching：_csrmm_batching
│       │
│       └────── dt2t.py
│               ├── csrmm_dt2t / cscmm_dt2t：矩阵输入的逐连接乘法
│               ├── csrmm_dt2t_p_call：底层分发
│               ├── 前向 numba：_csrmm_dt2t_numba_kernels
│               ├── 前向 jax_raw：_csrmm_dt2t_jax_kernel
│               └── 前向 cuda_raw：_csrmm_dt2t_cuda_kernel : dt2t.cu
│
└────── 其他
        │
        ├────── main.py
        │       ├── CompressedSparseData：通用稀疏数据运算
        │       ├── CSR / CSC：构造、格式转换、转置和矩阵乘法
        │       ├── CSR.update_on_pre / update_on_post：突触更新
        │       └── CSR.tocsc / CSC.tocsr：通过 _misc.csr_to_csc_index 转换结构 : csr_to_csc.cu
        │
        ├────── plasticity_binary.py
        │       ├── update_csr_on_binary_pre / update_csc_on_binary_pre：pre spike 更新
        │       ├── update_csr_on_binary_post / update_csc_on_binary_post：post spike 更新
        │       ├── _csr_on_pre_cuda_kernel : plasticity_binary_on_pre.cu
        │       └── _csr2csc_on_post_cuda_kernel : plasticity_binary_on_post.cu
        │
        ├────── slice.py
        │       ├── csr_slice_rows / csr_slice_rows_p_call：抽取 CSR 行
        │       ├── csr_slice_rows_grad / csr_slice_rows_grad_p_call：反向散射
        │       ├── _csr_slice_rows_transpose_rule / _csr_slice_rows_batching
        │       └── _csr_slice_rows_cuda_kernel_generator / _csr_slice_rows_grad_cuda_kernel_generator : slice_csr_slice_rows.cu
        │
        ├────── diag_add.py
        │       ├── csr_diag_position：规划对角元位置
        │       └── csr_diag_add：添加对角元
        │
        ├────── spsolve.py
        │       └── csr_solve：稀疏线性方程求解
        │
        ├────── hybrid_config.py / initialize.py
        │       ├── HybridConfig / get_hybrid_config / save_hybrid_config：hybrid 参数管理
        │       └── run_benchmark / init_csr_config：测速与选择参数
        │
        ├────── _test_util.py / __init__.py
        │       ├── get_csr / vector_csr / matrix_csr 等：测试数据构造
        │       └── __init__.py：统一导出
        │
        └────── 根目录测试文件
                └── _test_util_test.py、binary_test.py、binary_indexed_test.py、
                    float_test.py、dt2t_test.py、diag_add_test.py、slice_test.py、
                    plasticity_binary_test.py、hybrid_config_test.py、
                    initialize_test.py、main_test.py
```

普通 `binary` 的 MV 和 MM 各有 4 个前向 backend；`binary_indexed` 各有 3 个，没有独立的 cusparse backend。indexed CUDA 在共享权重分支复用普通 binary 的对应 `.cu` 文件。
