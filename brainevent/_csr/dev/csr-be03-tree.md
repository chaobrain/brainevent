# be0.3：CSR 目录与主要函数

层级：CSR → MV / MM / 其他 → 类型或文件 → 主要函数。

树形图采用加宽缩进，并在类型之间留空行。`函数 : 文件.cu` 表示该 CUDA 文件实现对应功能；只列主要函数，省略 JVP。

```text
csr
│
├────── mv
│       │
│       ├────── binary/binary.py
│       │       ├── binary_csrmv / binary_csrmv_p_call：binary 或 weighted 事件 MV
│       │       ├── 前向 numba：_csrmv_numba_kernel（weighted 分支：_csrmv_numba_weighted）
│       │       ├── 前向 jax_raw：_binary_csrmv_jax_kernel
│       │       ├── 前向 cusparse：_binary_csrmv_cusparse_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmv_cuda_kernel : binary_csrmv.cu、binary_csrmv_hybrid.cu
│       │       ├── 反向：_csrmv_transpose_rule（逐边权重梯度：_entry_weight_ct）
│       │       └── batching：_csrmv_batching（批量 MV 转 MM）
│       │
│       ├────── binary/binary_indexed.py
│       │       ├── binary_csrmv_indexed / binary_csrmv_indexed_p_call：带 perm 的事件 MV
│       │       ├── 前向 numba：_csrmv_indexed_numba_kernel
│       │       ├── 前向 jax_raw：_binary_csrmv_indexed_jax_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmv_indexed_cuda_kernel : binary_indexed_csrmv.cu、binary_indexed_csrmv_hybrid.cu
│       │       ├── 反向：_idx_transpose_rule
│       │       └── batching：_idx_batching
│       │
│       ├────── float/float.py
│       │       ├── csrmv / csrmv_p_call：普通数值 CSR × 向量
│       │       ├── 前向 numba：_csrmv_numba_kernel_generator
│       │       ├── 前向 jax_raw：_csrmv_jax_kernel
│       │       ├── 前向 cuda_raw：_csrmv_cuda_kernel : float_csrmv.cu
│       │       ├── 反向：_csrmv_transpose_rule
│       │       └── batching：_csrmv_batching
│       │
│       └────── dt2t/dt2t.py
│               ├── csrmv_dt2t / cscmv_dt2t：向量按连接展开并乘边权
│               ├── csrmv_dt2t_p_call：底层分发
│               ├── 前向 numba：_csrmv_dt2t_numba_kernels
│               ├── 前向 jax_raw：_csrmv_dt2t_jax_kernel
│               └── 前向 cuda_raw：_csrmv_dt2t_cuda_kernel : dt2t.cu
│
├────── mm
│       │
│       ├────── binary/binary.py
│       │       ├── binary_csrmm / binary_csrmm_p_call：binary 或 weighted 事件 MM
│       │       ├── 前向 numba：_csrmm_numba_kernel（weighted 分支：_csrmm_numba_weighted）
│       │       ├── 前向 jax_raw：_binary_csrmm_jax_kernel
│       │       ├── 前向 cusparse：_binary_csrmm_cusparse_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmm_cuda_kernel : binary_csrmm.cu、binary_csrmm_hybrid.cu
│       │       ├── 反向：_csrmm_transpose_rule（逐边权重梯度：_csrmm_entry_weight_ct）
│       │       └── batching：_csrmm_batching
│       │
│       ├────── binary/binary_indexed.py
│       │       ├── binary_csrmm_indexed / binary_csrmm_indexed_p_call：带 perm 的事件 MM
│       │       ├── 前向 numba：_csrmm_indexed_numba_kernel
│       │       ├── 前向 jax_raw：_binary_csrmm_indexed_jax_kernel
│       │       ├── 前向 cuda_raw：_binary_csrmm_indexed_cuda_kernel : binary_indexed_csrmm.cu、binary_indexed_csrmm_hybrid.cu
│       │       ├── 反向：_csrmm_idx_transpose_rule
│       │       └── batching：_csrmm_idx_batching
│       │
│       ├────── float/float.py
│       │       ├── csrmm / csrmm_p_call：普通数值 CSR × 矩阵
│       │       ├── 前向 numba：_csrmm_numba_kernel_generator
│       │       ├── 前向 jax_raw：_csrmm_jax_kernel
│       │       ├── 前向 cuda_raw：_csrmm_cuda_kernel : float_csrmm.cu
│       │       ├── 反向：_csrmm_transpose_rule
│       │       └── batching：_csrmm_batching
│       │
│       └────── dt2t/dt2t.py
│               ├── csrmm_dt2t / cscmm_dt2t：矩阵输入的逐连接乘法
│               ├── csrmm_dt2t_p_call：底层分发
│               ├── 前向 numba：_csrmm_dt2t_numba_kernels
│               ├── 前向 jax_raw：_csrmm_dt2t_jax_kernel
│               └── 前向 cuda_raw：_csrmm_dt2t_cuda_kernel : dt2t.cu
│
└────── 其他
        │
        ├────── main.py / _support/
        │       ├── CompressedSparseData / CSR / CSC：通用稀疏对象与矩阵运算
        │       ├── CSR.apply_on_pre / apply_on_post：自定义逐边可塑性规则
        │       ├── CSR.delay_deliver / delay_layout：延迟投递入口
        │       └── CSR.tocsc / CSC.tocsr：通过 _misc.csr_to_csc_index 转换结构 : _support/csr_to_csc.cu
        │
        ├────── delay/delay.py
        │       ├── csr_delay_deliver / csr_delay_deliver_p_call：按边延迟投递到环形缓冲区
        │       ├── 前向 jax_raw：_jax_kernel
        │       ├── 前向 numba：_numba_kernel_generator
        │       ├── 前向 cuda_raw：_cuda_kernel : csr_delay_deliver.cu
        │       ├── 反向：_transpose_rule
        │       └── batching：_batching_rule
        │
        ├────── delay/delay_layout.py
        │       ├── CSRDelayLayout.build / with_data / original_data / deliver：按 delay 重排和使用边
        │       ├── csr_delay_deliver_layout / csr_delay_deliver_layout_p_call：按预处理布局投递
        │       ├── 前向 cuda_raw：_cuda_kernel : csr_delay_deliver_layout.cu
        │       ├── 反向：_layout_transpose_rule
        │       └── batching：_batching_rule
        │
        ├────── plasticity/plasticity_binary.py
        │       ├── update_csr_on_binary_pre / update_csc_on_binary_pre：pre spike 更新
        │       ├── update_csr_on_binary_post / update_csc_on_binary_post：post spike 更新
        │       ├── _csr_on_pre_cuda_kernel : plasticity_binary_on_pre.cu
        │       ├── _csr2csc_on_post_cuda_kernel : plasticity_binary_on_post.cu
        │       └── _csr_on_pre_transpose_rule / _csr2csc_on_post_transpose_rule：反向
        │
        ├────── plasticity/plasticity_generic.py
        │       └── csr_apply_on_pre / csr_apply_on_post：用户自定义逐边更新规则
        │
        ├────── slice/slice.py
        │       ├── csr_slice_rows / csr_slice_rows_grad：抽行及其反向散射
        │       ├── _csr_slice_rows_transpose_rule / _csr_slice_rows_batching
        │       └── _csr_slice_rows_cuda_kernel_generator / _csr_slice_rows_grad_cuda_kernel_generator : slice_csr_slice_rows.cu
        │
        ├────── diag/diag_add.py
        │       ├── csr_diag_position：规划对角元位置
        │       └── csr_diag_add：添加对角元
        │
        ├────── solve/spsolve.py
        │       └── csr_solve：稀疏线性方程求解
        │
        ├────── binary/hybrid_config.py / binary/initialize.py
        │       ├── HybridConfig / get_hybrid_config / save_hybrid_config：hybrid 参数管理
        │       └── run_benchmark / init_csr_config：测速与选择参数
        │
        ├────── dev/delay_mv/
        │       ├── binary_delay.py：split_region_delay : region_delay_split.cu
        │       ├── binary_delay.py：_binary_csrmv_cuda_kernel : binary_csrmv_hybrid_region_delay.cu
        │       ├── binary_delay.py：_binary_csrmv_cusparse_kernel : region_delay_cusparse.cu
        │       └── delay_main.py：prepare_delay_metadata / delay_tcsr
        │
        ├────── _test_util.py / 各级 __init__.py
        │       ├── _test_util.py：CSR 与事件测试数据构造
        │       └── __init__.py：各子包导出
        │
        └────── test/ 与 dev/docs/
                ├── test/：main_test.py、_test_util_test.py；各功能目录有对应 test/ 文件夹
                └── dev/docs/：历史设计文档、说明和教程，无运行时函数
```

普通 `binary` 的 MV 和 MM 各有 4 个前向 backend；`binary_indexed` 各有 3 个，没有独立的 cusparse backend。indexed CUDA 在共享权重分支复用普通 binary 的对应 `.cu` 文件。
