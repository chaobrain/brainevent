# be0.3 相对 mmop 的 CSR 修改记录

层级：CSR → MV / MM / 其他 → 类型或文件 → 主要函数。

每行末尾的 `：` 后只填写相对 mmop 的修改内容；无修改则留空。方括号内保留对应的 CUDA 文件名，供定位使用。当前所有修改内容均待逐项确认。

```text
csr：主要修改提纲
│   1. binary / binary_indexed：新增 weighted 事件支持，完善前向、梯度与 batching 的事件语义。
│   2. main：整合事件类型分派，接入 weighted、delay 与突触可塑性接口。
│   3. delay：新增延迟投递及预处理布局功能。
│   4. 突触可塑性：完善 pre/post 更新及梯度支持，新增自定义连接状态更新接口。
│
├────── mv
│       │
│       ├────── binary/binary.py
│       │       ├── binary_csrmv / binary_csrmv_p_call：
│       │       ├── 前向 numba _csrmv_numba_kernel / _csrmv_numba_weighted：
│       │       ├── 前向 jax_raw _binary_csrmv_jax_kernel：event_active 将数值事件的激活标准统一为 != 0，event 为 NaN/±Inf 时仍激活；_event_contrib 通过 select_active 隔离异常权重：event 为 0 时贡献置 0，NaN/±Inf 权重不会传播；event 激活时异常权重仍参与运算并传播。
│       │       ├── 前向 cusparse _binary_csrmv_cusparse_kernel：新增异常权重回退：权重全部有限时使用 cuSPARSE，含 NaN/±Inf 时改用 JAX 逐连接选择贡献，避免未激活事件的 0 × NaN/Inf 污染输出。例外是 binary 共享标量权重：仍用 cuSPARSE 统计活跃事件数，再将零计数位置的输出置 0。
│       │       ├── 前向 cuda_raw _binary_csrmv_cuda_kernel [binary_csrmv.cu、binary_csrmv_hybrid.cu]：新增 weighted 支持；hybrid 的 weighted 任务记录起点与行号，按块长及 indptr 重建终点并回读事件值。后续拟推广为可扩展任务列表：binary 仅存起点，定位所属行后按 min(begin + 块长, indptr[row + 1]) 确定终点（行定位方式待定）；weighted 按需增加一列事件值列表，batch 按需增加一列批次索引列表。
│       │       ├── 反向 _csrmv_transpose_rule / _entry_weight_ct：新增 _entry_weight_ct / _event_ct 等辅助函数，为逐连接权重梯度套用 event_active（以非零判定激活）和 select_active（inactive 梯度置零，避免 NaN/Inf 污染）。后续拟删除该路径的辅助函数及包装逻辑，权重梯度统一交由 SDDMV/SDDMM 算子计算，由算子内部直接保证上述语义及 weighted 计算的正确性。
│       │       └── batching _csrmv_batching：
│       │
│       ├────── binary/binary_indexed.py
│       │       ├── binary_csrmv_indexed / binary_csrmv_indexed_p_call：
│       │       ├── 前向 numba _csrmv_indexed_numba_kernel：
│       │       ├── 前向 jax_raw _binary_csrmv_indexed_jax_kernel：
│       │       ├── 前向 cuda_raw _binary_csrmv_indexed_cuda_kernel [binary_indexed_csrmv.cu、binary_indexed_csrmv_hybrid.cu]：
│       │       ├── 反向 _idx_transpose_rule：
│       │       └── batching _idx_batching：
│       │
│       ├────── float/float.py
│       │       ├── csrmv / csrmv_p_call：
│       │       ├── 前向 numba _csrmv_numba_kernel_generator：
│       │       ├── 前向 jax_raw _csrmv_jax_kernel：
│       │       ├── 前向 cuda_raw _csrmv_cuda_kernel [float_csrmv.cu]：
│       │       ├── 反向 _csrmv_transpose_rule：
│       │       └── batching _csrmv_batching：
│       │
│       └────── dt2t/dt2t.py
│               ├── csrmv_dt2t / cscmv_dt2t：
│               ├── csrmv_dt2t_p_call：
│               ├── 前向 numba _csrmv_dt2t_numba_kernels：
│               ├── 前向 jax_raw _csrmv_dt2t_jax_kernel：
│               └── 前向 cuda_raw _csrmv_dt2t_cuda_kernel [dt2t.cu]：
│
├────── mm
│       │
│       ├────── binary/binary.py
│       │       ├── binary_csrmm / binary_csrmm_p_call：
│       │       ├── 前向 numba _csrmm_numba_kernel / _csrmm_numba_weighted：
│       │       ├── 前向 jax_raw _binary_csrmm_jax_kernel：event_active 将数值事件的激活标准统一为 != 0，event 为 NaN/±Inf 时仍激活；_event_contrib 通过 select_active 隔离异常权重：event 为 0 时贡献置 0，NaN/±Inf 权重不会传播；event 激活时异常权重仍参与运算并传播。
│       │       ├── 前向 cusparse _binary_csrmm_cusparse_kernel：新增异常权重回退：权重全部有限时使用 cuSPARSE，含 NaN/±Inf 时改用 JAX 按连接、按事件列选择贡献，避免未激活事件的 0 × NaN/Inf 污染输出。例外是 binary 共享标量权重：仍用 cuSPARSE 统计各列的活跃事件数，再将零计数位置的输出置 0。
│       │       ├── 前向 cuda_raw _binary_csrmm_cuda_kernel [binary_csrmm.cu、binary_csrmm_hybrid.cu]：
│       │       ├── 反向 _csrmm_transpose_rule / _csrmm_entry_weight_ct：计算权重梯度 dW 时，在 _csrmm_entry_weight_ct 中通过 jax.lax.cond(jnp.all(jnp.isfinite(ct)), sddmm, select, ct) 检查整个上游梯度 ct；无 NaN/±Inf 时走 sddmm 分支，调用 JAX 的 sddmm_coo_indices 包装；存在 NaN/±Inf 时走 select 分支，调用 _entry_weight_ct 按连接、按事件列选择贡献并求和，将未激活项置零，活跃项的异常值仍传播。
│       │       └── batching _csrmm_batching：
│       │
│       ├────── binary/binary_indexed.py
│       │       ├── binary_csrmm_indexed / binary_csrmm_indexed_p_call：
│       │       ├── 前向 numba _csrmm_indexed_numba_kernel：
│       │       ├── 前向 jax_raw _binary_csrmm_indexed_jax_kernel：
│       │       ├── 前向 cuda_raw _binary_csrmm_indexed_cuda_kernel [binary_indexed_csrmm.cu、binary_indexed_csrmm_hybrid.cu]：
│       │       ├── 反向 _csrmm_idx_transpose_rule：
│       │       └── batching _csrmm_idx_batching：
│       │
│       ├────── float/float.py
│       │       ├── csrmm / csrmm_p_call：
│       │       ├── 前向 numba _csrmm_numba_kernel_generator：
│       │       ├── 前向 jax_raw _csrmm_jax_kernel：
│       │       ├── 前向 cuda_raw _csrmm_cuda_kernel [float_csrmm.cu]：
│       │       ├── 反向 _csrmm_transpose_rule：
│       │       └── batching _csrmm_batching：
│       │
│       └────── dt2t/dt2t.py
│               ├── csrmm_dt2t / cscmm_dt2t：
│               ├── csrmm_dt2t_p_call：
│               ├── 前向 numba _csrmm_dt2t_numba_kernels：
│               ├── 前向 jax_raw _csrmm_dt2t_jax_kernel：
│               └── 前向 cuda_raw _csrmm_dt2t_cuda_kernel [dt2t.cu]：
│
└────── 其他
        │
        ├────── main.py / _support/
        │       ├── CompressedSparseData / CSR / CSC：
        │       ├── CSR.apply_on_pre / apply_on_post：
        │       ├── CSR.delay_deliver / delay_layout：
        │       └── CSR.tocsc / CSC.tocsr [_support/csr_to_csc.cu]：
        │
        │   delay 两种方案的反向规则一致：每次 event 向量输入为 N × 1，ct 为 D × N（D 为 ring 槽数），对应一份 N × N 的 dW，仅计算 CSR 已有连接的梯度。
        │   按连接的 delay 和目标位置，从 ct[(t + delay) % D, post] 抽取对应值，与源 event 相乘得到该连接的 dW；weighted 使用事件值，binary 使用激活掩码。
        │
        ├────── delay/delay.py：面向通用 CSR 的逐边延迟投递，使用循环缓冲区 ring 保存待到达贡献；indptr 标识 CSR 行边界，独立的 delays 数组按边保存延迟。调用者维护 t，以 t % L 定位当前前沿，并负责读取、清空和推进。CUDA 根据整张矩阵的平均行长度静态选择 TPR / WPR / BPR 调度。
        │       ├── csr_delay_deliver / csr_delay_deliver_p_call：
        │       ├── 前向 jax_raw _jax_kernel：
        │       ├── 前向 numba _numba_kernel_generator：
        │       ├── 前向 cuda_raw _cuda_kernel [csr_delay_deliver.cu]：
        │       ├── 反向 _transpose_rule：
        │       └── batching _batching_rule：
        │
        ├────── delay/delay_layout.py：为改善延迟投递的访问局部性，将每个 CSR 行内的连接按 delay 排序，以 delay_ptr 定位各延迟分段。CUDA 先压缩活跃行，再对每个 (delay, 活跃行) 工作组合固定分配 16 个线程，采用静态调度；ring 及当前前沿仍由调用者维护。
        │       ├── CSRDelayLayout.build / with_data / original_data / deliver：
        │       ├── csr_delay_deliver_layout / csr_delay_deliver_layout_p_call：
        │       ├── 前向 cuda_raw _cuda_kernel [csr_delay_deliver_layout.cu]：
        │       ├── 反向 _layout_transpose_rule：
        │       └── batching _batching_rule：
        │
        ├────── plasticity/plasticity_binary.py
        │       ├── update_csr_on_binary_pre / update_csc_on_binary_pre：
        │       ├── update_csr_on_binary_post / update_csc_on_binary_post：
        │       ├── _csr_on_pre_cuda_kernel [plasticity_binary_on_pre.cu]：
        │       ├── _csr2csc_on_post_cuda_kernel [plasticity_binary_on_post.cu]：
        │       └── _csr_on_pre_transpose_rule / _csr2csc_on_post_transpose_rule：
        │
        ├────── plasticity/plasticity_generic.py
        │       └── csr_apply_on_pre / csr_apply_on_post：
        │
        ├────── slice/slice.py
        │       ├── csr_slice_rows / csr_slice_rows_grad：
        │       ├── _csr_slice_rows_transpose_rule / _csr_slice_rows_batching：
        │       └── _csr_slice_rows_cuda_kernel_generator / _csr_slice_rows_grad_cuda_kernel_generator [slice_csr_slice_rows.cu]：
        │
        ├────── diag/diag_add.py
        │       ├── csr_diag_position：
        │       └── csr_diag_add：
        │
        ├────── solve/spsolve.py
        │       └── csr_solve：
        │
        ├────── binary/hybrid_config.py / binary/initialize.py
        │       ├── HybridConfig / get_hybrid_config / save_hybrid_config：
        │       └── run_benchmark / init_csr_config：
        │
        ├────── dev/delay_mv/
        │       ├── binary_delay.py split_region_delay [region_delay_split.cu]：
        │       ├── binary_delay.py _binary_csrmv_cuda_kernel [binary_csrmv_hybrid_region_delay.cu]：
        │       ├── binary_delay.py _binary_csrmv_cusparse_kernel [region_delay_cusparse.cu]：
        │       └── delay_main.py prepare_delay_metadata / delay_tcsr：
        │
        ├────── _test_util.py / 各级 __init__.py
        │       ├── _test_util.py：
        │       └── __init__.py：
        │
        └────── test/ 与 dev/docs/
                ├── test/：
                └── dev/docs/：
```
