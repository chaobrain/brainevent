# 分段延迟 cuSPARSE 对照算子

为[分段延迟 hybrid 算子](hybrid-region-delay.md)增加一条原生 cuSPARSE 对照路线。CUDA 算子保持不变，继续使用 `delay_count` 与 `delay_off`，不修改 `indptr`。cuSPARSE 路线复用同一个 `RegionDelay` 对象，写入语义与 CUDA 算子一致。

文件均位于 `brainevent/_tcsr/dev/with_delay/operator/`，使用说明见该目录的 `README.md`：

| 文件 | 内容 |
| --- | --- |
| `region_delay_split.cu` | 算子 1：拆分 kernel |
| `region_delay_cusparse.cu` | 算子 2：cuSPARSE SpMV 与 RingAdd |
| `region_delay.py` | 在原有封装中追加两个算子的 JAX FFI 封装、结构与 workspace |
| `region_delay_test.py` | 在原有测试中追加正确性测试 |

两个 `.cu` 的核心 `Launch` 只接收裸指针，后续原生 CUDA benchmark 可直接 include；`// @BE` 包装仅供 pytest 使用。

cuSPARSE 调用方式参考 [原生 benchmark](../../brainevent/_tcsr/dev/with_delay/benchmark/benchmark_binary_csr_vecmat_cusparse_native.cu)：`cusparseCreateCsr` 加 `cusparseSpMV(CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_SPMV_CSR_ALG1)`。本轮不接入 benchmark。

1. **符号与时间语义。**

   `M` 为源行数，`N` 为目标数，`K = len(delay_steps)`，`L = buffer_slots`，`nnz = indices.size`。环形缓冲 `B` 为 `[L, N]`，按时间优先存放。每步先完成所有投射，再由现有 `consume_region_delay` 读出并清零 `B[head]`，最后推进 `head`。第 k 段写入的元素偏移与 CUDA `Extract` 相同：

   ```text
   off = head * N + delay_off[k];  if (off >= L * N) off -= L * N
   ```

   两条路线都只累加、不清零，零延迟写入当前 `head` 槽。

2. **算子 1：拆分子矩阵 `region_delay_split_f32`。**

   Python 入口为 `prepare_region_delay_cusparse(structure: RegionDelay) -> RegionDelayCusparse`，在初始化时调用一次。

   | 方向 | 名称 | 形状 | dtype | 说明 |
   | --- | --- | --- | --- | --- |
   | 输入 | `weights` | `[nnz]` 或 `[1]` | f32 | 与 CUDA 算子相同 |
   | 输入 | `indices` | `[nnz]` | i32 / i64 | 与 CUDA 算子相同 |
   | 输入 | `indptr` | `[M+1]` | i32 / i64 | 与 CUDA 算子相同 |
   | 输入 | `delay_count` | `[M, K]` | 同 indptr | 与 CUDA 算子相同 |
   | 输出 | `sub_indptr` | `[K, M+1]` | SubT | 第 k 个子矩阵的行指针，从 0 开始 |
   | 输出 | `sub_base` | `[K+1]` | SubT | 第 k 个子矩阵在打包数组中的起点，`sub_base[K] = nnz` |
   | 输出 | `sub_indices` | `[nnz]` | SubT | 按 k 优先打包 |
   | 输出 | `sub_values` | `[nnz]` | f32 | 异质权重搬运；同质权重填充为 `w` |

   SubT 由总 nnz 静态决定：`nnz <= INT32_MAX` 时为 i32，否则为 i64。cuSPARSE SpMV 要求行指针与列索引同 dtype，int64 输入的列号也按此规则转换；列号小于 `N <= INT32_MAX`，转换不会溢出。所有输出形状只依赖 `M, K, nnz`。

   kernel 分三步，均在同一 stream 上执行：

   ```text
   ColumnScan  每个 k 一个 block，用 cub::BlockScan 分块扫描 delay_count[:, k]，
               写 sub_indptr[k, 0..M]
   BaseScan    单线程对 sub_indptr[k, M] 做前缀和，写 sub_base[0..K]
   Gather      每行一个线程：edge = indptr[r]，按 k 顺序
                 dst = sub_base[k] + sub_indptr[k, r]
                 拷贝 delay_count[r, k] 条 indices / weights 到 dst
                 edge += delay_count[r, k]
   ```

   不需要额外临时空间。初始化后由 host 读取一次 `sub_base`，得到静态元组 `base_k` 与 `nnz_k = base_{k+1} - base_k`。`cusparseCreateCsr` 拒绝 `nnz > rows * cols`，因此任一 `nnz_k > M*N`（只会由大量重复连接造成）时在此处抛出 `ValueError`；低于该上限的重复连接由 cuSPARSE 正确求和。CSR 数组只要求按自身 dtype 对齐，打包偏移 `base_k` 与行偏移 `k*(M+1)` 满足此要求。

   `RegionDelayCusparse` 注册为 pytree：

   | 字段 | 类别 | 内容 |
   | --- | --- | --- |
   | `sub_indptr, sub_indices, sub_values` | data | 算子 1 输出 |
   | `delay_off` | data | 直接引用 `structure.delay_off` |
   | `shape, buffer_slots` | meta | 与 `RegionDelay` 相同 |
   | `sub_base, sub_nnz` | meta | 长度为 K 的静态整数元组 |

3. **算子 2：逐段调用子矩阵并叠加 `region_delay_cusparse_spmv_f32`。**

   Python 入口为 `project_region_delay_cusparse(op, events, buffer, head, scratch) -> (buffer, scratch)`。先计算 `x = (events != 0).astype(float32)`，与 CUDA 算子一致：负值也视为激活，事件值不缩放权重。随后对每个 `nnz_k > 0` 的 k 调用一次 FFI，`buffer` 与 `scratch` 通过输入输出别名在调用间传递，顺序由数据依赖保证。

   | 方向 | 名称 | 形状 | dtype | 说明 |
   | --- | --- | --- | --- | --- |
   | 输入 | `sub_indptr` | `[K, M+1]` | SubT | 算子 1 输出 |
   | 输入 | `sub_indices` | `[nnz]` | SubT | 算子 1 输出 |
   | 输入 | `sub_values` | `[nnz]` | f32 | 算子 1 输出 |
   | 输入 | `x` | `[M]` | f32 | 由 `events` 转换 |
   | 输入 | `delay_off` | `[K]` | i32 / i64 | 与 CUDA 算子相同 |
   | 输入 | `head` | `[1]` | i32 | 与 CUDA 算子相同 |
   | 输入/输出 | `buffer` | `[L, N]` | f32 | 与 CUDA 算子相同，别名 |
   | 输入/输出 | `scratch` | `[N]` | f32 | workspace，别名 |
   | 静态属性 | `segment, base, nnz` | int64 | — | 第 k 段的打包偏移与长度 |

   每次调用：

   ```text
   1. cusparseSpMV(TRANSPOSE, alpha=1, A_k, x, beta=0, scratch)
      A_k: rows=M, cols=N, nnz=nnz_k,
           rowOffsets = sub_indptr + k*(M+1),
           colInd     = sub_indices + base_k,
           values     = sub_values  + base_k
   2. RingAdd：每个线程一列 j
      off = head*N + delay_off[k]; if (off >= L*N) off -= L*N
      buffer[off + j] += scratch[j]
   ```

   不让 cuSPARSE 以 `beta=1` 直接写环形缓冲行，原因有二：一是 `head` 在 device 上，cuSPARSE 的输出指针却要在 host 上确定，直接写需要每步做一次 D2H 拷贝和流同步；二是 `off` 一般不满足 16 字节对齐，而 cuSPARSE 12.4 存在输出向量未按 16 字节对齐时的越界访问问题。`scratch` 由 XLA 分配，满足对齐要求。代价是每段多一次 N 长度的读写。

   cuSPARSE handle 按设备缓存为静态对象，每次调用用 `cusparseSetStream` 绑定 XLA stream。描述符每次调用时创建、销毁；SpMV 外部 buffer 用 `cudaMallocAsync / cudaFreeAsync` 在同一 stream 上申请和释放。模块以 `extra_ldflags=["-lcusparse"]`、`allow_cuda_graph=False` 加载。

   `head` 越界时 RingAdd 不写入。cuSPARSE 路线没有 `status`，因此不报告此类错误，调用方负责保证 `head` 合法。

4. **不使用的 CUDA 算子数据。**

   `row_task_count`、`task_begin / task_end / task_out_off`、`status` 与调度配置 `RegionDelayConfig` 属于 hybrid 任务调度，cuSPARSE 路线不使用。

5. **边界情况。**

   | 情况 | 处理 |
   | --- | --- |
   | 空图（`nnz=0`，`delay_steps=(0,)`） | 算子 1 输出零长度打包数组；算子 2 不调用 FFI，`buffer` 原样返回 |
   | 某段 `nnz_k=0` | 跳过该段 FFI 调用 |
   | `N=0` | 算子 2 不调用 FFI |
   | `M=0` | 同空图 |
   | `buffer_slots > max(delay_steps) + 1` | 回绕公式按 `L*N` 处理，与 CUDA 相同 |
   | int64 `indptr` / `indices`，`nnz <= INT32_MAX` | SubT 取 i32 |
   | `L*N > INT32_MAX` | `delay_off` 为 i64，RingAdd 按 `delay_off` dtype 分派，偏移用 int64 计算 |

   cuSPARSE `CSR_ALG1` 转置 SpMV 使用原子累加，结果与 CUDA 算子不保证逐位一致，测试按 `rtol/atol` 比较。

6. **测试。**

   - 算子 1：K 个子矩阵按行拼回后等于原 CSR 的 `(indices, weights)`；`sub_base`、`sub_indptr` 与 numpy 参考一致；覆盖同质/异质权重、空段、int64 输入转 i32、空图。
   - 算子 2 单步：与 numpy 参考及 `project_region_delay` 对比；覆盖 bool/float 事件（含负值）、`head + d` 回绕、零延迟、`buffer_slots > max_delay + 1`、`buffer` 原有内容被保留。
   - 多步：在 `lax.scan` 中执行 project 加 `consume_region_delay`，与 CUDA 路线逐步对比 `current`。
   - 签名检查：错误的 `events / buffer / head / scratch` 形状或 dtype 抛出 `ValueError / TypeError`。
