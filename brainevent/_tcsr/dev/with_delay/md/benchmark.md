# 带延迟的 hybrid / cuSPARSE benchmark

每个 case 生成一份 CSR、逐连接独立随机延迟和一份完整事件矩阵。hybrid
直接使用延迟排序后的 CSR；cuSPARSE 在初始化时对同一结构 split 一次。
所有输入驻留内存，不保存或加载 split 文件。

## 一列事件等于一个 step

以 N 个源、T 个时间步和 K 个延迟类别为例：

- `events[N,T]` 是逻辑形状，**按列连续存储**；原生指针为
  `events + t*N`。T 与 K 独立，默认 T=100、K=3，延迟值为 `0、1、2`。
- 事件只初始化一次，值为 float32 的 0/1。两条路线共用同一份 GPU
  事件存储；循环中不生成、复制或转换事件列。
- hybrid：取第 t 列，调用一次包含 Extract + BlockScatter 的投射，计为
  一个 step。
- cuSPARSE：取同一列，调用 `region_delay_cusparse.cu` 的 `StepPlan.Project`
  一次；算子内部循环 K 个子矩阵，依次执行 SpMV + RingAdd，整组计为
  一个 step。空段跳过实际 SpMV；处理全部段之前不推进 head。

```text
case 初始化（计时外）
  CSR + 每条边独立选择整数 delay ∈ [0, K)
      → 按行内 (delay, column) 排序 → hybrid 静态输入
                                  → split → cuSPARSE 静态输入
  events[N,T]、buffer、任务队列、scratch、cuSPARSE 描述符/工作空间

每次 warmup / timed run
  重置 buffer、head 和错误状态（计时外）
  for t in range(T):
    start[t]
      hybrid Launch 一次 / cuSPARSE StepPlan.Project 一次（内部完成 K 段）
    stop[t]
    消费并清零当前槽，head 前进一次（计时外）
  同步一次，汇总 T 个 start/stop 区间
```

每轮耗时为 T 段投射时间之和，`timed_mean_step_ms = timed_mean_ms / T`。
cuSPARSE 的分母也是 T。数据生成、排序、split、描述符创建、上传、分配、
ring 消费、清零和 head 推进均不包含在计时中；hybrid Launch 内部每步
重置 status 则属于投射计时区间。CUDA event 放在同一 stream，循环中
不逐步同步。每轮从相同初始状态和事件序列开始。

hybrid 不调用 split。cuSPARSE 在首次运行前调用 `region_delay_split.cu`，
把四份完整设备输出交给 `region_delay_cusparse::StepPlan`；随后所有
warmup 和 timed run 复用此 plan。Python 和原生 benchmark 均不包含
子矩阵循环；K 循环只存在于 `operator/region_delay_cusparse.cu`。

## 入口和文件

| 文件 | 职责 |
| --- | --- |
| `benchmark_binary_csr_vecmat.py` | case 网格、随机输入、两个明确的调用入口、CSV 与汇总 |
| `native_runner.py` | 编译及 `NativeCase` 生命周期，`run_hybrid` / `run_cusparse` |
| `benchmark_binary_csr_vecmat_native.cu` | 输入上传、split/资源复用、原生时间循环与计时 |
| `delay_benchmark_api.h` | 原生输入与统计结构，两个投射 benchmark 函数 |
| `gini_row_lengths.py` | 沿用的行长分布生成器 |
| `../operator/region_delay.py` | 逐边延迟的 NumPy 输入准备；同时提供正确性测试用的三个 CUDA 调用封装 |

旧的 `BACKEND_CODES`、`BACKEND_LIST`、无延迟 cuSPARSE runner、
TPR/WPR/2end 接入、缺失 kernel 头文件依赖和旧分块辅助文件已移除。
当前只有带延迟 hybrid 和 cuSPARSE 两条明确的调用路线。

case 的 `K_LIST` 配置延迟段数列表，默认 `[3]`，可改为 `[1, 3, 8]`
分别测试三个 K。K 必须为正 int32 整数；每条连接独立均匀生成
`0～K−1` 的整数延迟，等价于在 `[0, K)` 上均匀随机取值再向下取整。
代码直接使用 `delay_rng.integers(0, k, size=nnz, dtype=np.int32)`。

完整延迟集合为 `tuple(range(k))`，即使某段没有连接也保留该段，ring
槽数为 K。K=1 时所有连接延迟为零。同一源或同一目标的不同连接可以
有不同延迟。列索引、权重、延迟和事件采用分开的随机流；两个算子使用
同一 case 输入，修改 K 不改变事件序列。`GINI_LIST` 控制行长分布；
当前生成器使用 float32 异质权重和有放回的列号采样。

预处理在 CPU 上完成，算子和 benchmark 时间循环在原生 CUDA 上执行。
大规模 case 需容纳完整 CSR、段计数、事件矩阵、ring、任务队列和 split；
初始化期间还有 host 排序副本。设备预算检查估算显式数组，cuSPARSE 的
内部工作空间由库查询后分配；分配失败会写入对应 case 的 error 字段。

## 运行

在仓库根目录，先验证小 case：

```bash
python -m brainevent._tcsr.dev.with_delay.benchmark.benchmark_binary_csr_vecmat --smoke --gpu 1 --steps 13
```

完整配置在脚本的 `CASE_LIST`；不传 `--smoke` 时运行完整网格。
`--output` 可指定结果目录。输出保留 `case_001/case_1.csv`、`summary.csv`
的组织方式；`backend` 使用 `cusparse`、`hybrid` 区分两种算子，`timing_scope` 固定为
`projection_only`。`k` 记录配置的段数，替代原 `delay_steps` 字符串；
`segment_count` 和 `buffer_slots` 均为 K。汇总按 backend、K 和其他输入
配置归组，不把随 seed 波动的 actual_gini 作为重复实验的分组键。

需要验证逐步输出时，直接使用：

```python
from brainevent._tcsr.dev.with_delay.benchmark.benchmark_binary_csr_vecmat import generate_case_inputs
from brainevent._tcsr.dev.with_delay.benchmark.native_runner import NativeCase

inputs, _ = generate_case_inputs(
    n=256, degree=38, event_density=0.1, duration_steps=13,
    k=3, seed=101,
)
with NativeCase(inputs) as case:
    hybrid = case.run_hybrid(warmup=2, runs=5, capture=True)
    cusparse = case.run_cusparse(warmup=2, runs=5, capture=True)
# consumed: [T,N]；ring: [L,N]，包含 T 步结束时尚未到达的贡献。
```

`capture=False` 为默认值，不保存整个输出轨迹。计时以外每步仍会消费和
清零；hybrid 每步错误会被累积检查，不会被后续无事件的 step 掩盖。

## 测试

```bash
python -m pytest brainevent/_tcsr/dev/with_delay/operator/test/delay_inputs_test.py brainevent/_tcsr/dev/with_delay/benchmark/test/native_runner_test.py brainevent/_tcsr/dev/with_delay/benchmark/test/benchmark_binary_csr_vecmat_test.py -q
```

原生测试将两条路线与原始逐边延迟的绝对到达时间参考比较，覆盖 T≠K、
空段/空图、单槽、零延迟、环绕、重复边、同质/异质权重、i32/i64 索引及
偏移、长短行边界、任务切块边界、多轮复用和早期错误保留。
