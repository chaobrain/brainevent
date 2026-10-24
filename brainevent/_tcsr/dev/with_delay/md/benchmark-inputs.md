# 延迟投射 benchmark 的输入与 step 语义

此文保留前一轮 benchmark 重组及其验证记录；Python 测试封装的最新接口
与精简后的验证结果见 [算子说明](operator.md) 和 [精简规格](python-test-wrappers.md)。

## backend 与 K 配置调整（2026-10-22）

按最新要求，结果和汇总的算子字段统一使用 `backend`，取值为
`cusparse`、`hybrid`。正常与失败记录均携带此字段，汇总按 backend
分别统计；删除结果中的 `runner` 字段。

`CASE_LIST` 以 `K_LIST` 指定 K 的候选值，替代手写延迟值元组，默认
`K_LIST: [3]`，保留用户当前的规模、稀疏度、发放率和 Gini 配置。
用户已确认 K 表示延迟段数：每条连接独立均匀选择 `0 <= delay < K`
的整数，分布等价于在 `[0, K)` 上均匀随机取值再向下取整。实现直接
使用整数采样，避免为每条连接额外创建浮点延迟数组。

benchmark 的输入生成和单 case 接口改接收 `k`，内部保留完整的
`tuple(range(k))`，即使部分延迟段未分配到连接也不缩小段数。K 必须为
正 int32 整数；ring 槽数为 K，K=1 时所有连接延迟为零。结果记录 `k`，
替代冗余的 `delay_steps` 字符串，保留原生返回的 `segment_count`。
汇总按 backend、K 和其他 case 配置分组，T 仍独立配置。

内存预算先使用标量 K 估计，再分配段数组和事件矩阵，避免超大 K 在
检查预算之前分配延迟列表。CUDA 输入、split 时机和投射计时范围沿用
现有实现。修改局限于 benchmark 配置/输入生成/结果字段及对应测试。

验收覆盖 CSV 字段、两种算子的独立分组、不同 K 的 case 展开、随机
延迟边界、空段和无连接情况，并用小规模真实 GPU case 验证原生衔接。

此前手写延迟集合容易混淆“段数”和“最大延迟”；今后 benchmark 统一
从 K 推导完整连续延迟范围，修改 K 不改变独立随机流生成的事件序列。

本轮验证结果：

- benchmark 配置/输入/CSV 与原生 GPU 回归共 62 项通过；入口文件行覆盖率
  99.28%，benchmark 目录整体 93.95%。本次复用了已有原生编译缓存。
- JAX 0.8.0 环境下 37 项配置与输入测试通过；两个修改的 Python 文件
  通过 `mypy --follow-imports=silent`。
- GPU 1 实测 N=256、T=13、`K_LIST=[1,3,8]`，hybrid 与 cuSPARSE
  共六条结果全部成功，明细和汇总均按 backend、K 区分。cuSPARSE 每轮
  分别执行 13、39、104 次 SpMV，符合 `T * 非空段数`。
- 边界覆盖 K=1、无连接、连接数小于 K、非法 K、超大 K 提前预算拒绝，
  以及改变 K 后事件随机流保持不变。后续可增加 K=16/64、包含大量空段
  的 case，继续核对输出和实际非空段调用数。

## 范围与已确认规则

按 2026-10-22 的讨论，清理 `brainevent/_tcsr/dev/with_delay/benchmark/`
中无延迟 benchmark 的旧 backend 注册、分派、编译与调用衔接，围绕
`operator/` 的 hybrid、split 和 cuSPARSE 延迟投射重新组织输入。

用户已明确取消 split 持久化。每个 case 在内存中初始化自己的带延迟
CSR 和完整事件序列；初始化完毕后使用对应算子的输入执行 benchmark。
不添加 split 的保存、加载、磁盘缓存或恢复接口。

用户已确认：每条连接从 case 配置的延迟值集合中独立、均匀随机选择延迟；
不使用脑区对延迟表。只统计投射：hybrid 一次完整调用；cuSPARSE 全部
非空延迟段的 SpMV + RingAdd。消费、清零和 head 推进执行但不计时。

## case 初始化

设源数量为 M，目标数量为 N，执行步数为 T，延迟段数为 K，环形缓冲
槽数为 L。方阵 case 中 M=N。事件列数 T 与延迟段数 K 独立配置，不共用
变量名，也不要求数值相等。

每个 case 的 seed 固定后：

1. 生成一份 CSR 权重、列索引、行指针以及对应延迟。
2. 按行内 `(delay, column)` 排序，权重随连接移动；生成
   `delay_count[M,K]`、`delay_steps[K]` 和 `delay_off[K]`。
3. hybrid 使用此结构及其任务元数据；cuSPARSE 对同一结构执行一次
   split，得到 K 个按延迟划分的 CSR 子矩阵。
4. 一次生成完整事件矩阵 `events[M,T]`，每列对应一个 step。使用
   float32 的 0/1 活动值，使两个算子可以共享完全相同的事件存储。
5. 分配环形缓冲、hybrid 任务数组、cuSPARSE scratch 与运行资源。

上述初始化不计入 benchmark 的运算时间。同一输入的 warmup 和 timed
run 复用预处理结果和事件序列；每轮开始时在计时外重置 buffer、head
及必要状态。case 结束后释放其资源。

事件矩阵按列连续存储，即逻辑形状 `[M,T]`、元素地址
`events[row,step] = events_base[step*M + row]`。第 t 步使用指针
`events_base + t*M`；时间循环内不重新随机生成事件，不进行事件列的
gather、复制、转置或 dtype 转换。

## 算子输入与输出

### 共同的静态输入

| 名称 | 形状 | dtype / 含义 |
| --- | --- | --- |
| weights | `[nnz]` 或 `[1]` | float32，异质或同质权重 |
| indices | `[nnz]` | int32 / int64，目标列号 |
| indptr | `[M+1]` | int32 / int64，CSR 行边界 |
| delay_count | `[M,K]` | 与 indptr 同型，每行每个延迟段的连接数 |
| delay_steps | `[K]` | 升序的非负整数延迟步数 |
| delay_off | `[K]` | int32 / int64，`delay_steps[k]*N` |

每行的计数和必须等于该行连接数；行内边的顺序必须与这些段计数一致。
`L > max(delay_steps)`，K 不等于 L 时仍使用实际延迟确定输出槽。

### hybrid

静态输入为上述排序 CSR、延迟元数据，以及 `row_task_count[M]`、
任务容量和调度配置。动态输入为当前事件列、`head[1]`、
`buffer[L,N]` 与可复用任务 workspace。

输出为更新的 buffer、任务数组和 `status[2]`。buffer 采用累加语义；
status 的错误不能在后续步骤重置后被忽略。原生 Launch 使用裸指针、
显式维度与 stream 参数；张量解包由 LaunchTensor 负责。

### split

输入为 weights、indices、indptr 和 delay_count。只在 case 初始化时执行。

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| sub_indptr | `[K,M+1]` | 每段 CSR 的行指针，各自从零开始 |
| sub_base | `[K+1]` | 打包数组中的边界，末项等于总 nnz |
| sub_indices | `[nnz]` | 按段打包的目标列号 |
| sub_values | `[nnz]` | float32，同质权重展开为逐边权重 |

cuSPARSE 的行指针和列索引使用统一的 SubT，沿用现有按总 nnz 选择
int32 / int64 的规则。`sub_nnz[k] = sub_base[k+1]-sub_base[k]`。
这些结果只在内存中保留并直接交给 cuSPARSE benchmark。

### cuSPARSE

静态输入为 split 输出和 delay_off；动态输入为当前 float32 事件列、
head、buffer 与 `scratch[N]`。第 k 段使用指针：

```text
row_ptr = sub_indptr + k*(M+1)
col_ind = sub_indices + sub_base[k]
values  = sub_values  + sub_base[k]
```

执行 `scratch = A_k^T @ events[:,t]` 后，通过 RingAdd 累加到
`buffer[(head+delay_steps[k]) % L, :]`。所有段共享同一事件列和同一
step 的 head。空段可以跳过实际 SpMV，但不改变 step 计数。

输出为更新的 buffer；scratch 只是复用的临时空间，不是该 step 的完整
结果。cuSPARSE 的描述符和外部 workspace 应在计时前准备并复用。

## 时间循环和统计

```text
hybrid:
  for t in [0,T):
    x = events[:,t]
    project_hybrid(x)                 # 一次完整 hybrid 投射
    consume_current_and_advance_head  # 每个 step 仅执行一次

cuSPARSE 初始化（计时外）:
  split(original_csr) -> sub_indptr, sub_base, sub_indices, sub_values
  StepPlan(split_outputs)             # 算子内部准备全部子矩阵描述符

cuSPARSE benchmark:
  for t in [0,T):
    x = events[:,t]
    plan.Project(x)                   # 一次调用完成整个 step
    consume_current_and_advance_head  # 全部段结束后才执行

region_delay_cusparse.cu 内部的 StepPlan.Project:
  for k in [0,K):
    if sub_base[k+1] > sub_base[k]:
      SpMV(A_k, x, scratch)
      RingAdd(scratch, delay_off[k], head, buffer)
```

K 段循环只属于 `operator/region_delay_cusparse.cu`，不允许由 Python
或原生 benchmark 控制。Python 先调用 split，再将四份完整输出交给
cuSPARSE；每个非零输出维度的 step 只发出一次 FFI 调用。删除单段
`segment/base/nnz` 入口和 Python 的逐段静态元数据。

`StepPlan` 接收四份 split 设备数组、维度、scratch 和 stream，在构造时
读取 `sub_base[K+1]`、校验边界并创建非空子矩阵描述符；其 `Project`
只切换事件向量并在内部遍历全部子矩阵。原生 benchmark 在计时前创建
一个 plan，跨 warmup、timed run 和全部 steps 复用。JAX FFI 是正确性
封装，每次调用创建临时 plan，因此会读取这份小型前缀数组；不用于性能
计时，也不启用 CUDA graph 捕获。两条入口调用同一个整步实现。

一次 hybrid 投射计为一个 step；cuSPARSE 遍历全部 K 段也只计为一个
step。执行 T 步时，两条路线都报告 T steps，cuSPARSE 不报告 T*K steps。
每步平均耗时使用 `total_run_time / T`；多轮平均使用
`total_timed_time / (runs*T)`。结果明确记录 `projection_only` 计时范围。

两条路线从相同的初始 buffer/head 出发，逐步消费同一时刻的输出；
零延迟在当前 step 消费，未来槽的贡献保留到对应时刻。消费后推进 head，
不能在每个 cuSPARSE 子矩阵之间推进。消费及推进统一放在每步计时终点之后、下一步计时起点之前。

## 验收

- 旧 backend 编号、分派和缺失 kernel 头文件的编译依赖全部清理。
- 使用列间明显不同的事件验证取列顺序；T 与 K 取不同数值，验证 step
  总数和平均耗时分母。
- 比较 hybrid、cuSPARSE 与独立逐边绝对到达时间参考；比较每步消费值
  以及尚未消费的 buffer 内容。
- 覆盖 K=1、空段、空图、无事件、全活跃、同质/异质权重、重复连接、
  非连续延迟、零延迟、L=1、环绕和 int32/int64 索引。
- 覆盖短行/长行阈值和任务切块边界；检查错误状态不被后续 step 掩盖。
- 多轮测试确认初始化仅发生一次、run 间状态复位、事件和 split 被复用。
- 新增回归约束：多段 JAX 投射只有一个 FFI 调用；直接调用 CUDA 整步
  接口一次就完成全部非空段，覆盖空段、不同事件列、环绕及已有 buffer。
- 同级后缀测试文件；新增 Python 路径覆盖率超过 90%，mypy 无错误，
  JAX 支持下限保持 0.8.0。真实 GPU 测试与 CPU 参考校验分别记录结果。

所有修改发生在 worktree 分支 `codex/delay-benchmark-inputs`，目录为
`/home/luruheng/BrainEvent/brainevent-mm进一步优化/brainevent-delay-benchmark`。
源目录原受 `.gitignore` 的 `dev/` 规则忽略；已把必要源码快照带入该
worktree，并为源码设置 Git intent-to-add，使改动可审阅。

## 实现组织

- `operator/delay_inputs.py`：NumPy 的逐边延迟输入校验、排序与任务计数，
  不依赖 JAX；所有静态和动态输入在初始化时完成，事件按列连续存放。
- `operator/*.cu`：保留三个算子的实现与 FFI 包装，hybrid 增加裸指针
  Launch；cuSPARSE 的 StepPlan 管理全部段描述符和工作空间，内部
  循环子矩阵，一次 Project 完成整个 step。
- `benchmark/native_runner.py`：编译、资源生命周期与两个明确的原生调用
  入口，GPU 初始化一次，split 在 cuSPARSE 首次运行前准备一次。
- `benchmark/benchmark_binary_csr_vecmat_native.cu`：输入上传、可复用资源、
  原生 step 循环和 CUDA event 计时；每步 start/stop 只包围投射。
- `benchmark/benchmark_binary_csr_vecmat.py`：case 的 seed、随机逐边延迟、
  完整事件序列、两种算子的调用、CSV 汇总；删除旧 backend 注册和调度。

主 benchmark 为 float32。CSR 列索引、行指针及环形偏移分别支持
int32/int64；零长度数组和空段仍有明确形状。小规模校验可取回逐步消费
输出和最终环形缓冲，大规模计时默认不保留整个输出轨迹。

## 验证结果（2026-10-22）

使用物理 GPU 1（NVIDIA RTX PRO 6000 Blackwell Server Edition）、CUDA
13.0、Python 3.12.13 完成验证：

- JAX 0.11.0：算子、原生接口、输入和 benchmark 控制器测试共
  **211 项全部通过，无跳过**；异常类型兼容调整后补跑 4 项边界测试通过。
- Python 覆盖率：所测 operator/benchmark 目录总计 **96.66%**；
  `region_delay.py` 为 99.16%，`delay_inputs.py` 为 100%，
  `native_runner.py` 为 95.88%（本次命中已编译库缓存），控制器为 99.26%。
  覆盖率按目录采集，避免包名探测提前导入 NumPy 的问题。
- JAX 0.8.0：在隔离环境中使用对应版本的 jax、jaxlib、CUDA plugin，
  **112 项算子测试全部通过，无跳过**。隔离解决了旧 namespace
  plugin 被共享环境中的 0.11.0 常规包覆盖的问题，没有修改共享环境。
- 9 个相关 Python 实现和测试文件通过 mypy，无错误。
- 命令行 smoke：N=256、nnz=9728、T=13、延迟值 (0,2,5)、K=3，
  两条路线均成功输出 CSV。cuSPARSE 每轮 39 次 SpMV，duration_steps=13。
  本次整步接口 smoke 输出位于 `/tmp/brainevent-delay-step.p9_7x52f/case_001/`。
- `git diff --check` 通过。没有运行完整的大规模 CASE_LIST；smoke 耗时
  只用于验证计时与输出流程，不作为正式性能结论。

发现并修正的契约问题均先复现再修复：

1. hybrid 的原生裸指针调用原先编译失败，与 README 不一致。新增同级
   CUDA 编译/运行测试后，将张量解包移到 LaunchTensor，原生 Launch
   接收显式指针和维度。以后以真实编译约束接口说明。
2. 空行数的事件矩阵可以具有超 int32 的列数而不占用存储，初版输入层
   未拒绝它。先添加 `(0,2**31)` 的失败测试，再把步数范围校验提前到
   输入构建阶段；以后对空数组同时检查逻辑维度和存储大小。
3. 初版 cuSPARSE 把 K 循环放在 Python 和原生 benchmark 中，算子
   只处理一个子矩阵，职责不符合用户要求。先复现一 step 出现两次
   FFI、原生整步接口不存在，再将完整 split 输入和 K 循环统一到
   StepPlan。以后同时验证调用次数和数值输出，防止仅数值正确却把
   控制循环放错层。直接 CUDA 测试还覆盖两种 SubT、两种 RingT、空段、
   环绕、已有 ring 内容和不同事件列间的描述符复用。
4. 新增非法 split 测试初次假设所有 JAX 版本都用 RuntimeError 包装
   FFI 错误，0.8 实际使用 ValueError。测试兼容两种具体异常，并严格
   匹配算子错误信息；以后验证跨版本的错误语义，而不假定包装类一致。

下次性能测量建议先选择目标规模的 CASE_LIST 子集，在独占 GPU 上运行，
再扩大到完整网格；继续保持逐步输出的正确性校验与正式计时分开。
