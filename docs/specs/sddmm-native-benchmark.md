# SDDMM 原生基准实现计划

日期：2026-10-19。工作分支：`mm-optimizing`。

实施授权：用户已通过“组织实现”确认执行本计划。实现和验收记录追加于本文。

现行修订：用户随后要求对齐 SDDMV 的清零口径，详见 [计时对齐规格](sddmm-sddmv-timing-alignment.md)。下文现行计时约定已更新，初版验收结果保留为旧口径历史记录。

在 `brainevent/_tcsr/dev/sddmm/benchmark` 建立原生 CUDA 基准，沿用 tilemm 的逐用例子进程、逐步 CUDA Event、整轮同步和逐 seed CSV 汇总方式。event 与 ct 均来自独立循环池，先复制到独立分配的输入缓冲，再调用 SDDMM。

本文是实施规格，完成状态见末尾的验收清单。它替代 [旧 SDDMM 规格](sddmm-cusparse-benchmark.md) 中的 JAX 调用计时、输入生成、运行网格和结果组织约定。

## 已确认的内存约定

“从池抽取”必须形成独立的设备数据副本。循环池和输入缓冲各自拥有显存分配，二者地址范围不得重叠。禁止用池内地址偏移、切片、reshape 或其他共享底层存储的 view 代替复制。

输入缓冲在用例初始化时分配一次，每步 gather 将新内容写入同一缓冲；不要求每步重新 `cudaMalloc`，也不保存所有时间步的完整矩阵。当前流中的上一步计算完成后，下一步才能覆盖该缓冲。

后端接口可以用 `BE::Tensor` 包装输入缓冲。该对象仅描述已独立分配的输入缓冲，不拥有内存，也不得指向循环池；真正的显存所有权归 native runner 的资源管理对象。

现有 tilemm 的依据位于
[`benchmark_binary_csr_matmat_native.cu`](../../brainevent/_tcsr/dev/new_tile_operator/baseline/benchmark_binary_csr_matmat_native.cu)：

- `Resources::allocate` 调用 `cudaMalloc`，并登记资源以便释放。
- `ALLOC(pool, pool_elements)` 与 `ALLOC(input_i8, window_elements)` 或 `ALLOC(input_typed, ...)` 是不同的显存分配。
- `gather_circular_input` 执行 `staging[destination] = input_pool[source]`，同时完成复制和布局组装。
- tilemm 当前输入采用 BN；本次 SDDMM 的 event 采用 NB，ct 采用 BN。

以上是现有 tilemm 的源码事实；下面各节规定待新增的 SDDMM 实现。

## 计算与布局

设 CSR 形状为 `R × C`，内积维为 `K=batch`，槽位 `s` 对应 `(i,j)`。正式网格使用 `R=C=N`；辅助函数和小规模测试保留矩形形状，以检验行列地址是否混淆。

| 数组 | 形状及类型 | 元素地址或含义 |
| --- | --- | --- |
| event 活动池 | `R × P_event` 个 uint8 | 固定 0/1 活动标志 |
| ct 循环池 | `C × P_ct` 个 ValueT | 固定稠密浮点数 |
| event 输入 | `[R,K]`，行主序 | `event[i*K+b]` |
| ct 输入 | `[K,C]`，行主序 | `ct[b*C+j]` |
| dweight | `[nnz]`，ValueT | 与 CSR 槽序一致 |

binary 计算 `dweight[s] = sum_b (event[i,b] > 0) * ct[b,j]`，bool 直接使用真值。float 计算 `sum_b event[i,b] * ct[b,j]`，保留有符号活动值。

正式 f32 对照包括 `be_binary_bool`、`be_binary_f32`、`be_float`、`cusparse_binary` 和 `cusparse_float`。实现和数据类型分别记录，避免将 binary-f32 与保留数值的 float 混为一类。小规模验收覆盖现有 BE 类型矩阵：binary 的 f32 输出支持 bool/f32 事件，f64 输出支持 bool/int8/f32/f64 事件；float 支持匹配的 f32/f64 输入输出。cuSPARSE 对照使用匹配的浮点精度。

## 循环池和 gather

默认 `P_event=P_ct=550`，分别记录在配置和结果中，允许配置为其他正整数。池长度是元素总数，按一维循环窗口访问，不将其解释成每个神经元独立移动的 550 列历史。

### 初始化

1. 分别分配 event 活动池、ct 池、event 输入、ct 输入以及输出和 scratch；分配前检查整数乘法及字节数溢出。
2. 使用独立随机流生成两个池。活动概率为 `firing_rate_hz * dt_ms / 1000`，池内为 Bernoulli 0/1；ct 为稠密 `U[-2,2)`，不受 event 活动标志影响。
3. 随机生成以流种子和池元素下标为键，固定池内容不包含当前 step 或测量轮次。记录网络、活动、float 活动值与 ct 的随机流种子；不同后端使用一致配置。
4. 分配 cuSPARSE 所需的数值输入及工作区。描述符和 workspace 在计时前准备，正式循环中不新增分配。

### 每步地址公式

对第 `t` 步、样本 `b`：

```text
event_source = (t * R * K + i * K + b) mod (R * P_event)
ct_source    = (t * C * K + j * K + b) mod (C * P_ct)

event_step[i * K + b] <- convert(event_pool[event_source])
ct_step[b * C + j]    <- ct_pool[ct_source]
```

模乘使用无溢出的实现，不先计算可能溢出的 `t*R*K`。GPU kernel 实际读池、写目标缓冲；即使窗口未回绕，也执行复制，不切换成指针别名。

- binary-bool 写 bool，binary 数值类型写 0/1。
- float 使用同一活动池；活动位置的幅值按独立随机流及 `event_source` 确定为 `±U[0.5,1.5)`，零活动写 0。这一数值生成与类型转换属于 gather 准备阶段。
- cuSPARSE binary 使用对应 0/1 浮点输入，float 使用同一组有符号数值。独立 worker 之间要求逻辑输入逐元素一致，不要求跨进程共享显存地址。
- event 和 ct 各自启动 gather，输出地址分别为 NB 和 BN。不得在计时内补做转置。

### 周期和重放

单池的窗口起点周期为 `P/gcd(P,K)`。默认 P=550，K 为 4 到 512 的 2 的幂时，周期均为 275 步，表示 step 0 与 step 275 的完整输入窗口相同。

每轮从 step 0 开始，默认执行 25 步，预热和测量重放相同输入。大 batch 下窗口会重叠；结果记录池长度和周期，不宣称各步输入相互独立。池很小、窗口比池更大或单次窗口多次回绕时，也按上述公式定义。

## 结构与分块

正式结构使用 `N=int(scale*1000)`、`degree=max(1,round(N*density_mat))`、`nnz=N*degree`，要求 `degree<=N`。生成固定行长、行内唯一且有序的列号；BE 与 cuSPARSE 使用相同 CSR 槽序，不沿用 tilemm 有放回取模生成列号的方式。

采用可按行复现的无放回生成流程，分块仅改变准备范围，不能改变行内列集合。初始化可分块生成和排序，禁止构造完整 `N×N` 稠密矩阵。独立小规模参考检查唯一性、固定行长及种子可复现性。

TCSR 保持 int64 `indptr`、uint16 `local_targets` 和 int32 的行内 `tile_offsets`。SDDMM 不使用 CSR 权重，不为其分配无消费者的随机权重数组。

运行支持 direct 与按行 chunk。先完成 direct，再用可降低的分块阈值验证 chunk：

- event 当前行块使用全局行号定位循环池，但写入局部 `[chunk_rows,K]` 独立输入缓冲；窗口推进仍以完整 `R*K` 为步长。
- ct 每步 gather 一次，保留完整 `[K,C]`，同一步所有行块共享该独立输入缓冲，切换 chunk 不推进 ct 窗口。
- BE 与 cuSPARSE 的 chunk 对照使用相同行划分、局部行指针与全局列号。cuSPARSE 块内使用 int32 索引；所有全局数量及偏移计算使用足够宽的整数。
- 各块对应不相交的 dweight 槽位，均独立覆盖，cuSPARSE 始终 `beta=0`。禁止复制 MM 的后续块 `beta=1` 累加规则。
- 复用块输出缓冲；全量输出只用于能容纳的小规模校验。大规模分块校验在独立验证流程中保留所需样本，避免下一块覆盖后丢失待校验值。
- 临时结构生成、排序、元数据构造放在事件外；分块模式和块数写入结果，不与 direct 无条件混合汇总。

cuSPARSE 使用 `CUSPARSE_SDDMM_ALG_DEFAULT`，不移植 SpMM 的 ALG1/ALG2 注册。初版使用与 BE 一致的 event NB、ct BN 布局，不启用 `cusparseSDDMM_preprocess`。此布局和预处理选择显式记录在 manifest 中，避免将结果解释成所有 cuSPARSE 配置的性能上限。

## 计时和资源生命周期

每轮流程为：

```text
for step:
    gather ct 到独立 BN 缓冲
    for row_chunk:             # direct 只有一块
        准备块结构和 TCSR 元数据（需要时）
        gather event 到独立 NB 缓冲
        start[step, chunk]
        完整 SDDMM 调用
        stop[step, chunk]
轮末同步一次
读取各事件耗时并求和
```

所有操作使用同一 CUDA stream。下一块或下一步的复制排在前一次读取输入的计算之后。事件初始化时创建、轮间复用；禁止每步同步后再启动下一步。

输出在 Operator 构造阶段通过 cudaMemset 初始化一次，位于验证和计时前。计时包含 BE 算子内部的清零、mask、活动行压缩与计算；cuSPARSE 以 `alpha=1,beta=0` 直接调用，每步和每块均不额外清零，与 SDDMV cuSPARSE benchmark 对齐。

池生成、gather、类型/布局转换、静态结构准备、分配、描述符创建、校验、主机拷贝和日志不计时。输入 gather 对缓存的影响属于该测试条件，不能因其位于计时外就宣称没有缓存影响。

默认 25 步、2 轮预热、5 轮测量。每轮耗时为各步各块事件之和；正式轮次取均值、总体标准差、最小和最大值，单步均值为轮次均值除以步数。跨 seed 汇总使用样本标准差。默认仅落盘汇总统计，不保存完整逐步耗时序列。

全部资源由 runner 管理；部分初始化失败、CUDA/cuSPARSE 错误和正常返回均释放已分配的资源。有排队工作时先完成必要同步再释放，不允许 C++ 异常越过 C ABI。

## 显存与调度

显存估算至少包括：

- event 活动池 `R*P_event` 字节，ct 池 `C*P_ct*sizeof(ValueT)`。
- 独立 event staging `working_rows*K*sizeof(EventT)` 和 ct staging `K*C*sizeof(ValueT)`；cuSPARSE 如需要额外数值转换缓冲，单独计入。
- CSR/TCSR 结构、dweight、binary mask、float 专用 scratch、结构准备临时区及事件资源。
- cuSPARSE/CUB 查询到的 workspace；分别计算准备阶段和测量阶段峰值，不能只计算最终常驻数组。

例如 N=1,000,000、K=512、f32 时，仅 ct 池和 ct staging 合计 `4*N*(550+512)` 字节，约 3.96 GiB。

Python 保留 CASE_LIST、配置文件和 smoke 入口。正式参数沿用 MM 的 scale、density、Hz、seed=101、dt=0.1 ms，batch 去掉 2，类型默认 f32；每个后端有 8192 个候选参数组合，再按 SDDMM 实际内存需求过滤。沿用 96 GiB 的可配置估算上限并在 worker 检查设备可用显存；估算通过不能代替真实分配结果。

父进程完成配置校验和编译，再串行启动全新的 worker 进程，每个 worker 完成一个后端/类型/参数/seed 用例。保留可配置超时，默认 1800 秒；崩溃、超时、运行错误和无效结果写失败行，后续用例继续。用户中断时回收当前进程并落盘已完成结果。

## 文件与输出

```text
brainevent/_tcsr/dev/sddmm/benchmark/
    benchmark_sddmm.py
    benchmark_sddmm_test.py
    benchmark_sddmm_native.cu
    benchmark_sddmm_native_test.py
    circular_inputs.cuh
    circular_inputs_test.py
    README.md
    build/                    编译产物
    runs/<timestamp>/
        manifest.json
        case_001/
            case_1.csv        第一个 seed
            case_2.csv        第二个 seed（配置时）
            summary.csv
            run.log
```

binary 和 float 原始 `.cu` 分别作为编译单元链接到 runner，避免其匿名命名空间和同名模板在一个编译单元内冲突。直接调用现有 native 入口，不复制修改计算内核。新 cuSPARSE native 适配支持上述计时边界和 dtype；现有 FFI 包装不作为计时入口。

复用 MM 的流程和字段含义，不通过运行时导入 MM 驱动建立跨实验目录依赖。构建依赖纳入实际 `.cu`、公共头文件与本地头文件；源文件变化触发重建。

CSV 保存后端、事件语义、event/value dtype、shape、网络和输入种子、两池长度、计时配置、执行模式/块数、正确性状态与错误。失败或跳过的耗时为空，不写 0；跳过内存或不支持组合记录具体原因。跨 seed 汇总键包含除 seed 外的完整输入及执行配置和 timing_mode，跳过、校验失败和运行失败分别计数，speedup 只使用参数匹配且校验成功的对照。

manifest 记录配置、源码摘要、编译命令、GPU/CUDA/cuSPARSE 版本、池复制和计时约定、输入分布以及完成/中断状态。父进程拥有 CSV 写权限，每行 flush/fsync，日志保留 seed 与文件的映射。

## 实施顺序和验收

按依赖顺序实施，问题先添加可复现测试，再修复。所有测试使用同目录 `*_test.py`，不新增 `tests/` 或 `test/` 子目录。

| 顺序 | 工作 | 必须验证 |
| --- | --- | --- |
| 1 | 循环池、独立分配与双布局 gather | 地址范围不重叠；双向修改互不影响；NB/BN 内容与独立 CPU 参考一致 |
| 2 | 小规模 direct runner 和后端适配 | 五项默认 f32 对照及支持的 f64 类型组合逐槽正确；重复调用覆盖输出 |
| 3 | 逐步事件及统计 | gather 在事件外；每步每块一对事件；每轮一次同步；无逐步分配；统计与探针一致 |
| 4 | 按行 chunk、显存估算与失败清理 | 尾块、全局输入定位、direct/chunk 一致、低阈值分块、溢出和注入故障 |
| 5 | Python 子进程及结果落盘 | 成功、崩溃、超时、损坏结果、中断、失败后继续、跨 seed 汇总不混组 |
| 6 | 集成验收和文档 | 小规模真实 worker smoke、覆盖率、mypy、JAX 兼容检查和可复现命令 |

### 必须证明独立数据副本

不能仅断言两个指针数值不同，池内偏移也会产生不同指针。测试需同时完成：

1. 用分配探针记录 allocation base 和字节范围，确认两池与两个 staging 的范围分别不重叠，且所有资源最终释放。
2. gather 后修改 staging 的若干元素，验证池内容未变。
3. 重新 gather 保存快照，然后修改被读取的池元素；验证 staging 保持原内容，直到下一次 gather 才反映池的变化。
4. 多步执行时 staging 地址保持稳定，分配次数不随步数增长；源池在正式测量期间保持不变。
5. CPU 参考独立实现模索引及布局；验证源、目标均不越界，覆盖未回绕、池尾回绕、窗口大于池和多次回绕。

上述内容在独立数值测试中同步、拷回验证，不把检查探针或主机拷贝放入性能测量流程。

### 数值和边界

- event 全零、全活动、float 正负值，以及 binary 对负值不活动的语义。
- K=4、8、16、32、64、128、256、512；拒绝 1、2、3、129 等不支持档位。
- 行数 127/128/129，列数 8191/8192/8193 与 32767/32768/32769，矩形结构、空行和零 nnz。
- 同一池下 step 274/275/276 的周期和回绕；预热、测量轮次重放一致。
- 不整除的行块、最后一块、无活动块，以及同一步不同 chunk 读取相同 ct。
- 稀疏数值校验按槽位进行，不构造大型稠密乘积。小规模独立验证每一步完整输出；正式用例在测量前后用独立验证调用全量检查小问题、大问题抽样，并保存样本覆盖信息。
- f32 参考使用更高精度累加和与内积项规模相关的误差界；f64 使用独立高精度小规模参考及明确误差界。不能把 BE 与 cuSPARSE 互相一致作为唯一正确性证据。
- binary-bool 与 binary 数值输入在同精度、相同活动标志下输出一致；float 参考必须包含乘法幅值。

Python 驱动和辅助逻辑覆盖率 >90%，native 主机侧 gcov 覆盖率 >90%；两者单独报告，主机覆盖率不代表设备内核覆盖率。mypy 检查新增 Python 驱动及测试，无错误；公共接口使用 NumPy 风格 docstring。

native benchmark 不依赖 JAX 调用计时。与现有 JAX 接口的互操作检查在 JAX 0.8.0 和当前可用版本分别进行，避免测试辅助代码使用仅新版存在的配置项。GPU smoke 不等同于正式大规模性能验收，全量网格不作为开发测试自动启动。

## 完成清单

- [x] 固定双循环池、独立 staging、布局和计时边界。
- [x] 核对现有 tilemm 的独立分配与复制实现。
- [x] 实现循环输入并通过双向修改不互相影响的测试。
- [x] 实现 direct、chunk 及各后端数值测试。
- [x] 实现计时探针、异常清理、进程隔离和结果写出。
- [x] 完成覆盖率、mypy、JAX 兼容及真实 worker smoke。
- [x] 更新运行说明，记录实际验证环境和未覆盖的大规模条件。


## 实施记录

新增文件均位于原 `benchmark` 目录，现有 binary/float CUDA 内核不修改；两者分别编译链接。所有 Python 测试采用同目录 `*_test.py` 命名。工作分支保持 `mm-optimizing`，沿用现有 dev 目录的本地 Git 排除规则，没有更改索引、提交或推送。

落实的额外细节：

- 每次显存分配前查询 `cudaMemGetInfo`，按 CUB/cuSPARSE 实际查询值分配 workspace。CSV 记录初始可用显存和两种 workspace 字节数；静态估算保留库工作区不确定性的说明。
- 共享库签名包含实现、原始 CUDA 文件、公共头文件及 nvcc 版本；manifest 记录编译命令、运行库版本、输入分布和 seed 到 CSV 的映射。
- 每行 CSV flush/fsync；worker 超时或用户中断时 kill/wait 回收。父进程继续处理失败后的配置，汇总不合并不同池长度、精度或分块阈值。
- 稀疏结构使用固定行长的 Floyd 无放回抽样，网络 seed 与全局行决定列集合；分块与 direct 读取相同逻辑结构。

### 回归修正与避免重复

1. 自定义输入验证入口最初把 binary 原始浮点幅值直接交给 cuSPARSE。正负非单位输入测试复现后，改为在调用前物化 `>0` 掩码；进一步用正数下溢测试固定“先转输入精度，再判断正值”的顺序。正式 gather 生成的 0/1 不受该问题影响，但验证入口必须遵守相同类型语义。
2. CSV 汇总最初未防御 NaN，可能导致标准差计算失败。先加入损坏数值回归用例，再将非有限、负数或不可解析的成功行计入失败，不混入时间统计。
3. manifest 最初保存展开后的全部用例，并随每行重复写入。回归用例固定紧凑网格格式，改为记录原始网格、默认配置、seeds 和文件映射。
4. JAX 辅助代码统一采用 0.8.0 已支持的 `jax.enable_x64(True)`，避免引用后续版本已移除的 experimental API。保留两个实际 JAX 环境的 FFI/native 对照检查。
5. 覆盖率探针使用专用 `--basetemp` 并提前创建父目录，避免其他 pytest 会话清理通用临时目录而丢失报告。


### 初版验收结果（2026-10-19，旧清零口径）

本节记录 `gpu_events_including_output_clear` 初版，不作为新口径的性能结果；现行 `gpu_events_operator_only_v2` 的验收见后续对齐规格。

环境：Python 3.12.13，CUDA toolkit/runtime 13.0，cuSPARSE 12.6.3，驱动 580.95.05，GPU 1 为 NVIDIA RTX PRO 6000 Blackwell Server Edition。

| 检查 | 结果 |
| --- | --- |
| Python 调度与结果测试 | 56 通过 |
| GPU native 与循环输入测试 | 231 通过 |
| JAX 0.8.0 现有 FFI/native 对照 | 16 通过 |
| JAX 0.11.0 现有 FFI/native 对照 | 16 通过 |
| 正式 O3 产物、真实隔离 worker | 24/24 通过，12 种后端/精度组合各检查 direct 和 chunk |
| worker 默认轮次 | 25 步 ×（2 轮预热 + 5 轮测量），direct 1 块、chunk 5 块 |
| 输出与元数据复核 | CSV/summary 各 24 行；全部 validation_status=passed；manifest 已完成且源码摘要匹配 |
| Python 驱动覆盖率 | 语句 100%；含分支 99.52% |
| native runner 主机源码行覆盖率 | 273/274，99.64% |
| circular_inputs 主机源码行覆盖率 | 67/67，100% |
| mypy | 4 个新增 Python 源文件，无错误 |

主机覆盖率使用 gcov，模板实例按同一源码行合并，并排除 `__global__` 设备函数及其未使用的 host launch stub；原始 gcov 报告同时保留，不把该指标解释成设备内核或所有异常分支覆盖率。没有为了覆盖率触发真实 OOM，显存不足路径使用配置筛选与故障注入验证。

24 个 worker 每项验证 465 个槽（3 次 × 155 nnz），最大误差／误差界比为 0.01151 以下。该检查只证明这些小用例的数值和运行流程，不作为正式规模性能结论。

可复核产物位于：

- `brainevent/_tcsr/dev/sddmm/benchmark/runs/validation-20261019-native/`：manifest、逐项 CSV、summary、worker 日志。
- `brainevent/_tcsr/dev/sddmm/benchmark/build/validation/validation-summary.json`：汇总验证记录。
- `build/validation/python-coverage.json` 与 `python.coverage`：Python 覆盖率。
- `build/validation/native/sddmm-native-probe0/`：gcov 原始报告及 `host-source-coverage.json`。
- `build/validation/jax080/` 与 `jax011/`：各版本互操作探针产物。

以上后三项路径相对于 benchmark 目录。运行与复验命令详见 [README](../../brainevent/_tcsr/dev/sddmm/benchmark/README.md)。尚未执行正式 40960 候选配置／seed 的全量网格；建议后续在独占 GPU 上验证百万级结构、真实显存临界规模和多 seed 性能分布。当前小规模验收覆盖所有支持 batch，并覆盖回绕、空行、零 nnz、无活动、全活动、短尾块、整数溢出和异常退出等边界。
