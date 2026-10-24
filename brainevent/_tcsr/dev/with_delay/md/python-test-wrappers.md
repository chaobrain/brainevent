# 延迟算子的 Python 正确性测试封装精简

状态：已实现并验证。文档位置按用户要求统一到本目录。

## 逐 step NumPy 对照（2026-10-22）

在 `operator/test/region_delay_test.py` 补充独立的 NumPy 单步投射实现，
输入为排序前的 CSR、逐边 delay、一个事件列和当前 ring/head。按非零
事件选择源行，通过 `np.add.at` 累加重复目标，使用 float64 参考结果；
不使用 CUDA split 输出、排序后的段计数或 GPU 结果生成期望值。

新增测试先用可手算的重复边、负值活动、head 环绕和共享权重验证 NumPy
参考本身；随后初始化完整事件矩阵，按列顺序逐步调用 NumPy、hybrid
和 split → cuSPARSE，每步即时比较投射后的完整 ring 和当前槽输出，
再由测试消费、清零、推进 head。split 只在一轮开始时调用一次。

现有绝对到达时间参考保留，用于交叉检查 NumPy 单步参考整轮输出。
测试失败信息包含 backend、step 和 head。用户已确认使用 `events[N,T]`，
按列顺序读取 `events[:,t]`，覆盖 T=K 和 T≠K。
按用户最新要求，连接矩阵扩大为 `10000×10000`，K 固定为 8，每列
事件独立按 30% 活跃概率生成。移除本用例首两列强制全活跃/全静默的
覆盖设置；这些边界继续由现有其他用例验证。

参数组合为 `(8,8)`、`(8,17)`、`(8,3)`，并另保留 `(8,3)` 下
`delay=4` 为空段的边界变体。普通用例包含 `0～7` 全部八种延迟。
继续组合 bool/f32 非零事件和共享/逐边权重，覆盖重复边、128/129
短长行边界、4097 长行和 head 环绕。使用 CSR 稀疏存储，沿用原有
行长生成规则，只扩大矩阵维度，不分配 `10000×10000` 稠密数组。

一轮中 hybrid 和 cuSPARSE 整步算子各调用 T 次，cuSPARSE 子矩阵循环
仍在 CUDA 文件内部；每步有 A 个非空段时，内部 SpMV 总数为 T×A。
T=K 且全部段非空时为 K²。pytest 逐步核对数值，不通过重复调用整个
cuSPARSE 算子模拟段循环，也不把该理论次数当作库调用拦截测量结果。
生产算子与 benchmark 的调用流程保持原样。

此前整轮断言可以检查输出序列，但难以直接指出最早出错时刻的完整
延迟状态；本次补上逐步 ring 断言，避免只比较最终结果而遗漏中间状态。

扩容前验证：新增 17 项（1 项手算参考、16 项逐 step 参数组合）全部通过。
JAX 0.11.0 下整个 `region_delay_test.py` 的 82 项测试加 42 项输入准备
测试，共 124 项通过，`region_delay.py` 行覆盖率为 100%。JAX 0.8.0
环境下同一算子测试文件的 82 项全部通过。实现和测试两个文件通过
`mypy --follow-imports=silent`，本轮修改文件通过 `git diff --check`。
后续可增加 K=16/64、更多 seed 和更长 T，继续检查每步完整 ring。

10k 扩容后重新验证：JAX 0.11.0 与 JAX 0.8.0 下，本算子测试文件各
82 项全部通过；修改后的测试文件通过 mypy，本轮修改文件通过
`git diff --check`。全部事件列使用统一的 30% 概率参数，使配置与
实际生成规则一致；保留共享/逐边权重、空段和跨 ring 周期的对照。

## 输入准备合并（2026-10-22）

按用户要求，将 `operator/delay_inputs.py` 的全部输入准备实现合并到
`operator/region_delay.py`，删除独立的 `delay_inputs.py`。下文精简记录
中的文件划分和行数是合并前的历史结果，以本节和 operator.md 为当前接口说明。

- 保留 `DelayInputs`、`prepare_delay_inputs`、`validate_delay_steps`、
  调度常量及原有 NumPy 校验和排序行为；三个 CUDA 调用封装保持不变。
- benchmark、原生 runner 和正确性测试统一从 `region_delay.py` 导入。
- 保留当前 `operator/test/` 和 `benchmark/test/` 测试布局，修正移动后
  CUDA 测试的源文件和 include 路径；输入准备测试继续独立覆盖空图、
  空段、dtype 和溢出边界。
- 不增加兼容模块或持久化层。split 仍只在 cuSPARSE 初始化时调用，
  子矩阵循环仍由 CUDA 控制。
- 验收运行现有 GPU/CPU 回归、覆盖率、mypy 和 JAX 0.8.0 兼容性检查。

此前将共享输入准备拆成独立模块，增加了阅读这组三算子调用的文件跳转。
本次按用户要求集中到一个 Python 入口；后续新增输入辅助逻辑先在该入口
内组织，不再自行拆出模块。

合并验证结果：

- `region_delay.py` 共 383 行，包含原输入准备和三个调用封装。
- JAX 0.11.0：164 项回归全部通过；`region_delay.py` 的 116 个语句
  覆盖率为 100%，operator 与 benchmark 合计为 96.49%。
- JAX 0.8.0：65 项 GPU 算子测试和 42 项输入准备测试，共 107 项通过。
- 本次 8 个相关 Python 文件通过 `mypy --follow-imports=silent`。
  未限制导入范围的检查在仓库其他 25 个文件中报告 216 个类型错误，
  因此此处通过结论仅覆盖本次指定文件，不表示全仓类型检查通过。
- 两个原生 CUDA 测试先复现源文件/include 路径错误，再通过现有用例
  验证修复。测试路径以当前目录布局为准，避免继续引用迁移前的位置。
- 当前覆盖空图/空段、零维、非连续延迟、共享权重、事件列连续性、
  128/4096 任务边界和 i32/i64 溢出校验；后续可增加 K=16/64 且
  T≠K 的随机组合，与原始逐边 CPU 参考比较。

## 目标

`operator/region_delay.py` 只服务同级 `region_delay_test.py`，负责加载和
调用三个 CUDA 算子，并直接串联 split 与 cuSPARSE。原有两个文件为
694 / 712 行，目标显著缩减到约 200 / 300 行，保留可读的类型标注和
NumPy 风格接口说明，避免仅通过压缩排版减少行数。

此前 Python 层保留了脑区延迟预处理和完整的结构、配置、状态管理，
测试也承担了大量辅助 API 校验，职责超出测试封装。此次按 CUDA 输入
输出组织接口；测试负责生成输入和独立判断结果。

## 保留接口

| Python 接口 | CUDA 调用与返回 |
| --- | --- |
| `split_region_delay(weights, indices, indptr, delay_count)` | 调用 `region_delay_split_f32`，返回四份原始 split 数组 |
| `project_region_delay(weights, indices, indptr, events, delay_count, delay_off, row_task_count, head, buffer, workspace)` | 调用 `region_delay_f32`，返回 buffer 和任务 workspace/status |
| `project_region_delay_cusparse(split, events, delay_off, head, buffer, scratch)` | 将完整 split 结果交给一次 `region_delay_cusparse_f32`，返回 buffer 和 scratch |

保留一个私有 CUDA 加载缓存，以及必要的 FFI 输出形状、dtype 和别名
声明。split 输出和 hybrid workspace 最多采用简单的命名元组组织数组，
不附加构建、校验、缓存或执行策略。保留 bool / float32 事件的二值语义，
cuSPARSE 输入事件转换为 float32 0/1。

cuSPARSE 使用方式为：测试初始化调用一次 `split_region_delay`；后续
每步把原样 split 结果和当前事件列传给 `project_region_delay_cusparse`。
K 个子矩阵循环继续完全由 `region_delay_cusparse.cu` 控制。

## 已删除内容

- 脑区边界/脑区对延迟接口、CSR 排序和预处理实现。
- `RegionDelayConfig`、`RegionDelay`、`RegionDelayCusparse` 及其 JAX
  dataclass 注册、存储统计、配置派生逻辑。
- Python 中重复的 CSR/形状/dtype 校验框架、host 内容检查和状态异常
  包装；保留 FFI 正确调用所必需的静态输出声明。
- workspace 工厂、消费及 head 推进的管理接口。测试直接分配所需数组，
  并在测试辅助代码中维护每步消费/清零/head。
- 为上述辅助 API 编写的测试、脑区专用场景和 HLO 文本/存储地址检查。

本轮修改两个 Python 文件及相关说明。CUDA 算法与原生 benchmark 的
调用流程保持既有职责；旧的测试专用 Python API 不保留兼容别名。

## 正确性测试

测试输入复用已有 `delay_inputs.py` 的逐边延迟预处理；不把旧的脑区
预处理搬到新文件，也不重写一份生产侧排序实现。独立参考从原始逐边
CSR、逐边 delay 和 `events[M,T]` 计算绝对到达时间，不以 hybrid 与
cuSPARSE 彼此相等作为唯一判据。

保留并合并以下测试：

1. split 的四份输出与 CPU 分段参考逐项一致，包含空段、同质权重展开。
2. hybrid 与 split → cuSPARSE 的逐步输出、剩余 ring 都符合独立参考。
3. 参数化覆盖 bool/f32 事件、同质/异质权重、i32/i64 索引/行指针/偏移，
   无事件、全活跃、负值活动、重复连接、短长行和任务切块边界。
4. 覆盖零延迟、非连续延迟、空行/空图/零维、单槽、head 环绕、已有
   ring 内容和 T≠K；按列顺序执行，split 只调用一次。
5. 保留每步一次 cuSPARSE FFI 调用的约束，以及直接来自 CUDA 的关键
   状态/边界错误检查。

## 验收

- 实现和首次验证在 `codex/delay-benchmark-inputs` 工作树中完成；最终
  交付至用户当前 `brainevent-main` 工作目录的 `mm-optimizing` 分支。
- 改写后的算子正确性测试通过，原生 benchmark 相关回归通过。
- 精简后的 Python 封装覆盖率超过 90%，相关 mypy 检查无错误。
- JAX 0.8.0 及当前验证环境 0.11.0 都运行真实 GPU 正确性测试。
- 将 operator 说明集中到本目录，移除被删除的 Python API 示例；记录实际行数、
  验证结果和保留的边界用例。

## 实现与验证结果（2026-10-22）

| 文件 | 原行数 | 现行数 | 当前职责 |
| --- | --- | --- | --- |
| `operator/region_delay.py` | 694 | 179 | 三个 FFI 调用函数、一个加载缓存、两个数组命名元组 |
| `operator/region_delay_test.py` | 712 | 253 | 原始逐边参考、split 检查、两条路线的逐步结果与关键边界 |

合计由 1406 行缩减至 432 行。没有把删除的预处理、配置或校验框架
搬到其他 Python 文件。测试复用既有 `delay_inputs.py` 准备设备输入，
独立参考始终从排序前的逐边 CSR 和 delay 计算结果。

在 GPU 1、CUDA 13.0、Python 3.12.13 上完成：

- JAX 0.11.0：算子测试 65 项，加原生接口、输入和 benchmark 回归，
  共 **164 项通过，无跳过**。
- JAX 0.8.0：隔离环境中 **65 项算子测试全部通过，无跳过**。
- `region_delay.py` 行覆盖率 **100%**（37 个语句）；所测 operator 与
  benchmark 目录整体 **96.50%**，保留的旧行长生成器拉低整体覆盖率。
- 9 个相关 Python 实现/测试文件通过 mypy；`git diff --check` 和本目录
  Markdown 相对链接检查通过。
- Python 和 CUDA 仍保持相同的 step 边界：split 初始化一次，cuSPARSE
  每步一次 FFI，CUDA 内部执行所有非空段；原生 benchmark 回归通过。

目录内集中保存 `operator.md`、`benchmark.md`、`benchmark-inputs.md`
和本规格，原 operator/benchmark README 与 docs/specs 中的相关文档
已迁入此处。`benchmark-inputs.md` 保留前一轮验证记录，并链接本轮说明。

此前的职责偏差是把测试入口逐渐扩展成了完整预处理 API；本轮用直接
映射 CUDA 输入输出的三个函数收缩职责。后续新增测试数据构造和参考
计算应放在测试中，避免再次扩展调用封装。

后续扩展用例可选 K=16/64、包含大量空段且 T≠K 的随机图，继续比较
每步消费值和剩余 ring。当前已覆盖的边界列于“正确性测试”一节。

## 当前目录交付

用户查看的目录为
`/home/luruheng/BrainEvent/brainevent-mm进一步优化/brainevent-main`。
此前只修改旁边的 `brainevent-delay-benchmark`，没有同步到用户当前目录，
导致用户看到的仍是旧文件。交付以当前目录的实际文件、导入路径和测试
结果为准；后续不再仅以另一个工作树中的验证结果宣告交付完成。

同步完成后已按用户要求删除 `codex/delay-benchmark-inputs` 分支及
`brainevent-delay-benchmark` 工作树。后续只使用当前 `mm-optimizing` 分支。

同步限于本任务的 operator、benchmark 源码和四份 Markdown；原有
`md/hybrid_with_delay.md`、`md/hybrid_region_delay.md`、fig、ref 及其他
未提交内容予以保留。被替换和移除的旧文件先在临时目录中备份。

交付核验已完成：22 份文件与已验证实现逐字节一致，其中 19 份为新建或
替换，6 份旧接入/说明文件已清理；95 份原有文档、图与参考文件及其他
Git 未提交状态保持一致。当前目录实际导入的 `region_delay.py` 为 179 行，
同级测试为 253 行。在当前目录重新运行 164 项测试全部通过，9 个文件
的 mypy 检查无错误。旧文件备份位于
`/tmp/brainevent-delay-before-sync-qf19xn1m`。
