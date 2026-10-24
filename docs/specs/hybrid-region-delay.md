# Hybrid 分段延迟算子实现计划

在现有 hybrid 的 Extract 和 Block Scatter 两阶段调度上，增加按延迟段遍历与环形输出缓冲。每行的边在构建时排序，运行时用 `row * K` 定位段连接数；短行直接散射，长行注册不跨段的任务。

基线为 [原始 hybrid](../../brainevent/_tcsr/dev/with_delay/operator/ori/binary_csrmv_hybrid.cu)，新算子放在 `brainevent/_tcsr/dev/with_delay/operator/`。本轮交付范围为 CUDA 算子、JAX 封装和同级正确性测试。本计划按本轮确认的固定段计数布局细化[原方案](../../brainevent/_tcsr/dev/with_delay/md/hybrid_region_delay.md)。

1. **统一数据布局和时间语义。**

   `M` 为源行数，`N` 为目标数，`K` 为全图不同整数延迟值的数量。`delay_steps[K]` 按实际延迟升序排列，各行使用同一段编号；没有该延迟连接的段计数为零。槽数 `L = max(delay_steps) + 1`，与 `K` 分开定义，例如延迟为 `[0, 2, 5]` 时 `K=3`、`L=6`。无连接时保留一个零延迟类别，使缓冲和元数据形状明确。

   首版接收已经离散化的非负整数延迟步数，支持零延迟。每步先由已知 `events(t)` 完成所有投射，再消费并清零 `B[head]`，更新神经元得到 `events(t+1)`，最后推进 `head`。这一约定与[延迟图规格](hybrid-region-delay-figure.md)一致，零延迟在本步消费。

   | 数据 | 形状与含义 |
   | --- | --- |
   | `weights, indices, indptr` | 按行内延迟段顺序排列的 CSR；异质权重与列索引同步排序 |
   | `delay_count` | `[M, K]`，行优先存储；`delay_count[row*K+k]` 表示展平后的访问方式 |
   | `delay_off` | `[K]`，`delay_steps[k] * N`，各行共享 |
   | `row_task_count` | `[M]`，每行需要的任务数，短行为零 |
   | `B` | `[L, N]`，时间优先、每个时间槽连续存放 N 个元素 |
   | `head` | device `int32[1]`，范围为 `[0, L)` |
   | `task_begin, task_end, task_out_off` | `[task_capacity]`，任务的边起止位置和输出槽基址 |
   | `status` | `int32[2]`，本次任务数与溢出标记 |

   固定段计数直接决定边界，不需要 `row_seg_ptr` 或 `seg_end`。静态新增存储为 `M*K` 个段计数、`M` 个行任务数和 `K` 个共享延迟偏移；构建时报告实际字节数。

2. **完成构建与容量计算。**

   构建器接收 CSR、源/目标脑区边界及脑区对延迟表；第一版要求脑区连续编号。逐行按 `(实际延迟, 目标列号)` 排序，相同键保持稳定顺序，同质权重继续使用一个标量。排序只发生在构建阶段，运行时不保留逐边延迟标签。

   生成每行的 `K` 个连接数，并验证计数非负、计数和等于 `indptr[row+1]-indptr[row]`、目标列合法以及权重与边的对应关系。输入已满足排序要求时复用其边数组。延迟排序后的结构属于新实验算子，不直接交给要求整行列号升序的现有 TCSR tile 元数据。

   设整行阈值为 `T`，任务最大边数为 `C`，初始采用基线默认值 `T=128`、`C=4096`。构建器和 CUDA 编译参数共用同一份配置。

   ```text
   行长 <= T：row_task_count[row] = 0
   行长 >  T：row_task_count[row] = sum(ceil(delay_count[row, k] / C))
   task_capacity = sum(row_task_count)
   ```

   容量按所有长行同时活跃计算。零连接段不生成任务。向上取整使用商和余数，容量求和使用宽位整数，并在分配前检查 int32 任务计数器上限；配置变化后重建任务数与 workspace。

3. **修改 Extract 阶段，完成短行散射与长行注册。**

   一个线程负责一行。先检查事件，再读取 `indptr` 得到行范围；空行直接结束，其他行按整行长度分流。这里的 TPR 阶段指 Extract kernel，包括短行直接计算和长行任务注册。

   短行：用 `row*K` 定位首段，边指针从 `indptr[row]` 开始。每段读取连接数，空段跳过；非空段读取共享 `delay_off[k]`，计算 `base = head*N + delay_off[k]`，若 `base >= L*N` 则减去 `L*N`。段内逐边原子累加到 `B + base`，达到本段连接数后切换下一段。`head` 在段循环外读取；同质权重也在段循环外加载。

   长行：读取 `row_task_count[row]`，用一次 `atomicAdd` 预留整行任务位置。随后顺序读取各段连接数与输出偏移，每段按最多 `C` 条边切块，写入任务三元组：

   ```text
   task_begin   = 当前边指针
   task_end     = 当前边指针 + min(C, 本段剩余边数)
   task_out_off = 当前段的输出时间槽基址
   ```

   每登记一块，边指针推进到 `task_end`，任务指针加一；一段结束后读取下一段。注册过程不读取逐边 `indices/weights`。长行中的短段也必须生成任务，不能调用带有短行阈值过滤的原 `TaskChunksForRowNnz` 来计算段任务数。

   固定示例作为验收用例：行范围 `[10000,15400)`、段计数 `[100,5000,300]`、延迟 `[0,1,2]`、`head=2`、`L=3`、队列初始计数为 8。一次预留四项，得到：

   | 任务位置 | begin | end | out_off |
   | --- | --- | --- | --- |
   | 8 | 10000 | 10100 | `2*N` |
   | 9 | 10100 | 14196 | `0` |
   | 10 | 14196 | 15100 | `0` |
   | 11 | 15100 | 15400 | `N` |

4. **接入 WPR 和 JAX 状态传递。**

   WPR 沿用基线的 Block kernel：固定数量 CTA 以 grid-stride 方式取任务，每个任务多读取一个 `task_out_off`，再协作执行该范围内的散射。本轮不改变 CTA/warp 调度粒度；文中的 WPR 指这个协作处理阶段。

   删除投射入口对整个输出缓冲的清零，保留每次调用的 `status` 清零，包括空结构路径。Extract 与 Block kernel 使用同一 CUDA stream 顺序执行，消费发生在整个投射完成后。保留任务容量检查，溢出时置位状态并避免消费未初始化的任务，不能把截断计算视为成功。

   Python 提供构建结构、分配 workspace、投射累加、消费清零四个阶段的接口，集中在 `region_delay.py`。复用仓库 CUDA 加载器和 FFI；返回更新后的 `B` 与 workspace，通过 `input_output_aliases` 声明对应别名。`head` 作为 device operand，`K/L/N` 和调度配置为静态信息。

   `lax.scan` 显式携带 `B/head/workspace`；闭环仿真另携带神经元状态与当前事件。多个投射共享同一缓冲时依次传递返回值，最后统一消费一次。初版动态索引组合的 GPU 编译检查已复现循环内整块 `L*N` 复制，消费改为独立的 CUDA 读出并清零 kernel，通过 FFI 为环形缓冲声明输入输出别名；仅输出当前 N 个元素，head 仍由 JAX 推进。用编译结果与实际指针复用测试约束该行为，禁止 `jnp.roll`。

   新增入口首先覆盖 float32 同质/异质权重、bool 与 float32 事件（非零表示活跃，不乘事件幅值），以及 int32/int64 CSR 索引和行指针。段计数与行指针同型，行任务数为 int32。环形偏移类型按 `L*N` 单独选择，与列索引类型解耦；`row*K`、地址计算和边遍历的中间运算采用 int64。需用 int64 的 JAX 输入必须实际保留该类型，禁止静默截断。

5. **按阶段验证正确性与兼容性。**

   | 顺序 | 交付文件，相对 `brainevent/_tcsr/dev/with_delay/` | 验收重点 |
   | --- | --- | --- |
   | 1 | `operator/region_delay.py`、`operator/region_delay_test.py` | 排序、计数、容量、权重对齐与输入校验 |
   | 2 | `operator/binary_csrmv_hybrid_region_delay.cu`、同级 `binary_csrmv_hybrid_region_delay_test.cu`，扩充同级 Python 测试 | 短行散射、任务覆盖、原算子边界复现与真实 GPU 尾部遍历 |
   | 3 | 扩充上述封装和测试 | WPR、FFI、环形缓冲、多步与多投射组合 |

   CPU 参考按原始脑区对延迟和绝对到达时刻累加，不依赖 GPU 的排序结果、段计数或环形寻址。至少运行 `2*L+1` 步，逐步比较消费结果，并检查其余未来槽的保留情况。

   必测边界包括：零延迟、最大延迟、非连续延迟值、`L=1`、环绕、空行/空段/空图、无事件、重复目标、同格不同来源累加、`T/T+1`、`C-1/C/C+1`、长行包含多个短段、多个长行同时注册、零任务容量、跨步 workspace 复用和多投射共享缓冲。检查实际任务数与无重叠、无遗漏、无跨段覆盖；并行行的预留顺序不作为断言条件。

   对原 hybrid 的 int32 中间运算风险，先补失败用例，再在新实现中使用安全边界计算与宽位遍历。合成大偏移用于任务注册测试，另用实际 GPU 路径验证散射；主机算术测试不代替 GPU 验证。状态复用测试以非零旧状态启动，防止空结构路径遗留旧计数。

   新增 Python 实现的有意义路径覆盖率大于 90%，实现与测试通过 mypy；公共接口使用 NumPy 风格 docstring。分别验证 JAX 0.8.0 与当前安装版本，并记录实际 GPU 验证结果。所有 Python 测试与模块同级，使用 `foo_test.py` 命名。

6. **同步方案描述。**

   实现时将原方案正文更新为固定 `delay_count[M,K]` 布局和零延迟步进顺序，说明行内排序键及 `K` 与 `L` 的区别。对结构的更正同时核对预处理、任务容量和 kernel 遍历，避免再次把固定段计数解释成可变段指针；对时间语义的更正同时核对槽数、输出偏移和消费顺序。

7. **实现与验证结果。**

   2026-10-19 已完成上述 CUDA 算子、Python 封装、同级 Python/CUDA 测试，以及原方案正文和可运行示例的同步。新增接口提供 float32 权重、bool/float32 非零事件、int32/int64 CSR 及独立环形偏移类型的 CUDA 前向计算。

   在物理 GPU 1（NVIDIA RTX PRO 6000 Blackwell Server Edition）、CUDA 13.0、Python 3.12.13 上验证：JAX 0.11.0 和 JAX 0.8.0 各通过 68 项完整测试及随后补充的 4 项多 block 消费边界测试，均无跳过。新增 Python 实现覆盖率为 99.44%（完整 68 项测试采集），实现和测试文件的 mypy 均无错误。

   原始 Extract 在 `INT32_MAX` 附近的任务终点溢出已在 GPU 复现；新实现通过相同边界、超 32 位合成偏移、全部 524288 个大行任务覆盖，以及实际 8 GiB 列索引数组最后 500 条边的散射验证。任务示例还使用空权重/索引指针执行注册，确认注册路径不读取逐边数据。

   动态索引消费组合的初版在 scan 编译中出现两次整块 B 复制。先用失败的编译结果测试复现，再改为独立读出清零 kernel；两版 JAX 均通过循环体无整块 B copy、外层 donation 复用原设备指针的检查。多 block 消费另覆盖 255、256、257、1031 列，head 更新在 kernel 外执行。

   辅助 CUDA 测试首次编译缺少同级源码的 include 路径，现已显式传入源码目录，并将被包含源码纳入测试模块命名摘要，避免路径与缓存遗漏。后续继续用真实编译、真实 GPU 和内存复用检查验证这些约束，而不根据 Python 表达式推断零拷贝。

   正确性测试入口：`python -m pytest brainevent/_tcsr/dev/with_delay/operator/region_delay_test.py -q`。文档中的三步示例已实际执行，输出依次为 `[1,0]`、`[0,0]`、`[0,2]`。本轮没有新增 benchmark 文件。
