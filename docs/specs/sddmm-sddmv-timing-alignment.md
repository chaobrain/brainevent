# SDDMM 与 SDDMV 输出清零口径对齐

日期：2026-10-19。工作分支：`mm-optimizing`。用户通过“对齐sddmv，请你确定当前mv mm的计时口径差异，并确定 cudaMemsetAsync 操作的位置”授权实施。

## 核对结论和范围

本次 mv/mm 指 `dev/sddmv/benchmark` 与 `dev/sddmm/benchmark`。上一版 SDDMM 在 cuSPARSE 适配层每步、每块额外执行输出清零，并计入事件耗时；SDDMV cuSPARSE 每步只更新输入描述符并以 beta=0 调用 SDDMM，没有这项清零。

按 SDDMV 对齐输出生命周期：所有后端的输出缓冲在分配后、首次验证和计时前初始化为零一次；BE 保留原算子 launch 内部的每次输出清零；cuSPARSE 每次调用不再显式清零，依靠 beta=0 覆盖全部有效输出槽。初始化使用 cudaMemset，位于 Operator 构造阶段。不得将原来的逐步清零简单移动到每步事件前。

保留先前批准的双循环池、每步每块事件与整轮同步方案。本次比较并记录这些与 SDDMV 不同的测试条件，不将它们描述为完全相同的计时方式。

## 计时差异表

| 项目 | SDDMV 当前 benchmark | SDDMM 本次对齐后 |
| --- | --- | --- |
| 默认执行 | 100 步，2 轮预热、5 轮测量 | 25 步，2 轮预热、5 轮测量 |
| CUDA Event 窗口 | 最多 32 个连续 step 一对事件 | 每个 step 的每个行块一对事件 |
| 同步 | 每个窗口末尾同步 stop event | 一整轮结束后同步 stream |
| CUDA stream | 单独创建的 non-blocking stream | 默认 stream |
| 输入准备 | 每窗口生成最多 32 步 event/ct，bool 对照转换也在事件外 | 每步从固定双池复制；ct 一次、event 按行块，均在事件外 |
| 输出初始化 | DeviceBuffer 分配时同步清零一次，事件外 | Operator 输出分配后同步清零一次，事件外 |
| BE baseline 输出清零 | 原算子入口内部，计时内 | 原算子入口内部，计时内 |
| cuSPARSE 输出清零 | 无逐步显式清零；beta=0 | 无逐步显式清零；beta=0 |
| cuSPARSE 描述符 | 创建在计时外；每步更新 A/B 地址发生在窗口内 | 创建在计时外；staging 地址固定，无逐步 SetValues |
| cuSPARSE 算法 | DEFAULT，未启用 preprocess | DEFAULT，未启用 preprocess |
| 结构准备 | 构造网络在计时前 | direct 计时前；chunk 每块在对应事件前准备 |
| 轮次统计 | 所有窗口耗时之和 | 所有 step/chunk 耗时之和 |
| prezeroed 变体 | 单独方案，非活动位置保留旧值 | 当前仅验证每步完整输出语义，无此变体 |

两种事件粒度和输入准备方式会改变提交间隙及缓存状态，原始整轮时间还对应不同步数；不能直接拿两者整轮时间作同工作量比较。单步时间也分别对应 MV 和 batch MM 的不同运算量。

## 修改和回归要求

1. 先将探针断言改成“native 适配层计时内没有额外清零，输出初始化恰好一次且发生在首事件前”，在旧库上复现失败。
2. 从 `Operator::launch()` 的 cuSPARSE 分支删除输出 cudaMemsetAsync；在输出分配后初始化一次。
3. 测试填充非零/NaN 输出后，连续执行“活动输入 → 全零输入”，证明 beta=0 覆盖结果，不能依赖逐步 memset 掩盖旧值；覆盖 f32/f64 与 binary/float。
4. 更新 CSV 的 timing_mode、manifest 输出初始化/计时说明。summary 分组键包含 timing_mode，防止新旧清零口径混合。
5. 更新 README、原规格中的现行描述，明确旧验证目录属于旧口径；旧 CSV、manifest 和耗时不改写。新验证写独立目录。
6. 运行 CPU/native 回归、mypy、主机覆盖率以及生产 O3 direct/chunk 小用例。

旧 FFI 包装 `baseline/cusparse_sddmm.cu:81` 仍有清零，但当前 native benchmark 未编译或调用它；本次不修改该独立入口。SDDMM native 的 hash 表 cudaMemsetAsync 属于结构准备，位于事件外，与输出无关。

## 原因与防止重犯

此前把“双方都显式清零”误作统一比较条件，给 cuSPARSE 增加了参考实现没有的工作。以后先逐层核对内存初始化、算子 launch 和事件边界；使用探针按后端固定清零次数/位置，并将计时口径纳入结果分组键。

## 验收记录

已完成。输出初始化位于 `benchmark_sddmm_native.cu:203`，使用 cudaMemset；cuSPARSE launch 分支已删除额外输出 cudaMemsetAsync。BE binary/float 的输出异步清零分别仍位于 `baseline/sddmm_binary.cu:240`、`baseline/sddmm_float.cu:353`。native 的 hash 表 cudaMemsetAsync 位于第 160 行，结构准备在第 366 行执行、start event 在第 369 行，因此不计入事件耗时。

验收：57 项 CPU 测试、214 项 native GPU 测试全部通过，mypy 检查 4 个文件无错误；生产 O3 产物 24 个真实 worker 全部成功（12 种后端/精度 × direct/chunk，25 步、2 轮预热、5 轮测量）。CSV、summary、manifest 已确认采用 `gpu_events_operator_only_v2`，源文件摘要一致。最大误差界比例 <0.01151。

覆盖率：Python 语句 100%，含分支 99.76%；gcov 主机源码行 runner 274/275（99.64%）、circular_inputs 67/67（100%）。模板实例按源码行合并，排除设备函数；不表示 GPU 指令覆盖率。

新增/更新边界测试：初始化清零失败时释放全部资源；适配层首次事件前只初始化一次；direct/chunk、验证调用、预热和多轮计时期间均无额外输出清零；cuSPARSE f32/f64 binary/float 覆盖 NaN 旧输出以及“活动 → 全零”；新旧计时口径不得合并汇总。已有空行、零 nnz、短尾块和事件同步测试继续通过。

结果目录：`brainevent/_tcsr/dev/sddmm/benchmark/runs/validation-20261019-sddmv-aligned/`。详细报告：`build/validation/alignment-summary.json`（相对于 benchmark）。初版 `validation-20261019-native` 目录不改写，README 已标为旧口径。

尚未运行完整大规模网格。后续性能对比应使用一致的 timing_mode、工作量和输入条件；若要消除窗口粒度、同步或 stream 差异，需要单独变更这些已列明的测试条件。
