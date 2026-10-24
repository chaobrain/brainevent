# prezore_sddmm：覆盖活动行并保留未更新输出

日期：2026-10-19。目标分支：`mm-optimizing`。目录名按用户指定拼写为 `brainevent/_tcsr/dev/sddmm/dev/prezore_sddmm`。

状态：实现与验证完成。用户确认 float、binary 均按活动行覆盖；最新计时要求已同时应用于 cuSPARSE、baseline SDDMM 和 prezero SDDMM。

## 输出语义

调用者传入已有 dweight，与输出共享同一设备缓冲。对 CSR 槽 s=(i,j)：

- binary：若行 i 的 batch 中存在正事件（bool 为 true），覆盖为本次 sum_b active(event[i,b])*ct[b,j]；否则保留传入 dweight[s]。
- float：若行 i 的 batch 中存在非零事件，覆盖为本次 sum_b event[i,b]*ct[b,j]；否则保留传入 dweight[s]。
- 活动行的结果即使等于零，也必须覆盖为零。不能按“结果是否非零”决定写入。
- “第一个活动 phase”相对于本次调用，每次重新由输入决定。不能跨调用累加旧 dweight。
- 初始输出可为调用者提供的零、非零值或既有历史结果；不要求每次清零。没有初始化过的设备存储不构成有效历史结果。
- kernel/launch 内不调用 cudaMemsetAsync 或 cudaMemset 清除 dweight，不把清零隐蔽地移到调用前。mask、活动行等 scratch 仍由已有准备 kernel 正常覆盖生成。

## 文件与接口

新增目录内容：

```text
prezore_sddmm/
    sddmm_binary.cu
    sddmm_float.cu
    prezore_sddmm.py
    prezore_sddmm_test.py
    README.md
```

CUDA 以 baseline binary/float 为来源分别建立变体，使用独立的导出名称避免与 baseline 链接冲突。保留支持的 batch=4/8/16/32/64/128/256/512、binary 的 6 种输入输出类型配对、float 的 f32/f64，保留 int32/int64 indptr 支持。

提供 JAX factory `make_prezore_sddmm(kind)`，返回接收 `(event, ct, local_targets, indptr, tile_offsets, dweight)` 的 JIT 调用。逻辑 event 沿用 baseline 的 `[batch, rows]`，ct 为 `[batch, cols]`；FFI 接收 event.T 的连续 NB 布局。native 接口仍是 event NB、ct BN。

FFI 增加 dweight_in 输入并 alias 到 dweight 输出；Python 通过 input_output_aliases 和 donate_argnums 明确转移缓冲，调用者后续使用返回值，不再使用被 donate 的旧 JAX 数组。native 检查输入输出地址、shape、dtype 一致，避免声明保留旧值但实际使用未初始化输出。

## 内核实现

### binary

当前 baseline 在 `sddmm_binary.cu:214` 对每个活动 phase 执行 `dweight[slot] += partial`，依赖 launch 中的全输出清零。直接删除清零会产生错误的跨调用累加。

新 kernel 在当前 phase 存在活动事件时，从已经生成的 phase masks 判断此前是否存在活动 phase：

1. 当前 phase 不活动：不写输出。
2. 本轮首次活动 phase：`dweight[slot] = partial`。
3. 本轮后续活动 phase：`dweight[slot] += partial`。

判断按行进行，由 group lane 0 读取此前 masks 并广播；不添加额外的输出清零 kernel。K<=128 时只有一个 phase，所有活动行直接覆盖；K=256/512 时重点验证首个活动出现在后续 phase 的情况。

### float

现有 `first_phase` 与 `first_bit` 已实现第一次活动乘积赋值、后续乘积累加，见 baseline `sddmm_float.cu:261` 与 `:312` 附近。移除 launch 中的 dweight memset，保留其 mask、first_phase、活动行压缩、乘积与加法顺序。

## 与 benchmark 的关系

该目录提供独立算子和测试，与现有 baseline 共存，并接入 benchmark。新增 prezero_binary_bool/f32/i8/f64 与 prezero_float backend。分块时 prezero 为完整全局 CSR 分配历史输出，算子接收对应块的切片；增加的内存计入用例预算。baseline/cuSPARSE 仍复用块输出。输出只在分配后初始化一次。

各 backend 每 step（分块时每 step/chunk）一对 CUDA Event，最后累加；整轮末同步一次。event NB 与 ct BN 的 gather 在 start 前；cuSPARSE 在 start/stop 内更新 A/B 描述符指针并调用 cusparseSDDMM(beta=0)，无显式清零；SDDMM 在区间内调用完整算子，baseline 内部清零，prezero 无输出清零。CSV 使用新 timing_mode，防止与未逐步更新指针的历史结果混合。

prezero 数值检查在计时外保存被抽查位置的调用前值；有活动核对本次乘积，无活动核对旧值。另以独立 CPU 序列参考检查整个运行的历史语义，覆盖分块隔离。

## SDDMV 计时桩的实际包裹内容

`benchmark/timing_common.cuh:333` 记录 start，随后循环执行最多 32 次 `step(...)`，第 337 行记录 stop，第 338 行等待 stop。输入生成和 bool→float 转换位于 start 前；同步等待、计时读取、最终数值校验位于 stop 后。CUDA Event 测量该 stream 两个事件间的 GPU 时间，包含该区间内的 kernel/memset 及可能出现的提交空隙，不是 Python/JAX 调用墙钟耗时。

| 方案 | 一次 step 内、位于事件窗口中的操作 |
| --- | --- |
| baseline binary/float | active_count 清零；dweight 清零；活动行压缩；按 macro tile 启动计算 kernel |
| sddmv_prezeroed | active_count 清零；活动行压缩；按 macro tile 启动计算 kernel；无 dweight 清零 |
| hybrid | dweight 清零；status 清零；extract kernel；task_capacity>0 时启动 block kernel |
| hybrid_prezeroed | status 清零；extract kernel；task_capacity>0 时启动 block kernel；无 dweight 清零 |
| cuSPARSE | 更新 A/B 输入描述符地址；调用 cusparseSDDMM(alpha=1,beta=0)；无逐步输出 memset |

BE step 在 `tcsr_timing.cu:62`，hybrid step 在 `hybrid_timing.cu:64`，cuSPARSE step 在 `cusparse_timing.cu:83`。SetValues 是事件记录之间的主机 API 调用，不能把其 CPU 执行时间简单等同为单独 GPU kernel 耗时。

所有方案的输出与 scratch 在 DeviceBuffer 分配时统一初始化一次（`generate.cuh:215`），位于预热和测量之前。prezeroed 后续保留非活动位置旧值，其计时不包含每步全输出初始化。

## 验证

测试同目录后缀命名，包含独立 CPU 参考与真实 CUDA/JAX 调用：

- 全部合法 batch/type 组合，矩形输入、空行和零 nnz。
- 非零哨兵历史值；整行无活动原样保留；活动行结果为零仍覆盖。
- 连续调用：活动→无活动→再次活动；再次活动必须覆盖而非累加旧值。
- K=256/512 的 phase 0 空、phase 1/2/3 首次活动、多 phase 活动以及跨 phase 正负抵消。
- NaN/Inf 旧值在活动行被新有限结果覆盖；非活动槽保留其位模式。
- 127/128/129 行边界，8192/32768 列边界，int32/int64 指针。
- 探针确认变体 launch 不调用输出 memset；检查 native alias 契约拒绝不同输出地址。
- JAX 0.8.0 与当前版本验证 alias/donation；mypy 无错误；有意义的 Python 与 native 主机源码行覆盖率 >90%。

发现问题先加入复现测试，再修正并在此记录原因。

## 开发验证记录

- cuSPARSE 指针更新探针先失败：36 次计时调用的 SetValues 计数为 0，预期 72。修正每步绑定位置；以后同时验证调用次数和所在区间。
- 编译检查发现扩大函数签名的文本替换误及 run；局部修正，接口修改改为限定具体函数。
- 位级保留测试最初比较 CPU 高精度舍入值与 GPU 历史，误报一个 ULP 差异；改为调用前真实 GPU 值的位模式。数值正确性仍独立用高精度乘积验证。

## 验收结果

2026-10-19：58 项驱动测试、342 项原生 GPU 测试通过；prezero 在 JAX 0.8.0 与 0.11.0 各通过 170 项测试，mypy 检查 5 个文件无错误。40 个 O3 隔离 worker（20 direct、20 chunk；25 步、2 轮预热、5 轮测量）均通过。

Python 接口语句/分支覆盖率 100%；benchmark 驱动语句 100%、含分支 99.76%。gcov 主机源码行覆盖率：prezero binary 98.94%、float 98.86%、runner 99.66%；不代表 GPU 指令覆盖率。原有 baseline JAX 互操作测试本轮未启用，其 16 项跳过；新 prezero 在两个版本均实际运行全部 170 项。

记录：`brainevent/_tcsr/dev/sddmm/benchmark/build/validation/prezore/validation-summary.json`；生产验证：`benchmark/runs/validation-20261019-prezero-v3/`。正式大规模网格未运行，性能验收建议使用独占 GPU、多 seed 和显存边界用例。
