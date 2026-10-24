# TCSR SDDMV（batch=1）benchmark 按 ref 重组

> 事件类型与实现划分已由 [hybrid-sddmv-float.md](hybrid-sddmv-float.md) 修改：实现名只保留方案名，`event_type ∈ {bool, float}` 选择 binary / float 算子；float 事件为正的 U[0.5,1.5)。本文第 3 节的 `float` 取值与第 4 节的 cuSPARSE 两种输入以该文为准。

日期：2026-10-16。分支：`mm-optimizing`。取代 [sddmv-cusparse-benchmark.md](sddmv-cusparse-benchmark.md) 第 4 节（输入与计时）和第 5 节之后的驱动、输出部分；方案、step 内容（第 3 节）不变。参照 `brainevent/_tcsr/dev/sddmv/ref/benchmark_binary_csr_vecmat{.py,_native.cu}`。

## 1 分工

| 层 | 职责 |
| --- | --- |
| C++（`benchmark/generate.cuh` + 各计时 `.cu`） | 生成网络、行内排序、构建 CSR 与 TCSR 元数据、计算 hybrid task_capacity、按 tile 生成事件与 ct、计时、统计 |
| Python（`benchmark/run.py`） | `CASE_LIST`、seed 循环、静态过滤、nvcc 编译、ctypes 调用、CSV / summary / log / manifest |

Python 只向计时库传标量：`n, n_conn, seed, firing_rate_hz, dt_ms, duration_steps, warmup, runs, event_kind`。

## 2 生成（与 ref 逐位一致的部分）

```
mix64(v):  v += 0x9e3779b97f4a7c15
           v = (v ^ v>>30) * 0xbf58476d1ce4e5b9
           v = (v ^ v>>27) * 0x94d049bb133111eb
           return v ^ v>>31                                  // uint64 回绕
random_unit(v) = float(mix64(v) >> 40) * 2^-24               // [0, 1)，24 位
```

- **规模**：`n = int(scale * 1000)`，`n_conn = max(1, round(n * density_mat))`，`nnz = n * n_conn`，行长均匀，`indptr[row] = row * n_conn`。
- **列号**：`key = seed ^ row*0xd6e8feb86659fd93 ^ k*0xa5a3564e27f8862d`，`col = mix64(key) % n`。有放回，保留重复列。
- **排序**：TCSR 要求行内有序，用 `cub::DeviceSegmentedSort::SortKeys` 行内升序。只改槽序，不改矩阵；所有方案与 cuSPARSE 用同一份排序后的 CSR。
- **事件**：`key = kSpikeSeed ^ step*0x9e3779b97f4a7c15 ^ row*0xbf58476d1ce4e5b9`，`active = random_unit(key) < density`，`kSpikeSeed = 0x4f1bbcdc6766c4db` 固定，不随网络 seed 变。`density = Hz * dt_ms / 1000`（float32），`dt_ms = 0.1`。`step` 为全局步号 `0..duration_steps-1`，每轮相同。

## 3 生成（SDDMV 新增，ref 中没有）

- **ct**：稠密，与 ref 的 weight 同分布 U[-2,2)。`key = kCtSeed ^ step*0x9e3779b97f4a7c15 ^ col*0xbf58476d1ce4e5b9`，`ct = random_unit(key)*4 - 2`，`kCtSeed = 0x2545f4914f6cdd1d` 固定（与事件一样属于输入，不随网络 seed 变），每步不同。
- **事件 kind**：`binary_bool`（bool）、`binary_f32`（0/1）、`float`。`float` 在活跃位置取 `±U[0.5,1.5)`：`h = mix64(key ^ 0x632be59bd9b4e019)`，`u = (h>>40)*2^-24`，符号取 `h & 1`；非活跃为 0。
- **TCSR 元数据**（与 `brainevent/_tcsr/preprocess.py::build_tcsr_structure` 定义相同）：`tcsr_indptr` int64；`local_targets = col & 8191`（uint16）；`tile_offsets[row, t] = lower_bound(row, min(t*8192, n))`，形状 `[n, ceil(n/8192)+1]`，int32。
- **hybrid task_capacity**：`n_conn > 128` 时 `n * ceil(n_conn/4096)`，否则 0。

## 4 计时（照 ref）

```
dW、scratch：分配时清零一次，之后不重置
cudaDeviceSynchronize
for run in warmup + runs:
    for step_begin in 0, 32, 64, ... < duration_steps:
        填充本 tile 的事件和 ct [≤32, n]                     ← 窗口外
        cudaEventRecord(start)
        for step in tile: 调用方案入口                        ← 窗口内
        cudaEventRecord(stop) → 同步 → 累加
    每轮耗时 = 各 tile 窗口之和
```

- 默认 `DURATION_STEPS = 100`、`WARMUP = 2`、`RUNS = 5`。
- 统计在 C++ 中计算（ref 的 `write_stats`，总体标准差）：`warmup_total_s, timed_total_s, timed_mean_ms, timed_std_ms, timed_min_ms, timed_max_ms`；另返回 `nnz, task_capacity, active_per_step_mean`（第一轮各步活跃数均值）、`output_l1`（最终 dW 的 L1，窗口外计算）。
- 每步 step 的内容与原规格第 3 节相同。cuSPARSE 有两种输入：`cusparse_binary_f32`（0/1）、`cusparse_float`（带符号值）。

## 5 编译

- 显式调用 `nvcc`：`-O3 -std=c++17 -shared -Xcompiler -fPIC`、`-gencode`（kernix 解析的架构）、`-ccbin`、`-I`（brainevent include、XLA FFI include、CUDA include）、方案宏 `-D...`、`-include <方案 .cu>`、`-lcusparse`。
- 每个 (scheme, family) 一个 `.so`，放在计时源码旁边：`benchmark/<timing>_<scheme>_<family>.so`；另有生成器库 `benchmark/generate_host.so`。
- 与 ref 相同，按 mtime 判断：`.so` 不存在或早于任一源码（计时文件、`timing_common.cuh`、`generate.cuh`、方案 `.cu`）时重编。`.so` 已被 `.gitignore` 的 `*.so` 忽略。

## 6 驱动与输出（照 ref）

- `run.py` 顶部：`gpu_id`、`RNG_SEED_LIST`、`CASE_LIST`；每个 case 含 `REPEAT_COUNT`（须等于 seed 个数）、`SCALE_LIST`、`DENSITY_MAT_LIST`、`DENSITY_VEC_LIST`（Hz）、`IMPL_LIST`。不用的 case 放进字符串保留。无 CLI。
- **静态过滤**：`nnz <= INT32_MAX`（cuSPARSE 与 hybrid 使用 int32 偏移，不做 chunk），且估算设备占用 `< MATRIX_MEMORY_LIMIT_BYTES`；不满足的组合不运行、不写记录，只计入 manifest 和日志。
- **目录**：`results/<时间戳>/manifest.json`；`case_XXX/case_<k>.csv`（第 k 个 seed）、`summary.csv`（跨 seed 聚合 `timed_mean_s` 的 mean/std/min/max 与成功、失败次数）、`run.log`（stdout/stderr 重定向）。每行写完 flush + fsync。
- **CSV 列**：`impl, scheme, event_kind, clears_dw_in_kernel, scale, n_neuron, density_mat, firing_rate_hz, density_vec, dt_ms, seed, n_conn, nnz, task_capacity, active_per_step_mean, duration_steps, warmup_runs, warmup_total_s, timed_runs, timed_total_s, timed_mean_s, timed_mean_step_ms, timed_std_ms, timed_min_ms, timed_max_ms, output_l1, status, error`；`status ∈ {timed, failed_runtime}`。
- **manifest**：计时口径、方案表、`CASE_LIST`、seeds、常数、nvcc 命令、设备与版本、源码 sha256、每个 case 的运行与过滤计数、`status ∈ {running, completed, interrupted}`。

## 7 Python 接口（供测试与 sweep 使用）

- `run.generate(n, n_conn, seed, firing_rate_hz, *, step_begin=0, steps=1, dt_ms=DT_MS)` → `(Structure, inputs)`；数据全部来自 `generate_host.so`，`inputs` 为 `[steps, n]` 的 `binary_bool / binary_f32 / float / ct`。
- `run.run_impl(impl, n, n_conn, seed, firing_rate_hz, *, duration_steps, warmup, runs, dt_ms, copy_dweight)` → `(stats, dweight | None)`。
- 删除：`build_structure`、`build_dense`、`build_steps`、`check_slots`、`reference`、`_check`、动态内存估算、JSONL、argparse。

## 8 调用方迁移

- 4 个方案测试、`pytest_for_other_operator.py`、`block_sweep/sweep.py` 及其测试改用 `run.generate`；事件率用 Hz 表示（`dt = 0.1 ms`：10000 Hz = 1，1 Hz = 1e-4）；活跃数为二项分布。
- `block_sweep` 的采样校验函数从 run.py 移入 `sweep.py`。
- `ncu/20261015_150430_*` 是冻结的采集记录（manifest 固定了旧 run.py 的 sha256），不迁移工作负载：把旧 numpy 生成函数原样复制进同目录的 `legacy_workload.py`，脚本改为从它导入，使其仍可运行并复现采集时的工作负载。

## 9 测试

- 生成器与 ref 公式逐位一致：用 numpy uint64 独立实现第 2、3 节的公式，小规模下比对列号（排序后）、事件、ct、float 值。
- C++ 构建的 TCSR 元数据与 `TCSR.from_sorted_csr` 逐位一致。
- 每个实现：`duration_steps` 跨 tile 边界（如 34）时，最终 dW 等于各步覆盖后的 numpy 结果；最后一步与方案的 JAX 算子一致。
- 统计量自洽；参数错误变为 `RuntimeError`；静态过滤、CSV、summary、log、manifest 写出正确；mtime 触发重编。
