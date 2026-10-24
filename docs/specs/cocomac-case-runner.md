# CoCoMac case 组合调用

日期：2026-10-22。工作分支：`mm-optimizing`。

## 目标

参照 `new_tile_operator/benchmark/benchmark_binary_csr_matmat.py`，在 `case.py` 顶部直接填 `CASE_LIST`，通过 `itertools.product` 展开笛卡尔积。脚本不使用 `argparse` 或其他 CLI parser。运行 `python case.py` 即顺序执行所有组合，`run_all_cases(cases=..., run_directory=...)` 可供测试或 Python 调用。

每个组合都在两个独立 worker 中调用本地实现函数：

1. 训练：`braintrace_rnn_split_timing.run()`，传入所选 n 模型、`runs`、`warmup`、`dt_ms`、`timing_mode` 等。
2. 纯前向：`trace.run()`，从同一个原始模型包重新初始化并写 `forward/simulation/spikes/`、`statistics.npz`、`summary.json`。训练权重不持久化，也不会进入前向。

## 配置与输出

- `CASE_LIST` 的每个对象包含列表字段 `SCALE_LIST`、`RUNS_LIST`、`WARMUP_LIST`、`DT_MS_LIST`、`TIMING_MODE_LIST`；可加 `STEPS_LIST`、`DURATION_MS_LIST` 等 `DEFAULT_CASE` 中的字段。不同列表之间取笛卡尔积。
- `DEFAULT_CASE` 放固定参数：训练 steps、前向 duration/burn-in、5 Hz 目标、学习率、随机种子、GPU 和缓存目录。默认示例只选择 `n0.05`，避免无意启动所有大图；用户编辑列表可选 `k0.39` 的任意 n。
- 在启动 worker 前验证组合和模型包；输出目录只新建不覆盖。每个 `case_NNN` 保存完整配置、训练 JSON 与日志、前向日志和脉冲记录。`summary.csv` 逐项写入状态。一个阶段失败后仍运行另一个，后续组合继续。
- worker 使用 `spawn` 独立进程，在各自进程内导入本地 `brainevent` 与 `braintrace`，并直接调用 `run()`。GPU 可见性在导入 JAX 之前设置。

## 验收

1. 测试验证列表笛卡尔积、输入检查、目录布局、结果与日志分流、失败后继续，以及前向记录存在且完成。
2. 用小型真实模型分别验证 `gradient_only` 和 `with_sgd` 的训练 worker，以及独立前向 worker；不启动完整 254 群体图。
3. 同级 `case_test.py`、mypy、覆盖率超过 90%。

## 修正缘由

首版将配置当作 CLI 参数转发给两个入口，偏离了参考脚本的列表组合和 worker 直接调用结构。新实现明确每个组合和阶段，测试在实现层验证 `run()` 调用，避免仅验证命令拼接。
