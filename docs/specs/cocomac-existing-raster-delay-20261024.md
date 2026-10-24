# 使用现有 plot_raster.py 绘制 D/A

用户指定调用 `plot/plot_raster.py`，使用该脚本原有全脑单面板样式，
输出到 D、A 各自现有的 `result/figures/`。

## 必要兼容

- 保留原有按脑区比例、无放回随机抽样；5000 个，seed=20260926。
- 两份录制都是完整 20 s，包含前 10 s 预热，横轴单位 ms。
- 原脚本只读取 graph/complete.json，但延迟图保存 graph/metadata.json。
  新增对已知 delay-csr-eight-buckets-v1 格式的识别，验证模型哈希、
  双轴神经元数量、群体顺序和 dt，保留原无延迟身份核验。
- 标题根据 simulation/config.json 的 transmission_delay（fallback launch）
  使用 Delayed、No-delay 或 Recorded，不能给延迟录制标注 No-delay。
- 不修改抽样、原始数据或绘图样式；不运行模拟。

## 验证

先用同级 plot_raster_test.py 复现缺少 complete.json 和错误标题，再修复。
保留既有 test/plot_raster_test.py 并运行其测试以验证旧路径；新增测试均同级。
覆盖延迟正确身份以及哈希、尺寸、群体表、dt 错误，标题的延迟/无延迟/未知分支。
组合测试语句覆盖率 >90%，mypy 无错误；不引入 JAX 依赖。
真实结果核验 5000 个独立编号、相同种子相同编号、32 脑区、完整时间范围、
抽样 spike 数与原始总计数，并检查 SVG 标签。

## 产物

各 `result/figures/raster_5000_proportional.svg` 及同名前缀的 sampled_neurons.csv、
area_ranges.csv、summary.json。核验日志保存在各案例 verification/plot_raster/。
