# CoCoMac 本地路径修复

日期：2026-10-22。范围：`brainevent/_tcsr/dev/cocomac/utils.py` 中的路径约定。

## 问题与目标

- `ensure_local_packages()` 把 `ROOT.parents[4]` 当作仓库根目录，实际仓库根目录是 `ROOT.parents[3]`。独立启动 `trace.py` 时，在导入本地 BrainEvent 和 BrainTrace 前即报错。
- 默认模型目录指向 `stabilized_model/`，但实际参数包在 `stabilized_model/ori/`；未传 `--model-dir` 的前向与训练入口均无法加载模型。
- `render_plot()` 查找 `ROOT.parent/plot/plot_fig3_5.py`，实际脚本在 `ROOT/plot/`，使默认绘图失败。
- 绘图相关入口仍有迁移前目录：投射矩阵的默认模型包、Fig. 3/5 的默认 runs 根目录及无 `launch.json` 时的模型包回退路径；合成绘图测试也指向旧脚本和旧 `populations.csv`。

修复这些本地路径，不改变连接采样、TCSR 布局、神经元动力学或模型参数。保持显式 `--model-dir` 和 `--cache-dir` 的现有行为。

## 验收

1. 在工作区外、清除 `PYTHONPATH` 的新进程中调用本地包解析，再导入 BrainEvent 与 BrainTrace；两个模块的 `__file__` 必须指向当前工作区。
2. 不传目录时 `load_model()` 应从 `stabilized_model/ori/` 读到完整模型包；显式目录仍有效。
3. 绘图入口应从 `cocomac/plot/` 加载脚本，并把运行目录与既有参数传给 `process_run`。
4. 独立 `trace.py --help` 可启动；路径回归测试使用同级 `utils_test.py`，并在当前 JAX 与 JAX 0.8.0 环境可用时验证。
5. 绘图默认值指向 `cocomac/stabilized_model/ori` 与 `cocomac/runs`；缺少 `launch.json` 时从运行目录相邻的 `ori` 回退；254 群体合成图测试能找到同目录的绘图器和人口表。
6. 独立仿真测试显式传 `--dt-ms 0.1`，使 0.2 ms 时长与当前默认 1 ms 步长解耦。
7. 修改文件通过 mypy；绘图样式需要显式字段类型，独立测试的本地 `trace.py` 导入需要说明与标准库同名。

## 回归缘由

这些路径来自代码迁移后的旧目录层级。第一次验证只覆盖了 `utils.py`，随后运行绘图测试才暴露相邻路径；独立仿真测试还暗含了旧时间步默认值。以后对仓库相对路径扫描调用链和相邻测试，并让时间网格等测试依赖显式化，再用同级回归测试覆盖工作区外的独立进程。
