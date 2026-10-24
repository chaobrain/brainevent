# Hybrid MV baseline 接入与标准测试方案

## 目标与范围

通过 `jax.ffi` 接入用户指定的 `brainevent/_tcsr/dev/new_mv_operator/baseline/binary_csrmv_hybrid.cu`，建立可直接运行的 Python baseline 测试程序。实现采用少量函数和普通循环，以已有 [测试标准](../../brainevent/_tcsr/dev/new_mv_operator/benchmark/测试标准/README.md) 的输入组合为基础。

用户已确认本方案，所有改动位于 `mm-optimizing` 工作分支。

## 文件组织

| 文件 | 职责 |
| --- | --- |
| `brainevent/_tcsr/dev/new_mv_operator/baseline/baseline.py` | 加载 CUDA 文件，准备任务 workspace，提供 JIT 可调用的 FFI 封装 |
| `brainevent/_tcsr/dev/new_mv_operator/baseline/test/baseline_test.py` | 用小规模真实 GPU 用例检查输出、任务边界和 workspace 复用 |
| `brainevent/_tcsr/dev/new_mv_operator/benchmark/run.py` | 枚举标准组合，生成输入，计算参考结果，重复计时，逐条写出 JSONL |
| `brainevent/_tcsr/dev/new_mv_operator/benchmark/test/run_test.py` | 检查组合枚举、Gini 分支、资源预检、参考结果和结果记录 |

复用已有 `benchmark/测试标准/gini_row_lengths.py`。公共函数使用 NumPy 风格 docstring，类型标注覆盖函数接口及主要数据结构。

## FFI 接入

复用 `brainevent._op.load_cuda_file` 完成编译、缓存和目标注册，Python 使用 `jax.ffi.ffi_call` 调用注册目标。这与仓库现有 `_tcsr/binary.py` 的接入方式一致；接口参考 [JAX FFI 文档](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html)。

首版提供 float32 异质及同质权重、bool 事件、int32 列索引和 int32/int64 行指针，对应 CUDA 的 `binary_csrmv_wat_hybrid_hetero_f32_bool` 与 `binary_csrmv_wat_hybrid_homo_f32_bool`。

一次调用包含七个输入：权重、列索引、行指针、事件、任务起点 scratch、任务终点 scratch、状态 scratch。四个输出依次为结果、任务起点、任务终点、状态；scratch 的输入输出别名为 `{4: 1, 5: 2, 6: 3}`。计时循环复用返回的 workspace，并明确处理 JAX donation 后旧数组失效的问题。

任务容量使用主机 int64 行长计算：`sum(ceil(length / 4096) for length > 128)`，与该 CUDA 文件默认宏一致。两个任务数组与行指针同型，状态为两个 int32 元素。容量计算和首次分配在计时前完成，启用 JAX x64 以保留 int64 偏移。

封装信任程序自行生成的合法 CSR，仅保留必要的参数范围、任务容量和内存预算检查。全量输出正确性比较是测试的一部分，放在计时之外。

## 数据与结果

按测试标准 1.1 保留 24 个结构组合（边长 10k/100k/1000k/2000k，密度 1e-4/1e-3/1e-2，Gini 0/0.4），每个结构遍历活动率 1e-4 和 1e-3，共 48 条记录；固定 seed=0、float32 异质 `±1` 权重，使用原有 Gini 生成器，Gini 为零时构造等长行。列索引逐行无放回采样并排序；使用独立随机流生成行序、列、权重和事件。

使用 NumPy 按活动 CSR 行累加参考输出，并检查完整结果。不可实现的结构（`N=10k、d=1e-4、Gini=0.4`，两个活动率共 2 条）明确记录原因。申请大数组前按主机和设备可用内存进行简单预算预检；可配置更小的运行预算，结果中保留该预算，不以缩小矩阵代替原组合。

命令行提供变体（`hybrid` 或 `two_ended`）、尺寸、密度、Gini、活动率、权重模式、预热次数、正式调用次数、内存预算和输出路径。默认枚举完整标准；子集执行明确记录筛选条件。

JSONL 逐条写出用例编号、参数、实际 Gini、nnz、活动边数、任务容量、数据类型、正确性结果、计时原始值及状态。环境和运行配置单独保存；错误或跳过使用文字原因，耗时为空。

## 计时口径

为保持 Python 接入程序简单，采用主机单调时钟：预热 100 次并同步，然后连续正式调用 100 次，最后同步并记录整批总耗时 `jax_total_ms`，`jax_call_ms = jax_total_ms / repeat`。不做 round 重复。

计时字段明确命名为 `jax_call_ms`，包含 JAX 提交、输出分配、设备执行及完成等待。生成输入、参考计算、首次编译、首次 workspace 分配和数据上传在计时之前完成。循环在 Python 中逐次提交真实调用，避免不依赖循环状态的 JIT 循环被优化掉。

此项是用户已确认的实施范围调整：暂不实现标准中的纯设备事件计时、阶段拆分计时和逐项峰值内存采集。因此首版报告只称为 baseline 的正确性与 JAX 完整调用测试，不把 `jax_call_ms` 命名为标准的设备端 `t_total`，也不声称完成全部性能仪器要求。后续变体采用相同计时口径即可先进行可比测试。

## 验证与边界

- 数据侧验证 48 条记录的枚举、2 条已知不可实现分类、零 Gini 和正 Gini、精确 nnz、可复现 seed，以及资源跳过记录。
- 真实 GPU 测试覆盖短行、长行、`128/129` 和 `4096/4097` 分界、零任务容量、空行、全静默与全活动、同质与异质、int32/int64 偏移及跨调用复用。
- 先运行 10k 子集检查接入，再按可用资源执行 1.1 全量清单。
- 对新增 Python 实现与测试运行 mypy，对新增实现的有意义路径取得大于 90% 的覆盖率。JAX 最低支持 0.8.0，当前已有 GPU 测试环境为 JAX/jaxlib 0.11.0；最低版本验证与实际 GPU 验证分开记录。
- 若发现 baseline 缺陷，先补可复现测试，再做必要的最小修复并说明原因。

## 实现中的修正

GPU 回归测试发现原 CUDA launcher 在 `nnz=0` 时于状态清零之前返回，导致输出已经为零但任务计数和溢出标记仍是上次调用的值。先用非零状态构造失败用例，再将同质和异质 launcher 的提前返回移动到状态清零之后。短行 scatter、重行分块及调度参数保持原算法；回归测试同时覆盖零行、零列和普通维度的空结构。今后的空输入验收同时检查输出与状态缓冲区。

测试使用稳定的 `jax.config.update("jax_enable_x64", ...)`，避免依赖 JAX 0.11 已移除的实验性入口。覆盖率按文件目录采集，避免通过模块名称解析覆盖范围时提前导入 JAX 扩展。这些问题分别由真实测试失败暴露并修正。

## 验证结果

2026-10-14，在物理 GPU 3（NVIDIA RTX PRO 6000 Blackwell Server Edition，97887 MiB）验证，驱动为 580.95.05，nvcc 为 13.0.88。实际执行命令使用 `CUDA_VISIBLE_DEVICES=3` 和 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。

- JAX/jaxlib 0.11.0：40 项测试通过，无跳过；新增 Python 实现覆盖率 99.56%，其中 FFI 封装 100%，runner 99.47%。
- JAX/jaxlib 0.8.0 及对应 CUDA 13 插件：同一组 40 项真实 GPU 测试通过，无跳过。临时依赖安装于 `/tmp/brainevent-mv-jax080`，通过 `PYTHONPATH` 使用，未改动原环境。测试期间宿主环境的 0.11 CUDA 插件被发现并按版本规则忽略。
- 四个新增实现及测试 Python 文件的 mypy 检查无错误；`git diff --check` 通过。

运行说明见 [benchmark/README.md](../../brainevent/_tcsr/dev/new_mv_operator/benchmark/README.md)。
