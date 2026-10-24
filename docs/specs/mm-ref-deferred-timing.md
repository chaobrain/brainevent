# MM ref：逐步计时，整轮同步

日期：2026-10-16。工作分支：`mm-optimizing`。

后续状态：框架已迁至 `new_tile_operator/baseline/`，本规格保留先前计时改动的历史记录。
当前实现与运行命令见 [迁移规格](mm-baseline-benchmark-migration.md)。

## 范围

用户已批准将刚讨论的计时方式先应用到 `new_tile_operator/ref`，并评估修改面。
实现集中在 `benchmark_binary_csr_matmat_native.cu`；新增同目录
`benchmark_binary_csr_matmat_native_test.py`。Python 驱动、C ABI、CSV、输入公式、
算子代码和现有清零边界不变。这里记录的时间仍是各计算区间之和，不是整轮墙钟时间。

## 执行协议

1. 普通路径为每个时间步创建独立 start/stop 事件；cuSPARSE 分块路径为每个
   `(step, chunk)` 创建独立事件。事件在预热前创建，各轮读取完成后复用。
2. 沿用原有默认 stream。输入整理、必要的输出初始化、事件记录和算子执行
   都按原顺序提交，输入缓冲仍仅容纳一步（分块路径容纳一块）。
3. 删除逐步/逐块的输入同步、stop 同步和耗时读取。整轮提交完成后同步该 stream
   一次，再读取所有事件对并求和；预热和正式测量沿用同一循环。
4. 原有 cuSPARSE 输出清零位于 start 前；自定义算子的原有清零位置保持不变。
5. grid_y / largescale / uint16_local 原本逐步拷回并检查状态：改为在 stop 后
   把每步状态异步保存到设备侧小型历史数组，轮末一次拷回并逐步检查。不能只检查
   最后一步，因为后续算子会重置状态。记录量为 `steps * batch * 2 * sizeof(int32_t)`。
6. cuSPARSE 分块描述符和最大工作区在运行前创建/分配；块的 CSR 和输入仍逐块
   生成到复用缓冲。循环中不再创建/销毁描述符，也不分配/释放工作区。
7. 中途提交失败时，资源清理先等待已提交工作完成，防止描述符及缓冲提前释放。

## 测试与验收

- 先添加测试，复现旧代码逐步同步；再修改计时循环使测试通过。
- GPU 测试以 CUDA API 计数验证事件独立性、每轮统一同步和耗时读取次数。
- 小规模覆盖所有后端、f32/f64 支持组合、多个步数、零/全活动输入、循环池回绕、
  多轮预热、输出覆盖，以及强制小阈值触发的 cuSPARSE 分块（含不满块尾部）。
- 输出与独立 NumPy 参考对比；统计量应满足总时间、均值及极值关系。
- 验证早期步骤的错误状态不会被最后一步正常状态覆盖；测试部分资源创建失败及清理。
- 新增 Python 测试有必要类型注解，mypy 通过；CUDA 主机侧修改行争取超过 90%
  执行覆盖率，未覆盖项按实际证据报告。此次不改变 JAX 接口。

## 依赖与限制

ref 现有副本缺失若干 kernel 头文件。验证通过 `MM_REF_INCLUDE_DIR` 指定本机已有
原始 MM 源码目录，仅作为编译 include 路径，不复制或修改外部 kernel。
原 Python 驱动依赖文件缺失的问题不并入本次计时改动。

原理依据：同一 CUDA stream 内按顺序执行，下一步输入覆盖排在上一步算子后面；
每个事件只保留最近一次记录，因此同轮中的各步骤必须使用不同事件对象。

## 验证结果

- 修改前：7 步、预热 1 轮、测量 2 轮，回归测试记录到 21 次事件同步，
  与目标的 3 次轮末同步不符，测试失败。
- 修改后：GPU 0 上 60 个测试全部通过（CUDA 13.0）。相同用例记录到 3 次同步、
  21 次耗时读取；事件在各轮之间复用。8 种 cuSPARSE 配置的直接及分块执行均已覆盖。
- 独立 NumPy 参考检查了最终完整输出；故障注入覆盖事件创建、事件记录、耗时读取、
  轮末同步失败，以及第一步异常、后续步骤正常的状态历史。
- mypy：新增测试文件检查无错误。
- gcov：修改涉及的可执行 CUDA 主机侧代码行覆盖 75/77（97.40%）；未覆盖两处
  地址空间溢出保护。整个 native 文件主机侧行覆盖 549/657（83.56%）。该测量
  不代表 GPU kernel 的设备代码覆盖率；kernel 未修改。
- 生产代码仅改一个 `.cu`，新增 110 行、删除 244 行，净减少 134 行；另新增本规格
  和同目录测试文件。现有 `.gitignore` 的 `dev/` 规则仍然忽略 ref 目录。

验证命令（Python 环境需有 pytest、NumPy，仓库导入依赖需兼容）：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
MM_REF_INCLUDE_DIR='/path/to/original/mm' MM_REF_COVERAGE=1 \
python -m pytest brainevent/_tcsr/dev/new_tile_operator/ref/benchmark_binary_csr_matmat_native_test.py -q

python -m mypy --follow-imports=silent --ignore-missing-imports \
  brainevent/_tcsr/dev/new_tile_operator/ref/benchmark_binary_csr_matmat_native_test.py
```

`MM_REF_COVERAGE=1` 会在 pytest 临时构建目录保留 gcov 文件；退出测试进程后，
在该目录执行 `gcov --json-format --branch-probabilities probe.gcno` 可读取主机侧覆盖数据。

后续大规模测试建议：使用实际 batch 和网络尺寸分别比较旧版与新版的计时稳定性；
另外记录整轮墙钟时间，用于观察逐步主机同步移除后的影响，不将它与算子区间和混用。
