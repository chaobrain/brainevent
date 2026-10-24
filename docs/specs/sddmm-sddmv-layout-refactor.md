# SDDMM benchmark 按 SDDMV 结构精简

日期：2026-10-19。分支：`mm-optimizing`。状态：实现与验收完成。

## 问题与目标

重构前 `benchmark_sddmm_native.cu` 共 732 行，混合 CSR 生成、后端资源、运行时分派、输入验证、测试专用求值入口和计时。采用 SDDMV 的组织方式：数据生成、公共计时循环、TCSR step、cuSPARSE step 分开；删除跨后端的通用 Operator，避免每步重新判断后端。

## 文件布局

```text
benchmark/
  run.py                    # 用例、按需编译、隔离 worker、CSV
  generate.cuh              # CSR/TCSR 生成、设备缓冲、双池和 gather
  timing_common.cuh         # 参数、统计、验证、唯一的 run_steps 循环
  tcsr_timing.cu            # baseline/prezero 的资源准备与直接算子调用
  cusparse_timing.cu        # cuSPARSE 资源准备、SetValues 与 SDDMM
  benchmark_sddmm.py        # 旧命令的薄兼容入口
  *_test.py                 # 同目录测试，保留有意义的数值与计时探针
```

迁移完成后移除旧 native 大文件及被 generate.cuh 接替的 circular_inputs.cuh，不保留两套实现。已有结果保留。测试专用的自定义输入 C ABI 随测试探针编译，不进入正式动态库。

## 调用方式

参考 SDDMV 的 tcsr_timing.cu/cusparse_timing.cu：每个后端准备一次资源，提供短小的 step，再调用公共 run_steps。按方案/算子家族构建独立库，使用统一 C ABI；kind/dtype 选择在进入计时循环前完成，保留现有全部合法组合。

```text
构建结构、双池、独立 event/ct 目标缓冲、输出和 scratch
每轮每 step：
  gather ct
  每行块：准备块结构，gather event
    start → step(event, ct, block) → stop
  整轮末同步、读取各 step/chunk 耗时并累加
```

TCSR step 调用所选的原始 CUDA 入口；cuSPARSE step 只更新 A/B 指针并调用 SDDMM(beta=0)。稳定的 Tensor 包装在准备阶段建立，step 只处理本次输入地址与必要的块视图。

## 保留的已确认行为

- 逐 step/chunk 计时，整轮同步一次；不采用 SDDMV 的 32 步合并窗口。
- event NB、ct BN 从独立循环池实际复制到独立目标缓冲；gather 在计时外。
- baseline 内部清零；prezero binary/float 活动覆盖（包括零结果），整行无活动保留。
- prezero 分块使用全局输出的独立切片，内存计入预算。
- 保留 direct/chunk、已有 batch/dtype、随机输入分布、配置/CSV 语义及逐用例子进程。
- 该重构不修改算子算法与 JAX 接口，不迁入 SDDMV 的 hybrid 算子。

## 验收

先迁移现有回归测试，再以独立 CPU 参考检查数值与跨步历史；探针继续核对 gather 边界、SetValues 次数、内部清零次数、逐步计时累加和分块隔离。测试后缀命名、覆盖率 >90%、mypy 无错误；保留 JAX >=0.8 兼容性。README 只说明当前结构与运行方式，历史验收详情链接到既有记录。

本次纠正：此前为了集中复用而让一个 Operator 同时承担多种后端和测试职责。重构以明确的 step 接口约束职责，不继续给该类增加分支或配置项。

## 完成情况

已移除旧 native 大文件和 circular_inputs.cuh；公共计时入口为 timing_common.cuh::run_steps，TCSR/cuSPARSE 文件分别为 88/73 行。旧命令保留 6 行转发入口。五个后端库按需构建，正式库不导出自定义输入求值或测试探针。数据生成、输入分布、算子与输出语义保持一致。

验收通过：61 项驱动/兼容入口测试、342 项原生算子与计时测试、22 项独立复制测试；prezero 在 JAX 0.8.0/0.11.0 各 170 项通过。mypy 检查 6 文件无错误。Python 语句覆盖率 100%、含分支 99.78%；host gcov：generate 100%、公共计时 98.55%、TCSR 97.92%、cuSPARSE 100%，不代表 GPU 指令覆盖率。

正式 O3 的 40 个 worker 全部通过（direct/chunk 各 20，默认 25 步、2 轮预热、5 轮测量）；manifest 的源码摘要与当前实现一致。记录：`brainevent/_tcsr/dev/sddmm/benchmark/build/validation/layout-refactor/validation-summary.json`；运行结果：`benchmark/runs/validation-20261019-sddmv-layout/`。正式大规模性能网格未运行。
