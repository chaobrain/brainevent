# Delay 三后端接入

2026-10-23，按用户批准计划在 `mm-optimizing` worktree 分支实施。

## 初始化与两套输入

保留 `delay_tcsr(source, edge_delays, *, buffer_slots=None, binary_backend=None)`。
先校验 CSR，再按显式参数、当前平台可用的全局配置、primitive 默认值确定并固定
backend。GPU 默认 cuda_raw，CPU/TPU 默认 jax；显式不可用后端报错，无运行时替换。

共享 NamedTuple 定义放在 binary_delay：

- `_DelayCSRInputs(data, indices, indptr, delay_count, row_task_count)`：CUDA/JAX。
- `_SplitCSR(sub_indptr, sub_base, sub_indices, sub_values)`：cuSPARSE。
- `_DelayWorkspace(task_capacity, task_begin, task_end, task_out_off, status)`：
  CUDA/JAX，任务阈值 128、任务大小 4096，数组类型沿用现有约定。
- `_CusparseWorkspace(scratch)`：float32[N]，cuSPARSE 各段共用并返回复用。

保留 CPU 上的 delay 元数据转换和 CSR 原始边顺序。cuSPARSE 初始化时调用一次
`split_region_delay(data, indices, indptr, delay_count)`，包装现有 CUDA split；
同质权重展开，整数输出按总 nnz 选择一致的 int32/int64，保留权重单位。
预先检查每段 nnz <= M*N；需要 int64 时检查 JAX x64。实例保存 `_inputs` 和对应
`_workspace`，以及全零 float32 buffer[L,N]、int32 head[1]。结构、权重、backend
固定，更换时重建实例。

## 一步接口与后端

```python
binary_csrmv_delay_step(
    inputs, events, *, shape, delay_off, buffer, head, workspace, backend,
) -> (y, new_buffer, new_head, new_workspace)
```

类只处理 NumPy/JAX/BinaryArray 的向量输入并保存返回状态。binary 统一进行
events > 0 归一化、权重单位处理和显式状态 primitive 分派。旧普通 MV 包装、
transpose/mirror/permutation、普通 hybrid 调优依赖移除，不保留旧接口兼容分支。
primitive 不注册 AD/vmap 规则；类成员更新拒绝外层 jit/scan，显式状态底层可 JIT。

- cuda_raw：调用现有 `region_delay_f32`，完整返回 buffer 和四个任务数组。
- cusparse：输入 split 四数组，调用 `region_delay_cusparse_f32`，CUDA 内循环 K 段，
  返回 buffer/scratch；保留临时 StepPlan，不增加跨步 plan 缓存或直接写 ring。
- jax：根据 indptr/delay_count 按原边序展开源行和延迟段映射，静态 nnz，scatter-add
  到 ring；偏移类型跟随 delay_off，避免 int32 加法中间值溢出；原 workspace 返回。

所有后端先投射，再消费并清零当前槽，最后 head=(head+1)%L。GPU 消费使用已有
`consume_region_delay_f32`，JAX 使用数组索引。FFI 明确 buffer/workspace 别名。
GPU head 保留为设备数组，不为状态处理引入每步 CPU 同步。status 原样返回供检查。
空图/零维度可跳过投射，但仍消费已有贡献并推进 head。buffer 保持 float32 数值，
y 恢复权重单位。CUDA 模块按需加载缓存，cuSPARSE 链接 -lcusparse，禁用 CUDA graph。

## 验证

测试保留在同级 binary_delay_test.py/delay_main_test.py。独立 NumPy 参考直接按
原始边和延迟计算，不使用被测 split/JAX 实现生成期望值。

- 后端优先级、固定选择、两套输入/workspace 匹配、split 一次且只用于 cuSPARSE。
- split 四数组、同质展开、异质权重、空行/空图、int32/int64、x64 禁用。
- 三后端逐步输出与剩余 ring：零/非连续延迟、单槽/多次回绕、预存贡献、负值/
  零/NaN 事件、重复目标、单位、128/129 与 4096/4097 边界。
- 真实 CUDA/cuSPARSE FFI、完整 workspace 流转、显式状态底层 JIT、三类左乘；
  拒绝矩阵、AD、vmap 和外层成员状态 JIT。替身仅验证契约和调用次数。
- 修改模块有效 Python 行覆盖率 >90%，实现与测试 mypy 零错误，NumPy 文档/doctest。
- 当前 JAX 0.11.0 CPU 和 GPU；隔离 JAX 0.8.0 CPU，不修改共享依赖环境。

## 实施结果

已接通三后端，沿用现有三个 CUDA 源文件；本轮未修改 CUDA 算法或 StepPlan 生命周期。
原始 CSR 的公开字段保持原引用，初始化时另行准备 `_inputs` 中的设备数组。
正值判断放在 binary 的各 primitive kernel 内，避免在 primitive 外先转换 bool
导致事件梯度被静默视为零；回归测试验证事件和权重的 AD 均明确拒绝。

2026-10-23 验证结果（仅运行本次两个同级测试模块）：

| 环境 / 检查 | 结果 |
| --- | --- |
| JAX 0.11.0，真实 GPU | 156 passed，包含 CUDA hybrid、cuSPARSE 和 JAX 数值对照 |
| JAX 0.11.0，CPU | 123 passed，33 GPU 项跳过 |
| 隔离 JAX 0.8.0，CPU | 123 passed，33 GPU 项跳过 |
| Python 行覆盖率 | binary_delay 98.40%，delay_main 99.07%，合计 98.64% |
| mypy，实现及测试四文件 | Success: no issues found |
| doctest | 13 passed |

覆盖率未覆盖的四行是无效 split 维度检查及需要超大数组才触发的三个容量保护分支。
JAX 0.8.0 环境继承的 CUDA 0.11 插件会报告版本不兼容并跳过加载；本次仅验证其
CPU/JAX 路径，不声称验证了 0.8.0 GPU。未进行全仓库测试或性能基准测试。

实际边界测试包括空图/零维度、单槽、非连续延迟、多次回绕、已有 ring 贡献、
负值/零/NaN 事件、重复目标、同质/异质/带单位权重、任务阈值和任务切块边界、
int32 偏移中间溢出、int64 split ABI，以及完整 workspace 连续返回。
若后续修改 StepPlan 生命周期，建议另加跨实例交替调用和资源释放测试。

## 辅助函数作用域整理

按用户要求，将仅属于一个上层操作的辅助函数收进其作用域：

- `_wrapped_offsets` 嵌入 `_binary_csrmv_jax_kernel`，保留 int32 防溢出算法。
- `_resolve_backend` 嵌入 `delay_tcsr.__init__`；split 只检查 cuSPARSE 是否在
  当前平台可用，不再借用包含全局配置和默认值选择的初始化逻辑。
- `_validate_backend`、`_check`、`_load`、`_consume_gpu` 有多个实际调用者，
  继续作为共享函数；CUDA 加载缓存仍保持模块级生命周期。

本次仅调整 Python 作用域，接口、数值计算、状态流转和后端选择规则不变。
此前测试直接调用局部辅助函数，导致内部整理牵动测试；改为通过构造函数和
JAX kernel 验证选择优先级、平台限制及大偏移防溢出。沿用当前测试文件位置，
验证真实 GPU、当前 CPU、JAX 0.8 CPU、覆盖率和 mypy。

整理后验证：真实 GPU 156 passed；JAX 0.11.0 / 0.8.0 CPU 分别 123 passed、
33 GPU 项跳过；实现及测试四文件 mypy 无错误。合计覆盖率 98.31%
（binary_delay 97.70%，delay_main 99.18%）；新增独立的 split 平台拒绝分支未在
本轮 GPU 覆盖率运行中触发。后续继续从上层入口测试局部逻辑，重点覆盖显式/
全局/默认选择顺序、不可用平台和 int32 回绕偏移。
