# 延迟算子的调用与正确性检查

`operator/region_delay.py` 集中提供逐边延迟输入准备和三个 CUDA 算子的
调用封装，供 `operator/test/` 的正确性测试使用。benchmark 复用其中的
CPU 输入准备函数，计时循环直接调用原生 CUDA。随机输入生成、workspace
分配、事件列循环、消费及 head 推进由调用者负责。

| Python 调用 | CUDA 文件 | 输出 |
| --- | --- | --- |
| `split_region_delay` | `region_delay_split.cu` | `SplitCSR` 四份设备数组 |
| `project_region_delay` | `binary_csrmv_hybrid_region_delay.cu` | 更新的 ring、任务数组和 status |
| `project_region_delay_cusparse` | `region_delay_cusparse.cu` | 更新的 ring 和 scratch |

源文件均在 `../operator/`。原生 benchmark 直接调用 CUDA，详见
[benchmark 说明](benchmark.md)；本次精简的范围与验收见
[Python 封装规格](python-test-wrappers.md)。

## 输入与串联

设源数量 M、目标数量 N、延迟段数 K、ring 槽数 L、事件列数 T。
`region_delay.py` 中的 `prepare_delay_inputs` 校验原始 CSR 和逐边延迟，
按每行 `(delay, column)` 排序，返回 `DelayInputs`；同时计算段计数、
ring 偏移和任务容量，将完整事件矩阵转成按列连续的 float32 二值数组。
`validate_delay_steps` 校验延迟集合。原 `delay_inputs.py` 已删除，相关
调用统一从 `region_delay.py` 导入。T 与 K 独立，准备过程不调用 CUDA。

| 输入 | 形状与类型 |
| --- | --- |
| `weights` | float32 `[nnz]` 或 `[1]` |
| `indices / indptr` | int32/int64 `[nnz]` / `[M+1]` |
| `delay_count` | `[M,K]`，同 indptr dtype |
| `delay_off` | int32/int64 `[K]`，值为 delay_steps × N |
| `row_task_count` | int32 `[M]`；按 CUDA 默认阈值 128、任务大小 4096 准备 |
| `events[:,t]` | bool/float32 `[M]`，非零值表示活动，不缩放权重 |
| `head / buffer` | int32 `[1]` / float32 `[L,N]` |

hybrid 接收上述数组和 `HybridWorkspace(task_begin, task_end,
task_out_off, status)`；前三个 workspace 数组长度为 capacity，前两个
与 indptr 同 dtype，第三个与 delay_off 同 dtype，status 为 int32 `[2]`。
输出 status[1] 非零表示该步失败，每次调用会重置 status。

cuSPARSE 初始化只调用一次 split，保留全部返回数组：

| `SplitCSR` 字段 | 形状 | 用途 |
| --- | --- | --- |
| `sub_indptr` | `[K,M+1]` | 各子矩阵从零开始的行指针 |
| `sub_base` | `[K+1]` | packed 区间边界，末项等于 nnz |
| `sub_indices` | `[nnz]` | 按段打包的列号 |
| `sub_values` | float32 `[nnz]` | 同质权重也展开到每条边 |

前三份整数数组统一使用 int32；总 nnz 超过 int32 范围时使用 int64，
需要启用 JAX x64。之后每步把整个 `SplitCSR` 原样交给 cuSPARSE：

```python
split = split_region_delay(weights, indices, indptr, delay_count)
scratch = jnp.zeros(N, jnp.float32)
for t in range(T):
    buffer, scratch = project_region_delay_cusparse(
        split, events[:, t], delay_off, head, buffer, scratch)
    # 测试读取当前槽、清零，并将 head 推进一次。
```

一次 Python 调用对应一次 FFI 调用。CUDA 内部用
`sub_indptr + k*(M+1)` 和 `sub_base[k:k+2]` 找到子矩阵，循环全部 K 段：
每个非空段执行 SpMV + RingAdd，写入 `(head+delay_steps[k]) % L`。
整个循环只算一个 step。hybrid 同样每列调用一次；两条路径都累加已有
buffer，不自行消费或推进 head，不提供 split 文件持久化。

FFI 调用保留 ring/workspace 别名声明。形状、dtype 和关键边界检查由
现有 CUDA 入口执行；测试调用者负责提供合法的 CSR 内容和匹配的元数据。
JAX cuSPARSE 封装每次创建临时 StepPlan，读取 split 边界，仅用于正确性
检查。原生 benchmark 在计时前准备并复用 StepPlan。

## 正确性测试

测试从原始 CSR、逐边 delay 和事件矩阵计算独立 CPU 参考，比较：

- split 的四份数组，而非仅比较两个投射结果是否相等；
- hybrid 与 cuSPARSE 每步消费值、运行结束后的 ring；
- 每步一次 cuSPARSE FFI、相同编译结果接受不同 split 边界；
- 空行、空图、零维、空段、单槽、零/非连续延迟、head 环绕、已有 ring；
- bool/f32 事件、负值活动、同质/异质权重、重复边、i32/i64 元数据；
- 短长行和 4096 切块边界、多 block 并发任务分配、CUDA 错误与状态重置。

`test_each_event_column_matches_numpy_step` 专门逐 step 对照：一次初始化
`events[N,T]` 并上传，每步读取 `events[:,t]`，分别调用一次 hybrid 和
一次 cuSPARSE 整步算子。NumPy 使用排序前的 CSR 和逐边 delay，通过
`np.add.at` 独立计算 ring，两个 GPU 结果都立即与 NumPy 比较当前槽输出
和完整 ring；随后测试消费、清零并推进 head。失败信息包含 backend、
step、head、K 和 T。split 在循环之前只调用一次。

该测试使用 `10000×10000` CSR 矩阵、K=8，每列事件独立按 30% 活跃
概率生成。覆盖 T=8、17、3，另有 T=3 且 `delay=4` 为空段的边界变体；
普通用例包含 `0～7` 全部八种延迟。
T=K 且全部段非空时，整轮各调用 K 次 hybrid/cuSPARSE 整步算子，
cuSPARSE 内部执行 K² 次 SpMV。一般情况为 T×非空段数；子矩阵循环
保持在 CUDA 中。测试还组合 bool/f32 非零事件、共享/逐边权重、
重复边、长行及环绕；NumPy 单步参考另有手算结果和绝对到达时间
参考的交叉核对。

```bash
python -m pytest brainevent/_tcsr/dev/with_delay/operator/test/region_delay_test.py -q
# 只运行新增的 NumPy 单步与逐列 GPU 对照
python -m pytest brainevent/_tcsr/dev/with_delay/operator/test/region_delay_test.py -k numpy -q
```

另外，`operator/test/region_delay_cusparse_test.py` 编译并运行同目录 CUDA
测试，直接核对 split → StepPlan 的原生接口。
`operator/test/delay_inputs_test.py` 负责 `region_delay.py` 的输入准备校验，
`benchmark/test/native_runner_test.py` 检查原生计时循环的数值。

2026-10-22 扩容为 10k×10k、K=8、30% 活跃概率后，JAX 0.11.0 和
JAX 0.8.0 下本算子测试文件各 82 项全部通过，修改后的测试文件通过
mypy。此前包含输入准备回归的 124 项验证及 100% 封装行覆盖率记录
保留在规格中。
检查范围与详细结果见 [精简规格](python-test-wrappers.md)。
