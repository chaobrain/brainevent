# delay_tcsr 精简封装

2026-10-23，按已批准计划在 worktree 分支 `mm-optimizing` 实施。

## 范围与接口

本阶段只实现 `delay_mv/delay_main.py` 及同级 `delay_main_test.py`，不接入
真实 delay 后端，不修改现有普通 MV、CUDA 或包级导出。模块包含一个转换函数、
一个内部 NamedTuple workspace 和一个独立类，不继承矩阵体系，不注册 PyTree，
不引入 BrainState 或额外工厂方法。

`prepare_delay_metadata(edge_delays, indptr, *, n_post, buffer_slots=None)`
返回 `(delay_steps, delay_count, delay_off)`。输入已按行、按延迟升序分段；
在 CPU 上校验和统计，不排序或重排 CSR。delay_steps 为升序整数元组，
delay_count[M,K] 跟随 indptr 的 int32/int64 类型；delay_off[K] 等于
delay_steps*N，按实际 L*N 选择 int32/int64。空图用 `(0,)` 与零计数。
L 默认最大延迟加一，显式 L 必须更大；需要 int64 时要求启用 JAX x64。
校验 CSR 行指针、非负整数延迟、每行顺序和长度，维度/槽数采用当前 CUDA 的
int32 范围，缓冲字节数必须可用 int64 表示。

`delay_tcsr(source, edge_delays, *, buffer_slots=None, binary_backend=None)`
仅实现 `__init__` 和 `__rmatmul__`。source 必须是 PlainCSR；支持同质/异质
float32 权重并保留已有单位。保存 CSR 数组、shape、后端及 delay 元数据。
源数组顺序保持不变，构造阶段不接受事件序列。

## 初始化与状态流转

固定采用 delay CUDA 默认阈值 T=128、任务大小 Q=4096，避免普通 TCSR 的
自动调优配置改变容量计算。短行 row_task_count=0；长行按各延迟段分别向上
取整后求和。以宽位整数计算任务容量并检查 int32 上限。

`_DelayWorkspace` 保存 task_capacity、task_begin、task_end、task_out_off、
status；三个任务数组长度等于容量，begin/end 与 indptr 同型，out_off 与
delay_off 同型，status 为 int32[2]。初始均为零。buffer 为全零 float32[L,N]，
head 为 device int32[1]，初值 0。元数据及工作区在初始化时构建。

左乘接收 NumPy/JAX bool/float32 向量和 BinaryArray，要求形状 [M]，转换为
`events > 0` 的 bool 激活。矩阵输入被拒绝。设置数组运算优先级，使三种输入
进入 `__rmatmul__`。

约定后续 `binary_delay.binary_csrmv_delay_step` 的调用签名：

```text
step(data, indices, indptr, events,
     shape=shape, delay_count=delay_count, delay_off=delay_off,
     row_task_count=row_task_count, buffer=buffer, head=head,
     workspace=_workspace, backend=binary_backend)
    -> (y, new_buffer, new_head, new_workspace)
```

delay_steps 由构造阶段使用；底层运行时由 delay_off 与 buffer 形状得到所需信息。
底层负责投射、消费清零、推进 head，类只保存返回的三个状态并返回 y。
完整 workspace 必须返回下一步，包含 begin/end/out_off/status。
底层入口缺失时抛出包含“delay 后端尚未接入”的 NotImplementedError，成员不变。
左乘限定在 Python 循环中使用；拒绝带 tracer 的调用，避免外层 jit/scan/vmap/AD
导致隐式成员更新。底层内部编译不受此限制。

## 测试与验收

- 元数据：零/非连续延迟、空行/空图、零目标、行间重新从小延迟开始；非法
  延迟、长度、行指针、排序、维度、槽数与 x64 设置；偏移类型按实际 L*N 判定。
- 初始化：同质/异质及带单位权重、CSR 顺序保留、形状/dtype/初值，128/129
  行长分界、4095/4096/4097 段长度，以及长行中多个短段各自生成任务。
- 左乘：NumPy/JAX/BinaryArray 分派、严格正值激活、参数传递、缺失后端与非法
  输入；用明确的测试替身跨多步更新 buffer/head/workspace，覆盖回绕。
- 不将替身测试描述为真实 CUDA 正确性验证。新增有效 Python 路径覆盖率 >90%，
  实现及测试通过 mypy，公共接口使用 NumPy docstring 并验证 doctest。
- 使用当前 JAX 0.11.0 与隔离的 JAX 0.8.0 CPU 环境验证，不修改共享依赖版本。

## 验证结果

- JAX 0.11.0 / CPU：同级测试 77 项通过；新增 `delay_main.py` 行覆盖率
  99.02%（102 条有效语句，未覆盖实际超大图的任务容量溢出分支）。
- JAX 0.8.0 / CPU：同样 77 项通过。隔离环境继承的 CUDA 0.11.0 插件与旧版
  JAX 不兼容并被忽略；本次未运行 CUDA 后端，也未改动共享环境。
- `mypy --follow-imports=silent` 检查实现及测试两份文件：零错误。
- 公共转换函数的 6 条 doctest 全部通过。
- 初次验证纠正了测试导入方式和 float32/bool 参数化变量的类型声明；后续
  延续绝对导入，并在多 dtype 测试中明确数组类型，避免重复出现同类问题。

后续接入真实后端时，需要补充 GPU 数值对照测试，验证零延迟当步到达、
最大延迟回绕、重复目标累加、同质/异质及带单位权重、任务容量和 status 错误
传播。当前替身测试只验收接口契约与 Python 成员状态流转。
