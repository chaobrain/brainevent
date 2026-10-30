# LIF delay 示例

按用户限定，仅在 `brainevent/_tcsr/delay_mv/example` 内新增文件。

实现一个可直接运行的 `lif_delay.py`：10,000 个 LIF 神经元组成随机稀疏网络，
矩阵形状为 10,000 × 10,000，密度 1e-2，恰好 1,000,000 条边。
从所有矩阵位置中均匀无放回采样，允许自连接，禁止重复位置；直接构造 CSR，
不构造稠密权重矩阵。每条边的 delay 独立均匀采样自整数 0…8（最大延迟 8 步，
buffer 共 9 槽），随后按源行、delay 排序，保持目标索引与 delay 的对应关系。

固定随机种子 2026。同质权重为 float32 0.2，初始约 10% 神经元随机放电，
初始膜电位全零；后续输入全部来自上一轮的 LIF 放电。权重和初始放电率是
可在示例中调整的参数。

每步在 Python 循环中依次执行：

1. `arrivals = spikes @ W`，通过 `delay_tcsr.__rmatmul__` 投射和消费。
2. `W` 自行清空当前槽并推进 head；示例只读取 head 用于输出。
3. LIF 更新 `v = exp(-dt/tau) * v + arrivals`，阈值为 1，放电后重置到 0。
4. 打印消费前后的 head、输入/输出 spike 数量、加权到达量之和、重置后膜电位均值。
5. 输出 spike 作为下一步输入，包括全零向量。

这里使用无量纲膜电位、`dt=1 ms`、`tau=10 ms` 和瞬时突触电位跳变；不设置
不应期。`arrivals` 是已消费槽的加权输入，不是新 head 所指槽的值。
示例不操作 buffer，不写 head，不直接调用 binary 或其他内部后端入口。

在仓库根目录运行：

```bash
python -m brainevent._tcsr.delay_mv.example.lif_delay --steps 16
python -m brainevent._tcsr.delay_mv.example.lif_delay --steps 16 --backend jax
python -m brainevent._tcsr.delay_mv.example.lif_delay --steps 16 --backend cuda_raw
python -m brainevent._tcsr.delay_mv.example.lif_delay --steps 16 --backend cusparse
```

默认沿用 `delay_tcsr` 的后端选择。验证精确形状/密度、唯一连接、延迟范围与排序，
对照独立 NumPy 的到达时序、LIF 反馈和 head 回绕；运行三个后端、mypy 和 JAX 0.8 CPU。

后续修改示例参数时，建议继续检查最大延迟到达时刻、连续无输入时的膜电位
衰减，以及产生的新 spike 是否恰好成为下一步输入。

## 扩大后的验证结果

- 精确验证 10,000 × 10,000、1,000,000 个唯一位置、行内延迟非递减和全部
  0…8 延迟段。固定种子下各段边数在 110,790 至 111,857 之间，三个后端
  得到完全相同的图和延迟数组。
- JAX、CUDA hybrid、cuSPARSE 各运行 24 步。使用独立 NumPy 绝对时间队列
  对照完整到达向量和剩余 buffer，容差 rtol=atol=2e-5；head 两次回绕正确。
  用实际到达量计算 NumPy LIF，并逐元素验证下一步输入 spike 与其输出一致。
- 当前 JAX CPU 的 24 步运行通过；隔离 JAX 0.8 CPU 的 10 步运行通过。
  旧环境仍输出继承 CUDA 0.11 插件的不兼容日志，但 CPU 运行正常结束。
- mypy 无错误；无单位示例将左乘返回值显式转换为 JAX 数组，避免统计函数
  接口接收到泛型 Data 中的 Quantity 类型。示例 38 个有效 Python 行覆盖率
  100%，包括 `--steps 0` 拒绝分支；未修改示例目录外的源码或覆盖率文件。
- 默认种子产生 964 个初始 spike；纯兴奋、无不应期的当前参数在第 5 步起
  全体放电。这是当前演示参数的实际运行结果。
