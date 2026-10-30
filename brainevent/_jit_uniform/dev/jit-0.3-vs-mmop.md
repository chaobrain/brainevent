# JIT Uniform：be0.3 相对 mm-optimizing 的更改

基线：2026-10-26 核对的两个工作区，含 be0.3 未提交的目录迁移；本包对照了全部 19 个原有 Python/CUDA 文件。

三个 JIT 的共同变化：

- 对象层统一输入识别；`WeightedArray` 转为权重 dtype，走 float MV/MM。
- binary 统一按 `x != 0` 判断活跃，负值和 NaN 也算事件；Numba 接收 bool 掩码，CUDA 在打包前生成掩码。
- float MV/MM 补齐跳零，避免非活跃输入的 `0 * NaN/Inf` 污染输出；Numba fastmath 排除 `nnan/ninf`。
- 参数梯度改用 `active_sum`：binary 只计活跃，float 保留幅值，非活跃项选零。
- 7 个 CUDA 文件的宿主 grid 越界检查由 `abort()` 改为 `BE_CHECK`；binary MM 打包新增 `n <= 65535` 检查。

Uniform 特有变化：

- 参数反向由“基矩阵乘事件，再与上游梯度 `ct` 归约”，改为“基矩阵转置乘 `ct`，再按输入活跃性归约”。
- 设 `C` 为连接矩阵、`U` 为连接上的均匀样本：high 梯度归约 `U.T @ ct`，low 梯度归约 `C.T @ ct - U.T @ ct`。binary 反向也使用 float 基算子，以保留 `ct` 数值。
- 旧 binary 已通过基算子处理事件幅值；这里主要修复非活跃贡献的 NaN/Inf 隔离，不是 Scalar/Normal 的误乘幅值问题。参数 JVP 未改。

依据：[binary 反向](../binary/binary.py#L610)、[float 反向](../float/float.py#L1213)。

目录调整：实现按 `binary/float/csr/dt2t` 归组，测试进入各自 `test/`。CSR、dt2t 的 Python 计算逻辑未改，仅调整导入。详见[结构计划](jit-uniform-structure-plan.md)。

测试差异：新增 **14 个测试定义**。此前选定的 CPU 回归，本包 **64 项通过**（三包共 202 项）；GPU 未验证。
