# JIT Normal：be0.3 相对 mm-optimizing 的更改

基线：2026-10-26 核对的两个工作区，含 be0.3 未提交的目录迁移；本包对照了全部 19 个原有 Python/CUDA 文件。

三个 JIT 的共同变化：

- 对象层统一输入识别；`WeightedArray` 转为权重 dtype，走 float MV/MM。
- binary 统一按 `x != 0` 判断活跃，负值和 NaN 也算事件；Numba 接收 bool 掩码，CUDA 在打包前生成掩码。
- float MV/MM 补齐跳零，避免非活跃输入的 `0 * NaN/Inf` 污染输出；Numba fastmath 排除 `nnan/ninf`。
- 参数梯度改用 `active_sum`：binary 只计活跃，float 保留幅值，非活跃项选零。
- 7 个 CUDA 文件的宿主 grid 越界检查由 `abort()` 改为 `BE_CHECK`；binary MM 打包新增 `n <= 65535` 检查。

Normal 特有变化：

- 参数 JVP：loc 方向由 `(loc_dot, scale)` 改为 `(loc_dot, 0)`；scale 方向由 `(loc, scale_dot)` 改为 `(0, scale_dot)`，去掉无关参数原值带来的额外贡献。
- 修正覆盖 binary/float、MV/MM、loc/scale，共 8 个 JVP 函数；稠密生成 `jitn` 的 JVP 未改。
- binary 参数 VJP 修复误乘事件幅值；float VJP 保留幅值。两者只改最终归约，没有 Uniform 的反向计算顺序重写。

依据：[binary JVP/VJP](../binary/binary.py#L445)、[float JVP/VJP](../float/float.py#L845)。

目录调整：实现按 `binary/float/csr/dt2t` 归组，测试进入各自 `test/`。CSR、dt2t 的 Python 计算逻辑未改，仅调整导入。详见[结构计划](jit-normal-structure-plan.md)。

测试差异：新增 **16 个测试定义**。此前选定的 CPU 回归，本包 **72 项通过**（三包共 202 项）；GPU 未验证。
