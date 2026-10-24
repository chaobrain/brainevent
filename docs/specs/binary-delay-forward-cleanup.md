# binary_delay 第一阶段：移除训练、MM 和批处理

## 已确认的范围

2026-10-23：先清理 `brainevent/_tcsr/delay_mv/binary_delay.py`，之后再讨论
delay 接口和从 `main.py` 提取的轻量封装。所有修改在现有 worktree 分支
`mm-optimizing` 上完成，保留用户已有修改。

- 事件统一按 `events > 0` 判定，bool 的 True 表示激活；负值不激活。
- 删除 `binary_csrmm`、其 primitive、所有 MM 后端和变换规则。
- 删除 Numba MV 后端与注册，后端集合仅保留 jax、cusparse、cuda_raw。
- 删除 MV 的梯度后端选择、JVP、transpose rule、训练算法参数及依赖。
- 删除专用 batching，并取消 XLACustomKernel 自动注册的通用 batching；
  grad、jvp、vmap 均不支持，不能静默产生零梯度或转入 MM。
- 删除仅用于 MV 转 MM 的 tile operands、校验、dtype 辅助函数和静态属性。
- 按前一轮命名要求将 MV 入口、调用包装与 primitive 改为
  binary_csrmv_delay、binary_csrmv_delay_p_call、binary_csrmv_delay_p。
- 保留当前 MV 的 transpose/镜像前向路径及三种后端实现，修复迁入子目录后
  的依赖导入和普通 MV CUDA 源文件定位。拒绝二维事件输入。

## 阶段边界

本阶段保留的算术仍是普通单步 MV，是供下一阶段设计 delay 接口的过渡实现。
不接入环形缓冲，不迁入 main.py 的类，不修改同目录的三个 delay CUDA 文件，
也不修改原有 `_tcsr/binary.py` 或共享 XLACustomKernel 的变换行为。
delay CUDA 中现有的非零事件判断将在接入阶段与已确认的 `> 0` 语义统一。
本阶段文档必须明确这一限制，不能把函数更名描述成已经支持 delay。

## 验证

先用同级 `binary_delay_test.py` 复现当前模块的导入错误，再进行修复。
验证保留前向的正值/零/负值/bool 事件、同质/异质权重、空图、重复连接、
两种遍历方向与 JIT；验证二维输入、Numba、grad/jvp/vmap 被拒绝。
GPU 可用时对三后端使用相同参考结果进行验证，覆盖直接和镜像的 FFI 路径。
测试不要求 CUDA 原子累加逐位一致。

有效 Python 路径覆盖率目标大于 90%，实现及同级测试通过 mypy。
验证可用的 JAX >= 0.8.0 环境，并记录实际版本与结果。
公共接口保持 NumPy docstring，例子使用带提示符的 python code-block。

## 修正记录

文件从父目录复制后仍使用单点相对导入和当前目录 CUDA 路径，导致导入或加载失败。
本阶段以真实包导入和真实前向调用验证，后续移动模块时同步检查 Python 依赖、
CUDA 源文件路径、primitive 名称和文档示例，避免仅做文本删除而遗留失效路径。

JAX 0.11 的 `primitive_batchers` 是代理对象，没有字典的 `pop` 方法。
导入回归测试捕获这一假设错误后，改为其支持的 `del registry[primitive]`；
它也适用于旧版字典注册表。后续不根据变量名称推断容器接口，以实际版本实现
及导入、vmap 行为测试验证。高层入口将去单位后的输入转为 JAX 数组再调用
带类型注解的低层入口，避免通过扩大 Any 或忽略类型错误掩盖输入边界。

## 实施结果

- `binary_delay.py` 从 1972 行缩减为 582 行；保留 8 个函数和前向 primitive
  注册，移除训练、MM、Numba、tile 参数和所有 batching 能力。
- 当前执行环境为 Python 3.12.9、JAX 0.11.0，使用 `multiplatform` 环境。
- GPU：`CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest
  brainevent/_tcsr/delay_mv/binary_delay_test.py -q -m ''
  --cov=brainevent/_tcsr/delay_mv --cov-report=term-missing --cov-fail-under=90`：
  59 项通过，无跳过；模块覆盖率 96.58%。覆盖三后端的真实执行与 JIT。
- CPU：`JAX_PLATFORMS=cpu python -m pytest
  brainevent/_tcsr/delay_mv/binary_delay_test.py -q -m ''`：31 项通过，28 项 GPU
  用例按平台跳过。
- `python -m mypy --follow-imports=silent` 检查实现和同级测试，0 错误；
  模块 doctest 的 10 个输入全部通过。
- 本次未运行 JAX 0.8.0 环境；下一阶段仍须保持最低版本兼容。

覆盖率按目录路径采集。以完整模块名指定 coverage source 时，本环境发生
NumPy 重入导入错误；换成目录路径后，真实测试与覆盖率采集均正常。
未修改共享环境或依赖版本。

下一阶段接口确定后，再补充 delay 专属的零延迟、最大延迟、环形回绕、
多投射累加、消费清零、跨步 workspace 复用，以及 delay CUDA 的负值不激活测试。
