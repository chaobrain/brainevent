# MM baseline benchmark 迁移实施规格

日期：2026-10-16；分支：`mm-optimizing`。

用户已通过“组织实现”批准执行
[迁移方案](../../brainevent/_tcsr/dev/new_tile_operator/benchmark/dev/md/mm-ref-migration.md)。

## 实施边界

- 将 ref 的驱动、native runner、两个辅助头和测试迁入 baseline，取消旧后端，
  保留 baseline（30）和 cuSPARSE（21–28）共九项注册。
- baseline 保留现有 f32 测试能力，直接使用本地内核；cuSPARSE 保留 f32/f64、
  direct/chunk 及全部 NB/BN、ALG1/ALG2 组合。
- 删除无消费者的 n_tail、output_l1，以及被移除后端的队列、缓冲、状态历史和分支。
  在测试侧补充 baseline 所需的局部目标与 tile offsets 准备函数。
- 保持 GPU 逐步/逐块计时、每轮统一同步求和；保留清零、准备操作原有计时边界。
- 父进程编译和串行调度，一个新 Python 子进程运行一个参数组合的完整预热与测量。
  默认超时 1800 秒；崩溃、超时、错误返回和损坏结果写失败行后继续。父进程统一落盘。
- 不改 baseline 计算内核、父目录 JAX benchmark/runtime 和其他实验源码。

## 验收

先为新增进程保护编写失败测试，再实现。迁移后的测试必须独立构建，
不使用外部 MM 内核目录。数值验证覆盖 cuSPARSE 32 个基本组合及 baseline 边界；
计时测试检查事件次数、轮末同步、统计关系和失败清理。
进程测试覆盖成功、返回错误、崩溃、超时、结果异常、中断以及失败后继续。
Python mypy 无错误；受影响 Python 和 CUDA 主机代码分别争取超过 90% 覆盖率。
测试使用同目录 `*_test.py`，真实超大 nnz 与性能稳定性不冒充已验证。

## 实施结果

- 已迁移五个框架源文件，ref 对应文件已移除；新增 Python 测试，更新本地来源说明。
- native：1649 → 709 行；Python 驱动：644 → 377 行；运行代码净减少 1207 行。
  baseline 计算内核和两个辅助头内容未变，构建无需外部 MM 头文件。
- 200 项测试通过（144 native + 56 Python），mypy 对三个 Python 文件检查无错误。
- Python 驱动行覆盖率 253/254（99.61%），native 主机侧 gcov 行覆盖率 268/293（91.47%）。
  gcov 不代表设备内核覆盖率。
- 真实子进程端到端小规模检查覆盖全部 17 个注册/dtype 组合；使用 GPU 0、CUDA 13.0。
- 新增回归测试发现并修复：未知配置项被默默接受、非整数执行元数据被接受、
  不同 warmup/runs/chunk 设置被混入同一汇总组。现以配置白名单、结果类型检查及完整分组键防止复发。
- 初次迁移曾改为 raw.csv（带 seed）和每用例 JSON/日志目录；该输出组织已按用户要求撤回，
  当前恢复 ref 的配置组/逐 seed CSV，见 [输出对齐规格](mm-ref-output-alignment.md)。
- 真实超大 nnz 和大规模性能稳定性尚未验证，chunk 正确性使用较小编译阈值触发。
- 操作命令、结果结构和详细验收记录见迁移方案第 10 节。
