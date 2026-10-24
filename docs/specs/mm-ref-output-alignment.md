# MM benchmark 输出机制对齐 ref

日期：2026-10-16；工作分支：`mm-optimizing`。

用户在输出审查后要求“对齐 ref 的机制”。本规格覆盖 Python 输出调度，
不修改 CUDA 算子、计时、参数轴或用户当前只选择 batch_major 的配置，不重启已停止的正式测试。

## 输出约定

```text
runs/<时间>/
├── case_001/                 # CASE_LIST 的第一组，不是单个参数点
│   ├── case_1.csv            # 第一个 seed 的全部参数点，逐行写入
│   ├── case_2.csv            # 如果配置第二个 seed
│   ├── summary.csv           # 只汇总本组重复实验
│   └── run.log               # seed 映射、参数、结果和 worker 输出
└── case_002/                 # 第二组，即使参数相同也单独统计
```

- 执行顺序恢复为配置组 → seed → 全参数网格。显存筛选保留原公式。
- CSV 表头恢复 ref 的字段顺序，只保留此前明确删除 n_tail 的差异。
  不再新增 seed 列；每个重复编号对应的 seed 写入 run.log。
- summary.csv 从本组已落盘的 CSV 读取，均值、样本标准差、极值及
  `failure_count = repeat_count - success_count` 沿用 ref。不同配置组绝不合并。
  汇总列同样恢复 ref（去除 n_tail）。为保护已有自定义入口，内部仍区分
  warmup/runs/chunk 设置，避免将执行条件不同的结果混为一项。
- CSV 表头和每条结果均 flush/fsync。每组结束立即汇总；Python 中断/异常退出时
  尝试汇总已落盘结果。默认 SIGTERM/SIGKILL 不保证执行 Python finally，和 ref 一样。
- 每个参数点仍启动独立 worker，保留超时、崩溃后继续和中断时回收子进程。
  config/result JSON 放在一个临时目录，正常返回、失败、超时及 Python 中断后清理；
  worker 输出追加到本组 run.log，不永久建立参数点目录。强制终止可能遗留当时的临时目录。
- --config/--smoke 的显式参数列表作为一个配置组，按 seed 分开 CSV。
  不恢复已去重的 CASE_LIST 块，不改变用户当前后端选择，不转换或删除已有历史结果。

## 回归与验收

此次问题来自将进程隔离与结果目录改造绑定，旧测试只验证了新布局。
先新增失败回归，再修正实现，独立固定 ref 的表头、目录层级、执行顺序和跨 seed 汇总结果。

- 两个相同配置组、两个 seed、多个参数点：两组独立 CSV/汇总，无跨组合并。
- 每个 seed 执行完完整网格，再进入下一个 seed；组结束即有 summary。
- 单 seed、多 seed、全失败、缺失重复记录、坏数值、不同运行设置、空/超限网格。
- 成功、native 错误、进程崩溃、超时、损坏结果和 Python 中断均不遗留通信文件；
  组日志保留必要诊断，CSV 失败行耗时为空，下一参数点仍能执行。
- 驱动测试及 mypy 通过，Python 行覆盖率超过 90%。仅用小规模 CLI smoke 验证真实调用链，
  不恢复后台正式测试；CUDA 内容不变，无需扩大 native 数值测试。

## 实施结果

- 已恢复配置组目录、逐 seed CSV、组日志和组内汇总。当前只选 batch_major 的配置保持原样。
- 先运行回归，确认旧实现的 seed 顺序、表头、独立组汇总三个测试失败，再完成修正。
  实现阶段测试也捕获了 seed 参数被循环变量遮蔽的问题；现用独立变量名，
  并保留不同 seed、重复 seed、相同配置组的顺序回归，避免同类错误复发。
- 66 项 Python 测试通过，驱动和测试文件 mypy 无错误；驱动行覆盖率 295/295（100%）。
- 与迁移前 ref 备份直接比较：原始及汇总 CSV 表头除 n_tail 外逐列一致；
  双 seed 的均值、样本标准差、失败和缺失重复记录的汇总结果逐字段一致。
- GPU 0 上真实 CLI smoke 的 18 个后端/类型用例全部成功；结果目录
  `/tmp/mm-ref-output-smoke-20261016/case_001/` 仅有 `case_1.csv`、`summary.csv`、`run.log`。
  两份 CSV 均有 18 条记录，没有永久参数点目录或 JSON 文件。
- 未修改 CUDA 文件或 GPU 计时；未重启 GPU 2 的正式任务，未修改旧任务的 253 条 CSV 记录。
- 后续维护建议保留多配置组、多 seed 的输出回归；新后端应复用这套输出约定。
