# Hybrid MV 基线测试子进程隔离

日期：2026-10-14。状态：已按用户确认的方案实施。

## 目标

`brainevent/_tcsr/dev/new_mv_operator/benchmark/run.py` 原先在同一进程内依次执行全部用例。段错误、CUDA 非法访问或进程被系统终止会使整个清单中断；显存不足后的分配器状态也可能影响后续用例。改为父进程调度、子进程执行：每条记录在独立子进程中运行，子进程退出后父进程再启动下一条，单个用例崩溃不终止测试。

所有更新在当前工作分支 `mm-optimizing` 完成。

## 已确认的取舍

| 事项 | 决定 |
| --- | --- |
| 隔离粒度 | 每条运行记录（结构 × 活动率）一个子进程 |
| 崩溃处理 | 只记录，不重试 |
| 超时 | 不限时，不提供超时选项 |
| 内存预算 | 开始时由探测子进程测一次，全程共用 |

## 设计

### 父进程

1. 解析并校验参数，口径与原实现相同。
2. 启动一个探测子进程，开启 x64 后计算主机与设备预算，并返回设备、JAX 版本等环境信息。父进程不初始化 GPU 后端，不占用显存。探测失败时抛出 `RuntimeError`，此时尚未创建输出目录。
3. 创建输出目录、写 manifest（新增 `isolation` 字段）。
4. 逐条枚举记录，在父进程内执行不可实现判定和资源预检；这类记录不启动子进程，直接写入。
5. 对可运行记录，经临时目录中的 JSON 文件传入记录、预算和 `warmup/repeat/weight_mode`，启动 `sys.executable -u run.py --worker <task.json> <result.json>`，继承环境变量和工作目录。子进程 stdout 直接继承；stderr 逐行转发，同时保留最后 20 行。
6. 子进程退出后：
   - 结果文件存在且是 JSON 对象：采用该记录。若返回码非零（例如结果写出后在解释器退出阶段崩溃），追加 `worker_returncode`。
   - 结果文件缺失或无法解析：记为 `status="crashed"`，`reason` 包含返回码或信号名及 stderr 尾部。
7. 父进程收到中断时终止当前子进程，manifest 标记为 `interrupted`。
8. 返回码：存在 `failed_*`、`generation_failed` 或 `crashed` 时返回 1。

### 子进程

`--worker` 模式读取任务文件，开启 x64：探测任务调用 `_memory_limits`，用例任务复用原 `_run_case`（计时与正确性逻辑不变），结果先写临时文件再重命名，保证父进程不会读到半个文件。

## 代价

每条记录都要重新导入 JAX、加载 CUDA 动态库并编译 XLA；CUDA 源码有编译缓存。该开销不在计时区间内。

## 验证

- 真实子进程测试（父进程测试不依赖 GPU，使用替换的子进程命令）：
  - `os.abort` 崩溃后记为 `crashed`，含信号名，并继续执行后续记录，CLI 返回 1；
  - 结果写出后非零退出，保留结果并记录返回码；
  - 结果文件损坏记为 `crashed`；
  - 预检跳过和不可实现记录不启动子进程；
  - 探测失败抛出错误且不创建输出目录；
  - 中断时终止子进程并保存 manifest。
- 进程内直接调用 worker，覆盖探测和用例两类任务。
- GPU 端到端：通过真实探测子进程和用例子进程完成一个 10k 用例并通过正确性检查。无 GPU 时跳过。
- 覆盖率 >90%、mypy 无错误。
- 本次实施期间 GPU 由他人占用，按用户要求只以 `JAX_PLATFORMS=cpu` 运行测试，GPU 相关测试跳过，留待 GPU 空闲时补测。

## 实现与验证结果

实现位于 `benchmark/run.py`（`_spawn`、`_isolated_case`、`_probe`、`_worker` 和 `main` 中的 `--worker` 入口），测试位于相邻的 `run_test.py`，说明同步到 `benchmark/README.md`。`_run_case` 的计时与正确性逻辑及算子实现均未修改。

- 2026-10-14，`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""`：JAX 0.11.0 与 0.8.0（`/tmp/brainevent-mv-jax080`）下，`baseline_test.py` 和 `benchmark/run_test.py` 均为 40 项通过、25 项因需要 CUDA 跳过。JAX 0.11.0 下 runner 覆盖率 98.03%（未覆盖部分为仅由 GPU 测试触达的原有失败分支及 `__main__` 行），总覆盖率 93.90%。
- 四个 Python 文件通过 mypy，`git diff --check` 通过。
- GPU 被他人占用，按要求未在 GPU 上运行测试。GPU 空闲后需补跑完整测试，包括新增的 `test_gpu_cli_runs_case_in_child`。
