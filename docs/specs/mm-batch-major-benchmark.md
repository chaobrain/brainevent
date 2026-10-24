# MM benchmark 接入 batch_major

日期：2026-10-16。分支：`mm-optimizing`。

用户要求挂载 `new_tile_operator/dev/batch_major`，并核对测试参数与迁移前 ref 一致。

## 实施范围

- 新增 `batch_major` 注册，编号 31；直接包含现有 kernels.cuh 并调用
  `batch_major::launch`，不复制或修改算子源码，不经过 JAX FFI。
- 与 baseline 共用 BN 输入/输出、f32 类型范围、CSR 排序、tile 元数据、活动行工作区、
  输入池和逐步事件计时。沿用每轮统一同步及子进程隔离。
- 增加 batch_major 的 grid.y tile 数上限检查；其他已有参数限制沿用 baseline。
- 编译依赖纳入 batch_major/kernels.cuh，显式提供 baseline include 路径。
- 正式 CASE_LIST 增加 batch_major，保留原 cusparse_nb_alg2_nb 对照；参数轴不变。
  --smoke 增加 batch_major f32，共 18 个后端/类型组合。

## ref 参数核对

对照 `/tmp/mm_baseline_migration_lo824h64/ref/benchmark_binary_csr_matmat.py`：

| 参数 | 参考值 |
| --- | --- |
| scale | 100、200、400、…、3000；n = scale × 1000 |
| 矩阵密度 | 1e-5、3e-5、1e-4、3e-4、1e-3、3e-3、1e-2、3e-2 |
| 事件频率 Hz | 1、3、10、30、100、300、1000、3000 |
| batch | 2、4、8、16、32、64、128、256、512 |
| dtype / seed | f32 / [101] |
| 步数 / 预热 / 测量 | 25 / 2 / 5 |
| dt / 输入池 | 0.1 ms / 550 列 |
| direct / chunk nnz 限制 | 2^31−1 / 2^30 |
| 显存估算上限 / GPU | 96 GiB / 1 |

当前原 cuSPARSE 对照的 9216 个有效组合与 ref 单组完全一致，集合差异为零。
新算子采用相同的 9216 个候选参数组合，再按其实际缓冲需求过滤显存估算超限的用例。
参数值一致不等于运行次数完全一致：此前迁移已将 ref 的三份完全重复配置合并成一份，
删除了未参与计算的 n_tail=8。本次不恢复这两项。
DEFAULT_CASE 和 --smoke 的小规模设置用于自定义/验收入口，不是正式参数网格。

## 验收

- 参数回归测试固定上述参考值及 9216 个原对照组合，避免后续误改；测试不依赖 /tmp 备份。
- GPU 对照独立 NumPy 参考，覆盖多 batch、多 tile、4096 行活动提取边界、8192 列 tile
  边界、输入回绕、零/全活动、多步和事件计数；检查 batch_major 的非法 dtype/grid 参数。
- 确认编译依赖包含实际 kernel 文件，旧的 9 个后端继续通过测试。
- 驱动及测试通过 mypy；Python 和 native 主机侧覆盖率分别超过 90%。
- 实际子进程小规模验证 18 个组合，不自动运行全部大规模网格。

## 实施结果

- 已注册 `batch_major=31`，接入时正式默认列表为 `batch_major` 和 `cusparse_nb_alg2_nb`。
  原 baseline 和 8 个 cuSPARSE 注册仍可通过配置选择。batch_major 算子源码保持不变。
- 两个后端各有 9216 个候选组合。按当前 96 GiB 显存估算上限，batch_major 保留 7272 个、
  跳过 1944 个；cuSPARSE 保留全部 9216 个，正式默认合计 16488 个用例。
  这是缓冲需求不同导致的筛选差异，不是参数轴或计时口径不同。
- GPU 0、CUDA 13.0：236 项测试通过（58 项 Python、178 项 native）。
  多 tile 对照额外覆盖 batch=1、7、300；其余边界见上述验收清单。
- 驱动及两个测试文件通过 mypy。Python 驱动行覆盖率 257/257（100%）；
  native 主机侧 gcov 行覆盖率 272/297（91.58%），不代表 GPU 内核覆盖率。
- 真实子进程 smoke 的 18 个用例全部成功，结果位于
  `/tmp/mm-batch-major-smoke-20261016/`。该小规模检查不用于性能结论。
- 尚未运行全量正式网格。后续性能验收建议先选择代表性大规模用例，确认实际显存峰值、
  高 batch 下的正确性和运行稳定性，再运行全量；显存估算不包含不透明库 workspace。

从仓库根目录运行：

```bash
python brainevent/_tcsr/dev/new_tile_operator/baseline/benchmark_binary_csr_matmat.py \
  --smoke --gpu 0
```

用户随后将当前列表调整为仅 `batch_major`，正式筛选后为 7272 个用例；输出修正保留此选择。
省略 `--smoke` 和 `--config` 即运行文件当前 CASE_LIST；`--gpu` 只覆盖设备选择。
详细的配置文件与运行方式见 `new_tile_operator/benchmark/mm-ref-migration.md`。
输出现已恢复 ref 的配置组/逐 seed CSV，见 [输出对齐规格](mm-ref-output-alignment.md)。
