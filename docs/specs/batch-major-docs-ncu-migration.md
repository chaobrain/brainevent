# batch_major 文档与 NCU 记录归档

日期：2026-10-17。工作分支：`mm-optimizing`。

## 目标与范围

把 `new_tile_operator/dev/batch_major` 使用的实验分析集中到该目录的 `docs/`，
把 E3 实际使用的 TileMM NCU 记录集中到 `new_tile_operator/ncu/batch_major/`。
本次仅移动文档、历史记录并维护索引与链接，不修改算子或捕获工具代码。

以下路径除特别说明外均相对 `brainevent/_tcsr/dev/new_tile_operator/`。

## 文档清单

| 原路径 | 目标路径 | 与 batch_major 的关系 |
| --- | --- | --- |
| `dev/batch_major/findings.md` | `dev/batch_major/docs/findings.md` | 汇总分析，包含 E3 高/低事件率结果和产物索引 |
| `dev/md/e1_global_atomic.md` | `dev/batch_major/docs/e1_global_atomic.md` | 汇总文档引用的写侧对照证据 |
| `dev/md/e2_rotated_rows.md` | `dev/batch_major/docs/e2_rotated_rows.md` | 汇总文档引用的读侧对照证据 |
| `dev/md/e3_batch_major.md` | `dev/batch_major/docs/e3_batch_major.md` | E3 详细分析、计时与 NCU 数据 |

新增 `dev/batch_major/docs/README.md`，列出文档、共享规格、计时和 NCU 证据。
在 `dev/md/README.md` 加入新入口。所有受影响的本地 Markdown 链接按新位置更新。

仓库 `docs/specs/` 中的规格继续集中保存：
`tilemm-e1-e2-variants.md`、`hybrid-mv-atomic-ncu.md` 和
`mm-batch-major-benchmark.md` 均通过索引链接访问。
原生 benchmark 的说明、运行结果说明及其他实验文档保留其归属位置。

## NCU 清单

下列 10 个有效原记录目录整体移入 `ncu/batch_major/`，目录名保持不变。
它们均来自物理 GPU 1；有效性依据现有 E3 分析及整理集 README。

| 原记录目录 | 变体 | d | p |
| --- | --- | --- | --- |
| `20261016_175909_791840_full_m100000_n100000_b128_d0.01_p0.03_baseline` | baseline | 0.01 | 0.03 |
| `20261016_175950_305924_full_m100000_n100000_b128_d0.03_p0.03_baseline` | baseline | 0.03 | 0.03 |
| `20261016_180048_861306_full_m100000_n100000_b128_d0.1_p0.03_baseline` | baseline | 0.1 | 0.03 |
| `20261016_180236_273019_full_m100000_n100000_b128_d0.01_p0.03_batch_major` | batch_major | 0.01 | 0.03 |
| `20261016_180316_794642_full_m100000_n100000_b128_d0.03_p0.03_batch_major` | batch_major | 0.03 | 0.03 |
| `20261016_180414_418644_full_m100000_n100000_b128_d0.1_p0.03_batch_major` | batch_major | 0.1 | 0.03 |
| `20261016_182554_574036_full_m100000_n100000_b128_d0.003_p0.003_baseline` | baseline | 0.003 | 0.003 |
| `20261016_182628_026581_full_m100000_n100000_b128_d0.003_p0.003_batch_major` | batch_major | 0.003 | 0.003 |
| `20261016_182702_023450_full_m100000_n100000_b128_d0.01_p0.003_baseline` | baseline | 0.01 | 0.003 |
| `20261016_182743_454764_full_m100000_n100000_b128_d0.01_p0.003_batch_major` | batch_major | 0.01 | 0.003 |

现有整理副本目录 `ncu/20261016_e3_batch_major_gpu1/` 整体移动到
`ncu/batch_major/20261016_e3_batch_major_gpu1/`，保留其 10 份副本和 `summary.csv`。
迁移前已按逐文件 SHA-256 核对，10 组副本均与对应原记录完全相同。
保留副本不做去重，原目录与副本的关系写入新索引。

新增 `ncu/batch_major/README.md`，包含完整原记录/副本映射和有效参数组合。
整理集的 README 更新原记录位置以及分析入口。

## 历史证据与边界

- 受干扰、失败或中断的 GPU 3 记录 `20261016_175036_*`、`20261016_175138_*`、
  `20261016_175352_*`、`20261016_175527_*` 保留在原位置；索引说明其排除原因。
- `brainevent/_tcsr/dev/new_mv_operator/ncu/20261016_115426_full_random_n1e5_p3e-2_mv_atomic/`
  是共享的 MV 对照证据，保留在 MV 目录；新文档索引提供链接。
- NCU 报告、CSV、日志、配置、provenance、manifest、源码快照和历史捕获脚本保持字节不变。
  其中的绝对路径记录捕获时的位置，不改写为迁移后路径，不作为当前执行入口。
- 文档中的历史命令和实验结论不因归档而更改。涉及导航的路径、目录索引按新位置更新。
- 移动前检查目标冲突，保持现有工作区的其他改动，全部操作在当前非 main 工作分支完成。

## 验收与建议测试

1. 迁移前保存所有被移动 NCU 文件的相对路径、大小和 SHA-256；迁移后逐项验证。
   只有整理集的导航 README 可以改动，报告、数据和历史证据文件不得改变。
2. 确认 10 份原报告和 10 份整理副本完整到位，原目录均已移走，`summary.csv` 内容不变。
3. 确认 4 份分析文档都位于目标 docs，旧路径已移走。
4. 检查修改或新建文档的本地 Markdown 链接、图片链接，以及 `summary.csv` 引用的目录。
5. 检查共享规格、MV 记录、4 份排除记录和算子源码内容不变。
6. 该改动不涉及运行时代码，不重新运行 GPU benchmark 或改写捕获产物。
   有意义的边界检查包括目标重名、隐藏/被 Git 忽略的文件、相对路径层级变化和重复证据保全。

## 实施状态

已在 `mm-optimizing` 分支完成：4 份分析文档迁入目标 docs，10 个有效 NCU 原记录目录和
1 个整理副本集迁入 `ncu/batch_major/`。新增两处 README 索引，并更新公共文档入口。

迁移集合共 910 个历史 NCU 文件：整理集的导航 README 更新了位置说明，
其余 909 个文件的大小与 SHA-256 均与迁移前一致。225 个受保护文件通过哈希校验，
包括算子源码、其他规格、共享 MV 记录以及 4 份排除记录。
10 份原报告和 10 份副本均完整到位，10 组原记录与副本仍逐文件相同；
`summary.csv` 保持 20 行，引用的 10 个子目录全部存在。

文档复查时发现低事件率记录列表有一项原先省略了目录前缀，已补为
`ncu/batch_major/20261016_182702_023450_*`。路径验收同时检查 Markdown 链接和
此类行内目录缩写，避免只检查显式链接而遗漏导航文字。
