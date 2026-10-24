# TCSR SDDMV（batch=1）各方案与 cuSPARSE 的单步正确性测试

日期：2026-10-16。分支：`mm-optimizing`。补充 [sddmv-cusparse-benchmark.md](sddmv-cusparse-benchmark.md)：benchmark 不校验输出，由本测试把 `dev/` 下各方案的单步输出与 cuSPARSE 对齐。

## 1 工作负载

- 结构：`run.generate(10_000, 1000, RNG_SEED_LIST[0], hz)`（C++ 生成，ref 哈希，见 [sddmv-benchmark-ref-layout.md](sddmv-benchmark-ref-layout.md)），均匀行长 1000，有放回抽列后行内排序，nnz = 1e7。hybrid 全部走 heavy 路径；TCSR 每行跨 2 个 8192 列 tile。
- 输入：同一调用的第 0 步，发放率 `hz ∈ {10000, 1000, 100, 10, 1}` Hz（`dt = 0.1 ms`，脉冲概率 1 到 1e-4）；活跃行数服从二项分布，实测为 10000、979、84、8、1；`b`（ct）为 U[-2,2)。每个方案的测试对 5 个发放率各跑一次。
- batch（k）= 1，只执行一步。

## 2 参考

- `benchmark/cusparse_timing.cu`（`cusparseSDDMM`，`alpha = 1`、`beta = 0`）以 `duration_steps = 1, warmup = 0, runs = 1` 在同一生成用例上运行，经 `run.run_impl("cusparse", "bool" | "float", ..., copy_dweight=True)` 取回 `C[nnz]`；`bool` 时 cuSPARSE 读到的是 bool 转换后的 float 0/1。
- binary 方案对应 `bool` 参考；float 方案（`float_f32`、hybrid 的 `hybrid_sddmv_float_f32`）对应 `float` 参考，`a` 为正的 U[0.5,1.5)。
- 参考先与 float64 numpy 结果在同一容差下比对，防止参考本身出错使测试失效。

## 3 比较

`np.testing.assert_allclose(got, ref, rtol=1e-7, atol=0)`，float32，逐元素相对容差。每个槽只有一次乘法，正确实现应与 cuSPARSE 逐位一致；非活动行必须严格为 0。

## 4 覆盖的方案

| 目录 | 参数 | dW 输入 |
| --- | --- | --- |
| `dev/hybrid-sddmv` | binary：`event ∈ {bool, float}`；float 算子（`test_float_single_step_matches_cusparse_10k`） | 脏（全 7.0），算子自行清零 |
| `dev/hybrid-sddmv-prezeroed` | binary：`event ∈ {bool, float}`；float 算子 | 全零 |
| `dev/sddmv-prezeroed` | `binary_f32_bool`、`binary_f32_float`、`float_f32` | 全零 |
| `dev/block_sweep` | 基线配置 `Config(256, 16, 4, 2)` | 无（算子自带输出） |

f64 kind 不在范围内（cuSPARSE 参考为 f32）。

## 5 文件

- `benchmark/test/pytest_for_other_operator.py`：共用驱动（工作负载、cuSPARSE 参考、比较），每个发放率只生成一次工作负载，每个 `(kind, hz)` 只生成一次参考。文件名不匹配 `*_test.py`，不会被单独收集。
- 各方案 `test/*_test.py` 新增 `test_single_step_matches_cusparse_10k`。
- 不修改 `benchmark/` 下其他脚本。
