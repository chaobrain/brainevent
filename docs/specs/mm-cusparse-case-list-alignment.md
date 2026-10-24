# MM benchmark 用例列表对齐 cuSPARSE 与 tile

日期：2026-10-22。分支：`mm-optimizing`。

## 范围

直接修改 `benchmark/benchmark_binary_csr_matmat.py` 的 `CASE_LIST`，对齐
`baseline/runs/nohup_gpu2_20261016_214626/data/cusparse/case_1.csv` 的规模、矩阵密度与 batch，
向量密度采用同一 data 目录下 `tile/case_1.csv` 的范围。

- scale：100、200、400、600、800、1000、1200、1400、1600、1800、2000。
- 矩阵密度：3e-5、1e-4、3e-4、1e-3、3e-3、1e-2、3e-2。
- 发放率：10、30、100、300、1000 Hz；现有 dt=0.1 ms 下对应 tile CSV 的
  density_vec=0.001、0.003、0.01、0.03、0.1。
- batch：2、4、8、16、32、64、128、256、512。
- 保留 dispatch、f32、seed=101；统一沿用现有 duration_steps=20、warmup=2、runs=5。

## 验收与边界

展开后的 3465 个唯一组合应覆盖 693 个原有规模/矩阵密度/batch 组合与 5 档向量密度
的笛卡尔积，全部通过参数校验；不要求数量等于已经过滤过的 tile CSV。
逐项核对计时参数及向量密度两端 0.001、0.1 的单位换算。
现有 96 GiB 显存估算过滤预计保留 2475 个、跳过 990 个，
不修改过滤逻辑，也不启动全量 GPU benchmark。
本次仅修改配置，无新增运行逻辑；使用完整组合核对与 mypy 验证。
