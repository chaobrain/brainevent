# Hybrid MV 全局原子加瓶颈 NCU 记录

## 目标

判断 `new_mv_operator/baseline/binary_csrmv_hybrid.cu` 的全局 `atomicAdd`（SASS 为 RED）
是否为吞吐瓶颈。RED 无 scoreboard 依赖，warp stall 采样基本看不到它，因此以硬件单元
利用率为主要证据，而不是 `lg_throttle` 或 long scoreboard。

结果用于回答：TileMM 把全局原子改成 shared 累加，收益是否可能来自原子加本身。
本记录只做第一步（单元利用率与外推），不修改 CUDA 源码；消融另行立项。

## 工作负载

与 TileMM `baseline_b128` NCU 记录对齐的单向量 MV：

| 参数 | 值 |
| --- | --- |
| M=N | 100,000 |
| 密度 | 0.01 / 0.03 / 0.1（每行固定度数 1000/3000/10000，Gini=0） |
| 活动行 | 恰好 3000 行（事件率 0.03），bool |
| 权重 | float32 异质 ±1（累加精确） |
| 索引 | int32 列索引，行内无放回、排序；int32/int64 行指针 |
| TPR 阈值 / task nnz | 128 / 4096，三个密度全部为重行，由 BlockKernel 完成 scatter |
| seed | 0，`SeedSequence(0).spawn(4)` |
| warmup / 正式调用 | 10 / 2 |

每个密度一份报告。物理 GPU 3（RTX PRO 6000 Blackwell，L2 128 MB，188 SM）。

## 记录结构

遵守 `brainevent/_tcsr/dev/ncu-record-standard.md`，目录：
`new_mv_operator/ncu/<timestamp>_full_random_n1e5_p3e-2_mv_atomic/`

| 文件 | 职责 |
| --- | --- |
| `profile_two_steps.py` | 固定工作负载，`--density` 选择三档之一，NVTX 两步，正确性检查 |
| `profile_two_steps_test.py` | 配置、输入生成、调用链、捕获脚本守卫 |
| `capture_all.sh` | GPU 3，`--set full` + 额外指标，导出 details 与 raw CSV，校验 kernel 集合 |
| `manifest.md` | 环境、命令、源码与产物哈希、完整性 |
| `result_summary.md` | 指标表与判读 |

## 额外指标

- L2 原子 ALU：`lts__d_atomic_input_cycles_active` avg/max pct of peak
- L2 事务：`lts__t_{requests,sectors}_op_{red,read,atom,write}`、red/read 命中率、`lts__t_sectors` pct
- crossbar：`lts__xbar2lts_cycles_active`、`lts__lts2xbar_cycles_active`、
  `l1tex__m_l1tex2xbar_req_cycles_{active,stalled}`、`l1tex__m_l1tex2xbar_write_sectors_mem_global_op_red`
- L1 合并：`l1tex__t_{requests,sectors}_pipe_lsu_mem_global_op_red`、`smsp__inst_executed_op_global_red`
- stall：drain / lg_throttle / long_scoreboard 每发射比例及平均发射间隔

## 判读

任一 RED 相关单元接近峰值即为吞吐瓶颈。否则用外推 `u × r_mm / r_mv`（u 为原子 ALU 利用率，
r 为边处理速率）估计 TileMM 速度下改用全局原子时所需利用率。注意 MV 每边 8 B（int32 索引），
TileMM 每边 6 B（uint16 局部列），读侧不可直接等同。
