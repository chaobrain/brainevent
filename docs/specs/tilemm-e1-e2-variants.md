# TileMM 单变量对照：全局原子（E1）与错位扫描（E2）

## 目标

把 TileMM 相对顺序 MV 的收益拆成可单独测量的来源，每个实验只改一个变量，
grid、线程映射、occupancy 和总工作量都保持与正式 baseline 相同。

| 实验 | 对比 | 唯一变量 | 回答的问题 |
| --- | --- | --- | --- |
| E1 | baseline vs `global_atomic` | 累加位置：shared 还是全局 RED | 写的拥塞是否是主要开销 |
| E2 | baseline vs `rotated_rows` | 并发 batch 访问 W 的时间对齐 | 碰巧发生的跨 batch L2 读共享贡献多大 |

本轮先做 E1；E2 的算子同时实现并测试，计时另行进行。

## 目录与接入

两个变体各自一个目录，沿用 `warp_tile` 的组织方式：`kernels.cuh` 复用正式
`bn_tile_kernels.cuh` 的活动行提取，只定义自己的 scatter 与 launch；`operator.cu`
用 `#define bn_tile <ns>` 复用正式 `binary_csrmm_tile.cu` 的 ABI 检查和四个入口。

```text
new_tile_operator/dev/global_atomic/{kernels.cuh, operator.cu, global_atomic_test.py}
new_tile_operator/dev/rotated_rows/{kernels.cuh, operator.cu, rotated_rows_test.py}
```

`runtime.py` 增加这两个变体名：从 `dev/<variant>/` 编译，编译名包含本目录与正式
源码的哈希，避免复用过期的编译缓存。Python 接口、反向传播、输入契约与 baseline
完全相同。

## 内核差异（相对 `bn_tile::scatter_kernel`）

`global_atomic`：
- 去掉 shared 清零、`__syncthreads` 和最后的写回；
- 每条边执行 `atomicAdd(&output[b*cols + tile_begin + local], value)`，返回值不用，编译为 RED；
- launch 前对整个输出做 `cudaMemsetAsync`，计入耗时，这是全局累加本身的成本；
- **仍申请 32 KiB 动态 shared**，保证每个 SM 驻留的 CTA 数与 baseline 相同。

`rotated_rows`：
- 每个 CTA 从活动行表的第 `shift(b)` 项开始循环扫描，其中
  `shift(b) = floor(frac(b·φ⁻¹) · count)`，φ⁻¹ 取 32 位定点的 2654435769/2³²；
- 相邻 batch 的起点相隔约 0.38 或 0.62 个列表长度，同一 wave 内并发的 batch
  均匀分散在整个活动行表上，不再同时读同一段 W；
- 每个活动行仍然恰好处理一次，只是顺序不同；其余代码不变。

已知局限：错位只破坏"时间对齐"。当读复用距离小于 L2 容量时（低密度），
即使错位，L2 仍可能保留部分跨 batch 复用。因此 E2 测得的是"对齐相关"的那部分共享。

## 测试（与 `operator.cu` 同级的 `<variant>_test.py`；用目录名命名，避免 pytest 模块名冲突）

- 与稠密参考比较：跨 8192 列的 tile 尾部、重复列、负事件、float32/64、同质/异质权重；
  f64 会覆盖一个 metadata tile 拆成两个计算 tile 的中点二分路径；
- 全静默事件得到精确零（`global_atomic` 依赖 memset），空结构；
- 活动行数跨越 128/256/4096 等边界，并覆盖多个 batch。对 `rotated_rows`，
  这同时检查了取模回绕；
- 整数权重下与 baseline 逐位相等（求和顺序不同也必须精确一致）；
- batch=1、非 2 的幂 batch；JIT 与权重梯度；
- 新增 Python 代码 mypy 无错误。

## E1 计时

复用 `benchmark.py`：GPU 3，M=N=1e5，B=128，float32 异质权重，seed=101，
d ∈ {0.003, 0.01, 0.03, 0.1} × p ∈ {0.003, 0.03}，预热 10 次，40 轮随机交错执行，
与 `results/20261008_154828_136114` 的 baseline_b128 配置相同。

判读：加速比 = global_atomic / baseline 的耗时比，配对样本 ≥75% 达到 1.05 即为稳定差异。

- `global_atomic` 明显变慢：写拥塞是 TileMM 收益的主要来源，shared 累加必须保留；
- 两者相当：写不是压力，收益来自并行度或读侧。

E1 的 NCU（`--clock-control none`，加上原子与 crossbar 指标）在计时结果确认后另行捕获。

## 顺带修复

`dev/` 重组后，`benchmark.py` 和 `test/conftest.py` 找不到 `adaptive`。这里只补上
`dev/` 的搜索路径，并把 `dev/` 下的源码纳入 `benchmark.py` 的源码哈希。
`edge_split`、`subtile` 的 CUDA 相对 include 和 runtime 路径同样失效，但本轮不修改，单独报告。

## E3：batch_major（只交换 grid 维度）

E2 证明了"共同扫描前沿"带来的跨 batch L2 读复用。波次模型认为：在现在 tile 维变化最快的 grid 下，
每一波只有约 43 个 batch，各自覆盖全部 13 个 tile，所以 W 每一波都要从 DRAM 重读一遍，
预测 DRAM 读量约为 2.2 倍 W（实测 2.4–2.8 倍）。

`dev/batch_major/`：只把 `blockIdx.x`/`blockIdx.y` 的含义对调，launch 改为 `dim3(batch, tile 数)`。
按硬件实际的线性发射顺序，同一时刻驻留的是 128 个 batch × 约 4.4 个 tile，
λ 从 1.29 升到 3.84，每个 tile 的 W 预期只读一次。其余代码不变。由于 tile 数放到了 y 维，
tile 数超过 65535 时（cols > 65535×8192）launch 返回错误，实际规模远不会触及这个上限。

### NCU 记录

用 `ncu/capture.py` 新增的 `--extra-metrics`，为 baseline 与 batch_major 各捕获一份：
p=0.03，d ∈ {0.01, 0.03, 0.1}，B=128，seed=101，`--set full`、`--clock-control none`。
额外指标：DRAM 读字节/吞吐、L2 读 sector 与读命中率、crossbar 上下行、L1→xbar 请求受阻、
long scoreboard / lg_throttle。核心比较量是 scatter kernel 的 `dram__bytes_read.sum` 除以
W 的字节数（nnz×6）。

### 计时

与 E1/E2 相同的 8 个 case，比较 baseline 与 batch_major。

### 测试

`dev/batch_major/batch_major_test.py`：与 E1/E2 相同的边界、静默、活动行边界和逐位相等测试。
`ncu/test/capture_test.py`：新增变体的完整性检查，以及额外指标确实进入 NCU 命令。
