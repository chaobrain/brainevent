# hybrid SDDMV 的 float 计算型算子与 benchmark 事件类型参数

日期：2026-10-16。分支：`mm-optimizing`。修改 [sddmv-benchmark-ref-layout.md](sddmv-benchmark-ref-layout.md) 的实现划分与事件生成。

## 1 两种计算型

| 计算型 | 事件 | 活跃判断 | 写入活跃行的每个槽 |
| --- | --- | --- | --- |
| binary | bool 0/1 | `event != 0` | `dW[j] = ct[col]`（抄写，不读事件值） |
| float | float32 | `event != 0` | `dW[j] = event[row] * ct[col]`（必须带事件值） |

TCSR 已有两份文件（`sddmv_binary.cu` / `sddmv_float.cu`）。hybrid 按同样方式新增 float 文件。

## 2 hybrid float 文件

- `dev/hybrid-sddmv/hybrid_sddmv_float.cu`：复制 `hybrid_sddmv.cu`；`dev/hybrid-sddmv-prezeroed/hybrid_sddmv_prezeroed_float.cu`：复制 `hybrid_sddmv_prezeroed.cu`。两者之间的差别与 binary 两份之间相同（只有 dW memset）。
- 机制与 binary 相同：Extract 按行判断活跃；`nnz ≤ 128` 的行由一个线程写完；长行按 4096 槽切成 task，Block kernel 用固定数量的 block 以 grid-stride 处理 task。
- 只改数据：
  - 短行：Extract 已读到 `v = vector[row]`，写 `v * ct[col]`。
  - 长行：task 只存 begin/end。Block kernel 中线程 0 用 `upper_bound(indptr, begin) - 1` 求出行号，读 `v` 后经 shared memory 广播给全 block。task 缓冲区与 binary 相同，`prepare_workspace` 不变。
- 入口：`hybrid_sddmv_float_f32`（事件 f32，ct / dW f32），参数列表与 binary 入口相同。
- Python：`hybrid_sddmv.py` 增加 `make_hybrid_sddmv_float(nnz)`，`hybrid_sddmv_prezeroed.py` 增加 `make_hybrid_sddmv_prezeroed_float(nnz)`，事件须为 float32。

## 3 benchmark：一个事件类型参数

- 每个 case 增加 `EVENT_TYPE_LIST`（如 `["bool", "float"]`，同 ref 的 `VALUE_DTYPE_LIST`）。实现名只保留方案名：`baseline, sddmv_prezeroed, hybrid, hybrid_prezeroed, cusparse`。删除 `*_binary_bool / *_binary_f32 / *_float` 与 `cusparse_binary_f32 / cusparse_float`。
- **bool**：初始化只生成 bool 0/1。TCSR 与 hybrid 调 binary 文件的 `*_f32_bool` 入口；cuSPARSE 在窗口外把该 tile 的 bool 转成 float 0/1 后调用 SDDMM。
- **float**：初始化只生成 float，活跃位置取正的 U[0.5,1.5)，其余为 0。TCSR 调 `tcsr_sddmv_dweight_float_f32_t`，hybrid 调 `hybrid_sddmv_float_f32`，cuSPARSE 直接使用。
- 两种类型的脉冲位置相同（同一 `kSpikeSeed` 判定）；float 值：`h = mix64(key ^ 0x632be59bd9b4e019)`，`v = 0.5 + (h>>40)·2^-24`。
- `(scheme, event_type)` 各编译一个 `.so`，cuSPARSE 一个 `.so` 处理两种类型。CSV 增加 `event_type` 列，`IDENTITY_FIELDS` 含 `event_type`。
- `run.generate` 返回 `bool`、`float`、`ct` 三个 `[steps, n]` 数组。

## 4 测试

- hybrid float：短行、阈值行、多 chunk 行、空行、链式 donate、零事件；与 TCSR float 及 cuSPARSE float 逐位一致。
- 10k 正确性测试（`pytest_for_other_operator.py`）：bool 组以"bool 转 float 后的 cuSPARSE"为参考，float 组以 cuSPARSE float 为参考；hybrid 两个方案均加入 float 组。
- benchmark：每个 `(scheme, event_type)` 跨 tile 的多步结果与 numpy 一致；bool 下 cuSPARSE 的输入为 bool 转换结果。
