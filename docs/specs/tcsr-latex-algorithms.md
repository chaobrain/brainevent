# TCSR 算法 LaTeX 说明与正确性审查

## 范围

按 `brainevent/_tcsr/dev/Latex_alg/algorithms.tex` 的 `algorithm2e` 风格，
说明 Hybrid SpMV、Tile SpMM、binary/float SDDMM，以及同目录单向量
SDDMV 的运行机制。修订伪代码并生成 PDF，不修改 Python/CUDA 计算实现。

按用户要求，所有改动位于当前 `mm-optimizing` 分支。`Latex_alg` 被
`dev/` 忽略规则覆盖，因此需将完成的 LaTeX 源文件和 PDF 显式纳入
当前分支的版本控制视图。

## 数学与存储约定

- CSR 物理形状为 `(m,n)`，每个存储槽 `j` 独立，重复列索引仍是不同槽。
- 行指针为 `p`，列索引为 `c`，异质权重使用 `w[j]`，同质使用 `w[0]`。
- TileCSR 每行列索引非递减；基础 tile 宽度为 8192，`lambda[j]` 为局部列号，
  `Theta[r,t]` 为行内相对槽边界。槽号必须加 `p[r]`，全局列号必须加 `8192*t`。
- 前向事件算子只累加活动行的权重。浮点事件以 `x > 0` 判活；Tile SpMM
  CUDA 的 int8 输入已由 Python 转为 0/1，CUDA 内检查 `!= 0`。
- SDDMM 输出权重梯度，不读取前向权重：binary 为
  `dw[j] = sum_b [X[b,r] > 0] * G[b,c[j]]`，float 为
  `dw[j] = sum_{b: X[b,r] != 0} X[b,r] * G[b,c[j]]`。
  float 不丢弃负值。实数公式与普通内积的等价性限于有限输入；跳过零值
  对 NaN/Inf 的传播可能不同于稠密乘法。
- batch API 支持 4、8、16、32、64、128、256、512；单向量走独立 SDDMV API。
  SDDMM/SDDMV 低层接口均要求 `transpose=True`。float 的 X 与 G 类型相同，
  均为 float32 或 float64；binary 的事件类型组合按 Python 分派表说明。

## 已发现的文档缺口与修订

1. 原 Hybrid 伪代码未定义 `active`。依据 `include/cuda_common.h` 明确
   bool/int8 存储路径为非零，float32 路径为正值；不能套用文件头的非零描述。
2. 原 Hybrid 容量截断可能被误读为溢出后仍正确。明确容量由静态结构预计算，
   并必须覆盖全部重行任务。队列溢出时 CUDA 设置状态位，但消费者不检查该位；
   不足容量不在正确性保证范围内。`Q=0` 跳过第二个 kernel。
3. 原 Tile SpMM 固定共享数组长度 8192，只覆盖 float32。float64 的共享预算
   仍为 32 KiB，计算 tile 为 4096，通过二分边界将基础 tile 分成两半。
   补充局部下标减去半块起点，以及尾 tile 有效长度。
4. 原 Tile SpMM 仅写 `v[j]`，遗漏已实现的同质 `v[0]` 路径。
5. 原 ballot 表达式中的 lane 作用域不清晰。将 lane 集体投票、段计数、
   exclusive scan 与活动行写出分开表达，保证每个活动行恰好占一个槽。
6. 补充 SDDMM 的 128-batch phase、每行 4 个 uint32 掩码、4 个基础 tile
   构成 macro tile、16 线程分组和 CSR 槽唯一写入者。
7. binary 以槽为外层累计 phase 局部和；float 以活动 batch bit 为外层，
   先压缩每个 128 行 chunk 的活动行，再广播输入值、成对访问槽。
   float 首次活动 phase 的首个 bit 对当前 macro tile 各槽赋值，之后累加。
   该描述忠实反映赋值/累加顺序；输出仍然预先清零，不能误称此优化替代了
   清零。后续 phase 重新赋值才会覆盖已有的有效贡献。
8. macro tile 与 phase 的 kernel 在同一 stream 依次提交。不能将 phase
   画为并行且仍使用普通 `+=`。多 batch kernel 网格是 `ceil(m/128)`；
   只有掩码等辅助网格和单向量 panel 网格使用 `2*SM` 上限。

这些修订源于以 dtype、输入语义和实际 launcher 为依据的逐路径核对。
今后调整内核常量、分派或循环顺序时，应同步检查算法的输入约定、并行层级
和边界，不仅根据内核名称或文件头注释更新文档。

## 实现文件映射

| 内容 | 实现 |
| --- | --- |
| Hybrid 提取、任务队列、scatter | `brainevent/_tcsr/binary_csrmv_hybrid.cu` |
| 判活宏 | `brainevent/include/cuda_common.h` |
| Tile 计数、scan、压缩、scatter | `brainevent/_tcsr/tile_mm/bn_tile_kernels.cuh` |
| Tile 类型与形状检查 | `brainevent/_tcsr/tile_mm/binary_csrmm_tile.cu` |
| 排序、基础 tile 元数据 | `brainevent/_tcsr/preprocess.py` |
| 前向事件规范化、同质梯度归约 | `brainevent/_tcsr/binary.py` |
| SDDMM 输入、布局、scratch 与分派 | `brainevent/_tcsr/sddmm/sddmm.py` |
| 多 batch binary / float | `brainevent/_tcsr/sddmm/sddmm_binary.cu`、`sddmm_float.cu` |
| 单向量 binary / float | `brainevent/_tcsr/sddmm/sddmv_binary.cu`、`sddmv_float.cu` |

## 验收与边界用例

- LaTeX 编译成功；交叉引用正确；无超出页面的算法、公式或代码路径。
- 用现有 SDDMM API/dtype 测试检查文档描述的输入语义；GPU 数值结果与
  手算公式一致。记录实际运行环境与结果，不把 Python mock 测试当作内核验证。
- Hybrid：空结构、全静默、行长 0/128/129/4096/4097、同质与异质权重、
  重复列、多块任务和恰好足够的队列容量。
- Tile：m 跨越 32、256、4096；n 跨越 4096、8192；float32/float64；
  空行、全静默、重复列、尾 tile、同质权重。当前 Tile MM 拒绝零 nnz。
- SDDMM：batch 4/128/256/512，活动 bit 位于 31/32/127/128/255/256/511；
  行数 127/128/129；列跨基础 tile 和第 4/5 个基础 tile；槽数量 15/16/17/31/32/33；
  首个活动 phase 非零、全静默行、正负值抵消、重复列、零 nnz。
- SDDMV：活动行压缩跨 256 行块边界，binary 负值不活动，float 负值参与乘法。
- 错误输入：非排序/不一致 tile 元数据，以及不足的 Hybrid 任务容量属于前置
  条件违例；不声称现有低层接口能检查所有元数据内容或安全恢复。

仅修改文档，不新增生产代码或新的 Python 模块，故覆盖率与 mypy 不作为
本次文档变更的度量；不改变 JAX >= 0.8.0 的兼容性要求。

## 验证结果

2026-10-07，使用现有 `oper-benchmark` 环境的 JAX 0.11.0，在物理 GPU 3
上关闭 JAX 显存预分配后执行：

```bash
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/luruheng/miniconda3/envs/oper-benchmark/bin/python -m pytest \
  brainevent/_tcsr/test/sddmm_test.py \
  brainevent/_tcsr/sddmm/test/sddmm_dtype_test.py -q --disable-warnings

CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/luruheng/miniconda3/envs/oper-benchmark/bin/python -m pytest \
  brainevent/_tcsr/test/binary_test.py -q \
  -k 'cuda_tile_mm_matches_dense_across_f64_subtile_boundaries or cuda_tile_mm_f64_bptt_weight_gradient_matches_dense_reference or cuda_raw_sampled_weight_gradient_matches_dense_reference' \
  --disable-warnings
```

第一组 40 项通过，第二组 21 项通过、42 项未选取，无跳过或失败。
数值检查实际执行 CUDA；API 路由检查部分使用 mock。覆盖 binary/float
SDDMM 和 SDDMV、支持的类型分派、正负事件、Tile float32/float64、
同质/异质权重、重复列、4096/8192 列边界，以及两种方向的权重梯度。
更大 batch 的跨 phase、跨 macro tile 和极端队列容量仍是上述建议用例，
本次未将这些条件宣称为已动态验证。

LaTeX 使用 `latexmk -pdf -interaction=nonstopmode -halt-on-error` 编译为
8 页 PDF，包含 7 个 `algorithm2e` 算法块。最终编译日志无 warning、
overfull/underfull box 或未定义引用；已查看符号约定、Tile 累加、float
SDDMM 和 SDDMV 页面的渲染。`git diff --check` 通过。

在已执行的数值用例中未复现内核计算错误。静态审查的正确性结论依赖合法
CSR/TileCSR 元数据、足够的任务容量、可表示的索引算术和保持顺序的 kernel
提交；文档明确了这些前置条件，没有将容量截断描述为错误恢复机制。
