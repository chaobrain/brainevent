# CSR hybrid 配置、CUDA 接入与边界修复

日期：2026-11-04。本文按本次提交的最终状态，合并当天四份设计记录：`csr-hybrid-cuda-assembly`、`csr-spmm-plan-and-layout`、`csr-hybrid-config`、`csr-boundaries-and-api-cleanup`。已替代的早期建议在第 7 节注明，不能作为当前契约使用。

## 1. 目标与完成范围

以 `CSRConfig` 统一 CSR/CSC hybrid 调度参数、primary/mirror 结构计划及 MM 调用计划。构造或显式准备负责持久化读取与具体结构扫描；算子、AD 和 batching 只消费配置快照。新 MV scatter 和 v3 MM 已接入生产调用，旧 CSR hybrid scatter 已删除。

本轮覆盖普通/indexed、binary/weighted、homo/hetero 的 hybrid MV/MM，以及 SDDMV 的共享调度参数传递。float、TCSR、全局 `brainevent.config` 和 SDDMM-MM 独立计划算法不合并。

| 文件/区域 | 最终职责 |
| --- | --- |
| [_csr/config.py](../../brainevent/_csr/config.py) | `CSRConfig`、设备/结构描述、MM 调用计划、校验、纯标量缓存及 pytree。 |
| [binary/hybrid_config.py](../../brainevent/_csr/binary/hybrid_config.py) | `load_csr_config()` / `save_csr_config()`，保持已有 JSON 格式。 |
| [binary/initialize.py](../../brainevent/_csr/binary/initialize.py) | 保留初始化入口；候选与返回类型改为 `CSRConfig`，负载、候选参数、测速和选优算法不变。 |
| [_csr/main.py](../../brainevent/_csr/main.py) | CSR/CSC 持有配置、准备两个方向、传递快照及管理受控重建。 |
| [binary.py](../../brainevent/_csr/binary/binary.py)、[binary_indexed.py](../../brainevent/_csr/binary/binary_indexed.py) | 原语、AD、batching 的配置传递及调用计划派生。 |
| [_hybrid_mm.py](../../brainevent/_csr/binary/_hybrid_mm.py) | 设备查询、模块加载、NB/BN 适配、scratch 与 FFI 检查。 |
| [hybrid_mv](../../brainevent/_csr/binary/hybrid_mv)、[hybrid_mm](../../brainevent/_csr/binary/hybrid_mm) | 包内可加载的原生 CUDA 源码与传递头文件，支持 mirror 权重 perm 读取。 |
| [_sddmm](../../brainevent/_sddmm)、[_fcn/main.py](../../brainevent/_fcn/main.py)、[_tcsr/preprocess](../../brainevent/_tcsr/preprocess) | SDDMV 沿用前向调度快照，FCN indexed CSR 和 TCSR 普通 CSR 重建补齐配置。 |

## 2. 公开接口与保护契约

- `CSRConfig(...)` 直接保存四个 hybrid 参数：`block_size`、`fixed_scatter_blocks`、`tpr_threshold`、`task_nnz`。允许暂未绑定结构，供 benchmark 候选使用；MM 几何与这四个参数独立。
- `CSR/CSC(..., config=None)` 加载持久化参数；显式传入 config 时继承参数并绑定当前结构。保存新配置不改变已有矩阵的快照。
- `M.prepare_config(device=...)` 返回新矩阵，继承当前参数，不读持久化、不修改原对象。对具体结构在 JIT 外调用。
- 裸数组四个 binary CSR 入口的 `config=None` 使用固定默认调度，不读盘、不扫描、不隐式准备。实际进入非空 hybrid MM 才要求结构、mask 和必要设备信息完整，否则明确报准备错误；不静默回退到已删除的旧 MM。
- 零 batch、零输出或零 nnz 无需未使用的计划；JAX、Numba 和保留的非 hybrid gather 不要求 MM plan。MV 被 vmap 转为 hybrid MM 后，同样受准备检查约束。
- 四个入口均删除已忽略的 `workspace`。普通/indexed MM 删除无调度作用的 `max_row_nnz` 及转发包装；MV 保留该提示，作为未准备结构时的可信容量上界，不扫描 indptr。
- `HybridConfig` 及旧读写/辅助接口已删除，不保留别名。持久化白名单仅包含四参数与测速记录，不保存结构、设备、mask 或调用计划。

完整使用契约和示例见 [API 参考](../reference/apis/utilities.rst)。原始数组调用者必须保证配置对应实际结构；相同 shape/nnz 不能证明行分布和 long mask 相同。

## 3. 构造、计划与 JAX 生命周期

```text
CSR/CSC.__init__ 或 prepare_config
  → 选取调度参数快照，校验具体结构，获取目标设备描述
  → CSRConfig.prepare_structure：primary + mirror 统计与 long mask
  → self.config
_stored_event_product
  → 选择遍历方向与普通/indexed 入口
  → p_call → config.mm_plan(batch, dtype, weighted)
  → primitive(mask operand, hashable config descriptor, immutable call plan)
  → hybrid lowering → 完整准备/设备/缓冲区校验 → FFI
```

primary 行长度来自 `diff(indptr)`，mirror 来自 `bincount(indices, minlength=storage_cols)`；复用已有主机数组，避免重复设备回读。完整 mirror 索引和 perm 继续惰性构建。tracer 结构不进行主机扫描，不能继承不适用的精确计划。

目标设备来自显式参数或结构数组的单一设备，不直接选择 `jax.devices()[0]`。CPU/非 CUDA 不强制加载 CUDA。实际 hybrid MM 校验所依赖的设备特征；换设备需要显式准备。

完整 config 是普通 pytree：mask 为数组叶，参数/结构统计/几何/设备描述为静态元数据。primitive 不以整个含数组对象作为 static 参数。MM 调用计划使用有界纯标量缓存（256 项），不缓存 tracer 或 scratch，不在 tracing 中回写矩阵。

| 场景 | 最终行为 |
| --- | --- |
| `with_data`、数据算术、逻辑 `.T` | 共享配置与结构计划；逻辑转置不交换实际压缩布局身份。 |
| `tocsc/tocsr` 物理格式转换 | 交换 primary/mirror 计划，维护重排后的权重与 perm。 |
| 受控切片、插入对角线等结构变化 | eager 新结构重新准备，traced 新结构使精确计划失效。 |
| `_from_parts`、pytree 恢复 | 纯装配，不扫描、不读盘。 |
| 初始传入 mirror / 后续替换 mirror buffer | 具体初始 mirror 在准备阶段绑定；后续替换使对应计划失效。 |
| RHS vmap、MV→MM、MM batch 折叠 | 重入 p_call，按变换后的 batch 派生计划。 |
| weights-only vmap、JVP/VJP、indexed 重入 | 携带同一结构描述与相关数组，按新调用规格重新规划。 |
| 不同 indptr 的 batching | 不新增异结构精确 hybrid 计划批处理；清除共享计划，实际 hybrid MM 报准备错误，其他后端保留能力。 |

`precompute_weight_indices=False` 只关闭构造时预建 mirror，并不禁止惰性构建；traced 参数需要在外部携带所需 mirror。任意直接赋值主结构属性的缓存一致性仍属第 8 节的已知缺口，上表不对这种绕过正式入口的修改作保证。

## 4. MM 几何与工作区公式

令当前遍历行长为 `L`，固定 `K=16`、task 长度 `T=32*K=512`。Python 与 CUDA 保持相同整数规则：

1. 校验维度、offset 单调性与末项 nnz、dtype、整数规格及容量；空尺寸单独处理。
2. 只取 `0 < L <= 512` 的行。先算 `a=floor(sum(L)/count)`，再算 `q=(a+3)//4`；按 `q<=4/8/16` 选 `G=4/8/16`，否则 32；无候选行取 32。例如 `[16,17]` 得 `G=4`。
3. 长行阈值 `H=K*G`；`L>H` 的位置写入 `uint32[ceil(rows/32)]` 位图。
4. 单 batch 长任务数 `Q=sum(ceil(L/T), L>H)`。
5. 非空调用 `C=max(1,min((L2//4)//(cols*value_bytes),batch,65535))`；容量 `P=C*Q`，组数 `groups=ceil(batch/C)`。
6. binary 工作区字节数为 `align8(8*batch)+8*groups+12*P`；weighted 额外增加 `align4(P*value_bytes)`。工作区使用 uint32，short-row 表需要 `batch*rows` 个 int32。低精度事件队列保持四字节对齐。

primary/mirror 使用各自的行分布。权重数值和事件活动率不决定结构计划。相同 shape/nnz 的 `[513,512]` 与 `[1025,0]` 可分别需要 2、3 个长任务；仅缓存容量而丢掉源统计不足以在 batch/dtype 改变后重新规划。容量乘法和对齐均有溢出检查，int64 offsets 不代表无限尺寸。

## 5. CUDA 接入、布局与删除范围

MV direct/mirror scatter 使用包内 `hybrid_mv` 和同一组 `BE_HYBRID_*` 调度定义；初始化 benchmark 编译同一新 MV 源文件。MM 使用包内 v3 binary/weighted、ordinary/permuted 模块；hetero mirror 按 `weights[perm[j]]` 读权重，homo 复用普通模块而不读取 perm。

生产 CSR 普通/indexed MM 原语仍使用 NB：输入 `(rows,batch)`，输出 `(cols,batch)`。原生 v3 使用 BN，adapter 执行 `event_abi.prepare(events).T`，返回 `output_bn.T`。`col_major` 描述稀疏遍历方向，不改变 dense BN 约定；TCSR 的布局规则也不能直接套用到普通 CSR。

scratch 在 lowering 中按调用建立，mask 来自已准备结构。普通 MV 每次调用都清零 status，包括零行/零边的早退；新 MM 空调用只要求正确的零输出，不能把未定义的内部 scratch 状态当作公开结果。

已删除四个无生产引用的旧 CSR hybrid 文件：`binary_csrmv_hybrid.cu`、`binary_indexed_csrmv_hybrid.cu`、`binary_csrmm_hybrid.cu`、`binary_indexed_csrmm_hybrid.cu`。保留有调用者的普通/indexed gather、float、TCSR 独立实现。新源码的本地 include 与 package-data 覆盖包内传递依赖。

逻辑转置是否变成物理拷贝由 XLA 编译结果决定，需要端到端测量，不能仅凭 `.T` 的数量判断性能。

## 6. 边界修复与接口清理

- 零维 `(0,0)/(0,n)/(n,0)` 贯通 mirror、`build_weight_indices()`、`diag_add()` 与对象级 event MV/MM；空 mirror 无需设备查询或 CUDA 启动。
- `CSR/CSC.solve()` 在求解前拒绝非方阵；裸 `csr_solve()` 在空系统返回前也检查 RHS 必须为 `(n,)` 向量。
- ordinary MV binary/weighted、homo/hetero 在空输入前清零 status；新增 dirty scratch 和活动→静默→恢复的复用回归。
- 删除无效参数、兼容警告、失效测试及 MM 空转发层；修正文档的结构 dtype、实际后端、事件活动语义、dt2t 反向求导及 CUDA 文件名。

这些边界问题均先写测试复现，再修改实现；后续变更需同时核对对象入口、原语早退、AD/batching 和文档，避免仅底层支持空输入而对象链仍拒绝。

## 7. 当天早期方案如何演进

| 早期阶段提议/状态 | 最终处理 |
| --- | --- |
| CUDA 装配阶段先迁移 MV，MM 生产暂留旧实现 | 完成 config/plan/adapter 后，MM 已切入 v3。 |
| 在 `hybrid_config.py` 建独立计划，不扩展 `HybridConfig` | 改为 `_csr/config.py` 的统一 `CSRConfig`；旧类型删除，持久化独立。 |
| 由 StructureCache 持有 MM 计划 | 计划归 config，结构缓存继续管理 mirror 等结构资产。 |
| traced 缺计划时回退旧 MM | 实际 hybrid MM 明确报准备错误；动态位图/保守计划路径未实现。 |
| 保留旧 scatter 等待迁移 | 生产与调参调用迁移、回归通过后删除；仍需要的 gather 保留。 |
| NB/BN adapter、计划缓存、AD/batching、空 MV status 尚待接入 | 均已完成；公开 BN 接口切换和性能优化仍待组织。 |

## 8. 未实施的重构与能力边界

| 问题 | 原因/影响 | 后续方向 |
| --- | --- | --- |
| 直接修改主结构属性可留下旧缓存 | `indices/indptr/shape` 可写，`prepare_config()` 仍可能沿用旧 mirror，存在结果错误风险。 | 首先明确只读结构或统一替换入口；同步失效 config、mask、mirror、perm、diag。 |
| 自动与显式 CUDA 的 mirror 路由不同 | 对象路由仅识别显式 `cuda_raw`，`None` 到原语层才解析。 | 统一实际后端与遍历选择；保留缺 mirror 时的明确策略，不能先删 gather。 |
| 设备查询加载整个 MM 模块 | `_query_device()` 为 SM/L2 调用 `load_module()`。 | 独立轻量查询，保留真实设备绑定和缓存；避免 MV-only 构造承担 MM 冷编译。 |
| MM 性能与显存未独立验收 | short-row scratch 按 `batch*rows`，每次调用建立，现有初始化调参只覆盖 MV。 | 分测构造、首次编译、稳态、布局和峰值显存；测量后再决定 scratch 改造。 |
| CSC 类移除 | 逻辑转置、物理重排和外部格式接口尚需统一契约。 | 同步 CSR、Dense、FCN、TCSR、JITC 转换和 pytree；删除类不等于删除 mirror/perm。 |
| 公开/原语 BN 切换 | 当前公开 NB 与原生 BN 通过 adapter 对接。 | 先定切换边界，保持矩阵乘法语义；同步全部后端、AD、RHS vmap、MM 折叠及调用计划轴。 |

超大规模 diag offset/perm 的 int64 全链路、不同 indptr 的精确 hybrid vmap、solve vmap、稀疏乘稀疏、新 slice 后端按需独立排期。现有 SDDMM 独立算法问题不因本次配置传递而视为解决。

建议先固定正确性/性能基线并解决结构绑定，再独立组织 CSC 和路由、BN 切换；设备查询可并行解耦。MM 峰值显存、最低版本 GPU 与干净安装仍需验收。本次不修改这些重构项。

## 9. 验证记录与范围

以下是不同阶段的已完成运行，存在范围重叠，**不能相加为一次最终全仓测试**。主要环境为 JAX 0.11.0、CUDA 13、RTX PRO 6000 Blackwell。

| 阶段 | 已完成验证 | 范围说明 |
| --- | --- | --- |
| 初始 CUDA 装配 | 分次选择性测试 771 项；四个 MM translation unit 与 standalone C++ caller 编译链接通过。 | 当时生产 MM 尚未迁移，不能当作最终生产 MM 回归。 |
| 原始 MM/布局检查 | loader-built col-major binary/weighted 非默认 stream 原生执行；memcheck 0 错误/泄漏、racecheck 0 hazards；另有 NB→BN→NB 矩形/perm/长行等 32 组验证。 | sanitizer 面向当时原生 caller，布局验证也早于生产接入。 |
| 配置迁移阶段 | 完整普通/indexed CPU 699/597、GPU 1026/602；GPU adapter/FFI/变换联合 125；CSR/CSC、FCN、生命周期 GPU 分片覆盖 1019。 | 对应阶段版本的套件；后续 API 清理删除了失效测试，计数已变化。 |
| 共享链与兼容性 | SDDMV/dispatch CPU 356 passed、1 skipped；结构/索引/TCSR 预处理 85 passed；JAX 0.9.0 CPU config/生命周期/集成 137 passed、12 skipped，float8 专项另 8 passed。 | 跳过项为对应环境未运行的用例；不代表 JAX 0.9 GPU 验收。 |
| 最终 API/边界清理 | 两套 binary 算子与 config 集成 CPU **1369 passed、38 skipped**（含 slow）；最终 MM 入口 GPU 集成 **82 passed**。 | 最新清理后的选定完整文件范围，不是全仓测试。 |
| 最终专项 | 零尺寸 GPU 54 passed，CPU 与 JAX 0.9 均 42 passed、12 skipped；既有 misc/diag/config 97 passed、2 skipped；solve 71 加对象 solve 7 passed；MV FFI 22 加公开 CUDA 长行/int64 offset 10 passed。 | 38 个 solve 非法输入及 16 个 dirty-status 空调用案例先失败后通过。 |
| 最低版本与静态检查 | JAX 0.9.0 最终入口/batching/单位测试 50 passed；项目固定 mypy 2.4.0 检查 174 源文件通过；diff whitespace 检查通过。 | CPU 环境禁用了不兼容的 CUDA 0.11 插件并产生提示；未用该组合执行 GPU。 |

配置模块覆盖率 100%、持久化模块 91.3%（配置迁移阶段）；最终 `spsolve.py` 100%、`diag_add.py` 98.04%。diag 数字合并正常 Numba 运行及 `NUMBA_DISABLE_JIT=1` 实际执行 Python 主体的数据，不代表 CUDA 或整个 CSR 包的覆盖率。

初始装配阶段做过 wheel 资源检查（当时 18 个 MV/MM 资源），初始化单测与 GPU 小负载 smoke 通过。当前 package-data 覆盖 19 个 MV/MM CUDA 源码/头文件；资源发现与本地编译不替代干净安装、全部支持 CUDA/GPU/JAX 组合和性能验收。
