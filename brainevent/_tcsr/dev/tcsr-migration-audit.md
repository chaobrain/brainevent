# TCSR 迁移与全类型隔离审计

日期：2026-10-26。对应方案：[文件归属与迁移计划](tcsr-structure-plan.md)。

TCSR 目录迁移已完成。本轮没有从 dev 提取、恢复或导入算子实现，正式计算逻辑保持不变。
发现并修复了原有的 dev 打包泄漏和 tile 头文件漏打包问题。
全局开启 x64 的扩展检查发现了三组既有梯度类型问题，详见第 5 节；不能据此宣称所有 dtype 组合均已通过。

## 1. 文件归属与来源

| 范围 | 实施结果 |
| --- | --- |
| 根对象层 | `__init__.py`、`main.py`；对象测试进入 `test/` |
| 事件计算 | `binary/binary.py`、三个 hybrid CUDA；tile 资源进入 `binary/tile_mm/` |
| 数值与逐槽位计算 | `float/`、`dt2t/`，各自同目录保存实现及 CUDA |
| 结构支持 | `preprocess/` 同时保存结构构建、镜像/workspace 和运行时配置 |
| 权重梯度 | `sddmm/` 保留正式 dWeight API，测试进入 `sddmm/test/` |
| 实验输入梯度 | `bpsa/` 保留原有显式 API，测试进入 `bpsa/test/`；稳定对象不自动导入 |
| 开发归档 | `resource_monitor.py` 进入 `dev/tools/`；重复 `csr_to_csc.cu` 进入 `dev/legacy/` |

以本轮实施前工作区为基线，而非 Git HEAD：工作区已包含此前 CSR、Dense、FCN、JIT
迁移及用户修改。保存了全包 **381 个文件**的哈希，其中 **84 个位于 dev**，并复制了迁移前源码。

- 36 个 TCSR 原文件逐一映射，24 个移动、12 个原位保留，没有丢失文件。
- 14 个 CUDA、2 个 `.cuh`、归档工具均与原文件字节一致。
- 9 个 Python 文件发生文本修改，逐一按允许的导入替换、私有模块路径及 `_CUDA_SOURCE`
  路径替换核对全文；没有算法、断言、参数化或 skip 条件变化。
- 新增 11 个初始化文件，仅包含说明、直接转导出及 `__all__`，没有新增计算实现。
- TCSR 之外的正式 Python/CUDA/头文件与本轮基线一致。项目级修改为
  [AGENTS.md](../../../AGENTS.md) 的测试布局规则和 [pyproject.toml](../../../pyproject.toml) 的隔离/资源配置。
- 原有 84 个 dev 文件没有被移除。本轮更新 TCSR 方案并新增本报告；另有三份 JIT
  `dev/jit-0.3-vs-mmop.md` 在工作期间发生并发文档变化，原样保留，未用作迁移来源。

BPSA 在迁移前就是独立的正式树实验包，保留它不涉及从 dev 恢复代码。
归档的 `csr_to_csc.cu` 与 CSR 正式资源虽然字节相同，但正式转换一直使用
`_csr/_support/csr_to_csc.cu`；本次没有改变该来源。

## 2. 全部公开数据类型

从 `brainevent.__all__` 枚举数据/事件类，包含公开基类和辅助表示，共 **21 个**。
在迁移前源码、迁移后源码和安装后的 wheel 中分别核对公开名称、类所属模块及注册信息。

| 家族 | 核对类型 |
| --- | --- |
| 事件表示 | `EventRepresentation`、`BinaryArray`、`WeightedArray`、`BitPackedBinary`、`CompactBinary` |
| 基础与显式矩阵 | `DataRepresentation`、`CSR`、`CSC`、`TCSR`、`Dense` |
| 固定连接数 | `FixedNumConn`、`FixedNumPerPre`、`FixedNumPerPost` |
| JIT 基类 | `JITCMatrix`、`JITCScalarMatrix` |
| JIT Scalar | `JITCScalarR`、`JITCScalarC` |
| JIT Normal | `JITCNormalR`、`JITCNormalC` |
| JIT Uniform | `JITCUniformR`、`JITCUniformC` |

三份运行时快照完全一致：公开导出不变，显式加载 TCSR 功能后共 **59 个 primitive**，
注册名、tags、各平台后端及默认后端不变，无重复注册。
各 facade 导出的函数/primitive 与实现模块中的对象一致；SDDMM 服务仍由原包持有。

数值检查覆盖现有参数化中的 bool、int8、uint8、int16、int32、float16、bfloat16、
float32、float64，以及结构索引和编码所用的 int64、uint16、uint32、uint8。
这是对现有支持范围、转换与拒绝行为的检查，不是新增任意 dtype 的支持承诺。

| TCSR 部分 | 保留的数据契约及检查边界 |
| --- | --- |
| canonical 与 mirror | `indices:int32`、`indptr:int64`、`local_targets:uint16`、`tile_offsets:int32`；mirror permutation 为 int64；无法保留显式 int64 时明确报错 |
| hybrid workspace | task begin/end 与 indptr 同 dtype，status 为 int32；保持类型验证和容量计算 |
| binary / weighted | 保留非零活动语义、weighted 数值语义、整数/半精度活动转换、浮点输出、AD 和非活跃 NaN/Inf 隔离 |
| tile MM | float32/float64 权重 ABI；活动输入规范化为 int8；不支持的类型在加载 CUDA 前拒绝 |
| float | 保留按权重 dtype 输出和转换输入的规则；原 CUDA float32/float64 分支与非原生精度的 JAX fallback 不变 |
| dt2t | 保留逐槽位 MV/MM 及转置镜像；CUDA 结构要求 int32 indices、int64 indptr/permutation |
| SDDMM binary | cotangent 为 float32/float64；原生事件组合不变，其余支持的整数/浮点活动输入转换为布尔掩码 |
| SDDMM float | events 与 cotangent 使用匹配的 float32 或 float64；MM 仍只接受既有 batch 集合及方向 |
| BPSA | 权重/cotangent 为 float32，event 为 bool 或 float32，打包掩码为 uint8；保留 indptr int32/int64 及 tile 元数据验证 |

GPU 对应检查包含静态 ABI、资源路径、loader/FFI mock、合法与非法 dtype 输入。
本环境没有 CUDA-enabled JAX，未验证 GPU 内核实际编译及执行。

## 3. dev 隔离

| 层面 | 结果 |
| --- | --- |
| 文件来源 | 没有原 dev 文件迁出；新正式文件只有 11 个初始化文件，其他文件均来自原正式源码的明确映射 |
| 静态引用 | 检查全部非 dev 的 235 个 Python 文件及 69 个 CUDA/头文件；无指向 dev 的导入、动态路径字符串或 include |
| 既有测试中的 dev 字面量 | CSR 的源码 dtype 测试用 `'dev' not in path.parts` 排除开发文件，属于隔离断言，不是依赖 |
| 运行时 | fresh process 导入稳定包、显式导入各 TCSR 功能后，所有已加载 brainevent 模块均不来自 dev；稳定导入不加载 BPSA |
| pytest | 保留默认忽略项并追加 `dev`；正式用例不减项，开发实验不进入默认递归收集 |
| mypy | 显式排除 dev；正式包检查通过 |
| 包发现 | 排除 `*.dev`、`*.dev.*`，覆盖所有数据家族的嵌套 dev |
| 构建与安装 | 干净构建的 wheel 和 sdist 均为 **0 个 dev 文件**；源码目录外安装后再次验证导出、资源及导入隔离 |

迁移前快照重建的 wheel 含 **8 个 CSR dev 文件**：`delay_mv` 内的 3 份 CUDA、
`binary_delay.py`、`delay_main.py`、`example/lif_delay.py`、`test/delay_main_test.py`，
以及 `dev/docs/conf.py`。本次统一包发现排除规则已去除这些文件。

新 wheel 包含 **13 个活跃 TCSR CUDA 和 2 个 tile 头文件**，没有旧路径或归档副本。
16 处 `load_cuda_file` 调用按实际路径求值，包含 indexed 动态选择的两个分支；
资源都存在、与迁移前内容一致且位于正式目录。两层本地 include 文件均随包安装。

## 4. 测试与类型检查结果

环境：Python 3.12.13，JAX/JAXLIB 0.12.0.1（CPU），Numba 0.68.0，pytest 9.1.1，mypy 2.4.0。

TCSR 迁移前后均收集 **1,128 个参数化节点**，应用文件映射后节点集合完全一致。
七份测试完整执行 `-m ""`，没有排除 slow：

| 功能 | passed | skipped | failed |
| --- | ---: | ---: | ---: |
| binary | 741 | 70 | 0 |
| float | 51 | 0 | 0 |
| dt2t | 31 | 10 | 0 |
| preprocess | 6 | 0 | 0 |
| sddmm | 33 | 13 | 0 |
| bpsa | 31 | 2 | 0 |
| main | 83 | 57 | 0 |
| 合计 | **976** | **152** | **0** |

与 CSR、Dense、FCN、JIT 三家族、事件包及 data/misc/registry 合同联合收集 **9,090 个节点**，
没有 dev 测试或同名模块冲突。联合收集数不等于全部执行通过数。

扩展检查分别执行每个家族的默认 `not slow` 用例，以及全局 x64 开启时按 dtype 关键词选出的
slow 用例。这里的“默认”指全局 `JAX_ENABLE_X64=false`，测试内部可自行开启 x64。
扩展检查没有覆盖各家族剩余的非 dtype slow 用例。

| 家族 | 默认检查 passed / skipped / failed | x64 dtype slow：passed / skipped / failed |
| --- | --- | --- |
| CSR / CSC | 499 / 79 / 0 | 1045 / 4 / 0 |
| Dense | 352 / 57 / 0 | 186 / 0 / 0 |
| FCN | 662 / 26 / 0 | 257 / 4 / **36** |
| JIT Normal | 31 / 24 / 0 | 270 / 0 / **112** |
| JIT Uniform | 31 / 24 / 0 | 254 / 0 / **136** |
| JIT Scalar | 3 / 26 / 0 | 386 / 0 / 0 |
| Event | 199 / 14 / 0 | 54 / 0 / 0 |

其他验证：

- `_data_test.py`、`_misc_test.py`、`_registry_test.py`、`_numba_random_test.py` 在默认配置下 **385 passed**。
- 安装后的 preprocess、SDDMM、BPSA 及稳定导入检查：**71 passed、15 skipped**。
- `python -m mypy --no-site-packages`：**163 个源文件通过，无错误**；迁移前基线为 157 个。
  数量变化包含新增初始化文件、工具归档和统一排除已有 dev。

## 5. 发现的既有 x64 问题

以下失败全部在迁移前源码副本中复现，逐用例名称和失败信息与迁移后完全一致。
同一批 **284 个**梯度用例在迁移后关闭全局 x64 时全部通过。本次未修改这些算子的计算代码。

| 范围 | x64 失败数 | 表现 |
| --- | ---: | --- |
| FCN `test_fcn_weighted_array_matmul_and_grads` | 36 | dense reference 的 COO 转置求导断言失败：行列索引出现 int64/int32 不一致 |
| JIT Normal `test_weighted_jitc_grads_match_dense` | 112 | `todense()` 参考梯度合并 tangent 时出现 float32/float64 不一致 |
| JIT Uniform 梯度路径 | 136 | 对 w_low/w_high 求导时生成的参数类型不一致，触发 `w_low and w_high must have the same dtype` |

另将四份跨包合同测试整体强制置于 x64 环境时，得到 **380 passed、5 failed**，
五项失败也在基线逐一复现：

- 三项 `TestDiagAddBufferIntegration` 用 float32 数据配合未显式指定 dtype 的对角输入，
  x64 下后者成为 float64，触发既有类型一致性断言。
- 两项 `_misc_test.py` 明确检查“x64 关闭时应报错”，与强制开启的运行环境冲突。

恢复默认配置后，这四份文件的 385 项全部通过。以上配置差异与梯度问题分别记录，
没有通过改变生产算子、测试断言或 skip 条件来消除失败。

本次结论限定为：文件迁移未引入已检查范围内的回归，dev 隔离与打包验证通过。
既有 x64 梯度问题、GPU 数值执行以及未运行的其他 slow 用例仍需分别验证。

## 6. 复现与证据

从 be0.3 根目录执行（依赖已安装）：

```bash
JAX_PLATFORMS=cpu python -m pytest --collect-only -q -m "" brainevent/_tcsr
JAX_PLATFORMS=cpu JAX_ENABLE_X64=false python -m pytest -q -m "" brainevent/_tcsr
JAX_PLATFORMS=cpu JAX_ENABLE_X64=false python -m pytest -q -m "" brainevent/_data_test.py brainevent/_misc_test.py brainevent/_registry_test.py brainevent/_numba_random_test.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m pytest -q -m slow -k 'dtype or bool or int8 or uint8 or int16 or uint16 or int32 or uint32 or int64 or uint64 or float16 or bfloat16 or float32 or float64' brainevent/_fcn
python -m mypy --no-site-packages
```

同一条 dtype 命令可分别用于 `_csr`、`_dense`、三个 `_jit_*` 和 `_event`。
构建使用独立干净副本运行 `python -m build`，随后 `pip install --no-deps --target ...`，
从源码目录外运行安装检查，避免已有 egg-info 或源码导入掩盖打包问题。

本机证据目录：`/tmp/brainevent-tcsr-layout-bfq0a6vd`。
这是本次检查的临时产物，未写入正式包；长期结论保存在本文。

| 证据 | 内容 |
| --- | --- |
| `before-package.json`、`before-workspace.json`、`baseline-package/` | 实施前哈希及源码副本 |
| `manifest.json`、`audit_sources.py`、`source-audit.json` | 36 文件映射、逐字节/允许替换核对及全包 dev 静态审计 |
| `verify_runtime.py`、`runtime-before.json`、`runtime-after.json`、`runtime-installed.json` | 三次导出、类型、注册及资源快照 |
| `collect-before.log`、`collect-after.log`、`collect-all-data.log` | 收集节点及跨包收集 |
| `pytest-*.log/xml`、`baseline-*.log/xml` | 分组回归、x64 基线复现及默认配置复查 |
| `test-results.json`、`baseline-repros.json`、`dtype-final-results.json` | 分组执行状态 |
| `mypy-before.log`、`mypy-after.log` | 类型检查结果 |
| `baseline-dist/`、`dist/`、`installed/` | 迁移前 wheel、新 wheel/sdist 及安装副本 |
