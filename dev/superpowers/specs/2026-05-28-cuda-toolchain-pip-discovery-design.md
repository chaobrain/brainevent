# GPU 工具链发现：复用 `jax[cuda*]` 自带的 pip nvcc + host 编译器策略

- 日期：2026-05-28
- 状态：设计待实现
- 涉及模块：`brainevent/_op/kernix_toolchain.py`、`brainevent/_op/kernix_compiler.py`、`brainevent/_op/kernix_runtime.py`、`brainevent/_error.py`、`brainevent/config.py`、README/FAQ

## 1. 背景与问题

brainevent 的 GPU 后端把 `.cu` 内核（含 host 侧 XLA FFI 注册代码）在运行时用 `nvcc` 编成 `.so`（`kernix_*` 流水线，ninja 优先、直连 nvcc 兜底）。

当前 `detect_cuda_toolchain()` 只在 `PATH` / `$CUDA_HOME` / `$BRAINEVENT_NVCC_PATH` 找 `nvcc`，找不到即报错。因此**用户必须额外安装系统级 CUDA Toolkit** 才能用 GPU——这是要消除的痛点。

关键事实（已验证）：

- `pip install jax[cuda12|cuda13]` 会自动安装一整套 `nvidia-*` pip 包，其中**已包含 nvcc / ptxas / cudart / 头文件 / libdevice**：
  - **cu13（及未来 cuNN）合并式布局**：全部在 `site-packages/nvidia/cu13/` 下——`bin/nvcc`、`include/cuda_runtime.h`、`lib/libcudart.so.13`、`nvvm/libdevice/libdevice.10.bc`。其 `bin/nvcc.profile` 已自动注入 `-L .../lib`、`-L .../lib/stubs`（libcuda 桩）和 cccl 的 `-isystem`。
  - **cu12 分散式布局**：每包一目录——`nvidia/cuda_nvcc/bin/nvcc`（自带 `include/` 仅 crt 头）、`cuda_runtime.h` 在 `nvidia/cuda_runtime/include`、cudart 库在 `nvidia/cuda_runtime/lib`、cccl 头在 `nvidia/cuda_cccl/include`（若安装）。
- pip 的 nvcc 包**不含 host C++ 编译器**；`nvcc.profile` 也不指定 host 编译器，nvcc 默认到 `PATH` 找 `gcc/g++`。
- 当前 CUDA 编译路径**没有传 `-ccbin`**：`detect_cuda_toolchain()` 探测了 `cxx` 但只有 CPU 后端用它，nvcc 实际用的是 PATH 上第一个 `g++`——"探测到的"与"实际用的"可能不一致（隐藏缺口）。

## 2. 目标 / 非目标

**目标**

- 让仅通过 `pip install brainevent[cuda12]` / `[cuda13]`（含 `jax[cuda*]`）的用户，**无需单独安装系统 CUDA Toolkit** 即可编译现有 CUDA 内核。
- 通用覆盖**所有** `jax[cuda*]` 安装（cu12 分散布局、cu13 合并布局，并向后兼容未来 cuNN），不依赖某台机器的具体布局。
- host C++ 编译器依赖做到：可靠发现、确定性使用、版本兼容处理、清晰诊断、文档说明。
- **分层报错**：编译链路每一层（驱动/算力、nvcc 发现、host 编译器发现、头文件、编译/链接、加载）都有独立异常类型与统一格式诊断，清楚指出"哪一层失败、试过什么、为什么否、怎么修"。

**非目标（本次不做）**

- 不消除对 host C++ 编译器（g++/clang++）的依赖。彻底免 host 编译器需改走 NVRTC + 预编译 host shim（见 §10）。
- 不实现 HIP/AMD 路径。
- 不改 CPU(`CPPBackend`) / numba 路径的行为。

## 3. 已确定的决策

1. **nvcc 发现优先级：pip 优先于系统 PATH**（保证 nvcc 与 jax 加载的 CUDA runtime 版本一致，避免 `libcudart.so.12` vs `.13` 冲突）。
2. **提供用户快捷函数**：默认 pip 优先，但用户可切换为"系统 PATH 优先"。
3. **host 编译器发现：conda 优先于系统**（conda 的 gxx 与 conda libstdc++ ABI 匹配）。
4. **给 nvcc 显式传 `-ccbin`**，让探测到的 host 编译器就是实际使用的（消除隐藏缺口、构建可复现）。

## 4. 设计

### 4.1 通用 nvcc + include 发现

在 `kernix_toolchain.py` 新增 `_find_pip_cuda() -> tuple[str, list[str]] | None`，返回 `(nvcc_path, include_dirs)`：

```text
用 importlib.util.find_spec("nvidia") 取得 nvidia 命名空间包的所有 search 根目录
（跨 venv / conda / user-site 均可靠）。对每个根 root：

A. 合并式（cu13 及未来 cuNN，向后兼容）：
   glob root/cu*/bin/nvcc(.exe)；若有多个 cuNN，按数字版本取最高。
   命中 → nvcc = 该路径；include_dirs = [<that cuNN>/include]

B. 分散式（cu12）：
   若 root/cuda_nvcc/bin/nvcc(.exe) 存在 →
     nvcc = 该路径
     include_dirs = [cuda_nvcc/include]
       + (cuda_runtime/include 若存在)     # 提供 cuda_runtime.h
       + (cuda_cccl/include   若存在)       # thrust/cub
   其中 cuda_runtime / cuda_cccl 也用 find_spec("nvidia.cuda_runtime")
   等定位，避免硬编码相对路径。

找到即返回；A 优先于 B。
```

改写 `detect_cuda_toolchain()` 的 nvcc 选择顺序为：

1. `$BRAINEVENT_NVCC_PATH`（显式覆盖，最高）
2. 显式设置的 `$CUDA_HOME/bin/nvcc`（用户明确指向某 toolkit）
3. 由偏好决定（见 §4.3）的两步：
   - 默认 `pip`：`_find_pip_cuda()` → 再 `shutil.which("nvcc")`
   - 切换 `system`：`shutil.which("nvcc")` → 再 `_find_pip_cuda()`

`cuda_home` 与 `cuda_include_dirs` 的确定：

- 若 nvcc 来自 `_find_pip_cuda()`：`cuda_include_dirs` 用其返回的 list；`cuda_home` 由 nvcc 路径上推（`bin/nvcc` → 上两级），仅用于诊断。
- 若来自系统/CUDA_HOME：`cuda_home` 按现逻辑推导，`cuda_include_dirs = [cuda_home/include]`。
- nvcc 通过 `nvcc.profile` 自相对定位 `nvvm/libdevice`，无需额外处理（已验证 cu13/cu12 均成立）。

> 注：不再用环境里可能过时的 `$CUDA_HOME` 覆盖 pip 选中的 home/include；以"实际选中的 nvcc"为准来推导，保证一致。

### 4.2 host 编译器：发现 + `-ccbin` + 版本兼容 + 诊断

**发现优先级**（`detect_cuda_toolchain()` 与 `detect_cpp_toolchain()` 共用一个 helper `_find_host_cxx()`）：

1. `$CXX`（显式覆盖）
2. conda：若 `$CONDA_PREFIX` 存在，依次找
   `$CONDA_PREFIX/bin/{<triplet>-g++, g++, c++, clang++}`
   （`<triplet>` 如 `x86_64-conda-linux-gnu`；可用 glob `*-g++` 兜底匹配）
3. 系统 PATH：`g++` → `c++` → `clang++`

**`-ccbin`**：在 `NinjaBuild._cuda_flags()` 与 `CUDABackend.compile_source()` 的直连 nvcc 命令中，均加入 `["-ccbin", toolchain.cxx]`，使探测到的编译器即实际使用的。

**版本兼容**：

- 在 `CudaToolchain` 记录 `cxx_version`（运行 `cxx --version` 取首行；同时进编译缓存 key，使换编译器后能重新编译）。
- 不硬编码 CUDA↔gcc 版本矩阵。改为：当 nvcc 失败且 stderr 命中 "unsupported"/"is not supported" 这类 host 版本信号时，抛 `HostCompilerIncompatibleError`（见 §6）给出明确指引：装受支持的 gcc 并设 `CXX`，或设 `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1`。
- 新增 env `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER`：为真时给 nvcc 追加 `-allow-unsupported-compiler`。

**诊断**：所有发现/编译错误统一走 §6 的分层报错机制（含环境感知的"如何修复"，例如 host 编译器缺失时明确"pip 不提供 host 编译器"，并给 `conda install -c conda-forge gxx` / `apt-get install g++` / `dnf install gcc-c++` / 设 `CXX`）。

### 4.3 用户切换快捷函数（pip / system 优先）

在 `kernix_toolchain.py` 保存进程内偏好：

```python
_NVCC_DISCOVERY: str = "pip"   # "pip" | "system"

def set_nvcc_discovery(prefer: str) -> None: ...   # 校验取值
def get_nvcc_discovery() -> str: ...
```

`detect_cuda_toolchain()` 在 §4.1 第 3 步读取该偏好。读取优先级：
**函数显式设置 > 环境变量 `BRAINEVENT_NVCC_PREFER`(pip|system) > 默认 pip**。

在 `brainevent/config.py` 暴露面向用户的快捷函数并加入其 `__all__`，与既有 `set_numba_parallel` 等同风格：

```python
def prefer_system_nvcc(enable: bool = True) -> None:
    """切换 nvcc 发现优先级：True→系统 PATH 优先，False→pip 优先（默认）。"""
```

用户用法：`brainevent.config.prefer_system_nvcc()`。该函数内部转调 `kernix_toolchain.set_nvcc_discovery(...)`。

> 缓存交互：偏好须在首次编译前设置；`detect_cuda_toolchain()` 每次调用都重新读偏好（现状即每次重新探测，不额外引入 memo）。

### 4.4 数据结构变更：`CudaToolchain`

- `cuda_include_dir: str` → `cuda_include_dirs: tuple[str, ...]`。
- 新增 `cxx_version: str = ""`。
- 同步更新引用（共 4 处，已确认）：dataclass 定义、`detect_cuda_toolchain()` 构造、`kernix_compiler.py:178`（ninja `_cuda_flags`）、`kernix_compiler.py:342`（直连 nvcc）——后两处由单个 `-I` 改为遍历 `cuda_include_dirs` 生成多个 `-I`。

### 4.5 运行时链接（cu12 / cu13）

- **cu13**：`nvcc.profile` 已注入 `-L .../lib`（含 `libcudart.so.13`）与 `-L .../lib/stubs`（libcuda 桩），链接 OK；运行时 JAX 已加载同 soname 的 cudart，加载 OK。
- **cu12（分散）**：cudart 在 `nvidia/cuda_runtime/lib`，不在 nvcc 自身 `../lib`。若链接/加载失败，需为 cu12 追加 `-L <nvidia/cuda_runtime/lib>` 并写 rpath（`--linker-options -rpath,<...>`）。**列为验证项**，确认确有需要再加，不预先投机加入。

## 5. 改动点汇总（文件 / 行）

- `brainevent/_error.py`
  - 新增异常类（见 §6.1）：`GpuArchDetectionError`、`NvccNotFoundError`、`HostCompilerNotFoundError`、`HeaderNotFoundError`、`HostCompilerIncompatibleError`、`KernelLoadError`，并入 `__all__`。
  - `CompilationError`：新增 `stage` 字段（"compile"/"link"/"build"）。
- `brainevent/_op/kernix_toolchain.py`
  - 新增 `_find_pip_cuda()`、`_find_host_cxx()`、`set/get_nvcc_discovery()`、`_NVCC_DISCOVERY`。
  - 新增诊断基础设施（见 §6）：`CandidateProbe`、`render_toolchain_error()`、`collect_toolchain_diagnostics()`。
  - 改写 `detect_cuda_toolchain()`：nvcc 发现顺序、include_dirs、cxx 发现、cxx_version；各层失败抛对应分层异常（带 probe 列表）。
  - `detect_cpp_toolchain()` 复用 `_find_host_cxx()`、抛 `HostCompilerNotFoundError`。
  - `detect_cuda_arch()`：失败抛 `GpuArchDetectionError`。
  - `CudaToolchain`：`cuda_include_dirs` + `cxx_version`。
- `brainevent/_op/kernix_compiler.py`
  - `NinjaBuild._cuda_flags()`、`CUDABackend.compile_source()`：遍历 `cuda_include_dirs`；加 `-ccbin`；按 `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER` 追加 `-allow-unsupported-compiler`。
  - 编译/链接失败：标注 `stage`；命中 host 版本信号时抛 `HostCompilerIncompatibleError`，否则 `CompilationError`。
- `brainevent/_op/kernix_runtime.py`
  - `.so` dlopen 失败：包成 `KernelLoadError`，附 `.so` 路径与缺库提示（见 §6.5 E-LOAD）。
  - FFI 注册失败：沿用现有 `KernelRegistrationError`，改走统一渲染器输出。
- `brainevent/config.py`：新增 `prefer_system_nvcc()` 并入 `__all__`。
- README / FAQ：GPU 依赖三件套说明（§7）。

## 6. 分层报错与诊断机制

编译链路每一层失败都给出**独立异常类型 + 统一格式**的诊断，用户一眼看清"哪一层、为什么、试了什么、怎么修"。

### 6.1 错误分层与异常类型（均向后兼容，继承现有基类）

| 阶段码 | 层 | 异常（新增/扩展） | 基类 | 触发 |
|---|---|---|---|---|
| `E-ARCH` | 设备 / 驱动 | `GpuArchDetectionError` | `KernelToolchainError` | `nvidia-smi`/驱动缺失、算力探测失败 |
| `E-NVCC` | nvcc 发现 | `NvccNotFoundError` | `KernelToolchainError` | 所有候选都没找到 nvcc |
| `E-CXX` | host 编译器发现 | `HostCompilerNotFoundError` | `KernelToolchainError` | 找不到 g++/clang++ |
| `E-HDR` | 头文件解析 | `HeaderNotFoundError` | `KernelToolchainError` | `cuda_runtime.h` / XLA FFI `ffi.h` / brainevent include 缺失 |
| `E-COMPILE` | 编译 / 链接 | `CompilationError`（扩展 `stage`） | `KernelCompilationError` | nvcc/g++ 编译或链接失败 |
| `E-CXXVER` | host 版本不兼容 | `HostCompilerIncompatibleError` | `CompilationError` | 编译失败且命中 "unsupported host compiler" 信号 |
| `E-LOAD` | 加载 `.so` | `KernelLoadError` | `KernelError` | dlopen `.so` 失败（如 `libcudart.so.X` 找不到） |
| `E-REG` | FFI 注册 | `KernelRegistrationError`（**复用现有**） | `KernelError` | XLA FFI target 注册失败 |

> 全部继承现有 `KernelToolchainError` / `CompilationError` / `KernelError`，旧的 `except KernelToolchainError`/`except CompilationError` 仍能兜住，不破坏现有捕获。`KernelRegistrationError` 已存在，仅让其消息走统一渲染器（§6.2）。

### 6.2 统一诊断输出格式

上述异常都经同一渲染器 `render_toolchain_error(...)` 生成消息，固定分区：

```text
[brainevent GPU 工具链] <阶段> 失败  (code=E-XXX)

原因: <一句话点明本层失败>

已尝试 (按优先级):
  ✗ BRAINEVENT_NVCC_PATH         未设置
  ✗ $CUDA_HOME/bin/nvcc          未设置
  ✗ pip: nvidia/cu*/bin/nvcc     未找到 (未安装 jax[cuda13]?)
  ✗ pip: nvidia/cuda_nvcc/bin    未找到
  ✗ PATH: nvcc                   未找到

命令:                # 仅编译/链接层
  <command>
编译器输出:           # 仅编译/链接层
  <stderr+stdout>

如何修复:
  1) <按当前环境给出的可执行步骤>
  2) ...

工具链快照:           # 可选，见 6.4
  nvcc=...  host_cxx=...(ver)  includes=[...]  discovery=pip  env=...
```

- **原因**：一句话点明本层失败。
- **已尝试（按优先级）**：列出本层每个候选及状态——这是"清晰"的关键，用户能看到 pip/系统/conda 各候选分别试了哪条路径、为何被否。
- **命令 / 编译器输出**：仅编译/链接层有（沿用 `CompilationError` 现有 `command`/`compiler_output` 字段）。
- **如何修复**：按当前环境给可执行步骤（见 6.5）。
- **工具链快照**：可选（见 6.4）。

### 6.3 候选探测记录 `CandidateProbe`

发现类 helper（`_find_pip_cuda` / `_find_host_cxx` / nvcc 选择）在尝试每个候选时记录一条 probe：

```python
@dataclass(frozen=True)
class CandidateProbe:
    source: str   # "BRAINEVENT_NVCC_PATH" / "pip:nvidia/cu13" / "PATH:nvcc" / "$CONDA_PREFIX/bin/g++" ...
    path: str     # 实际检查的路径（env 未设置则为空）
    status: str   # "unset" | "not-found" | "not-a-file" | "rejected:<why>" | "ok"
```

helper 返回 `(result | None, list[CandidateProbe])`；失败时把 probe 列表交给异常，由渲染器输出成"已尝试"区。这样发现逻辑与报错文案解耦，且顺序与实际尝试一致。

### 6.4 统一快照 `collect_toolchain_diagnostics()`

返回结构化快照（nvcc 路径与版本、host cxx 与版本、`cuda_include_dirs`、发现偏好、相关 env、检测到的 cu 布局、算力），作为诊断的**单一真相源**：

- 供既有 `print_diagnostics()` 复用（去掉其各自拼装逻辑）；
- 当 `BRAINEVENT_TOOLCHAIN_DEBUG=1` 时，附到任意错误末尾的"工具链快照"区，便于一次性贴报告排障。

### 6.5 各层修复指引（remediation）要点

- **E-NVCC**：装 `jax[cuda12]`/`jax[cuda13]`（自带 nvcc，免系统 CUDA）；或设 `BRAINEVENT_NVCC_PATH`/`CUDA_HOME`；或 `brainevent.config.prefer_system_nvcc()`。
- **E-CXX**：pip 不提供 host 编译器 → `conda install -c conda-forge gxx` / `apt-get install g++` / `dnf install gcc-c++` / 设 `CXX`。
- **E-HDR**：指出缺哪个头、搜了哪些目录；多为 jax 与 CUDA wheel 不配套或包损坏 → 重装对应 `jax[cuda*]` / jaxlib。
- **E-CXXVER**：装受支持的 gcc 并设 `CXX`，或 `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1`。
- **E-LOAD**：若缺 `libcudart.so.X`（典型 cu12）→ 提示 §4.5 的 `-L`/rpath 修复，或设 `LD_LIBRARY_PATH`；附实际 `.so` 路径与 dlopen 原文。
- **E-ARCH**：装/修驱动使 `nvidia-smi` 可用，或设 `BRAINEVENT_COMPUTE_CAPABILITIES`。

### 6.6 设计取舍

- **为何用子类而非单类 + `stage` 字段**：用户明确要"每一层很清晰"，独立异常类型便于 `except` 精准捕获、便于测试断言、IDE/文档可见；同时全部继承现有基类，零破坏。
- **渲染器集中**：所有文案走一处 `render_toolchain_error`，保证格式一致、便于统一改版与本地化。

## 7. 文档更新（README / FAQ）

GPU 编译需要三件套：
1. **NVIDIA 驱动（libcuda）**——系统/驱动层，任何方案都需要；`nvidia-smi` 用于架构探测（`BRAINEVENT_COMPUTE_CAPABILITIES` 兜底）。
2. **`jax[cuda12]` / `jax[cuda13]`**——提供 nvcc/ptxas/cudart/头，**免系统 CUDA Toolkit**。
3. **host C++ 编译器（g++/clang++）**——pip 不提供，用 conda/apt/dnf 安装。

并记录可选项：`BRAINEVENT_NVCC_PREFER`、`prefer_system_nvcc()`、`BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER`、`CXX`、`BRAINEVENT_NVCC_PATH`。

## 8. 风险与验证计划

- **真实端到端验证**：在仅装 `jax[cuda13]`（无系统 CUDA）的环境中，跑一个现有 GPU 内核，确认 nvcc 被发现、编译成功、`.so` 能 import 并运行。
- **cu12 链接验证**：在 `jax[cuda12]` 环境验证 cudart 链接/加载；据结果决定是否加 §4.5 的 `-L`/rpath。
- **回归**：系统 CUDA 在 PATH 的环境，确认 `prefer_system_nvcc()` 后行为与旧版一致。
- **无编译器/旧 gcc**：确认诊断信息与 `-allow-unsupported-compiler` 逃生口生效。
- **分层报错单测**：逐层模拟失败并断言"异常类型 + 阶段码 + 含修复指引"——
  - nvcc 不可见（清空候选）→ `NvccNotFoundError`，消息含"已尝试"全部候选与 `jax[cuda*]` 指引；
  - 无 host 编译器（`CXX`/conda/PATH 皆空）→ `HostCompilerNotFoundError`，含 conda/apt/dnf 指引；
  - 缺头文件 → `HeaderNotFoundError`，含缺失头名与搜索目录；
  - 注入"unsupported host compiler"样例 stderr → `HostCompilerIncompatibleError`；
  - dlopen 缺 `libcudart.so` → `KernelLoadError`，含 `.so` 路径；
  - `nvidia-smi` 不可用且未设算力 → `GpuArchDetectionError`。
  - 校验 `BRAINEVENT_TOOLCHAIN_DEBUG=1` 时错误末尾出现"工具链快照"。

## 9. 兼容性

- 对已有系统 CUDA 用户：默认改为 pip 优先属行为变更，提供 `prefer_system_nvcc()` 与 `BRAINEVENT_NVCC_PREFER=system` 还原旧优先级。
- `cuda_include_dir`→`cuda_include_dirs` 为内部数据结构，非公共 API。

## 10. 非目标 / 未来：NVRTC 方案 B

要彻底去掉 host 编译器与 nvcc 依赖，需改架构：host 侧 FFI launcher 在 brainevent 构建期（manylinux）预编译并随 wheel 分发，运行时仅用 **NVRTC** 把设备内核 JIT 成 PTX/cubin，并经 driver API（`cuLaunchKernel`）启动。代价：NVRTC 只编设备代码、host 侧 thrust 不可用、需要新的启动路径与较大重构。本次不做，仅作记录。
