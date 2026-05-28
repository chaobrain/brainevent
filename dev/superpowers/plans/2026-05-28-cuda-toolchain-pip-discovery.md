# GPU 工具链 pip 发现 + 分层报错 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让仅装了 `jax[cuda*]`（无系统 CUDA Toolkit）的用户也能编译 brainevent 的 CUDA 内核，并为编译链路每一层提供独立异常类型与统一格式的诊断。

**Architecture:** 在 `kernix_toolchain.py` 增加"发现层"（通用扫描 `nvidia/*` pip 包里的 nvcc 与头文件、按优先级选择 host 编译器），所有发现/编译/加载失败统一经一个渲染器输出分层错误（带"已尝试候选"与"如何修复"）；`kernix_compiler.py` 用 `-ccbin` 锁定 host 编译器并遍历多个 include 目录；`config.py` 暴露 pip/system 切换快捷函数。

**Tech Stack:** Python 3.10+、pytest、jax/jaxlib、nvcc/ninja、ctypes。

**Spec:** `dev/superpowers/specs/2026-05-28-cuda-toolchain-pip-discovery-design.md`

---

## File Structure

- `brainevent/_error.py` — 新增分层异常类；`CompilationError` 加 `stage` 字段。
- `brainevent/_op/kernix_toolchain.py` — 发现与诊断核心：`CandidateProbe`、`render_toolchain_error`、`collect_toolchain_diagnostics`、`_find_pip_cuda`、`_find_host_cxx`、`_select_nvcc`、`set/get_nvcc_discovery`；重写 `detect_cuda_toolchain`/`detect_cpp_toolchain`/`detect_cuda_arch`；`CudaToolchain` 改 `cuda_include_dirs`+`cxx_version`。
- `brainevent/_op/kernix_compiler.py` — 遍历 `cuda_include_dirs`、加 `-ccbin`、`-allow-unsupported-compiler`、编译失败分类（`HostCompilerIncompatibleError`）、`stage` 标注。
- `brainevent/_op/kernix_runtime.py` — `CompiledModule` dlopen 失败包成 `KernelLoadError`。
- `brainevent/_op/kernix_pipeline.py` — cache key 纳入 host `cxx_version`；`print_diagnostics` 复用 `collect_toolchain_diagnostics`。
- `brainevent/config.py` — `prefer_system_nvcc()`。
- `brainevent/__init__.py` — 重导出新异常类。
- 测试：`brainevent/_op/kernix_toolchain_test.py`、`brainevent/_op/kernix_compiler_test.py`、`brainevent/_op/kernix_runtime_load_test.py`、`brainevent/_error_test.py`。
- 文档：`README.md`（GPU 依赖三件套）。

测试命令统一用：`python -m pytest <path> -v`。

---

## Task 1: 分层异常类 + `CompilationError.stage`

**Files:**
- Modify: `brainevent/_error.py`
- Test: `brainevent/_error_test.py` (create)

- [ ] **Step 1: 写失败测试**

Create `brainevent/_error_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for the layered toolchain/compilation exception hierarchy."""

from brainevent._error import (
    KernelError, KernelToolchainError, CompilationError,
    NvccNotFoundError, HostCompilerNotFoundError, HeaderNotFoundError,
    GpuArchDetectionError, HostCompilerIncompatibleError, KernelLoadError,
)


def test_toolchain_subclasses():
    assert issubclass(NvccNotFoundError, KernelToolchainError)
    assert issubclass(HostCompilerNotFoundError, KernelToolchainError)
    assert issubclass(HeaderNotFoundError, KernelToolchainError)
    assert issubclass(GpuArchDetectionError, KernelToolchainError)


def test_compilation_subclasses():
    assert issubclass(HostCompilerIncompatibleError, CompilationError)


def test_load_subclass():
    assert issubclass(KernelLoadError, KernelError)


def test_compilation_error_stage_field():
    e = CompilationError("boom", compiler_output="out", command="cmd", stage="link")
    assert e.stage == "link"
    assert "cmd" in str(e)
    assert "out" in str(e)


def test_compilation_error_default_stage():
    assert CompilationError("x").stage == "compile"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest brainevent/_error_test.py -v`
Expected: FAIL — ImportError (新类未定义)。

- [ ] **Step 3: 修改 `brainevent/_error.py`**

在 `__all__`（约第 19–32 行）追加新名字。把现有 `__all__` 列表末尾的 `'CUDANotInstalledError',` 之后补：

```python
    'NvccNotFoundError',
    'HostCompilerNotFoundError',
    'HeaderNotFoundError',
    'GpuArchDetectionError',
    'HostCompilerIncompatibleError',
    'KernelLoadError',
```

把 `CompilationError`（约第 283–295 行）替换为带 `stage` 的版本：

```python
class CompilationError(KernelCompilationError):
    """CUDA or C++ compilation failed."""

    def __init__(self, message: str, compiler_output: str = "",
                 command: str = "", stage: str = "compile"):
        self.compiler_output = compiler_output
        self.command = command
        self.stage = stage
        full_msg = message
        if command:
            full_msg += f"\n\nCommand:\n  {command}"
        if compiler_output:
            full_msg += f"\n\nCompiler output:\n{compiler_output}"
        super().__init__(full_msg)


class HostCompilerIncompatibleError(CompilationError):
    """Host C++ compiler version is not supported by this CUDA/nvcc (E-CXXVER)."""
    __module__ = 'brainevent'
```

在 `KernelToolchainError`（约第 278–280 行）之后追加：

```python
class NvccNotFoundError(KernelToolchainError):
    """The CUDA compiler (nvcc) could not be located (E-NVCC)."""
    __module__ = 'brainevent'


class HostCompilerNotFoundError(KernelToolchainError):
    """No host C++ compiler (g++/clang++) could be located (E-CXX)."""
    __module__ = 'brainevent'


class HeaderNotFoundError(KernelToolchainError):
    """A required header (cuda_runtime.h / XLA FFI / brainevent) is missing (E-HDR)."""
    __module__ = 'brainevent'


class GpuArchDetectionError(KernelToolchainError):
    """GPU compute capability could not be detected (E-ARCH)."""
    __module__ = 'brainevent'


class KernelLoadError(KernelError):
    """A compiled .so failed to load via dlopen (E-LOAD)."""
    __module__ = 'brainevent'
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest brainevent/_error_test.py -v`
Expected: PASS。

- [ ] **Step 5: 重导出到顶层包**

在 `brainevent/__init__.py` 的 `from ._error import (` 块内加入这 6 个名字：`NvccNotFoundError, HostCompilerNotFoundError, HeaderNotFoundError, GpuArchDetectionError, HostCompilerIncompatibleError, KernelLoadError`；并把同样 6 个名字加入该文件的 `__all__` 列表（在已有错误类名附近）。

验证：`python -c "import brainevent; brainevent.NvccNotFoundError"`
Expected: 无错误。

- [ ] **Step 6: Commit**

```bash
git add brainevent/_error.py brainevent/_error_test.py brainevent/__init__.py
git commit -m "feat(_error): add layered toolchain/compilation exceptions and CompilationError.stage"
```

---

## Task 2: `CandidateProbe` + `render_toolchain_error`

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`
- Test: `brainevent/_op/kernix_toolchain_test.py` (create)

- [ ] **Step 1: 写失败测试**

Create `brainevent/_op/kernix_toolchain_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for kernix_toolchain discovery and diagnostics."""

from brainevent._op import kernix_toolchain as kt
from brainevent._op.kernix_toolchain import CandidateProbe, render_toolchain_error


def test_render_sections_present():
    msg = render_toolchain_error(
        stage="nvcc 发现", code="E-NVCC", summary="未找到 nvcc。",
        probes=[
            CandidateProbe("BRAINEVENT_NVCC_PATH", "", "unset"),
            CandidateProbe("PATH:nvcc", "", "not-found"),
        ],
        remediation=["安装 jax[cuda13]"],
    )
    assert "E-NVCC" in msg
    assert "原因" in msg
    assert "已尝试" in msg
    assert "BRAINEVENT_NVCC_PATH" in msg
    assert "如何修复" in msg
    assert "安装 jax[cuda13]" in msg


def test_render_includes_command_for_compile():
    msg = render_toolchain_error(
        stage="编译", code="E-COMPILE", summary="失败",
        command="nvcc x.cu", compiler_output="error: boom", remediation=["fix it"],
    )
    assert "命令" in msg and "nvcc x.cu" in msg
    assert "编译器输出" in msg and "boom" in msg
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -v`
Expected: FAIL — ImportError (`CandidateProbe`/`render_toolchain_error` 未定义)。

- [ ] **Step 3: 实现**

在 `brainevent/_op/kernix_toolchain.py` 顶部导入区补 `import re`（已有 `import os, shutil, subprocess, sys`、`from dataclasses import dataclass`、`from pathlib import Path`）。

把 `from brainevent._error import KernelToolchainError` 改为：

```python
from brainevent._error import (
    KernelToolchainError, NvccNotFoundError, HostCompilerNotFoundError,
    HeaderNotFoundError, GpuArchDetectionError,
)
```

在文件中（紧接 import 之后、`CudaToolchain` 定义之前）加入：

```python
@dataclass(frozen=True)
class CandidateProbe:
    """One attempted toolchain candidate and its outcome."""
    source: str   # "BRAINEVENT_NVCC_PATH" / "pip:nvidia/cu13" / "PATH:nvcc" ...
    path: str     # examined path ("" if the env var is unset)
    status: str   # "unset" | "not-found" | "not-a-file" | "rejected:<why>" | "ok"


_PROBE_LABEL = {
    "unset": "未设置",
    "not-found": "未找到",
    "not-a-file": "不是文件",
    "ok": "命中",
}


def _probe_line(p: CandidateProbe) -> str:
    mark = "✓" if p.status == "ok" else "✗"
    if p.status.startswith("rejected:"):
        label = "拒绝(" + p.status.split(":", 1)[1] + ")"
    else:
        label = _PROBE_LABEL.get(p.status, p.status)
    loc = f"  [{p.path}]" if p.path else ""
    return f"  {mark} {p.source}{loc}  {label}"


def render_toolchain_error(
    *,
    stage: str,
    code: str,
    summary: str,
    probes: "list[CandidateProbe] | None" = None,
    command: str = "",
    compiler_output: str = "",
    remediation: "list[str] | None" = None,
    snapshot: "dict | None" = None,
) -> str:
    """Render a uniform, layered toolchain/compilation error message."""
    if snapshot is None and os.environ.get("BRAINEVENT_TOOLCHAIN_DEBUG"):
        try:
            snapshot = collect_toolchain_diagnostics()
        except Exception:
            snapshot = None

    lines = [f"[brainevent GPU 工具链] {stage} 失败  (code={code})", "", f"原因: {summary}"]
    if probes:
        lines += ["", "已尝试 (按优先级):"]
        lines += [_probe_line(p) for p in probes]
    if command:
        lines += ["", "命令:", f"  {command}"]
    if compiler_output:
        lines += ["", "编译器输出:", compiler_output]
    if remediation:
        lines += ["", "如何修复:"]
        lines += [f"  {i}) {r}" for i, r in enumerate(remediation, 1)]
    if snapshot:
        lines += ["", "工具链快照:"]
        lines += [f"  {k} = {v}" for k, v in snapshot.items()]
    return "\n".join(lines)
```

> `collect_toolchain_diagnostics` 在 Task 9 定义；由于只在调用时解析名字，Task 2–8 期间它不存在也不影响这些任务的测试（测试未设置 `BRAINEVENT_TOOLCHAIN_DEBUG`，不会走到该分支）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): add CandidateProbe and unified toolchain error renderer"
```

---

## Task 3: nvcc 发现偏好 + `config.prefer_system_nvcc`

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`, `brainevent/config.py`
- Test: `brainevent/_op/kernix_toolchain_test.py`

- [ ] **Step 1: 追加失败测试**

Append to `brainevent/_op/kernix_toolchain_test.py`:

```python
import pytest


def test_nvcc_discovery_default_and_env(monkeypatch):
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", None)
    monkeypatch.delenv("BRAINEVENT_NVCC_PREFER", raising=False)
    assert kt.get_nvcc_discovery() == "pip"
    monkeypatch.setenv("BRAINEVENT_NVCC_PREFER", "system")
    assert kt.get_nvcc_discovery() == "system"


def test_nvcc_discovery_function_overrides_env(monkeypatch):
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", None)
    monkeypatch.setenv("BRAINEVENT_NVCC_PREFER", "system")
    kt.set_nvcc_discovery("pip")
    assert kt.get_nvcc_discovery() == "pip"


def test_nvcc_discovery_invalid(monkeypatch):
    with pytest.raises(ValueError):
        kt.set_nvcc_discovery("bogus")


def test_config_prefer_system_nvcc(monkeypatch):
    import brainevent.config as cfg
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", None)
    cfg.prefer_system_nvcc(True)
    assert kt.get_nvcc_discovery() == "system"
    cfg.prefer_system_nvcc(False)
    assert kt.get_nvcc_discovery() == "pip"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k nvcc_discovery -v`
Expected: FAIL — `get_nvcc_discovery`/`set_nvcc_discovery` 未定义。

- [ ] **Step 3: 实现偏好状态**

在 `brainevent/_op/kernix_toolchain.py`（CandidateProbe 之后）加入：

```python
_VALID_NVCC_DISCOVERY = ("pip", "system")
_NVCC_DISCOVERY: "str | None" = None  # None → fall back to env then default


def set_nvcc_discovery(prefer: str) -> None:
    """Set nvcc discovery preference: 'pip' (default) or 'system'."""
    if prefer not in _VALID_NVCC_DISCOVERY:
        raise ValueError(
            f"Invalid nvcc discovery preference {prefer!r}. "
            f"Valid: {_VALID_NVCC_DISCOVERY}"
        )
    global _NVCC_DISCOVERY
    _NVCC_DISCOVERY = prefer


def get_nvcc_discovery() -> str:
    """Resolve preference: explicit set > BRAINEVENT_NVCC_PREFER env > 'pip'."""
    if _NVCC_DISCOVERY is not None:
        return _NVCC_DISCOVERY
    env = os.environ.get("BRAINEVENT_NVCC_PREFER", "").strip().lower()
    if env in _VALID_NVCC_DISCOVERY:
        return env
    return "pip"
```

- [ ] **Step 4: 实现 config 快捷函数**

在 `brainevent/config.py` 末尾加入，并把 `'prefer_system_nvcc'` 加入文件顶部 `__all__`：

```python
# ──────────────────────────────────────────────────────────────────────
#  CUDA nvcc discovery preference
# ──────────────────────────────────────────────────────────────────────

def prefer_system_nvcc(enable: bool = True) -> None:
    """切换 nvcc 发现优先级。

    Parameters
    ----------
    enable : bool
        ``True`` → 优先使用系统 ``PATH`` 上的 nvcc；
        ``False`` → 优先使用 ``jax[cuda*]`` 自带的 pip nvcc（默认）。

    Examples
    --------
    .. code-block:: python

        >>> import brainevent
        >>> brainevent.config.prefer_system_nvcc()      # 改为系统优先
        >>> brainevent.config.prefer_system_nvcc(False) # 改回 pip 优先
    """
    from brainevent._op.kernix_toolchain import set_nvcc_discovery
    set_nvcc_discovery("system" if enable else "pip")
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k "nvcc_discovery or prefer_system" -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/config.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): add nvcc discovery preference and config.prefer_system_nvcc"
```

---

## Task 4: `_find_pip_cuda`（通用扫描 pip nvidia 包）

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`
- Test: `brainevent/_op/kernix_toolchain_test.py`

- [ ] **Step 1: 追加失败测试**

Append:

```python
def _touch_exec(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    p.chmod(0o755)


def test_find_pip_cuda_consolidated(tmp_path):
    _touch_exec(tmp_path / "cu13" / "bin" / "nvcc")
    (tmp_path / "cu13" / "include").mkdir(parents=True)
    res, probes = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res is not None
    path, includes = res
    assert path == str(tmp_path / "cu13" / "bin" / "nvcc")
    assert includes == [str(tmp_path / "cu13" / "include")]


def test_find_pip_cuda_consolidated_picks_highest(tmp_path):
    _touch_exec(tmp_path / "cu13" / "bin" / "nvcc")
    (tmp_path / "cu13" / "include").mkdir(parents=True)
    _touch_exec(tmp_path / "cu14" / "bin" / "nvcc")
    (tmp_path / "cu14" / "include").mkdir(parents=True)
    res, _ = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res[0] == str(tmp_path / "cu14" / "bin" / "nvcc")


def test_find_pip_cuda_split(tmp_path):
    _touch_exec(tmp_path / "cuda_nvcc" / "bin" / "nvcc")
    (tmp_path / "cuda_nvcc" / "include").mkdir(parents=True)
    (tmp_path / "cuda_runtime" / "include").mkdir(parents=True)
    res, _ = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res is not None
    path, includes = res
    assert path == str(tmp_path / "cuda_nvcc" / "bin" / "nvcc")
    assert str(tmp_path / "cuda_nvcc" / "include") in includes
    assert str(tmp_path / "cuda_runtime" / "include") in includes


def test_find_pip_cuda_absent(tmp_path):
    res, probes = kt._find_pip_cuda(roots=[str(tmp_path)])
    assert res is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k find_pip_cuda -v`
Expected: FAIL — `_find_pip_cuda` 未定义。

- [ ] **Step 3: 实现**

加入：

```python
def _nvcc_name() -> str:
    return "nvcc.exe" if sys.platform == "win32" else "nvcc"


def _nvidia_roots() -> "list[str]":
    """Return search roots of the 'nvidia' namespace package (empty if absent)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return list(spec.submodule_search_locations)


def _pip_include_for(pkg: str, roots: "list[str]") -> "str | None":
    for root in roots:
        inc = Path(root) / pkg / "include"
        if inc.is_dir():
            return str(inc)
    return None


def _find_pip_cuda(roots: "list[str] | None" = None):
    """Locate a pip-installed CUDA nvcc + its include dirs.

    Returns ``((nvcc_path, [include_dirs]) | None, [CandidateProbe])``.
    Handles consolidated ``nvidia/cuNN`` (cu13+) and split ``nvidia/cuda_nvcc``
    (cu12) layouts; consolidated wins, highest cuNN wins.
    """
    if roots is None:
        roots = _nvidia_roots()
    probes: "list[CandidateProbe]" = []
    exe = _nvcc_name()

    # A. consolidated cuNN layout (cu13 and future cuNN)
    best = None  # (version:int, nvcc_path:str, include_dir:str)
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        for entry in sorted(rp.iterdir()):
            if not re.match(r"cu\d+$", entry.name):
                continue
            cand = entry / "bin" / exe
            src = f"pip:nvidia/{entry.name}"
            if cand.is_file():
                ver = int(entry.name[2:])
                probes.append(CandidateProbe(src, str(cand), "ok"))
                if best is None or ver > best[0]:
                    best = (ver, str(cand), str(entry / "include"))
            else:
                probes.append(CandidateProbe(src, str(cand), "not-found"))
    if best is not None:
        return (best[1], [best[2]]), probes

    # B. split cu12 layout
    for root in roots:
        cand = Path(root) / "cuda_nvcc" / "bin" / exe
        src = "pip:nvidia/cuda_nvcc"
        if cand.is_file():
            includes = [str(Path(root) / "cuda_nvcc" / "include")]
            for pkg in ("cuda_runtime", "cuda_cccl"):
                inc = _pip_include_for(pkg, roots)
                if inc:
                    includes.append(inc)
            probes.append(CandidateProbe(src, str(cand), "ok"))
            return (str(cand), includes), probes
        probes.append(CandidateProbe(src, str(cand), "not-found"))

    if not roots:
        probes.append(CandidateProbe("pip:nvidia/*", "", "not-found"))
    return None, probes
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k find_pip_cuda -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): discover pip-installed nvcc across cu12/cuNN layouts"
```

---

## Task 5: `_find_host_cxx`（CXX → conda → 系统）

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`
- Test: `brainevent/_op/kernix_toolchain_test.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_find_host_cxx_prefers_cxx_env(monkeypatch, tmp_path):
    fake = tmp_path / "mygcc"
    _touch_exec(fake)
    monkeypatch.setenv("CXX", str(fake))
    cxx, probes = kt._find_host_cxx()
    assert cxx == str(fake)


def test_find_host_cxx_conda_before_system(monkeypatch, tmp_path):
    monkeypatch.delenv("CXX", raising=False)
    gpp = tmp_path / "conda" / "bin" / "g++"
    _touch_exec(gpp)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "conda"))
    monkeypatch.setattr(kt.shutil, "which", lambda n: "/usr/bin/" + n)
    cxx, probes = kt._find_host_cxx()
    assert cxx == str(gpp)


def test_find_host_cxx_system_fallback(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(kt.shutil, "which", lambda n: "/usr/bin/g++" if n == "g++" else None)
    cxx, probes = kt._find_host_cxx()
    assert cxx == "/usr/bin/g++"


def test_find_host_cxx_none(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(kt.shutil, "which", lambda n: None)
    cxx, probes = kt._find_host_cxx()
    assert cxx is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k find_host_cxx -v`
Expected: FAIL — `_find_host_cxx` 未定义。

- [ ] **Step 3: 实现**

```python
def _find_host_cxx(self_probes=None):
    """Find a host C++ compiler: CXX > conda > system PATH.

    Returns ``(cxx_path | None, [CandidateProbe])``.
    """
    probes: "list[CandidateProbe]" = []

    # 1. CXX
    cxx_env = os.environ.get("CXX")
    if cxx_env:
        resolved = cxx_env if os.path.isfile(cxx_env) else shutil.which(cxx_env)
        if resolved:
            probes.append(CandidateProbe("CXX", resolved, "ok"))
            return resolved, probes
        probes.append(CandidateProbe("CXX", cxx_env, "not-found"))
    else:
        probes.append(CandidateProbe("CXX", "", "unset"))

    # 2. conda
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        bindir = Path(conda) / "bin"
        names = [p.name for p in sorted(bindir.glob("*-g++"))] if bindir.is_dir() else []
        names += ["g++", "c++", "clang++"]
        for name in names:
            cand = bindir / name
            src = f"$CONDA_PREFIX/bin/{name}"
            if cand.is_file():
                probes.append(CandidateProbe(src, str(cand), "ok"))
                return str(cand), probes
            probes.append(CandidateProbe(src, str(cand), "not-found"))
    else:
        probes.append(CandidateProbe("$CONDA_PREFIX", "", "unset"))

    # 3. system PATH
    for name in ("g++", "c++", "clang++"):
        found = shutil.which(name)
        if found:
            probes.append(CandidateProbe(f"PATH:{name}", found, "ok"))
            return found, probes
        probes.append(CandidateProbe(f"PATH:{name}", "", "not-found"))

    return None, probes
```

> 注：参数 `self_probes` 仅占位以兼容关键字调用方式，无实际用途——**删除它**，签名应为 `def _find_host_cxx():`。（按 YAGNI，直接写无参版本。）

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k find_host_cxx -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): host compiler discovery (CXX > conda > system)"
```

---

## Task 6: `_select_nvcc` + 辅助（include/version 推导）

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`
- Test: `brainevent/_op/kernix_toolchain_test.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_select_nvcc_env_override(monkeypatch, tmp_path):
    nvcc = tmp_path / "cudahome" / "bin" / "nvcc"
    _touch_exec(nvcc)
    monkeypatch.setenv("BRAINEVENT_NVCC_PATH", str(nvcc))
    path, includes, probes = kt._select_nvcc()
    assert path == str(nvcc)
    assert includes == [str(tmp_path / "cudahome" / "include")]


def test_select_nvcc_pip_first(monkeypatch):
    monkeypatch.delenv("BRAINEVENT_NVCC_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", "pip")
    monkeypatch.setattr(kt, "_find_pip_cuda",
                        lambda roots=None: (("/pip/nvcc", ["/pip/include"]), []))
    monkeypatch.setattr(kt.shutil, "which", lambda n: "/usr/bin/nvcc")
    path, includes, _ = kt._select_nvcc()
    assert path == "/pip/nvcc"


def test_select_nvcc_system_pref(monkeypatch):
    monkeypatch.delenv("BRAINEVENT_NVCC_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", "system")
    monkeypatch.setattr(kt, "_find_pip_cuda",
                        lambda roots=None: (("/pip/nvcc", ["/pip/include"]), []))
    monkeypatch.setattr(kt.shutil, "which",
                        lambda n: "/usr/local/cuda/bin/nvcc" if n == "nvcc" else None)
    path, includes, _ = kt._select_nvcc()
    assert path == "/usr/local/cuda/bin/nvcc"
    assert includes == ["/usr/local/cuda/include"]


def test_select_nvcc_not_found(monkeypatch):
    monkeypatch.delenv("BRAINEVENT_NVCC_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(kt, "_NVCC_DISCOVERY", "pip")
    monkeypatch.setattr(kt, "_find_pip_cuda", lambda roots=None: (None, []))
    monkeypatch.setattr(kt.shutil, "which", lambda n: None)
    path, includes, probes = kt._select_nvcc()
    assert path is None and includes == []
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k select_nvcc -v`
Expected: FAIL — `_select_nvcc` 未定义。

- [ ] **Step 3: 实现**

```python
def _include_from_nvcc(nvcc_path: str) -> str:
    return str(Path(nvcc_path).resolve().parent.parent / "include")


def _cxx_version(cxx: str) -> str:
    try:
        proc = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=10)
        return proc.stdout.splitlines()[0].strip() if proc.stdout else ""
    except Exception:
        return ""


def _select_nvcc():
    """Pick nvcc per priority. Returns ``(nvcc|None, [include_dirs], [probes])``."""
    probes: "list[CandidateProbe]" = []
    exe = _nvcc_name()

    # 1. BRAINEVENT_NVCC_PATH
    env = os.environ.get("BRAINEVENT_NVCC_PATH")
    if env:
        if os.path.isfile(env):
            probes.append(CandidateProbe("BRAINEVENT_NVCC_PATH", env, "ok"))
            return env, [_include_from_nvcc(env)], probes
        probes.append(CandidateProbe("BRAINEVENT_NVCC_PATH", env, "not-a-file"))
    else:
        probes.append(CandidateProbe("BRAINEVENT_NVCC_PATH", "", "unset"))

    # 2. explicit CUDA_HOME
    home = os.environ.get("CUDA_HOME")
    if home:
        cand = os.path.join(home, "bin", exe)
        if os.path.isfile(cand):
            probes.append(CandidateProbe("$CUDA_HOME/bin/nvcc", cand, "ok"))
            return cand, [os.path.join(home, "include")], probes
        probes.append(CandidateProbe("$CUDA_HOME/bin/nvcc", cand, "not-found"))
    else:
        probes.append(CandidateProbe("$CUDA_HOME/bin/nvcc", "", "unset"))

    # 3. preference-ordered pip vs system PATH
    order = ("pip", "system") if get_nvcc_discovery() == "pip" else ("system", "pip")
    for src in order:
        if src == "pip":
            res, pip_probes = _find_pip_cuda()
            probes.extend(pip_probes)
            if res is not None:
                return res[0], res[1], probes
        else:
            which = shutil.which("nvcc")
            if which:
                probes.append(CandidateProbe("PATH:nvcc", which, "ok"))
                return which, [_include_from_nvcc(which)], probes
            probes.append(CandidateProbe("PATH:nvcc", "", "not-found"))

    return None, [], probes
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k select_nvcc -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): nvcc selection with overrides + pip/system preference"
```

---

## Task 7: `CudaToolchain` 改造 + 重写 `detect_cuda_toolchain`

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`
- Test: `brainevent/_op/kernix_toolchain_test.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_detect_cuda_toolchain_no_nvcc(monkeypatch):
    from brainevent._error import NvccNotFoundError
    monkeypatch.setattr(
        kt, "_select_nvcc",
        lambda: (None, [], [CandidateProbe("PATH:nvcc", "", "not-found")]),
    )
    with pytest.raises(NvccNotFoundError) as ei:
        kt.detect_cuda_toolchain()
    msg = str(ei.value)
    assert "E-NVCC" in msg
    assert "jax[cuda" in msg


def test_detect_cuda_toolchain_no_host_cxx(monkeypatch, tmp_path):
    from brainevent._error import HostCompilerNotFoundError
    nvcc = tmp_path / "cu13" / "bin" / "nvcc"
    _touch_exec(nvcc)
    monkeypatch.setattr(
        kt, "_select_nvcc",
        lambda: (str(nvcc), [str(tmp_path / "cu13" / "include")], []),
    )
    monkeypatch.setattr(kt.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "Cuda release 13.0", "stderr": ""})())
    monkeypatch.setattr(kt, "_find_host_cxx", lambda: (None, []))
    with pytest.raises(HostCompilerNotFoundError) as ei:
        kt.detect_cuda_toolchain()
    assert "E-CXX" in str(ei.value)
    assert "conda install" in str(ei.value)


def test_cuda_toolchain_dataclass_fields():
    tc = kt.CudaToolchain(
        nvcc="/n", cxx="/c", cuda_home="/h",
        cuda_include_dirs=("/i1", "/i2"),
        xla_ffi_include_dir="/x", brainevent_include_dir="/b",
        nvcc_version="v", cxx_version="g",
    )
    assert tc.cuda_include_dirs == ("/i1", "/i2")
    assert tc.cxx_version == "g"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k "detect_cuda_toolchain or dataclass_fields" -v`
Expected: FAIL — 字段不存在 / 抛错类型不对。

- [ ] **Step 3: 改 `CudaToolchain`**

把 `CudaToolchain`（约第 30–39 行）替换为：

```python
@dataclass(frozen=True)
class CudaToolchain:
    """Immutable description of available CUDA compilation tools."""
    nvcc: str
    cxx: str
    cuda_home: str
    cuda_include_dirs: tuple[str, ...]
    xla_ffi_include_dir: str
    brainevent_include_dir: str
    nvcc_version: str = ""
    cxx_version: str = ""
```

- [ ] **Step 4: 重写 `detect_cuda_toolchain`**

整体替换 `detect_cuda_toolchain()`（约第 51–118 行）为：

```python
def detect_cuda_toolchain() -> CudaToolchain:
    """Auto-detect nvcc, host C++ compiler, and include paths.

    Raises the matching layered exception (NvccNotFoundError /
    HostCompilerNotFoundError / HeaderNotFoundError) when a stage fails.
    """
    nvcc, cuda_include_dirs, nvcc_probes = _select_nvcc()
    if nvcc is None:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc 发现", code="E-NVCC",
            summary="未找到 CUDA 编译器 nvcc。",
            probes=nvcc_probes,
            remediation=[
                '安装 jax[cuda13]（自带 nvcc，无需系统 CUDA Toolkit）：pip install -U "jax[cuda13]"',
                "或设 BRAINEVENT_NVCC_PATH 指向 nvcc，或设 CUDA_HOME 指向 CUDA 安装目录",
                "若想优先用系统 PATH 上的 nvcc：brainevent.config.prefer_system_nvcc()",
            ],
        ))

    cuda_home = str(Path(nvcc).resolve().parent.parent)

    # nvcc version
    nvcc_version = ""
    proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
    for line in proc.stdout.splitlines():
        if "release" in line.lower():
            nvcc_version = line.strip()
            break
    if not nvcc_version:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc 发现", code="E-NVCC",
            summary=f"nvcc 存在但无法获取版本：{nvcc}",
            probes=nvcc_probes,
            compiler_output=(proc.stdout + proc.stderr),
            remediation=["确认该 nvcc 可执行且未损坏，或改用其它 nvcc。"],
        ))

    # host compiler
    cxx, cxx_probes = _find_host_cxx()
    if cxx is None:
        raise HostCompilerNotFoundError(render_toolchain_error(
            stage="host 编译器发现", code="E-CXX",
            summary="未找到 host C++ 编译器（nvcc 需要它编译 host 侧代码并链接）。pip 不提供 host 编译器。",
            probes=cxx_probes,
            remediation=[
                "conda 环境：conda install -c conda-forge gxx",
                "Debian/Ubuntu：sudo apt-get install g++",
                "RHEL/Fedora：sudo dnf install gcc-c++",
                "或设 CXX=/path/to/g++",
            ],
        ))
    cxx_version = _cxx_version(cxx)

    # XLA FFI include (from jaxlib)
    xla_ffi_include = jax.ffi.include_dir()
    ffi_header = os.path.join(xla_ffi_include, "xla", "ffi", "api", "ffi.h")
    if not os.path.isfile(ffi_header):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="未找到 XLA FFI 头 ffi.h（jaxlib 与 CUDA wheel 可能不配套或损坏）。",
            probes=[CandidateProbe("jaxlib:xla/ffi/api/ffi.h", ffi_header, "not-found")],
            remediation=['重装与 CUDA 匹配的 jaxlib：pip install -U "jax[cuda13]"'],
        ))

    # brainevent include
    be_include = str(Path(__file__).resolve().parent.parent / "include")
    if not os.path.isdir(be_include):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="brainevent include 目录缺失（包安装可能损坏）。",
            probes=[CandidateProbe("brainevent/include", be_include, "not-found")],
            remediation=["重装 brainevent：pip install -U --force-reinstall brainevent"],
        ))

    return CudaToolchain(
        nvcc=nvcc,
        cxx=cxx,
        cuda_home=cuda_home,
        cuda_include_dirs=tuple(cuda_include_dirs),
        xla_ffi_include_dir=xla_ffi_include,
        brainevent_include_dir=be_include,
        nvcc_version=nvcc_version,
        cxx_version=cxx_version,
    )
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k "detect_cuda_toolchain or dataclass_fields" -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): rewrite detect_cuda_toolchain with layered errors + multi-include"
```

---

## Task 8: `detect_cpp_toolchain` + `detect_cuda_arch` 分层错误

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`
- Test: `brainevent/_op/kernix_toolchain_test.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_detect_cpp_toolchain_no_cxx(monkeypatch):
    from brainevent._error import HostCompilerNotFoundError
    monkeypatch.setattr(kt, "_find_host_cxx", lambda: (None, []))
    with pytest.raises(HostCompilerNotFoundError) as ei:
        kt.detect_cpp_toolchain()
    assert "E-CXX" in str(ei.value)


def test_detect_cuda_arch_failure(monkeypatch):
    from brainevent._error import GpuArchDetectionError
    monkeypatch.delenv("BRAINEVENT_COMPUTE_CAPABILITIES", raising=False)

    def fake_run(*a, **k):
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "no smi"})()

    monkeypatch.setattr(kt.subprocess, "run", fake_run)
    with pytest.raises(GpuArchDetectionError) as ei:
        kt.detect_cuda_arch()
    assert "E-ARCH" in str(ei.value)


def test_detect_cuda_arch_env_override(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_COMPUTE_CAPABILITIES", "8.6,8.0")
    assert kt.detect_cuda_arch() == ["sm_86", "sm_80"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k "detect_cpp_toolchain or cuda_arch" -v`
Expected: FAIL — 抛错类型不对 / FileNotFoundError 未处理。

- [ ] **Step 3: 重写 `detect_cpp_toolchain`**

整体替换 `detect_cpp_toolchain()`（约第 121–157 行）为：

```python
def detect_cpp_toolchain() -> CppToolchain:
    """Auto-detect a host C++ compiler and include paths for CPU compilation."""
    cxx, cxx_probes = _find_host_cxx()
    if cxx is None:
        raise HostCompilerNotFoundError(render_toolchain_error(
            stage="host 编译器发现", code="E-CXX",
            summary="未找到 C++ 编译器（CPU 后端需要 g++/clang++）。",
            probes=cxx_probes,
            remediation=[
                "conda 环境：conda install -c conda-forge gxx",
                "Debian/Ubuntu：sudo apt-get install g++",
                "RHEL/Fedora：sudo dnf install gcc-c++",
                "或设 CXX=/path/to/g++",
            ],
        ))
    cxx_version = _cxx_version(cxx)

    xla_ffi_include = jax.ffi.include_dir()
    ffi_header = os.path.join(xla_ffi_include, "xla", "ffi", "api", "ffi.h")
    if not os.path.isfile(ffi_header):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="未找到 XLA FFI 头 ffi.h。",
            probes=[CandidateProbe("jaxlib:xla/ffi/api/ffi.h", ffi_header, "not-found")],
            remediation=["重装 jaxlib：pip install -U jax jaxlib"],
        ))

    be_include = str(Path(__file__).resolve().parent.parent / "include")
    if not os.path.isdir(be_include):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="brainevent include 目录缺失。",
            probes=[CandidateProbe("brainevent/include", be_include, "not-found")],
            remediation=["重装 brainevent：pip install -U --force-reinstall brainevent"],
        ))

    return CppToolchain(
        cxx=cxx,
        xla_ffi_include_dir=xla_ffi_include,
        brainevent_include_dir=be_include,
        cxx_version=cxx_version,
    )
```

- [ ] **Step 4: 重写 `detect_cuda_arch` 结尾**

把 `detect_cuda_arch()`（约第 229–254 行）的 `subprocess.run(...)` 调用包进 try/except，并把最后的 `raise KernelToolchainError(...)` 改为 `GpuArchDetectionError` + 渲染。替换函数体内"探测"部分为：

```python
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        result = None

    if result is not None and result.returncode == 0:
        caps = set()
        for line in result.stdout.strip().splitlines():
            cap = line.strip().replace(".", "")
            caps.add(f"sm_{cap}")
        if caps:
            return sorted(caps)

    raise GpuArchDetectionError(render_toolchain_error(
        stage="算力探测", code="E-ARCH",
        summary="无法通过 nvidia-smi 探测 GPU 算力（驱动缺失或 nvidia-smi 不可用）。",
        probes=[CandidateProbe("nvidia-smi", shutil.which("nvidia-smi") or "", "not-found")],
        remediation=[
            "安装/修复 NVIDIA 驱动，使 nvidia-smi 可用",
            "或设 BRAINEVENT_COMPUTE_CAPABILITIES（如 '8.6,8.0'）跳过自动探测",
        ],
    ))
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k "detect_cpp_toolchain or cuda_arch" -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): layered errors for cpp toolchain and arch detection"
```

---

## Task 9: `collect_toolchain_diagnostics` + `print_diagnostics` 复用

**Files:**
- Modify: `brainevent/_op/kernix_toolchain.py`, `brainevent/_op/kernix_pipeline.py`
- Test: `brainevent/_op/kernix_toolchain_test.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_collect_diagnostics_keys(monkeypatch):
    monkeypatch.setattr(kt, "_select_nvcc", lambda: ("/n/nvcc", ["/n/include"], []))
    monkeypatch.setattr(kt, "_find_host_cxx", lambda: ("/usr/bin/g++", []))
    monkeypatch.setattr(kt, "_cxx_version", lambda c: "g++ 12")
    snap = kt.collect_toolchain_diagnostics()
    assert snap["nvcc"] == "/n/nvcc"
    assert snap["host_cxx"] == "/usr/bin/g++"
    assert snap["discovery"] in ("pip", "system")
    assert "env:CUDA_HOME" in snap


def test_render_appends_snapshot_when_debug(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_TOOLCHAIN_DEBUG", "1")
    monkeypatch.setattr(kt, "collect_toolchain_diagnostics", lambda: {"nvcc": "/n"})
    msg = kt.render_toolchain_error(stage="x", code="E-X", summary="s")
    assert "工具链快照" in msg and "/n" in msg
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k "collect_diagnostics or appends_snapshot" -v`
Expected: FAIL — `collect_toolchain_diagnostics` 未定义。

- [ ] **Step 3: 实现 `collect_toolchain_diagnostics`**

在 `kernix_toolchain.py` 加入：

```python
def collect_toolchain_diagnostics() -> dict:
    """Single source of truth for a toolchain snapshot (never raises)."""
    snap: "dict[str, str]" = {}
    snap["discovery"] = get_nvcc_discovery()
    try:
        nvcc, includes, _ = _select_nvcc()
    except Exception:
        nvcc, includes = None, []
    snap["nvcc"] = nvcc or "<not found>"
    snap["cuda_include_dirs"] = ", ".join(includes) if includes else "<none>"
    try:
        cxx, _ = _find_host_cxx()
    except Exception:
        cxx = None
    snap["host_cxx"] = cxx or "<not found>"
    if cxx:
        snap["host_cxx_version"] = _cxx_version(cxx)
    for var in ("BRAINEVENT_NVCC_PATH", "CUDA_HOME", "CXX", "CONDA_PREFIX",
                "BRAINEVENT_NVCC_PREFER", "BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER",
                "BRAINEVENT_COMPUTE_CAPABILITIES"):
        snap[f"env:{var}"] = os.environ.get(var, "<unset>")
    return snap
```

- [ ] **Step 4: `print_diagnostics` 复用快照**

在 `brainevent/_op/kernix_pipeline.py` 把 import（约第 40 行）改为也引入 `collect_toolchain_diagnostics`：

```python
from .kernix_toolchain import (
    collect_toolchain_diagnostics, detect_cpp_toolchain, detect_cuda_arch,
    detect_cuda_toolchain, so_ext,
)
```

把 `print_diagnostics()`（约第 456–492 行）中"Toolchain"那段 `try/except`（约第 468–477 行）替换为：

```python
    # Toolchain (single-source snapshot)
    for k, v in collect_toolchain_diagnostics().items():
        print(f"{k}: {v}")
    try:
        archs = detect_cuda_arch()
        print(f"GPU architectures: {', '.join(archs)}")
    except Exception as e:
        print(f"GPU architectures: ERROR ({e.__class__.__name__})")
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_toolchain_test.py -k "collect_diagnostics or appends_snapshot" -v`
Expected: PASS。
冒烟：`python -c "import brainevent; brainevent.print_diagnostics()"` → 不抛异常（无 CUDA 时各项显示 `<not found>`）。

- [ ] **Step 6: Commit**

```bash
git add brainevent/_op/kernix_toolchain.py brainevent/_op/kernix_pipeline.py brainevent/_op/kernix_toolchain_test.py
git commit -m "feat(kernix): unified toolchain diagnostics snapshot + print_diagnostics reuse"
```

---

## Task 10: `kernix_compiler` — `-ccbin`、多 include、版本兼容

**Files:**
- Modify: `brainevent/_op/kernix_compiler.py`
- Test: `brainevent/_op/kernix_compiler_test.py` (create)

- [ ] **Step 1: 写失败测试**

Create `brainevent/_op/kernix_compiler_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for kernix_compiler flag/error helpers (hermetic, no nvcc needed)."""

from brainevent._op import kernix_compiler as kc


def test_is_host_incompat_true():
    assert kc._is_host_incompat(
        "error: unsupported GNU version! gcc versions later than 13 are not supported"
    )


def test_is_host_incompat_false():
    assert not kc._is_host_incompat("kernel.cu(10): error: expected a ';'")


def test_allow_unsupported_compiler(monkeypatch):
    monkeypatch.setenv("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", "1")
    assert kc._allow_unsupported_compiler() is True
    monkeypatch.setenv("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", "0")
    assert kc._allow_unsupported_compiler() is False
    monkeypatch.delenv("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", raising=False)
    assert kc._allow_unsupported_compiler() is False
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_compiler_test.py -v`
Expected: FAIL — `_is_host_incompat`/`_allow_unsupported_compiler` 未定义。

- [ ] **Step 3: 加辅助 + 导入**

在 `brainevent/_op/kernix_compiler.py` 顶部把 import 改为也引入新异常：

```python
from brainevent._error import CompilationError, HostCompilerIncompatibleError, KernelToolchainError
```

在文件靠上的位置（`_find_ninja` 附近）加入：

```python
_HOST_INCOMPAT_SIGNALS = (
    "unsupported gnu version",
    "unsupported clang version",
    "is not supported",
    "no longer supported",
)


def _is_host_incompat(output: str) -> bool:
    low = output.lower()
    return any(sig in low for sig in _HOST_INCOMPAT_SIGNALS)


def _allow_unsupported_compiler() -> bool:
    return os.environ.get("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _raise_compile_error(output: str, command: str, stage: str) -> None:
    if _is_host_incompat(output):
        msg = (
            "host C++ 编译器与当前 CUDA/nvcc 不兼容。\n"
            "如何修复:\n"
            "  1) 安装受支持版本的 gcc 并设 CXX=/path/to/g++\n"
            "  2) 或设 BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1 后重试"
        )
        raise HostCompilerIncompatibleError(msg, compiler_output=output, command=command, stage=stage)
    raise CompilationError("compilation failed", compiler_output=output, command=command, stage=stage)
```

- [ ] **Step 4: 运行辅助测试确认通过**

Run: `python -m pytest brainevent/_op/kernix_compiler_test.py -v`
Expected: PASS。

- [ ] **Step 5: 接入 `NinjaBuild._cuda_flags`**

替换 `NinjaBuild._cuda_flags()`（约第 161–185 行）中固定 include 段。把：

```python
        flags += [
            "--std=c++17",
            f"-O{self.optimization_level}",
            f"-I{self.toolchain.brainevent_include_dir}",
            f"-I{self.toolchain.xla_ffi_include_dir}",
            f"-I{self.toolchain.cuda_include_dir}",
        ]
```

改为：

```python
        flags += [
            "--std=c++17",
            f"-O{self.optimization_level}",
            "-ccbin", self.toolchain.cxx,
            f"-I{self.toolchain.brainevent_include_dir}",
            f"-I{self.toolchain.xla_ffi_include_dir}",
        ]
        for inc in self.toolchain.cuda_include_dirs:
            flags.append(f"-I{inc}")
        if _allow_unsupported_compiler():
            flags.append("-allow-unsupported-compiler")
```

并把 `NinjaBuild.build()` 里失败分支（约第 251–256 行）：

```python
        if result.returncode != 0:
            raise CompilationError(
                "ninja build failed",
                compiler_output=result.stderr + result.stdout,
                command=" ".join(cmd),
            )
```

改为：

```python
        if result.returncode != 0:
            _raise_compile_error(result.stderr + result.stdout, " ".join(cmd), stage="build")
```

- [ ] **Step 6: 接入直连 nvcc（`CUDABackend.compile_source`）**

替换该函数内构造 `cmd` 的固定段（约第 330–343 行）。把：

```python
        cmd = [
            self.toolchain.nvcc,
            src_path,
            "-shared",
            "-o", output_path,
            f"-arch={gpu_arch}",
            *nvcc_host_pic_flags(),  # -fPIC on Linux/macOS, nothing on Windows
            "--std=c++17",
            f"-O{optimization_level}",
            # Include paths (order matters for override semantics)
            "-I", self.toolchain.brainevent_include_dir,
            "-I", self.toolchain.xla_ffi_include_dir,
            "-I", self.toolchain.cuda_include_dir,
        ]
```

改为：

```python
        cmd = [
            self.toolchain.nvcc,
            src_path,
            "-shared",
            "-o", output_path,
            f"-arch={gpu_arch}",
            *nvcc_host_pic_flags(),  # -fPIC on Linux/macOS, nothing on Windows
            "--std=c++17",
            f"-O{optimization_level}",
            "-ccbin", self.toolchain.cxx,
        ]
        if _allow_unsupported_compiler():
            cmd.append("-allow-unsupported-compiler")
        # Include paths (order matters for override semantics)
        cmd += ["-I", self.toolchain.brainevent_include_dir,
                "-I", self.toolchain.xla_ffi_include_dir]
        for inc in self.toolchain.cuda_include_dirs:
            cmd += ["-I", inc]
```

并把该函数末尾失败分支（约第 367–372 行）：

```python
        if result.returncode != 0:
            raise CompilationError(
                "nvcc compilation failed",
                compiler_output=result.stderr + result.stdout,
                command=cmd_str,
            )
```

改为：

```python
        if result.returncode != 0:
            _raise_compile_error(result.stderr + result.stdout, cmd_str, stage="compile")
```

- [ ] **Step 7: 全量编译模块测试确认未回归**

Run: `python -m pytest brainevent/_op/kernix_compiler_test.py brainevent/_op/kernix_toolchain_test.py -v`
Expected: PASS（语法/导入无误；`cuda_include_dir` 已无残留引用——用 `grep -rn "cuda_include_dir\b" brainevent/` 确认仅剩 `cuda_include_dirs`）。

- [ ] **Step 8: Commit**

```bash
git add brainevent/_op/kernix_compiler.py brainevent/_op/kernix_compiler_test.py
git commit -m "feat(kernix): -ccbin + multi include dirs + host-incompat compile errors"
```

---

## Task 11: `kernix_runtime` — dlopen 失败包成 `KernelLoadError`

**Files:**
- Modify: `brainevent/_op/kernix_runtime.py`
- Test: `brainevent/_op/kernix_runtime_load_test.py` (create)

- [ ] **Step 1: 写失败测试**

Create `brainevent/_op/kernix_runtime_load_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for CompiledModule load-time error wrapping."""

import pytest

from brainevent._op import kernix_runtime as kr
from brainevent._error import KernelLoadError


def test_dlopen_failure_wrapped(monkeypatch):
    def boom(path):
        raise OSError("libcudart.so.12: cannot open shared object file: No such file or directory")
    monkeypatch.setattr(kr.ctypes, "CDLL", boom)
    with pytest.raises(KernelLoadError) as ei:
        kr.CompiledModule("/tmp/does-not-exist.so", ["f"])
    msg = str(ei.value)
    assert "E-LOAD" in msg
    assert "/tmp/does-not-exist.so" in msg
    assert "LD_LIBRARY_PATH" in msg
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest brainevent/_op/kernix_runtime_load_test.py -v`
Expected: FAIL — 现在 `CDLL` 的 `OSError` 未被包成 `KernelLoadError`。

- [ ] **Step 3: 实现**

在 `brainevent/_op/kernix_runtime.py` 顶部 import 改为：

```python
from brainevent._error import KernelError, KernelLoadError, KernelRegistrationError
```

在 `CompiledModule` 之前加一个格式化辅助：

```python
def _format_load_error(so_path: str, err: Exception) -> str:
    msg = str(err)
    low = msg.lower()
    lines = [
        "[brainevent GPU 工具链] 加载 .so 失败  (code=E-LOAD)",
        "",
        f"原因: 无法 dlopen 编译产物：{so_path}",
        f"dlopen: {msg}",
        "",
        "如何修复:",
    ]
    if "cudart" in low or "cannot open shared object" in low:
        lines += [
            "  1) 缺少 CUDA 运行库（典型 cu12）。确保 jax[cuda*] 已正确安装。",
            "  2) 把 CUDA 运行库目录加入 LD_LIBRARY_PATH（如 site-packages/nvidia/cuda_runtime/lib）。",
        ]
    else:
        lines += ["  1) 确认编译成功且依赖库可用；设 BRAINEVENT_TOOLCHAIN_DEBUG=1 查看工具链快照。"]
    return "\n".join(lines)
```

把 `CompiledModule.__init__`（约第 149–167 行）里：

```python
        self._so_path = str(so_path)
        self._lib = ctypes.CDLL(self._so_path)
        self._functions: dict[str, ctypes._CFuncPtr] = {}
```

改为：

```python
        self._so_path = str(so_path)
        try:
            self._lib = ctypes.CDLL(self._so_path)
        except OSError as e:
            raise KernelLoadError(_format_load_error(self._so_path, e)) from e
        self._functions: dict[str, ctypes._CFuncPtr] = {}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest brainevent/_op/kernix_runtime_load_test.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add brainevent/_op/kernix_runtime.py brainevent/_op/kernix_runtime_load_test.py
git commit -m "feat(kernix): wrap dlopen failures into KernelLoadError (E-LOAD)"
```

---

## Task 12: cache key 纳入 host `cxx_version`

**Files:**
- Modify: `brainevent/_op/kernix_pipeline.py`

- [ ] **Step 1: 改 cache key**

在 `load_cuda_inline`（约第 159–170 行）把：

```python
    cache_key = _cache.cache_key(
        source=user_source,
        arch=gpu_arch,
        cxx_version=toolchain.nvcc_version,
```

改为（同时纳入 nvcc 与 host 编译器版本，host 编译器变化也会触发重编）：

```python
    cache_key = _cache.cache_key(
        source=user_source,
        arch=gpu_arch,
        cxx_version=f"{toolchain.nvcc_version}|{toolchain.cxx_version}",
```

- [ ] **Step 2: 冒烟验证导入无误**

Run: `python -c "import brainevent._op.kernix_pipeline as p; print('ok')"`
Expected: 打印 `ok`。

- [ ] **Step 3: Commit**

```bash
git add brainevent/_op/kernix_pipeline.py
git commit -m "fix(kernix): include host cxx version in CUDA cache key"
```

---

## Task 13: 文档 — GPU 依赖三件套

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 加 GPU 依赖说明**

在 `README.md` 的安装/GPU 相关章节追加一节（中英按 README 现有语言风格择一；以下为中文示例）：

```markdown
### GPU 编译依赖

在 GPU 上首次运行内核时，brainevent 会即时编译 CUDA 源码。需要三样东西：

1. **NVIDIA 驱动**（提供 `libcuda` 与 `nvidia-smi`）—— 系统层，任何方案都需要。
2. **`jax[cuda12]` 或 `jax[cuda13]`** —— 它会自动安装 `nvidia-*` pip 包，其中自带
   `nvcc`/`ptxas`/CUDA 运行库/头文件，**因此无需单独安装系统 CUDA Toolkit**。
3. **host C++ 编译器（`g++`/`clang++`）** —— pip 不提供，请用
   `conda install -c conda-forge gxx` / `sudo apt-get install g++` / `sudo dnf install gcc-c++` 安装。

可选环境变量 / 配置：

- `brainevent.config.prefer_system_nvcc()`：改为优先使用系统 `PATH` 上的 nvcc（默认优先 pip 自带）。
- `BRAINEVENT_NVCC_PREFER=pip|system`、`BRAINEVENT_NVCC_PATH`、`CUDA_HOME`、`CXX`。
- `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1`：host gcc 版本超出 nvcc 支持范围时强制编译。
- `BRAINEVENT_COMPUTE_CAPABILITIES=8.6,8.0`：跳过 `nvidia-smi` 自动探测。
- `BRAINEVENT_TOOLCHAIN_DEBUG=1`：所有工具链错误末尾附"工具链快照"，便于排障。
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document GPU compile dependencies and toolchain env vars"
```

---

## Self-Review

**Spec coverage（逐节核对）:**

- §3 决策1（pip 优先）→ Task 6 `_select_nvcc` + Task 3 偏好。✓
- §3 决策2（切换快捷函数）→ Task 3 `prefer_system_nvcc`。✓
- §3 决策3（conda 优先 host）→ Task 5 `_find_host_cxx`。✓
- §3 决策4（`-ccbin`）→ Task 10。✓
- §4.1 通用发现（cu12/cuNN）→ Task 4。✓
- §4.4 dataclass（`cuda_include_dirs`+`cxx_version`）→ Task 7 + Task 10 接入。✓
- §4.5 运行时链接 → 由 Task 11 的 `KernelLoadError` 给出 cu12 缺库指引；实际 `-L`/rpath 是验证项（见下）。✓（验证项）
- §6 分层报错（7+1 类、渲染器、probe、快照、各层指引）→ Task 1/2/9 + 各 detect 接入。✓
- §8 验证项 → 下方"Manual verification"。
- §4.2 `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER`、版本不兼容 → Task 10。✓
- 缓存纳入 host cxx_version → Task 12。✓
- §7 文档 → Task 13。✓

**Placeholder scan:** 无 TBD/TODO；Task 5 Step 3 显式纠正了占位参数（按无参实现）。

**Type consistency:** `CudaToolchain.cuda_include_dirs: tuple[str, ...]`（Task 7）在 Task 10 两处以 `for inc in self.toolchain.cuda_include_dirs` 遍历，名称一致；`CompilationError(stage=...)`（Task 1）在 Task 10 `_raise_compile_error` 使用一致；`render_toolchain_error` 关键字参数（Task 2）在所有调用点一致。

## Manual verification（需真实环境，非 CI）

- 仅装 `jax[cuda13]`、无系统 CUDA：跑一个现有 GPU 内核，确认 nvcc 被 `_find_pip_cuda` 发现并编译、`.so` 能 import、`print_diagnostics()` 显示 pip nvcc。
- 仅装 `jax[cuda12]`：验证 cudart 链接/加载；若出现 `libcudart.so.12` dlopen 失败（E-LOAD），按 §4.5 决定是否在 cu12 分支追加 `-L <cuda_runtime/lib>` + rpath（届时补一个后续 Task）。
- 系统 CUDA 在 PATH：`brainevent.config.prefer_system_nvcc()` 后确认走系统 nvcc。
- `BRAINEVENT_TOOLCHAIN_DEBUG=1`：制造一次 nvcc 缺失，确认错误末尾出现"工具链快照"。
