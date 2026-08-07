# Contributing to BrainEvent

Thanks for your interest in `brainevent`! This project provides data structures and
algorithms for event-driven computation on CPUs, GPUs, and TPUs, and it is developed
in the open — bug reports, documentation fixes, new sparse formats, and new kernel
backends are all welcome.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

- **Repository**: https://github.com/chaobrain/brainevent
- **Documentation**: https://brainx.chaobrain.com/brainevent/
- **Issue tracker**: https://github.com/chaobrain/brainevent/issues


## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running the tests](#running-the-tests)
- [Type checking](#type-checking)
- [Building the documentation](#building-the-documentation)
- [Code style](#code-style)
- [Submitting a pull request](#submitting-a-pull-request)
- [Contributing GPU kernels](#contributing-gpu-kernels)


## Ways to contribute

| I want to... | Do this |
| --- | --- |
| Report a bug | Open a [bug report](https://github.com/chaobrain/brainevent/issues/new?template=bug_report.md) — include your Python, JAX, and `brainevent` versions, the platform (CPU/GPU/TPU), and a minimal reproducer. |
| Request a feature | Open a [feature request](https://github.com/chaobrain/brainevent/issues/new?template=feature_request.md) describing the use case, not just the proposed API. |
| Report a security issue | Follow [SECURITY.md](SECURITY.md). |
| Fix something | Send a pull request — see [Submitting a pull request](#submitting-a-pull-request). |
| Improve the docs | Docs live in `docs/`; prose fixes are just as valuable as code. |

For large or architectural changes (a new sparse format, a new kernel backend, a
breaking API change), please **open an issue to discuss the design first**. It is much
cheaper to agree on an approach before the implementation exists.


## Development setup

`brainevent` requires **Python >= 3.11** (3.11–3.14 are supported) and **JAX >= 0.8.0**.

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/brainevent.git
cd brainevent

# 2. Create an isolated environment (conda, venv, uv — your choice)
conda create -n brainevent python=3.13
conda activate brainevent

# 3. Install the development dependencies
pip install -r requirements-dev-cpu.txt   # CPU: JAX + numba backends
# pip install -r requirements-dev-gpu.txt # GPU: adds warp-lang

# 4. Install brainevent itself in editable mode
pip install -e .

# 5. Install the git hooks
pip install pre-commit
pre-commit install
```

The two development requirement files differ by backend:

- `requirements-dev-cpu.txt` — pulls in `numba`, which powers the CPU kernel path.
- `requirements-dev-gpu.txt` — pulls in `warp-lang` for the GPU kernel path.

Both include `pytest` and `pytest-cov`. If you prefer extras, the equivalent
declarations live in `pyproject.toml`: `brainevent[cpu]`, `[cuda12]`, `[cuda13]`,
`[tpu]`, and `[testing]`.


## Running the tests

Tests are **co-located with the code** as `*_test.py` files next to the module they
cover (for example `brainevent/_op/numba_ffi_test.py`), rather than in a separate
top-level `tests/` directory.

```bash
# Fast run — this is the sensible default while developing.
# The `slow` marker is deselected automatically (see [tool.pytest.ini_options]).
pytest brainevent/

# A single module or a single test
pytest brainevent/_op/numba_ffi_test.py
pytest brainevent/_op/numba_ffi_test.py -k csr -q

# FULL suite, including the `slow` variants. THIS IS WHAT CI RUNS —
# run it before opening a pull request.
pytest brainevent/ -m ""

# Only the slow variants
pytest brainevent/ -m slow

# With coverage (config lives in [tool.coverage] in pyproject.toml)
pytest brainevent/ -m "" --cov=brainevent --cov-report=term
```

The `slow` marker covers compilation-heavy backend variants (numba / native kernels)
that recompile per test and dominate wall-clock time. The root `conftest.py` applies
the marker automatically based on the backend parameter, so you do not need to mark
tests by hand.

On Windows, CI adds `-p no:faulthandler`; do the same locally if you hit
faulthandler-related crashes.

**Test expectations for a pull request:**

- New features and bug fixes come with tests. For a bug fix, write a test that
  reproduces the bug *first*, then fix until it passes.
- Aim for >90% coverage of new code, but favour meaningful tests over trivial ones —
  cover edge cases and critical paths, not merely lines.
- Be explicit about dtypes in tests. Relying on implicit floating-point dtypes makes
  tests fragile under `jax_enable_x64`.


## Type checking

`brainevent` ships inline type information (PEP 561, via `brainevent/py.typed`), and
CI enforces a mypy gate:

```bash
pip install -r requirements-typecheck.txt   # pins mypy==2.3.0
mypy brainevent/
```

The gate deliberately runs with **only mypy installed** — no runtime dependencies — so
every third-party import collapses to `Any` via `ignore_missing_imports`. This keeps
the verdict reproducible. `python_version` and `platform` are pinned in
`[tool.mypy]`, and `*_test.py` files are excluded.

Prioritise meaningful annotations that improve readability over exhaustive coverage of
every local variable.


## Building the documentation

```bash
pip install -r requirements-doc.txt
cd docs
make html          # output lands in docs/_build/html
```

The documentation follows the [Diátaxis](https://diataxis.fr/) structure — put your
change in the right place:

| Directory | Purpose |
| --- | --- |
| `docs/getting-started/` | Installation and quickstart |
| `docs/tutorials/` | Learning-oriented, step-by-step |
| `docs/how-to/` | Task-oriented recipes for a concrete problem |
| `docs/explanation/` | Understanding-oriented background and design rationale |
| `docs/reference/` | API and kernel reference, changelog |


## Code style

- **Docstrings**: all public classes, methods, and functions use
  [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html),
  with sections in the canonical order: Short summary, Extended summary, Parameters,
  Returns/Yields, Raises, See Also, Notes, References, Examples. Wrap example code in a
  `.. code-block:: python` directive, prefix input lines with `>>>`, and make examples
  self-contained (include the imports).
- **License header**: every new source file starts with the Apache-2.0 header used
  throughout the codebase.
- **Linting**: `flake8` plus end-of-file / trailing-whitespace / debug-statement hooks
  run through `pre-commit`. Run them across the tree with:

  ```bash
  pre-commit run --all
  ```

- **Comments**: explain *why*, not *what*. The existing code documents non-obvious
  ABI, caching, and numerical decisions inline — please match that density.

`AGENTS.md` in the repository root records the same conventions in the form used by
AI coding assistants; it is the single source of truth for the docstring rules above.


## Submitting a pull request

1. **Branch off `main`** in your fork — use a descriptive name such as
   `fix-csr-transpose` or `add-fixed-conn-vmap`.
2. **Make focused commits.** One logical change per pull request keeps review tractable.
3. **Run the full local gate** before pushing:

   ```bash
   pytest brainevent/ -m ""
   mypy brainevent/
   pre-commit run --all
   ```

4. **Update `changelog.md`** for user-visible changes. The format follows
   [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
   [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
5. **Update the docs** if you changed behaviour or public API.
6. **Open the pull request** and fill in
   [the template](.github/PULL_REQUEST_TEMPLATE.md) — describe what changed, why, and
   how you tested it. Link the issue it fixes with `Fixes #<issue_number>`.
7. **Green CI is required.** Continuous integration runs the full test suite on Linux,
   macOS, and Windows (Python 3.13), plus the mypy gate and a Codecov upload.
8. **Respond to review.** Push follow-up commits to the same branch; the pull request
   updates automatically.

All contributions are licensed under the project's [Apache 2.0 License](LICENSE).


## Contributing GPU kernels

GPU work has extra requirements beyond the Python dependencies, because
`brainevent` compiles CUDA sources on the fly the first time a kernel runs:

1. An **NVIDIA driver** (provides `libcuda` and `nvidia-smi`).
2. **`jax[cuda12]` or `jax[cuda13]`** — these pull in the `nvidia-*` pip packages that
   bundle `nvcc`, `ptxas`, and the CUDA runtime/headers, so a separate system CUDA
   Toolkit is *not* required.
3. A **host C++ compiler** (`g++` / `clang++`), which pip does not provide:
   `conda install -c conda-forge gxx`, `sudo apt-get install g++`, or
   `sudo dnf install gcc-c++`.

Useful environment variables while debugging a toolchain problem:

| Variable | Effect |
| --- | --- |
| `BRAINEVENT_TOOLCHAIN_DEBUG=1` | Append a toolchain snapshot to every toolchain error |
| `BRAINEVENT_NVCC_PREFER=pip\|system` | Choose the pip-bundled or system `nvcc` |
| `BRAINEVENT_NVCC_PATH`, `CUDA_HOME`, `CXX` | Point at a specific toolchain |
| `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1` | Force compilation when host gcc is newer than nvcc supports |
| `BRAINEVENT_COMPUTE_CAPABILITIES=8.6,8.0` | Skip `nvidia-smi` auto-detection |

Note that the project's CI runs on CPU only. If your change touches a GPU code path,
please say so explicitly in the pull request and describe the hardware you tested on —
reviewers cannot verify it from CI alone.

Background reading before writing a kernel:

- `docs/explanation/custom-kernel-architecture.rst` — how the backends fit together
- `docs/how-to/building-extending/compile-raw-cuda-cpp.rst` — compiling raw CUDA/C++
- `docs/reference/kernels/` — argument specs, caching, and compiler options


## Questions?

Open an issue or start a discussion on the
[issue tracker](https://github.com/chaobrain/brainevent/issues). `brainevent` is part
of the [BrainX](https://brainx.chaobrain.com/) brain modeling ecosystem.
