# Security Policy

## Supported versions

Security fixes are applied to the latest released version of `brainevent`. We do not
backport fixes to older minor versions — please upgrade before reporting an issue.

| Version | Supported |
| --- | --- |
| Latest release on [PyPI](https://pypi.org/project/brainevent/) | ✅ |
| Older releases | ❌ — please upgrade |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.** Doing so discloses the problem to everyone before a
fix is available.

Instead, use one of these private channels:

1. **GitHub private vulnerability reporting** (preferred) — go to the
   [Security tab](https://github.com/chaobrain/brainevent/security/advisories/new) and
   click **Report a vulnerability**. This keeps the report visible only to the
   maintainers until an advisory is published.
2. **Email** — <chao.brain@qq.com>.

Please include as much of the following as you can:

- The type of issue and the affected component or module.
- Full paths of the source files involved, and the version/commit affected.
- Step-by-step instructions to reproduce, ideally a minimal proof of concept.
- Your environment: `brainevent`, `jax`/`jaxlib`, and Python versions; OS; and
  platform (CPU / GPU / TPU).
- The impact you believe the issue has, including how an attacker might exploit it.

## What to expect

| Stage | Target |
| --- | --- |
| Acknowledgement of your report | within 5 days |
| Detailed response with next steps | within 10 days |
| Progress updates | until the issue is resolved |

These timelines may extend when maintainers are away, particularly around the end of
the year. After the initial reply we will keep you informed of progress toward a fix
and a public announcement, and may ask for additional information or guidance.

We support coordinated disclosure: we ask that you give us a reasonable opportunity to
release a fix before publishing details. With your permission, we will credit you in
the resulting security advisory.

## Scope

### What is in scope

Bugs in `brainevent` itself that let an attacker affect confidentiality, integrity, or
availability beyond what the documented API allows — for example memory-safety defects
in the compiled CPU/GPU kernels (out-of-bounds reads or writes driven by *data* rather
than by attacker-supplied source code), or flaws in the on-disk kernel cache that let
one user's build influence another's.

### What is out of scope

`brainevent` **compiles and executes native code at runtime by design.** The public
API includes `load_cpp_inline`, `load_cuda_inline`, `load_cuda_file`, `load_cuda_dir`,
and `load_cpp_file`, which pass source you supply to a host C++/CUDA compiler and load
the result into the process. Likewise, the `numba` backends JIT-compile Python kernel
functions.

Consequently:

- **Passing untrusted or attacker-controlled source, kernel functions, or compiler
  flags to these APIs is arbitrary code execution — and is not a vulnerability.** Treat
  kernel source with exactly the same trust you would give any other code you import
  and run. Do not build services that compile source supplied by untrusted users.
- Reports that amount to "I passed malicious C++/CUDA to `load_cpp_inline` and it ran"
  will be closed as working-as-intended.
- The compilation toolchain (`nvcc`, `g++`/`clang++`) and the environment variables
  that select it (`BRAINEVENT_NVCC_PATH`, `CUDA_HOME`, `CXX`, and related settings) are
  part of the trusted local environment. An attacker who can already set your
  environment variables or write to your `PATH` has broader access than `brainevent`
  can defend against.

Also out of scope: vulnerabilities in third-party dependencies (JAX, jaxlib, numba,
warp-lang, NumPy, and the CUDA toolchain). Report those to their respective
maintainers — though we appreciate a heads-up if `brainevent` is affected, so we can
adjust our version constraints.

## Reporting a bug in a third party module

Security bugs in third-party modules should be reported to their respective
maintainers. If the issue reaches users *through* `brainevent`, please tell us too.
