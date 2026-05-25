# 2026-05-25

## Commit: `test: align sparse masks with fromdense structure`

修复 CI 上 macOS 偶发的 CSR plasticity 测试失败。根因是测试使用 `mask = mat < threshold` 表示稀疏连接存在，但 `CSR.fromdense()` 和 `COO.fromdense()` 实际只保留 `mat != 0` 的元素；当随机数恰好生成精确 `0.0` 时，dense 期望会把它当作存在的连接更新，而 sparse 结构已经丢掉该项，导致结果不一致。

本次将相关测试中的 mask 改为同时满足阈值条件和非零条件，例如 `(mat < 0.5) & (mat != 0.)`，使测试期望与 sparse `fromdense()` 的真实结构一致。覆盖范围包括 CSR/COO plasticity 测试，以及 CSR `diag_add` 测试中同类的 mask 构造。

验证记录：

- `JAX_PLATFORM_NAME=cpu pytest -q 'brainevent/_csr/plasticity_binary_test.py::Test_on_post::test_csr_on_post_v2[None-0.1-shape1-jax_raw]' --tb=short`: 1 passed
- `pytest -q brainevent/_csr/plasticity_binary_test.py --tb=short`: 108 passed
- `pytest -q brainevent/_coo/plasticity_binary_test.py --tb=short`: 67 passed, 1 skipped
- `JAX_PLATFORM_NAME=cpu pytest -q brainevent/_csr/plasticity_binary_test.py -k jax_raw --tb=short`: 54 passed, 54 deselected

备注：`Test_diag_add` 的同类 mask 已同步修复，但当前本地环境缺少 `numba`，该测试组无法完成运行，失败原因为 `ModuleNotFoundError: No module named 'numba'`。
