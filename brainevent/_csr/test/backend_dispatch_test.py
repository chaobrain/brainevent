"""Object and global backend choices select the same CSR mirror route."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent as be
import brainevent._csr.main as csr_main


@pytest.fixture(autouse=True)
def restore_global_backends():
    previous = {platform: be.config.get_backend(platform) for platform in ('cpu', 'gpu', 'tpu')}
    be.config.clear_backends()
    try:
        yield
    finally:
        be.config.clear_backends()
        for platform, backend in previous.items():
            if backend is not None:
                be.config.set_backend(platform, backend)


def _product(matrix, values, weighted):
    events = be.WeightedArray(values) if weighted else be.BinaryArray(values)
    return matrix @ events if isinstance(matrix, be.CSR) else events @ matrix


def _spy_products(monkeypatch):
    calls = []

    def fake(indexed):
        def product(*args, shape, transpose, backend, weighted, **kwargs):
            calls.append((indexed, transpose, backend, weighted))
            rows = shape[1] if transpose else shape[0]
            return jnp.zeros((rows,) + args[-1].shape[1:], dtype=args[0].dtype)
        return product

    for name in ('binary_csrmv', 'binary_csrmm'):
        monkeypatch.setattr(csr_main, name, fake(False))
        monkeypatch.setattr(csr_main, name + '_indexed', fake(True))
    return calls


@pytest.mark.parametrize('cls', [be.CSR, be.CSC])
@pytest.mark.parametrize('ndim', [1, 2], ids=['mv', 'mm'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('backend,global_backend,prebuilt,indexed', [
    (None, 'cuda_raw', False, True),
    (None, 'cuda_raw', True, True),
    (None, None, False, False),
    (None, 'jax_raw', False, False),
    ('jax_raw', 'cuda_raw', False, False),
    ('cuda_raw', 'jax_raw', False, True),
], ids=['global_cuda_lazy', 'global_cuda_prebuilt', 'unset', 'global_jax',
        'object_jax', 'object_cuda'])
def test_event_backend_precedence(monkeypatch, cls, ndim, weighted, backend,
                                  global_backend, prebuilt, indexed):
    matrix = cls.fromdense(jnp.array([[1., 0., 2.], [0., 3., 4.]]), backend=backend,
                           precompute_weight_indices=prebuilt)
    be.config.set_backend('gpu', global_backend)
    calls = _spy_products(monkeypatch)
    size = matrix.shape[1] if cls is be.CSR else matrix.shape[0]
    shape = (size,) if ndim == 1 else ((size, 4) if cls is be.CSR else (4, size))

    _product(matrix, jnp.ones(shape), weighted)

    assert calls == [(indexed, indexed, backend, weighted)]
    assert matrix.backend == backend  # Do not turn the GPU default into an explicit CPU request.
    assert (matrix._cache.mirror is not None) == (prebuilt or indexed)


def test_global_changes_retrace_existing_jit_route(monkeypatch):
    matrix = be.CSR.fromdense(jnp.array([[1., 0., 2.], [0., 3., 4.]]))
    calls = _spy_products(monkeypatch)
    product = jax.jit(lambda values: matrix @ be.BinaryArray(values))
    values = jnp.ones(3)
    for backend in ('jax_raw', 'cuda_raw', None):
        be.config.set_backend('gpu', backend)
        product(values).block_until_ready()
    assert [call[0] for call in calls] == [False, True, False]
    assert all(call[2] is None for call in calls)


def test_gpu_global_backend_keeps_cpu_execution_on_cpu(monkeypatch):
    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        dense = jnp.array([[1., 0., 2.], [0., 3., 4.]])
        matrix = be.CSR.fromdense(dense)
        values = jnp.array([1., 0., -2.])
        be.config.set_backend('cpu', 'jax_raw')
        be.config.set_backend('gpu', 'cuda_raw')
        calls = []
        original = csr_main.binary_csrmv_indexed

        def indexed(*args, **kwargs):
            calls.append(kwargs['backend'])
            return original(*args, **kwargs)

        monkeypatch.setattr(csr_main, 'binary_csrmv_indexed', indexed)
        np.testing.assert_allclose(matrix @ be.WeightedArray(values), dense @ values)
        assert calls == [None]


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='CUDA hybrid lowering needs a GPU')
@pytest.mark.parametrize('cls', [be.CSR, be.CSC])
@pytest.mark.parametrize('ndim', [1, 2], ids=['mv', 'mm'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
def test_global_cuda_uses_hybrid_forward_and_weight_gradient(cls, ndim, weighted):
    dense = jnp.array([[1., 0., 2.], [0., 3., 4.]])
    matrix = cls.fromdense(dense, precompute_weight_indices=True)
    be.config.set_backend('gpu', 'cuda_raw')
    size = matrix.shape[1] if cls is be.CSR else matrix.shape[0]
    shape = (size,) if ndim == 1 else ((size, 4) if cls is be.CSR else (4, size))
    values = (jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) % 3) - 1
    factors = values if weighted else (values != 0).astype(values.dtype)
    expected = dense @ factors if cls is be.CSR else factors @ dense
    product = jax.jit(lambda mat, x: _product(mat, x, weighted))
    lowered = product.lower(matrix, values)
    symbol = 'hybrid_spmv_col_major_' if ndim == 1 else 'batched_spmm_v3_col_major_'
    assert symbol in lowered.as_text()
    np.testing.assert_allclose(lowered.compile()(matrix, values), expected, rtol=1e-5, atol=1e-6)

    gradient = jax.jit(jax.grad(lambda data: _product(matrix.with_data(data), values, weighted).sum()))
    lowered_gradient = gradient.lower(matrix.data)
    symbol = 'hybrid_sddmv_' if ndim == 1 else 'batched_v3_sddmm'
    assert symbol in lowered_gradient.as_text()
    coo = matrix.tocoo()
    per_input = factors if ndim == 1 else factors.sum(axis=1 if cls is be.CSR else 0)
    expected_gradient = per_input[coo.col if cls is be.CSR else coo.row]
    np.testing.assert_allclose(lowered_gradient.compile()(matrix.data), expected_gradient,
                               rtol=1e-5, atol=1e-6)
