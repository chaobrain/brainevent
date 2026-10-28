# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JITC ``tocsr`` / ``dt2t`` under ``jit``, ``vmap`` and ``grad``, and prepared structures."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._jit_normal.csr import jitn_to_csr
from brainevent._jit_scalar.csr import jits_structure, jits_to_csr
from brainevent._jit_uniform.csr import jitu_to_csr
from brainevent._jitc_structure import JITCStructure
from brainevent._test_util import jax_x64_enabled

SHAPE = (12, 9)
SEED = 7

#: ``(class name, weight parameters)`` for every family and orientation class.
CASES = [
    ('JITCScalarR', (1.5,)),
    ('JITCScalarC', (1.5,)),
    ('JITCNormalR', (1.5, 0.3)),
    ('JITCNormalC', (1.5, 0.3)),
    ('JITCUniformR', (0.5, 1.5)),
    ('JITCUniformC', (0.5, 1.5)),
]
CASE_IDS = [c for c, _ in CASES]


def _matrix(cls_name, params, prob, corder, seed=SEED):
    return getattr(brainevent, cls_name)(*params, prob=prob, seed=seed, shape=SHAPE, corder=corder)


def _assert_same_csr(got, expected):
    np.testing.assert_array_equal(np.asarray(got.data), np.asarray(expected.data))
    np.testing.assert_array_equal(np.asarray(got.indices), np.asarray(expected.indices))
    np.testing.assert_array_equal(np.asarray(got.indptr), np.asarray(expected.indptr))


@pytest.fixture(autouse=True)
def _quiet_performance_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', brainevent.PerformanceWarning)
        yield


@pytest.mark.parametrize('prob', [0.25, 0.3])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('cls_name, params', CASES, ids=CASE_IDS)
def test_tocsr_matches_the_dense_matrix_of_the_matmul_kernels(cls_name, params, prob, corder):
    """``tocsr`` (deterministic row-stream fill) and ``todense`` (matmul kernels) draw one matrix."""
    mat = _matrix(cls_name, params, prob, corder)
    csr = mat.tocsr()
    np.testing.assert_array_equal(np.asarray(csr.todense()), np.asarray(mat.todense()))
    # Rows are column-sorted.
    for r in range(SHAPE[0]):
        row = np.asarray(csr.indices[int(csr.indptr[r]):int(csr.indptr[r + 1])])
        assert np.all(np.diff(row) > 0)


@pytest.mark.parametrize('prob', [0.25, 0.3])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('cls_name, params', CASES, ids=CASE_IDS)
def test_closed_over_matrix_under_jit_and_vmap_matches_eager(cls_name, params, prob, corder):
    mat = _matrix(cls_name, params, prob, corder)
    y = jnp.linspace(0.5, 2., SHAPE[0])
    x = jnp.linspace(-1., 1., SHAPE[1])

    _assert_same_csr(jax.jit(lambda: mat.tocsr())(), mat.tocsr())
    np.testing.assert_array_equal(jax.jit(lambda y_: mat.dt2t(y_, None))(y), mat.dt2t(y, None))
    np.testing.assert_array_equal(
        jax.jit(lambda x_: mat.dt2t_transposed(x_, None))(x), mat.dt2t_transposed(x, None)
    )
    ys = jnp.stack([y, 2. * y])
    np.testing.assert_array_equal(
        jax.vmap(lambda y_: mat.dt2t(y_, None))(ys),
        jnp.stack([mat.dt2t(y_, None) for y_ in ys]),
    )


@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('cls_name, params', CASES, ids=CASE_IDS)
def test_vmap_over_weights_matches_per_weight_calls(cls_name, params, corder):
    mat = _matrix(cls_name, params, 0.3, corder)
    batch = jax.tree.map(lambda p: jnp.asarray([p, 2. * p]), mat.data)
    got = jax.vmap(lambda d: mat.with_data(d).tocsr().data)(batch)
    for i in range(2):
        expected = mat.with_data(jax.tree.map(lambda b: b[i], batch)).tocsr().data
        np.testing.assert_array_equal(got[i], expected)


@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('cls_name, params', CASES, ids=CASE_IDS)
def test_weight_gradients_agree_across_jit_and_grad_orderings(cls_name, params, corder):
    mat = _matrix(cls_name, params, 0.3, corder)
    y = jnp.linspace(0.5, 2., SHAPE[0])
    loss = lambda d: mat.with_data(d).dt2t(y, None).sum()
    eager = jax.grad(loss)(mat.data)
    for g in (jax.jit(jax.grad(loss))(mat.data), jax.grad(jax.jit(loss))(mat.data)):
        for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(eager)):
            np.testing.assert_allclose(a, b, rtol=1e-6)


@pytest.mark.parametrize('to_csr, params', [(jitn_to_csr, (1.5, 0.3)), (jitu_to_csr, (0.5, 1.5))],
                         ids=['normal', 'uniform'])
@pytest.mark.parametrize('corder', [True, False])
def test_to_csr_data_passes_check_grads_in_reverse_mode(to_csr, params, corder):
    from jax.test_util import check_grads

    with jax_x64_enabled():
        f = lambda a, b: to_csr(a, b, 0.3, SEED, shape=SHAPE, corder=corder).data
        check_grads(f, tuple(jnp.asarray(p) for p in params), order=1, modes=['rev'])
        check_grads(jax.jit(f), tuple(jnp.asarray(p) for p in params), order=1, modes=['rev'])


@pytest.mark.parametrize('prob', [0.25, 0.3])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('cls_name, params', CASES, ids=CASE_IDS)
def test_prepared_matrix_passed_into_jit_matches_eager(cls_name, params, prob, corder):
    mat = _matrix(cls_name, params, prob, corder)
    prepared = mat.prepare_structure()
    y = jnp.linspace(0.5, 2., SHAPE[0])
    x = jnp.linspace(-1., 1., SHAPE[1])

    _assert_same_csr(jax.jit(lambda m: m.tocsr())(prepared), mat.tocsr())
    np.testing.assert_array_equal(jax.jit(lambda m, y_: m.dt2t(y_, None))(prepared, y), mat.dt2t(y, None))
    np.testing.assert_array_equal(
        jax.jit(lambda m, x_: m.dt2t_transposed(x_, None))(prepared, x), mat.dt2t_transposed(x, None)
    )
    # The matrix leaves (prob and seed included) are traced inside the jit above;
    # gradients flow through the prepared matrix too.
    loss = lambda m: m.dt2t(y, None).sum()
    g_jit = jax.jit(jax.grad(lambda d: loss(prepared.with_data(d))))(prepared.data)
    g_eager = jax.grad(lambda d: loss(mat.with_data(d)))(mat.data)
    for a, b in zip(jax.tree.leaves(g_jit), jax.tree.leaves(g_eager)):
        np.testing.assert_allclose(a, b, rtol=1e-6)


@pytest.mark.parametrize('cls_name, params', CASES, ids=CASE_IDS)
def test_unprepared_matrix_passed_into_jit_raises_a_clear_error(cls_name, params):
    mat = _matrix(cls_name, params, 0.3, True)
    with pytest.raises(ValueError, match='prepare_structure'):
        jax.jit(lambda m: m.tocsr().data)(mat)
    with pytest.raises(ValueError, match='prepare_structure'):
        jax.jit(lambda m: m.dt2t(jnp.ones(SHAPE[0]), None))(mat)


@pytest.mark.parametrize('cls_name, params', CASES, ids=CASE_IDS)
def test_structure_survives_weight_operations_and_drops_on_connectivity_changes(cls_name, params):
    prepared = _matrix(cls_name, params, 0.3, True).prepare_structure()
    structure = prepared.structure
    assert isinstance(structure, JITCStructure)
    for kept in (prepared.with_data(prepared.data), 2. * prepared, prepared * 3., -prepared,
                 prepared._new_mat(*params)):
        assert kept.structure is structure
    assert prepared.T.structure is None
    assert prepared._new_mat(*params, seed=SEED + 1).structure is None
    assert prepared._new_mat(*params, prob=0.5).structure is None
    # The structure travels as pytree leaves and survives a round trip.
    leaves, treedef = jax.tree.flatten(prepared)
    assert len(leaves) == len(jax.tree.leaves(_matrix(cls_name, params, 0.3, True))) + 4
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.structure.nnz == structure.nnz


def test_prepare_structure_fixes_a_missing_seed():
    mat = brainevent.JITCNormalR(1.5, 0.3, prob=0.3, seed=None, shape=SHAPE)
    prepared = mat.prepare_structure()
    assert prepared.seed is not None
    y = jnp.ones(SHAPE[0])
    np.testing.assert_array_equal(prepared.dt2t(y, None), jax.jit(lambda m: m.dt2t(y, None))(prepared))


def test_zero_probability_and_empty_shapes_prepare_an_empty_structure():
    prepared = brainevent.JITCScalarR(1.5, prob=0., seed=SEED, shape=SHAPE).prepare_structure()
    assert prepared.structure.nnz == 0
    out = jax.jit(lambda m: m.dt2t(jnp.ones(SHAPE[0]), None))(prepared)
    assert out.shape == (0,)
    empty = jits_structure(0.3, SEED, shape=(0, 5), corder=True)
    assert empty.nnz == 0 and empty.indptr.shape == (1,)


def test_to_csr_rejects_a_structure_of_another_shape():
    structure = jits_structure(0.3, SEED, shape=(5, 4), corder=True)
    with pytest.raises(ValueError, match='indptr'):
        jits_to_csr(1.5, 0.3, SEED, shape=SHAPE, corder=True, structure=structure)


def test_prepared_structure_is_a_program_input_not_a_constant():
    """Closing over a matrix embeds its connectivity; passing a prepared one does not."""
    shape = (2000, 1000)
    mat = brainevent.JITCNormalR(1.5, 0.3, prob=0.1, seed=SEED, shape=shape)
    prepared = mat.prepare_structure()
    y = jnp.ones(shape[0])
    as_argument = len(jax.jit(lambda m, y_: m.dt2t(y_, None)).lower(prepared, y).as_text())
    closed_over = len(jax.jit(lambda y_: mat.dt2t(y_, None)).lower(y).as_text())
    assert prepared.structure.nnz > 100_000
    assert as_argument < 50_000
    assert closed_over > 10 * as_argument
