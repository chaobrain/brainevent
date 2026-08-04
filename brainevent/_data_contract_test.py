# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""Tests for the common-API contract on :class:`brainevent.DataRepresentation`.

Covers (1) that every concrete data representation overrides or deliberately
refuses each contract method (no silent inheritance of a bare base stub),
(2) the deliberate JIT-connectivity refusals, and (3) the conversion
round-trips (``tocsr`` / ``tocsc`` / ``tocoo`` / ``fromdense``) across the
compressed-sparse, fixed-num-connection, and JIT-connectivity families.
"""

import inspect

import jax
import jax.numpy as jnp
import pytest

import brainunit as u

import brainevent as be
from brainevent import BrainEventError, UnsupportedOperationError
from brainevent._data import DataRepresentation

SPARSE_BASE = u.sparse.SparseMatrix

# Concrete subclasses of DataRepresentation (the in-scope families).
CONCRETE_CLASSES = [
    be.Dense,
    be.CSR, be.CSC,
    be.FixedNumPerPre, be.FixedNumPerPost,
    be.JITCScalarR, be.JITCScalarC,
    be.JITCNormalR, be.JITCNormalC,
    be.JITCUniformR, be.JITCUniformC,
]

# The common-API contract surface. ``with_data`` / ``transpose`` / ``todense``
# are declared by the saiunit base; the rest by DataRepresentation.
CONTRACT_METHODS = [
    'todense', 'fromdense', 'tocoo', 'tocsr', 'tocsc',
    'dt2t', 'dt2t_transposed',
    'update_on_pre', 'update_on_post',
    'with_data', 'transpose',
]

# (class, constructor-data) for each concrete JIT-connectivity family.
JITC_INSTANCES = [
    (be.JITCScalarR, (1.5, 0.2, 42)),
    (be.JITCScalarC, (1.5, 0.2, 42)),
    (be.JITCNormalR, (0.0, 1.0, 0.2, 42)),
    (be.JITCNormalC, (0.0, 1.0, 0.2, 42)),
    (be.JITCUniformR, (0.0, 1.0, 0.2, 42)),
    (be.JITCUniformC, (0.0, 1.0, 0.2, 42)),
]
JITC_IDS = [c.__name__ for c, _ in JITC_INSTANCES]

# Every JITC family materializes mode-dependently: the mv (32-lane) and mm
# (4-thread AW-T4) light kernels draw different matrices, so bare
# ``todense()/tocsr()/tocsc()/tocoo()`` raise and callers must go through the
# ``mat.mv`` / ``mat.mm`` views.  Their CSR conversion is CUDA-only (the
# per-family ``*_csr_count`` primitives register only a CUDA backend).
def _csr_count_backends(cls):
    """CSR-materialization backends available for ``cls`` on the current platform.

    Returns an empty tuple when the family's ``csr_count`` primitive has no
    backend for the active platform (e.g. CPU), so callers can skip the
    CUDA-only conversion assertions.
    """
    try:
        if cls in (be.JITCScalarR, be.JITCScalarC):
            from brainevent._jit_scalar.csr import jits_csr_count_p as count_p
        elif cls in (be.JITCNormalR, be.JITCNormalC):
            from brainevent._jit_normal.csr import jitn_csr_count_p as count_p
        elif cls in (be.JITCUniformR, be.JITCUniformC):
            from brainevent._jit_uniform.csr import jitu_csr_count_p as count_p
        else:  # pragma: no cover - defensive
            return ()
        return tuple(count_p.available_backends(jax.default_backend()))
    except Exception:  # pragma: no cover - defensive: import/registration failure
        return ()

_DENSE = jnp.array([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]])


def _defining_class(cls, method):
    """Return the first class in ``cls.__mro__`` that defines ``method``."""
    for klass in cls.__mro__:
        if method in vars(klass):
            return klass
    return None


# --------------------------------------------------------------------------- #
# Contract coverage
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('method', CONTRACT_METHODS)
@pytest.mark.parametrize('cls', CONCRETE_CLASSES, ids=[c.__name__ for c in CONCRETE_CLASSES])
def test_contract_method_is_overridden_or_refused(cls, method):
    # Every contract method must resolve to a definition inside brainevent's
    # hierarchy -- never the bare saiunit base stub nor the bare
    # DataRepresentation stub. A deliberate refusal counts as an override.
    defn = _defining_class(cls, method)
    assert defn is not None, f'{cls.__name__} is missing contract method {method!r}'
    assert defn is not SPARSE_BASE, (
        f'{cls.__name__}.{method} silently inherits the saiunit SparseMatrix stub'
    )
    assert defn is not DataRepresentation, (
        f'{cls.__name__}.{method} silently inherits the DataRepresentation stub'
    )


def test_data_representation_declares_dt2t_contract():
    assert 'dt2t' in vars(DataRepresentation)


def test_data_representation_declares_yw_to_w_deprecated_aliases():
    assert 'yw_to_w' in vars(DataRepresentation)
    assert 'yw_to_w_transposed' in vars(DataRepresentation)


@pytest.mark.parametrize('cls', CONCRETE_CLASSES, ids=[c.__name__ for c in CONCRETE_CLASSES])
def test_yw_to_w_signatures_align_dt2t_contract(cls):
    for alias, canonical in (('yw_to_w', 'dt2t'), ('yw_to_w_transposed', 'dt2t_transposed')):
        sig = inspect.signature(getattr(cls, alias))
        canonical_sig = inspect.signature(getattr(cls, canonical))
        assert list(sig.parameters) == list(canonical_sig.parameters)


def test_yw_to_w_warns_and_delegates_to_dt2t():
    csr = be.CSR.fromdense(_DENSE)
    y = jnp.ones(csr.shape[0])
    w = csr.data
    with pytest.warns(DeprecationWarning, match='yw_to_w is deprecated'):
        out = csr.yw_to_w(y, w)
    assert jnp.allclose(out, csr.dt2t(y, w))


def test_yw_to_w_transposed_warns_and_delegates_to_dt2t_transposed():
    csr = be.CSR.fromdense(_DENSE)
    y = jnp.ones(csr.shape[1])
    w = csr.data
    with pytest.warns(DeprecationWarning, match='yw_to_w_transposed is deprecated'):
        out = csr.yw_to_w_transposed(y, w)
    assert jnp.allclose(out, csr.dt2t_transposed(y, w))


@pytest.mark.parametrize('method', ['dt2t', 'dt2t_transposed'])
@pytest.mark.parametrize('cls', CONCRETE_CLASSES, ids=[c.__name__ for c in CONCRETE_CLASSES])
def test_dt2t_signatures_align_data_representation_contract(cls, method):
    sig = inspect.signature(getattr(cls, method))
    base_sig = inspect.signature(getattr(DataRepresentation, method))
    assert list(sig.parameters) == list(base_sig.parameters)
    assert sig.parameters['y_dim_arr'].annotation == base_sig.parameters['y_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].annotation == base_sig.parameters['w_dim_arr'].annotation
    assert sig.parameters['w_dim_arr'].default is inspect._empty
    assert sig.return_annotation == base_sig.return_annotation


def test_unsupported_operation_error_is_brainevent_error():
    assert issubclass(UnsupportedOperationError, BrainEventError)


# --------------------------------------------------------------------------- #
# JIT-connectivity deliberate refusals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('cls,data', JITC_INSTANCES, ids=JITC_IDS)
def test_jitc_refuses_plastic_update_protocols(cls, data):
    m = cls(data, shape=(16, 16))
    y = jnp.ones(16)
    with pytest.raises(UnsupportedOperationError):
        m.update_on_pre(y, y)
    with pytest.raises(UnsupportedOperationError):
        m.update_on_post(y, y)


@pytest.mark.parametrize('cls,data', JITC_INSTANCES, ids=JITC_IDS)
def test_jitc_refuses_fromdense(cls, data):
    with pytest.raises(UnsupportedOperationError):
        cls.fromdense(jnp.ones((16, 16)))


# --------------------------------------------------------------------------- #
# JIT-connectivity conversions (delegate through tocsr)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('cls,data', JITC_INSTANCES, ids=JITC_IDS)
def test_jitc_conversions_agree_with_todense(cls, data):
    m = cls(data, shape=(16, 16))
    # Bare materialization is ambiguous for every JITC family and must raise; the
    # mv/mm views resolve the mode. tocsr/tocsc/tocoo are CUDA-only.
    with pytest.raises(NotImplementedError):
        m.todense()
    with pytest.raises(NotImplementedError):
        m.tocsr()
    dense = m.mv.todense()
    assert dense.shape == m.shape
    if not _csr_count_backends(cls):
        pytest.skip('JITC CSR conversion is CUDA-only')
    assert jnp.allclose(m.mv.tocsr().todense(), dense)
    assert jnp.allclose(m.mv.tocsc().todense(), dense)
    assert jnp.allclose(m.mv.tocoo().todense(), dense)
    assert m.mv.tocsc().shape == m.shape
    assert m.mv.tocoo().shape == m.shape


# --------------------------------------------------------------------------- #
# Compressed-sparse conversions
# --------------------------------------------------------------------------- #

def test_csr_conversions_roundtrip():
    csr = be.CSR.fromdense(_DENSE)
    assert csr.tocsr() is csr
    assert jnp.allclose(csr.tocsc().todense(), _DENSE)
    assert jnp.allclose(csr.tocoo().todense(), _DENSE)


def test_csc_conversions_roundtrip():
    csc = be.CSC.fromdense(_DENSE)
    assert csc.tocsc() is csc
    assert jnp.allclose(csc.tocsr().todense(), _DENSE)
    assert jnp.allclose(csc.tocoo().todense(), _DENSE)


def test_tocsc_preserves_shape_unlike_transpose():
    # tocsc re-encodes the *same* logical matrix (shape unchanged); transpose
    # swaps the shape. A non-square matrix makes the distinction unambiguous.
    dense = jnp.array([[1., 2., 0., 0.], [0., 0., 3., 0.], [0., 4., 0., 5.]])  # (3, 4)
    csr = be.CSR.fromdense(dense)
    assert csr.tocsc().shape == (3, 4)
    assert csr.transpose().shape == (4, 3)
    assert jnp.allclose(csr.tocsc().todense(), dense)


def test_csr_tocoo_homogeneous_value_broadcast():
    # A size-1 (shared) value must broadcast to one entry per stored element.
    csr = be.CSR((jnp.array([2.0]), jnp.array([0, 2, 1]), jnp.array([0, 2, 3])), shape=(2, 3))
    coo = csr.tocoo()
    assert coo.row.size == 3
    assert jnp.allclose(coo.todense(), csr.todense())


# --------------------------------------------------------------------------- #
# Fixed-num-connection conversions and fromdense
# --------------------------------------------------------------------------- #

def test_fcn_pre_fromdense_uniform_roundtrip():
    dense = jnp.array([[1., 2., 0.], [0., 3., 4.], [5., 0., 6.]])  # 2 conns / pre
    pre = be.FixedNumPerPre.fromdense(dense)
    assert pre.num_conn == 2
    assert jnp.allclose(pre.todense(), dense)
    assert jnp.allclose(pre.tocsr().todense(), dense)
    assert jnp.allclose(pre.tocsc().todense(), dense)
    assert jnp.allclose(pre.tocoo().todense(), dense)


def test_fcn_post_fromdense_uniform_roundtrip():
    dense = jnp.array([[1., 0., 5.], [2., 3., 0.], [0., 4., 6.]])  # 2 conns / post
    post = be.FixedNumPerPost.fromdense(dense)
    assert post.num_conn == 2
    assert jnp.allclose(post.todense(), dense)
    assert jnp.allclose(post.tocsr().todense(), dense)
    assert jnp.allclose(post.tocoo().todense(), dense)


def test_fcn_fromdense_irregular_requires_num_conn():
    irr = jnp.array([[1., 2., 3.], [0., 4., 0.], [5., 0., 6.]])  # 3, 1, 2 nnz
    with pytest.raises(ValueError):
        be.FixedNumPerPre.fromdense(irr)


def test_fcn_fromdense_padding_roundtrip():
    irr = jnp.array([[1., 2., 3.], [0., 4., 0.], [5., 0., 6.]])
    pre = be.FixedNumPerPre.fromdense(irr, num_conn=3)
    assert pre.num_conn == 3
    # zero-weight sentinel padding does not change the dense matrix.
    assert jnp.allclose(pre.todense(), irr)


def test_fcn_fromdense_overflow_raises():
    irr = jnp.array([[1., 2., 3.], [0., 4., 0.], [5., 0., 6.]])
    with pytest.raises(ValueError):
        be.FixedNumPerPre.fromdense(irr, num_conn=2)


def test_fcn_fromdense_preserves_units():
    dense = jnp.array([[1., 2., 0.], [0., 3., 4.], [5., 0., 6.]]) * u.mV
    pre = be.FixedNumPerPre.fromdense(dense)
    assert u.get_unit(pre.todense()) == u.get_unit(dense)
    assert u.math.allclose(pre.todense(), dense)


def test_fcn_fromdense_rejects_non_2d():
    with pytest.raises(ValueError):
        be.FixedNumPerPre.fromdense(jnp.ones((3,)))


@pytest.mark.parametrize('name', ['to_csr', 'to_csc', 'to_dense'])
def test_fcn_deprecated_aliases_removed(name):
    pre = be.FixedNumPerPre.fromdense(_DENSE, num_conn=2)
    assert not hasattr(pre, name)
