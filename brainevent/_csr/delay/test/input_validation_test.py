"""Keep prepared-input validation consistent across backends and consumed outputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.delay import csr_delay_deliver_layout_p, delay as operator
from brainevent._csr.delay.test._helpers import BACKENDS


def _operands():
    # One long row needs a nonempty task queue, so consuming only a workspace
    # array cannot disappear merely because its shape is zero.
    queue = jnp.zeros(1, jnp.int32)
    return [
        jnp.ones((2, 2), jnp.float32), jnp.array([True]),
        jnp.array([.5], jnp.float32), jnp.zeros(129, jnp.int32),
        jnp.array([0, 129], jnp.int32), jnp.array([[129]], jnp.int32),
        jnp.array([2], jnp.int32), jnp.array([0], jnp.int32),
        jnp.ones(1, jnp.int32), queue, queue, queue, jnp.zeros(2, jnp.int32),
    ]


def _project(args, *, backend, entry, check_status, weighted=False):
    if entry == 'call':
        return csr_delay_deliver_layout_p.call(
            *args, backend=backend, weighted=weighted, check_status=check_status)
    outputs = tuple(jax.core.ShapedArray(value.shape, value.dtype)
                    for value in (args[0], *args[9:]))
    return csr_delay_deliver_layout_p.primitive.bind(
        *args, outs=outputs, backend=backend, weighted=weighted, check_status=check_status)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('entry', ['call', 'bind'])
@pytest.mark.parametrize('output', ['ring', 'status', 'workspace'])
@pytest.mark.parametrize('check_status', [True, False])
def test_dynamic_head_checked_for_each_consumed_output(backend, entry, output, check_status):
    args = _operands()
    position = {'ring': 0, 'status': -1, 'workspace': 1}[output]
    run = jax.jit(lambda *xs: _project(
        xs, backend=backend, entry=entry, check_status=check_status)[position])
    valid = jax.block_until_ready(run(*args))
    if output == 'ring':
        np.testing.assert_array_equal(valid, [[1, 1], [65.5, 1]])
    elif output == 'status':
        np.testing.assert_array_equal(valid[1], 0)
    else:
        assert valid.shape == (1,)

    invalid = list(args)
    invalid[7] = jnp.array([2], jnp.int32)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(*invalid))
    np.testing.assert_array_equal(jax.block_until_ready(run(*args)), valid)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('field,output', [
    ('indptr_start', 'status'), ('indptr_end', 'workspace'),
    ('counts', 'status'), ('tasks', 'workspace'), ('indices', 'ring'),
    ('offset_range', 'status'), ('offset_alignment', 'workspace'), ('capacity', 'ring'),
])
def test_metadata_validation_and_unchecked_status(backend, field, output):
    args = _operands()
    if field == 'counts':
        args[5] = jnp.array([[130]], jnp.int32)
    elif field == 'tasks':
        args[8] = jnp.zeros(1, jnp.int32)
    elif field == 'capacity':
        args[9:12] = [value[:0] for value in args[9:12]]
    else:
        # Metadata validity is independent of activity. Inactive rows also
        # keep the legacy CUDA kernel from dereferencing malformed pointers
        # while this regression test runs against the unfixed implementation.
        args[1] = jnp.array([False])
        if field == 'indptr_start':
            args[4] = jnp.array([1, 129], jnp.int32)
        elif field == 'indptr_end':
            args[4] = jnp.array([0, 130], jnp.int32)
        elif field == 'indices':
            args[3] = args[3].at[0].set(2)
        else:
            args[6] = jnp.array([4 if field == 'offset_range' else 1], jnp.int32)

    unchecked = jax.jit(lambda *xs: _project(
        xs, backend=backend, entry='bind', check_status=False)[-1])
    status = jax.block_until_ready(unchecked(*args))
    np.testing.assert_array_equal(status[1], 1 if field == 'capacity' else 3)

    position = {'ring': 0, 'status': -1, 'workspace': 1}[output]
    checked = jax.jit(lambda *xs: _project(
        xs, backend=backend, entry='bind', check_status=True)[position])
    with pytest.raises(Exception, match='capacity|metadata|task'):
        jax.block_until_ready(checked(*args))
    jax.block_until_ready(checked(*_operands()))


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('check_status', [True, False])
def test_weighted_dynamic_head_checked_when_only_status_is_used(backend, check_status):
    args = _operands()
    args[1] = jnp.array([-2], jnp.float32)
    args.insert(-1, jnp.zeros(1, jnp.float32))
    run = jax.jit(lambda head: _project(
        (*args[:7], head, *args[8:]), backend=backend, entry='bind',
        check_status=check_status, weighted=True)[-1])
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(jnp.array([-1], jnp.int32)))
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('check_status', [True, False])
def test_empty_ring_dynamic_head_checked_when_only_status_is_used(backend, check_status):
    args = _operands()
    args[0] = jnp.zeros((2, 0), jnp.float32)
    args[3] = jnp.zeros(0, jnp.int32)
    args[4] = jnp.zeros(2, jnp.int32)
    args[5] = jnp.zeros((1, 1), jnp.int32)
    args[6] = jnp.zeros(1, jnp.int32)
    args[8] = jnp.zeros(1, jnp.int32)
    run = jax.jit(lambda head: _project(
        (*args[:7], head, *args[8:]), backend=backend, entry='call',
        check_status=check_status)[-1])
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(jnp.array([2], jnp.int32)))
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
def test_unchecked_metadata_error_prevents_partial_ring_writes(backend, weighted):
    args = _operands()
    args[1] = jnp.array([-2, .5], jnp.float32)
    args[3] = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.ones(129, jnp.int32)))
    args[4] = jnp.array([0, 1, 130], jnp.int32)
    args[5] = jnp.array([[1], [130]], jnp.int32)
    args[8] = jnp.array([0, 1], jnp.int32)
    if weighted:
        args.insert(-1, jnp.zeros(1, jnp.float32))

    def project(*xs):
        result = _project(xs, backend=backend, entry='bind', check_status=False, weighted=weighted)
        return result[0], result[-1]

    run = jax.jit(project)
    ring, status = jax.block_until_ready(run(*args))
    np.testing.assert_array_equal(status[1], 3)
    # The first row is valid and active. It must not write before validation
    # rejects the second row's mismatched segment length.
    np.testing.assert_array_equal(ring, args[0])
    args[5] = jnp.array([[1], [129]], jnp.int32)
    ring, status = jax.block_until_ready(run(*args))
    np.testing.assert_array_equal(status[1], 0)
    np.testing.assert_array_equal(ring, [[1, 1], [0, 33.25] if weighted else [1.5, 65.5]])
    np.testing.assert_array_equal(args[0], 1)


@pytest.mark.parametrize('backend', BACKENDS[1:])
def test_jitted_vmapped_head_checks_each_batch_element(backend):
    args = _operands()
    run = jax.jit(jax.vmap(lambda head: _project(
        (*args[:7], head, *args[8:]), backend=backend, entry='call', check_status=True)[-1]))
    valid = jnp.array([[0], [1]], jnp.int32)
    np.testing.assert_array_equal(jax.block_until_ready(run(valid))[:, 1], 0)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(valid.at[1, 0].set(2)))
    np.testing.assert_array_equal(jax.block_until_ready(run(valid))[:, 1], 0)


@pytest.mark.parametrize('field', ['counts', 'indptr', 'many_counts'])
def test_input_metadata_overflow_is_rejected_without_x64(field):
    with jax.enable_x64(False):
        args = _operands()
        args[3] = jnp.zeros(1, jnp.int32)
        if field == 'counts':
            args[4] = jnp.array([0, 1], jnp.int32)
            args[5] = jnp.array([[2**31 - 1, 2**31 - 1, 3]], jnp.int32)
            args[6] = jnp.zeros(3, jnp.int32)
            args[8] = jnp.zeros(1, jnp.int32)
        elif field == 'indptr':
            args[1] = jnp.zeros(3, jnp.bool_)
            args[4] = jnp.array([0, 1431655765, -1431655766, 1], jnp.int32)
            args[5] = jnp.diff(args[4])[:, None]
            args[6] = jnp.zeros(1, jnp.int32)
            args[8] = args[5][:, 0] // 4096 + (args[5][:, 0] % 4096 != 0)
        else:
            args[1] = jnp.array([False])
            args[3] = jnp.zeros(100000, jnp.int32)
            args[4] = jnp.array([0, 100000], jnp.int32)
            args[5] = jnp.concatenate((jnp.full(42950, 100000, jnp.int32),
                                       jnp.array([67296], jnp.int32)))[None]
            args[6] = jnp.zeros(args[5].shape[1], jnp.int32)
            args[8] = jnp.sum(args[5] // 4096 + (args[5] % 4096 != 0), axis=1)
        # Probe validation alone: these malformed pointers must never reach
        # a delivery implementation, even while testing the unfixed code.
        code = jax.jit(operator._input_error_code)(*args)
        np.testing.assert_array_equal(code, 3)


@pytest.mark.parametrize('check_status', [True, False])
@pytest.mark.parametrize('output', ['status', 'workspace'])
def test_backend_execution_status_uses_shared_output_check(check_status, output):
    def fake_generator(**params):
        def kernel(*args):
            return (args[0], *(jnp.full_like(value, 7) for value in args[9:-1]),
                    jnp.array([0, 3], jnp.int32))
        return kernel

    checked = operator._validated_kernel(fake_generator)(check_status=check_status)
    position = -1 if output == 'status' else 1
    run = jax.jit(lambda *args: checked(*args)[position])
    if check_status:
        with pytest.raises(Exception, match='metadata'):
            jax.block_until_ready(run(*_operands()))
    else:
        np.testing.assert_array_equal(jax.block_until_ready(run(*_operands())),
                                      [0, 3] if output == 'status' else [7])
