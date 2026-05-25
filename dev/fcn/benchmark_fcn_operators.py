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
"""
Direct FCN operator benchmark with fixed W/indices and reproducible spikes.

This script intentionally does not use argparse. Edit the constants below to
change the benchmark scale, connection count, batch width, seed, number of
runs, or preprocessing modes.

It benchmarks the FCN operator family directly:

    fcnmv / fcnmm
    binary_fcnmv / binary_fcnmm
    bitpack_binary_fcnmv / bitpack_binary_fcnmm
    compact_binary_fcnmv / compact_binary_fcnmm

For bitpack and compact operators, PREPROCESS_MODES controls whether packing
and compaction are measured inside the timed function ("included") or prepared
once outside the timed function ("excluded"). Bitpack FCN-MM also supports
"prepacked", which calls the low-level primitive with precomputed packed spike
matrices so the timed region is as close as possible to the FCN-MM kernel only.
"""

import sys
import time
from collections import namedtuple
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

import brainevent
from brainevent._fcn.bitpack_binary import bitpack_binary_fcnmm_p_call


# ---------------------------------------------------------------------------
# Hard-coded benchmark parameters
# ---------------------------------------------------------------------------

SEED = 42

# COBA EI-style size controls. SCALE follows the COBA EI convention where
# scale=1 means 3200 excitatory + 800 inhibitory neurons. This direct operator
# benchmark keeps a square FCN shape and uses CONN as the fixed per-row
# connection count; probability-based connection controls are intentionally not
# used here.
COBA_EI_N_EXC = 3200
COBA_EI_N_INH = 800
COBA_EI_N = COBA_EI_N_EXC + COBA_EI_N_INH
SCALE = 106
CONN = 2515
N_B = 32

SPIKE_RATE_HZ = 60.23233413696289
SPIKE_DT_MS = 0.1
SPIKE_RATE = SPIKE_RATE_HZ * SPIKE_DT_MS / 1000.0
SPIKE_STEPS = 3
COBA_DURATION_MS = 50.0
COBA_DT_MS = 0.1
# COBA evaluates excitatory and inhibitory connections separately, but this
# direct benchmark uses one combined square matrix with the same total number
# of source rows (n_exc + n_inh). One direct step therefore corresponds to the
# full E/I connection work for one COBA simulation step.
COBA_CONN_UPDATES_PER_STEP = 1

N_WARMUP = 3
N_RUNS = 3
# Timed step control. None means use the COBA-equivalent step count computed
# from COBA_DURATION_MS / COBA_DT_MS. Set an integer such as 32 to run exactly
# that many timed steps.
BENCHMARK_STEPS = 3

PLATFORM = "gpu"
PREPROCESS_MODES = ("included",) #included
PACK_AXIS = 1
DTYPE = jnp.float32

# Set to a tuple such as ("jax_raw", "cuda_raw") to restrict backend testing.
BACKENDS = None
OPERATOR_FILTER = None
TRANSPOSE_FILTER = None
HOMO_FILTER = None

# COBA_EI_binary_fcnmm_CsvOuput.py-style hard-coded sweep controls. Edit these
# lists or pass explicit values to benchmark_conn() below to launch a smaller
# or larger direct-operator run.
scales = [SCALE]

conn_nums = [CONN]
default_batch_sizes = [N_B]

# ncu target. Running this file directly benchmarks only this route, so
# profiler output is focused on the target FCN operator kernel.
MAIN_DATA_TYPE = "binary"
MAIN_KIND = "mm"
MAIN_MODE = "post"
MAIN_HOMO = True

MAIN_MV_LAYOUT = "row_gather"

backends = ["test_colmajor_fullwarp_nocap"]
MAIN_BACKEND = "test_colmajor_fullwarp_nocap"

FixedData = namedtuple(
    "FixedData",
    [
        "shape",
        "weights_homo",
        "weights_hetero",
        "indices",
        "spike_vectors",
        "spike_matrices",
        "float_vectors",
        "float_matrices",
    ],
)

BenchmarkSpec = namedtuple(
    "BenchmarkSpec",
    [
        "operator",
        "kind",
        "backend",
        "preprocess",
        "transpose",
        "homo",
        "fn",
        "step_args",
    ],
)

BenchmarkCase = namedtuple(
    "BenchmarkCase",
    [
        "scale",
        "conn_num",
        "batch_size",
        "backend",
    ],
)

BenchmarkRow = namedtuple(
    "BenchmarkRow",
    [
        "operator",
        "kind",
        "backend",
        "preprocess",
        "transpose",
        "homo",
        "elapsed_s",
        "std_s",
        "min_s",
        "success",
        "error",
    ],
)


def _scaled_size(scale=None):
    if scale is None:
        scale = SCALE
    size = int(COBA_EI_N * scale)
    if size <= 0:
        raise ValueError("SCALE must produce at least one neuron.")
    return size


def _benchmark_dimensions():
    conn = int(CONN)
    n_b = int(N_B)
    if conn <= 0:
        raise ValueError("CONN must be a positive integer.")
    if n_b <= 0:
        raise ValueError("N_B must be a positive integer.")
    size = _scaled_size()
    return size, size, conn, n_b


def make_fixed_data():
    """Return deterministic W/indices/spike inputs from SEED.

    The generated spike series is reproducible for the same seed, while each
    run step uses a different pre-generated vector/matrix.
    """
    n_pre, n_post, conn, n_b = _benchmark_dimensions()
    if SPIKE_STEPS > min(n_pre, n_post):
        raise ValueError("SPIKE_STEPS must be <= SCALE-derived FCN size to guarantee unique spike inputs.")

    rng = np.random.default_rng(SEED)
    shape = (n_pre, n_post)
    indices = jnp.asarray(
        rng.integers(0, n_post, size=(n_pre, conn), dtype=np.int32)
    )
    weights_homo = jnp.ones(1, dtype=DTYPE) if _needs_homo_weights() else None
    weights_hetero = (
        jnp.asarray(
            rng.standard_normal(size=(n_pre, conn)),
            dtype=DTYPE,
        )
        if _needs_hetero_weights()
        else None
    )
    spike_vectors_np = (
        rng.random(size=(SPIKE_STEPS, max(n_pre, n_post))) < SPIKE_RATE
        if _needs_vector_spikes()
        else None
    )
    spike_matrices_np = (
        rng.random(size=(SPIKE_STEPS, max(n_pre, n_post), n_b)) < SPIKE_RATE
        if _needs_matrix_spikes()
        else None
    )
    marker = np.arange(SPIKE_STEPS)
    if spike_vectors_np is not None:
        spike_vectors_np[:, :SPIKE_STEPS] = False
        spike_vectors_np[marker, marker] = True
        spike_vectors = jnp.asarray(spike_vectors_np, dtype=jnp.bool_)
    else:
        spike_vectors = None
    if spike_matrices_np is not None:
        spike_matrices_np[:, :SPIKE_STEPS, 0] = False
        spike_matrices_np[marker, marker, 0] = True
        spike_matrices = jnp.asarray(spike_matrices_np, dtype=jnp.bool_)
    else:
        spike_matrices = None
    # Float FCN uses the same reproducible spike sequence, represented as
    # floating-point activity, so every operator sees the same event pattern.
    float_vectors = spike_vectors.astype(DTYPE) if _needs_float_vectors() else None
    float_matrices = spike_matrices.astype(DTYPE) if _needs_float_matrices() else None
    return FixedData(
        shape=shape,
        weights_homo=weights_homo,
        weights_hetero=weights_hetero,
        indices=indices,
        spike_vectors=spike_vectors,
        spike_matrices=spike_matrices,
        float_vectors=float_vectors,
        float_matrices=float_matrices,
    )


def _weight(data, homo):
    return data.weights_homo if homo else data.weights_hetero


def _source_size(data, transpose):
    n_pre, n_post = data.shape
    return n_pre if transpose else n_post


def _vector(values, data, transpose):
    return values[:_source_size(data, transpose)]


def _matrix(values, data, transpose):
    return values[:_source_size(data, transpose), :]


def _available_backends(primitive, platform):
    available = tuple(primitive.available_backends(platform))
    if BACKENDS is None:
        return available
    requested = tuple(BACKENDS)
    return tuple(backend for backend in requested if backend in available)


def _binary_fcnmm_backend_supported(backend, transpose):
    return True


def _spec_enabled(operator, transpose, homo):
    if OPERATOR_FILTER is not None and operator not in OPERATOR_FILTER:
        return False
    if TRANSPOSE_FILTER is not None and transpose not in TRANSPOSE_FILTER:
        return False
    if HOMO_FILTER is not None and homo not in HOMO_FILTER:
        return False
    return True


def _any_spec_enabled(operators, *, transpose_values=(False, True), homo_values=(True, False)):
    for operator in operators:
        for transpose in transpose_values:
            for homo in homo_values:
                if _spec_enabled(operator, transpose, homo):
                    return True
    return False


def _needs_homo_weights():
    return HOMO_FILTER is None or True in HOMO_FILTER


def _needs_hetero_weights():
    return HOMO_FILTER is None or False in HOMO_FILTER


def _needs_vector_spikes():
    return _any_spec_enabled(
        (
            "fcnmv",
            "binary_fcnmv",
            "bitpack_binary_fcnmv",
            "compact_binary_fcnmv",
        )
    )


def _needs_matrix_spikes():
    return _any_spec_enabled(
        (
            "fcnmm",
            "binary_fcnmm",
            "bitpack_binary_fcnmm",
            "bitpack_binary_fcnmm_p",
            "compact_binary_fcnmm",
        )
    )


def _needs_float_vectors():
    return _any_spec_enabled(("fcnmv",))


def _needs_float_matrices():
    return _any_spec_enabled(("fcnmm",))


def _bitpack_vector_entries(data, transpose):
    entries = []
    for spikes in data.spike_vectors:
        value = _vector(spikes, data, transpose)
        bp = brainevent.BitPackedBinary(value)
        entries.append((bp.packed[0], bp.value))
    return tuple(entries)


def _bitpack_matrix_entries(data, transpose):
    entries = []
    for spikes in data.spike_matrices:
        value = _matrix(spikes, data, transpose)
        bp = brainevent.BitPackedBinary(value)
        entries.append((bp.packed[PACK_AXIS], bp.value))
    return tuple(entries)


def _bitpack_matrix_prepacked_entries(data, transpose):
    return _bitpack_matrix_entries(data, transpose)


def _compact_vector_excluded_entries(data, transpose):
    entries = []
    for spikes in data.spike_vectors:
        value = _vector(spikes, data, transpose)
        cb = brainevent.CompactBinary.from_array(value) if transpose else brainevent.CompactBinary.from_array_light(value)
        entries.append((cb.packed, cb.active_ids, cb.n_active, cb.value))
    return tuple(entries)


def _compact_matrix_excluded_entries(data, transpose):
    entries = []
    for spikes in data.spike_matrices:
        value = _matrix(spikes, data, transpose)
        cb = brainevent.CompactBinary.from_array(value)
        entries.append((cb.packed, cb.active_ids, cb.n_active, cb.value))
    return tuple(entries)


def _make_fcnmv_spec(data, *, transpose, homo):
    weights = _weight(data, homo)

    def run(vector):
        return brainevent.fcnmv(
            weights,
            data.indices,
            vector,
            shape=data.shape,
            transpose=transpose,
        )

    step_args = tuple(
        (_vector(data.float_vectors[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def _make_fcnmm_spec(data, *, transpose, homo):
    weights = _weight(data, homo)

    def run(matrix):
        return brainevent.fcnmm(
            weights,
            data.indices,
            matrix,
            shape=data.shape,
            transpose=transpose,
        )

    step_args = tuple(
        (_matrix(data.float_matrices[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def _make_binary_fcnmv_spec(data, *, backend, transpose, homo):
    weights = _weight(data, homo)

    def run(spikes):
        return brainevent.binary_fcnmv(
            weights,
            data.indices,
            spikes,
            shape=data.shape,
            transpose=transpose,
            backend=backend,
        )

    step_args = tuple(
        (_vector(data.spike_vectors[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def _make_binary_fcnmm_spec(data, *, backend, transpose, homo):
    weights = _weight(data, homo)

    def run(matrix):
        return brainevent.binary_fcnmm(
            weights,
            data.indices,
            matrix,
            shape=data.shape,
            transpose=transpose,
            backend=backend,
        )

    step_args = tuple(
        (_matrix(data.spike_matrices[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def _make_bitpack_fcnmv_excluded_spec(data, *, backend, transpose, homo):
    weights = _weight(data, homo)

    def run(packed, spikes):
        return brainevent.bitpack_binary_fcnmv(
            weights,
            data.indices,
            packed,
            spikes,
            shape=data.shape,
            transpose=transpose,
            backend=backend,
        )

    return run, _bitpack_vector_entries(data, transpose)


def _make_bitpack_fcnmv_included_spec(data, *, backend, transpose, homo):
    weights = _weight(data, homo)

    def run(spikes):
        bp = brainevent.BitPackedBinary(spikes)
        return brainevent.bitpack_binary_fcnmv(
            weights,
            data.indices,
            bp.packed[0],
            bp.value,
            shape=data.shape,
            transpose=transpose,
            backend=backend,
        )

    step_args = tuple(
        (_vector(data.spike_vectors[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def _make_bitpack_fcnmm_excluded_spec(data, *, backend, transpose, homo):
    weights = _weight(data, homo)

    def run(packed, matrix):
        return brainevent.bitpack_binary_fcnmm(
            weights,
            data.indices,
            packed,
            matrix,
            shape=data.shape,
            transpose=transpose,
            pack_axis=PACK_AXIS,
            backend=backend,
        )

    return run, _bitpack_matrix_entries(data, transpose)


def _make_bitpack_fcnmm_included_spec(data, *, backend, transpose, homo):
    weights = _weight(data, homo)

    def run(matrix):
        bp = brainevent.BitPackedBinary(matrix)
        return brainevent.bitpack_binary_fcnmm(
            weights,
            data.indices,
            bp.packed[PACK_AXIS],
            bp.value,
            shape=data.shape,
            transpose=transpose,
            pack_axis=PACK_AXIS,
            backend=backend,
        )

    step_args = tuple(
        (_matrix(data.spike_matrices[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def _make_bitpack_fcnmm_prepacked_spec(data, *, backend, transpose, homo):
    weights = _weight(data, homo)

    def run(packed, matrix):
        return bitpack_binary_fcnmm_p_call(
            weights,
            data.indices,
            packed,
            matrix,
            shape=data.shape,
            transpose=transpose,
            pack_axis=PACK_AXIS,
            backend=backend,
        )[0]

    return run, _bitpack_matrix_prepacked_entries(data, transpose)


def _make_compact_fcnmv_excluded_spec(data, *, transpose, homo):
    weights = _weight(data, homo)

    def run(packed, active_ids, n_active, spikes):
        return brainevent.compact_binary_fcnmv(
            weights,
            data.indices,
            packed,
            active_ids,
            n_active,
            spikes,
            shape=data.shape,
            transpose=transpose,
        )

    return run, _compact_vector_excluded_entries(data, transpose)


def _make_compact_fcnmv_included_spec(data, *, transpose, homo):
    weights = _weight(data, homo)

    def run(spikes):
        cb = brainevent.CompactBinary.from_array(spikes) if transpose else brainevent.CompactBinary.from_array_light(spikes)
        return brainevent.compact_binary_fcnmv(
            weights,
            data.indices,
            cb.packed,
            cb.active_ids,
            cb.n_active,
            cb.value,
            shape=data.shape,
            transpose=transpose,
        )

    step_args = tuple(
        (_vector(data.spike_vectors[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def _make_compact_fcnmm_excluded_spec(data, *, transpose, homo):
    weights = _weight(data, homo)

    def run(packed, active_ids, n_active, matrix):
        return brainevent.compact_binary_fcnmm(
            weights,
            data.indices,
            packed,
            active_ids,
            n_active,
            matrix,
            shape=data.shape,
            transpose=transpose,
            pack_axis=PACK_AXIS,
        )

    return run, _compact_matrix_excluded_entries(data, transpose)


def _make_compact_fcnmm_included_spec(data, *, transpose, homo):
    weights = _weight(data, homo)

    def run(matrix):
        cb = brainevent.CompactBinary.from_array(matrix)
        return brainevent.compact_binary_fcnmm(
            weights,
            data.indices,
            cb.packed,
            cb.active_ids,
            cb.n_active,
            cb.value,
            shape=data.shape,
            transpose=transpose,
            pack_axis=PACK_AXIS,
        )

    step_args = tuple(
        (_matrix(data.spike_matrices[i], data, transpose),)
        for i in range(SPIKE_STEPS)
    )
    return run, step_args


def build_benchmark_specs(data, platform=PLATFORM):
    specs = []
    for transpose in (False, True):
        for homo in (True, False):
            if _spec_enabled("fcnmv", transpose, homo):
                fn, step_args = _make_fcnmv_spec(data, transpose=transpose, homo=homo)
                specs.append(BenchmarkSpec("fcnmv", "mv", "jax", "none", transpose, homo, fn, step_args))
            if _spec_enabled("fcnmm", transpose, homo):
                fn, step_args = _make_fcnmm_spec(data, transpose=transpose, homo=homo)
                specs.append(BenchmarkSpec("fcnmm", "mm", "jax", "none", transpose, homo, fn, step_args))

            if _spec_enabled("binary_fcnmv", transpose, homo):
                for backend in _available_backends(brainevent.binary_fcnmv_p, platform):
                    fn, step_args = _make_binary_fcnmv_spec(data, backend=backend, transpose=transpose, homo=homo)
                    specs.append(BenchmarkSpec("binary_fcnmv", "mv", backend, "none", transpose, homo, fn, step_args))
            if _spec_enabled("binary_fcnmm", transpose, homo):
                for backend in _available_backends(brainevent.binary_fcnmm_p, platform):
                    if not _binary_fcnmm_backend_supported(backend, transpose):
                        continue
                    fn, step_args = _make_binary_fcnmm_spec(data, backend=backend, transpose=transpose, homo=homo)
                    specs.append(BenchmarkSpec("binary_fcnmm", "mm", backend, "none", transpose, homo, fn, step_args))

            if _spec_enabled("bitpack_binary_fcnmv", transpose, homo):
                for backend in _available_backends(brainevent.bitpack_binary_fcnmv_p, platform):
                    if "excluded" in PREPROCESS_MODES:
                        fn, step_args = _make_bitpack_fcnmv_excluded_spec(data, backend=backend, transpose=transpose, homo=homo)
                        specs.append(BenchmarkSpec("bitpack_binary_fcnmv", "mv", backend, "excluded", transpose, homo, fn, step_args))
                    if "included" in PREPROCESS_MODES:
                        fn, step_args = _make_bitpack_fcnmv_included_spec(data, backend=backend, transpose=transpose, homo=homo)
                        specs.append(BenchmarkSpec("bitpack_binary_fcnmv", "mv", backend, "included", transpose, homo, fn, step_args))
            if _spec_enabled("bitpack_binary_fcnmm", transpose, homo):
                for backend in _available_backends(brainevent.bitpack_binary_fcnmm_p, platform):
                    if "excluded" in PREPROCESS_MODES:
                        fn, step_args = _make_bitpack_fcnmm_excluded_spec(data, backend=backend, transpose=transpose, homo=homo)
                        specs.append(BenchmarkSpec("bitpack_binary_fcnmm", "mm", backend, "excluded", transpose, homo, fn, step_args))
                    if "included" in PREPROCESS_MODES:
                        fn, step_args = _make_bitpack_fcnmm_included_spec(data, backend=backend, transpose=transpose, homo=homo)
                        specs.append(BenchmarkSpec("bitpack_binary_fcnmm", "mm", backend, "included", transpose, homo, fn, step_args))
            if _spec_enabled("bitpack_binary_fcnmm_p", transpose, homo):
                for backend in _available_backends(brainevent.bitpack_binary_fcnmm_p, platform):
                    if "prepacked" in PREPROCESS_MODES:
                        fn, step_args = _make_bitpack_fcnmm_prepacked_spec(data, backend=backend, transpose=transpose, homo=homo)
                        specs.append(BenchmarkSpec("bitpack_binary_fcnmm_p", "mm", backend, "prepacked", transpose, homo, fn, step_args))

            if _spec_enabled("compact_binary_fcnmv", transpose, homo) and _available_backends(brainevent.compact_binary_fcnmv_p, platform):
                if "excluded" in PREPROCESS_MODES:
                    fn, step_args = _make_compact_fcnmv_excluded_spec(data, transpose=transpose, homo=homo)
                    specs.append(BenchmarkSpec("compact_binary_fcnmv", "mv", "default", "excluded", transpose, homo, fn, step_args))
                if "included" in PREPROCESS_MODES:
                    fn, step_args = _make_compact_fcnmv_included_spec(data, transpose=transpose, homo=homo)
                    specs.append(BenchmarkSpec("compact_binary_fcnmv", "mv", "default", "included", transpose, homo, fn, step_args))
            if _spec_enabled("compact_binary_fcnmm", transpose, homo) and _available_backends(brainevent.compact_binary_fcnmm_p, platform):
                if "excluded" in PREPROCESS_MODES:
                    fn, step_args = _make_compact_fcnmm_excluded_spec(data, transpose=transpose, homo=homo)
                    specs.append(BenchmarkSpec("compact_binary_fcnmm", "mm", "default", "excluded", transpose, homo, fn, step_args))
                if "included" in PREPROCESS_MODES:
                    fn, step_args = _make_compact_fcnmm_included_spec(data, transpose=transpose, homo=homo)
                    specs.append(BenchmarkSpec("compact_binary_fcnmm", "mm", "default", "included", transpose, homo, fn, step_args))
    return specs


def _block_until_ready(output):
    jax.block_until_ready(output)
    return output


def _timed_step_count():
    if BENCHMARK_STEPS is None:
        return _coba_step_count()
    steps = int(BENCHMARK_STEPS)
    if steps <= 0:
        raise ValueError("BENCHMARK_STEPS must be None or a positive integer.")
    return steps


def benchmark_spec(spec):
    """Benchmark one spec while cycling through deterministic spike steps."""
    run_jit = jax.jit(spec.fn)

    try:
        _block_until_ready(run_jit(*spec.step_args[0]))
        for i in range(N_WARMUP):
            _block_until_ready(run_jit(*spec.step_args[i % SPIKE_STEPS]))

        timed_steps = _timed_step_count()
        times = []
        for i in range(timed_steps):
            args = spec.step_args[i % SPIKE_STEPS]
            start = time.perf_counter()
            _block_until_ready(run_jit(*args))
            times.append(time.perf_counter() - start)

        times_s = np.asarray(times, dtype=np.float64)
        return BenchmarkRow(
            spec.operator,
            spec.kind,
            spec.backend,
            spec.preprocess,
            spec.transpose,
            spec.homo,
            float(np.mean(times_s)),
            float(np.std(times_s)),
            float(np.min(times_s)),
            True,
            "",
        )
    except Exception as exc:
        return BenchmarkRow(
            spec.operator,
            spec.kind,
            spec.backend,
            spec.preprocess,
            spec.transpose,
            spec.homo,
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            f"{type(exc).__name__}: {exc}",
        )


def _format_bool(value):
    return "T" if value else "NT"


def _format_homo(value):
    return "homo" if value else "hetero"


def _coba_step_count():
    return int(round(COBA_DURATION_MS / COBA_DT_MS))


def _coba_elapsed_s(single_step_s):
    return single_step_s * _timed_step_count() * COBA_CONN_UPDATES_PER_STEP


def _format_row(row):
    if row.success:
        single_step_s = row.elapsed_s
        return {
            "operator": row.operator,
            "kind": row.kind,
            "backend": row.backend,
            "preprocess": row.preprocess,
            "transpose": _format_bool(row.transpose),
            "weight": _format_homo(row.homo),
            "elapsed_s": f"{_coba_elapsed_s(single_step_s):.6f}",
            "single_step_s": f"{single_step_s:.6f}",
            "std_s": f"{row.std_s:.6f}",
            "min_s": f"{row.min_s:.6f}",
            "timed_steps": f"{_timed_step_count()}",
            "coba_steps": f"{_coba_step_count()}",
            "conn_updates": f"{COBA_CONN_UPDATES_PER_STEP}",
            "status": "OK",
        }
    return {
        "operator": row.operator,
        "kind": row.kind,
        "backend": row.backend,
        "preprocess": row.preprocess,
        "transpose": _format_bool(row.transpose),
        "weight": _format_homo(row.homo),
        "elapsed_s": "",
        "single_step_s": "",
        "std_s": "",
        "min_s": "",
        "timed_steps": f"{_timed_step_count()}",
        "coba_steps": f"{_coba_step_count()}",
        "conn_updates": f"{COBA_CONN_UPDATES_PER_STEP}",
        "status": row.error,
    }


def print_rows(rows):
    columns = (
        "operator",
        "kind",
        "backend",
        "preprocess",
        "transpose",
        "weight",
        "elapsed_s",
        "single_step_s",
        "std_s",
        "min_s",
        "timed_steps",
        "coba_steps",
        "conn_updates",
        "status",
    )
    table = [_format_row(row) for row in rows]
    if not table:
        print("No benchmark rows to display.")
        return
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in table))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    rule = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(rule)
    for row in table:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))


def _runtime_device(platform):
    if platform == "gpu":
        try:
            return jax.devices("gpu")[0]
        except RuntimeError:
            return None
    return jax.devices(platform)[0]


def build_benchmark_cases(scale=None, conn_num=None, batch_size=None, backend=None):
    scales_to_use = [scale] if scale is not None else scales
    conns_to_use = [conn_num] if conn_num is not None else conn_nums
    batches_to_use = [batch_size] if batch_size is not None else default_batch_sizes
    backends_to_use = [backend] if backend is not None else backends
    cases = []
    for s in scales_to_use:
        for cn in conns_to_use:
            for batch in batches_to_use:
                for back in backends_to_use:
                    cases.append(BenchmarkCase(s, cn, batch, back))
    return tuple(cases)


def _run_benchmark_case(case, *, platform=PLATFORM):
    global SCALE, CONN, N_B, BACKENDS

    old_scale = SCALE
    old_conn = CONN
    old_n_b = N_B
    old_backends = BACKENDS
    try:
        SCALE = case.scale
        CONN = case.conn_num
        N_B = case.batch_size
        BACKENDS = None if case.backend is None else (case.backend,)

        n_pre, n_post, conn, n_b = _benchmark_dimensions()
        print(
            f"seed={SEED}, scale={SCALE}, shape=({n_pre}, {n_post}), "
            f"conn={conn}, n_b={n_b}, backend={case.backend}, "
            f"spike_rate={SPIKE_RATE:.1%}, spike_steps={SPIKE_STEPS}"
        )

        data = make_fixed_data()
        specs = build_benchmark_specs(data, platform=platform)
        rows = []
        for i, spec in enumerate(specs, 1):
            print(
                f"[{i}/{len(specs)}] {spec.operator} "
                f"{_format_bool(spec.transpose)} {_format_homo(spec.homo)} "
                f"{spec.backend} preprocess={spec.preprocess}"
            )
            rows.append(benchmark_spec(spec))
        print_rows(rows)
        return rows
    finally:
        SCALE = old_scale
        CONN = old_conn
        N_B = old_n_b
        BACKENDS = old_backends


def _operator_filter_for_data_type(data_type, *, kind=None):
    if kind is not None and kind not in ("mv", "mm"):
        raise ValueError("kind must be one of None, 'mv', or 'mm'.")
    if data_type is None:
        if kind is None:
            return None
        if kind == "mv":
            return (
                "fcnmv",
                "binary_fcnmv",
                "bitpack_binary_fcnmv",
                "compact_binary_fcnmv",
            )
        return (
            "fcnmm",
            "binary_fcnmm",
            "bitpack_binary_fcnmm",
            "bitpack_binary_fcnmm_p",
            "compact_binary_fcnmm",
        )
    if kind is None:
        kind = "mm"
    if data_type == "float":
        return ("fcnmv",) if kind == "mv" else ("fcnmm",)
    if data_type == "binary":
        return ("binary_fcnmv",) if kind == "mv" else ("binary_fcnmm",)
    if data_type in ("bitpack", "bitpack_a0", "bitpack_a1"):
        if kind == "mv":
            return ("bitpack_binary_fcnmv",)
        return ("bitpack_binary_fcnmm", "bitpack_binary_fcnmm_p")
    if data_type == "compact":
        return ("compact_binary_fcnmv",) if kind == "mv" else ("compact_binary_fcnmm",)
    raise ValueError(
        "data_type must be one of None, 'float', 'binary', "
        "'bitpack', 'bitpack_a0', 'bitpack_a1', or 'compact'."
    )


def _transpose_filter_for_mode(mode):
    if mode is None:
        return None
    if mode == "post":
        # COBA EI post mode applies Binary/BitPacked operand @ FixedPostNumConn.
        # FixedPostNumConn.__rmatmul__ transposes the batch operand internally
        # and calls the primitive with transpose=True.
        return (True,)
    if mode == "pre":
        # COBA EI pre mode applies FixedPostNumConn @ operand, which calls the
        # primitive with transpose=False.
        return (False,)
    raise ValueError("mode must be one of None, 'post', or 'pre'.")


def benchmark_conn(
    *,
    data_type=None,
    kind=None,
    mode=None,
    scale=None,
    conn_num=None,
    batch_size=None,
    homo=None,
    backend=None,
    mv_layout="row_gather",
    platform=PLATFORM,
):
    global PACK_AXIS, OPERATOR_FILTER, TRANSPOSE_FILTER, HOMO_FILTER

    old_pack_axis = PACK_AXIS
    old_operator_filter = OPERATOR_FILTER
    old_transpose_filter = TRANSPOSE_FILTER
    old_homo_filter = HOMO_FILTER
    if mv_layout not in ("row_gather", "col_scatter", "auto"):
        raise ValueError("mv_layout must be one of 'row_gather', 'col_scatter', or 'auto'.")
    if data_type == "bitpack_a1":
        PACK_AXIS = 1
    elif data_type == "bitpack_a0":
        PACK_AXIS = 0
    OPERATOR_FILTER = _operator_filter_for_data_type(data_type, kind=kind)
    TRANSPOSE_FILTER = _transpose_filter_for_mode(mode)
    HOMO_FILTER = None if homo is None else (bool(homo),)

    device = _runtime_device(platform)
    try:
        if device is None:
            print("ERROR: No GPU device found.")
            return []

        print(f"FCN direct operator benchmark - platform={platform}, device={device}")
        print(
            f"data_type={data_type}, kind={kind}, mode={mode}, homo={homo}, "
            f"mv_layout={mv_layout}"
        )
        print(
            f"warmup={N_WARMUP}, runs={N_RUNS}, timed_steps={_timed_step_count()}, "
            f"benchmark_steps={BENCHMARK_STEPS}, preprocess_modes={PREPROCESS_MODES}"
        )

        rows = []
        cases = build_benchmark_cases(
            scale=scale,
            conn_num=conn_num,
            batch_size=batch_size,
            backend=backend,
        )
        for i, case in enumerate(cases, 1):
            print(
                f"\ncase [{i}/{len(cases)}] "
                f"scale={case.scale}, conn={case.conn_num}, "
                f"n_b={case.batch_size}, backend={case.backend}"
            )
            rows.extend(_run_benchmark_case(case, platform=platform))
        return rows
    finally:
        PACK_AXIS = old_pack_axis
        OPERATOR_FILTER = old_operator_filter
        TRANSPOSE_FILTER = old_transpose_filter
        HOMO_FILTER = old_homo_filter


def main():
    benchmark_conn(
        data_type=MAIN_DATA_TYPE,
        kind=MAIN_KIND,
        mode=MAIN_MODE,
        scale=SCALE,
        conn_num=CONN,
        batch_size=N_B,
        homo=MAIN_HOMO,
        backend=MAIN_BACKEND,
        mv_layout=MAIN_MV_LAYOUT,
        platform=PLATFORM,
    )


if __name__ == "__main__":
    main()
