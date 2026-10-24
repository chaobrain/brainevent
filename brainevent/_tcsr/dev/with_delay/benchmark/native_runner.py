"""Call the two native delay benchmarks with one reusable case allocation."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import TracebackType
from typing import Any
import weakref

import numpy as np
from numpy.typing import NDArray

from brainevent._tcsr.dev.with_delay.operator.region_delay import DelayInputs, TASK_NNZ, TPR_THRESHOLD

_DIRECTORY = Path(__file__).resolve().parent
_INCLUDE = Path(__file__).resolve().parents[4] / "include"


class _Input(ctypes.Structure):
    _fields_ = [
        *[(name, ctypes.c_int64) for name in (
            "rows", "cols", "steps", "segments", "slots", "nnz", "weight_count", "capacity")],
        *[(name, ctypes.c_int32) for name in ("index_bits", "edge_bits", "ring_bits")],
        *[(name, ctypes.c_void_p) for name in (
            "weights", "indices", "indptr", "counts", "offsets", "row_tasks", "events")],
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        *[(name, ctypes.c_double) for name in (
            "warmup_total_ms", "timed_total_ms", "timed_mean_ms", "timed_std_ms",
            "timed_min_ms", "timed_max_ms")],
        *[(name, ctypes.c_int64) for name in ("duration_steps", "segment_count", "spmv_calls_per_run")],
    ]


@lru_cache(maxsize=1)
def _library() -> Any:
    compiler = shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError("nvcc is required for the native delay benchmark")
    sources = [
        _DIRECTORY / "benchmark_binary_csr_vecmat_native.cu",
        _DIRECTORY / "delay_benchmark_api.h",
        *sorted((_DIRECTORY.parent / "operator").glob("*.cu")),
        *sorted(_INCLUDE.rglob("*.h")),
    ]
    digest = hashlib.sha256(subprocess.check_output([compiler, "--version"]))
    configuration = [f"-DBE_DELAY_TPR_THRESHOLD={TPR_THRESHOLD}", f"-DBE_DELAY_TASK_NNZ={TASK_NNZ}"]
    digest.update(" ".join(configuration).encode())
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    build = _DIRECTORY / ".build"
    build.mkdir(exist_ok=True)
    target = build / ("delay_" + digest.hexdigest()[:20] + ".so")
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=build) as temporary:
            result = Path(temporary) / "delay.so"
            subprocess.run([
                compiler, "-std=c++17", "-O3", "-arch=native", "--shared", "-Xcompiler=-fPIC",
                *configuration, "-I", str(_INCLUDE), str(sources[0]), "-lcusparse", "-o", str(result),
            ], check=True, capture_output=True, text=True, timeout=600)
            result.replace(target)
    library = ctypes.CDLL(str(target))
    library.create_delay_benchmark.argtypes = [
        ctypes.POINTER(_Input), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    library.create_delay_benchmark.restype = ctypes.c_int
    library.destroy_delay_benchmark.argtypes = [ctypes.c_void_p]
    library.destroy_delay_benchmark.restype = None
    for function in (library.benchmark_delay_hybrid, library.benchmark_delay_cusparse):
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Stats),
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_int
    return library


@dataclass(frozen=True)
class BenchmarkResult:
    """Report projection-only timing and optional final-run outputs.

    Parameters
    ----------
    timing : dict
        CUDA event statistics in milliseconds, plus step/segment counts.
    consumed : numpy.ndarray or None
        Float32 output of shape (T, N) from the final run, when captured.
    ring : numpy.ndarray or None
        Final ring of shape (L, N), including arrivals after step T-1.
    """

    timing: dict[str, float | int]
    consumed: NDArray[np.float32] | None
    ring: NDArray[np.float32] | None

    @property
    def mean_step_ms(self) -> float:
        """Return mean projection time per complete step.

        Returns
        -------
        float
            Mean run time divided by T, for both hybrid and cuSPARSE.
        """
        return self.timing["timed_mean_ms"] / self.timing["duration_steps"]


class NativeCase:
    """Own GPU inputs and resources for one delay benchmark case.

    Parameters
    ----------
    inputs : DelayInputs
        Validated arrays returned by ``prepare_delay_inputs``.

    Notes
    -----
    Upload once. The first cuSPARSE call prepares split and descriptors before
    timing; subsequent calls reuse them. Calls on one instance are sequential.
    Use as a context manager to release GPU resources promptly.
    """

    def __init__(self, inputs: DelayInputs) -> None:
        if not isinstance(inputs, DelayInputs):
            raise TypeError("inputs must be DelayInputs")
        self.inputs = inputs
        self._lib = _library()
        arrays = (inputs.weights, inputs.indices, inputs.indptr, inputs.delay_count,
                  inputs.delay_off, inputs.row_task_count, inputs.events)
        view = _Input(
            *inputs.shape, inputs.duration_steps, len(inputs.delay_steps), inputs.buffer_slots,
            inputs.indices.size, inputs.weights.size, inputs.task_capacity,
            inputs.indices.dtype.itemsize * 8, inputs.indptr.dtype.itemsize * 8,
            inputs.delay_off.dtype.itemsize * 8, *(array.ctypes.data for array in arrays),
        )
        self._handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(2048)
        if self._lib.create_delay_benchmark(ctypes.byref(view), ctypes.byref(self._handle), error, len(error)):
            raise RuntimeError(error.value.decode("utf-8"))
        self._finalizer = weakref.finalize(self, self._lib.destroy_delay_benchmark, self._handle)

    def __enter__(self) -> NativeCase:
        """Enter the case's resource scope.

        Returns
        -------
        NativeCase
            This live case.
        """
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        """Release GPU resources on leaving a context.

        Parameters
        ----------
        kind, value, traceback : object or None
            Context exception information; exceptions propagate unchanged.
        """
        self.close()

    def close(self) -> None:
        """Release the case's GPU resources, allowing repeated close calls."""
        self._finalizer()

    def _run(self, function: Any, warmup: int, runs: int, capture: bool) -> BenchmarkResult:
        if not self._finalizer.alive:
            raise RuntimeError("native case is closed")
        if type(warmup) is not int or not 0 <= warmup <= 2**31 - 1:
            raise ValueError("warmup must be a nonnegative int32")
        if type(runs) is not int or not 1 <= runs <= 2**31 - 1:
            raise ValueError("runs must be a positive int32")
        steps, columns = self.inputs.duration_steps, self.inputs.shape[1]
        output = np.empty((steps, columns), np.float32) if capture else None
        ring = np.empty((self.inputs.buffer_slots, columns), np.float32) if capture else None
        stats = _Stats()
        error = ctypes.create_string_buffer(2048)
        code = function(self._handle, warmup, runs, ctypes.byref(stats),
                        None if output is None else output.ctypes.data,
                        None if ring is None else ring.ctypes.data, error, len(error))
        if code:
            raise RuntimeError(error.value.decode("utf-8"))
        return BenchmarkResult({field[0]: getattr(stats, field[0]) for field in _Stats._fields_}, output, ring)

    def run_hybrid(self, *, warmup: int = 2, runs: int = 5, capture: bool = False) -> BenchmarkResult:
        """Benchmark one complete hybrid projection per event column.

        Parameters
        ----------
        warmup, runs : int, optional
            Untimed-report warmup runs and measured runs; each executes T steps.
        capture : bool, optional
            Return final-run consumed outputs and remaining ring contents.

        Returns
        -------
        BenchmarkResult
            Projection-only timings and requested outputs.
        """
        return self._run(self._lib.benchmark_delay_hybrid, warmup, runs, capture)

    def run_cusparse(self, *, warmup: int = 2, runs: int = 5, capture: bool = False) -> BenchmarkResult:
        """Benchmark all delay segments as one step for each event column.

        Parameters
        ----------
        warmup, runs : int, optional
            Warmup and measured runs, each with the same T event columns.
        capture : bool, optional
            Return final-run outputs for correctness checks.

        Returns
        -------
        BenchmarkResult
            Timings covering all nonempty segments' SpMV plus RingAdd. Split,
            descriptor setup, consumption and head advancement are excluded.
        """
        return self._run(self._lib.benchmark_delay_cusparse, warmup, runs, capture)
