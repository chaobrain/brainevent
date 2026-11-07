"""Strict source and compiled-call checks for the hybrid network example.

This module uses only the standard library until its context manager is entered,
so the negative checks can run without initializing CUDA or importing JAX.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import importlib
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch


FORBIDDEN_SOURCES = frozenset({
    "binary_csrmv.cu", "binary_csrmm.cu",
    "binary_indexed_csrmv.cu", "binary_indexed_csrmm.cu",
})
FORBIDDEN_TARGETS = (
    "csr_binary_csrmv.", "csr_binary_csrmm.",
    "csr_binary_indexed_csrmv.", "csr_binary_indexed_csrmm.",
)
_LOAD_MODULES = (
    "brainevent._csr.binary.binary",
    "brainevent._csr.binary.binary_indexed",
    "brainevent._csr.float.float",
    "brainevent._sddmm._mv.sddmv",
    "brainevent._sddmm._mm.sddmm",
)
_TARGET_PATTERNS = (
    re.compile(r'\b(?:stablehlo|mhlo)\.custom_call\s+@(?:"([^"]+)"|([^\s(]+))'),
    re.compile(r'\b(?:custom_call_target|call_target_name)\s*=\s*"([^"]+)"'),
)


def _targets(text: str) -> list[str]:
    """Read both StableHLO assembly and executable HLO target spellings."""
    targets = []
    for pattern in _TARGET_PATTERNS:
        for match in pattern.finditer(text):
            targets.append(next(group for group in match.groups() if group is not None))
    return targets


def _check_source(source: str | Path) -> None:
    path = Path(source).resolve()
    if path.name in FORBIDDEN_SOURCES and "/_csr/binary/" in path.as_posix():
        raise AssertionError(f"Forbidden non-hybrid CUDA source: {path}")


def inspect_ir(text: str, *, backward: bool, layout: str, operand: str,
               expected_calls: int | None = None) -> dict[str, Any]:
    """Require hybrid event calls and reject legacy calls in a compiled graph.

    Parameters
    ----------
    text : str
        StableHLO assembly or the executable's HLO text, including loop bodies.
    backward : bool
        Whether this graph includes the example's weight and input gradients.
    layout : {'CSR', 'CSC'}
        Storage of the model's matrices. CSC must use the indexed mirror route.
    operand : {'mv', 'mm', 'vector', 'batch', 'batched'}
        Vector or matrix inputs. Matrix examples must have batch size above one.
    expected_calls : int or None, optional
        Expected sites per category in the example's StableHLO. Its sparse
        layer shapes are distinct, so each has one function definition. Do not
        enforce this on optimized HLO, which may clone/unroll computations.

    Returns
    -------
    dict
        JSON-safe target names, occurrence counts, and semantic categories.

    Raises
    ------
    AssertionError
        A forbidden/unrecognized custom call appears or a required route is
        missing. Positive checks also catch a silent change to pure JAX.
    ValueError
        The layout or operand selector is invalid.
    """
    layout = layout.upper()
    operand = {"vector": "mv", "batch": "mm", "batched": "mm"}.get(operand, operand)
    if layout not in ("CSR", "CSC") or operand not in ("mv", "mm"):
        raise ValueError("layout must be CSR/CSC and operand must be mv/mm")
    targets = _targets(text)
    counts = Counter(targets)
    forbidden = sorted(target for target in counts if target.startswith(FORBIDDEN_TARGETS))
    if forbidden:
        raise AssertionError(f"Forbidden non-hybrid custom-call targets: {forbidden}")

    forward = sorted(target for target in counts if (
        target.startswith("csr_hybrid_spmv_")
        or (target.startswith("csr_hybrid_mm_") and ".batched_spmm_v3" in target
            and not target.endswith("_device_info"))
    ))
    weight = sorted(target for target in counts if target.startswith((
        "csr_sddmv_hybrid_", "csr_sddmm_hybrid_",
    )))
    float_dx = sorted(target for target in counts if "cusparse" in target.lower()
                      and "csr" in target.lower())
    # XLA may add runtime calls for fusion, math, or sharding around our kernels.
    standard = sorted(target for target in counts if target.startswith((
        "__gpu$", "__cublas$", "__cudnn$", "xla.gpu.",
    )) or target in {"AllocateBuffer", "Sharding", "SPMDFullToShardShape", "SPMDShardToFullShape"})
    recognized = set(forward + weight + float_dx + standard)
    unexpected = sorted(set(counts) - recognized)
    if unexpected:
        raise AssertionError(f"Unrecognized custom-call targets: {unexpected}")
    if not forward:
        raise AssertionError("Missing hybrid forward custom calls; JAX fallback cannot pass the audit")
    expected_prefix = "csr_hybrid_spmv_" if operand == "mv" else "csr_hybrid_mm_"
    if any(not target.startswith(expected_prefix) for target in forward):
        raise AssertionError(f"Expected {operand} hybrid forward calls, got {forward}")
    if any(("col_major" in target) != (layout == "CSC") for target in forward):
        raise AssertionError(f"Unexpected hybrid storage route for {layout}: {forward}")
    if backward:
        prefix = "csr_sddmv_hybrid_" if operand == "mv" else "csr_sddmm_hybrid_"
        if not weight or any(not target.startswith(prefix) for target in weight):
            raise AssertionError(f"Missing or mismatched hybrid weight-gradient calls for {operand}: {weight}")
        if not float_dx:
            raise AssertionError("Missing cuSPARSE float input-gradient calls for heterogeneous weights")
    elif weight or float_dx:
        raise AssertionError(f"Forward-only graph unexpectedly contains gradient calls: {weight + float_dx}")
    category_counts = {
        'hybrid_forward': sum(counts[t] for t in forward),
        'hybrid_weight_gradient': sum(counts[t] for t in weight),
        'float_input_gradient': sum(counts[t] for t in float_dx),
    }
    if expected_calls is not None:
        expected = dict(hybrid_forward=expected_calls,
                        hybrid_weight_gradient=expected_calls if backward else 0,
                        float_input_gradient=expected_calls if backward else 0)
        if category_counts != expected:
            raise AssertionError(f'Missing or extra layer calls: {category_counts}; expected {expected}')
    return {
        "layout": layout, "operand": operand, "backward": bool(backward),
        "targets": dict(sorted(counts.items())), "custom_call_count": len(targets),
        "hybrid_forward": forward, "hybrid_weight_gradient": weight,
        "float_input_gradient": float_dx, "standard_runtime": standard,
        "category_counts": category_counts, "expected_calls": expected_calls, "forbidden_targets": [],
    }


class HybridAudit:
    """Record source requests and enforce native event-gradient dispatch.

    Attributes
    ----------
    phase : str
        User-assigned stage label copied into subsequent records.
    sources : list of dict
        JSON-safe CUDA source requests, including module and importer names.
    gradient_dispatches : list of dict
        JSON-safe sampled weight-gradient dispatch records. These record tracing
        calls, not the number of kernels executed by a compiled training step.

    Notes
    -----
    Source-loader interception alone cannot audit cached compilation. Always run
    :func:`inspect_ir` as well, and execute the compiled example numerically.
    The context temporarily patches module globals and is for a single-threaded
    diagnostic process, not concurrent training jobs in the same Python process.
    """

    def __init__(self, phase: str = "setup") -> None:
        self.phase = phase
        self.sources: list[dict[str, Any]] = []
        self.gradient_dispatches: list[dict[str, Any]] = []
        self._stack: ExitStack | None = None

    def __enter__(self) -> HybridAudit:
        if self._stack is not None:
            raise RuntimeError("HybridAudit contexts cannot be nested on the same instance")
        stack = ExitStack()
        try:
            for name in _LOAD_MODULES:
                module = importlib.import_module(name)
                original = module.load_cuda_file

                def load(source, *args, _original=original, _importer=name, **kwargs):
                    self.sources.append({
                        "phase": str(self.phase), "importer": _importer,
                        "source": str(Path(source).resolve()),
                        "module": str(kwargs.get("name", "")),
                    })
                    _check_source(source)
                    return _original(source, *args, **kwargs)

                stack.enter_context(patch.object(module, "load_cuda_file", load))
            dispatch = importlib.import_module("brainevent._sddmm.dispatch")
            original_sampled = dispatch.sddmm

            def sampled(a, b, indices, indptr=None, **kwargs):
                backend = kwargs.get("backend")
                batch = int(a.shape[1]) if a.ndim == 2 else 1
                columns = int(b.shape[-1])
                record = {
                    "phase": str(self.phase), "mode": kwargs.get("mode", "float"),
                    "transpose": kwargs.get("transpose", True), "batch": batch,
                    "columns": columns, "slots": int(indices.size),
                    "backend": getattr(backend, "requested", backend),
                }
                self.gradient_dispatches.append(record)
                if (record["mode"] not in ("binary", "weighted")
                        or record["transpose"] is not True or indptr is None
                        or not isinstance(backend, dispatch._CSRBackend)
                        or record["backend"] != "cuda_raw"
                        or batch < 1 or batch > 65535 or columns > 2**31 - 1):
                    raise AssertionError(f"Non-hybrid weight-gradient dispatch: {record}")
                return original_sampled(a, b, indices, indptr, **kwargs)

            stack.enter_context(patch.object(dispatch, "sddmm", sampled))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        assert self._stack is not None
        stack, self._stack = self._stack, None
        return bool(stack.__exit__(exc_type, exc, traceback))

    def inspect_ir(self, text: str, *, backward: bool, layout: str, operand: str,
                   expected_calls: int | None = None) -> dict[str, Any]:
        """Check one graph and attach the current phase to its JSON report."""
        return {"phase": str(self.phase), **inspect_ir(
            text, backward=backward, layout=layout, operand=operand, expected_calls=expected_calls,
        )}


def check_negative_guards() -> dict[str, int]:
    """Verify rejection paths without importing JAX or executing CUDA.

    Returns
    -------
    dict
        Counts of rejected sources, legacy targets, and missing-route graphs.
    """
    rejected_sources = rejected_targets = missing_routes = 0
    for source in sorted(FORBIDDEN_SOURCES):
        try:
            _check_source(Path("brainevent/_csr/binary") / source)
        except AssertionError:
            rejected_sources += 1
        else:
            raise AssertionError(f"Source guard failed to reject {source}")
    for prefix in FORBIDDEN_TARGETS:
        for syntax in (f"stablehlo.custom_call @{prefix}kernel()",
                       f'custom-call(), custom_call_target="{prefix}kernel"'):
            try:
                inspect_ir(syntax, backward=False, layout="CSR", operand="mv")
            except AssertionError as exc:
                if "Forbidden non-hybrid custom-call" not in str(exc):
                    raise AssertionError("Target parser failed to identify the forbidden call") from exc
                rejected_targets += 1
            else:
                raise AssertionError(f"Target guard failed to reject {prefix}")
    forward = "stablehlo.custom_call @csr_hybrid_spmv_binary.binary_csrmv_wat_hybrid_hetero_f32()"
    weight = "stablehlo.custom_call @csr_sddmv_hybrid_binary.hybrid_sddmv_binary_f32()"
    dx = 'custom-call(), custom_call_target="cusparse_csr_matvec_ffi"'
    inspect_ir("\n".join((forward, weight, dx)), backward=True, layout="CSR", operand="mv")
    for ir, backward in (("", False), (forward, True), (forward + "\n" + weight, True)):
        try:
            inspect_ir(ir, backward=backward, layout="CSR", operand="mv")
        except AssertionError:
            missing_routes += 1
        else:
            raise AssertionError("Audit accepted a graph with missing required routes")
    # A graph retaining one hybrid layer must not hide another layer's JAX fallback.
    try:
        inspect_ir("\n".join((forward, weight, dx)), backward=True,
                   layout="CSR", operand="mv", expected_calls=2)
    except AssertionError as exc:
        if 'Missing or extra layer calls' not in str(exc):
            raise
        missing_routes += 1
    else:
        raise AssertionError('Audit accepted an incomplete set of layer calls')
    return {"sources_rejected": rejected_sources, "legacy_targets_rejected": rejected_targets,
            "missing_routes_rejected": missing_routes}
