import re
from pathlib import Path


SCALAR_LIGHT_DIR = Path(__file__).resolve().parents[2] / "dev" / "scalar"


def _be_exports(source: str) -> set[str]:
    return set(re.findall(r"^// @BE ([A-Za-z0-9_]+)$", source, re.MULTILINE))


def test_dev_scalar_light_cuda_exports_non_f32_weight_kernels():
    for filename in (
        "float_jits.cu",
        "float_jitsmv.cu",
        "float_jitsmm.cu",
        "binary_jitsmv.cu",
        "binary_jitsmm.cu",
        "csr.cu",
        "dt2t.cu",
    ):
        exports = _be_exports((SCALAR_LIGHT_DIR / filename).read_text())
        f32_exports = {name for name in exports if name.endswith("_f32")}

        assert f32_exports, f"{filename} should expose at least one f32 light kernel."
        for name in sorted(f32_exports):
            prefix = name.removesuffix("_f32")
            for suffix in ("_f64", "_f16", "_bf16"):
                assert prefix + suffix in exports
