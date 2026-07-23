import re
from pathlib import Path


NORMAL_LIGHT_DIR = Path(__file__).resolve().parents[2] / "dev" / "normal"


def _be_exports(source: str) -> set[str]:
    return set(re.findall(r"^// @BE ([A-Za-z0-9_]+)$", source, re.MULTILINE))


def test_dev_normal_light_cuda_exports_non_f32_weight_kernels():
    for filename in (
        "float_jitn.cu",
        "float_jitnmv.cu",
        "float_jitnmm.cu",
        "binary_jitnmv.cu",
        "binary_jitnmm.cu",
        "csr.cu",
        "dt2t.cu",
    ):
        exports = _be_exports((NORMAL_LIGHT_DIR / filename).read_text())
        f32_exports = {name for name in exports if name.endswith("_f32")}

        assert f32_exports, f"{filename} should expose at least one f32 light kernel."
        for name in sorted(f32_exports):
            prefix = name.removesuffix("_f32")
            for suffix in ("_f64", "_f16", "_bf16"):
                assert prefix + suffix in exports
