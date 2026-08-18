"""Tests for the selected tutorial notebook execution scope."""

from __future__ import annotations

import ast
import json
from pathlib import Path


DOCS_ROOT = Path(__file__).parents[1]
EXECUTED_NOTEBOOKS = {
    Path("getting-started/quickstart.ipynb"),
    Path("tutorials/data-structures/02_sparse_matrices.ipynb"),
    Path("tutorials/data-structures/03_jit_connectivity.ipynb"),
    Path("tutorials/data-structures/04_fixed_connections.ipynb"),
    Path("tutorials/events/binary-events.ipynb"),
    Path("tutorials/events/synaptic-plasticity.ipynb"),
}
EXECUTING_MODES = {"auto", "force", "cache", "inline"}


def _notebook_metadata(path: Path) -> dict:
    """Return top-level notebook metadata without executing any cells."""
    return json.loads(path.read_text(encoding="utf-8"))["metadata"]


def test_only_quickstart_data_and_events_execute_during_build() -> None:
    """Keep global execution off and enforce the exact six-file whitelist."""
    config_tree = ast.parse((DOCS_ROOT / "conf.py").read_text(encoding="utf-8"))
    global_mode = next(
        node.value.value
        for node in config_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "nb_execution_mode"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    )
    assert global_mode == "off"

    source_notebooks = (
        path
        for path in DOCS_ROOT.rglob("*.ipynb")
        if "_build" not in path.relative_to(DOCS_ROOT).parts
    )
    discovered_executed: set[Path] = set()
    for notebook_path in source_notebooks:
        relative_path = notebook_path.relative_to(DOCS_ROOT)
        metadata = _notebook_metadata(notebook_path)
        if relative_path in EXECUTED_NOTEBOOKS:
            assert metadata["mystnb"] == {
                "execution_mode": "force",
                "execution_timeout": 120,
            }, relative_path
            assert metadata["kernelspec"]["name"] == "python3", relative_path
            discovered_executed.add(relative_path)
        else:
            mode = metadata.get("mystnb", {}).get("execution_mode")
            assert mode not in EXECUTING_MODES, relative_path

    assert discovered_executed == EXECUTED_NOTEBOOKS
