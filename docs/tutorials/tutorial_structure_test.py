"""Regression tests for the BrainEvent tutorial information architecture."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import nbformat
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
BASELINE_COMMIT = "7a4d5fc"

TARGET_NOTEBOOKS = (
    DOCS_ROOT / "getting-started" / "getting-started-with-brainevent.ipynb",
    DOCS_ROOT / "tutorials" / "events" / "binary-events.ipynb",
    DOCS_ROOT / "tutorials" / "events" / "synaptic-plasticity.ipynb",
    DOCS_ROOT / "tutorials" / "data-structures" / "02_sparse_matrices.ipynb",
    DOCS_ROOT / "tutorials" / "data-structures" / "04_fixed_connections.ipynb",
    DOCS_ROOT / "tutorials" / "data-structures" / "03_jit_connectivity.ipynb",
)

REQUIRED_NOTEBOOK_HEADINGS = {
    "tutorials/data-structures/02_sparse_matrices.ipynb": (
        "# CSR and CSC Sparse Matrices",
        "## Why Use Sparse Connectivity Data?",
        "## COO Input, CSR Storage, and CSC Storage",
        "## Constructing CSR and CSC Data",
        "## Combining Sparse Data with Binary Events",
        "## Memory, Correctness, and Performance",
        "## Build a Sparse Event-Driven Network",
        "## Inspect the Connectivity Structure",
        "## Choosing CSR or CSC",
        "## Summary and Next Steps",
    ),
    "tutorials/data-structures/04_fixed_connections.ipynb": (
        "# Fixed Connection Count Structures",
        "## Why Fix the Number of Connections?",
        "## Fixed Fan-Out with FixedNumPerPre",
        "## Fixed Fan-In with FixedNumPerPost",
        "## Combining Fixed Connectivity with Binary Events",
        "## Build a Fixed-Degree Network",
        "## Memory and Performance Characteristics",
        "## Choosing Fan-In or Fan-Out Constraints",
        "## Summary and Next Steps",
    ),
    "tutorials/data-structures/03_jit_connectivity.ipynb": (
        "# Just-in-Time Connection Matrices",
        "## Why Generate Connections Just in Time?",
        "## Homogeneous-Weight JIT Connectivity",
        "## Normally Distributed JIT Connectivity",
        "## Uniformly Distributed JIT Connectivity",
        "## Memory and Performance Trade-offs",
        "## Build a Large Random Network",
        "## Row- and Column-Oriented Connectivity",
        "## Summary and Next Steps",
    ),
}

RETIRED_NOTEBOOKS = (
    DOCS_ROOT / "tutorials" / "data-structures" / "01_eventarray_basics.ipynb",
    DOCS_ROOT / "tutorials" / "data-structures" / "05_synaptic_plasticity.ipynb",
)


def _read(path: Path) -> str:
    """Return UTF-8 text from ``path``."""
    return path.read_text(encoding="utf-8")


def _active_document_paths() -> list[Path]:
    """Return active RST and notebook sources, excluding historical records."""
    excluded = {DOCS_ROOT / "specs", DOCS_ROOT / "superpowers"}
    paths: list[Path] = []
    for suffix in ("*.rst", "*.ipynb"):
        for path in DOCS_ROOT.rglob(suffix):
            if any(parent == root for parent in path.parents for root in excluded):
                continue
            paths.append(path)
    return sorted(paths)


def _git_bytes(revision: str, path: Path) -> bytes:
    """Read a repository-relative file as stored in ``revision``."""
    relative = path.relative_to(REPO_ROOT).as_posix()
    return subprocess.check_output(
        ["git", "show", f"{revision}:{relative}"], cwd=REPO_ROOT
    )


def _require(condition: bool, message: str) -> None:
    """Fail concisely without expensive assertion-introspection rendering."""
    if not condition:
        pytest.fail(message, pytrace=False)


def test_target_documents_exist_and_retired_documents_are_removed() -> None:
    """Require the approved target set and reject superseded notebooks."""
    missing = [str(path.relative_to(REPO_ROOT)) for path in TARGET_NOTEBOOKS if not path.is_file()]
    remaining = [
        str(path.relative_to(REPO_ROOT)) for path in RETIRED_NOTEBOOKS if path.exists()
    ]
    _require(not missing, f"Missing target notebooks: {missing}")
    _require(not remaining, f"Retired notebooks still present: {remaining}")


def test_top_level_navigation_exposes_getting_started_data_events_and_custom() -> None:
    """Require the approved navigation groups in learning order."""
    index = _read(DOCS_ROOT / "index.rst")
    getting_started = "getting-started/getting-started-with-brainevent"
    data = "tutorials/data-structures/index"
    events = "tutorials/events/index"
    custom = "tutorials/custom-operators/index"

    _require(getting_started in index, f"Missing navigation entry: {getting_started}")
    _require(data in index, f"Missing navigation entry: {data}")
    _require(events in index, f"Missing navigation entry: {events}")
    _require(custom in index, f"Missing navigation entry: {custom}")
    navigation = index[index.index(".. toctree::") :]
    _require(
        navigation.index(getting_started)
        < navigation.index(data)
        < navigation.index(events)
        < navigation.index(custom),
        "Getting Started, Data, Events, and Custom operators are out of order",
    )


def test_data_and_events_toctrees_have_the_approved_order() -> None:
    """Require stable Data paths and descriptive Events paths in order."""
    data_index = _read(DOCS_ROOT / "tutorials" / "data-structures" / "index.rst")
    _require("Data structures & operators" not in data_index, "Retired Data heading remains")
    _require(
        data_index.index("02_sparse_matrices") < data_index.index("04_fixed_connections"),
        "CSR/CSC must precede fixed-count connectivity",
    )
    _require(
        data_index.index("04_fixed_connections") < data_index.index("03_jit_connectivity"),
        "Fixed-count connectivity must precede JIT connectivity",
    )

    events_index = _read(DOCS_ROOT / "tutorials" / "events" / "index.rst")
    _require(
        events_index.index("binary-events") < events_index.index("synaptic-plasticity"),
        "Binary events must precede synaptic plasticity",
    )


def test_active_tutorial_titles_are_not_globally_numbered() -> None:
    """Reject global ``Tutorial N`` prefixes in active notebook titles."""
    numbered = re.compile(r"^# Tutorial \d+:", re.MULTILINE)
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _active_document_paths()
        if path.suffix == ".ipynb" and numbered.search(_read(path))
    ]
    _require(not offenders, f"Globally numbered tutorial titles: {offenders}")


def test_active_documents_do_not_reference_retired_paths_or_names() -> None:
    """Reject removed notebook paths, labels, and deprecated JIT names."""
    retired_terms = (
        "tutorials/data-structures/01_eventarray_basics",
        "tutorials/data-structures/05_synaptic_plasticity",
        "Data structures & operators",
        "JITCHomoR",
        "JITCHomoC",
    )
    offenders: dict[str, list[str]] = {}
    for path in _active_document_paths():
        matches = [term for term in retired_terms if term in _read(path)]
        if matches:
            offenders[str(path.relative_to(REPO_ROOT))] = matches
    _require(not offenders, f"Retired documentation terms remain: {offenders}")


def test_target_notebooks_are_valid_notebook_v4_documents() -> None:
    """Validate every target notebook as notebook format version 4."""
    for path in TARGET_NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)


def test_data_notebooks_follow_the_approved_heading_map() -> None:
    """Require every mapped Data heading in the approved teaching order."""
    for relative_path, required_headings in REQUIRED_NOTEBOOK_HEADINGS.items():
        notebook = nbformat.read(DOCS_ROOT / relative_path, as_version=4)
        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        positions = [markdown.find(heading) for heading in required_headings]
        _require(all(position >= 0 for position in positions), f"Missing heading in {relative_path}")
        _require(positions == sorted(positions), f"Heading order changed in {relative_path}")


def test_binary_event_tutorial_uses_raw_values_for_general_array_operations() -> None:
    """Keep general reductions and logic outside BinaryArray's focused API."""
    path = DOCS_ROOT / "tutorials" / "events" / "binary-events.ipynb"
    source = _read(path)
    unsupported_forms = ("events_2d.sum(", "event_a & event_b", "event_a | event_b")
    offenders = [form for form in unsupported_forms if form in source]
    _require(not offenders, f"Unsupported BinaryArray operations remain: {offenders}")


def test_custom_operator_tutorials_match_the_approved_baseline() -> None:
    """Prevent any change to Custom operators during this restructure."""
    root = DOCS_ROOT / "tutorials" / "custom-operators"
    changed: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        baseline = _git_bytes(BASELINE_COMMIT, path)
        current = path.read_bytes()
        if hashlib.sha256(current).digest() != hashlib.sha256(baseline).digest():
            changed.append(str(path.relative_to(REPO_ROOT)))
    _require(not changed, f"Custom operator tutorials changed: {changed}")
