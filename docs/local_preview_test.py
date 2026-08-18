"""Tests for local BrainEvent documentation navigation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).with_name("local_preview.py")
CONF_PATH = Path(__file__).with_name("conf.py")


def _load_local_preview():
    spec = importlib.util.spec_from_file_location("brainevent_local_preview", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, False),
        ({"BRAINEVENT_DOCS_LOCAL": ""}, False),
        ({"BRAINEVENT_DOCS_LOCAL": "0"}, False),
        ({"BRAINEVENT_DOCS_LOCAL": "true"}, False),
        ({"BRAINEVENT_DOCS_LOCAL": "1"}, True),
    ],
)
def test_local_preview_requires_exact_opt_in(environment, expected: bool) -> None:
    local_preview = _load_local_preview()
    assert local_preview.local_preview_enabled(environment) is expected


@pytest.mark.parametrize(
    ("href", "document_path", "expected"),
    [
        ("https://brainx.chaobrain.com/brainevent/", "index.html", "index.html"),
        (
            "https://brainx.chaobrain.com/brainevent/",
            "tutorials/events/index.html",
            "../../index.html",
        ),
        (
            "https://brainx.chaobrain.com/brainevent/getting-started/quickstart.html?mode=local#install",
            "tutorials/events/index.html",
            "../../getting-started/quickstart.html?mode=local#install",
        ),
        (
            "/brainevent/tutorials/events/",
            "getting-started/quickstart.html",
            "../tutorials/events/index.html",
        ),
        (
            "/research/papers-about-brainx#cite",
            "tutorials/events/index.html",
            "https://brainx.chaobrain.com/research/papers-about-brainx#cite",
        ),
        (
            "https://brainx.chaobrain.com/brainunit/",
            "index.html",
            "https://brainx.chaobrain.com/brainunit/",
        ),
        (
            "https://github.com/chaobrain/brainevent",
            "index.html",
            "https://github.com/chaobrain/brainevent",
        ),
        (
            "https://doi.org/10.1007/s10827-007-0038-6",
            "index.html",
            "https://doi.org/10.1007/s10827-007-0038-6",
        ),
        ("../index.html", "tutorials/events/index.html", "../index.html"),
        ("#section", "index.html", "#section"),
        ("mailto:test@example.com", "index.html", "mailto:test@example.com"),
    ],
)
def test_rewrite_href_localizes_only_brainevent_navigation(
    href: str, document_path: str, expected: str
) -> None:
    local_preview = _load_local_preview()
    assert local_preview.rewrite_href(href, document_path) == expected


def test_rewrite_html_navigation_changes_only_anchor_href_attributes() -> None:
    local_preview = _load_local_preview()
    content = (
        '<a class="brand" href="https://brainx.chaobrain.com/brainevent/">Home</a>'
        "<a HREF='/research/papers-about-brainx#cite'>Research</a>"
        '<a data-href="https://brainx.chaobrain.com/brainevent/">No href</a>'
        '<img src="https://brainx.chaobrain.com/brainevent/logo.png">'
        '<code>href="https://brainx.chaobrain.com/brainevent/"</code>'
    )
    rewritten = local_preview.rewrite_html_navigation(
        content, "tutorials/events/index.html"
    )
    assert 'href="../../index.html"' in rewritten
    assert "HREF='https://brainx.chaobrain.com/research/papers-about-brainx#cite'" in rewritten
    assert 'data-href="https://brainx.chaobrain.com/brainevent/"' in rewritten
    assert 'src="https://brainx.chaobrain.com/brainevent/logo.png"' in rewritten
    assert '<code>href="https://brainx.chaobrain.com/brainevent/"</code>' in rewritten


def test_rewrite_built_html_rewrites_root_and_nested_pages(tmp_path: Path) -> None:
    local_preview = _load_local_preview()
    root_page = tmp_path / "index.html"
    nested_page = tmp_path / "tutorials" / "events" / "index.html"
    nested_page.parent.mkdir(parents=True)
    hosted_link = '<a href="https://brainx.chaobrain.com/brainevent/">Home</a>'
    root_page.write_text(hosted_link, encoding="utf-8")
    nested_page.write_text(hosted_link, encoding="utf-8")
    app = SimpleNamespace(outdir=str(tmp_path), builder=SimpleNamespace(format="html"))
    local_preview.rewrite_built_html(app, exception=None)
    assert 'href="index.html"' in root_page.read_text(encoding="utf-8")
    assert 'href="../../index.html"' in nested_page.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("builder_format", "exception"),
    [("latex", None), ("html", RuntimeError("build failed"))],
)
def test_rewrite_built_html_skips_non_html_or_failed_builds(
    tmp_path: Path, builder_format: str, exception: Exception | None
) -> None:
    local_preview = _load_local_preview()
    page = tmp_path / "index.html"
    original = '<a href="https://brainx.chaobrain.com/brainevent/">Home</a>'
    page.write_text(original, encoding="utf-8")
    app = SimpleNamespace(
        outdir=str(tmp_path), builder=SimpleNamespace(format=builder_format)
    )
    local_preview.rewrite_built_html(app, exception=exception)
    assert page.read_text(encoding="utf-8") == original


def test_sphinx_config_wires_the_opt_in_local_preview_contract() -> None:
    tree = ast.parse(CONF_PATH.read_text(encoding="utf-8"))
    assignments = {
        target.id: value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        for value in [node.value]
    }
    assert ast.unparse(assignments["local_preview"]) == "local_preview_enabled()"
    assert ast.unparse(assignments["html_baseurl"]) == (
        "'' if local_preview else 'https://brainx.chaobrain.com/brainevent/'"
    )
    assert ast.unparse(assignments["brainx_inject_base"]) == "not local_preview"
    setup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    )
    setup_source = ast.unparse(setup)
    assert "if local_preview:" in setup_source
    assert "app.connect('build-finished', rewrite_built_html)" in setup_source
