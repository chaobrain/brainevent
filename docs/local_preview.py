"""Keep BrainEvent documentation navigation inside local HTML builds."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import posixpath
import re
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit


LOCAL_PREVIEW_ENV = "BRAINEVENT_DOCS_LOCAL"
BRAINX_HOST = "brainx.chaobrain.com"
BRAINEVENT_PATH_PREFIX = "/brainevent"

_ANCHOR_TAG = re.compile(r"<a\b[^>]*>", flags=re.IGNORECASE | re.DOTALL)
_HREF_ATTRIBUTE = re.compile(
    r"(?<![\w:-])(?P<name>href)(?P<separator>\s*=\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    flags=re.IGNORECASE | re.DOTALL,
)


def local_preview_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether explicit local-documentation mode is enabled.

    Parameters
    ----------
    environ : mapping of str to str, optional
        Environment mapping to inspect. The process environment is used when
        omitted.

    Returns
    -------
    bool
        ``True`` only when ``BRAINEVENT_DOCS_LOCAL`` is exactly ``"1"``.
    """
    environment = os.environ if environ is None else environ
    return environment.get(LOCAL_PREVIEW_ENV) == "1"


def _is_brainevent_path(path: str) -> bool:
    return path == BRAINEVENT_PATH_PREFIX or path.startswith(
        f"{BRAINEVENT_PATH_PREFIX}/"
    )


def _local_target(path: str) -> str:
    target = path[len(BRAINEVENT_PATH_PREFIX) :].lstrip("/")
    if not target:
        return "index.html"
    if target.endswith("/"):
        return f"{target}index.html"
    if not posixpath.splitext(target)[1]:
        return f"{target}.html"
    return target


def _with_relative_target(
    parts: SplitResult, target: str, document_path: str
) -> str:
    document_directory = posixpath.dirname(document_path) or "."
    relative_target = posixpath.relpath(target, start=document_directory)
    return urlunsplit(("", "", relative_target, parts.query, parts.fragment))


def rewrite_href(href: str, document_path: str) -> str:
    """Rewrite one navigation target for a local BrainEvent HTML page.

    Parameters
    ----------
    href : str
        Original value of an HTML ``href`` attribute.
    document_path : str
        POSIX path of the containing HTML document relative to the Sphinx
        output directory.

    Returns
    -------
    str
        A local relative path for BrainEvent documentation, an absolute BrainX
        URL for root-relative ecosystem links, or the original external link.
    """
    parts = urlsplit(href)
    hosted_brainevent = (
        parts.netloc == BRAINX_HOST
        and parts.scheme in {"", "https"}
        and _is_brainevent_path(parts.path)
    )
    root_relative_brainevent = (
        not parts.scheme
        and not parts.netloc
        and _is_brainevent_path(parts.path)
    )

    if hosted_brainevent or root_relative_brainevent:
        return _with_relative_target(parts, _local_target(parts.path), document_path)

    if not parts.scheme and not parts.netloc and parts.path.startswith("/"):
        return urlunsplit(
            ("https", BRAINX_HOST, parts.path, parts.query, parts.fragment)
        )

    return href


def rewrite_html_navigation(content: str, document_path: str) -> str:
    """Rewrite anchor navigation in one generated HTML document.

    Parameters
    ----------
    content : str
        Generated HTML text.
    document_path : str
        POSIX path of the document relative to the Sphinx output directory.

    Returns
    -------
    str
        HTML with BrainEvent anchor targets localized and unrelated markup
        unchanged.
    """

    def rewrite_anchor(anchor_match: re.Match[str]) -> str:
        anchor = anchor_match.group(0)

        def rewrite_attribute(attribute_match: re.Match[str]) -> str:
            quote = attribute_match.group("quote")
            rewritten = rewrite_href(attribute_match.group("value"), document_path)
            return (
                f"{attribute_match.group('name')}"
                f"{attribute_match.group('separator')}"
                f"{quote}{rewritten}{quote}"
            )

        return _HREF_ATTRIBUTE.sub(rewrite_attribute, anchor)

    return _ANCHOR_TAG.sub(rewrite_anchor, content)


def rewrite_built_html(app: Any, exception: Exception | None) -> None:
    """Localize navigation after a successful Sphinx HTML build.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Sphinx application containing the builder and output directory.
    exception : Exception or None
        Build exception supplied by the ``build-finished`` event. Failed builds
        are left untouched.
    """
    if exception is not None or getattr(app.builder, "format", None) != "html":
        return

    output_directory = Path(app.outdir)
    for html_path in output_directory.rglob("*.html"):
        document_path = html_path.relative_to(output_directory).as_posix()
        content = html_path.read_text(encoding="utf-8")
        rewritten = rewrite_html_navigation(content, document_path)
        if rewritten != content:
            html_path.write_text(rewritten, encoding="utf-8")
