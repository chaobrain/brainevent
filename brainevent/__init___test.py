# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Tests keeping the public surface of :mod:`brainevent` and its API reference in sync.

The Sphinx reference under ``docs/reference/apis`` is hand-maintained: an
``autosummary`` entry that outlives its symbol renders as a broken stub, and a
new export without an entry is invisible to users. Both directions are checked
here so the drift is caught by the test suite rather than by a reader.
"""

import re
from pathlib import Path
from typing import Dict, List

import pytest

import brainevent
from brainevent import config as be_config

# Repository root -> docs; absent in a wheel-only installation, where these tests
# are skipped rather than failed.
_DOCS_APIS = Path(__file__).resolve().parents[1] / 'docs' / 'reference' / 'apis'

# Names exported for convenience that are deliberately not autosummary entries.
_NOT_DOCUMENTED = frozenset({
    'config',  # submodule handle; documented as a page of its own (config.rst)
})

# ``currentmodule`` of each reference page.
_PAGE_MODULES = {'config.rst': be_config}

_ENTRY = re.compile(r'^ {3}([A-Za-z_][A-Za-z0-9_]*)$')


def _autosummary_entries(rst: Path) -> List[str]:
    """Collect the symbol names listed in every ``autosummary`` block of ``rst``.

    Parameters
    ----------
    rst : Path
        Reference page to parse.

    Returns
    -------
    list of str
        Entry names, in document order, with duplicates preserved.
    """
    names: List[str] = []
    in_block = False
    for line in rst.read_text(encoding='utf-8').splitlines():
        if line.strip().startswith('.. autosummary::'):
            in_block = True
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith(':'):
            continue  # blank separator or block option (:toctree:, :template:, ...)
        match = _ENTRY.match(line)
        if match:
            names.append(match.group(1))
        else:
            in_block = False  # prose, directive or heading closes the block
    return names


def _pages() -> Dict[str, List[str]]:
    """Map each reference page filename to its ``autosummary`` entries."""
    return {
        rst.name: _autosummary_entries(rst)
        for rst in sorted(_DOCS_APIS.glob('*.rst'))
        if rst.name != 'index.rst'
    }


pytestmark = pytest.mark.skipif(
    not _DOCS_APIS.is_dir(),
    reason='API reference sources are not shipped with the installed package',
)


def test_every_documented_symbol_exists():
    """No ``autosummary`` entry outlives the symbol it documents."""
    dangling = {}
    for page, names in _pages().items():
        module = _PAGE_MODULES.get(page, brainevent)
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            dangling[page] = missing
    assert not dangling, f'API reference documents symbols that no longer exist: {dangling}'


def test_every_public_export_is_documented():
    """Every name in ``brainevent.__all__`` appears on some reference page."""
    documented = {name for names in _pages().values() for name in names}
    undocumented = sorted(
        name for name in brainevent.__all__
        if name not in documented and name not in _NOT_DOCUMENTED
    )
    assert not undocumented, (
        'public exports missing from docs/reference/apis: '
        f'{undocumented}'
    )


def test_every_config_export_is_documented():
    """Every name in ``brainevent.config.__all__`` appears on ``config.rst``."""
    documented = set(_pages()['config.rst'])
    undocumented = sorted(n for n in be_config.__all__ if n not in documented)
    assert not undocumented, f'brainevent.config exports missing from config.rst: {undocumented}'


def test_no_symbol_is_documented_twice():
    """A symbol is documented on exactly one page, so ``generated/`` has one stub for it."""
    seen: Dict[str, str] = {}
    duplicates = []
    for page, names in _pages().items():
        for name in names:
            if name in seen:
                duplicates.append(f'{name} ({seen[name]} and {page})')
            else:
                seen[name] = page
    assert not duplicates, f'symbols documented on more than one page: {duplicates}'


def test_deprecated_aliases_are_not_documented():
    """Deprecated aliases stay out of the reference so users land on the replacements."""
    documented = {name for names in _pages().values() for name in names}
    deprecated = {'FixedPreNumConn', 'FixedPostNumConn'}
    assert not (documented & deprecated), (
        'deprecated aliases must not be documented; use FixedNumPerPre / FixedNumPerPost'
    )
