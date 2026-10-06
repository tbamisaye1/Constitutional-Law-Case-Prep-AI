"""Fail if MCP code hardcodes workspace IDs, Bronner, or kind-name lists."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MCP_DIR = Path(__file__).resolve().parents[2] / "app" / "mcp"

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_BRONNER_RE = re.compile(r"bronner", re.I)
# Kind names inside list/set/frozenset literals only (not dict keys or SQL).
_KIND_NAMES = (
    "arguments|notebook|guide_edits|facts|openings|opinions|"
    "case_facts|cites|timeline|note_tabs"
)
_KIND_LIST_RE = re.compile(
    rf"""(?:frozenset|set)\s*\(\s*[\[{{]\s*["'](?:{_KIND_NAMES})["']"""
    rf"""|\[\s*["'](?:{_KIND_NAMES})["']\s*,"""
)


@pytest.mark.parametrize("path", sorted(MCP_DIR.rglob("*.py")))
def test_no_hardcoded_workspace_or_seed(path: Path):
    text = path.read_text(encoding="utf-8")
    if _UUID_RE.search(text):
        pytest.fail(f"{path.relative_to(MCP_DIR.parent.parent)} contains a UUID literal")
    if _BRONNER_RE.search(text):
        pytest.fail(f"{path.relative_to(MCP_DIR.parent.parent)} contains 'bronner'")
    if _KIND_LIST_RE.search(text):
        pytest.fail(
            f"{path.relative_to(MCP_DIR.parent.parent)} hardcodes library kinds in a list/set"
        )
