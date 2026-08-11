"""Regression checks for the documentation navigation hierarchy."""

from __future__ import annotations

import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
VALIDATION_ROOT = DOCS_ROOT / "validation"
TOCTREE_BLOCK = re.compile(r"```\{toctree\}\n(?P<body>.*?)```", re.DOTALL)


def _toctree_targets(path: Path) -> list[Path]:
    targets: list[Path] = []
    text = path.read_text(encoding="utf-8")
    for block in TOCTREE_BLOCK.finditer(text):
        for raw_line in block.group("body").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if "<" in line and line.endswith(">"):
                line = line.rsplit("<", maxsplit=1)[1][:-1]
            target = (path.parent / line).resolve()
            if not target.suffix:
                target = target.with_suffix(".md")
            targets.append(target)
    return targets


def test_validation_ledgers_have_one_toctree_parent() -> None:
    index = VALIDATION_ROOT / "index.md"
    ledgers = set(VALIDATION_ROOT.rglob("*.md")) - {index}
    parents: dict[Path, list[Path]] = {ledger: [] for ledger in ledgers}

    for source in DOCS_ROOT.rglob("*.md"):
        if "_build" in source.parts:
            continue
        for target in _toctree_targets(source):
            if target in parents:
                parents[target].append(source)

    assert all(sources == [index] for sources in parents.values())
    assert "orphan: true" not in "".join(
        ledger.read_text(encoding="utf-8") for ledger in ledgers
    )


def test_validation_sidebar_is_configured_as_one_semantic_leaf() -> None:
    config = (DOCS_ROOT / "conf.py").read_text(encoding="utf-8")
    css = (DOCS_ROOT / "_static" / "css" / "custom.css").read_text(encoding="utf-8")

    assert '"navigation_depth": 1' in config
    assert "_check_validation_sidebar" in config
    assert "legacy-audit.html" not in css
    assert "reproducing-simulations.html" not in css
