"""Regression checks for mathematical markup in authored documentation."""

from __future__ import annotations

import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
INLINE_CODE = re.compile(r"`[^`\n]*`")
TEX_COMMAND = re.compile(r"\\[A-Za-z]+")
UNSUPPORTED_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")
DIRECT_AUTODOC = re.compile(
    r"^```\{auto(?:attribute|class|function|method|module)\}", re.MULTILINE
)


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(DOCS_ROOT)
    except ValueError:
        return path


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _math_markup_violations(path: Path) -> list[str]:
    violations: list[str] = []
    in_fence = False
    fence = ""
    in_display_math = False

    for line_number, original_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = original_line.lstrip()
        if not in_fence and stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            in_fence = True
            continue
        if in_fence:
            if stripped.startswith(fence):
                in_fence = False
                fence = ""
            continue

        line = INLINE_CODE.sub("", original_line)
        for delimiter in UNSUPPORTED_DELIMITERS:
            if delimiter in line:
                violations.append(
                    f"{_display_path(path)}:{line_number}: "
                    f"unsupported math delimiter {delimiter!r}"
                )

        prose: list[str] = []
        in_inline_math = False
        index = 0
        while index < len(line):
            if line.startswith("$$", index) and not _is_escaped(line, index):
                if in_inline_math:
                    violations.append(
                        f"{_display_path(path)}:{line_number}: "
                        "display delimiter inside inline math"
                    )
                in_display_math = not in_display_math
                index += 2
                continue
            if line[index] == "$" and not _is_escaped(line, index):
                if not in_display_math:
                    in_inline_math = not in_inline_math
                index += 1
                continue
            if not in_display_math and not in_inline_math:
                prose.append(line[index])
            index += 1

        if in_inline_math:
            violations.append(
                f"{_display_path(path)}:{line_number}: unmatched inline math delimiter"
            )

        command = TEX_COMMAND.search("".join(prose))
        if command is not None:
            violations.append(
                f"{_display_path(path)}:{line_number}: "
                f"TeX command {command.group()!r} outside math markup"
            )

    if in_fence:
        violations.append(f"{_display_path(path)}: unclosed code fence")
    if in_display_math:
        violations.append(f"{_display_path(path)}: unmatched display math delimiter")
    return violations


def test_authored_docs_use_supported_math_markup() -> None:
    violations = [
        violation
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if "_build" not in path.parts
        for violation in _math_markup_violations(path)
    ]
    assert not violations, "\n" + "\n".join(violations)


def test_math_markup_check_detects_the_failure_modes(tmp_path: Path) -> None:
    broken = tmp_path / "broken.md"
    broken.write_text(
        "Inline \\(x\\), raw \\alpha, unmatched $y, and display:\n\\[z\\]\n",
        encoding="utf-8",
    )

    messages = "\n".join(_math_markup_violations(broken))
    assert "unsupported math delimiter" in messages
    assert "outside math markup" in messages
    assert "unmatched inline math delimiter" in messages


def test_autodoc_uses_the_rst_compatibility_wrapper() -> None:
    violations = [
        str(_display_path(path))
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if "_build" not in path.parts
        and DIRECT_AUTODOC.search(path.read_text(encoding="utf-8")) is not None
    ]
    assert not violations, (
        "Sphinx autodoc emits reStructuredText. Wrap it in a MyST "
        "`{eval-rst}` fence:\n" + "\n".join(violations)
    )
