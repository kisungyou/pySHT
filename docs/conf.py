"""Sphinx configuration for the pySHT project website."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path
from xml.sax.saxutils import escape

from sphinx.application import Sphinx
from sphinx.errors import ExtensionError

project = "pySHT"
author = "Kisung You"
copyright = "2026, Kisung You"
release = version("pysht")
version = release

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinxext.opengraph",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autosummary_generate = True
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

html_theme = "pydata_sphinx_theme"
html_title = "pySHT"
html_short_title = "pySHT"
html_baseurl = "https://www.kisungyou.com/pysht/"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.png"
html_extra_path = ["robots.txt"]
html_show_sourcelink = True
html_last_updated_fmt = "%B %d, %Y"
html_meta = {
    "description": (
        "pySHT provides literature-traceable, independently validated "
        "statistical hypothesis tests for Python."
    ),
    "keywords": (
        "pySHT, pysht, statistical hypothesis testing, Python statistics, "
        "multivariate statistics, permutation tests"
    ),
}
html_context = {
    "default_mode": "light",
    "doc_path": "docs",
    "github_repo": "pySHT",
    "github_user": "kisungyou",
    "github_version": "master",
}
html_theme_options = {
    "back_to_top_button": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "github_url": "https://github.com/kisungyou/pySHT",
    "header_links_before_dropdown": 3,
    "logo": {
        "text": "pySHT",
    },
    "navbar_align": "content",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navigation_depth": 1,
    "search_bar_text": "Search pySHT…",
    "show_nav_level": 2,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}

ogp_site_url = html_baseurl
ogp_image = f"{html_baseurl}_static/og.png"
ogp_type = "website"
ogp_enable_meta_description = True
ogp_description_length = 200


def _check_validation_sidebar(app: Sphinx, exception: BaseException | None) -> None:
    """Require the validation directory to remain one semantic sidebar leaf."""
    if exception is not None or app.builder.name != "html":
        return

    output_root = Path(app.outdir)
    pages = sorted((output_root / "user-guide").rglob("*.html"))
    pages.extend(sorted((output_root / "validation").rglob("*.html")))
    anchor_pattern = re.compile(r"<a\b[^>]*>\s*Validation ledgers\s*</a>", re.DOTALL)
    failures: list[str] = []

    for page in pages:
        html = page.read_text(encoding="utf-8")
        sidebar_start = html.find('<div id="pst-primary-sidebar"')
        sidebar_end = html.find('<main id="main-content"', sidebar_start)
        if sidebar_start < 0 or sidebar_end < 0:
            failures.append(f"{page.relative_to(output_root)}: primary sidebar missing")
            continue

        sidebar = html[sidebar_start:sidebar_end]
        matches = list(anchor_pattern.finditer(sidebar))
        if len(matches) != 1:
            failures.append(
                f"{page.relative_to(output_root)}: expected one Validation ledgers "
                f"entry, found {len(matches)}"
            )
            continue

        match = matches[0]
        item_start = sidebar.rfind("<li", 0, match.start())
        item_tag_end = sidebar.find(">", item_start)
        item_end = sidebar.find("</li>", match.end())
        item_tag = sidebar[item_start:item_tag_end]
        item_body = sidebar[match.end() : item_end]
        if item_start < 0 or item_tag_end < 0 or item_end < 0:
            failures.append(
                f"{page.relative_to(output_root)}: malformed Validation ledgers entry"
            )
        elif "has-children" in item_tag or "<details" in item_body:
            failures.append(
                f"{page.relative_to(output_root)}: Validation ledgers has children"
            )

    if failures:
        raise ExtensionError(
            "Validation sidebar must be a single leaf entry:\n" + "\n".join(failures)
        )


def _write_sitemap(app: Sphinx, exception: BaseException | None) -> None:
    """Write a deterministic sitemap without network or process dependencies."""
    if exception is not None or app.builder.name != "html":
        return

    base_url = html_baseurl.rstrip("/") + "/"
    locations = [
        base_url + app.builder.get_target_uri(docname)
        for docname in sorted(app.env.found_docs)
    ]
    entries = "\n".join(
        f"  <url><loc>{escape(location)}</loc></url>" for location in locations
    )
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n"
        "</urlset>\n"
    )
    (Path(app.outdir) / "sitemap.xml").write_text(sitemap, encoding="utf-8")


def setup(app: Sphinx) -> dict[str, bool]:
    """Register small project-local build hooks."""
    app.connect("build-finished", _check_validation_sidebar)
    app.connect("build-finished", _write_sitemap)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
