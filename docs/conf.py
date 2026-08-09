"""Sphinx configuration for the pySHT documentation."""

from __future__ import annotations

from importlib.metadata import version

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
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autosummary_generate = True
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

html_theme = "pydata_sphinx_theme"
html_title = f"pySHT {release}"
html_static_path: list[str] = []
html_theme_options = {
    "github_url": "https://github.com/kisungyou/pySHT",
    "show_toc_level": 2,
}
