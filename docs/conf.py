"""Sphinx configuration for the public Pulsim documentation site."""

from __future__ import annotations

import importlib.util
from datetime import date

project = "PulsimCore"
copyright = f"{date.today().year}, Pulsim Contributors"
author = "Pulsim Contributors"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "python/**",
    "refactor-python-only-v1-hardening/**",
    "mainpage.md",
    "convergence-algorithms.md",
    "determinism.md",
    "performance-tuning.md",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_title = "PulsimCore Docs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Keep local builds working even if the optional docs theme is not installed.
if importlib.util.find_spec("furo"):
    html_theme = "furo"
    html_theme_options = {
        "navigation_with_keys": True,
        "top_of_page_button": "edit",
        "light_css_variables": {
            "color-brand-primary": "#0F766E",
            "color-brand-content": "#0F766E",
            "color-api-name": "#0F766E",
        },
        "dark_css_variables": {
            "color-brand-primary": "#34D399",
            "color-brand-content": "#34D399",
            "color-api-name": "#34D399",
        },
    }
else:
    html_theme = "sphinx_rtd_theme"
