"""Shared notebook-cell helpers for project notebook generators.

Each ``_build_pulsim_validation.py`` script imports ``md``, ``code``,
and ``write_notebook`` from here so the boilerplate isn't repeated
8 times. The existing ``_build_notebooks.py`` modeling generators
still inline their own copies for backward compatibility — this
module is for the new Pulsim validation notebooks only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def md(text: str) -> dict[str, Any]:
    """Markdown cell."""
    return {"cell_type": "markdown", "metadata": {},
            "source": _split(text)}


def code(text: str) -> dict[str, Any]:
    """Code cell with empty outputs (filled in by ``nbconvert --execute``)."""
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": _split(text)}


def _split(text: str) -> list[str]:
    text = text.lstrip("\n")
    return text.splitlines(keepends=True)


def write_notebook(cells: list[dict[str, Any]], path: Path) -> None:
    """Serialise the cells as a Jupyter notebook on disk."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                            "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
