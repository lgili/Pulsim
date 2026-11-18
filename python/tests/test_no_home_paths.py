"""Committed notebooks and docs carry no absolute home-directory path
(a re-run notebook stored /Users/<name>/... in its outputs). openspec/
probe logs are working records and are left out."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOME_PATH = re.compile(r"/(Users|home)/[A-Za-z0-9_.-]+/")


def test_no_committed_notebook_or_doc_has_a_home_path():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "*.ipynb", "*.md", "*.rst",
         ":!openspec"], capture_output=True, text=True, check=True).stdout.split()
    hits = [f"{f}:{n}" for f in files if (ROOT / f).is_file()
            for n, line in enumerate((ROOT / f).read_text(errors="ignore").splitlines(), 1)
            if HOME_PATH.search(line)]
    assert not hits, "absolute home paths (make them relative or clear the output):\n" \
        + "\n".join(hits)
