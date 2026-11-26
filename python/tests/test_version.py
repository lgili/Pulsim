"""The release version is one string: package, compiled module, pyproject."""

import pathlib
import re

import pulsim
from pulsim import _pulsim


def test_compiled_module_and_package_report_one_version():
    # The compiled module said 1.6.4 through 2.0.0: update_version.py now covers it.
    assert _pulsim.__version__ == pulsim.__version__


def test_pyproject_version_matches_package():
    pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():  # an installed wheel tested outside the tree
        return
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"$', pyproject.read_text())
    assert m and m.group(1) == pulsim.__version__
