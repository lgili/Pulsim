"""add-immutable-circuit-ir 3.4: no module keeps side state on a builder.

A Circuit is a snapshot of the C++ builder; anything a module stores as a
Python attribute on a builder instance (``builder._x = ...``,
``setattr(builder, ...)``) is invisible to it, so ``build()``,
``simulate(circuit)``, ``to_yaml()`` and a pickle would drop it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pulsim

PKG = Path(pulsim.__file__).resolve().parent

# `<name containing builder>._attr = ...` / `b._attr = ...` and setattr on
# one. A result's `_builder` (an attribute OF a result) and a method
# installed on the class (`builder_cls._x = f`) do not match.
_OBJ = r"(?!\w*_cls\b)(\w*builder\w*|b)"
ASSIGN = re.compile(r"\b" + _OBJ + r"\._\w+\s*(\[[^\]]*\])?\s*=(?!=)")
SETATTR = re.compile(r"setattr\(\s*" + _OBJ + r"\s*,\s*['\"]_")
# Copying a builder's __dict__ onto another (the side-state hand-over).
DICT = re.compile(r"\b" + _OBJ + r"\.__dict__\.update\(|\bvars\(\s*" + _OBJ + r"\s*\)")

# c_block.py (`_c_blocks`) and blockchain.py (`_graph_driven`) are being
# consolidated onto the control graph on another branch (DECISIONS 14);
# `build()` turns both into ControlDecls, so a Circuit carries them
# meanwhile. Delete these entries when that lands.
ALLOWED = {("c_block.py", "_c_blocks"), ("blockchain.py", "_graph_driven")}
# Any other receiver name escapes the patterns; the check is a tripwire
# for the forms this package used, not a proof.


def test_no_module_sets_a_private_attribute_on_a_builder():
    found = []
    for path in sorted(PKG.rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if DICT.search(code):
                found.append(f"{path.relative_to(PKG)}:{n}: {line.strip()}")
            for m in (*ASSIGN.finditer(code), *SETATTR.finditer(code)):
                attr = re.search(r"\._(\w+)|['\"]_(\w+)", m.group(0))
                name = "_" + next(g for g in attr.groups() if g)
                if (str(path.relative_to(PKG)), name) not in ALLOWED:
                    found.append(f"{path.relative_to(PKG)}:{n}: {line.strip()}")
    assert found == []


def test_the_check_sees_an_assignment():
    """Control: the pattern matches the form it forbids."""
    assert ASSIGN.search("    builder._pulsim_ic = d")
    assert SETATTR.search("setattr(builder, '_x', 1)")
    assert not ASSIGN.search("res._builder = builder")
    assert not ASSIGN.search("builder_cls._snapshot = native")
    assert DICT.search("builder.__dict__.update(side)")
