# Pulsim v1 — Legacy Archive

This directory holds the **v1 user-facing surface** (Python tests +
examples) that was moved out of the active tree as part of the
v1 → v2 retirement effort (Phase 1).

## Why it's here

The v1 kernel itself (``core/include/pulsim/v1/runtime_circuit.hpp``)
and its C++ test suite still live in their original locations —
they're built by CMake and exposed to Python via the legacy binding
path. Removing those requires CMake surgery + ``__init__.py``
rewrite, scheduled for a later phase.

The Python tests and pure-Python examples gathered here used the
top-level ``import pulsim`` surface. They:

* tested features that v2 doesn't have (codegen, FMU, templates,
  presets, robustness profiles, …),
* exercised v1-specific APIs (``RuntimeCircuit``, ``vcswitch``,
  ``PulseParams``, ``add_node`` returning node IDs, …),
* or duplicated work that now has a v2 equivalent under
  ``python/tests/v2/`` and ``examples/v2/scripts/``.

Moving them out of ``python/tests/`` removes them from pytest's
auto-discovery path (configured to ``testpaths = ["python/tests"]``
in ``pyproject.toml``) without deleting any history — every move is
a ``git mv`` so the file history follows.

## Layout

```
legacy/v1/
├── python/
│   └── tests/
│       ├── test_*.py          (51 top-level v1 tests)
│       ├── validation/        (level1..5 + framework, ~17 files)
│       └── properties/        (KCL / passivity / energy property tests)
└── examples/
    └── python/                (16 v1 numerical examples)
```

## How to restore (if a future regression points back here)

```bash
git mv legacy/v1/python/tests/test_foo.py python/tests/test_foo.py
```

The migration guide at ``docs/v2/migration-from-v1.md`` documents
the v1 → v2 API mapping for anyone porting a specific test or
example forward.

## What's NOT here yet

| Item | Where | Why kept in place |
|------|-------|-------------------|
| v1 C++ kernel header | ``core/include/pulsim/v1/runtime_circuit.hpp`` | Still used by ``python/pulsim/__init__.py`` top-level binding |
| v1 C++ tests | ``core/tests/test_*.cpp`` | CMakeLists.txt explicitly lists ~89 files; move = surgery |
| ``python/pulsim/__init__.py`` | ``python/pulsim/__init__.py`` | Currently exposes v1 binding; needs rewrite to point at v2 |
| Internal build scripts | ``projects/*/_build_notebooks.py`` | Regenerate notebooks; keep until notebooks are fully migrated |

These are the targets for **Phase 2** (selective ``git rm`` of
unused test infrastructure) and **Phase 3** (kernel removal +
``__init__.py`` rewrite). See ``docs/v2/migration-from-v1.md`` §
"Known gaps" for which v1 features are deliberately not in v2.
