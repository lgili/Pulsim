"""Regression: `import pulsim` must work even when PyYAML isn't
installed.

PyYAML lives under `pyproject.toml`'s `[project.optional-dependencies]
dev` extra — it's NOT a runtime dependency. Prior to v1.6.1,
`pulsim.yaml_chain` did a top-level `import yaml as _yaml`, which
broke the cibuildwheel smoke test (`python -c "import pulsim"`) on
every platform and made the PyPI publish of v1.6.0 fail.

We pin the contract two ways:

1. **Subprocess test**: spawn a Python interpreter with a
   bytecode-monkey-patched `import` hook that pretends `yaml` isn't
   installed, and verify `import pulsim` succeeds.
2. **In-process test**: install the same `__import__` hook in the
   running interpreter, force a fresh re-import of `pulsim.yaml_chain`,
   and confirm the module loads + `wire_chain_from_yaml(..., list)`
   still works without PyYAML.

The actionable-error path — calling `wire_chain_from_yaml(...,
'yaml_string')` without PyYAML — is also pinned: must raise
`ModuleNotFoundError` pointing at `pulsim[dev]`.
"""
from __future__ import annotations

import builtins
import importlib
import subprocess
import sys
import textwrap


# ---------------------------------------------------------------------------
# Subprocess: fresh interpreter, no PyYAML in scope.
# ---------------------------------------------------------------------------
def test_import_pulsim_without_pyyaml_in_subprocess() -> None:
    """Spawn a fresh Python and block `import yaml`. `import pulsim`
    must still succeed — this is the exact scenario cibuildwheel
    runs after building each wheel."""
    script = textwrap.dedent("""
        import builtins, sys
        _real = builtins.__import__
        def _no_yaml(name, *a, **kw):
            if name == 'yaml' or name.startswith('yaml.'):
                raise ModuleNotFoundError("No module named 'yaml'")
            return _real(name, *a, **kw)
        builtins.__import__ = _no_yaml

        import pulsim
        assert hasattr(pulsim, 'wire_chain_from_yaml')
        sys.stdout.write(f'OK {pulsim.__version__}')
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"`import pulsim` failed without PyYAML:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    # In editable installs, scikit-build-core prepends rebuild output
    # to stdout. The contract we care about is that the marker line is
    # present somewhere — not that it's the first line.
    assert "OK " in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# In-process: drive `wire_chain_from_yaml` paths with PyYAML blocked.
# ---------------------------------------------------------------------------
def _install_no_yaml_hook():
    """Return (real_import, restore_fn) and install a hook that
    blocks `import yaml`."""
    real = builtins.__import__

    def _no_yaml(name, *a, **kw):
        if name == "yaml" or name.startswith("yaml."):
            raise ModuleNotFoundError("No module named 'yaml'")
        return real(name, *a, **kw)

    builtins.__import__ = _no_yaml
    return real


def _restore_import(real_import):
    builtins.__import__ = real_import


def test_wire_chain_from_yaml_with_list_works_without_pyyaml() -> None:
    """When the chain spec is a Python list, PyYAML is never
    touched — must work even with `yaml` blocked."""
    real = _install_no_yaml_hook()
    try:
        # Pop any cached `yaml` so the hook actually fires on next
        # import. `pulsim.yaml_chain` is loaded already, but the
        # lazy import is deferred to the function body — exactly
        # the contract we want.
        sys.modules.pop("yaml", None)
        # Re-import pulsim.yaml_chain to confirm module loads cleanly.
        import pulsim.yaml_chain as yc
        importlib.reload(yc)

        import pulsim as p
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 0.0)
        b.add_resistor("R", "a", "gnd", 1.0)

        loaded = type("Loaded", (), {"builder": b})()
        chain = p.wire_chain_from_yaml(loaded, [])  # empty list path
        # Just confirm we got a CxxBlockChain back; concrete type
        # name is "CxxBlockChain" but we keep the assertion loose.
        assert type(chain).__name__ == "CxxBlockChain"
    finally:
        _restore_import(real)


def test_wire_chain_from_yaml_string_raises_friendly_error() -> None:
    """When the chain spec is a YAML *string* and PyYAML is missing,
    the user gets `ModuleNotFoundError` pointing at the `dev`
    extra — not a confusing `_yaml` NameError."""
    real = _install_no_yaml_hook()
    try:
        sys.modules.pop("yaml", None)
        import pulsim.yaml_chain as yc
        importlib.reload(yc)

        import pulsim as p
        try:
            p.wire_chain_from_yaml(object(), "- type: foo")
        except ModuleNotFoundError as exc:
            msg = str(exc)
            assert "pulsim[dev]" in msg or "PyYAML" in msg, msg
        else:
            raise AssertionError(
                "wire_chain_from_yaml('...yaml string...') should raise "
                "ModuleNotFoundError when PyYAML is missing")
    finally:
        _restore_import(real)
