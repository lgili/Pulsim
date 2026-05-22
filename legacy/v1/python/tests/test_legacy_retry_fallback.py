# ruff: noqa: E402
"""Tests for refactor-pwl-switching-engine Phase 7.4 — Python retry-wrapper
deprecation gate.

The retry / auto-bleeder layer wrapped by `pulsim.run_transient` is the
legacy crutch for switching circuits that struggle through the Behavioral
Newton path. PWL `SwitchingMode.Ideal` makes it unnecessary. Phase 7.4
ships:

- An opt-out via env var ``PULSIM_LEGACY_RETRY_FALLBACK=0`` that bypasses
  the entire retry loop and runs a single transient pass.
- A ``DeprecationWarning`` emitted (once per process) on the first retry,
  pointing to ``docs/pwl-switching-migration.md``.

These tests pin both contracts without depending on a circuit actually
hitting a retry, which would require a non-deterministic failure mode.
"""

import warnings

import pytest

import pulsim as ps
from pulsim import (
    _emit_legacy_retry_deprecation_warning,
    _legacy_retry_fallback_enabled,
)
import pulsim as _pulsim_module  # type: ignore  # noqa: F401  (module attr access)


@pytest.fixture(autouse=True)
def _reset_deprecation_state(monkeypatch):
    """Each test starts with the deprecation warning re-armed and the env
    variable unset; the module-level flag is module-global so we must reset
    it manually between tests.
    """
    monkeypatch.delenv("PULSIM_LEGACY_RETRY_FALLBACK", raising=False)
    _pulsim_module._LEGACY_RETRY_DEPRECATION_WARNED = False
    yield
    _pulsim_module._LEGACY_RETRY_DEPRECATION_WARNED = False


def test_legacy_retry_fallback_enabled_default_true(monkeypatch):
    """Default (env unset) keeps the retry layer active for backward compat."""
    monkeypatch.delenv("PULSIM_LEGACY_RETRY_FALLBACK", raising=False)
    assert _legacy_retry_fallback_enabled() is True


def test_legacy_retry_fallback_enabled_env_one(monkeypatch):
    """Explicit "1" keeps the retry layer (same as unset)."""
    monkeypatch.setenv("PULSIM_LEGACY_RETRY_FALLBACK", "1")
    assert _legacy_retry_fallback_enabled() is True


def test_legacy_retry_fallback_enabled_env_zero(monkeypatch):
    """Explicit "0" disables the retry layer."""
    monkeypatch.setenv("PULSIM_LEGACY_RETRY_FALLBACK", "0")
    assert _legacy_retry_fallback_enabled() is False


def test_deprecation_warning_fires_once():
    """`_emit_legacy_retry_deprecation_warning` raises a DeprecationWarning
    on first call and a no-op on subsequent calls within the same process."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)

        _emit_legacy_retry_deprecation_warning()
        _emit_legacy_retry_deprecation_warning()
        _emit_legacy_retry_deprecation_warning()

    deprecation_warnings = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    msg = str(deprecation_warnings[0].message)
    assert "PULSIM_LEGACY_RETRY_FALLBACK" in msg
    assert "pwl-switching-migration" in msg


def test_run_transient_env_opt_out_single_pass(monkeypatch):
    """With PULSIM_LEGACY_RETRY_FALLBACK=0, `run_transient` runs a single
    pass and does not install auto-bleeders even when the circuit topology
    would otherwise trigger them (we check the bleeder registry membership).
    """
    monkeypatch.setenv("PULSIM_LEGACY_RETRY_FALLBACK", "0")

    ckt = ps.Circuit()
    gnd = ps.Circuit.ground()
    n1 = ckt.add_node("n1")
    ckt.add_voltage_source("V1", n1, gnd, 5.0)
    ckt.add_resistor("R1", n1, gnd, 1000.0)

    # Sanity: bleeder registry starts not containing this circuit
    assert ckt not in _pulsim_module._AUTO_BLEEDER_CIRCUITS

    result = ps.run_transient(ckt, 0.0, 1e-6, 1e-7)
    # Tuple shape preserved (t, x, success, message)
    assert isinstance(result, tuple)
    assert len(result) == 4
    success = result[2]
    assert success in (True, False)

    # Crucial: opt-out path must NOT install auto-bleeders
    assert ckt not in _pulsim_module._AUTO_BLEEDER_CIRCUITS


def test_run_transient_legacy_path_no_warning_on_first_pass(monkeypatch):
    """A simple circuit that succeeds on the first attempt does not trigger
    the DeprecationWarning, even under the legacy retry path. Only retries
    emit the warning.
    """
    monkeypatch.setenv("PULSIM_LEGACY_RETRY_FALLBACK", "1")

    ckt = ps.Circuit()
    gnd = ps.Circuit.ground()
    n1 = ckt.add_node("n1")
    ckt.add_voltage_source("V1", n1, gnd, 5.0)
    ckt.add_resistor("R1", n1, gnd, 1000.0)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        ps.run_transient(ckt, 0.0, 1e-6, 1e-7)

    legacy_warnings = [
        w for w in captured
        if issubclass(w.category, DeprecationWarning)
        and "PULSIM_LEGACY_RETRY_FALLBACK" in str(w.message)
    ]
    assert legacy_warnings == [], (
        "Successful first-pass run should not emit the retry deprecation warning"
    )
