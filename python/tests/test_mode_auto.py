"""add-complementarity-mode-selection: ``mode_selection='auto'`` — 'lcp'
wherever it applies, the legacy settle where 'lcp' would refuse, the choice
in ``result.diagnostics['mode_selection_auto']``. Each fallback reason is
pinned to the explicit 'lcp' refusal it stands for."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import UnsupportedError


def _rect(**diode):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 1e3)
    b.add_diode("D", "a", "k", 1e3, 1e-9, **diode)
    b.add_capacitor("C", "k", "gnd", 1e-5)
    b.add_resistor("R", "k", "gnd", 50.0)
    return b


def _auto(r):
    return r.diagnostics["mode_selection_auto"]


def test_auto_equals_lcp_where_lcp_applies():
    kw = dict(t_end=2e-3, dt=1e-6, engine="pwl")
    ra = p.simulate(_rect(), mode_selection="auto", **kw)
    rl = p.simulate(_rect(), mode_selection="lcp", **kw)
    assert _auto(ra) == {"requested": "auto", "chosen": "lcp", "reason": None}
    # the same run (one code path): identical on this machine
    assert np.array_equal(np.asarray(ra.states), np.asarray(rl.states))
    assert ra.diagnostics["mode_selection"] == rl.diagnostics["mode_selection"]


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "k", 1.0)
    b.add_capacitor("C", "k", "gnd", 1e-6)
    return b


def _newton():
    b = _rect()
    b.add_shockley_diode("DS", "k", "gnd")
    return b


def _gated():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 1e3)
    b.add_thyristor("T", "a", "k")
    b.add_diode("D", "gnd", "k", 1e3, 1e-9)
    b.add_resistor("R", "k", "gnd", 5.0)
    return b


@pytest.mark.parametrize("make,reason", [
    (_rc, "no_diodes"),
    (_newton, "newton"),
    (_gated, "gated (T)"),
])
def test_circuit_fallbacks_match_the_lcp_refusals(make, reason):
    # the thyristor's gate is undriven on purpose (the subject is the mode rule)
    kw = dict(t_end=1e-4, dt=1e-6, engine="pwl", undriven="closed")
    with pytest.raises(UnsupportedError) as ei:
        p.simulate(make(), mode_selection="lcp", **kw)
    assert ei.value.feature == "mode_selection"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        ra = p.simulate(make(), mode_selection="auto", **kw)
    # a circuit reason is recorded, not warned about
    assert not any("mode_selection='auto'" in str(w.message) for w in seen)
    assert _auto(ra) == {"requested": "auto", "chosen": "legacy", "reason": reason}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = p.simulate(make(), mode_selection="legacy", **kw)   # the settle, named
    assert np.array_equal(np.asarray(ra.states), np.asarray(rd.states))


def test_max_event_iterations_zero_warns():
    kw = dict(t_end=1e-4, dt=1e-6, engine="pwl", max_event_iterations=0)
    with pytest.raises(UnsupportedError):
        p.simulate(_rect(), mode_selection="lcp", **kw)
    with pytest.warns(UserWarning, match="max_event_iterations=0"):
        ra = p.simulate(_rect(), mode_selection="auto", **kw)
    assert _auto(ra)["chosen"] == "legacy"


def test_trbdf2_falls_back_to_the_settle():
    # explicit 'lcp' / 'shadow' run on trbdf2; only 'auto' has no rule there
    rl = p.simulate(_rect(), t_end=1e-4, engine="trbdf2", mode_selection="lcp")
    assert rl.diagnostics["mode_selection"]["lcp_count"] > 0
    rs = p.simulate(_rect(), t_end=1e-4, engine="trbdf2", mode_selection="shadow")
    assert rs.diagnostics["mode_shadow"]["lcp_count"] > 0
    ra = p.simulate(_rect(), t_end=1e-4, engine="trbdf2", mode_selection="auto")
    assert _auto(ra) == {"requested": "auto", "chosen": "legacy",
                         "reason": "engine 'trbdf2' (no 'auto' rule there yet)"}


def test_dsed_auto():
    ra = p.simulate(_rect(), t_end=1e-3, engine="dsed", mode_selection="auto")
    rl = p.simulate(_rect(), t_end=1e-3, engine="dsed", mode_selection="lcp")
    assert _auto(ra)["chosen"] == "lcp"
    assert np.array_equal(np.asarray(ra.states), np.asarray(rl.states))


def test_auto_is_validated_and_the_default():
    with pytest.raises(ValueError, match="'auto'"):
        p.simulate(_rect(), t_end=1e-5, dt=1e-6, engine="pwl", mode_selection="Auto")
    # the release default (DECISIONS F5); 'legacy' is the settle, no auto record
    r = p.simulate(_rect(), t_end=1e-5, dt=1e-6, engine="pwl")
    assert _auto(r) == {"requested": "auto", "chosen": "lcp", "reason": None}
    r = p.simulate(_rect(), t_end=1e-5, dt=1e-6, engine="pwl", mode_selection="legacy")
    assert "mode_selection_auto" not in r.diagnostics


def test_sensitivities_record_the_fallback():
    r = p.simulate(_rc(), t_end=1e-5, engine="dsed", sensitivities=["R.R"],
                   mode_selection="auto")
    assert _auto(r) == {"requested": "auto", "chosen": "legacy", "reason": "sensitivities"}
