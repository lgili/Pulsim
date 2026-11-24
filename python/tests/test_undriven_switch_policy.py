"""Owner decisions 2.1 and 2.3 (update-number-changing-release):
- One undriven-switch policy via ``pulsim._plan.DEFAULT_UNDRIVEN_SWITCH_POLICY``
  (``"warn_closed"`` today; ``"refuse"`` at the release flip across every engine).
- One unified post-run voltage-sanity check (``_auto_check_voltage_sanity``) across
  ``pwl``, ``trbdf2``, and ``dsed``/``exact``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim import _plan

_TOOLS = Path(__file__).resolve().parents[2] / "benchmarks" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import next_defaults as nd  # noqa: E402
import next_defaults_pytest as nd_pytest  # noqa: E402


def _single_switch(form: str = "conductance") -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 12.0)
    if form == "branch":
        b.add_switch("Q_main", "in", "out", R_on=1e-3, g_off=1e-9, form="branch")
    else:
        b.add_switch("Q_main", "in", "out", 1e3, 1e-9)
    b.add_resistor("Rload", "out", "gnd", 10.0)
    return b


def _half_bridge_with_diode() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q_hi", "in", "sw", 1e3, 1e-9)
    b.add_diode("D_fw", "gnd", "sw", 1e3, 1e-9)
    b.add_switch("Q_lo", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rload", "sw", "gnd", 10.0)
    return b


def _diode_only(form: str = "conductance") -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 1e3, 0.0)
    if form == "branch":
        b.add_diode("D1", "ac", "out", R_on=1e-3, g_off=1e-9, form="branch")
    else:
        b.add_diode("D1", "ac", "out", 1e3, 1e-9)
    b.add_capacitor("C1", "out", "gnd", 10e-6)
    b.add_resistor("R1", "out", "gnd", 100.0)
    return b


def _passive_rc() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 5.0)
    b.add_resistor("R1", "in", "out", 1e3)
    b.add_capacitor("C1", "out", "gnd", 1e-6)
    return b


_ALL_ENGINE_CALLS = [
    pytest.param({"engine": "pwl", "dt": 1e-6}, "pwl", id="pwl"),
    pytest.param({"engine": "trbdf2"}, "trbdf2", id="trbdf2"),
    pytest.param({"engine": "dsed"}, "dsed", id="dsed"),
    pytest.param({"engine": "exact"}, "dsed", id="exact"),
    pytest.param({"engine": "auto", "dt": 1e-6}, "auto", id="auto-with-dt"),
    pytest.param({"engine": "auto"}, "auto", id="auto-no-dt"),
]


# ---------------------------------------------------------------------------
# Default policy ("warn_closed"): today's behaviour preserved unchanged
# ---------------------------------------------------------------------------

def test_default_policy_is_warn_closed():
    assert _plan.DEFAULT_UNDRIVEN_SWITCH_POLICY == "warn_closed"
    assert _plan.UNDRIVEN_SWITCH_POLICIES == ("warn_closed", "refuse")


@pytest.mark.parametrize("kw", [
    pytest.param({"engine": "pwl", "dt": 1e-6}, id="pwl"),
    pytest.param({"engine": "dsed"}, id="dsed"),
    pytest.param({"engine": "exact"}, id="exact"),
    pytest.param({"engine": "auto", "dt": 1e-6}, id="auto-with-dt"),
])
def test_warn_closed_runs_all_closed_with_warning(kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(_single_switch(), 1e-5, **kw)
    assert any("no switch_fn was given" in str(w.message) for w in caught)
    assert float(np.asarray(res.v("out"))[-1]) == pytest.approx(12.0, rel=2e-3)


def test_warn_closed_preserves_trbdf2_and_auto_no_dt_refusals():
    with pytest.raises(p.UnsupportedError, match="no switch_fn") as exc_trb:
        p.simulate(_single_switch(), 1e-5, engine="trbdf2")
    assert exc_trb.value.feature == "switch_fn"

    with pytest.raises(ValueError, match="no switch_fn"):
        p.simulate(_single_switch(), 1e-5, engine="auto")


# ---------------------------------------------------------------------------
# Release policy ("refuse"): uniform UnsupportedError across every engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw,expected_engine", _ALL_ENGINE_CALLS)
@pytest.mark.parametrize("form", ["conductance", "branch"])
def test_refuse_policy_rejects_undriven_controlled_switch_on_every_engine(
    monkeypatch, kw, expected_engine, form
):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN_SWITCH_POLICY", "refuse")
    with pytest.raises(p.UnsupportedError) as exc:
        p.simulate(_single_switch(form=form), 1e-5, **kw)
    err = exc.value
    assert err.feature == "undriven_switch"
    assert err.engine == expected_engine
    assert err.device == "Q_main"
    assert "Q_main" in str(err)


@pytest.mark.parametrize("kw,_expected_engine", _ALL_ENGINE_CALLS)
def test_refuse_policy_names_only_controlled_switches_not_diodes(
    monkeypatch, kw, _expected_engine
):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN_SWITCH_POLICY", "refuse")
    with pytest.raises(p.UnsupportedError) as exc:
        p.simulate(_half_bridge_with_diode(), 1e-5, **kw)
    msg = str(exc.value)
    assert exc.value.feature == "undriven_switch"
    assert "Q_hi" in msg and "Q_lo" in msg
    assert "D_fw" not in msg


@pytest.mark.parametrize("kw,_expected_engine", _ALL_ENGINE_CALLS)
def test_refuse_policy_allows_passive_circuits(monkeypatch, kw, _expected_engine):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN_SWITCH_POLICY", "refuse")
    res = p.simulate(_passive_rc(), 1e-5, **kw)
    assert len(res.times) > 1
    assert np.isfinite(res.v("out")).all()


@pytest.mark.parametrize("kw,_expected_engine", _ALL_ENGINE_CALLS)
@pytest.mark.parametrize("form", ["conductance", "branch"])
def test_refuse_policy_allows_diode_only_circuits(
    monkeypatch, kw, _expected_engine, form
):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN_SWITCH_POLICY", "refuse")
    res = p.simulate(_diode_only(form=form), 5e-5, **kw)
    assert len(res.times) > 1
    assert np.isfinite(res.v("out")).all()


@pytest.mark.parametrize("kw,_expected_engine", _ALL_ENGINE_CALLS)
def test_refuse_policy_allows_explicit_switch_fn(monkeypatch, kw, _expected_engine):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN_SWITCH_POLICY", "refuse")
    b = _single_switch()
    sf = p.make_pwm_switch_fn(
        frequency=100e3,
        duty=0.5,
        switch_idx=b.switch_index_of("Q_main"),
        num_switches=b.graph.num_switches,
    )
    res = p.simulate(b, 2e-5, switch_fn=sf, **kw)
    assert len(res.times) > 1
    assert np.isfinite(res.v("out")).all()


def test_refuse_policy_allows_closed_loop_drive(monkeypatch):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN_SWITCH_POLICY", "refuse")
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L1", "sw", "vout", 100e-6)
    b.add_capacitor("C1", "vout", "gnd", 47e-6)
    b.add_resistor("R1", "vout", "gnd", 10.0)
    pi = p.control.PIController(
        Kp=0.05, Ki=100.0, output_min=0.05, output_max=0.95,
    )
    loop = p.control.bind_pi_to_switch(
        b, pi=pi, measured="vout", setpoint=12.0, switch="Q1", freq=50e3,
    )
    res = p.simulate(b, 1e-4, dt=1e-6, engine="pwl", closed_loops=[loop])
    assert len(res.times) > 1
    assert np.isfinite(res.v("vout")).all()


def test_invalid_policy_raises_options_error(monkeypatch):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN_SWITCH_POLICY", "bogus")
    with pytest.raises(p.OptionsError, match="DEFAULT_UNDRIVEN_SWITCH_POLICY"):
        p.simulate(_single_switch(), 1e-5, dt=1e-6, engine="pwl")


# ---------------------------------------------------------------------------
# Next-defaults harness wiring & clean teardown
# ---------------------------------------------------------------------------

def test_next_defaults_policy_defaults_context_manager():
    assert nd.POLICY["DEFAULT_UNDRIVEN_SWITCH_POLICY"].value == "refuse"
    assert _plan.DEFAULT_UNDRIVEN_SWITCH_POLICY == "warn_closed"
    with nd.policy_defaults():
        assert _plan.DEFAULT_UNDRIVEN_SWITCH_POLICY == "refuse"
        with pytest.raises(p.UnsupportedError) as exc:
            p.simulate(_single_switch(), 1e-5, dt=1e-6, engine="pwl")
        assert exc.value.feature == "undriven_switch"
    assert _plan.DEFAULT_UNDRIVEN_SWITCH_POLICY == "warn_closed"


def test_next_defaults_pytest_install_and_uninstall_restore_policy():
    orig_sim = p.simulate
    orig_sw = p.CircuitBuilder.add_switch
    orig_di = p.CircuitBuilder.add_diode
    orig_policy = _plan.DEFAULT_UNDRIVEN_SWITCH_POLICY
    orig_restart = _plan.DEFAULT_DISCONTINUITY_RESTART
    orig_split = _plan.DEFAULT_SUBSTEP_STATE_CORRECTION
    try:
        nd_pytest._install()
        assert _plan.DEFAULT_UNDRIVEN_SWITCH_POLICY == "refuse"
        assert _plan.DEFAULT_DISCONTINUITY_RESTART is True
        assert _plan.DEFAULT_SUBSTEP_STATE_CORRECTION is True
        with pytest.raises(p.UnsupportedError) as exc:
            p.simulate(_single_switch(), 1e-5, engine="auto")
        assert exc.value.feature == "undriven_switch"
    finally:
        nd_pytest._uninstall()
    assert _plan.DEFAULT_UNDRIVEN_SWITCH_POLICY == orig_policy
    assert _plan.DEFAULT_DISCONTINUITY_RESTART == orig_restart
    assert _plan.DEFAULT_SUBSTEP_STATE_CORRECTION == orig_split
    assert p.simulate is orig_sim
    assert p.CircuitBuilder.add_switch is orig_sw
    assert p.CircuitBuilder.add_diode is orig_di


# ---------------------------------------------------------------------------
# Owner decision 2.3: unified post-run voltage-sanity check across engines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    pytest.param({"engine": "pwl", "dt": 1e-6}, id="pwl"),
    pytest.param({"engine": "trbdf2"}, id="trbdf2"),
    pytest.param({"engine": "exact"}, id="exact"),
])
def test_unified_voltage_sanity_warns_at_caller_frame_and_honours_werror(kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(_passive_rc(), 1e-5, voltage_sanity_factor=1e-9, **kw)
    hits = [w for w in caught if "largest voltage any source" in str(w.message)]
    assert len(hits) == 1
    assert hits[0].filename == __file__
    assert res._implausible_voltage.node >= 0

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="largest voltage any source"):
            p.simulate(_passive_rc(), 1e-5, voltage_sanity_factor=1e-9, **kw)
