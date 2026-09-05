"""SMPS topology factories — `add_buck` / `add_boost` / `add_flyback`.

These collapse 5–8 `add_*` calls into one. Tests cover three angles:

1. Smoke — the function builds the right number of branches and the
   returned `*Result` carries the deterministic device names.
2. End-to-end — a closed-loop PI controller drives the buck to its
   setpoint within the simulated window.
3. Validation — bad params raise `ValueError` cleanly.

The new top-level entry points (`p.add_buck`, etc.) live in
`pulsim.topology` and are re-exported from `pulsim`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# -----------------------------------------------------------------------
# Smoke tests
# -----------------------------------------------------------------------

def test_add_buck_builds_six_devices_with_deterministic_names() -> None:
    b = p.CircuitBuilder()
    out = p.add_buck(b, name="my_buck",
                       V_in=48.0, L=100e-6, C=100e-6, R_load=4.0,
                       f_sw=100e3)

    # 1 Vin + 1 switch + 1 diode + 1 inductor + 1 cap + 1 resistor = 6.
    assert b.graph.num_branches == 6
    assert out.switch_name == "my_buck__HS"
    assert out.diode_name == "my_buck__D"
    assert out.inductor_name == "my_buck__L"
    assert out.capacitor_name == "my_buck__C"
    assert out.load_name == "my_buck__R"
    assert out.switch_branch_id >= 0
    assert out.inductor_branch_id >= 0
    # 100 samples per period @ 100 kHz → 100 ns
    assert math.isclose(out.suggested_dt, 1.0 / (100.0 * 100e3),
                          rel_tol=1e-9)


def test_add_boost_builds_six_devices() -> None:
    b = p.CircuitBuilder()
    out = p.add_boost(b, name="my_boost",
                        V_in=12.0, L=100e-6, C=100e-6, R_load=50.0,
                        f_sw=100e3)
    assert b.graph.num_branches == 6
    assert out.switch_name == "my_boost__LS"
    assert out.diode_name == "my_boost__D"
    assert out.inductor_name == "my_boost__L"


def test_add_flyback_couples_primary_and_secondary() -> None:
    b = p.CircuitBuilder()
    out = p.add_flyback(b, name="fb", V_in=48.0,
                          L_pri=200e-6, L_sec=200e-6, k=0.99,
                          C=100e-6, R_load=10.0)
    # 1 Vin + 1 Lp + 1 LS + 1 Ls + 1 D + 1 C + 1 R = 7.
    assert b.graph.num_branches == 7
    assert out.primary_name == "fb__Lp"
    assert out.secondary_name == "fb__Ls"
    assert out.primary_branch_id != out.secondary_branch_id
    # The name says "couples"; this used to check only the counts
    # above, and the coupling call sat behind a `hasattr` for a
    # method that did not exist — so every flyback this factory built
    # had two INDEPENDENT inductors and the test was green. Now the
    # coupling is asserted where it shows: energy reaches the output.
    pwm = p.make_pwm_switch_fn(frequency=100e3, duty=0.4, switch_idx=0,
                               num_switches=b.graph.num_switches, phase=0.0)
    res = p.simulate(b, t_end=1e-3, dt=2e-8, switch_fn=pwm)
    v_out = float(np.asarray(res.v("vout"))[-1])
    assert v_out > 5.0, f"no transformer action: V_out = {v_out} V"


def test_add_inductor_coupling_refuses_by_name() -> None:
    b = p.CircuitBuilder()
    b.add_inductor("La", "a", "gnd", 1e-3)
    b.add_inductor("Lb", "b", "gnd", 1e-3)
    b.add_saturable_inductor("Ls", "c", "gnd", 1e-3, 1.0)
    b.add_resistor("R", "a", "b", 1.0)
    with pytest.raises(Exception, match=r"\[0, 1\]"):
        b.add_inductor_coupling("La", "Lb", 1.5)
    with pytest.raises(Exception, match="itself"):
        b.add_inductor_coupling("La", "La", 0.5)
    with pytest.raises(Exception, match="not a linear inductor"):
        b.add_inductor_coupling("La", "Ls", 0.5)
    with pytest.raises(Exception, match="not a linear inductor"):
        b.add_inductor_coupling("La", "R", 0.5)
    b.add_inductor_coupling("La", "Lb", 0.9)      # and the good one works


def test_add_buck_default_name_when_unspecified() -> None:
    b = p.CircuitBuilder()
    out = p.add_buck(b)  # default name="buck"
    assert out.switch_name == "buck__HS"


# -----------------------------------------------------------------------
# End-to-end: open-loop buck reaches expected v_out
# -----------------------------------------------------------------------

def test_buck_open_loop_50pct_duty_reaches_v_in_over_two() -> None:
    """50 % duty on a buck (V_in=48, L=100 µH, C=100 µF, R_load=4 Ω)
    settles to ~24 V — the textbook D · V_in answer for CCM."""
    b = p.CircuitBuilder()
    out = p.add_buck(b, V_in=48.0, L=100e-6, C=100e-6, R_load=4.0,
                      f_sw=100e3)

    F_SW = 100e3
    T_SW = 1.0 / F_SW
    DUTY = 0.5

    def pwm(t: float):
        m = p.SwitchStateMask(b.graph.num_switches)
        on = (t % T_SW) < (DUTY * T_SW)
        m.set(0, on)
        return m

    res = p.simulate(b, t_end=5e-3, dt=out.suggested_dt,
                       switch_fn=pwm)
    v_out_final = float(np.asarray(res.v("vout"))[-1])
    # Settling to ~D · V_in. Allow ±15 % because of switch / diode
    # conduction drops + finite simulation window.
    assert 0.85 * (DUTY * 48.0) < v_out_final < 1.15 * (DUTY * 48.0), (
        f"Open-loop buck v_out_final = {v_out_final:.2f} V; expected "
        f"≈ {DUTY * 48.0:.2f} V")


# -----------------------------------------------------------------------
# SimulationResult.plot smoke test (no mpl backend required)
# -----------------------------------------------------------------------

def test_result_plot_resolves_node_and_branch_names() -> None:
    """`res.plot('vout', 'L')` must not raise — drives the same
    name-resolution path as `res.v` / `res.i`."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    b = p.CircuitBuilder()
    p.add_buck(b, V_in=48.0)
    n = b.graph.num_switches
    res = p.simulate(b, t_end=1e-4, dt=1e-6,
                       switch_fn=lambda _t:
                           (lambda m: (m.set(0, True), m)[1])(
                               p.SwitchStateMask(n)))
    fig = res.plot("vout", show=False)
    assert fig is not None


def test_result_plot_without_builder_attr_raises() -> None:
    """If the result wasn't produced via `simulate()` and lacks a
    `_builder` back-reference, `res.plot(...)` must raise
    `AttributeError` with an actionable pointer to `pulsim.scope`."""
    b = p.CircuitBuilder()
    p.add_buck(b, V_in=48.0)
    n = b.graph.num_switches
    res = p.simulate(b, t_end=1e-4, dt=1e-6,
                       switch_fn=lambda _t:
                           (lambda m: (m.set(0, True), m)[1])(
                               p.SwitchStateMask(n)))
    # Strip the back-reference simulate() attached.
    if hasattr(res, "_builder"):
        del res._builder

    with pytest.raises(AttributeError, match="pulsim.scope"):
        res.plot("vout", show=False)


# -----------------------------------------------------------------------
# Validation — bad params raise loudly
# -----------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected", [
    ({"V_in": -1.0}, "V_in"),
    ({"L": 0.0}, "L"),
    ({"C": -1.0}, "C"),
    ({"R_load": 0.0}, "R_load"),
    ({"f_sw": 0.0}, "f_sw"),
])
def test_add_buck_validates_positive_params(kwargs, expected) -> None:
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=expected):
        p.add_buck(b, **kwargs)


@pytest.mark.parametrize("k", [-0.1, 0.0, 1.1, 2.0])
def test_add_flyback_validates_coupling(k) -> None:
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="k must be"):
        p.add_flyback(b, k=k)
