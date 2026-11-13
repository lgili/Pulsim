"""Refusals added after the review of the exact-path sensitivities: every
case where a gradient would have been silently approximate."""

import numpy as np
import pytest

import pulsim as p
from pulsim import sensitivity as sens

from . import _sens_reference as ref
from .test_sensitivity_exact import _buck, _pwm

T = ref.T


def test_duty_of_a_dead_time_pair_is_refused():
    # the duty moves the low side's rising edge too: not dtau/dD = T on
    # one edge
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("HS", "vin", "sw", ref.GON, ref.GOFF)
    b.add_switch("LS", "sw", "gnd", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    sf = p.make_dead_time_pwm_pair_fn(
        1 / T, 0.4, b.switch_index_of("HS"), b.switch_index_of("LS"),
        b.graph.num_switches, 1e-7)
    for name in ("HS.duty", "LS.duty"):
        with pytest.raises(p.UnsupportedError, match="alone") as e:
            p.simulate(b, 5 * T, engine="dsed", switch_fn=sf,
                       sensitivities=[name])
        assert e.value.feature == "sensitivity_parameter"
    # the element parameters of the same circuit are fine
    r = p.simulate(b, 5 * T, engine="dsed", switch_fn=sf,
                   sensitivities=["L.L"])
    assert np.isfinite(r.sens("vout", "L.L")).all()


def test_a_numeric_step_is_refused(monkeypatch):
    real = p._pulsim.run_ped_native_builder

    def one_numeric_step(*a, **kw):
        r = dict(real(*a, **kw))
        r["n_exact_steps"] = int(r["n_exact_steps"]) - 1
        return r

    monkeypatch.setattr(p._pulsim, "run_ped_native_builder",
                        one_numeric_step)
    b = _buck()
    with pytest.raises(p.UnsupportedError, match="numeric"):
        p.simulate(b, 5 * T, engine="dsed", switch_fn=_pwm(b),
                   sensitivities=["L.L"])


def test_moebius_check_is_per_row(monkeypatch):
    # a cubic (not linear-fractional) entry of 1e-3 in the recovery, next
    # to 1e12 1/s entries of A: one scale for the whole vector let it
    # through
    real = sens._snapshot

    def cubic(builder, masks, h_max, ic):
        out = real(builder, masks, h_max, ic)
        R = {c["name"]: c for c in builder.components()}["RL"]["params"][
            "R_ohms"]
        for key in masks:
            out[key] = out[key].copy()
            out[key][-1] += 1e-3 * (R / 50.0) ** 3
        return out

    monkeypatch.setattr(sens, "_snapshot", cubic)
    b = _buck()
    with pytest.raises(p.UnsupportedError, match="linear-fractional"):
        p.simulate(b, 5 * T, engine="dsed", switch_fn=_pwm(b),
                   sensitivities=["RL.R"])


def test_orbit_sensitivity_run_must_reproduce_the_solver():
    b = _buck()
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b))
    s = r._ctx
    s.orbit_traj.events = [e for e in s.orbit_traj.events
                           if e["name"] == "gate_edge"]   # drop the diode
    with pytest.raises(p.UnsupportedError, match="events"):
        sens.orbit_sensitivities(r, s, ["RL.R"])


def test_window_edge_on_a_moving_jump_is_refused():
    b = _buck()
    r = p.simulate(b, 10 * T, engine="dsed", switch_fn=_pwm(b),
                   sensitivities=["S.duty"])
    j = r.sensitivity.params.index("S.duty")
    edge = [e["t"] for e in r.sensitivity.event_dtau
            if e["name"] == "gate_edge" and e["dtau"][j] != 0][-1]
    with pytest.raises(p.UnsupportedError, match="one-sided"):
        r.kpi("sw", "avg", t_from=0.0, t_to=edge)
    with pytest.raises(p.UnsupportedError, match="one-sided"):
        r.kpi("sw", "final", t_to=edge)
    # a state is continuous there: allowed
    v, g = r.kpi("vout", "avg", t_from=0.0, t_to=edge)
    assert np.isfinite(g["S.duty"])


def test_orbit_ends_are_not_a_tie():
    b = _buck(R=5.0)                      # CCM: one valley per period
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b),
                                sensitivities=["L.L"])
    assert r.sensitivity.periodic
    for kind in ("peak", "valley", "ripple"):
        assert not r.kpi("L", kind).non_unique


def test_evaluator_drops_a_non_unique_kpi_gradient():
    from pulsim.design import Evaluator

    def build(d):
        b = _buck(L=d["L"])
        return b, _pwm(b)

    rec = Evaluator(build, period=T, output="vout",
                    gradients={"L": "L.L"}).evaluate({"L": 100e-6})
    assert rec.status == "ok"
    assert "ripple_i.L" not in rec.grads          # the DCM valley is flat
    assert "ripple_i.L" in rec.counters["grads_refused"]
    assert set(rec.grads) == {"v_out", "ripple_v"}


def test_evaluator_refuses_a_variable_that_feeds_two_elements():
    from pulsim.design import Evaluator

    def build(d):
        b = _buck(R=d["R"])
        b.add_resistor("Rx", "vout", "gnd", 20 * d["R"])
        return b, _pwm(b)

    ev = Evaluator(build, period=T, output="vout", gradients={"R": "RL.R"})
    with pytest.raises(ValueError, match="alone"):
        ev.evaluate({"R": 50.0})


def test_the_extrema_grid_brackets_a_fast_mode_peak():
    """Batch 6c review (b1): y = e^{-3e8 s} - 3 e^{-3e7 s} + 2.5 e^{-3e6 s}
    over a 10 us step peaks at 1.707 near 92 ns with y' < 0 at both ends
    of the first of 16 uniform cells: the uniform grid reported 0.5. The
    same grid is detail::extrema_grid in period_integrals.hpp."""
    from pulsim.sensitivity import _extrema_grid
    mu = np.array([-3e8, -3e7, -3e6])
    al = np.array([1.0, -3.0, 2.5])

    def dy(s):
        return float((al * mu * np.exp(mu * s)).sum())

    def brackets(ss):
        d = [dy(s) for s in ss]
        return [(a, b) for a, b, da, db in zip(ss[:-1], ss[1:], d[:-1], d[1:])
                if da > 0 > db]
    assert brackets(np.linspace(0.0, 1e-5, 17)) == []          # the miss
    br = brackets(_extrema_grid(0.0, 1e-5, 16, 3e8))
    assert len(br) == 1 and br[0][0] < 9.2e-8 < br[0][1], br
    # a slow step adds nothing: the grid is the uniform one
    assert np.array_equal(_extrema_grid(0.0, 1e-5, 16, 1e5),
                          np.array([1e-5 * c / 16 for c in range(17)]))
