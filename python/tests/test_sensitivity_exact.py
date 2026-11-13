"""add-parameter-sensitivities, groups 2 and 3: the exact path.

Transient sensitivities of linear-element parameters and of the duty on
the ``dsed`` exact stepper (Daleckii-Krein with phi1 divided
differences, the block exponential on a near-defective mode, the jump
terms at diode events and at the duty's gate edges), and the
steady-state gradients ``dx*/dp = (I - M)^-1 dP/dp``.

Oracles: the closed form of the RC step, the prototype table
(``_sens_reference``), and central differences of the same exact map
(the finite-difference harness, step chosen by its Richardson error).
"""

import math
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import sensitivity as sens

from . import _sens_reference as ref

T = ref.T


def _buck(L=100e-6, R=50.0, C=47e-6, Vin=24.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", Vin)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", C)
    b.add_resistor("RL", "vout", "gnd", R)
    return b


def _pwm(b, D=0.3):
    return p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"),
                                b.graph.num_switches)


_DEFAULTS = dict(L=100e-6, R=50.0, C=47e-6, Vin=24.0, D=0.3)
_KEY = {"L.L": "L", "RL.R": "R", "C.C": "C", "Vin.V": "Vin",
        "S.duty": "D"}


def _buck_run(names, **kw):
    v = dict(_DEFAULTS, **kw)
    D = v.pop("D")
    b = _buck(**v)
    return p.simulate(b, 40 * T, engine="dsed", switch_fn=_pwm(b, D),
                      sensitivities=names)


@pytest.fixture(scope="module")
def dcm():
    return _buck_run(["L.L", "RL.R", "S.duty", "C.C", "Vin.V"])


# ---------------------------------------------------------------------
# RC: the closed form
# ---------------------------------------------------------------------

def test_rc_step_against_the_closed_form():
    V, R, C = ref.RC["V"], ref.RC["R"], ref.RC["C"]
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", V)
    b.add_resistor("R1", "in", "o", R)
    b.add_capacitor("C1", "o", "gnd", C)
    res = p.simulate(b, R * C, engine="dsed",
                     sensitivities=["R1.R", "C1.C"])
    dR, dC = res.sens("o", "R1.R"), res.sens("o", "C1.C")
    assert dR[-1] == pytest.approx(ref.RC_DV_DR, rel=1e-9)
    assert dC[-1] == pytest.approx(ref.RC_DV_DC, rel=1e-9)
    t = np.asarray(res.times)
    exact = -V * t / (R * R * C) * np.exp(-t / (R * C))
    k = t > 0
    assert np.abs(dR[k] / exact[k] - 1).max() < 1e-9
    assert res.sensitivity.states.shape == (len(t), 1, 2)


# ---------------------------------------------------------------------
# DCM buck transient (task 2.4)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name,key", [("L.L", "L"), ("RL.R", "R"),
                                      ("S.duty", "D")])
def test_dcm_transient_matches_the_prototype_table(dcm, name, key):
    s = dcm.sens("vout", name)[-1]
    assert s == pytest.approx(ref.DCM_TRANSIENT[key], rel=1e-7)


@pytest.mark.parametrize("name", ["L.L", "RL.R", "S.duty", "C.C",
                                  "Vin.V"])
def test_dcm_transient_equals_a_central_difference(dcm, name):
    key = _KEY[name]

    def vend(v):
        r = _buck_run(["L.L"], **{key: v})
        return float(np.asarray(r.v("vout"))[-1])

    fd, err, _h = ref.central_difference(vend, _DEFAULTS[key],
                                         rel_steps=(1e-3, 1e-4))
    s = dcm.sens("vout", name)[-1]
    assert abs(s - fd) <= 1e-7 * abs(fd), (s, fd, err)


def test_duty_has_no_sensitivity_without_the_gate_edge_jump(monkeypatch):
    monkeypatch.setattr(sens, "_duty_period", lambda *a: (0.0, 0.3))
    r = _buck_run(["S.duty"])
    assert np.abs(r.sensitivity.states).max() == 0.0


def test_one_run_and_the_event_terms(dcm):
    c = dcm.sensitivity.counters
    assert c["n_engine_runs"] == 1
    assert c["n_sens_propagations"] == 5 * (len(dcm.times) - 1)
    # every diode turn-off carries dtau/dp; the duty moves gate edges by T
    offs = [e for e in dcm.sensitivity.event_dtau if "turns OFF" in e["name"]]
    assert offs and all(np.all(np.isfinite(e["dtau"])) for e in offs)
    j = dcm.sensitivity.params.index("S.duty")
    gate = [e["dtau"][j] for e in dcm.sensitivity.event_dtau
            if e["name"] == "gate_edge"]
    assert set(np.round(np.asarray(gate) / T, 12)) == {0.0, 1.0}


def test_the_builder_is_restored_bit_for_bit():
    b = _buck()
    plain = p.simulate(b, 10 * T, engine="dsed", switch_fn=_pwm(b))
    p.simulate(b, 10 * T, engine="dsed", switch_fn=_pwm(b),
               sensitivities=["L.L", "RL.R", "C.C", "Vin.V"])
    again = p.simulate(b, 10 * T, engine="dsed", switch_fn=_pwm(b))
    assert np.array_equal(np.asarray(plain.states), np.asarray(again.states))
    assert np.array_equal(np.asarray(plain.times), np.asarray(again.times))


# ---------------------------------------------------------------------
# Near-defective and close eigenvalues: series RLC
# ---------------------------------------------------------------------

def _rlc(R, L=1e-3, C=1e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", R)
    b.add_inductor("L", "a", "o", L)
    b.add_capacitor("C", "o", "gnd", C)
    return b


R_CRIT = 2 * math.sqrt(1e-3 / 1e-6)


@pytest.mark.parametrize("rel,tier", [(0.0, 2), (1e-3, 1), (0.5, 1)])
def test_rlc_near_critical_damping_equals_central_differences(rel, tier):
    R0 = R_CRIT * (1 + rel)
    names = ["R.R", "L.L", "C.C"]
    r = p.simulate(_rlc(R0), 2e-4, engine="dsed", sensitivities=names)
    assert set(r.sensitivity.counters["tiers"].values()) == {tier}
    for name, v0 in zip(names, (R0, 1e-3, 1e-6)):
        def vend(v, name=name):
            kw = {"R": R0, "L": 1e-3, "C": 1e-6}
            kw[name[0]] = v
            res = p.simulate(_rlc(kw["R"], kw["L"], kw["C"]), 2e-4,
                             engine="dsed", sensitivities=["R.R"])
            return float(np.asarray(res.v("o"))[-1])

        fd, err, _h = ref.central_difference(vend, v0,
                                             rel_steps=(1e-3, 1e-4))
        s = r.sens("o", name)[-1]
        assert abs(s - fd) <= 1e-7 * abs(fd), (name, s, fd, err)


def test_eigenbasis_and_block_exponential_agree():
    """The two propagators on one regular mode (kappa ~ 1)."""
    rng = np.random.default_rng(7)
    n1 = 4
    Ab = np.zeros((n1, n1))
    Ab[:3, :3] = -np.diag([1.0, 2.0, 5.0]) + 0.3 * rng.standard_normal((3, 3))
    Ab[:3, 3] = rng.standard_normal(3)
    dAb = np.zeros((2, n1, n1))
    dAb[:, :3, :] = rng.standard_normal((2, 3, n1))
    m1 = sens._Mode("m", Ab, dAb, np.ones(n1), 1.0)
    m2 = sens._Mode("m", Ab, dAb, np.ones(n1), 1.0)
    assert m1.tier == 1
    xb = np.append(rng.standard_normal(3), 1.0)
    Sb = np.vstack([rng.standard_normal((3, 2)), np.zeros((1, 2))])
    for cs in ([1.0, 1.0], [2.0, 0.5]):     # the balancing is a similarity
        m2.tier, m2.cs, m2._pade = 2, np.array(cs), {}
        for h in (1e-3, 0.37, 2.0):
            a, b = m1.advance(xb, Sb, h), m2.advance(xb, Sb, h)
            assert np.allclose(a[0], b[0], rtol=1e-12, atol=1e-13)
            assert np.allclose(a[1], b[1], rtol=1e-11, atol=1e-12)


def test_divided_differences_are_continuous_and_exact():
    a = np.array([0.7, -3.0, -4e6, 0.2 + 30j])
    for eps in (0.0, 1e-12, 1e-6, 0.999e-3, 1.001e-3):
        got = sens._dd_phi1(a, a + eps * np.maximum(1, np.abs(a.real)))
        m = a
        want = (np.exp(m) * (m - 1) + 1) / m ** 2       # phi1'(a)
        assert np.all(np.abs(got / want - 1) < 1e-2 * max(eps, 1e-300) * 1e3
                      + 1e-9)
    # e^a phi1(b - a) on the decaying side: no overflow, no NaN
    v = sens._dd_exp(np.array([-5e6, 0.0]), np.array([0.0, -5e6]))
    assert np.all(np.isfinite(v)) and v[0] == pytest.approx(1 / 5e6)


# ---------------------------------------------------------------------
# Events: grazing refused, near misses flagged
# ---------------------------------------------------------------------

class _FakeCtx:
    def __init__(self, fm, fp):
        self._f = {"old": fm, "new": fp}
        self.diodes = {7: dict(name="D9", nodes=(0, -1))}

    def f(self, key, x):
        return self._f[key]

    def pair_row(self, key, a, k):
        return np.array([1.0, 0.0, 0.0]), np.zeros((1, 3))

    def diode_of(self, name):
        return self.diodes[7]


class _Key(str):
    def get(self, bit):
        return False


def test_grazing_event_is_refused():
    ctx = _FakeCtx(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    ev = dict(t=1e-6, name="D9 (branch #7) turns OFF", type=2,
              old=_Key("old"), new=_Key("new"))
    with pytest.raises(p.UnsupportedError) as ei:
        sens._jump(ctx, ev, np.zeros(2), np.zeros((2, 1)), [None])
    assert ei.value.feature == "grazing_event"
    assert ei.value.device == "D9"
    # transversal: the saltation of D3
    ctx = _FakeCtx(np.array([2.0, 1.0]), np.array([0.0, 1.0]))
    S1, dtau = sens._jump(ctx, ev, np.zeros(2), np.array([[3.0], [0.0]]),
                          [None])
    assert dtau[0] == pytest.approx(-1.5)
    assert np.allclose(S1[:, 0], [3.0 + 2.0 * -1.5, 0.0])


@pytest.fixture(scope="module")
def boundary_R():
    """The load bisected to the CCM/DCM boundary (CCM side)."""
    def valley(R):
        b = _buck(R=R)
        r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b))
        x = r.orbit
        return float(np.min(np.asarray(x.i("L"))))

    lo, hi = 20.0, 30.0       # CCM at 20 Ohm, DCM at 30 Ohm
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if valley(mid) > 1e-7 else (lo, mid)
    return lo


def test_dcm_ccm_boundary_is_flagged_as_a_near_miss(boundary_R):
    """At the CCM/DCM boundary the inductor current touches zero at its
    valley without crossing: the gradient is one-sided there."""
    b = _buck(R=boundary_R)
    with pytest.warns(UserWarning, match="near miss"):
        r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b),
                                    sensitivities=["RL.R"])
    flags = r.sensitivity.flags
    assert flags and flags[0]["device"] == "D1"
    # away from the boundary nothing is flagged
    b = _buck(R=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b),
                                    sensitivities=["RL.R"])
    assert r.sensitivity.flags == []


def test_near_misses_are_also_scanned_on_block_exponential_modes(
        boundary_R):
    # the same boundary orbit with every mode on the tier-2 propagator:
    # the scan must not skip a mode without an eigenbasis
    b = _buck(R=boundary_R)
    with pytest.warns(UserWarning, match="near miss"):
        r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b),
                                    sensitivities=["RL.R"])
    sr = r.sensitivity
    for m in sr._ctx.modes.values():
        m.tier, m.cs, m._pade = 2, np.ones(len(sr.params)), {}
    flags = sens._near_misses(sr._ctx, sr._traj)
    assert [f["device"] for f in flags] == [f["device"] for f in sr.flags]
    assert [f["t"] for f in flags] == pytest.approx(
        [f["t"] for f in sr.flags], rel=1e-12)


# ---------------------------------------------------------------------
# Steady state (task 3.2)
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def orbit():
    b = _buck()
    return p.periodic_steady_state(b, period=T, switch_fn=_pwm(b),
                                   sensitivities=["L.L", "RL.R", "S.duty"])


@pytest.mark.parametrize("name,key", [("L.L", "L"), ("RL.R", "R"),
                                      ("S.duty", "D")])
def test_orbit_gradients(orbit, name, key):
    iv = orbit.state_labels.index("v(vout)")
    g = orbit.dx0[name]
    assert g[iv] == pytest.approx(ref.DCM_ORBIT[key], rel=1e-7)

    def xs(v):
        kw = dict(_DEFAULTS, **{_KEY[name]: v})
        D = kw.pop("D")
        b = _buck(**kw)
        return p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, D),
                                       start_from=orbit.x0, rtol=1e-13).x0

    fd, err, _h = ref.central_difference(xs, _DEFAULTS[_KEY[name]],
                                         rel_steps=(1e-3, 1e-4))
    assert ref.rel_err(g, fd) < 1e-7, (g, fd, err)


def test_orbit_costs_one_more_period_run(orbit):
    b = _buck()
    plain = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b))
    assert orbit.n_period_runs == plain.n_period_runs + 1
    assert set(orbit.dPdp) == {"L.L", "RL.R", "S.duty"}


# ---------------------------------------------------------------------
# Refusals (by name, typed)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "auto"])
def test_other_engines_refuse_sensitivities(engine):
    b = _buck()
    kw = {"dt": T / 100} if engine == "pwl" else {}
    with pytest.raises(p.UnsupportedError) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.simulate(b, 2 * T, engine=engine, switch_fn=_pwm(b),
                       sensitivities=["L.L"], **kw)
    assert (ei.value.engine, ei.value.feature) == (engine, "sensitivities")


def test_unknown_component_is_named():
    b = _buck()
    with pytest.raises(ValueError, match="Lx"):
        p.simulate(b, 2 * T, engine="dsed", switch_fn=_pwm(b),
                   sensitivities=["Lx.L"])


@pytest.mark.parametrize("name,dev,fld", [("L.turns", "L", "turns"),
                                          ("S.g_on", "S", "g_on"),
                                          ("D1.V_th", "D1", "V_th")])
def test_non_differentiable_field_is_refused(name, dev, fld):
    b = _buck()
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 2 * T, engine="dsed", switch_fn=_pwm(b),
                   sensitivities=[name])
    assert (ei.value.feature, ei.value.device, ei.value.kind) == (
        "sensitivity_parameter", dev, fld)


def test_duty_of_a_non_pwm_schedule_is_refused():
    b = _buck()
    n, k = b.graph.num_switches, b.switch_index_of("S")

    class Burst:                     # one pulse, then off: not a PWM
        def __call__(self, t):
            m = p.SwitchStateMask(n)
            m.set(k, t < 3e-6)
            return m

        def next_edge_after(self, t):
            return 3e-6 if t < 3e-6 else math.inf

    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 2 * T, engine="dsed", switch_fn=Burst(),
                   sensitivities=["S.duty"])
    assert (ei.value.device, ei.value.kind) == ("S", "duty")


def test_opaque_gate_and_callbacks_are_refused():
    b = _buck()
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def gate(t):
        m = p.SwitchStateMask(n)
        m.set(k, (t / T) % 1 < 0.3)
        return m

    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 2 * T, engine="dsed", switch_fn=gate,
                   sensitivities=["L.L"])
    assert ei.value.feature == "switch_fn"
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 2 * T, engine="dsed", switch_fn=_pwm(b),
                   integrator="bdf2", sensitivities=["L.L"])
    assert ei.value.feature in ("integrator",)


def test_time_varying_source_and_newton_device_are_refused():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 1e-3, engine="dsed", sensitivities=["R.R"])
    assert ei.value.device == "V"
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_shockley_diode("D", "o", "gnd")
    b.add_capacitor("C", "o", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError):
        p.simulate(b, 1e-3, engine="dsed", sensitivities=["R.R"])
