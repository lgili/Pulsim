"""add-parameter-sensitivities, group 7: KPI gradients on the exact path.

``res.kpi(signal, kind)`` returns the KPI and its exact gradient (closed
forms of the exact trajectory plus the event-time terms); the oracle is a
central difference of the same KPI value (the finite-difference harness,
step chosen by its Richardson error). Orbit KPIs, ``value_and_grad`` with
``scipy.optimize.minimize(jac=True)``, and the counter gate of task 7.5.
"""

import numpy as np
import pytest

import pulsim as p
from pulsim import sensitivity as sens

from . import _sens_reference as ref
from .test_sensitivity_exact import _DEFAULTS, _KEY, _buck, _pwm

T = ref.T
W0, W1 = 30 * T, 40 * T          # the last ten periods
NAMES = ["L.L", "RL.R", "S.duty", "Vin.V"]


def _run(names, **kw):
    v = dict(_DEFAULTS, **kw)
    D = v.pop("D")
    b = _buck(**v)
    return p.simulate(b, 40 * T, engine="dsed", switch_fn=_pwm(b, D),
                      sensitivities=names)


@pytest.fixture(scope="module")
def dcm():
    return _run(NAMES)


def _fd(name, signal, kind, t0=W0, t1=W1):
    key = _KEY[name]

    def val(x):
        return _run(["L.L"], **{key: x}).kpi(signal, kind, t_from=t0,
                                             t_to=t1, with_grad=False)

    fd, err, _h = ref.central_difference(val, _DEFAULTS[key],
                                         rel_steps=(1e-3, 1e-4))
    return fd, err


# ---------------------------------------------------------------------
# 7.4: average and RMS of v_out on the DCM buck
# ---------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["avg", "rms"])
@pytest.mark.parametrize("name", NAMES)
def test_avg_and_rms_of_vout_equal_central_differences(dcm, name, kind):
    g = dcm.kpi("vout", kind, t_from=W0, t_to=W1).grad[name]
    fd, err = _fd(name, "vout", kind)
    assert abs(g - fd) <= 1e-6 * abs(fd), (g, fd, err)


@pytest.mark.parametrize("signal,kind", [
    ("vout", "final"), ("vout", "energy"), ("vout", "peak"),
    ("vout", "ripple"), ("L", "peak"), ("L", "rms"),
    ("sw", "avg"), ("sw", "rms")])
@pytest.mark.parametrize("name", ["L.L", "S.duty"])
def test_every_kpi_kind_equals_a_central_difference(dcm, signal, kind, name):
    g = dcm.kpi(signal, kind, t_from=W0, t_to=W1).grad[name]
    fd, err = _fd(name, signal, kind)
    assert abs(g - fd) <= 1e-6 * abs(fd), (g, fd, err)


def test_kpi_value_is_the_time_weighted_value_of_the_run(dcm):
    # the closed form against np.trapezoid of the samples refined by the
    # exact propagator (the run's own grid is one sample per mode)
    v = dcm.kpi("vout", "avg", t_from=W0, t_to=W1, with_grad=False)
    t = np.linspace(W0, W1, 4001)
    y = [dcm.kpi("vout", "final", t_from=0.0, t_to=tt, with_grad=False)
         for tt in t[1:]]
    y = np.array([y[0]] + y)
    assert v == pytest.approx(np.trapezoid(y, t) / (W1 - W0), rel=1e-6)


def test_the_switch_node_needs_its_event_terms(dcm, monkeypatch):
    # v(sw) jumps at every edge: without the (y- - y+) dtau/dp terms the
    # duty gradient of its average is wrong by the jump times T
    exact = dcm.kpi("sw", "avg", t_from=W0, t_to=W1).grad["S.duty"]
    monkeypatch.setattr(sens, "_event_terms",
                        lambda sr, row, t0, t1, power: np.zeros(
                            len(sr.params)))
    without = dcm.kpi("sw", "avg", t_from=W0, t_to=W1).grad["S.duty"]
    fd, _err = _fd("S.duty", "sw", "avg")
    assert abs(exact - fd) <= 1e-6 * abs(fd)
    assert abs(without - fd) > 1e-2 * abs(fd)


def test_non_unique_marks_a_flat_extremum(dcm):
    # the DCM inductor current sits at its valley through every idle
    # interval: the valley is attained at many instants
    assert dcm.kpi("L", "ripple", t_from=W0, t_to=W1).non_unique
    assert dcm.kpi("L", "valley", t_from=W0, t_to=W1).non_unique
    assert not dcm.kpi("vout", "peak", t_from=W0, t_to=W1).non_unique


def test_kpi_argument_errors(dcm):
    with pytest.raises(ValueError, match="kind"):
        dcm.kpi("vout", "median")
    with pytest.raises(ValueError, match="window"):
        dcm.kpi("vout", "avg", t_from=0.0, t_to=50 * T)
    with pytest.raises(ValueError, match="not requested"):
        dcm.sens("vout", "C.C")
    with pytest.raises(ValueError, match="signal"):
        dcm.kpi("nowhere", "avg")
    v, g = dcm.kpi("vout", "avg")
    assert set(g) == set(NAMES) and np.isfinite(v)


# ---------------------------------------------------------------------
# 7.2: orbit KPIs (the second sensitivity period)
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def orbit():
    b = _buck()
    return p.periodic_steady_state(b, period=T, switch_fn=_pwm(b),
                                   sensitivities=["L.L", "RL.R", "S.duty"])


@pytest.mark.parametrize("kind", ["avg", "rms", "ripple"])
@pytest.mark.parametrize("name", ["L.L", "RL.R", "S.duty"])
def test_orbit_kpis_equal_central_differences_of_the_pss(orbit, name, kind):
    key = _KEY[name]

    def val(x):
        v = dict(_DEFAULTS, **{key: x})
        D = v.pop("D")
        b = _buck(**v)
        r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, D),
                                    start_from=orbit.x0, rtol=1e-13,
                                    sensitivities=["L.L"])
        return r.kpi("vout", kind, with_grad=False)

    g = orbit.kpi("vout", kind).grad[name]
    fd, err, _h = ref.central_difference(val, _DEFAULTS[key],
                                         rel_steps=(1e-3, 1e-4))
    assert abs(g - fd) <= 1e-6 * abs(fd), (g, fd, err)


def test_orbit_kpi_needs_sensitivities():
    b = _buck()
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b))
    with pytest.raises(ValueError, match="sensitivities"):
        r.kpi("vout")


# ---------------------------------------------------------------------
# 7.5: eight parameters, one run
# ---------------------------------------------------------------------

_EIGHT = {"Rin.R": ("Rin", 0.05), "L.L": ("L", 100e-6),
          "RLs.R": ("RLs", 0.02), "C.C": ("C", 47e-6),
          "Resr.R": ("Resr", 0.01), "RL.R": ("RL", 50.0),
          "Vin.V": ("Vin", 24.0), "S.duty": ("D", 0.3)}


def _buck8(Rin=0.05, L=100e-6, RLs=0.02, C=47e-6, Resr=0.01, RL=50.0,
           Vin=24.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vs", "gnd", Vin)
    b.add_resistor("Rin", "vs", "vin", Rin)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "lx", L)
    b.add_resistor("RLs", "lx", "vout", RLs)
    b.add_capacitor("C", "vout", "cx", C)
    b.add_resistor("Resr", "cx", "gnd", Resr)
    b.add_resistor("RL", "vout", "gnd", RL)
    return b


def _run8(names, D=0.3, **kw):
    b = _buck8(**kw)
    return p.simulate(b, 20 * T, engine="dsed", switch_fn=_pwm(b, D),
                      sensitivities=names)


def test_eight_parameters_cost_one_run():
    names = list(_EIGHT)
    r = _run8(names)
    c = r.sensitivity.counters
    steps = len(r.times) - 1
    assert c["n_engine_runs"] == 1
    assert c["n_engine_steps"] == steps
    # one back-substitution per parameter per stage
    assert c["n_sens_propagations"] == 8 * steps
    # one base assembly + four per element parameter (the duty has none)
    assert c["n_assemblies"] == 1 + 4 * 7
    g = r.kpi("vout", "avg", t_from=10 * T, t_to=20 * T).grad
    # central differences need 2 runs per parameter: 16 runs
    for name, (key, x0) in _EIGHT.items():
        def val(x, key=key):
            return _run8(["L.L"], **{key: x}).kpi(
                "vout", "avg", t_from=10 * T, t_to=20 * T, with_grad=False)

        fd, err, _h = ref.central_difference(val, x0, rel_steps=(1e-3, 1e-4))
        assert abs(g[name] - fd) <= 1e-6 * abs(fd), (name, g[name], fd, err)


# ---------------------------------------------------------------------
# value_and_grad and scipy.optimize.minimize(jac=True)
# ---------------------------------------------------------------------

L0, C0, RIPPLE = 100e-6, 47e-6, 0.02     # an LC filter on a CCM buck


def _lc_objective(vals):
    b = _buck(L=vals["L.L"], C=vals["C.C"], R=5.0)
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.5),
                                sensitivities=["L.L", "C.C"])
    rip, g = r.kpi("vout", "ripple")
    # ripple on target, then the smallest L and C
    u = rip / RIPPLE - 1.0
    cost = u * u + 1e-2 * (vals["L.L"] / L0 + vals["C.C"] / C0)
    grad = {"L.L": 2 * u * g["L.L"] / RIPPLE + 1e-2 / L0,
            "C.C": 2 * u * g["C.C"] / RIPPLE + 1e-2 / C0}
    return cost, grad


def test_minimize_an_lc_filter_with_exact_gradients():
    opt = pytest.importorskip("scipy.optimize")
    vg = p.value_and_grad(_lc_objective, ["L.L", "C.C"])
    scale = np.array([L0, C0])

    def f(u):
        v, g = vg(u * scale)
        return v, g * scale

    u0 = np.array([1.0, 1.0])
    # the gradient at the start equals a central difference
    _v, g0 = f(u0)
    for j in range(2):
        def fj(x, j=j):
            u = u0.copy()
            u[j] = x
            return f(u)[0]

        fd, err, _h = ref.central_difference(fj, 1.0, rel_steps=(1e-3, 1e-4))
        assert abs(g0[j] - fd) <= 1e-6 * abs(fd), (j, g0[j], fd, err)
    res = opt.minimize(f, u0, jac=True, method="L-BFGS-B",
                       bounds=[(0.2, 5.0), (0.2, 5.0)])
    assert res.success, res.message
    b = _buck(L=res.x[0] * L0, C=res.x[1] * C0, R=5.0)
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.5),
                                sensitivities=["L.L"])
    assert r.kpi("vout", "ripple", with_grad=False) == pytest.approx(
        RIPPLE, rel=0.1)
    assert res.fun < f(u0)[0]


def test_value_and_grad_never_invents_a_gradient():
    vg = p.value_and_grad(lambda v: (1.0, {"L.L": 2.0}), ["L.L", "C.C"])
    with pytest.raises(ValueError, match="no gradient"):
        vg([1.0, 2.0])
    with pytest.raises(ValueError, match="entries"):
        vg([1.0])
    v, g = p.value_and_grad(lambda v: sens.KpiValue(3.0, {"L.L": 4.0}),
                            ["L.L"])([1.0])
    assert v == 3.0 and g.tolist() == [4.0]


# ---------------------------------------------------------------------
# add-design-optimization-workflow 7.3: Evaluator(gradients=...)
# ---------------------------------------------------------------------

def _design_buck(d):
    b = _buck(L=d["L"], R=d["R"])
    return b, _pwm(b, d["D"])


_GRADS = {"L": "L.L", "R": "RL.R", "D": "S.duty"}
_D0 = dict(L=100e-6, R=50.0, D=0.3)


def test_evaluator_gradients_are_the_orbit_gradients(orbit):
    from pulsim.design import Evaluator
    ev = Evaluator(_design_buck, period=T, output="vout", gradients=_GRADS)
    r = ev.evaluate(_D0)
    assert r.status == "ok", r.reason
    assert set(r.grads) == {"v_out", "ripple_v", "ripple_i.L"}
    for var, name in _GRADS.items():
        for k, kind, sig in (("v_out", "avg", "vout"),
                             ("ripple_v", "ripple", "vout")):
            assert r.grads[k][var] == pytest.approx(
                orbit.kpi(sig, kind).grad[name], rel=1e-9)
    assert r.kpis["v_out"] == pytest.approx(
        orbit.kpi("vout", "avg", with_grad=False), rel=1e-12)
    # and against a central difference of the evaluator's own KPI
    for var in ("L", "D"):
        def val(x, var=var):
            return ev.evaluate({**_D0, var: x}).kpis["v_out"]

        fd, err, _h = ref.central_difference(val, _D0[var],
                                             rel_steps=(1e-3, 1e-4))
        assert abs(r.grads["v_out"][var] - fd) <= 1e-6 * abs(fd), (var, fd)
    # without gradients= the record has none
    assert Evaluator(_design_buck, period=T, output="vout").evaluate(
        _D0).grads == {}


def test_evaluator_refuses_gradients_off_the_pss_route():
    from pulsim.design import Evaluator, SteadySpec, ThermalSpec
    r = Evaluator(_design_buck, period=T, output="vout", gradients=_GRADS,
                  steady=SteadySpec(method="transient")).evaluate(_D0)
    assert r.status == "refused" and r.refusal[1] == "gradients"
    assert r.grads == {}
    with pytest.raises(p.UnsupportedError) as e:
        Evaluator(_design_buck, period=T, output="vout", gradients=_GRADS,
                  thermal=ThermalSpec(r_th={"S": 1.0}))
    assert e.value.feature == "gradients"


def test_evaluator_refuses_a_scaled_design_variable():
    from pulsim.design import Evaluator

    def scaled(d):
        return _design_buck({**d, "L": d["L_uH"] * 1e-6})

    ev = Evaluator(scaled, period=T, output="vout",
                   gradients={"L_uH": "L.L"})
    with pytest.raises(ValueError, match="scaled mapping"):
        ev.evaluate({"L_uH": 100.0, "R": 50.0, "D": 0.3})
