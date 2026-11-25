"""Branch-form (ideal) switches and diodes in the exact period map, the
shooting and the sensitivities (update-number-changing-release blocker 1.4).

The period map reads a branch-form diode's turn-OFF predicate on its solved
current and its saltation gradient on that current's recovery row; the
observables read the current unknown itself. Checked against a closed form
(the ideal buck's mean output is D*Vin), the transient run to steady state,
the conductance-form twin in the limit, central differences, and the
transient's own refusal names.
"""

import numpy as np
import pytest

import pulsim as p

from . import _sens_reference as ref

T = 1e-5


def _sw(b, name, a, c, kind, g_on=None, R_on=0.0, g_off=1e-9):
    add = b.add_switch if kind == "S" else b.add_diode
    if g_on is None:
        add(name, a, c, form="branch", R_on=R_on, g_off=g_off)
    else:
        add(name, a, c, g_on, g_off)


def _buck(R, g_on=None, L=100e-6, C=47e-6, Vin=24.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", Vin)
    _sw(b, "S", "vin", "sw", "S", g_on)
    _sw(b, "D1", "gnd", "sw", "D", g_on)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", C)
    b.add_resistor("RL", "vout", "gnd", R)
    return b


def _boost(g_on=None):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 20e-6)
    _sw(b, "S", "sw", "gnd", "S", g_on)
    _sw(b, "D", "sw", "out", "D", g_on)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 100.0)
    return b


def _bridge(g_on=None):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "ac", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("Rs", "ac", "a", 0.5)
    for n, a, c in (("D1", "a", "p"), ("D2", "gnd", "p"), ("D3", "n", "a"),
                    ("D4", "n", "gnd")):
        _sw(b, n, a, c, "D", g_on)
    b.add_capacitor("C", "p", "n", 100e-6)
    b.add_resistor("RL", "p", "n", 50.0)
    return b


def _pwm(b, D):
    return p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"),
                                b.graph.num_switches)


CASES = {  # name: (builder(g_on), period, duty or None)
    "buck_ccm": (lambda g=None: _buck(1.0, g), T, 0.5),
    "buck_dcm": (lambda g=None: _buck(50.0, g), T, 0.3),
    "boost_dcm": (_boost, T, 0.4),
}


def _pss(mk, period, D, g_on=None, **kw):
    b = mk(g_on)
    return p.periodic_steady_state(
        b, period=period, switch_fn=None if D is None else _pwm(b, D), **kw)


@pytest.mark.parametrize("case", list(CASES))
def test_orbit_is_the_transient_steady_state(case):
    mk, period, D = CASES[case]
    r = _pss(mk, period, D)
    s = r._ctx
    x = np.zeros(len(r.x0))
    # the same exact stepper, period by period, until mu^N < 1e-12
    n = int(np.ceil(np.log(1e-12) / np.log(r.floquet_radius)))
    assert n < 15000
    for _ in range(n):
        x = s.run(x).x_end
    assert np.abs(x - r.x0).max() <= 1e-9 * max(1.0, np.abs(r.x0).max())
    # and one plain simulate() period from the orbit stays on it
    b = mk(None)
    one = p.simulate(b, t_end=period, dt=period, engine="dsed",
                     switch_fn=_pwm(b, D), initial_state=r.x0)
    assert np.abs(np.asarray(one.states_reduced[-1]) - r.x0).max() \
        <= 1e-9 * max(1.0, np.abs(r.x0).max())


@pytest.mark.parametrize("case", list(CASES))
def test_orbit_and_multipliers_equal_the_conductance_twin_in_the_limit(case):
    mk, period, D = CASES[case]
    r = _pss(mk, period, D)
    tw = _pss(mk, period, D, g_on=1e6)
    assert r.state_labels == tw.state_labels
    scale = np.abs(tw.x0).max()
    assert np.abs(r.x0 - tw.x0).max() <= 1e-5 * scale
    mu, mu_t = np.sort(np.abs(r.multipliers)), np.sort(np.abs(tw.multipliers))
    assert np.abs(mu - mu_t).max() <= 1e-5


def test_ideal_buck_mean_output_is_the_closed_form():
    # no conduction loss and no series resistance: <v_out> = D * Vin exactly
    b = _buck(1.0)
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.5),
                                sensitivities=["S.duty"])
    k = r.kpi("vout", "avg")
    assert k.value == pytest.approx(0.5 * 24.0, rel=1e-9)
    assert k.grad["S.duty"] == pytest.approx(24.0, rel=1e-7)


def test_ideal_diode_bridge_with_a_sine_source():
    # the sine source takes the Python map (its saltation through _g); the
    # conductance twin is refused from g_on = 1e5 up (a wrong-mode settle),
    # so the limit is read at g_on = 1e4: R_on = 1e-4 Ohm against Rs = 0.5
    mk = _bridge
    r = _pss(mk, 1e-3, None)
    assert r.stable
    tw = _pss(mk, 1e-3, None, g_on=1e4)
    assert np.abs(r.x0 - tw.x0).max() <= 1e-4 * np.abs(tw.x0).max()
    assert np.abs(np.abs(r.multipliers) - np.abs(tw.multipliers)).max() < 1e-4
    s = r._ctx
    x = np.zeros(len(r.x0))
    for _ in range(40):          # |mu| = 0.008
        x = s.run(x).x_end
    assert np.abs(x - r.x0).max() <= 1e-9 * np.abs(r.x0).max()


_D0 = dict(L=100e-6, R=50.0, C=47e-6, Vin=24.0, D=0.3)
_KEY = {"L.L": "L", "RL.R": "R", "C.C": "C", "Vin.V": "Vin", "S.duty": "D"}


def _buck_run(names, **kw):
    v = dict(_D0, **kw)
    D = v.pop("D")
    R = v.pop("R")
    b = _buck(R, **v)
    return p.simulate(b, 40 * T, engine="dsed", switch_fn=_pwm(b, D),
                      sensitivities=names)


@pytest.fixture(scope="module")
def dcm_sens():
    return _buck_run(list(_KEY))


@pytest.mark.parametrize("name", list(_KEY))
def test_transient_sensitivities_equal_central_differences(dcm_sens, name):
    key = _KEY[name]

    def vend(v):
        return float(np.asarray(_buck_run(["L.L"], **{key: v}).v("vout"))[-1])

    fd, err, _h = ref.central_difference(vend, _D0[key],
                                         rel_steps=(1e-3, 1e-4))
    s = dcm_sens.sens("vout", name)[-1]
    assert abs(s - fd) <= 1e-7 * abs(fd), (s, fd, err)


@pytest.mark.parametrize("name", ["L.L", "RL.R", "S.duty"])
def test_orbit_gradients_equal_central_differences(name):
    b = _buck(50.0)
    o = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.3),
                                sensitivities=["L.L", "RL.R", "S.duty"])
    key = _KEY[name]

    def xs(v):
        kw = dict(_D0, **{key: v})
        D, R = kw.pop("D"), kw.pop("R")
        bb = _buck(R, **kw)
        return p.periodic_steady_state(bb, period=T, switch_fn=_pwm(bb, D),
                                       start_from=o.x0, rtol=1e-13).x0

    fd, err, _h = ref.central_difference(xs, _D0[key], rel_steps=(1e-3, 1e-4))
    assert ref.rel_err(o.dx0[name], fd) < 1e-7, (o.dx0[name], fd, err)


def _gated(build):
    b = p.CircuitBuilder()
    build(b)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


def _vloop(b):     # a closed ideal switch across the source
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_resistor("R", "a", "x", 1.0)
    b.add_inductor("L", "x", "gnd", 1e-4)
    b.add_switch("S", "a", "gnd", form="branch", R_on=0.0, g_off=1e-9)


def _floating(b):  # an open g_off = 0 switch isolates an LC
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_resistor("R0", "a", "m", 1.0)
    b.add_switch("S", "m", "c", form="branch", R_on=1e-3, g_off=0.0)
    b.add_capacitor("C", "c", "f", 1e-6)
    b.add_inductor("L", "f", "c", 1e-4)


@pytest.mark.parametrize("build,feature", [(_vloop, "ideal_voltage_loop"),
                                           (_floating, "floating_node")])
def test_a_circuit_that_is_not_posed_is_refused_by_the_transients_name(
        build, feature):
    b, sf = _gated(build)
    with pytest.raises(p.UnsupportedError) as tr:
        p.simulate(b, 3 * T, engine="dsed", switch_fn=sf)
    b, sf = _gated(build)
    with pytest.raises(p.UnsupportedError) as ps:
        p.periodic_steady_state(b, period=T, switch_fn=sf)
    assert tr.value.feature == ps.value.feature == feature


def test_a_source_capacitor_loop_is_refused_as_dependent_states():
    # the transient projects onto K x = k; the period map has no Jacobian of
    # that projection, so it refuses by its existing name
    def build(b):
        b.add_voltage_source("V", "a", "gnd", 10.0)
        b.add_switch("S", "a", "c", form="branch", R_on=0.0, g_off=1e-9)
        b.add_capacitor("C", "c", "gnd", 1e-6)
        b.add_resistor("R", "c", "gnd", 10.0)
    b, sf = _gated(build)
    with pytest.raises(p.UnsupportedError) as ps:
        p.periodic_steady_state(b, period=T, switch_fn=sf)
    assert ps.value.feature == "dependent_states"
