"""add-branch-form-ideal-switch groups 3 and 4: the branch form on the engines.

* 3.4 with ``R_on = 1e-3``, ``g_off = 1e-9`` the branch form reproduces the conductance form
  (``g_on = 1e3``) within 1e-10 on the buck, the bridge rectifier and the three-phase inverter;
* 4.4 the ideal switch (``R_on = 0``) charges an RL to the ideal closed form; the structural pass
  refuses a floating node and an ideal voltage loop by name, and runs once per mode;
* 4.1 / 4.3 API, YAML and the engine column of the branch form.
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p


def _mask(b, on_names):
    m = p.SwitchStateMask(b.graph.num_switches)
    for nm in on_names:
        m.set(b.switch_index_of(nm), True)
    return m


def _sim(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, **kw)


# ---------------------------------------------------------------------------------------------
# 3.4: same values, same numbers
# ---------------------------------------------------------------------------------------------
def _buck(branch):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    if branch:
        b.add_ideal_switch("S", "vin", "sw", 1e-3, 1e-9)
        b.add_diode("D", "gnd", "sw", form="branch", R_on=1e-3, g_off=1e-9)
    else:
        b.add_switch("S", "vin", "sw", 1e3, 1e-9)
        b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    T = 1e-5
    k = b.switch_index_of("S")
    n = b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, (t % T) < 0.4 * T)
        return m
    return b, dict(t_end=2e-3, dt=T / 100, switch_fn=fn, engine="pwl"), "out"


def _bridge(branch):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "ac", "gnd", 0.0, 100.0, 50.0)
    for nm, a, c in (("D1", "ac", "p"), ("D2", "gnd", "p"), ("D3", "n", "ac"), ("D4", "n", "gnd")):
        if branch:
            b.add_diode(nm, a, c, form="branch", R_on=1e-3, g_off=1e-9)
        else:
            b.add_diode(nm, a, c, 1e3, 1e-9)
    b.add_capacitor("C", "p", "n", 1000e-6)
    b.add_resistor("R", "p", "n", 100.0)
    b.add_resistor("Rg", "n", "gnd", 1e6)
    return b, dict(t_end=40e-3, dt=10e-6, engine="pwl",
                   switch_fn=lambda t: p.SwitchStateMask(4)), ("p", "n")


def _inverter(branch):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 400.0)
    names = []
    for ph in "abc":
        for hi in (True, False):
            nm = f"S{ph}{'h' if hi else 'l'}"
            a, c = ("dc", ph) if hi else (ph, "gnd")
            if branch:
                b.add_ideal_switch(nm, a, c, 1e-3, 1e-9)
                b.add_diode(nm + "d", c, a, form="branch", R_on=1e-3, g_off=1e-9)
            else:
                b.add_switch(nm, a, c, 1e3, 1e-9)
                b.add_diode(nm + "d", c, a, 1e3, 1e-9)
            names.append(nm)
        b.add_resistor("R" + ph, ph, "m" + ph, 5.0)
        b.add_inductor("L" + ph, "m" + ph, "star", 2e-3)
    idx = {nm: b.switch_index_of(nm) for nm in names}
    n = b.graph.num_switches
    f_s, f_o = 10e3, 50.0

    def fn(t):
        m = p.SwitchStateMask(n)
        carrier = 2 * abs((t * f_s) % 1 - 0.5) * 2 - 1
        for j, ph in enumerate("abc"):
            ref = 0.8 * math.sin(2 * math.pi * f_o * t - 2 * math.pi * j / 3)
            up = ref > carrier
            m.set(idx[f"S{ph}h"], up)
            m.set(idx[f"S{ph}l"], not up)
        return m
    return b, dict(t_end=4e-3, dt=1e-6, switch_fn=fn, engine="pwl"), "ma"


@pytest.mark.parametrize("make", [_buck, _bridge, _inverter], ids=["buck", "bridge", "inverter"])
def test_branch_form_reproduces_the_conductance_form(make):
    bc, kw, node = make(False)
    bb, kwb, _ = make(True)
    rc, rb = _sim(bc, **kw), _sim(bb, **kwb)
    # The bridge reads its bus differentially: with every diode OFF its common
    # mode is set through Rg = 1 MOhm by currents that cancel to the solver's
    # residual (~7.6e-12 A in both forms, scaling with Rg), not by the form.
    def v(r):
        if isinstance(node, tuple):
            return np.asarray(r.v(node[0])) - np.asarray(r.v(node[1]))
        return np.asarray(r.v(node))
    vc, vb = v(rc), v(rb)
    assert vc.shape == vb.shape
    span = np.ptp(vc)
    assert span > 1.0                                  # the circuit does something
    assert np.max(np.abs(vb - vc)) <= 1e-10 * max(1.0, np.max(np.abs(vc)))


# ---------------------------------------------------------------------------------------------
# 4.4: the ideal switch, the structural refusals, one pass per mode
# ---------------------------------------------------------------------------------------------
def _rl(R_on=0.0, g_off=1e-9):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", R_on, g_off)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    return b


def test_ideal_rl_on_trbdf2():
    b = _rl()
    r = _sim(b, t_end=10e-3, switch_fn=lambda t: _mask(b, ["S"]), engine="trbdf2",
             rtol=1e-9, atol=1e-12)
    assert r.i("L")[-1] == pytest.approx(10 * (1 - math.exp(-10)), abs=1e-6)   # 9.99954600


def test_ideal_rl_on_pwl_at_tau_over_80():
    b = _rl()
    h = 1e-3 / 80
    r = _sim(b, t_end=10e-3, dt=h, switch_fn=lambda t: _mask(b, ["S"]), engine="pwl",
             discontinuity_restart=True)
    t = np.asarray(r.times)
    err = np.max(np.abs(np.asarray(r.i("L")) - 10 * (1 - np.exp(-t / 1e-3))))
    assert err <= 2e-3            # prototype 1.51e-3 (two BE half-steps then trapezoid)
    # the solved switch current is the inductor current while closed
    assert np.max(np.abs(np.asarray(r.i("S"))[1:] - np.asarray(r.i("L"))[1:])) < 1e-9


def test_two_open_ideal_switches_in_series_float_their_node():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S1", "vin", "mid", 0.0, 0.0)
    b.add_ideal_switch("S2", "mid", "gnd", 0.0, 0.0)
    b.add_resistor("R", "vin", "gnd", 10.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(b, t_end=1e-5, dt=1e-6, switch_fn=lambda t: _mask(b, []), engine="pwl")
    e = ei.value
    assert (e.engine, e.feature, e.kind) == ("pwl", "floating_node", "Switch")
    assert e.device == "S1, S2"
    assert "mid" in str(e) and "g_off > 0" in str(e)


def test_two_closed_ideal_switches_in_parallel_across_a_source():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S1", "vin", "gnd")
    b.add_ideal_switch("S2", "vin", "gnd")
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(b, t_end=1e-5, dt=1e-6, switch_fn=lambda t: _mask(b, ["S1", "S2"]), engine="pwl")
    assert (ei.value.feature, ei.value.kind) == ("ideal_voltage_loop", "Switch")
    assert ei.value.device == "S1"
    assert "R_on > 0" in str(ei.value)


def test_structural_pass_runs_once_per_mode():
    # two ideal switches sequenced through 4 modes, then a fifth (both on).
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S1", "vin", "a", 0.0, 1e-9)
    b.add_ideal_switch("S2", "a", "gnd", 1.0, 1e-9)
    b.add_resistor("R", "a", "gnd", 10.0)
    seq = [[], ["S1"], ["S2"], [], ["S1"], ["S1", "S2"], ["S2"]]

    def fn(t):
        return _mask(b, seq[min(int(t / 1e-4), len(seq) - 1)])
    r = _sim(b, t_end=1e-3, dt=1e-5, switch_fn=fn, engine="pwl")
    assert r.counters["mode_structure_checks"] == 4
    seq.insert(3, ["S1", "S2"])
    seq[-1] = []
    r2 = _sim(b, t_end=1e-3, dt=1e-5, switch_fn=fn, engine="pwl")
    assert r2.counters["mode_structure_checks"] == 4
    # control: a conductance-form circuit never runs the pass
    c = p.CircuitBuilder()
    c.add_voltage_source("V", "vin", "gnd", 10.0)
    c.add_switch("S1", "vin", "a", 1e3, 1e-9)
    c.add_resistor("R", "a", "gnd", 10.0)
    r3 = _sim(c, t_end=1e-4, dt=1e-5, switch_fn=lambda t: _mask(c, ["S1"]), engine="pwl")
    assert r3.counters["mode_structure_checks"] == 0


# ---------------------------------------------------------------------------------------------
# 4.1 API, YAML, components()
# ---------------------------------------------------------------------------------------------
def test_api_values_and_components():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_ideal_switch("S", "a", "b")
    b.add_diode("D", "b", "gnd", form="branch", V_f=0.7)
    b.add_diode("D2", "b", "gnd", g_on=1e3, form="branch")      # R_on = 1/g_on
    comps = {c["name"]: c["params"] for c in b.components()}
    assert comps["S"]["R_on"] == 0.0 and comps["S"]["g_off"] == 0.0
    assert comps["S"]["branch_form"] == 1 and math.isinf(comps["S"]["g_on"])
    assert comps["D"]["V_f"] == 0.7 and comps["D2"]["R_on"] == pytest.approx(1e-3)
    assert "I(S)" in b.state_var_names()
    with pytest.raises(ValueError, match=r'add_ideal_switch\("X"\): R_on = -1'):
        b.add_ideal_switch("X", "a", "b", -1.0)
    with pytest.raises(ValueError, match="form="):
        b.add_diode("Y", "a", "b", 1e3, 1e-9, form="ideal")
    with pytest.raises(ValueError, match="conductance form takes g_on"):
        b.add_diode("Y", "a", "b")


def test_yaml_form_branch():
    text = (
        "circuit:\n  devices:\n"
        "    - {type: voltage_source, name: V1, from: a, to: gnd, V: 1.0}\n"
        "    - {type: switch, name: S1, from: a, to: b, form: branch}\n"
        "    - {type: diode, name: D1, anode: b, cathode: gnd, form: branch, R_on: 0.01}\n")
    b = p.load_yaml_string(text)
    b = getattr(b, "builder", b)
    comps = {c["name"]: c["params"] for c in b.components()}
    assert comps["S1"]["branch_form"] == 1 and comps["S1"]["R_on"] == 0.0
    assert comps["D1"]["R_on"] == 0.01
    with pytest.raises(Exception, match="form: 'brnach'"):
        p.load_yaml_string(text.replace("form: branch}", "form: brnach}", 1))


# ---------------------------------------------------------------------------------------------
# 4.3 the engine column of the branch form (no silent cell)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ["dsed", "bdf1"])
def test_branch_switch_refused_by_name(engine):
    b = _rl()
    with pytest.raises(p.UnsupportedError) as ei:
        if engine == "bdf1":
            opts = p._pulsim.SimulationOptions(t_start=0.0, t_end=1e-4, dt=1e-6)
            p._pulsim.run_transient_bdf1(b, opts, lambda t: _mask(b, ["S"]))
        else:
            _sim(b, t_end=1e-4, dt=1e-6, switch_fn=lambda t: _mask(b, ["S"]), engine=engine)
    assert (ei.value.engine, ei.value.feature, ei.value.device, ei.value.kind) == (
        engine, "branch_form", "S", "Switch")


@pytest.mark.parametrize("engine", ["trbdf2", "dsed"])
def test_branch_diode_refused_by_name(engine):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 1e3)
    b.add_diode("D", "a", "o", form="branch", R_on=1e-3, g_off=1e-9)
    b.add_resistor("R", "o", "gnd", 10.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(b, t_end=1e-3, switch_fn=lambda t: p.SwitchStateMask(1), engine=engine)
    assert (ei.value.feature, ei.value.device, ei.value.kind) == ("branch_form", "D", "Diode")


def test_branch_switch_refused_by_the_ac_sweep():
    b = _rl()
    with pytest.raises(p.UnsupportedError) as ei:
        p._pulsim.run_mna_sweep_kernel(b.graph, b.pool, _mask(b, ["S"]), [1e2, 1e3],
                                       b.node_id_of("vin"), b.node_id_of("b"))
    assert (ei.value.engine, ei.value.feature) == ("ac_sweep", "branch_form")


@pytest.mark.parametrize("R_on, g_off, why", [(0.0, 0.0, "g_off = 0"), (1e-3, 0.0, "g_off = 0")])
def test_trbdf2_refuses_dependent_states(R_on, g_off, why):
    b = _rl(R_on, g_off)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(b, t_end=1e-4, switch_fn=lambda t: _mask(b, ["S"]), engine="trbdf2")
    assert (ei.value.engine, ei.value.feature, ei.value.device) == (
        "trbdf2", "dependent_states", "S")
    assert why in str(ei.value)


def test_trbdf2_refuses_an_ideal_switch_closing_a_capacitor_loop():
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", 1e-6)
    b.add_capacitor("C2", "y", "gnd", 3e-6)
    b.add_resistor("R", "x", "gnd", 1e3)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(b, t_end=1e-4, switch_fn=lambda t: _mask(b, ["S"]), engine="trbdf2")
    assert ei.value.feature == "dependent_states"
