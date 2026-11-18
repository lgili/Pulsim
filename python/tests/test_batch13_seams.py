"""Integration batch 13: snapshots, the ideal switch and the magnetic domain
at the seams of the tracks merged together.

C: one snapshot matrix, {pwl, trbdf2} x {stateful devices, closed loop
   with controller state, C block, latched thyristor, induction machine
   mid-start, loss port}: the resumed run equals the uninterrupted one to
   what each track claims (pwl with the same dt continues the grid: bit for
   bit; trbdf2 to tolerance) or is refused by name.
D: a branch-form ideal switch / diode with a loss port beside an IGBT, on
   pwl and trbdf2; mode_selection with a gated device refused by name;
   'shadow' bit-identical to the default.
E: a permeance-form transformer in a switching converter with a loss port
   and a C block in the same circuit (it drives an RC node, not the gate); YAML round trip; with_value on a permeance;
   the exact engine on a linear magnetic circuit.
"""

from __future__ import annotations

import math
import pickle
import warnings

import numpy as np
import pytest

import pulsim as p

DT = 1e-6


# ---------------------------------------------------------------- C ----------
def _stateful():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_nonlinear_capacitor("Coss", "vin", "sw", 1e-9, 25.0, 0.5)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_saturable_inductor("L", "sw", "out", 100e-6, 6.0, 1e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", 5.0)
    fn = p.make_pwm_switch_fn(50e3, 0.4, b.switch_index_of("S"), b.graph.num_switches)
    return b, dict(switch_fn=fn)


def _closed_loop():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    loop = p.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05, output_max=0.95),
        measured="out", setpoint=12.0, switch="Q", freq=10e3)
    return b, dict(closed_loops=[loop])


def _c_block():
    b = p.CircuitBuilder()
    b.add_resistor("R", "u", "v", 1e3)
    b.add_capacitor("C", "v", "gnd", 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p.add_c_block(b, inputs=[("v", "v")], outputs=[("v", "u", "gnd")], dt=1e-4,
                      fn=lambda t, dt, i, o, s: o.__setitem__(0, 2.0 * (1.0 - i[0])))
    return b, {}


def _thyristor():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 60.0)
    b.add_thyristor("T", "a", "k", 1e-3, 1e9)
    b.add_resistor("R", "k", "m", 10.0)
    b.add_inductor("L", "m", "gnd", 10e-3)
    sw = p.make_phase_control_fn(60.0, math.radians(30.0), 100e-6,
                                 [b.switch_index_of("T")], b.graph.num_switches)
    return b, dict(switch_fn=sw)


def _induction_machine():
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"V{k}", node, "gnd", 0.0, 100.0 * math.sqrt(2), 60.0,
                                  -2 * math.pi / 3 * k)
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w", R_s=0.5, L_ls=2e-3,
                                R_r=0.4, L_lr=2e-3, L_m=60e-3, pole_pairs=2, J=0.01)
    return b, {}


def _loss_port():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "o", 1e6, 1e-12)
    b.add_resistor("RL", "o", "gnd", 10.0)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 1.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    t = p.LossTable(v_axis=[100.0], i_axis=[1.0], tj_axis=[25.0],
                    energy=np.full((1, 1, 1), 1e-3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.set_device_losses("S1", junction="tj", E_on=t, E_off=t)
    return b, dict(switch_fn=p.make_pwm_switch_fn(10e3, 0.5, 0, 1))


# (factory, t1, t2, pwl dt); t1 is mid-period and off every tick
CASES = {
    "stateful": (_stateful, 0.53e-4, 1.2e-4, 1e-8),
    "closed_loop": (_closed_loop, 1.53e-3, 3e-3, DT),
    "c_block": (_c_block, 0.73e-3, 2e-3, DT),
    "thyristor": (_thyristor, 6e-3, 12e-3, DT),
    "induction_machine": (_induction_machine, 0.03, 0.06, 5e-5),
    "loss_port": (_loss_port, 2.53e-4, 5e-4, DT),
}
# trbdf2: the loss port is a pwl feature, refused by name (feature
# device_losses); a resume between two controller ticks of a closed loop
# or C block is refused by name (event-queue track: the snapshot lacks the
# tick origin)
TRBDF2_REFUSED = {"loss_port": "device_losses"}


def _run(factory, engine, t_end, dt, builder=None, kw=None, **extra):
    if builder is None:
        builder, kw = factory()
    if engine == "pwl":
        extra.setdefault("dt", dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return builder, kw, p.simulate(builder, t_end, engine=engine, **kw, **extra)


@pytest.mark.parametrize("case", sorted(CASES))
def test_pwl_resume_continues_the_grid_bit_for_bit(case):
    factory, t1, t2, dt = CASES[case]
    _, _, full = _run(factory, "pwl", t2, dt)
    b, kw, first = _run(factory, "pwl", t1, dt)
    snap = first.final_snapshot
    _, _, second = _run(factory, "pwl", t2, dt, builder=b, kw=kw, t_start=snap.t,
                        resume_from=snap)
    tf, ts = np.asarray(full.times), np.asarray(second.times)
    k = int(np.searchsorted(tf, ts[0]))
    assert tf[k] == ts[0] and len(tf) - k == len(ts)
    np.testing.assert_array_equal(tf[k:], ts)
    xf, xs = np.asarray(full.states)[k:], np.asarray(second.states)
    if case == "loss_port":
        # the heat in flight at the split is dropped (pinned gap, owner:
        # the event-losses track): the electrical node is bit-identical,
        # the junction is not
        np.testing.assert_array_equal(np.asarray(full.v("o"))[k:], np.asarray(second.v("o")))
        return
    np.testing.assert_array_equal(xf, xs)


@pytest.mark.parametrize("case", sorted(CASES))
def test_trbdf2_resume_matches_to_tolerance_or_is_refused_by_name(case):
    factory, t1, t2, _ = CASES[case]
    if case in TRBDF2_REFUSED:
        with pytest.raises(p.UnsupportedError, match=TRBDF2_REFUSED[case]):
            _run(factory, "trbdf2", t1, None)
        return
    _, _, full = _run(factory, "trbdf2", t2, None)
    b, kw, first = _run(factory, "trbdf2", t1, None)
    snap = first.final_snapshot
    try:
        _, _, second = _run(factory, "trbdf2", t2, None, builder=b, kw=kw,
                            t_start=snap.t, resume_from=snap)
    except p.UnsupportedError as e:
        # t1 is between two ticks by construction: the named refusal
        assert case in ("closed_loop", "c_block") and "between two" in str(e), str(e)
        return
    assert case not in ("closed_loop", "c_block"), "a between-ticks resume ran"
    xs = np.asarray(second.states)[-1]
    xf = np.array([np.interp(second.times[-1], full.times, col)
                   for col in np.asarray(full.states).T])
    scale = max(np.abs(np.asarray(full.states)).max(), 1e-12)
    assert np.abs(xs - xf).max() <= 2e-3 * scale, case


def test_every_snapshot_pickles_and_the_copy_resumes_or_refuses_by_name():
    for case, (factory, t1, t2, dt) in sorted(CASES.items()):
        b, kw, first = _run(factory, "pwl", t1, dt)
        snap = first.final_snapshot
        copy = pickle.loads(pickle.dumps(snap))
        assert copy.__getstate__()[:2] == ("SolverSnapshot", 4)
        assert list(copy.dev_state) == list(snap.dev_state)
        try:
            _, _, a = _run(factory, "pwl", t2, dt, builder=b, kw=kw, t_start=snap.t,
                           resume_from=copy)
        except p.UnsupportedError as e:
            # controllers' state does not pickle: refused by name
            assert case in ("closed_loop", "c_block") and "pickled" in str(e), (case, e)
            continue
        b2, kw2, first2 = _run(factory, "pwl", t1, dt)
        _, _, c = _run(factory, "pwl", t2, dt, builder=b2, kw=kw2,
                       t_start=first2.final_snapshot.t, resume_from=first2.final_snapshot)
        np.testing.assert_array_equal(np.asarray(a.states), np.asarray(c.states))


# ---------------------------------------------------------------- D ----------
def _ideal_with_igbt(loss=True, igbt=True):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 48.0)
    b.add_ideal_switch("S", "in", "sw", R_on=0.0, g_off=1e-9)
    b.add_diode("D", "gnd", "sw", form="branch", R_on=0.0, g_off=1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    # a second leg: an IGBT with a knee into its own load
    if igbt:
        b.add_igbt("Q", "in", "q", 1e3, 1e-9, V_ce0=0.9, reverse_blocking=True)
        b.add_resistor("RQ", "q", "gnd", 10.0)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 2.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    if loss:
        t = p.LossTable(v_axis=[48.0], i_axis=[1.0], tj_axis=[25.0],
                        energy=np.full((1, 1, 1), 1e-5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.set_device_losses("S", junction="tj", E_on=t, E_off=t)
    n = b.graph.num_switches
    iS = b.switch_index_of("S")
    iQ = b.switch_index_of("Q") if igbt else None

    def fn(t):
        m = p.SwitchStateMask(n)
        ph = (t * 50e3) % 1.0
        m.set(iS, bool(ph < 0.4))
        if iQ is not None:
            m.set(iQ, bool(ph < 0.7))
        return m
    return b, fn


def test_ideal_branch_switch_loss_port_and_igbt_on_pwl():
    b, fn = _ideal_with_igbt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 2e-4, dt=1e-7, switch_fn=fn)
        b2, fn2 = _ideal_with_igbt(igbt=False)
        r_sh = p.simulate(b2, 2e-4, dt=1e-7, switch_fn=fn2, mode_selection="shadow")
        r_ns = p.simulate(_ideal_with_igbt(igbt=False)[0], 2e-4, dt=1e-7, switch_fn=fn2)
    assert r.heat_injected > 0          # the port on the ideal switch works
    assert np.asarray(r.v("tj"))[-1] > 25.0
    # 'shadow' only shadows the default: bit-identical (no gated device)
    np.testing.assert_array_equal(np.asarray(r_ns.states), np.asarray(r_sh.states))
    # with the IGBT (a gated device) every mode_selection is refused by name
    with pytest.raises(p.UnsupportedError, match="gated device 'Q'"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.simulate(_ideal_with_igbt()[0], 2e-5, dt=1e-7, switch_fn=fn,
                       mode_selection="shadow")


def test_ideal_branch_switch_beside_an_igbt_on_trbdf2():
    b, fn = _ideal_with_igbt(loss=False)
    bp, _ = _ideal_with_igbt(loss=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 2e-4, engine="trbdf2", switch_fn=fn)
        rp = p.simulate(bp, 2e-4, dt=1e-8, switch_fn=fn, discontinuity_restart=True)
    v, vp = np.asarray(r.v("out"))[-1], np.interp(r.times[-1], rp.times, rp.v("out"))
    assert v == pytest.approx(vp, rel=2e-3)
    bl, _ = _ideal_with_igbt(loss=True)
    with pytest.raises(p.UnsupportedError, match="device_losses"):
        p.simulate(bl, 2e-5, engine="trbdf2", switch_fn=fn)


def test_ideal_branch_circuit_round_trips_yaml_and_pickle():
    b, fn = _ideal_with_igbt(loss=False)
    c = b.build()
    c2 = p.Circuit.from_yaml(c.to_yaml())
    c3 = pickle.loads(pickle.dumps(c))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rs = [p.simulate(x, 5e-5, dt=1e-7, switch_fn=fn) for x in (c, c2, c3)]
    for r in rs[1:]:
        np.testing.assert_array_equal(np.asarray(rs[0].states), np.asarray(r.states))


# ---------------------------------------------------------------- E ----------
def _magnetic_flyback(c_block=True, loss=True):
    # forward-style converter: a permeance-form transformer (two windings
    # on a gapped core), a switch with a loss port, a C-block duty source
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_winding("Wp", "in", "d", "m1", "gnd", 20)
    b.add_switch("S", "d", "gnd", 1e3, 1e-9)
    b.add_winding("Ws", "s", "gnd", "gnd", "m1", 10)
    b.add_permeance("Pc", "m1", "m2", 2.6e-6)
    b.add_air_gap("G", "m2", "gnd", 76e-6, 0.5e-3)
    b.add_diode("D", "s", "out", 1e3, 1e-9)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 2.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    if loss:
        t = p.LossTable(v_axis=[24.0], i_axis=[1.0], tj_axis=[25.0],
                        energy=np.full((1, 1, 1), 1e-6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.set_device_losses("S", junction="tj", E_on=t, E_off=t)
    if c_block:
        b.add_resistor("Rd", "duty", "gnd", 1e6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            p.add_c_block(b, inputs=[("v", "out")], outputs=[("v", "duty", "gnd")], dt=1e-5,
                          fn=lambda t, dt, i, o, s: o.__setitem__(0, 0.4))
    fn = p.make_pwm_switch_fn(50e3, 0.4, b.switch_index_of("S"), b.graph.num_switches)
    return b, fn


def test_permeance_transformer_in_a_converter_with_loss_port_and_c_block():
    b, fn = _magnetic_flyback()
    c = b.build()
    assert c.node_domain("m1") == "magnetic" and c.node_domain("tj") != "magnetic"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 4e-4, dt=1e-7, switch_fn=fn)
    assert r.heat_injected > 0
    assert np.asarray(r.v("out"))[-1] > 1.0      # the transformer transfers power
    assert np.all(np.isfinite(np.asarray(r.states)))


def test_permeance_circuit_yaml_pickle_with_value_and_the_exact_route():
    b, fn = _magnetic_flyback(c_block=False, loss=False)
    c = b.build()
    c2 = p.Circuit.from_yaml(c.to_yaml())
    c3 = pickle.loads(pickle.dumps(c))
    assert c2.structure_hash() == c.structure_hash() == c3.structure_hash()
    assert c2.node_domain("m2") == "magnetic"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rs = [p.simulate(x, 1e-4, dt=1e-7, switch_fn=fn) for x in (c, c2, c3)]
    for r in rs[1:]:
        np.testing.assert_array_equal(np.asarray(rs[0].states), np.asarray(r.states))
    # with_value on a permeance changes the run (the value reaches the stamp)
    cw = c.with_value("Pc.C", 1.3e-6)
    assert cw.value_hash() != c.value_hash()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rw = p.simulate(cw, 1e-4, dt=1e-7, switch_fn=fn)
    assert not np.array_equal(np.asarray(rw.states), np.asarray(rs[0].states))


def test_a_linear_magnetic_circuit_on_the_exact_route():
    # winding on a gapped core behind a resistor: i(t) = V/R (1 - e^{-t R/L})
    mu0 = 4e-7 * math.pi
    pc, pg = mu0 * 2000.0 * 76e-6 / 72e-3, mu0 * 76e-6 / 0.5e-3
    L = 25 ** 2 / (1 / pc + 1 / pg)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "e", 10.0)
    b.add_winding("W", "e", "gnd", "m1", "gnd", 25)
    b.add_permeance("Pc", "m1", "m2", pc)
    b.add_air_gap("G", "m2", "gnd", 76e-6, 0.5e-3)
    t_end = 3 * L / 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end, engine="exact")
    i = np.asarray(r.i("R"))[-1]
    assert i == pytest.approx(1.0 * (1 - math.exp(-t_end * 10.0 / L)), rel=1e-6)


def test_impulse_energy_on_a_loss_port_device_is_warned_not_heat():
    """Batch 13 review: a loss port does not carry the 24 uJ an ideal
    switch's capacitor loop dissipates (E_impulse_total), and its electrical
    conduction integral across the impulse step is not physical. simulate()
    now warns (it was silent); the numbers are pinned below."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", 1e-6, 10.0)
    b.add_capacitor("C2", "y", "gnd", 3e-6, 2.0)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 2.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    t = p.LossTable(v_axis=[10.0], i_axis=[1.0], tj_axis=[25.0],
                    energy=np.full((1, 1, 1), 1e-5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.set_device_losses("S", junction="tj", E_on=t, E_off=t)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(b, t_end=5e-6 + 1e-10, dt=1e-7, discontinuity_restart=True,
                       switch_fn=lambda t: m, auto_regularize=False)
    imp = p.losses.impulse_losses(b, r)
    assert imp["S"]["E_impulse_total"] == pytest.approx(24e-6, rel=1e-9)
    assert any("do not carry their impulsive" in str(x.message) for x in w)
    # the heat is the table energy (1e-5 J per event) only, not the 24 uJ
    # Pinned gap: the switch is ON from t = 0 (no table event) and the
    # electrical conduction integral across the impulse step reads -38.6 uJ,
    # so the junction COOLS below ambient (24.96 degC). Owner: the in-kernel
    # event-losses track (skip v*i across a re-initialisation step and
    # deposit dW instead); a fix shows up here.
    e = r.device_energy("S")
    assert e["n_on"] == 0 and e["E_cond_electrical"] < 0.0
    assert r.heat_injected == pytest.approx(e["E_cond_electrical"], rel=1e-12)
