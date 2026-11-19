"""Integration batch 12, seams D and E: a loss port (set_device_losses)
meets the circuit IR and the snapshot.

E: the IR's YAML dialect (and so `Circuit` pickling, the process backend)
has no loss ports. It refuses by name instead of dropping the port, whose
heat was silently lost before (heat_injected 9 mJ -> 0). `with_value` and
thread sweeps keep the lowered pool, so they carry it. Carrying the port
through YAML (and into value_hash) is left to the IR track.

D: a run resumed from a snapshot does not carry the heat still pending
at the split (owner: the in-kernel event-losses track, its "resumed run"
gap). Today's behaviour is pinned so a fix shows up as a test change."""

import pickle
import warnings

import numpy as np
import pytest

import pulsim as p

FSW = 10e3


def _builder():
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
        warnings.simplefilter("ignore")   # one temperature knot
        b.set_device_losses("S1", junction="tj", E_on=t, E_off=t)
    return b


def _fn():
    return p.make_pwm_switch_fn(FSW, 0.5, 0, 1)


def _gate(t):
    m = p.SwitchStateMask(1)
    m.set(0, bool((t * FSW) % 1.0 < 0.5))
    return m


def _heat(x, **kw):
    return p.simulate(x, 5e-4, dt=1e-6, switch_fn=_fn(), **kw).heat_injected


def test_a_built_circuit_and_with_value_carry_the_loss_port():
    h = _heat(_builder())
    assert h == pytest.approx(9e-3, rel=1e-5)
    c = _builder().build()
    assert _heat(c) == h
    assert _heat(c.with_value("RL.R", 10.0)) == h


def test_yaml_and_pickle_refuse_a_loss_port_by_name():
    c = _builder().build()
    with pytest.raises(ValueError, match='"S1" has a loss port'):
        c.to_yaml()
    with pytest.raises(ValueError, match='"S1" has a loss port'):
        pickle.dumps(c)


def _kpi(res, *_):
    return {"heat": float(res.heat_injected)}


def test_sweep_of_a_circuit_carries_or_refuses_the_loss_port():
    c = _builder().build()
    kw = dict(params={"RL.R": [10.0, 20.0]}, kpi_fn=_kpi, verbose=False,
              t_end=5e-4, dt=1e-6, switch_fn=_fn())
    for parallel in (False, True):
        r = p.sweep(c, parallel=parallel, **kw)
        assert not r.failed
        assert list(r.kpis["heat"]) == pytest.approx([9e-3, 9e-3], rel=1e-5)
    # a picklable gate, so the circuit is the part the process path meets
    kw["switch_fn"] = _gate
    try:
        r = p.sweep(c, parallel=True, backend="process", n_workers=2, **kw)
    except Exception as e:  # noqa: BLE001 - the message is the check
        assert '"S1" has a loss port' in str(e)
    else:
        assert len(r.failed) == 2
        assert all('"S1" has a loss port' in str(m) for _, m in r.failed)


def test_a_resumed_run_drops_the_heat_pending_at_the_split():
    full = p.simulate(_builder(), 5e-4, dt=1e-6, switch_fn=_fn())
    a = p.simulate(_builder(), 2.503e-4, dt=1e-6, switch_fn=_fn())
    snap = pickle.loads(pickle.dumps(a.final_snapshot))   # the snapshot pickles
    b = p.simulate(_builder(), 5e-4, dt=1e-6, switch_fn=_fn(), resume_from=snap)
    assert a.heat_pending == pytest.approx(1e-3, rel=1e-5)
    lost = full.heat_injected - (a.heat_injected + b.heat_injected)
    # pinned gap: exactly the pending energy of the first segment is lost
    assert lost == pytest.approx(a.heat_pending, rel=1e-5)


# ---- seam C: every stateful stamp survives beside a loss port ----------------
def _stateful_with_loss(loss, coss=True):
    """Saturable inductor + nonlinear Coss + branch-form diode (+ loss port)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    if coss:
        b.add_nonlinear_capacitor("Coss", "vin", "sw", 1e-9, 25.0, 0.5)
    else:
        b.add_capacitor("Coss", "vin", "sw", 1e-9)
    b.add_diode("D", "gnd", "sw", form="branch", R_on=1e-3, g_off=1e-9)
    b.add_saturable_inductor("L", "sw", "out", 100e-6, 6.0, 1e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", 5.0)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 2.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    if loss:
        t = p.LossTable(v_axis=[48.0], i_axis=[1.0], tj_axis=[25.0],
                        energy=np.full((1, 1, 1), 1e-4))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.set_device_losses("S", junction="tj", E_on=t, E_off=t)
    fn = p.make_pwm_switch_fn(50e3, 0.4, b.switch_index_of("S"), b.graph.num_switches)
    return b, fn


def _end(engine, loss, coss=True):
    b, fn = _stateful_with_loss(loss, coss)
    kw = {"dt": 1e-7} if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 4e-4, engine=engine, switch_fn=fn, **kw)
    return r


def test_stateful_stamps_and_a_loss_port_compose_on_pwl():
    """Values from the tree before the unified device interface merged
    (e2e55a71, byte-equal there and here on macOS); 1e-9 holds them across
    platforms and is far below what a lost stamp moves (Coss as a linear
    capacitor moves v(out) by more, asserted)."""
    r, r0 = _end("pwl", True), _end("pwl", False)
    out = np.asarray(r.v("out"))
    assert out[-1] == pytest.approx(17.070994170735837, rel=1e-9)
    assert np.asarray(r.i("L"))[-1] == pytest.approx(3.387895462470447, rel=1e-9)
    assert np.asarray(r.v("tj"))[-1] == pytest.approx(28.557469446748136, rel=1e-9)
    assert r.heat_injected == pytest.approx(0.00392922192281983, rel=1e-9)
    # the loss port only heats the thermal node: the electrical trace is r0's
    # (bit-equal on macOS and Linux; MSVC differs by 9.6e-15 relative)
    np.testing.assert_allclose(out, np.asarray(r0.v("out")), rtol=1e-12, atol=0.0)
    lin = np.asarray(_end("pwl", True, coss=False).v("out"))[-1]
    assert abs(lin - out[-1]) > 1e3 * 1e-9 * abs(out[-1])


def test_stateful_stamps_compose_on_trbdf2_and_its_loss_port_is_refused():
    r = _end("trbdf2", False)
    assert np.asarray(r.v("out"))[-1] == pytest.approx(17.08541162982808, rel=1e-5)
    assert np.asarray(r.i("L"))[-1] == pytest.approx(3.3922212010816595, rel=1e-5)
    with pytest.raises(p.UnsupportedError, match="loss port of 'S'"):
        _end("trbdf2", True)


# ---- seam D: a Circuit and a snapshot both cross to process workers ----------
def _v_end(res, *_):
    return {"v": float(np.asarray(res.v("o"))[-1]), "t0": float(res.times[0])}


def test_a_circuit_and_a_snapshot_reach_process_workers():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "x", 1e3, 1e-9)
    b.add_resistor("R", "x", "o", 10.0)
    b.add_capacitor("C", "o", "gnd", 1e-5)
    c = b.build()
    snap = p.simulate(c, 2.5e-4, dt=1e-6, switch_fn=_gate).final_snapshot
    kw = dict(params={"R.R": [10.0, 20.0]}, kpi_fn=_v_end, verbose=False,
              t_end=5e-4, dt=1e-6, switch_fn=_gate, resume_from=snap)
    thr = p.sweep(c, parallel=True, **kw)
    prc = p.sweep(c, parallel=True, backend="process", n_workers=2, **kw)
    assert not thr.failed and not prc.failed and prc.backend == "process"
    assert list(prc.kpis["v"]) == list(thr.kpis["v"])
    assert list(prc.kpis["t0"]) == pytest.approx([2.5e-4, 2.5e-4], abs=1e-12)


# ---- seam D: controller state + the flat device-state snapshot ----------------
def _sat_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_nonlinear_capacitor("Coss", "in", "sw", 1e-9, 25.0, 0.5)
    b.add_saturable_inductor("L", "sw", "out", 1e-3, 6.0, 1e-4)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    loop = p.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05, output_max=0.95),
        measured="out", setpoint=12.0, switch="Q", freq=10e3)
    return b, loop


def test_closed_loop_with_stateful_devices_resumes_and_its_pickle_is_refused():
    """The control graph's resume state and the device interface's flat
    device-state vector (pickle version 3) in one snapshot."""
    t1, t2, dt = 1.53e-3, 3e-3, 1e-6
    bf, lf = _sat_buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = p.simulate(bf, t2, dt=dt, closed_loops=[lf])
        b, lp = _sat_buck()
        first = p.simulate(b, t1, dt=dt, closed_loops=[lp])
        snap = first.final_snapshot
        assert len(snap.dev_state) > 0 and snap._control_state is not None
        second = p.simulate(b, t2, dt=dt, closed_loops=[lp], resume_from=snap)
    assert [d for _, d in lp.duty_history] == [d for _, d in lf.duty_history]
    tf = np.asarray(full.times)
    xf = np.asarray(full.states)[tf >= second.times[0] - 0.5 * dt]
    xs = np.asarray(second.states)
    n = min(len(xf), len(xs))
    assert np.abs(xf[:n] - xs[:n]).max() <= 1e-9 * np.abs(xf).max()
    copy = pickle.loads(pickle.dumps(snap))
    assert list(copy.dev_state) == list(snap.dev_state)
    with pytest.raises(p.UnsupportedError, match="was pickled"):
        p.simulate(b, t2, dt=dt, closed_loops=[lp], resume_from=copy)


def test_a_twelve_field_version_3_pickle_still_loads():
    """The device-interface branch wrote version 3 with 12 fields; the
    control graph's marker made it 13. Both load; the marker refuses."""
    b, _ = _sat_buck()
    snap = p.simulate(b, 2e-4, dt=1e-6, switch_fn=lambda t, n=b.graph.num_switches: p.SwitchStateMask(n)
                      ).final_snapshot
    st = snap.__getstate__()
    # version 4 (batch 13): the 13 fields of version 3 + the grid
    assert st[:2] == ("SolverSnapshot", 4) and len(st) == 14 and st[12] is False
    old = type(snap).__new__(type(snap))
    old.__setstate__(("SolverSnapshot", 3) + tuple(st[2:12]))
    assert list(old.dev_state) == list(snap.dev_state)
    assert old._control_state is None


# ---- seam G: an offset rate group resumed on trbdf2 (event queue x G26) -------
def _offset_chain_run(t1):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 12.0)
    b.add_resistor("R", "in", "out", 10.0)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4, offset=3e-5))
    ch.add("g", p.Gain(k=1.0), inputs=dict(x="out"), output="u")
    obs = ch.make_step_observer(b)
    ch._cxx_chain.record_channel("u")
    first = p.simulate(b, t1, engine="trbdf2", step_observer=obs)
    return b, ch, obs, first


def test_trbdf2_resume_of_an_offset_group_checks_the_offset_grid():
    """Ticks at 30 us + k*100 us. The queue resumes them at t_start +
    offset + k*T: from 1.5 ms (on the grid, next tick 1.53 ms) it lands
    them; from 1.53 ms (a tick instant, but 30 us past the grid origin)
    they would fall at 1.56, 1.66 ms, so it is refused by name."""
    b, ch, obs, first = _offset_chain_run(1.5e-3)
    p.simulate(b, 2e-3, engine="trbdf2", step_observer=obs,
               resume_from=first.final_snapshot)
    t = np.asarray(ch._cxx_chain.get_recording_times())
    np.testing.assert_allclose(t, 3e-5 + np.arange(len(t)) * 1e-4, atol=1e-12)
    assert len(t) == 20
    b, ch, obs, first = _offset_chain_run(1.53e-3)
    with pytest.raises(p.UnsupportedError, match="between two"):
        p.simulate(b, 2e-3, engine="trbdf2", step_observer=obs,
                   resume_from=first.final_snapshot)


@pytest.mark.parametrize("engine", ("pwl", "trbdf2"))
def test_periodic_steady_state_by_shooting_refuses_a_loss_port(engine):
    """Each period of the transient map is a fresh run: it would drop the
    turn-on at the period start and the heat in flight at its end."""
    kw = {"dt": 1e-6} if engine == "pwl" else {}
    with pytest.raises(p.UnsupportedError, match="loss port of 'S1'") as ei:
        p.periodic_steady_state(_builder(), period=1 / FSW, switch_fn=_fn(),
                                engine=engine, **kw)
    assert ei.value.feature == "device_losses"
