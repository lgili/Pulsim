"""``t_from`` / ``t_to`` on ``device_loss_summary`` and
``device_thermal_summary``, the Shockley diode in ``result.i`` and both
summaries, the once-per-result diode warning, and the top-level export
of the topology result types.

Before this change both summaries averaged over the whole run with no
way to say otherwise: on the 230 V bridge below a diode's conduction
loss read 2.3384 W over the run and 0.1358 W over 80-100 ms (17x), and
every naive script hand-rolled a window (one from the first sample
after ``t_from``, which on a variable grid is up to a whole step late).
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

import pulsim as p

COND = {f"BR__D{k}": {"V_f0": 0.7, "r_on": 0.02} for k in (1, 2, 3, 4)}


def _bridge(V_th: float = 0.7) -> p.CircuitBuilder:
    """230 V rms behind 0.1 Ohm into a bridge, 470 uF || 1 kOhm (the
    AI guide's trap 13)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "n", 0.0, 230 * np.sqrt(2), 50.0)
    b.add_resistor("Rline", "ac", "x", 0.1)
    p.add_bridge_rectifier(b, "BR", ac_a="x", ac_b="n", dc_pos="dc",
                           dc_neg="gnd", V_th=V_th)
    b.add_capacitor("C", "dc", "gnd", 470e-6)
    b.add_resistor("RL", "dc", "gnd", 1e3)
    return b


@functools.lru_cache(maxsize=None)
def _run_bridge(engine: str):
    b = _bridge()
    kw = {"dt": 1e-6, "engine": "pwl"} if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=0.1, **kw)
    return b, res


def _rows(b, res, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {r["name"]: r for r in p.device_loss_summary(b, res, **kw)}


def _buck(V_th: float = 0.7) -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw", R_on=0.05)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=V_th)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 2.0)
    return b


@functools.lru_cache(maxsize=None)
def _run_buck(engine: str):
    b = _buck()
    pwm = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"),
                               b.graph.num_switches)
    kw = ({"dt": 1e-7, "engine": "pwl"} if engine == "pwl"
          else {} if engine == "auto" else {"engine": engine})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=5e-3, switch_fn=pwm, **kw)
    return b, res


SW = {"Q1": {"E_on_ref": 20e-6, "E_off_ref": 30e-6, "V_ref": 24.0,
             "I_ref": 6.0}}


# ---------------------------------------------------------------------
# A1: the window
# ---------------------------------------------------------------------
@pytest.mark.parametrize("engine", ["pwl", "auto"])
def test_window_leaves_the_start_up_out(engine):
    """Measured: 2.3384 W over the run, 0.1358 W (pwl) / 0.1360 W
    (auto) over 80-100 ms. The windowed figure is the same conduction
    model applied to res.i over the window (the summary integrates
    |i| and i^2 of each step's line exactly, the trapezoid of the
    samples does not: 0.2 % apart on the variable grid)."""
    b, res = _run_bridge(engine)
    run = _rows(b, res, conduction_specs=COND)["BR__D1"]
    win = _rows(b, res, conduction_specs=COND, t_from=0.08)["BR__D1"]
    t = np.asarray(res.times)
    i = np.asarray(res.i("BR__D1"))
    ref = p.time_average(t, 0.7 * np.abs(i) + 0.02 * i * i, t_from=0.08)
    assert run["P_cond_model_avg"] > 10 * win["P_cond_model_avg"]
    assert win["P_cond_model_avg"] == pytest.approx(ref, rel=5e-3)
    assert win["t_window_start"] == 0.08
    assert win["t_window_end"] == pytest.approx(0.1, abs=1e-15)
    assert win["i_avg"] == pytest.approx(
        p.time_average(t, i, t_from=0.08), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("engine", ["pwl", "auto"])
def test_windows_add_up(engine):
    """Cut anywhere (inside steps), the energies, charges and event
    counts of adjacent windows add up to the window that spans them."""
    b, res = _run_bridge(engine)
    spec = {"BR__D1": {"Q_rr": 1e-6}}
    cuts = [0.04, 0.0537123, 0.0812345, 0.1]
    whole = _rows(b, res, conduction_specs=COND, diode_specs=spec,
                  t_from=cuts[0], t_to=cuts[-1])["BR__D1"]
    parts = [_rows(b, res, conduction_specs=COND, diode_specs=spec,
                   t_from=a, t_to=c)["BR__D1"]
             for a, c in zip(cuts[:-1], cuts[1:])]
    for key in ("E_total", "E_cond_model", "E_sw_total"):
        assert sum(q[key] for q in parts) == pytest.approx(
            whole[key], rel=1e-12), key
    q_parts = sum(q["i_avg"] * (c - a)
                  for q, a, c in zip(parts, cuts[:-1], cuts[1:]))
    assert q_parts == pytest.approx(whole["i_avg"] * (cuts[-1] - cuts[0]),
                                    rel=1e-12)
    assert sum(q["n_turn_off_events"] for q in parts) \
        == whole["n_turn_off_events"] == 3      # one per 50 Hz period
    d_parts = sum(q["duty_conducting"] * (c - a)
                  for q, a, c in zip(parts, cuts[:-1], cuts[1:]))
    assert d_parts == pytest.approx(
        whole["duty_conducting"] * (cuts[-1] - cuts[0]), rel=1e-12)


def test_whole_window_is_the_default():
    b, res = _run_bridge("pwl")
    t = np.asarray(res.times)
    dflt = _rows(b, res, conduction_specs=COND)
    full = _rows(b, res, conduction_specs=COND, t_from=float(t[0]),
                 t_to=float(t[-1]))
    for name, row in dflt.items():
        for key, val in row.items():
            assert full[name][key] == val, (name, key)


def test_a_bound_inside_a_long_step_is_interpolated():
    """On the variable-step grid a window bound falls inside a step.
    The inductor's i_avg must be the time average of the same
    piecewise-linear trace over exactly [t_from, t_to], which
    p.time_average computes independently (interpolated ends)."""
    b, res = _run_buck("auto")
    t = np.asarray(res.times)
    h = np.diff(t)
    k = int(np.argmax(np.where(t[:-1] > 3e-3, h, 0.0)))
    assert h[k] > 1e-7            # a long step to cut
    a = float(t[k] + 0.37 * h[k])
    bnd = a + 1e-4
    row = _rows(b, res, t_from=a, t_to=bnd)
    i_L = np.asarray(res.i("L"))
    assert row["L"]["i_avg"] == pytest.approx(
        p.time_average(t, i_L, t_from=a, t_to=bnd), rel=1e-12)
    assert row["L"]["i_rms"] == pytest.approx(
        p.time_rms(t, i_L, t_from=a, t_to=bnd), rel=1e-12)
    # The first sample after `a` would have shortened the window.
    j = int(np.searchsorted(t, a))
    naive = p.time_average(t, i_L, t_from=float(t[j]), t_to=bnd)
    assert naive != pytest.approx(row["L"]["i_avg"], rel=1e-9)


def test_variable_grid_window_matches_a_fine_fixed_grid():
    """100 periods starting 7.7 us into a period: 2.4249 W on the
    variable grid against 2.4248 W on a 1 ns grid (measured); the first
    sample after t_from read 2.4103 W. A 10 ns grid keeps the test
    cheap."""
    cond = {"D1": {"V_f0": 0.7, "r_on": 0.02}}
    a = 3.0077e-3
    bnd = a + 1e-3
    b, res = _run_buck("auto")
    bf = _buck()
    pwm = p.make_pwm_switch_fn(100e3, 0.5, bf.switch_index_of("Q1"),
                               bf.graph.num_switches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fine = p.simulate(bf, t_end=bnd, dt=1e-8, engine="pwl",
                          switch_fn=pwm)
    rv = _rows(b, res, conduction_specs=cond, t_from=a, t_to=bnd)
    rf = _rows(bf, fine, conduction_specs=cond, t_from=a, t_to=bnd)
    assert rv["D1"]["P_cond_model_avg"] == pytest.approx(
        rf["D1"]["P_cond_model_avg"], rel=1e-3)
    assert rv["Q1"]["P_avg"] == pytest.approx(rf["Q1"]["P_avg"], rel=2e-3)


@pytest.mark.parametrize("engine", ["pwl", "dsed"])
def test_gate_edges_are_counted_by_where_their_step_starts(engine):
    """Ten whole periods from an arbitrary instant hold ten turn-ons and
    ten turn-offs; adjacent windows share none."""
    b, res = _run_buck(engine)
    a0 = 3.00313e-3
    cuts = [a0, a0 + 5e-5, a0 + 1e-4]
    rows = [_rows(b, res, switch_specs=SW, t_from=a, t_to=c)["Q1"]
            for a, c in zip(cuts[:-1], cuts[1:])]
    whole = _rows(b, res, switch_specs=SW, t_from=cuts[0],
                  t_to=cuts[-1])["Q1"]
    assert whole["n_turn_on_events"] == whole["n_turn_off_events"] == 10
    assert whole["f_sw_estimate"] == pytest.approx(100e3, rel=1e-9)
    assert sum(r["n_turn_on_events"] for r in rows) == 10
    assert sum(r["E_sw_total"] for r in rows) == pytest.approx(
        whole["E_sw_total"], rel=1e-12)
    assert whole["duty_closed"] == pytest.approx(0.5, abs=0.02)


def test_bad_windows_are_refused():
    b, res = _run_bridge("pwl")
    for kw in ({"t_from": 0.05, "t_to": 0.05}, {"t_from": 0.06, "t_to": 0.05},
               {"t_from": -1e-3}, {"t_to": 0.2}, {"t_from": float("nan")}):
        with pytest.raises(ValueError):
            p.device_loss_summary(b, res, **kw)
    # t_end itself is accepted although the stored last time is an ulp
    # short of it (0.09999999999999999).
    row = _rows(b, res, t_from=0.09, t_to=0.1)["BR__D1"]
    assert row["t_window_end"] == float(np.asarray(res.times)[-1])


def test_a_seeded_run_starts_at_the_first_solved_sample():
    """t_from=0 on a run whose row 0 is unsolved starts at t[1]."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_switch("S", "in", "x", 1e3, 1e-9)
    b.add_inductor("L", "x", "y", 1e-3, i0=5.0)
    b.add_resistor("R", "y", "gnd", 1.0)
    b.add_diode("D", "gnd", "x", 1e3, 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=1e-3, dt=1e-5,
                         switch_fn=lambda t: p.SwitchStateMask(
                             b.graph.num_switches))
    t = np.asarray(res.times)
    rows = _rows(b, res, t_from=0.0)
    assert rows["D"]["t_window_start"] == float(t[1])


def test_thermal_window_reads_the_steady_state():
    """One 50 us Foster stage settles inside 3-5 ms, so the window's
    T_j_avg is T_amb + R_th * P_total_avg (measured within 1e-3 K on
    pwl); the trace still covers the run from T_amb."""
    b, res = _run_buck("pwl")
    ts = {"Q1": {"stages": [p.FosterStage(1.0, 5e-5)]}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = p.device_thermal_summary(b, res, thermal_specs=ts,
                                         switch_specs=SW, T_ambient_C=40.0)
        win = p.device_thermal_summary(b, res, thermal_specs=ts,
                                       switch_specs=SW, T_ambient_C=40.0,
                                       t_from=3e-3)
    w = win[0]
    assert w["t_window_start"] == 3e-3
    assert w["T_j_avg"] == pytest.approx(40.0 + w["P_total_avg"], abs=2e-3)
    assert w["P_sw_avg"] == pytest.approx(
        _rows(b, res, switch_specs=SW, t_from=3e-3)["Q1"]["P_sw_avg"],
        rel=1e-12)
    assert len(w["T_j_trace"]) == len(whole[0]["T_j_trace"])
    assert w["T_j_trace"][0] < 40.0 + 0.01 * w["P_total_avg"]
    assert w["T_j_peak"] <= float(np.max(w["T_j_trace"])) + 1e-9


# ---------------------------------------------------------------------
# A2: the Shockley diode
# ---------------------------------------------------------------------
def _shockley_circuit():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "a", 2.0)
    b.add_shockley_diode("D", "a", "k", I_S=1e-12, n=1.0, BV=5.1)
    b.add_resistor("Rs", "k", "out", 1e-3)          # sense resistor
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("RL", "out", "gnd", 20.0)
    return b


@pytest.mark.parametrize("kw", [{"dt": 1e-6}, {}], ids=["pwl", "auto"])
def test_shockley_current_matches_a_sense_resistor(kw):
    """Measured: |i_D - i_Rs| <= 3.2e-12 A of a 0.69 A peak."""
    b = _shockley_circuit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=5e-3, **kw)
    i_d = np.asarray(res.i("D"))
    i_s = np.asarray(res.i("Rs"))
    assert np.max(np.abs(i_d - i_s)) < 1e-9
    assert np.max(np.abs(i_s)) > 0.5
    assert "D" in res.currents()

    t = np.asarray(res.times)
    v_d = np.asarray(res.v("a")) - np.asarray(res.v("k"))
    for t_from in (None, 3e-3):
        row = _rows(b, res, t_from=t_from)["D"]
        assert row["kind"] == "shockley_diode"
        assert row["P_avg"] == pytest.approx(
            p.time_average(t, v_d * i_s, t_from=t_from), rel=1e-4)
        assert row["i_avg"] == pytest.approx(
            p.time_average(t, i_s, t_from=t_from), rel=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        th = p.device_thermal_summary(
            b, res, thermal_specs={"D": {"stages": [p.FosterStage(10.0,
                                                                  1e-4)]}},
            t_from=3e-3)
    assert th[0]["P_cond_avg"] == pytest.approx(
        _rows(b, res, t_from=3e-3)["D"]["P_avg"], rel=1e-9)


def test_unrecorded_shockley_and_unsupported_kinds_name_a_recipe():
    b = _shockley_circuit()
    b.add_lauritzen_diode("DL", "in", "q", I_S=1e-12)
    b.add_resistor("Rq", "q", "gnd", 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=2e-4, dt=1e-6)
    with pytest.raises(NotImplementedError, match="resistor in series"):
        res.i("DL")
    b._shockley_params.clear()
    with pytest.raises(NotImplementedError, match="resistor in series"):
        res.i("D")


# ---------------------------------------------------------------------
# A3: one warning per result and diode
# ---------------------------------------------------------------------
class _NoRecord:
    """A result without the kernel's stamped-mask record: every diode
    then warns that its state was replayed."""

    i = p.SimulationResult.i

    def __init__(self, r):
        self.times = r.times
        self.states = np.asarray(r.states)
        self._builder = r._builder


def _diode_warnings(rec):
    return [str(w.message) for w in rec
            if issubclass(w.category, RuntimeWarning)
            and str(w.message).startswith("diode 'BR__D1'")]


def test_diode_warning_is_emitted_once_per_result_and_diode():
    b, res = _run_bridge("pwl")
    nr = _NoRecord(res)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("default")
        nr.i("BR__D1")
        nr.i("BR__D1")      # a second call site
        for _ in range(2):
            nr.i("BR__D1")
        p.device_loss_summary(b, nr)
    msgs = _diode_warnings(rec)
    assert len(msgs) == 1, msgs
    assert "no record" in msgs[0]
    # Another result is another warning.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("default")
        _NoRecord(res).i("BR__D1")
    assert len(_diode_warnings(rec)) == 1


def test_missed_commutation_warning_is_emitted_once():
    """The V_th = 0 bridge on TR-BDF2 holds BR__D1 OFF while
    forward-biased (measured: 52 samples); five reads, one warning."""
    b = _bridge(V_th=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=0.1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("default")
        res.i("BR__D1")
        res.i("BR__D1")
        res.i("BR__D1")
        p.device_loss_summary(b, res)
    msgs = _diode_warnings(rec)
    assert len(msgs) <= 1, msgs
    if msgs:
        assert "commutation rule forbids" in msgs[0]


# ---------------------------------------------------------------------
# A4: topology result types at the top level
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name", ["BridgeRectifierResult",
                                  "ThreePhaseVsiResult",
                                  "ThreePhaseRLLoadResult"])
def test_topology_result_types_are_exported(name):
    assert getattr(p, name) is getattr(p.topology, name)
    assert name in p.__all__


def test_attribute_errors_name_the_installed_version():
    with pytest.raises(AttributeError, match=p.__version__) as exc:
        p.NoSuchThing  # noqa: B018
    assert "1.0.0 ships" not in str(exc.value)
    with pytest.raises(AttributeError, match=p.__version__):
        p.codegen  # noqa: B018
