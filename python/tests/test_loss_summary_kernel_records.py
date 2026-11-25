"""add-in-kernel-event-losses task 5.1: ``device_loss_summary`` and
``device_thermal_summary`` read kernel commutation records
(``result.loss_events``) when present.

Covers:
- reading ``res.loss_events`` across ``pwl``, ``trbdf2``, ``dsed`` / ``exact``
  (for switches and diodes, with and without ``t_from`` / ``t_to`` windowing);
- fixed-``Tj`` parity against the post-hoc ``switch_specs`` path (no behaviour
  change at fixed ``Tj``, identical key set, no double-counting);
- closed-loop ``Tj(t)`` fidelity when switching energies depend on temperature;
- ``device_thermal_summary`` consuming kernel ``res.loss_events`` cleanly;
- resumed runs filtering inherited pre-split events to the resumed window;
- untracked devices, circuits without ``set_device_losses``, conduction-only
  ports, and ideal branch-form switches.

Loss data here is synthetic.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

from .test_device_losses_kernel import T_AMB, _chopper, _table
from .test_exact_loss_port import E_OFF, E_ON, _buck


def _sim(b, t_end, fn, engine, **kw):
    if engine == "pwl_restart":    # the kernel reads a restart edge at its event limits
        engine, kw = "pwl", {**kw, "discontinuity_restart": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine=engine, **kw)


def _loss_by_name(b, res, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {r["name"]: r for r in p.device_loss_summary(b, res, **kw)}


def _thermal_by_name(b, res, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {r["name"]: r for r in p.device_thermal_summary(b, res, **kw)}


def _engine_kwargs(engine: str) -> dict:
    if engine in ("pwl", "pwl_restart"):
        return {"dt": 1e-7}
    if engine == "trbdf2":
        return {"rtol": 1e-8, "atol": 1e-10}
    return {}


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed", "exact"])
def test_device_loss_summary_reads_kernel_events_across_engines(engine):
    """Without switch_specs / diode_specs, device_loss_summary reads switching
    counts and energies directly from res.loss_events on every engine."""
    t_end = 202.5e-6
    b, fn, names = _buck()
    for n in names:
        b.set_device_losses(n, Tj=125.0, E_on=E_ON, E_off=E_OFF)
    res = _sim(b, t_end, fn, engine, **_engine_kwargs(engine))
    rows = _loss_by_name(b, res)
    span = float(res.times[-1] - res.times[0])

    for n in names:
        row = rows[n]
        e = res.device_energy(n)
        assert row["n_turn_on_events"] == e["n_on"] == 20
        assert row["n_turn_off_events"] == e["n_off"] == 20
        assert row["E_sw_on_total"] == pytest.approx(e["E_on"], rel=1e-12)
        assert row["E_sw_off_total"] == pytest.approx(e["E_off"], rel=1e-12)
        assert row["E_sw_total"] == pytest.approx(e["E_on"] + e["E_off"], rel=1e-12)
        assert row["P_sw_avg"] == pytest.approx((e["E_on"] + e["E_off"]) / span, rel=1e-12)
        expected_fsw = (
            float(e["n_off"]) / span
            if row["kind"] == "diode"
            else float(e["n_on"] + e["n_off"]) / (2.0 * span)
        )
        assert row["f_sw_estimate"] == pytest.approx(expected_fsw, rel=1e-12)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
def test_device_loss_summary_windowing_on_kernel_events(engine):
    """t_from / t_to filter res.loss_events with half-open [t_from, t_to)
    semantics, so adjacent windows add up even when a cut lands on an edge."""
    t_end = 202.5e-6
    b, fn, names = _buck()
    for n in names:
        b.set_device_losses(n, Tj=125.0, E_on=E_ON, E_off=E_OFF)
    res = _sim(b, t_end, fn, engine, **_engine_kwargs(engine))
    t = np.asarray(res.times)

    # Default whole-run matches explicit [t[0], t[-1]]
    whole_dflt = _loss_by_name(b, res)
    whole_expl = _loss_by_name(b, res, t_from=float(t[0]), t_to=float(t[-1]))
    for n in names:
        assert whole_dflt[n] == whole_expl[n]

    # Cut both inside steps and directly on a gate edge (100e-6 = 10 periods).
    cuts = [52.5e-6, 100.0e-6, 137.3e-6, 192.5e-6]
    span_whole = _loss_by_name(b, res, t_from=cuts[0], t_to=cuts[-1])
    parts = [_loss_by_name(b, res, t_from=a, t_to=c)
             for a, c in zip(cuts[:-1], cuts[1:])]

    # Truth: the gate schedule (100 kHz, D = 0.5, CCM; Q1 starts closed, no
    # event at t = 0). Q1 turns on at (k+1)*T and off at k*T + T/2; D1 the
    # other way round. The plain fixed-step engine samples switch_fn at the
    # end of a step and applies it over that step, so its commutations (and
    # i_L's kink) sit one dt earlier. An event belongs to a window by its
    # scheduled instant: the cut at 100 us (Q1 on, D1 off) books that edge in
    # the later window, on plain pwl (99.9 us) in the earlier one.
    T = 1e-5
    lag = 1e-7 if engine == "pwl" else 0.0
    k = np.arange(20)
    sched = {("Q1", "on"): (k + 1) * T - lag, ("Q1", "off"): k * T + T / 2 - lag,
             ("D1", "on"): k * T + T / 2 - lag, ("D1", "off"): (k + 1) * T - lag}
    for n in names:
        ev = res.loss_events(n)
        E_kind = {}
        for kind in ("on", "off"):
            sel = ev[ev["kind"] == kind]
            sel = sel[np.argsort(sel["t"])]
            s_k = sched[(n, kind)]
            assert len(sel) == len(s_k), (n, kind)
            assert np.abs(sel["t"] - s_k).max() < 1e-12, (n, kind)
            E_kind[kind] = (s_k, sel["E"])
        e_dev = res.device_energy(n)    # the kernel's own accumulator
        assert math.fsum(E_kind["on"][1]) == pytest.approx(e_dev["E_on"], rel=1e-12)
        assert math.fsum(E_kind["off"][1]) == pytest.approx(e_dev["E_off"], rel=1e-12)
        for a, c, part in zip(cuts[:-1], cuts[1:], parts):
            row = part[n]
            exp = {}
            for kind, (s_k, E) in E_kind.items():
                inside = (s_k >= a) & (s_k < c)
                exp[kind] = (int(np.count_nonzero(inside)), math.fsum(E[inside]))
            assert row["n_turn_on_events"] == exp["on"][0], (n, a, c)
            assert row["n_turn_off_events"] == exp["off"][0], (n, a, c)
            assert row["E_sw_on_total"] == pytest.approx(exp["on"][1], rel=1e-12, abs=1e-30)
            assert row["E_sw_off_total"] == pytest.approx(exp["off"][1], rel=1e-12, abs=1e-30)
            assert row["P_sw_avg"] == pytest.approx(
                (exp["on"][1] + exp["off"][1]) / (c - a), rel=1e-12, abs=1e-30)

        w = span_whole[n]
        assert sum(p_row[n]["n_turn_on_events"] for p_row in parts) == w["n_turn_on_events"]
        assert sum(p_row[n]["n_turn_off_events"] for p_row in parts) == w["n_turn_off_events"]
        for key in ("E_sw_on_total", "E_sw_off_total", "E_sw_total"):
            assert sum(p_row[n][key] for p_row in parts) == pytest.approx(w[key], rel=1e-12), (n, key)


@pytest.mark.parametrize("engine,rel_tol", [
    ("pwl", 1e-9),
    ("pwl_restart", 1e-9),
    ("trbdf2", 1e-5),
    ("dsed", 1e-9),
])
def test_fixed_tj_parity_with_posthoc_switch_specs(engine, rel_tol):
    """At fixed Tj, device_loss_summary on a circuit with set_device_losses
    (with or without switch_specs) matches the post-hoc switch_specs path on
    an identical circuit without set_device_losses — same keys and values."""
    t_end = 202.5e-6
    kw = _engine_kwargs(engine)

    b_plain, fn_plain, _ = _buck()
    res_plain = _sim(b_plain, t_end, fn_plain, engine, **kw)

    b_port, fn_port, _ = _buck()
    b_port.set_device_losses("Q1", Tj=125.0, E_on=E_ON, E_off=E_OFF)
    res_port = _sim(b_port, t_end, fn_port, engine, **kw)

    specs = {"Q1": {"E_on_table": E_ON, "E_off_table": E_OFF, "Tj": 125.0}}

    for win_kw in ({}, {"t_from": 52.5e-6, "t_to": 192.5e-6}):
        posthoc = _loss_by_name(b_plain, res_plain, switch_specs=specs, **win_kw)["Q1"]
        kernel_only = _loss_by_name(b_port, res_port, **win_kw)["Q1"]
        kernel_and_spec = _loss_by_name(b_port, res_port, switch_specs=specs, **win_kw)["Q1"]

        # Passing matching switch_specs alongside a loss port must be identical
        # (never double-counted).
        assert kernel_only == kernel_and_spec
        assert set(kernel_only.keys()) == set(posthoc.keys())
        assert kernel_only["n_turn_on_events"] == posthoc["n_turn_on_events"] > 10
        assert kernel_only["n_turn_off_events"] == posthoc["n_turn_off_events"] > 10
        for key in ("E_sw_on_total", "E_sw_off_total", "E_sw_total", "P_sw_avg",
                    "f_sw_estimate", "P_avg", "E_total"):
            assert kernel_only[key] == pytest.approx(posthoc[key], rel=rel_tol), (engine, win_kw, key)


def test_closed_loop_tj_kernel_events_preserved_in_summaries():
    """With closed-loop junction=, res.loss_events holds switching energies
    evaluated at the live Tj(t); device_loss_summary and device_thermal_summary
    preserve those energies instead of re-reading at a fixed temperature."""
    b, fn = _chopper(1.0, 0.01)
    tab = _table(1e-3, 2e-3)  # E(T) = 1 mJ * (1 + 0.01 * (T - 25))
    b.set_device_losses("S1", junction="tj", E_on=tab, E_off=tab)
    res = _sim(b, 0.05, fn, "pwl", dt=2e-6)

    ev = res.loss_events("S1")
    e_total_kernel = math.fsum(ev["E"])
    # Fixed Tj = 25 °C would give 1 mJ * len(ev); closed loop runs hotter (> 25 °C).
    assert e_total_kernel > 1.15 * (1e-3 * len(ev))

    row_no_spec = _loss_by_name(b, res)["S1"]
    assert row_no_spec["E_sw_total"] == pytest.approx(e_total_kernel, rel=1e-12)
    assert row_no_spec["n_turn_on_events"] + row_no_spec["n_turn_off_events"] == len(ev)

    # Even if a fixed-Tj switch_specs is also passed, the kernel's closed-loop
    # record takes precedence rather than overwriting with stale fixed-Tj values.
    stale_spec = {"S1": {"E_on_table": tab, "E_off_table": tab, "Tj": 25.0}}
    with pytest.warns(UserWarning, match="'S1'.*ignored"):
        rows = p.device_loss_summary(b, res, switch_specs=stale_spec)
    row_with_spec = {r["name"]: r for r in rows}["S1"]
    assert row_with_spec == row_no_spec

    # Steady-state window sees higher average switching power than early start-up.
    early = _loss_by_name(b, res, t_from=0.0, t_to=0.005)["S1"]
    late = _loss_by_name(b, res, t_from=0.04, t_to=0.05)["S1"]
    assert late["P_sw_avg"] > early["P_sw_avg"] * 1.15


@pytest.mark.parametrize("engine,rel_tol", [
    ("pwl", 1e-9),
    ("pwl_restart", 1e-9),
    ("trbdf2", 1e-5),
    ("dsed", 1e-9),
])
def test_device_thermal_summary_uses_kernel_events_and_matches_posthoc_at_fixed_tj(
        engine, rel_tol):
    """device_thermal_summary consumes kernel switching losses via
    res.loss_events (without switch_specs and without double-counting when
    switch_specs is passed), matching the post-hoc path at fixed Tj."""
    t_end = 202.5e-6
    kw = _engine_kwargs(engine)
    stages = [p.FosterStage(R_th_K_per_W=2.0, tau_s=5e-5)]
    t_specs = {"Q1": {"stages": stages}}
    s_specs = {"Q1": {"E_on_table": E_ON, "E_off_table": E_OFF, "Tj": 125.0}}

    b_plain, fn_plain, _ = _buck()
    res_plain = _sim(b_plain, t_end, fn_plain, engine, **kw)

    b_port, fn_port, _ = _buck()
    b_port.set_device_losses("Q1", Tj=125.0, E_on=E_ON, E_off=E_OFF)
    res_port = _sim(b_port, t_end, fn_port, engine, **kw)

    for win_kw in ({}, {"t_from": 52.5e-6, "t_to": 192.5e-6}):
        th_post = _thermal_by_name(
            b_plain, res_plain, thermal_specs=t_specs,
            switch_specs=s_specs, T_ambient_C=T_AMB, **win_kw)["Q1"]
        th_kern = _thermal_by_name(
            b_port, res_port, thermal_specs=t_specs,
            T_ambient_C=T_AMB, **win_kw)["Q1"]
        th_both = _thermal_by_name(
            b_port, res_port, thermal_specs=t_specs,
            switch_specs=s_specs, T_ambient_C=T_AMB, **win_kw)["Q1"]

        # No double-counting when switch_specs is also passed.
        assert th_kern["P_sw_avg"] == th_both["P_sw_avg"]
        assert th_kern["P_total_avg"] == th_both["P_total_avg"]
        assert th_kern["T_j_avg"] == th_both["T_j_avg"]
        np.testing.assert_array_equal(th_kern["T_j_trace"], th_both["T_j_trace"])

        # Matches post-hoc fixed-Tj thermal summary.
        assert th_kern["P_sw_avg"] > 100.0
        for key in ("P_cond_avg", "P_sw_avg", "P_total_avg", "T_j_avg", "T_j_peak"):
            assert th_kern[key] == pytest.approx(th_post[key], rel=rel_tol), (engine, win_kw, key)
        np.testing.assert_allclose(th_kern["T_j_trace"], th_post["T_j_trace"], rtol=rel_tol)


def test_diode_kernel_recovery_reaches_thermal_summary():
    """A PWL diode with a kernel E_off recovery table feeds its switching loss
    into both device_loss_summary and device_thermal_summary without diode_specs."""
    t_end = 202.5e-6
    b, fn, _ = _buck()
    b.set_device_losses("D1", Tj=125.0, E_off=E_OFF)
    res = _sim(b, t_end, fn, "pwl", dt=1e-7)

    loss = _loss_by_name(b, res)["D1"]
    e = res.device_energy("D1")
    span = float(res.times[-1] - res.times[0])
    assert loss["E_sw_on_total"] == 0.0
    assert loss["E_sw_off_total"] == pytest.approx(e["E_off"], rel=1e-12)
    assert loss["E_sw_total"] == pytest.approx(e["E_off"], rel=1e-12)
    assert loss["P_sw_avg"] == pytest.approx(e["E_off"] / span, rel=1e-12)
    assert loss["n_turn_off_events"] == e["n_off"] == 20

    th = _thermal_by_name(
        b, res,
        thermal_specs={"D1": {"stages": [p.FosterStage(1.0, 5e-5)]}},
        T_ambient_C=25.0,
    )["D1"]
    assert th["P_sw_avg"] == pytest.approx(loss["P_sw_avg"], rel=1e-12)
    assert th["P_total_avg"] == pytest.approx(th["P_cond_avg"] + loss["P_sw_avg"], rel=1e-12)
    assert th["T_j_avg"] > 25.0 + 0.5 * th["P_sw_avg"]


def test_untracked_devices_and_conduction_only_port_behaviour():
    """Devices without set_device_losses are completely unchanged; a
    conduction-only loss port (no E_on/E_off tables) reports 0 J without
    switch_specs and honours explicit post-hoc switch_specs when supplied."""
    t_end = 202.5e-6

    # 1. Circuit with NO set_device_losses at all: no switching keys without specs.
    b0, fn0, _ = _buck()
    r0 = _sim(b0, t_end, fn0, "pwl", dt=1e-7)
    rows0 = _loss_by_name(b0, r0)
    assert "P_sw_avg" not in rows0["Q1"]
    assert "P_sw_avg" not in rows0["D1"]

    # 2. Mixed: Q1 tracked, D1 untracked.
    b1, fn1, _ = _buck()
    b1.set_device_losses("Q1", Tj=125.0, E_on=E_ON, E_off=E_OFF)
    r1 = _sim(b1, t_end, fn1, "pwl", dt=1e-7)
    rows1 = _loss_by_name(b1, r1)
    assert rows1["Q1"]["P_sw_avg"] > 0.0
    assert "P_sw_avg" not in rows1["D1"]
    assert rows1["D1"] == rows0["D1"]

    # D1 still accepts post-hoc diode_specs on the mixed circuit.
    d_spec = {"D1": {"Q_rr": 5e-7}}
    rows1_d = _loss_by_name(b1, r1, diode_specs=d_spec)
    rows0_d = _loss_by_name(b0, r0, diode_specs=d_spec)
    assert rows1_d["D1"] == rows0_d["D1"]

    # 3. Conduction-only port on Q1 (no E_on / E_off tables).
    b2, fn2, _ = _buck()
    b2.set_device_losses("Q1", Tj=25.0)
    r2 = _sim(b2, t_end, fn2, "pwl", dt=1e-7)
    row_c_only = _loss_by_name(b2, r2)["Q1"]
    assert "P_sw_avg" not in row_c_only and "E_sw_total" not in row_c_only
    assert r2.device_energy("Q1")["n_on"] == r2.device_energy("Q1")["n_off"] == 20

    # Explicit post-hoc switch_specs on a conduction-only port is honoured.
    s_spec = {"Q1": {"E_on_ref": 1e-4, "E_off_ref": 1e-4, "V_ref": 24.0, "I_ref": 2.0}}
    row_c_spec = _loss_by_name(b2, r2, switch_specs=s_spec)["Q1"]
    row_plain_spec = _loss_by_name(b0, r0, switch_specs=s_spec)["Q1"]
    assert row_c_spec == row_plain_spec
    assert row_c_spec["E_sw_total"] > 0.0


@pytest.mark.parametrize("engine", ["pwl", "dsed"])
def test_resumed_run_filters_pre_split_kernel_events(engine):
    """A resumed run carries cumulative kernel events from the first leg;
    device_loss_summary restricts them to the resumed result's own span."""
    t_split, t_end = 102.5e-6, 202.5e-6
    kw = _engine_kwargs(engine)

    def make():
        b, fn, _ = _buck()
        b.set_device_losses("Q1", Tj=125.0, E_on=E_ON, E_off=E_OFF)
        return b, fn

    bf, fnf = make()
    full = _sim(bf, t_end, fnf, engine, **kw)
    b1, fn1 = make()
    first = _sim(b1, t_split, fn1, engine, **kw)
    b2, fn2 = make()
    second = _sim(b2, t_end, fn2, engine, resume_from=first.final_snapshot, **kw)

    # Resumed result holds cumulative events across [0, t_end)...
    assert len(second.loss_events("Q1")) == len(full.loss_events("Q1"))
    # ...but device_loss_summary reports only the events in [t_split, t_end).
    row_resumed = _loss_by_name(b2, second)["Q1"]
    row_window = _loss_by_name(bf, full, t_from=float(second.times[0]), t_to=t_end)["Q1"]
    assert row_resumed["n_turn_on_events"] == row_window["n_turn_on_events"] == 10
    assert row_resumed["n_turn_off_events"] == row_window["n_turn_off_events"] == 10
    assert row_resumed["E_sw_total"] == pytest.approx(row_window["E_sw_total"], rel=1e-9)
    assert row_resumed["P_sw_avg"] == pytest.approx(row_window["P_sw_avg"], rel=1e-9)


def test_ideal_branch_form_switch_with_loss_port():
    """An ideal branch-form switch with set_device_losses reports both its
    kernel switching losses and its impulsive re-initialisation energy."""
    c1, c2, v1, v2 = 1e-6, 3e-6, 10.0, 2.0
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", c1, v1)
    b.add_capacitor("C2", "y", "gnd", c2, v2)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(k, 1e-6 <= t < 2e-6)
        return m

    tab = _table(1e-5)
    b.set_device_losses("S", Tj=25.0, E_on=tab, E_off=tab)
    res = _sim(b, 5e-6, fn, "dsed", auto_regularize=False)

    row = _loss_by_name(b, res)["S"]
    dw = 0.5 * c1 * c2 / (c1 + c2) * (v1 - v2) ** 2
    assert row["form"] == "branch"
    assert row["n_turn_on_events"] == 1
    assert row["n_turn_off_events"] == 1
    assert row["E_sw_on_total"] == pytest.approx(1e-5, rel=1e-12)
    assert row["E_sw_off_total"] == pytest.approx(1e-5, rel=1e-12)
    assert row["E_sw_total"] == pytest.approx(2e-5, rel=1e-12)
    assert row["E_impulse_total"] == pytest.approx(dw, rel=1e-8)
    assert row["impulse_in_junction"] is True


def _kvi_table(k: float) -> p.LossTable:
    """Bilinear E(v, i, Tj) = k * v * i: strictly positive for v > 0, i > 0,
    and 0.0 J at ZVS (v = 0) or ZCS (i = 0)."""
    return p.LossTable(
        v_axis=[0.0, 50.0],
        i_axis=[0.0, 20.0],
        tj_axis=[25.0, 125.0],
        energy=np.array([
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [k * 1000.0, k * 1000.0]],
        ]),
    )


def _zero_table() -> p.LossTable:
    return p.LossTable(
        v_axis=[0.0, 50.0],
        i_axis=[0.0, 20.0],
        tj_axis=[25.0],
        energy=np.zeros((2, 2, 1)),
    )


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
def test_zvs_nonzero_table_is_not_overridden_by_switch_specs(engine):
    """Real ZVS (V = 0) with a non-zero LossTable evaluates to 0.0 J in the
    kernel; passing switch_specs alongside must keep 0.0 J and never override."""
    t_end = 202.5e-6
    kw = _engine_kwargs(engine)
    tab_kvi = _kvi_table(1e-6)  # 1 mJ at (50 V, 20 A), 0 J at V = 0
    # Non-vacuous caller switch_specs: constant 5 mJ per event even at V = 0.
    override_spec = {"S1": {"E_on_table": _table(5e-3), "E_off_table": _table(5e-3), "Tj": 25.0}}

    def make_zvs(with_port: bool):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", 0.0)
        b.add_switch("S1", "in", "sw", 1e3, 1e-9)
        b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
        b.add_inductor("L", "sw", "out", 100e-6)
        b.add_resistor("RL", "out", "gnd", 10.0)
        if with_port:
            b.set_device_losses("S1", Tj=25.0, E_on=tab_kvi, E_off=tab_kvi)
        fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("S1"), b.graph.num_switches)
        return b, fn

    # Control: without set_device_losses, override_spec produces positive energy.
    b_plain, fn_plain = make_zvs(with_port=False)
    r_plain = _sim(b_plain, t_end, fn_plain, engine, **kw)
    assert _loss_by_name(b_plain, r_plain, switch_specs=override_spec)["S1"]["E_sw_total"] > 0.1

    # With set_device_losses at ZVS, kernel records 0.0 J and wins over switch_specs.
    b_port, fn_port = make_zvs(with_port=True)
    r_port = _sim(b_port, t_end, fn_port, engine, **kw)
    rec = r_port._device_losses[0]
    assert rec.has_e_on is True and rec.has_e_off is True
    row_no_spec = _loss_by_name(b_port, r_port)["S1"]
    row_with_spec = _loss_by_name(b_port, r_port, switch_specs=override_spec)["S1"]
    assert row_no_spec["n_turn_on_events"] == 20
    assert row_no_spec["n_turn_off_events"] == 20
    assert row_no_spec["E_sw_total"] == 0.0
    assert row_with_spec == row_no_spec


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
def test_zero_valued_loss_table_is_not_overridden_by_specs(engine):
    """A zero-valued LossTable(0.0) on a live switching converter keeps 0.0 J
    even when caller switch_specs / diode_specs are also supplied."""
    t_end = 202.5e-6
    kw = _engine_kwargs(engine)
    ztab = _zero_table()
    s_spec = {"Q1": {"E_on_ref": 1e-3, "E_off_ref": 1e-3, "V_ref": 24.0, "I_ref": 2.0}}
    d_spec = {"D1": {"Q_rr": 1e-6}}

    b, fn, _ = _buck()
    b.set_device_losses("Q1", Tj=25.0, E_on=ztab, E_off=ztab)
    b.set_device_losses("D1", Tj=25.0, E_off=ztab)
    res = _sim(b, t_end, fn, engine, **kw)

    recs = {r.device: r for r in res._device_losses}
    assert recs["Q1"].has_e_on is True and recs["Q1"].has_e_off is True
    assert recs["D1"].has_e_on is False and recs["D1"].has_e_off is True

    rows_no_spec = _loss_by_name(b, res)
    rows_with_spec = _loss_by_name(b, res, switch_specs=s_spec, diode_specs=d_spec)
    for dev in ("Q1", "D1"):
        assert rows_no_spec[dev]["E_sw_total"] == 0.0
        assert rows_no_spec[dev]["P_sw_avg"] == 0.0
        assert rows_with_spec[dev] == rows_no_spec[dev]


def test_duck_typed_result_with_device_losses_but_no_loss_events_does_not_crash():
    """A duck-typed result carrying _device_losses without a callable
    loss_events() method falls back cleanly instead of raising AttributeError."""
    b, fn, _ = _buck()
    b.set_device_losses("Q1", Tj=125.0, E_on=E_ON, E_off=E_OFF)
    res = _sim(b, 52.5e-6, fn, "pwl", dt=1e-7)

    class _DuckResult:
        def __init__(self, r):
            self.times = np.asarray(r.times)
            self.states = np.asarray(r.states)
            self.switch_states = np.asarray(r.switch_states)
            self.engine_used = r.engine_used
            self._device_losses = list(r._device_losses)

    duck = _DuckResult(res)
    assert not hasattr(duck, "loss_events")

    rows = _loss_by_name(b, duck)
    assert "P_sw_avg" not in rows["Q1"]

    s_spec = {"Q1": {"E_on_table": E_ON, "E_off_table": E_OFF, "Tj": 125.0}}
    rows_spec = _loss_by_name(b, duck, switch_specs=s_spec)
    assert rows_spec["Q1"]["E_sw_total"] > 0.0

