"""Reference benchmark harness (openspec change restore-reference-benchmark-harness).

Metrics, the v1 reader and its triage manifest, the analytic reference set,
the golden checkpoints of the fixed engine and the counter gate. Every
assertion here is time-weighted or exact-integer; none reads a clock.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim

pytestmark = pytest.mark.reference  # CI job "reference-smoke"; also part of plain `pytest`

yaml = pytest.importorskip("yaml")
pytest.importorskip("scipy")

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path[:0] = [str(BENCH), str(BENCH / "tools")]

import counters  # noqa: E402
import differential_runner as dr  # noqa: E402
import goldens  # noqa: E402
import metrics  # noqa: E402
import v1_reader  # noqa: E402
from analytic import BY_ID, CASES  # noqa: E402

ANALYTIC = yaml.safe_load((BENCH / "manifests" / "analytic.yaml").read_text(encoding="utf-8"))["cases"]
TRIAGE = yaml.safe_load((BENCH / "manifests" / "v2_triage.yaml").read_text(encoding="utf-8"))["circuits"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------- metrics ---

def test_time_weighted_mean_on_a_clustered_grid():
    t = np.concatenate([np.linspace(0, 0.01, 1000), np.linspace(0.01, 1, 11)[1:]])
    assert t.mean() == pytest.approx(0.0104, abs=5e-5)          # what a sample mean says
    assert metrics.time_mean(t, t) == pytest.approx(0.5, abs=1e-12)


def test_err_rms_identity_and_known_offset():
    t = np.linspace(0, 1, 501) ** 2                                 # non-uniform grid
    y = np.sin(3 * t)
    span = y.max() - y.min()
    assert metrics.err_rms(t, y, (t, y)) == 0.0
    assert metrics.err_rms(t, y + 0.1 * span, (t, y)) == pytest.approx(0.1, abs=1e-12)
    assert metrics.err_rms(t, y + 0.1 * span, lambda s: np.sin(3 * s)) == pytest.approx(0.1, rel=1e-6)


def test_edge_measure_drops_only_the_steps_that_hold_a_jump():
    # The same unit step at t = 0.5, sampled densely and sparsely: both are
    # exact at their samples; interpolating the sparse one across its edge
    # step invents an error, the edge measure does not.
    tr = np.linspace(0.0, 1.0, 1001)
    yr = (tr >= 0.5).astype(float)
    tr, yr = np.insert(tr, 500, 0.5), np.insert(yr, 500, 0.0)    # left limit
    ts = np.array([0.0, 0.25, 0.45, 0.55, 0.75, 1.0])
    ys = (ts >= 0.5).astype(float)
    assert metrics.err_rms(ts, ys, (tr, yr)) > 0.05
    assert metrics.err_rms(ts, ys, (tr, yr), exclude=[(0.45, 0.55)]) == 0.0
    assert metrics.err_max(ts, ys, (tr, yr), exclude=[(0.45, 0.55)]) == 0.0
    # an error outside the excluded step is still seen, weighted by the kept window
    assert metrics.err_rms(ts, ys + 0.1, (tr, yr), exclude=[(0.45, 0.55)]) == pytest.approx(0.1)


def test_gate_observables_keep_source_forced_nodes():
    # The gate compares everything; only the edge diagnostic drops them.
    case = v1_reader.load(BENCH / "circuits/switching/buck/ll06_buck_vcswitch.yaml")
    assert dr.source_forced(case) == {"V(vin)", "V(ctrl)"}
    assert {"V(vin)", "V(ctrl)", "V(sw)", "I(L1)"} <= set(dr.observables_of(case))


def test_switching_steps_bracket_every_recorded_change():
    case = v1_reader.load(BENCH / "circuits/switching/buck/ll06_buck_vcswitch.yaml")
    r = case.run("pwl")
    t = np.asarray(r.times)
    s = dr.switch_record(r, t)
    steps = dr.switching_steps(s, t)
    assert len(steps) == int(np.any(s[1:] != s[:-1], axis=1).sum()) > 100
    assert all(b - a == pytest.approx(case.dt, rel=1e-6) for a, b in steps)


def _sw(t, on):
    return np.asarray(on, dtype=int).reshape(-1, 1)


def test_one_step_late_edge_on_a_resistive_load_disagrees_with_one_step_of_timing():
    # R load behind an ideal switch: the output IS the switch state. The
    # reference lands the edge at 0.5; the engine (dt = 0.02) applies it one
    # step late. Inside the excluded step the late edge is invisible, so the
    # edge diagnostic reads 0: only the gate and the timing column show it.
    tr_ = np.union1d(np.linspace(0.0, 1.0, 1001), [0.5 + 1e-9])
    ref = (tr_, {"V(out)": (tr_ > 0.5) * 1.0}, _sw(tr_, tr_ > 0.5))
    dt = 0.02
    t = np.linspace(0.0, 1.0, 51)
    eng = (t, {"V(out)": (t > 0.5 + dt / 2) * 1.0}, _sw(t, t > 0.5 + dt / 2))
    er, _ = dr.compare(eng, ref, ["V(out)"], 1e-3)
    assert er == pytest.approx(np.sqrt(dt / 3 / 0.98), rel=1e-3)
    assert er > dr.default_tol("pwl", True)                         # DISAGREE on the gate
    d = dr.diagnose(eng, ref, ["V(out)"], 1e-3, step=dt)
    assert d["edge"] == "ok" and d["edge_rms"] == pytest.approx(0.0, abs=1e-12)
    assert d["excluded_frac"] == pytest.approx((dt + 1e-9) / 0.98, rel=1e-6)
    assert d["edge_timing_steps"] == pytest.approx(1.0, abs=1e-6)


def test_a_chattering_trace_is_reported_chatter_not_agree():
    # The reference conducts throughout; the engine's diode mask flips on
    # every step of [0.3, 0.7] and is wrong on half of them. Dropping the
    # switching steps would drop exactly the wrong intervals.
    tr_ = np.linspace(0.0, 1.0, 1001)
    ref = (tr_, {"y": np.ones_like(tr_)}, _sw(tr_, np.ones_like(tr_)))
    t = np.linspace(0.0, 1.0, 101)
    on = np.where((t > 0.3) & (t < 0.7), np.arange(t.size) % 2, 1)
    eng = (t, {"y": on * 1.0}, _sw(t, on))
    er, _ = dr.compare(eng, ref, ["y"], 1e-3)
    assert er > dr.default_tol("pwl", True)
    d = dr.diagnose(eng, ref, ["y"], 1e-3)
    assert d["edge"] == "chatter" and d["edge_rms"] is None
    assert d["excluded_frac"] > 0.3


def test_a_trace_that_never_switches_is_missed_edges_not_chatter():
    t = np.linspace(0.0, 1.0, 101)
    ref = (t, {"y": np.ones_like(t)}, _sw(t, (t * 4).astype(int) % 2))  # 3 edges
    eng = (t, {"y": np.ones_like(t)}, _sw(t, np.zeros_like(t)))
    assert dr.diagnose(eng, ref, ["y"], 1e-3)["edge"] == "missed edges"


def test_union_of_excluded_steps_covering_the_span_does_not_raise():
    # Each trace alone switches on less than half its span; together they
    # cover it all (the old measure raised "every interval excluded").
    t = np.linspace(0.0, 1.0, 21)
    a = (t, {"y": np.ones_like(t)}, _sw(t, (t > 0.5) | ((t * 20).round() % 2 == 1) & (t < 0.5)))
    b = (t, {"y": np.ones_like(t)}, _sw(t, (t < 0.5) | ((t * 20).round() % 2 == 1) & (t > 0.5)))
    d = dr.diagnose(a, b, ["y"], 1e-3)
    assert d["edge"] == "undersampled" and d["edge_rms"] is None
    assert d["excluded_frac"] > 0.9


def test_exclude_t0_drops_only_the_initial_sample():
    t = np.linspace(0, 1, 11)
    y = np.ones_like(t)
    y[0] = 0.0                                                      # algebraic node read as 0 at t=0
    assert metrics.err_rms(t, y, lambda s: np.ones_like(s), abs_scale=1.0) > 0.1
    assert metrics.err_rms(t, y, lambda s: np.ones_like(s), abs_scale=1.0, exclude_t0=True) == 0.0


# ---------------------------------------------------------------- reader ---

def test_bridge_resistive_pure_through_the_reader():
    case = v1_reader.load(BENCH / "circuits/linear/bridge_resistive_pure.yaml")
    r = case.run("pwl")
    assert v1_reader.observable(r, "V(a)")[-1] == pytest.approx(6.6667, abs=1e-4)
    assert abs(r.v("a")[-1] - 20 / 3) < 1e-6 and abs(r.v("b")[-1] - 4.0) < 1e-6


def test_vcswitch_pwm_gate_becomes_a_native_schedule():
    # S1's control node is a 5 kHz, 50 % pwm source: period 200 us, duty 0.5
    case = v1_reader.load(BENCH / "circuits/switching/basic/chopper_with_rc_snubber.yaml")
    assert "S1: schedule f=5000 Hz duty=0.5 start=0 s" in case.notes
    _, sw, _ = case.build()
    assert isinstance(sw, pulsim.SwitchSchedule)
    assert sw.next_edge_after(0.0) == pytest.approx(100e-6, rel=1e-12)
    assert sw.next_edge_after(100e-6) == pytest.approx(200e-6, rel=1e-12)


def test_v1_pwm_phase_is_a_period_fraction_and_dead_time_shortens_on_time():
    # high while (t + phase*T) mod T < duty*T - dead_time. Re-pin (batch 6b): the v1 kernel
    # read radians, but every file that sets a pwm phase means a fraction (0.5 on a leg's
    # low side, 1/3 and 2/3 on three phases); as radians ll09/ll10 modelled shoot-through.
    f, duty, start = v1_reader._on_windows(
        {"type": "pwm", "frequency": 1e3, "duty": 0.5, "phase": 0.25, "dead_time": 1e-4}, 0.5)
    assert (f, duty) == (1e3, pytest.approx(0.4))
    assert start == pytest.approx(0.75e-3)


def test_v1_ideal_transformer_ratio_is_np_over_ns():
    # turns_ratio 0.5 = N_p/N_s: a 2x step-up
    case = v1_reader.load(BENCH / "circuits/magnetics/transformer_step_up.yaml")
    b, _, _ = case.build()
    r = case.run("pwl")
    pri = [c for c in case._components if c["type"] == "transformer"][0]["nodes"][0]
    k = len(r.times) // 4
    np.testing.assert_allclose(r.v("sec")[k:], 2 * np.asarray(r.v(pri))[k:], atol=1e-9)


def test_v1_current_source_drives_its_first_node():
    # 1 mA into `out` through 5 kOhm || C settles at +5 V (v1 and 2.0 agree)
    r = v1_reader.load(BENCH / "circuits/linear/current_source_rc.yaml").run("pwl")
    assert r.v("out")[-1] == pytest.approx(5.0, rel=1e-9)


def test_unmapped_construct_is_named():
    with pytest.raises(v1_reader.Unmapped, match="pmos"):
        v1_reader.load(BENCH / "circuits/switching/buck/buck_pmos.yaml")


# ---------------------------------------------------------------- triage ---

def test_triage_lists_every_circuit_on_disk():
    on_disk = {p.relative_to(BENCH).as_posix() for p in (BENCH / "circuits").rglob("*.yaml")}
    assert len(on_disk) == 126
    assert set(TRIAGE) == on_disk
    assert {e["status"] for e in TRIAGE.values()} <= {"runs", "skipped", "error"}
    for rel, e in TRIAGE.items():
        if e["status"] != "runs":
            assert e.get("reason"), rel


def test_every_known_cell_names_a_cause_that_resolves():
    """Task 3.2: a frozen disagree / error cell carries `cause: [id, ...]`; each id is a
    `- **id**` entry of KNOWN_LIMITATIONS.md's "Triage causes" (no outward issue tracker)."""
    kl = (BENCH.parent / "KNOWN_LIMITATIONS.md").read_text(encoding="utf-8")
    sec = kl[kl.index("#### Triage causes"):]
    sec = sec[:sec.index("\n###", 1)]
    defined = set(re.findall(r"^- \*\*(TC-[A-Z-]+)\*\*", sec, flags=re.M))
    assert len(defined) >= 5
    used = set()
    for rel, e in TRIAGE.items():
        if "known" in e or "known_err" in e:
            assert e.get("cause"), f"{rel}: a known cell without a cause"
            assert set(e["cause"]) <= defined, (rel, set(e["cause"]) - defined)
            used |= set(e["cause"])
        else:
            assert "cause" not in e, rel
    assert used == defined, defined - used          # no orphan cause


@pytest.mark.parametrize("rel", sorted(TRIAGE))
def test_triage_entry_is_true(rel):
    e = TRIAGE[rel]
    if e["status"] == "skipped":
        with pytest.raises(v1_reader.Unmapped) as ex:
            v1_reader.load(BENCH / rel)
        assert e["reason"] in str(ex.value)
        return
    case = v1_reader.load(BENCH / rel)
    if e["status"] == "error":
        with pytest.raises(Exception, match=e["match"]):
            case.run("pwl")
        return
    r = case.run("pwl")
    assert r.times[-1] == pytest.approx(case.t_end, rel=1e-6)
    for col in dr.observables_of(case):
        assert np.all(np.isfinite(v1_reader.observable(r, col))), col


# ------------------------------------------------------- analytic set ---

def _err(case, engine, k=1.0):
    r = case.run(engine, k)
    return metrics.err_rms(np.asarray(r.times), case.observe(r), case.exact,
                           abs_scale=case.abs_scale, exclude_t0=case.exclude_t0)


def test_every_analytic_case_is_unique_and_has_a_manifest_entry():
    # The count derives from the catalog: tracks that add cases (IGBT,
    # machines, permeance, ...) no longer collide on a literal.
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)) >= 26
    assert set(ANALYTIC) == {c.id for c in CASES}


def test_closed_form_self_checks():
    t = np.linspace(0, 10e-3, 400001)
    v = BY_ID["rlc_underdamped"].exact(t)
    wd = math.sqrt(1e8 - 500.0 ** 2)
    assert wd == pytest.approx(9987.49, abs=0.01)
    assert v.max() == pytest.approx(18.545, abs=1e-3)
    assert t[v.argmax()] == pytest.approx(math.pi / wd, abs=5e-8)
    assert BY_ID["rlc_underdamped"].exact(np.array([1.0]))[0] == pytest.approx(10.0, abs=1e-12)
    assert BY_ID["rlc_underdamped"].checks["peak"] == pytest.approx(18.545, abs=1e-3)
    assert BY_ID["wheatstone"].checks == {"V(a)": pytest.approx(6.6667, abs=1e-4), "V(b)": 4.0}
    r = BY_ID["pwl_diode_drop"].run("pwl")
    assert r.v("a")[-1] == pytest.approx(1.2, abs=1e-12)
    assert r.v("a")[-1] * 10.0 == pytest.approx(12.0, abs=1e-10)        # P = V I
    assert BY_ID["igbt_knee"].checks == {"V": pytest.approx(1.1, abs=1e-12), "P": 11.0}
    r = BY_ID["igbt_knee"].run("pwl")
    assert r.v("c")[-1] == pytest.approx(1.1, abs=1e-12)               # V_ce0 + r_ce I


def _cells():
    for c in CASES:
        for e, spec in ANALYTIC[c.id].items():
            yield pytest.param(c.id, e, spec, id=f"{c.id}-{e}")


@pytest.mark.parametrize("cid,engine,spec", list(_cells()))
def test_analytic(cid, engine, spec):
    case = BY_ID[cid]
    if "refused" in spec:
        with pytest.raises(Exception):
            case.run(engine)
        return
    if "known" in spec:
        # a recorded defect: it must still fail to agree; once fixed, give it a ceiling
        try:
            e = _err(case, engine)
        except Exception:  # noqa: BLE001 - the recorded error is still there
            assert spec["known"] == "error"
            return
        assert e > 1e-3, f"{cid}/{engine} now agrees ({e:.2e}): replace `known` by a ceiling"
        return
    assert _err(case, engine) <= spec["ceiling"]


@pytest.mark.parametrize("cid,engine,spec", [p for p in _cells() if "ceiling" in p.values[2]])
def test_analytic_detects_a_one_percent_component_error(cid, engine, spec):
    case = BY_ID[cid]
    try:
        e = _err(case, engine, k=1.01)
    except Exception:  # noqa: BLE001 - a run that breaks is detected too
        return
    assert e > spec["ceiling"], f"{cid}/{engine}: 1 % error gives {e:.3e} <= ceiling"


def test_analytic_ceilings_follow_the_policy():
    floor = {"pwl": 1e-9, "trbdf2": 1e-9, "dsed": 1e-6}
    for cid, cells in ANALYTIC.items():
        for e, s in cells.items():
            if "ceiling" in s and "reason" not in s:
                assert s["ceiling"] <= max(2.0 * s["measured"], floor[e]) * (1 + 1e-2), (cid, e)
    try:
        prev = subprocess.run(["git", "show", "HEAD:benchmarks/manifests/analytic.yaml"],
                              cwd=BENCH, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return  # no git history here: the in-file rule above still holds
    for cid, cells in yaml.safe_load(prev)["cases"].items():
        for e, s in cells.items():
            now = ANALYTIC.get(cid, {}).get(e, {})
            if "ceiling" in s and "ceiling" in now and "reason" not in now:
                assert now["ceiling"] <= s["ceiling"], f"{cid}/{e} loosened without a reason"


def test_rc_charge_agrees_on_three_engines_within_the_spec_ceilings():
    c = BY_ID["rc_charge"]
    assert _err(c, "pwl") <= 3.2e-3
    assert _err(c, "trbdf2") <= 1.2e-6
    assert _err(c, "dsed") <= 1e-6


# ------------------------------------------------------------- goldens ---

GOLDEN = list(goldens.cases())


def test_forty_goldens():
    assert len(GOLDEN) >= 40
    assert {p.stem for p in goldens.GOLDEN_DIR.glob("*.json")} == {g[0] for g in GOLDEN}


@pytest.mark.parametrize("name,run,names,newton", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden(name, run, names, newton):
    golden = json.loads((goldens.GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))
    assert golden["rtol"] <= (goldens.RTOL_NEWTON if newton else goldens.RTOL_LINEAR)
    bad = goldens.compare(golden, goldens.snapshot(run(), names, newton))
    assert not bad, "\n".join(bad[:10])


def test_golden_catches_one_part_per_million():
    c = BY_ID["rc_charge"]
    golden = json.loads((goldens.GOLDEN_DIR / "analytic__rc_charge.json").read_text(encoding="utf-8"))
    b, _ = c.build(1.0 + 1e-6)
    res = pulsim.simulate(b, c.t_end, c.dt, engine="pwl")
    bad = goldens.compare(golden, goldens.snapshot(res, b.state_var_names(), False))
    assert bad and "rel diff" in bad[0] and "V(o)" in "".join(bad)


def test_golden_catches_a_changed_switching_sequence_before_values():
    name, run, names, newton = next(g for g in GOLDEN if g[0] == "buck_switching")
    golden = json.loads((goldens.GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))
    res = run()

    class Shifted:  # the diode commutates one step earlier
        def __getattr__(self, k):
            return getattr(res, k)
        switch_states = np.asarray(res.switch_states).copy()
    k = int(np.argmax(np.any(np.diff(Shifted.switch_states, axis=0), axis=1)))
    Shifted.switch_states[k] = Shifted.switch_states[k + 1]
    bad = goldens.compare(golden, goldens.snapshot(Shifted(), names, newton))
    assert len(bad) == 1 and bad[0].startswith("switch_digest")


# ------------------------------------------------------------ counters ---

FROZEN = yaml.safe_load((BENCH / "manifests" / "perf_counters.yaml").read_text(encoding="utf-8"))["counters"]


@pytest.mark.parametrize("name", list(counters.CIRCUITS))
def test_counter_gate(name):
    for engine, frozen in FROZEN[name].items():
        if "refused" in frozen:
            with pytest.raises(Exception):
                counters.count(name, engine)
            continue
        bad, _ = counters.check(frozen, name, engine, counters.count(name, engine))
        assert not bad, bad


def test_counter_gate_names_an_extra_factorisation_per_step():
    got = counters.count("buck_ccm", "pwl")
    frozen = FROZEN["buck_ccm"]["pwl"]
    got["event_refactors"] += got["n_steps"]       # one more per step
    bad, _ = counters.check(frozen, "buck_ccm", "pwl", got)
    assert bad == [f"buck_ccm / pwl / event_refactors: {got['event_refactors']} > frozen "
                   f"{frozen['event_refactors']}"]
    got = counters.count("buck_ccm", "trbdf2")
    got["n_solves"] = int(FROZEN["buck_ccm"]["trbdf2"]["n_solves"] * 1.03) + 1
    assert counters.check(FROZEN["buck_ccm"]["trbdf2"], "buck_ccm", "trbdf2", got)[0]


# ------------------------------------------------- differential runner ---

def test_runner_classifies_a_junction_capacitor_rc_on_dsed_as_not_agree():
    def build():
        b = pulsim.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_resistor("R", "in", "out", 1e3)
        b.add_capacitor("C", "out", "gnd", 100e-9)
        b.add_nonlinear_capacitor("Cj", "out", "gnd", 1e-6, 1.0, 0.5)
        return b
    ref = pulsim.simulate(build(), 1e-4, 1e-7, engine="pwl")
    assert ref.v("out")[-1] == pytest.approx(1.02, abs=0.01)
    try:
        r = pulsim.simulate(build(), 1e-4, engine="dsed")
    except dr.REFUSAL:
        return                                                      # refused: fine
    e = metrics.err_rms(np.asarray(r.times), np.asarray(r.v("out")),
                        (np.asarray(ref.times), np.asarray(ref.v("out"))), exclude_t0=True)
    assert r.v("out")[-1] == pytest.approx(6.32, abs=0.01)          # silently drops Cj today
    assert e > dr.default_tol("dsed", False)                        # disagree, never agree


def test_runner_end_to_end_on_one_family(tmp_path):
    out = tmp_path / "results.json"
    rc = dr.main(["--circuits", "linear/rc_step.yaml", "--out", str(out)])
    recs = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert [(r["engine"], r["outcome"]) for r in recs] == [
        ("pwl", "agree"), ("trbdf2", "agree"), ("dsed", "agree")]
    assert all(r["reference"] == "trbdf2 rtol=1e-8" for r in recs)


def test_benchmark_runner_imports_without_the_retired_backend():
    src = (BENCH / "tools" / "benchmark_runner.py").read_text(encoding="utf-8")
    assert "import pulsim_python_backend" not in src and "from pulsim_python_backend" not in src
    import benchmark_runner  # noqa: F401


def test_benchmark_runner_runs_v1_files_against_their_stored_expectations(tmp_path):
    """The runner end to end on 2.0 (benchmarks.yaml -> v1 reader -> pulsim.simulate -> the
    file's own validation): an analytic RC step, a baseline-CSV bridge, a transformer. A scenario
    whose v1 solver selection or component has no 2.0 equivalent is refused by name."""
    import benchmark_runner as br

    ids = ["rc_step", "bridge_resistive_pure", "transformer_step_up", "saturating_inductor_step"]
    got = {(r.benchmark_id, r.scenario): r
           for r in br.run_benchmarks(BENCH / "manifests" / "benchmarks.yaml", tmp_path, selected=ids)}
    for bid in ids[:3]:
        r = got[(bid, "direct_trap")]
        assert r.status == "passed" and r.max_error is not None, (bid, r.message)
    assert got[("rc_step", "direct_trap")].max_error < 5e-3                 # measured 2.5e-3
    assert got[("bridge_resistive_pure", "direct_trap")].max_error < 1e-6  # measured 3.3e-10
    for key in [("rc_step", "gmres_trbdf2"), ("rc_step", "jfnk_trbdf2")]:
        assert got[key].status == "skipped"
        assert got[key].message.startswith("refused: v1 integrator 'trbdf2' (fixed step)")
    sat = [r for (b, _), r in got.items() if b == "saturating_inductor_step"]
    assert sat and all(r.status == "skipped" and "saturable_inductor" in r.message for r in sat)


def test_a_suspect_reference_grades_nothing_and_a_frozen_error_may_not_grow():
    # checkpoint review M1 / M7 (integration batch 6b)
    sus = {"status": "runs", "reference": "suspect", "reference_reason": "broken"}
    for out in ("agree", "disagree", "error"):
        r = dr.mark_suspect({"outcome": out}, sus)
        assert r["outcome"] == "unverified" and r["was"] == out
        assert dr.frozen_check(r, sus, "trbdf2") == ""
    assert dr.mark_suspect({"outcome": "refused"}, sus)["outcome"] == "refused"
    # cuk's diode is turned round (2026-11): its reference is graded again
    assert "reference" not in TRIAGE["circuits/switching/buck_boost/cuk_converter.yaml"]
    # sepic's reference grinds no more (TR-BDF2 roundoff floor): graded again
    assert "reference" not in TRIAGE["circuits/switching/buck_boost/sepic.yaml"]
    e = {"known": {"pwl+split": "disagree"}, "known_err": {"pwl+split": 0.01}}
    assert dr.frozen_check({"outcome": "disagree", "err_rms": 0.019}, e, "pwl+split") == ""
    assert "grew" in dr.frozen_check({"outcome": "disagree", "err_rms": 0.021}, e, "pwl+split")
    assert dr.frozen_check({"outcome": "disagree", "err_rms": 0.01}, e, "pwl") == "new outcome"
    r = {"outcome": "disagree", "err_rms": 0.004}
    assert dr.frozen_check(r, e, "pwl+split") == "" and "re-freeze" in r["note"]
    # every frozen disagreement carries its error
    for rel, ent in TRIAGE.items():
        for cfg, out in (ent.get("known") or {}).items():
            if out == "disagree":
                assert (ent.get("known_err") or {}).get(cfg), (rel, cfg)


def test_the_edge_diagnostic_is_symmetric_and_the_gate_is_not():
    # Two converged traces of a held 24 V output (6 mV ripple), each with its
    # stored initial state 0 V at t = 0 and a different first step: the gate keeps
    # the reference's first sample and divides by the reference's span, so it
    # depends on the side; the edge diagnostic does not, to the bit.
    def trace(t1, dt):
        t = np.concatenate(([0.0], np.arange(t1, 1e-3, dt)))
        y = 24.0 + 3e-3 * np.sin(2 * np.pi * 5e3 * t)
        y[0] = 0.0
        return (t, {"V(out)": y}, None)
    a, b = trace(1.25e-8, 1.25e-8), trace(5e-9, 1e-7)
    cols = ["V(out)"]
    ab, ba = dr.compare(a, b, cols, 1e-3)[0], dr.compare(b, a, cols, 1e-3)[0]
    assert max(ab, ba) > 3 * min(ab, ba)                       # the gate is one-sided
    da, db = dr.diagnose(a, b, cols, 1e-3), dr.diagnose(b, a, cols, 1e-3)
    assert da["edge_rms"] == db["edge_rms"] and da["excluded_frac"] == db["excluded_frac"]
    assert da["edge_rms"] < 1e-2                               # both traces are the same signal
    ya, yb = (a[0], a[1]["V(out)"]), (b[0], b[1]["V(out)"])
    assert metrics.err_rms_symmetric(ya, yb) == metrics.err_rms_symmetric(yb, ya)


def test_flip_evidence_resumes_per_circuit(tmp_path, monkeypatch):
    # a second run with the same cache measures nothing; a missing cell is measured again
    import ideal_switch_flip_evidence as ev
    pat = ["circuits/linear/rc_step.yaml"]
    recs, refs, tree = ev.run(pat, ("default",), tmp_path)
    assert recs and {r["variant"] for r in recs} == {"default"}
    assert refs["circuits/linear/rc_step.yaml"]["default"]["name"]
    monkeypatch.setattr(ev, "run_circuit", lambda *a, **k: pytest.fail("measured again"))
    again, _, _ = ev.run(pat, ("default",), tmp_path)
    assert again == recs
    monkeypatch.undo()
    more, _, _ = ev.run(pat, ("default", "auto"), tmp_path)
    assert {r["variant"] for r in more} == {"default", "auto"}
    assert [r for r in more if r["variant"] == "default"] == recs


def test_a_pulse_driven_rl_step_is_graded_against_its_ramp_closed_form():
    """rl_step / inductor_chain_step have no switch: their ideal twin is the circuit itself,
    and the converged ladder's reference (pwl+restart dt/32) drew the source's 1 ns edge as a
    ramp over one of its own steps: the exact engine (1e-11 of the closed form) was graded a
    disagreement (3.8e-3, 6.2e-3). Their reference is now the closed form of the ramp,
    declared exact in the circuit's validation block (exact engine session 22)."""
    import ideal_switch_flip_evidence as ev

    for rel, v0, tau in (("circuits/linear/rl_step.yaml", 10.0, 10e-3 / 100),
                         ("circuits/magnetics/inductor_chain_step.yaml", 5.0 * 20 / 30, 30e-3 / 100)):
        case = v1_reader.load(BENCH / rel)
        cols = dr.observables_of(case)
        t, ys, _ = ev.analytic_reference(case, cols)
        y = ys[cols[0]]
        # the ramp's closed form at t_r (v0 to O(t_r/tau)), the step's tail after it
        k = int(np.searchsorted(t, 1e-9))
        assert t[k] == 1e-9 and y[k] == pytest.approx(-v0 * (tau / 1e-9) * math.expm1(-1e-9 / tau), rel=1e-12)
        assert y[k] == pytest.approx(v0, rel=1e-5)
        assert y[-1] == pytest.approx(v0 * (tau / 1e-9) * math.expm1(1e-9 / tau) * math.exp(-t[-1] / tau),
                                      rel=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = case.run("dsed")
        rec = dr.classify({}, res, case, cols, (t, ys, None), 1e-3)
        # 2e-8: the gate's own measure, the engine's samples (1e-7 s apart, tau = 1e-4 s)
        # interpolated linearly onto the union grid; at its samples dsed is 2e-12 V off
        assert rec["outcome"] == "agree" and rec["err_rms"] < 1e-7


def test_a_right_limit_at_t0_belongs_to_the_initial_instant():
    """A restart-rule run keeps the right limit of its start, one ulp after
    t = 0 (with_right_limits). exclude_t0 drops it with the stored t = 0 row,
    so the gate reads the cell from its first step as before: the 0+ row
    against the reference's kept t = 0 row (0 V) read an exact DC trace
    2.24e-3 off on the restart / split columns."""
    t_ref = np.array([0.0, 5e-10, 1e-7, 2e-7])
    ref = (t_ref, {"y": np.array([0.0, 1.0, 1.0, 1.0])}, None)
    t = np.array([0.0, np.nextafter(0.0, 1.0), 1e-7, 2e-7])
    eng = (t, {"y": np.array([0.0, 1.0, 1.0, 1.0])}, None)
    assert dr.compare(eng, ref, ["y"], 1e-3)[0] == 0.0
