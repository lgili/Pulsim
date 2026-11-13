"""Reference benchmark harness (openspec change restore-reference-benchmark-harness).

Metrics, the v1 reader and its triage manifest, the analytic reference set,
the golden checkpoints of the fixed engine and the counter gate. Every
assertion here is time-weighted or exact-integer; none reads a clock.
"""

from __future__ import annotations

import json
import math
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

ANALYTIC = yaml.safe_load((BENCH / "manifests" / "analytic.yaml").read_text())["cases"]
TRIAGE = yaml.safe_load((BENCH / "manifests" / "v2_triage.yaml").read_text())["circuits"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_pulsim_under_test_is_this_checkout():
    assert Path(pulsim.__file__).resolve().parents[2] == BENCH.parent


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


def test_v1_pwm_phase_is_a_radian_advance_and_dead_time_shortens_on_time():
    # v1 kernel: high while (t + phase/2pi*T) mod T < duty*T - dead_time
    f, duty, start = v1_reader._on_windows(
        {"type": "pwm", "frequency": 1e3, "duty": 0.5, "phase": math.pi / 2, "dead_time": 1e-4}, 0.5)
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
    on_disk = {str(p.relative_to(BENCH)) for p in (BENCH / "circuits").rglob("*.yaml")}
    assert len(on_disk) == 126
    assert set(TRIAGE) == on_disk
    assert {e["status"] for e in TRIAGE.values()} <= {"runs", "skipped", "error"}
    for rel, e in TRIAGE.items():
        if e["status"] != "runs":
            assert e.get("reason"), rel


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


def test_there_are_twenty_analytic_cases_and_each_has_a_manifest_entry():
    assert len(CASES) == 20
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
    golden = json.loads((goldens.GOLDEN_DIR / f"{name}.json").read_text())
    assert golden["rtol"] <= (goldens.RTOL_NEWTON if newton else goldens.RTOL_LINEAR)
    bad = goldens.compare(golden, goldens.snapshot(run(), names, newton))
    assert not bad, "\n".join(bad[:10])


def test_golden_catches_one_part_per_million():
    c = BY_ID["rc_charge"]
    golden = json.loads((goldens.GOLDEN_DIR / "analytic__rc_charge.json").read_text())
    b, _ = c.build(1.0 + 1e-6)
    res = pulsim.simulate(b, c.t_end, c.dt, engine="pwl")
    bad = goldens.compare(golden, goldens.snapshot(res, b.state_var_names(), False))
    assert bad and "rel diff" in bad[0] and "V(o)" in "".join(bad)


def test_golden_catches_a_changed_switching_sequence_before_values():
    name, run, names, newton = next(g for g in GOLDEN if g[0] == "buck_switching")
    golden = json.loads((goldens.GOLDEN_DIR / f"{name}.json").read_text())
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

FROZEN = yaml.safe_load((BENCH / "manifests" / "perf_counters.yaml").read_text())["counters"]


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
    recs = json.loads(out.read_text())
    assert rc == 0
    assert [(r["engine"], r["outcome"]) for r in recs] == [
        ("pwl", "agree"), ("trbdf2", "agree"), ("dsed", "agree")]
    assert all(r["reference"] == "trbdf2 rtol=1e-8" for r in recs)


def test_benchmark_runner_imports_without_the_retired_backend():
    src = (BENCH / "tools" / "benchmark_runner.py").read_text()
    assert "import pulsim_python_backend" not in src and "from pulsim_python_backend" not in src
    import benchmark_runner  # noqa: F401
