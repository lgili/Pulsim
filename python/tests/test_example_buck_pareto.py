"""The worked Pareto example (add-design-optimization-workflow, task 6.2):
32 Latin-hypercube designs, without pymoo, through `main` as a reader
runs it. Checked: the status counts are printed and add up, the front
is non-empty, every member satisfies the constraints, and the
synthetic-part warning is printed."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

EXAMPLE = (Path(__file__).resolve().parents[2] / "examples" / "optimization"
           / "buck_pareto.py")


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("buck_pareto", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["buck_pareto"] = mod      # the process backend pickles by name
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("buck_pareto", None)


def test_32_designs_without_pymoo(ex, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "pymoo", None)     # not installed
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    records, front = ex.main(["buck_pareto.py", "32"])
    out = capsys.readouterr().out
    counts = {k: int(v) for k, v in re.findall(
        r"(ok|refused|failed|infeasible): (\d+)", out)}
    assert set(counts) == {"ok", "refused", "failed", "infeasible"}
    assert sum(counts.values()) == len(records) == 32
    assert counts == ex.status_counts(records)
    assert "SYNTHETIC" in out
    assert front, "the non-dominated set is empty"
    for r in front:
        assert r.status == "ok"
        assert r.kpis["ripple_v"] <= ex.RIPPLE_MAX
        assert r.kpis["r_th_sa"] >= ex.R_TH_SA_MIN
        assert all(v == 0.0 for v in r.violations.values())
    # nothing on the front is dominated by another feasible design
    ok = [r for r in records if r.status == "ok"]
    for r in front:
        for q in ok:
            assert not (q.kpis["efficiency"] >= r.kpis["efficiency"]
                        and q.kpis["volume"] <= r.kpis["volume"]
                        and (q.kpis["efficiency"] > r.kpis["efficiency"]
                             or q.kpis["volume"] < r.kpis["volume"]))


def test_parts_are_flagged_synthetic(ex):
    assert all(part.is_synthetic for part in ex.PARTS.values())
    assert "SYNTHETIC" in EXAMPLE.read_text()


def test_nsga2_path_when_pymoo_is_installed(ex):
    pytest.importorskip("pymoo")
    records, front = ex.run_pymoo(n_gen=2, pop=8)
    assert len(records) >= 8 and front
    assert all(r.status == "ok" for r in front)
