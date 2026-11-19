"""The next-defaults profile helper and the rehearsal's cell cache
(benchmarks/tools/next_defaults.py, next_defaults_rehearsal.py). Tooling only:
no library default is read or changed here."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import next_defaults as nd  # noqa: E402
import next_defaults_rehearsal as ndr  # noqa: E402
import reference_status as rs  # noqa: E402

import pulsim  # noqa: E402


def test_apply_profile_fills_only_what_the_caller_left_unset():
    kw = nd.apply_profile("pwl", {"discontinuity_restart": False, "dt": 1e-6})
    assert kw["discontinuity_restart"] is False          # the caller's value wins
    assert kw["dt"] == 1e-6
    assert kw["enable_substep_state_correction"] is True
    assert kw["mode_selection"] == "auto"
    assert nd.apply_profile("exact", {}) == nd.apply_profile("dsed", {})
    assert "device_path" not in nd.profile_kwargs("dsed")   # the exact engine refuses the keyword
    assert nd.profile_kwargs("auto") == {}                  # auto routes first
    assert nd.profile_kwargs("pwl", only={"control_tick"}) == {"control_tick": "state"}


def test_every_profile_entry_cites_its_change():
    changes = Path(__file__).resolve().parents[2] / "openspec" / "changes"
    for _eng, _k, e in nd.table():
        assert (changes / e.change).is_dir(), e.change
        assert e.decision


def test_configuration_a_is_reference_status_unchanged():
    for name, (eng, kw) in rs.CONFIGS.items():
        c = ndr.CONFIGS[name]
        assert (c["engine"], c["kw"], c["form"], c["group"]) == (eng, kw, False, "a")


def test_one_single_per_option_and_the_full_profile():
    for eng in ndr.ENGINES:
        singles = {c["option"] for c in ndr.CONFIGS.values()
                   if c["group"] == "b" and c["engine"] == eng}
        assert singles == set(nd.PROFILE[eng]) | {"form"}
        full = ndr.CONFIGS[f"{eng}+next"]
        assert full["kw"] == nd.profile_kwargs(eng) and full["form"] is True


def test_builder_form_patches_and_restores():
    before = pulsim.CircuitBuilder.add_switch
    with nd.builder_form():
        assert pulsim.CircuitBuilder.add_switch is not before
    assert pulsim.CircuitBuilder.add_switch is before
    with nd.builder_form(False):
        assert pulsim.CircuitBuilder.add_switch is before


def test_cell_cache_round_trip_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(ndr, "CACHE", tmp_path)
    unit = "circuits/x/y.yaml"
    assert ndr.missing(unit, ["pwl", "pwl:restart", "dsed"]) == ["pwl", "dsed"]  # alias never runs
    d = ndr.load_unit(unit)
    d["cells"]["pwl"] = {"config": "pwl", "outcome": "agree", "err_rms": 1e-4}
    ndr.save_unit(d)
    assert ndr.missing(unit, ["pwl", "dsed"]) == ["dsed"]
    assert ndr.load_unit(unit)["cells"]["pwl"]["err_rms"] == 1e-4
    text = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert '"agree"' in text


@pytest.mark.parametrize("a,b,verdict", [
    ({"outcome": "agree"}, {"outcome": "refused"}, "worse"),
    ({"outcome": "disagree"}, {"outcome": "agree"}, "better"),
    ({"outcome": "agree", "exact": 1e-6}, {"outcome": "agree", "exact": 3e-6}, "worse"),
    ({"outcome": "agree", "exact": 1e-6}, {"outcome": "agree", "exact": 1.9e-6}, None),
    ({"outcome": "agree", "exact": 1e-6}, {"outcome": "agree", "exact": 4e-7}, "better"),
    ({"outcome": "unverified"}, {"outcome": "agree"}, None),
])
def test_compare(a, b, verdict):
    assert ndr.compare(a, b) == verdict
