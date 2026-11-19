"""Pytest plugin: the number-changing release's no-op check and failure classifier.

    PYTHONPATH=python:benchmarks/tools python3 -m pytest python/tests -p next_defaults_pytest

Before the release this plugin injected the candidate "next defaults" profile
into every ``simulate()`` call (keywords per engine, the ``engine='auto'``
route, branch-form ``add_switch`` / ``add_diode``) to measure its blast radius.
Since the release (update-number-changing-release R1) that profile IS the
library default, so the plugin injects nothing: at configure time it asserts
``next_defaults.check_current()`` is empty (the flip is the profile; a
mismatch aborts the session), and a run with it equals a run without it. It
still classifies every failure and writes
``.scratch/rehearsal/pytest_profile.json`` (the input of the R2 bucket map).
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import next_defaults as nd

OUT = Path(__file__).resolve().parents[2] / ".scratch" / "rehearsal" / "pytest_profile.json"
_STATS = collections.Counter()
_FAILS: list = []


def _install():
    bad = nd.check_current()
    if bad:
        import pytest
        raise pytest.UsageError("next_defaults_pytest: the library defaults are not the "
                                "release profile:\n  " + "\n  ".join(bad))
    _STATS["profile == library defaults"] += 1


def classify(text: str) -> str:
    """One class per failure from its ``E`` lines (first match wins)."""
    t = text
    if re.search(r"DID NOT WARN|DID NOT RAISE|WarningMessage|\bwarn(ing)?s?\b", t, re.I):
        return "warning / raise expectation"
    if re.search(r"snapshot|pickle|SolverSnapshot|\bfields?\b", t):
        return "snapshot / layout"
    if re.search(r"^E\s+(\w+\.)*(UnsupportedError|NotImplementedError)|Refus", t, re.M) or re.search(
            r"^E\s+ValueError: .*(cannot|not supported|refus|requires|branch form|=True\): the)", t, re.M):
        return "new refusal by name"
    if re.search(r"'sha|digest", t):
        return "circuit digest (branch-form circuit)"
    if re.search(r"engine_used|engine_route_reason|'trbdf2'|'dsed'|'pwl'|_trbdf2_stats|"
                 r"^E\s+(AttributeError|KeyError)", t, re.M):
        return "engine route / result layout"
    if re.search(r"Not equal to tolerance|approx|Mismatched elements|Max absolute difference|golden|"
                 r"rel diff|array_equal|allclose|np\.float64\(|assert -?[0-9][0-9.e+-]* *[<>=!]=? *-?[0-9]", t):
        return "numeric assertion on an old value"
    if re.search(r"AssertionError|^E\s+assert ", t, re.M):
        return "other assertion"
    return "exception"


def summarize(fails, stats, applied):
    for f in fails:
        f["cls"] = classify(f["msg"])
    classes = collections.Counter(f["cls"] for f in fails)
    files = collections.defaultdict(collections.Counter)
    for f in fails:
        files[f["nodeid"].split("::")[0]][f["cls"]] += 1
    rows = sorted(((k, sum(c.values()), ", ".join(f"{a} {b}" for a, b in c.most_common()))
                   for k, c in files.items()), key=lambda r: (-r[1], r[0]))
    return dict(
        header=("Full `python/tests` with the profile injected by `benchmarks/tools/next_defaults_pytest.py`: "
                + ", ".join(f"{v} {k}" for k, v in stats.items())
                + f". Profile applications: {applied}."),
        stats=stats, applied=applied, classes=dict(classes.most_common()), files=rows, failures=fails)


def pytest_configure(config):
    _install()


def pytest_runtest_logreport(report):
    if report.failed:
        text = str(report.longrepr)
        last = [ln for ln in text.splitlines() if ln.startswith("E ")]
        msg = "\n".join(last[:4]) or (text.splitlines()[-1] if text else "")
        # the files the traceback passes through (a shared helper or conftest
        # that needs changing belongs to bucket 0 of the R2 map)
        entries = getattr(getattr(report.longrepr, "reprtraceback", None), "reprentries", None) or []
        paths = []
        for e in entries:
            loc = getattr(getattr(e, "reprfileloc", None), "path", None)
            if loc and loc not in paths:
                paths.append(str(loc))
        crash = getattr(report.longrepr, "reprcrash", None)
        _FAILS.append(dict(nodeid=report.nodeid, when=report.when, msg=msg[:600], tb=paths[:12],
                           crash=f"{crash.path}:{crash.lineno}" if crash else ""))


def pytest_sessionfinish(session, exitstatus):
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = {k: len(v) for k, v in (tr.stats.items() if tr else []) if k in
             ("passed", "failed", "error", "skipped", "xfailed", "xpassed")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summarize(_FAILS, stats, dict(_STATS)), indent=0), encoding="utf-8")


if __name__ == "__main__":     # re-classify a stored summary after a classify() change
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for f in d["failures"]:          # summaries written before E lines were kept one per line
        f["msg"] = f["msg"].replace(" | E", "\nE")
    OUT.write_text(json.dumps(summarize(d["failures"], d["stats"], d["applied"]), indent=0),
                   encoding="utf-8")
    print(json.dumps(summarize(d["failures"], d["stats"], d["applied"])["classes"], indent=1))
