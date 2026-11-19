"""Pytest plugin: the blast radius of the next-defaults profile on the suite.

    PYTHONPATH=python:benchmarks/tools python3 -m pytest python/tests -p next_defaults_pytest

No library or test edit. For the whole session it
(1) wraps ``pulsim.simulate`` so that every call gets the profile's keywords
    for the engine it lands on, only where the call did not set them;
    ``engine='auto'`` without ``dt`` takes the release route, approximated
    as: the exact engine with its profile, and trbdf2 with its profile when
    the exact engine refuses by name (D11's per-segment rule is approximated
    by the whole run); with ``dt`` (or a YAML that sets one) it is pwl;
(2) builds ``add_switch`` / ``add_diode`` in the branch form when the call
    names no ``form`` (a given ``g_on`` means ``R_on = 1/g_on``);
then classifies every failure and writes ``.scratch/rehearsal/pytest_profile.json``
(read by ``next_defaults_rehearsal.py render``). Calls made inside pulsim go
through the wrapper too (the release default would reach them as well).
"""

from __future__ import annotations

import collections
import functools
import json
import re
from pathlib import Path

import next_defaults as nd

OUT = Path(__file__).resolve().parents[2] / ".scratch" / "rehearsal" / "pytest_profile.json"
_STATS = collections.Counter()
_FAILS: list = []


def _route_kwargs(pulsim, args, kw):
    eng = nd.engine_key(kw.get("engine", "auto"))
    if eng != "auto":
        return [(eng, nd.apply_profile(eng, kw))]
    dt = args[2] if len(args) > 2 else kw.get("dt")
    b = args[0] if args else kw.get("builder")
    if dt is None and isinstance(b, pulsim.LoadedCircuit):
        try:
            dt = pulsim._resolve_loaded_circuit(b, kw.get("t_end", args[1] if len(args) > 1 else None),
                                                None, 0.0, None)[2]
        except Exception:  # noqa: BLE001 - leave the call to raise its own error
            dt = None
    if dt:
        return [("pwl", nd.apply_profile("pwl", {**kw, "engine": "pwl"}))]
    return [("dsed", nd.apply_profile("dsed", {**kw, "engine": "dsed"})),
            ("trbdf2", nd.apply_profile("trbdf2", {**kw, "engine": "trbdf2"}))]


def _install():
    import pulsim
    orig = pulsim.simulate

    @functools.wraps(orig)
    def simulate(*args, **kw):
        routes = _route_kwargs(pulsim, args, kw)
        for i, (eng, k) in enumerate(routes):
            try:
                out = orig(*args, **k)
                _STATS[f"applied:{eng}"] += 1
                return out
            except pulsim.UnsupportedError:
                if i == len(routes) - 1:
                    raise
                _STATS["auto: exact refused -> trbdf2"] += 1
    pulsim.simulate = simulate

    cb = pulsim.CircuitBuilder
    sw, di = cb.add_switch, cb.add_diode

    def add_switch(self, name, a, b, g_on=None, g_off=None, **k):
        if "form" in k:
            return sw(self, name, a, b, g_on, g_off, **k)
        _STATS["branch:add_switch"] += 1
        return sw(self, name, a, b, g_on, g_off, form="branch", **k)

    def add_diode(self, name, a, c, g_on=None, g_off=None, V_th=0.0, V_f=0.0, **k):
        if "form" in k:
            return di(self, name, a, c, g_on, g_off, V_th, V_f, **k)
        _STATS["branch:add_diode"] += 1
        if g_on is not None and k.get("R_on") is None:
            k["R_on"] = 1.0 / g_on
        return di(self, name, a, c, None, 1e-9 if g_off is None else g_off, V_th, V_f,
                  form="branch", **k)
    cb.add_switch, cb.add_diode = add_switch, add_diode


def classify(text: str) -> str:
    """One class per failure, from its report text (first match wins)."""
    t = text
    if re.search(r"snapshot|pickle|SolverSnapshot|field count|n_fields", t):
        return "snapshot / layout"
    if re.search(r"DID NOT WARN|DID NOT RAISE|warns\(|len\(w\)|warning", t, re.I) and "AssertionError" in t \
            or "DID NOT WARN" in t or "Failed: DID NOT RAISE" in t:
        return "warning / raise expectation"
    if "UnsupportedError" in t or "Refusal" in t:
        return "new refusal by name"
    if re.search(r"engine_used|engine_route_reason|route", t) and "AssertionError" in t:
        return "engine route assertion"
    if re.search(r"Not equal to tolerance|approx|assert_allclose|isclose|Mismatched elements|"
                 r"assert -?[0-9.e+-]+ [<>=!]=? -?[0-9.e+-]+", t):
        return "numeric assertion on an old value"
    if "AssertionError" in t or "assert " in t:
        return "other assertion"
    return "exception"


def pytest_configure(config):
    _install()


def pytest_runtest_logreport(report):
    if report.failed:
        text = str(report.longrepr)
        last = [ln for ln in text.splitlines() if ln.startswith("E ")]
        _FAILS.append(dict(nodeid=report.nodeid, when=report.when, cls=classify(text),
                           msg=(" | ".join(last[:3]) or text.splitlines()[-1:][0] if text else "")[:300]))


def pytest_sessionfinish(session, exitstatus):
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = {k: len(v) for k, v in (tr.stats.items() if tr else []) if k in
             ("passed", "failed", "error", "skipped", "xfailed", "xpassed")}
    classes = collections.Counter(f["cls"] for f in _FAILS)
    files = collections.defaultdict(collections.Counter)
    for f in _FAILS:
        files[f["nodeid"].split("::")[0]][f["cls"]] += 1
    rows = sorted(((k, sum(c.values()), ", ".join(f"{a} {b}" for a, b in c.most_common()))
                   for k, c in files.items()), key=lambda r: -r[1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        header=("Full `python/tests` with the profile injected by `benchmarks/tools/next_defaults_pytest.py`: "
                + ", ".join(f"{v} {k}" for k, v in stats.items())
                + f". Profile applications: {dict(_STATS)}."),
        stats=stats, applied=dict(_STATS), classes=dict(classes.most_common()), files=rows,
        failures=_FAILS), indent=0), encoding="utf-8")
