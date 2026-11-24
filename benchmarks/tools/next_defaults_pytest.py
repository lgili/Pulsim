"""Pytest plugin: the blast radius of the next-defaults profile on the suite.

    PYTHONPATH=python:benchmarks/tools python3 -m pytest python/tests -p next_defaults_pytest

No library or test edit. For the whole session it
(1) wraps ``pulsim.simulate`` so that every call gets the profile's keywords
    for the engine it lands on, only where the call did not set them (the
    restart rule and the step split instead set the library's own defaults,
    ``pulsim._plan.DEFAULT_*``, which ``simulate()`` resolves per run);
    ``engine='auto'`` without ``dt`` takes the release route, approximated
    as: trbdf2 when the exact engine cannot take the circuit structurally (a
    Newton device, a ``b_extra_fn``), else the exact engine, and trbdf2 when
    that run refuses by name (D11's per-segment rule approximated by the whole
    run; a refusal raised mid-run repeats the run's side effects on trbdf2);
    with ``dt`` (or a YAML that sets one) it is pwl. A ``solver=`` bundle's
    fields count as set;
(2) builds ``add_switch`` / ``add_diode`` in the branch form when the call
    names no ``form`` (a given ``g_on`` means ``R_on = 1/g_on``; an unset value
    takes the branch form's own default); YAML / IR-lowered circuits are not
    reached (their digests then differ from a patched builder's);
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


# Options whose default the library resolves itself (pulsim._plan): the
# profile sets the library default instead of the keyword, so the split
# follows the rule and mmc_arms turns the rule's default off, as at the release.
LIBRARY_DEFAULTS = {"discontinuity_restart": "DEFAULT_DISCONTINUITY_RESTART",
                    "enable_substep_state_correction": "DEFAULT_SUBSTEP_STATE_CORRECTION"}


def _inject(eng, kw):
    """The profile for ``eng`` on the keywords the call left unset, a ``solver=``
    bundle included (simulate() reads a bundle field only when the keyword is None)."""
    solver = kw.get("solver")
    prof = {k: v for k, v in nd.profile_kwargs(eng).items() if k not in LIBRARY_DEFAULTS}
    set_by_bundle = {k for k in prof
                     if solver is not None and getattr(solver, k, None) is not None}
    return {**{k: v for k, v in prof.items() if k not in set_by_bundle}, **kw}


def _exact_structurally(pulsim, b, kw):
    """False when the exact engine cannot take the circuit at all (a Newton
    device, a user source function): decided before any run, no side effect."""
    if kw.get("b_extra_fn") is not None:
        return False
    try:
        if isinstance(b, pulsim.Circuit):
            b = b.to_builder()
        pulsim._pulsim._require_supported(b.graph, b.pool, "exact")
    except pulsim.UnsupportedError:
        return False
    except Exception:  # noqa: BLE001 - a stand-in object: let the run decide
        return True
    return True


def _route_kwargs(pulsim, args, kw):
    eng = nd.engine_key(kw.get("engine", "auto"))
    if eng != "auto":
        return [(eng, _inject(eng, kw))]
    dt = args[2] if len(args) > 2 else kw.get("dt")
    b = args[0] if args else kw.get("builder")
    if dt is None and isinstance(b, pulsim.LoadedCircuit):
        try:
            dt = pulsim._resolve_loaded_circuit(b, kw.get("t_end", args[1] if len(args) > 1 else None),
                                                None, 0.0, None)[2]
        except Exception:  # noqa: BLE001 - leave the call to raise its own error
            dt = None
        b = getattr(b, "builder", b)
    if dt:
        return [("pwl", _inject("pwl", {**kw, "engine": "pwl"}))]
    trb = ("trbdf2", _inject("trbdf2", {**kw, "engine": "trbdf2"}))
    if not _exact_structurally(pulsim, b, kw):
        return [trb]
    return [("dsed", _inject("dsed", {**kw, "engine": "dsed"})), trb]


_SAVED: dict = {}


def _install():
    import pulsim
    from pulsim import _plan
    if _SAVED:
        return
    saved_plan = {}
    for k, attr in LIBRARY_DEFAULTS.items():
        saved_plan[attr] = getattr(_plan, attr)
        setattr(_plan, attr, nd.PROFILE["pwl"][k].value)
    for attr, entry in nd.POLICY.items():
        saved_plan[attr] = getattr(_plan, attr)
        setattr(_plan, attr, entry.value)
    orig = pulsim.simulate

    @functools.wraps(orig)
    def simulate(*args, **kw):
        routes = _route_kwargs(pulsim, args, kw)
        for i, (eng, k) in enumerate(routes):
            try:
                out = orig(*args, **k)
                _STATS[f"applied:{eng}"] += 1
                return out
            except pulsim.UnsupportedError as exc:
                if i == len(routes) - 1 or getattr(exc, "feature", "") == "undriven_switch":
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
        return di(self, name, a, c, None, g_off, V_th, V_f, form="branch", **k)
    cb.add_switch, cb.add_diode = add_switch, add_diode
    _SAVED.update(plan=saved_plan, simulate=orig, add_switch=sw, add_diode=di)


def _uninstall():
    if not _SAVED:
        return
    import pulsim
    from pulsim import _plan
    for attr, value in _SAVED["plan"].items():
        setattr(_plan, attr, value)
    pulsim.simulate = _SAVED["simulate"]
    cb = pulsim.CircuitBuilder
    cb.add_switch = _SAVED["add_switch"]
    cb.add_diode = _SAVED["add_diode"]
    _SAVED.clear()


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
        return "circuit digest (IR / YAML path not patched: harness artefact)"
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


def pytest_unconfigure(config):
    _uninstall()


def pytest_runtest_logreport(report):
    if report.failed:
        text = str(report.longrepr)
        last = [ln for ln in text.splitlines() if ln.startswith("E ")]
        msg = "\n".join(last[:4]) or (text.splitlines()[-1] if text else "")
        _FAILS.append(dict(nodeid=report.nodeid, when=report.when, msg=msg[:600]))


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
