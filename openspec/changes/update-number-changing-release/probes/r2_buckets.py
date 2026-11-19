#!/usr/bin/env python3
"""Generate R2_BUCKETS.md: the red test suite after the flip, split into six
disjoint buckets by test file (update-number-changing-release R1, task D).

    r2_buckets.py PYTEST_JSON CTEST_FAILED_TXT [--out R2_BUCKETS.md] [--resolved R1_RESOLVED.txt]

PYTEST_JSON: the summary ``benchmarks/tools/next_defaults_pytest.py`` writes
(every failure with its class and the files its traceback passes through).
CTEST_FAILED_TXT: one failing ctest name per line (from the run's log).
R1_RESOLVED.txt: ``<nodeid prefix> | <what fixed it in R1>`` per line; the
failures it matches are listed as resolved in R1 (bucket 0 / regenerated
goldens), counted in the totals, and given to no bucket.

Files that import each other (a test module importing another test module's
cases) go to the same bucket; a shared helper (conftest, a non-test module
under python/tests, a C++ test header) that a traceback crashes in is listed
as bucket 0. Buckets are balanced greedily by failure count, largest unit
first.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "python" / "tests"
CTESTS = ROOT / "core" / "tests"
N = 6
RULE = {
    "numeric assertion on an old value": "(a) physics: pass under the new defaults against its truth "
    "(tighten where earned, never loosen); a pinned old number of a legacy SUBJECT: (b) pin the legacy option",
    "new refusal by name": "(c) an accidental undriven switch / ideal loop: drive it or pass the policy meant; "
    "(d) a refusal that is the subject: expect the new refusal",
    "engine route / result layout": "(d) follow the new route (engine_used, route reasons, output grid); "
    "(b) pin engine= where the engine is the subject",
    "warning / raise expectation": "(d) expected warnings / refusals follow the new behaviour",
    "snapshot / layout": "(b)/(d) snapshot of a run whose defaults changed: re-record against the new defaults "
    "or pin the legacy options the snapshot was taken with",
    "circuit digest (IR / YAML path not patched: harness artefact)": "(d) digests of branch-form circuits: "
    "re-pin; check the YAML and builder paths build the same circuit",
    "other assertion": "read the assertion: classify as (a)-(d)",
    "exception": "read the traceback: an engine finding is reported, not patched",
}


def py_imports(path: Path) -> set:
    """Sibling modules under python/tests that ``path`` imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            parts = n.split(".")
            if parts[0] == "tests":
                parts = parts[1:]
            if parts and (TESTS / f"{parts[0]}.py").exists():
                out.add(f"python/tests/{parts[0]}.py")
            elif parts and (TESTS / parts[0]).is_dir():
                out.add(f"python/tests/{parts[0]}/")
    return out


def cpp_file_of(test_name: str, cache={}) -> str:
    """The C++ test source that declares TEST_CASE(test_name)."""
    if not cache:
        for f in CTESTS.rglob("*.cpp"):
            for m in re.finditer(r'TEST_CASE\(\s*"((?:[^"\\]|\\.)*)"', f.read_text(encoding="utf-8")):
                cache.setdefault(m.group(1).replace('\\"', '"'), str(f.relative_to(ROOT)))
    return cache.get(test_name, "core/tests/<unresolved>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pytest_json")
    ap.add_argument("ctest_failed")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "R2_BUCKETS.md"))
    ap.add_argument("--header", default="")
    ap.add_argument("--resolved")
    a = ap.parse_args(argv)
    d = json.loads(Path(a.pytest_json).read_text(encoding="utf-8"))
    resolved_rules = []
    if a.resolved:
        for ln in Path(a.resolved).read_text(encoding="utf-8").splitlines():
            if ln.strip() and not ln.startswith("#"):
                k, why = (x.strip() for x in ln.split("|", 1))
                resolved_rules.append((k, why))
    resolved = []
    fails = []
    for f in d["failures"]:
        why = next((w for k, w in resolved_rules if f["nodeid"].startswith(k)), None)
        (resolved.append((f, why)) if why else fails.append(f))
    n_all = len(d["failures"])
    by_file = collections.defaultdict(list)
    helpers = collections.Counter()
    for f in fails:
        rel = f["nodeid"].split("::")[0]
        by_file[rel].append(f)
        for p in f.get("tb", []):
            q = Path(p)
            try:
                r = str(q.resolve().relative_to(ROOT)) if q.is_absolute() else p
            except ValueError:
                continue
            if r.startswith("python/tests/") and r != rel and (
                    not Path(r).name.startswith("test_") or Path(r).name == "conftest.py"):
                helpers[r] += 1
    ctest = [ln.strip() for ln in Path(a.ctest_failed).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for t in ctest:
        by_file[cpp_file_of(t)].append(dict(nodeid=t, cls="C++ (read the assertion)", msg=""))
    # union of failing files that import each other
    parent = {k: k for k in by_file}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for rel in list(by_file):
        if rel.startswith("python/tests/"):
            for imp in py_imports(ROOT / rel):
                if imp in by_file and imp != rel:
                    parent[find(imp)] = find(rel)
    units = collections.defaultdict(list)
    for rel in by_file:
        units[find(rel)].append(rel)
    work = {u: sum(len(by_file[r]) for r in fs) for u, fs in units.items()}
    buckets = [dict(files=[], n=0) for _ in range(N)]
    for u in sorted(units, key=lambda u: (-work[u], u)):
        b = min(buckets, key=lambda b: (b["n"], len(b["files"])))
        b["files"] += sorted(units[u])
        b["n"] += work[u]
    n_py, n_cpp = len(fails), len(ctest)
    total = sum(b["n"] for b in buckets)
    assert total == n_py + n_cpp, (total, n_py, n_cpp)
    assert n_py + len(resolved) == n_all == d["stats"]["failed"], (n_py, len(resolved), n_all, d["stats"])
    L = ["# R2 buckets — the red suite after the flip", "",
         "Generated by `probes/r2_buckets.py` from ONE full pytest run and ONE full C++ test run on",
         "the flipped tree (not from the rehearsal's estimate). " + a.header, "",
         f"Run counts: pytest {d['stats']['failed']} failed / {d['stats']['passed']} passed / "
         f"{d['stats']['skipped']} skipped / {d['stats'].get('xfailed', 0)} xfailed; C++ {n_cpp} failed.", "",
         f"Totals: {n_all} pytest failures = **{len(resolved)} resolved in R1** (below) + "
         f"**{n_py} in buckets 1-6**; plus **{n_cpp} C++ test failures** in buckets 1-6. "
         f"Buckets 1-6 hold {total} = {n_py} + {n_cpp}.", "",
         "## Rules for fixing (the orchestrator's decision)", "",
         "- (a) A test about PHYSICS (a converter's output, a loss, a waveform) passes under the NEW",
         "  defaults against its truth: tighten its tolerance where the new engine earns it, never loosen;",
         "  if it cannot pass, that is an engine finding to report, not a number to paste.",
         "- (b) A test whose SUBJECT is a legacy behaviour (the conductance form's leakage, the restart-less",
         "  fixed step, an undriven switch being closed) pins the explicit legacy option and says so in one",
         "  comment (`next_defaults.LEGACY` lists the values).",
         "- (c) A test that relied on an undriven switch by accident drives the switch or passes the policy",
         "  it means (`undriven='closed'` / `'open'`).",
         "- (d) Expected warnings / refusals follow the new behaviour.",
         "- (e) No test is deleted or skipped without a line in the bucket's notes saying what it covered",
         "  and where that is covered now.",
         "- A bucket edits only its own files. A shared helper / conftest / C++ test header that needs a",
         "  change goes to the orchestrator (bucket 0), not into a bucket.", "",
         "## Failure classes (whole run)", "", "| class | pytest | rule |", "|---|---:|---|"]
    cls = collections.Counter(f["cls"] for f in d["failures"])
    for k, v in cls.most_common():
        L.append(f"| {k} | {v} | {RULE.get(k, '')} |")
    L += ["", f"C++: {n_cpp} failing test cases (classified per bucket below by reading).", "",
          "## Bucket 0 — shared helpers (R1 fixes these)", ""]
    if resolved:
        L += [f"Resolved in R1 ({len(resolved)} failures of the run; in no bucket):", "",
              "| test | class | fixed by |", "|---|---|---|"]
        L += [f"| `{f['nodeid']}` | {f['cls']} | {w} |" for f, w in resolved]
        L.append("")
    if helpers:
        L += ["Helpers a failing traceback passes through (count of failures):", ""]
        L += [f"- `{h}` ({n})" for h, n in helpers.most_common()]
    else:
        L.append("No failing traceback passes through a shared helper module or conftest.")
    for i, b in enumerate(buckets, 1):
        bc = collections.Counter(f["cls"] for r in b["files"] for f in by_file[r])
        dom = bc.most_common(1)[0][0] if bc else "-"
        L += ["", f"## Bucket {i} — {b['n']} failures, {len(b['files'])} files; dominant class: {dom}", "",
              "Rule: " + RULE.get(dom, "read each failure; rules (a)-(e)"), "",
              "| file | failures | classes |", "|---|---:|---|"]
        for r in b["files"]:
            c = collections.Counter(f["cls"] for f in by_file[r])
            L.append(f"| `{r}` | {len(by_file[r])} | " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + " |")
        L += ["", "<details><summary>failing tests</summary>", ""]
        for r in b["files"]:
            for f in by_file[r]:
                m = (f.get("msg") or "").splitlines()
                L.append(f"- `{f['nodeid']}` — {f['cls']}" + (f": {m[0][:160]}" if m else ""))
        L += ["", "</details>"]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {n_py} pytest + {n_cpp} C++ = {total}; buckets "
          + ", ".join(str(b["n"]) for b in buckets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
