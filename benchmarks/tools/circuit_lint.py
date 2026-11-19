#!/usr/bin/env python3
"""Structural lint of v1 benchmark circuits: authoring errors no engine fixes.

Two rules, on the netlist graph (element = edge, ground = node 0):

* ``source_free`` -- an inductor whose every loop is source-free: no
  independent source, switch or transformer winding shares a biconnected
  block with it, so from a zero start it never carries current (an
  uncoupled "secondary winding").
* ``inductor_cutset`` -- with one controlled switch open and the others
  closed (diodes taken as conducting, current sources open), a set of
  nodes whose remaining connections are inductors only: their currents
  are forced equal (one inductor: forced to zero) at every turn-off of
  that switch, which interrupts a current into its off-resistance.

    circuit_lint.py [GLOB ...]       # default: every circuit; exit 1 on a finding
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "circuits"
INDUCTOR = {"inductor", "saturable_inductor"}
SOURCE = {"voltage_source", "current_source"}
WINDING = {"transformer", "coupled_inductor"}
SWITCH = {"vcswitch", "switch", "mosfet", "pmos", "igbt", "thyristor"}
OPEN_ALWAYS = {"current_source"}
CONTROL = {"pi_controller", "pwm_generator", "pid_controller", "comparator",
           "integrator", "limiter", "gain", "sample_hold", "lookup_table",
           "rate_limiter", "hysteresis", "clarke_transform", "park_transform",
           "inverse_park_transform", "pll", "svm"}


def _gnd(n) -> str:
    return "0" if str(n) in ("0", "gnd", "GND") else str(n)


def edges_of(comps):
    """(name, type, a, b, ic) per two-terminal branch; a winding pair is two."""
    out = []
    for c in comps:
        t, nodes = c["type"], [_gnd(n) for n in c.get("nodes", [])]
        if t in CONTROL or len(nodes) < 2:
            continue
        if t in WINDING and len(nodes) >= 4:
            out += [(c["name"] + ".p", t, nodes[0], nodes[1], 0.0),
                    (c["name"] + ".s", t, nodes[2], nodes[3], 0.0)]
        elif t in SWITCH and len(nodes) >= 3:
            out.append((c["name"], t, nodes[1], nodes[2], 0.0))   # drain-source
        else:
            out.append((c["name"], t, nodes[0], nodes[1],
                        float(c.get("ic", 0.0) or 0.0)))
    return out


def _blocks(edges):
    """Biconnected blocks as sets of edge indices (edges subdivided, so
    parallel branches form a cycle)."""
    adj: dict = {}
    for i, (_n, _t, a, b, _ic) in enumerate(edges):
        m = ("e", i)
        for u in (a, b):
            adj.setdefault(u, []).append((m, i))
            adj.setdefault(m, []).append((u, i))
    disc, low, stack, blocks, t = {}, {}, [], [], [0]

    def dfs(u, parent):
        disc[u] = low[u] = t[0]
        t[0] += 1
        for v, i in adj[u]:
            if v == parent:
                continue
            if v not in disc:
                stack.append((u, v, i))
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    blk = set()
                    while True:
                        x = stack.pop()
                        blk.add(x[2])
                        if x[:2] == (u, v):
                            break
                    blocks.append(blk)
            elif disc[v] < disc[u]:
                stack.append((u, v, i))
                low[u] = min(low[u], disc[v])
    sys.setrecursionlimit(10000)
    for u in list(adj):
        if u not in disc:
            dfs(u, None)
    return blocks


def source_free(edges):
    found = []
    for blk in _blocks(edges):
        kinds = {edges[i][1] for i in blk}
        if kinds & (SOURCE | WINDING | SWITCH) or any(edges[i][4] for i in blk):
            continue                 # driven, or an initial condition (natural response)
        found += [edges[i][0] for i in blk if edges[i][1] in INDUCTOR]
    return sorted(set(found))


def inductor_cutsets(edges):
    """Per controlled switch opened alone: groups of nodes joined by the
    conducting non-inductor branches whose other connections are
    inductors only (the smaller side of each cut)."""
    nodes = {u for e in edges for u in e[2:4]}
    out = []
    for sw in [e for e in edges if e[1] in SWITCH]:
        parent = {u: u for u in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in edges:
            if e is not sw and e[1] not in INDUCTOR | OPEN_ALWAYS:
                parent[find(e[2])] = find(e[3])
        groups: dict = {}
        for e in edges:
            ga, gb = find(e[2]), find(e[3])
            if ga != gb:
                for g in (ga, gb):
                    groups.setdefault(g, set()).add(e)
        for g, inc in groups.items():
            live = [e for e in inc if e is not sw and e[1] not in OPEN_ALWAYS]
            members = sorted(u for u in nodes if find(u) == g)
            if sw in inc and live and all(e[1] in INDUCTOR for e in live) \
                    and "0" not in members:
                out.append((sw[0], members, sorted(e[0] for e in live)))
    return out


def lint(path: Path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    edges = edges_of(doc.get("components", []))
    found = [("source_free", f"{n} never carries current (every loop through it "
              "is source-free; an uncoupled winding?)") for n in source_free(edges)]
    for sw, members, inds in inductor_cutsets(edges):
        found.append(("inductor_cutset", f"{sw} open: node(s) {', '.join(members)} "
                      f"meet only inductors {', '.join(inds)}"))
    return found


def main(argv):
    pats = argv or ["**/*.yaml"]
    paths = sorted({p for pat in pats for p in ROOT.glob(pat)})
    bad = 0
    for p in paths:
        try:
            found = lint(p)
        except Exception as exc:  # noqa: BLE001 - reported per file
            print(f"{p.relative_to(ROOT)}: unreadable ({exc})")
            continue
        for rule, msg in found:
            bad += 1
            print(f"{p.relative_to(ROOT)}: {rule}: {msg}")
    print(f"{len(paths)} circuits, {bad} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
