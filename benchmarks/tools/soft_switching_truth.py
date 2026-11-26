"""Exact soft-switching KPIs of a benchmark circuit, from ``exact_truth.py`` (mpmath; never
imports pulsim): the truth beside ``kpi__zvs_fraction__*`` and ``kpi__switching_energy_j__*``.

For a gated switch between ``a`` and ``k`` (conductance form: i_D = g v_DS, g = g_on / g_off):

* every turn-on in (0, t_end): the LEFT limit of v_DS = v(a) - v(k) at the gate edge;
* the energy the KPI defines (docs/SOFT_SWITCHING.md): the sum over every gate edge of
  int g v_DS^2 dt over ``window_s`` after it (clipped at t_end), g that of the new gate state
  (assumed: the window ends before the switch's own next gate edge, so g is constant on it;
  true for the three references: 1 us / 10 us, 100 ns / 3.7 us, 50 ns / 4.92 us). One terminal
  must be ground or a DC bus (checked at the window's two ends only), so the integral is exact from the tool's window mean and RMS of the other
  node: int (Vb - v)^2 = (Vb^2 - 2 Vb mean + rms^2) T;
* with ``grid`` (the sample times of a trace): the same windows integrated by the trapezoid
  rule over the truth sampled after the edge's sub-picosecond C_oss collapse (JUMP_S) and at the
  grid points inside the window, the window end interpolated -- the quadrature the KPI applies to a trace, so
  ``|trapz - exact|`` is the part of a KPI's error that its sample grid alone causes.

``baseline`` writes a reference CSV for the benchmark runner: the truth on the netlist's dt grid
with both limits at every event (the right limit one ulp after the left), columns ``V_DS`` and
``I_D`` of the named switches (I_D = g v_DS of the mode on that side) plus the YAML's
observables.

    python benchmarks/tools/soft_switching_truth.py kpi <circuit.yaml> <switch> <window_s> [thr_v]
    python benchmarks/tools/soft_switching_truth.py baseline <circuit.yaml> <out.csv.gz> <switch>...
"""
from __future__ import annotations

import csv
import gzip
import math
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_truth as et  # noqa: E402


JUMP_S = mp.mpf("1e-11")


def _switch(sol, name):
    ck = sol.ck
    j = next(j for j, d in enumerate(ck.devs) if d["kind"] == "S" and d["name"] == name)
    return j, ck.devs[j]


def _col(dv):
    return f"V({dv['a']})" if dv["k"] == "gnd" else f"V({dv['a']},{dv['k']})"


def switch_kpi_truth(sol, name: str, window_s: float, threshold_v: float, grid=None) -> dict:
    _, dv = _switch(sol, name)
    t_end = sol.segs[-1].t0 + sol.segs[-1].tau
    a, k = dv["a"], dv["k"]
    col = _col(dv)

    def int_v2(t0, t1):
        if k == "gnd":                      # low side: v_DS = v(a)
            w = sol.window([f"V({a})"], t0, t1)[f"V({a})"]
            return w["rms"] ** 2 * (t1 - t0)
        vb = sol.sample([f"V({a})"], [t0])[f"V({a})"][0]       # a DC bus
        if abs(sol.sample([f"V({a})"], [t1])[f"V({a})"][0] - vb) > 1e-15 * abs(vb):
            raise et.Unsupported(f"{name}: terminal {a} is not a DC bus")
        w = sol.window([f"V({k})"], t0, t1)[f"V({k})"]
        return (vb ** 2 - 2 * vb * w["mean"] + w["rms"] ** 2) * (t1 - t0)

    def trapz_v2(t0, t1):
        # the integrand's first point is the truth extrapolated to te from te + 10 ps and
        # te + 20 ps: a turned-on switch collapses its node's C_oss (2 x 200 pF over 2 kS or
        # 1 kS: 0.2-0.4 ps, so 10 ps is >= 25 tau) -- a jump to every trace with event rows
        # (a grid row inside the collapse -- an event's right limit one ulp on -- is that limit)
        ts = [t0] + [mp.mpf(x) for x in grid if t0 + 2 * JUMP_S < x < t1] + [mp.mpf(x) for x in grid if x >= t1][:1]
        v = sol.sample([col], [t0 + JUMP_S, t0 + 2 * JUMP_S] + ts[1:])[col]
        v = [2 * v[0] - v[1]] + v[2:]       # back to te, linearly: the slow drift removed too
        s = mp.mpf(0)
        for i in range(len(ts) - 1):
            lo, hi = ts[i], ts[i + 1]
            p0, p1 = v[i] ** 2, v[i + 1] ** 2
            if hi > t1:                     # the window end, linearly interpolated
                p1, hi = p0 + (t1 - lo) / (hi - lo) * (p1 - p0), t1
            s += (p0 + p1) / 2 * (hi - lo)
        return s

    sched, t = dv["sched"], et.Fr(0)
    t_on, v_on, energy, trapz, n_edges = [], [], mp.mpf(0), mp.mpf(0), 0
    while True:
        t = sched.next_edge(t)
        if t is None or et.mpf(t) >= t_end:
            break
        te, on = et.mpf(t), sched.state(t)
        n_edges += 1
        if on:
            t_on.append(float(te))
            v_on.append(sol.sample([col], [te, te])[col][0])       # the left limit
        g = et.mpf(dv["g_on"] if on else dv["g_off"])
        t1 = min(te + mp.mpf(window_s), t_end)
        energy += g * int_v2(te, t1)
        if grid is not None:
            trapz += g * trapz_v2(te, t1)
    soft = sum(1 for v in v_on if abs(v) < threshold_v)
    out = dict(zvs_fraction=soft / len(v_on) if v_on else 0.0, turn_on_events=len(v_on),
               t_on=t_on, v_ds_at_turn_on=[float(v) for v in v_on], edges=n_edges,
               switching_energy_j=float(energy), t_end=float(t_end))
    if grid is not None:
        out["switching_energy_trapz_j"] = float(trapz)
    return out


def write_baseline(sol, switches, out_path) -> int:
    """The truth on the netlist's dt grid, both limits at every event: time, the YAML's
    observables, V_DS and I_D of each switch. Returns the row count."""
    ck = sol.ck
    n = int(et.Fr(ck.t_end) / et.Fr(ck.dt))
    ts = sorted([et.mpf(i * et.Fr(ck.dt)) for i in range(n + 1)]
                + [et.mpf(0)] + [e["t"] for e in sol.events[1:] for _ in (0, 1)])   # t = 0 too
    sw = [_switch(sol, s) for s in switches]
    vcols = [_col(dv) for _, dv in sw]
    obs = [c for c in ck.observables if c not in vcols]
    val = sol.sample(obs + vcols, ts)
    head = ["time"] + obs + vcols + [f"I({dv['name']})" for _, dv in sw]
    rows, last = [], -math.inf
    for i, t in enumerate(ts):
        side = 1 if i and ts[i - 1] == t else -1
        mode = sol.seg_at(t, side).mode
        tf = float(t)
        tf = tf if tf > last else math.nextafter(last, math.inf)   # right limit: one ulp later
        last = tf
        ids = [et.mpf(dv["g_on"] if mode[j] else dv["g_off"]) * val[c][i]
               for (j, dv), c in zip(sw, vcols)]
        rows.append([repr(tf)] + [f"{float(val[c][i]):.13g}" for c in obs + vcols]
                    + [f"{float(x):.13g}" for x in ids])
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)                   # quotes "V(vin,mid)"
        w.writerow(head)
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    sol = et.Solver(et.Circuit(path, "conductance"), 30).run()
    if cmd == "baseline":
        print(write_baseline(sol, sys.argv[4:], sys.argv[3]), "rows")
    else:
        w = float(sys.argv[4])
        thr = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        print(switch_kpi_truth(sol, sys.argv[3], w, thr))
