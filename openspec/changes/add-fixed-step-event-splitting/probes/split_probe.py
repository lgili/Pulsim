"""Engine probes for add-fixed-step-event-splitting.

    python split_probe.py save  OUT.npz      # record (any build)
    python split_probe.py check OUT.npz      # bit-identity vs OUT.npz
    python split_probe.py table [--split]    # error tables (restart rule on)
    python split_probe.py dcm  [--split]     # p15/p12: DCM buck at T/100, T/101, T/400
    python split_probe.py gates              # resonant gates with off-grid PWM edges
    python split_probe.py cost               # factorisations per period, buck T/100

`save` records the restart-rule probe set (../../add-discontinuity-restart-rule/
probes/pwl_baseline.py) twice -- flag absent and `discontinuity_restart=True`
-- plus the off-grid circuits of this change, with `enable_substep_state_
correction` never passed. `check` re-runs every fixed-engine run with the
keyword absent AND with `enable_substep_state_correction=False`, and the
trbdf2 / dsed runs as they are, and demands np.array_equal on states and
switch bits.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "add-discontinuity-restart-rule", "probes"))
import pwl_baseline as pb  # noqa: E402

import pulsim as p  # noqa: E402

V, R, L = 10.0, 1.0, 1e-3
TAU = L / (R + 1 / pb.GON)


def rl_offgrid(dt, frac, oracle=True):
    """Switch closes at (k + frac)·dt; error of i_L 0.5 ms later [A]."""
    te = (round(1e-3 / dt) + frac) * dt
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_switch("S", "in", "a", pb.GON, pb.GOFF)
    b.add_resistor("R", "a", "m", R)
    b.add_inductor("L", "m", "gnd", L)
    n, k = b.graph.num_switches, b.switch_index_of("S")
    if oracle:   # on from te for 3 ms (T = 4 ms, duty 0.75, phase te)
        fn = p.make_pwm_switch_fn(250.0, 0.75, k, n, te)
    else:
        fn = pb.mask_fn(b, "S", lambda t: t >= te - 1e-15)
    res = pb.sim(b, t_end=te + 0.6e-3 + 1e-12, dt=dt, switch_fn=fn)
    t, i = np.asarray(res.times), np.asarray(res.i("L"))
    j = int(np.argmin(abs(t - (te + 0.5e-3))))
    ex = V / (R + 1 / pb.GON) * (1 - math.exp(-(t[j] - te) / TAU))
    return res, i[j] - ex


def buck_sched(R_load, D, N, nper=40):
    """Buck driven by make_pwm_switch_fn (edge oracle), on from T0."""
    dt = pb.TB / N
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", pb.VIN)
    b.add_switch("S", "vin", "sw", pb.GON, pb.GOFF)
    b.add_diode("D1", "gnd", "sw", pb.GON, pb.GOFF)
    b.add_inductor("L", "sw", "vout", pb.LB)
    b.add_capacitor("C", "vout", "gnd", pb.CB)
    b.add_resistor("RL", "vout", "gnd", R_load)
    fn = p.make_pwm_switch_fn(pb.FSW, D, b.switch_index_of("S"),
                              b.graph.num_switches, pb.T0)
    return pb.sim(b, t_end=pb.T0 + nper * pb.TB + 1e-12, dt=dt, switch_fn=fn)


def buck_sched_err(R_load, D, N, nper=40):
    res = buck_sched(R_load, D, N, nper)
    xe = pb.buck_exact(R_load, D, nper)
    t = np.asarray(res.times)
    j = int(np.argmin(abs(t - (pb.T0 + nper * pb.TB))))
    return res, np.asarray(res.i("L"))[j] - xe[0], np.asarray(res.v("vout"))[j] - xe[1]


def dcm_grid(N, nper, **kw):
    """p15/p12: DCM buck (R = 50, D = 0.3) with a grid-indexed switch_fn."""
    dt = pb.TB / N
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", pb.VIN)
    b.add_switch("S", "vin", "sw", pb.GON, pb.GOFF)
    b.add_diode("D1", "gnd", "sw", pb.GON, pb.GOFF)
    b.add_inductor("L", "sw", "vout", pb.LB)
    b.add_capacitor("C", "vout", "gnd", pb.CB)
    b.add_resistor("RL", "vout", "gnd", 50.0)
    n, k, Don = b.graph.num_switches, b.switch_index_of("S"), round(0.3 * N)

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool(int(round(t / dt)) % N < Don))
        return m
    return pb.sim(b, t_end=nper * pb.TB + 1e-12, dt=dt, switch_fn=fn, **kw), Don / N


def runs():
    out = {}
    for flag in ({}, {"discontinuity_restart": True}):
        pb.FLAG.clear()
        pb.FLAG.update(flag)
        tag = "on" if flag else "abs"
        for name, res in pb.runs().items():
            if name.startswith(("trbdf2", "dsed")) and flag:
                continue
            out[f"{tag}_{name}"] = res
        out[f"{tag}_rl_off03"] = rl_offgrid(2e-6, 0.3)[0]
        out[f"{tag}_rl_off07_opaque"] = rl_offgrid(2e-6, 0.7, oracle=False)[0]
        out[f"{tag}_buck4567_T100"] = buck_sched(1.0, 0.4567, 100)
        out[f"{tag}_dcm_grid_T101"] = dcm_grid(101, 60)[0]
    return out


def arrays(res):
    return np.asarray(res.times), np.asarray(res.states), np.asarray(res.switch_states)


def save(path):
    data = {}
    for name, res in runs().items():
        for part, arr in zip(("t", "x", "sw"), arrays(res)):
            data[f"{name}/{part}"] = arr
    np.savez(path, **data)
    print(f"saved {len(data) // 3} runs to {path}")


def check(path):
    ref = np.load(path)
    bad = 0
    for extra in ({}, {"enable_substep_state_correction": False}):
        orig = pb.sim

        def sim(b, **kw):
            if kw.get("engine", "pwl") == "pwl":
                kw = {**kw, **extra}
            return orig(b, **kw)
        pb.sim = sim
        try:
            for name, res in runs().items():
                for part, arr in zip(("t", "x", "sw"), arrays(res)):
                    if not np.array_equal(ref[f"{name}/{part}"], arr):
                        bad += 1
                        print(f"DIFFERS {extra} {name}/{part}")
        finally:
            pb.sim = orig
        print(f"checked {extra or 'keyword absent'}")
    print("BIT-IDENTICAL" if bad == 0 else f"{bad} ARRAYS DIFFER")
    return bad


def table():
    for frac in (0.3, 0.7):
        for oracle in (True, False):
            e = [rl_offgrid(dt, frac, oracle)[1] for dt in (8e-6, 4e-6, 2e-6, 1e-6)]
            print(f"A. RL edge +{frac} dt ({'schedule' if oracle else 'opaque fn'}), err [A] dt=8,4,2,1 us: "
                  + " ".join(f"{v:+.3e}" for v in e) + "  order " + pb.order(e))
    ev, ei = [], []
    for N in (50, 100, 200, 400, 800):
        _, a, v = buck_sched_err(1.0, 0.4567, N)
        ei.append(a)
        ev.append(v)
    print("B. buck CCM D=0.4567 err vC T/50..T/800: " + " ".join(f"{v:+.2e}" for v in ev) + "  order " + pb.order(ev))
    print("                       err iL           : " + " ".join(f"{v:+.2e}" for v in ei) + "  order " + pb.order(ei))
    res = buck_sched(1.0, 0.4567, 100)
    print(f"   T/100: n_restart_steps={res.n_restart_steps} n_split_steps={getattr(res, 'n_split_steps', 'n/a')}")


def dcm(nper=3000):
    for N in (100, 101, 400):
        res, D = dcm_grid(N, nper)
        v = float(np.asarray(res.v("vout"))[-1])
        xe = pb.buck_exact(50.0, D, nper)[1]
        print(f"DCM T/{N} (D={D:.4f}): vC={v:.6f} exact={xe:.6f} err={v - xe:+.2e} ({(v / xe - 1) * 100:+.3f} %)"
              f" restarts={res.n_restart_steps} splits={getattr(res, 'n_split_steps', 'n/a')}")
    # p16: OFF events vs the inductor current's zero crossing (T/100, last 5 periods)
    res, _ = dcm_grid(100, 1000)
    t, iL = np.asarray(res.times), np.asarray(res.i("L"))
    dt = pb.TB / 100
    offs = []
    for e in res.commutation_events:
        if e.new_state or e.t_estimated < 995 * pb.TB:
            continue
        j = int(np.floor(e.t_estimated / dt + 1e-9))
        i1, i2 = iL[j - 1], iL[j]
        tz = t[j] + i2 / ((i1 - i2) / dt)    # extrapolate the falling ramp to zero
        offs.append((e.t_estimated / dt, (tz - e.t_estimated) / dt))
    print("p16 OFF events (t_est/dt, zero crossing - t_est in dt): "
          + " ".join(f"({a:.4f}, {b:+.4f})" for a, b in offs[:5]))


def gates():
    """The restart rule's resonant gates (resonant_gates.py) with every PWM
    edge moved 0.37 dt off the grid, so the split acts on each edge."""
    import resonant_gates as rg
    modes = {"off": {}, "rule": {"discontinuity_restart": True},
             "split": {"discontinuity_restart": True,
                       "enable_substep_state_correction": True}}

    def tank(N, **kw):
        b = p.CircuitBuilder()
        b.add_pwm_voltage_source("V", "a", "gnd", 50.0, -50.0, rg.F0, 0.5,
                                 0.37 * rg.T / N)
        b.add_resistor("R", "a", "m", rg.R)
        b.add_inductor("L", "m", "n", rg.L)
        b.add_capacitor("C", "n", "gnd", rg.C)
        r = p.simulate(b, t_end=400 * rg.T, dt=rg.T / N, engine="pwl", **kw)
        t, i = np.asarray(r.times), np.asarray(r.i("L"))
        return float(np.abs(i[t > 380 * rg.T]).max()), r

    ref = rg.tank_peak(1600)[0]
    print(f"A. tank, converged peak (T/1600, flag off, on-grid) {ref:.4f} A")
    for N in (50, 100):
        for name, kw in modes.items():
            pk, r = tank(N, **kw)
            print(f"   T/{N} {name:5s}: error {100 * (pk / ref - 1):+.3f} %, "
                  f"restarts {r.n_restart_steps}, splits {r.n_split_steps}")
    w = 1 / math.sqrt(1e-3 * 1e-6)
    Tp = 2 * math.pi / w
    for name, kw in modes.items():
        b = p.CircuitBuilder()
        b.add_inductor("L", "a", "gnd", 1e-3, i0=1.0)
        b.add_capacitor("C", "a", "gnd", 1e-6)
        b.add_pwm_voltage_source("Vp", "x", "gnd", 10.0, 0.0, 2.5 / Tp, 0.5,
                                 0.37 * Tp / 50)
        b.add_resistor("Rx", "x", "gnd", 10.0)
        r = p.simulate(b, t_end=100 * Tp, dt=Tp / 50, engine="pwl", **kw)
        i, v = np.asarray(r.i("L")), np.asarray(r.v("a"))
        E = 0.5e-3 * i ** 2 + 0.5e-6 * v ** 2
        print(f"B. isolated LC {name:5s}: E(100T)/E(0) = {E[-1] / E[0]:.4f}, "
              f"restarts {r.n_restart_steps}, splits {r.n_split_steps}")


class _MovingPwm:
    """A duty that changes every period; declares its edges."""

    def __init__(self, ds, k, n):
        self.ds, self.k, self.n = ds, k, n

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        j = int(math.floor(t / pb.TB + 1e-12))
        m.set(self.k, t - j * pb.TB < self.ds[min(j, len(self.ds) - 1)] * pb.TB - 1e-15)
        return m

    def next_edge_after(self, t):
        j = int(math.floor(t / pb.TB + 1e-12))
        for jj in (j, j + 1):
            for e in ((jj + self.ds[min(jj, len(self.ds) - 1)]) * pb.TB, (jj + 1) * pb.TB):
                if e > t + 1e-15:
                    return e
        return math.inf


def _cost_buck(duty, nper):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", pb.VIN)
    b.add_switch("S", "vin", "sw", pb.GON, pb.GOFF)
    b.add_diode("D1", "gnd", "sw", pb.GON, pb.GOFF)
    b.add_inductor("L", "sw", "vout", pb.LB)
    b.add_capacitor("C", "vout", "gnd", pb.CB)
    b.add_resistor("RL", "vout", "gnd", 1.0)
    k, n = b.switch_index_of("S"), b.graph.num_switches
    if duty == "fixed":
        return b, p.make_pwm_switch_fn(pb.FSW, 0.4567, k, n, 0.0)
    rng = np.random.default_rng(1)
    return b, _MovingPwm(0.40 + 0.1 * rng.random(nper + 1), k, n)


def perf(nper=2000, reps=5):
    """Interleaved wall time per step, rule vs split (report only)."""
    import time
    modes = (("rule", {"discontinuity_restart": True}),
             ("split", {"discontinuity_restart": True,
                        "enable_substep_state_correction": True}))
    for duty in ("fixed", "moving"):
        best = {m: math.inf for m, _ in modes}
        for _ in range(reps):
            for name, kw in modes:
                b, fn = _cost_buck(duty, nper)
                t0 = time.perf_counter()
                p.simulate(b, t_end=nper * pb.TB, dt=pb.TB / 100, engine="pwl",
                           switch_fn=fn, **kw)
                best[name] = min(best[name], time.perf_counter() - t0)
        ns = {m: 1e9 * v / (nper * 100) for m, v in best.items()}
        print(f"{duty:6s}: rule {ns['rule']:.0f} ns/step, split {ns['split']:.0f} ns/step "
              f"({100 * (ns['split'] / ns['rule'] - 1):+.1f} %)")


def cost(nper=200):
    """Factorisations per PWM period on the buck at T/100 (R = 1, CCM):
    edges fixed off the grid (D = 0.4567) and a duty that moves every
    period (a NativeMultiMaskPwm per period is not available, so the
    moving duty is a Python schedule declaring `next_edge_after`)."""
    for name, kw in (("rule", {"discontinuity_restart": True}),
                     ("split", {"discontinuity_restart": True,
                                "enable_substep_state_correction": True})):
        for duty in ("fixed", "moving"):
            caches = []
            N = 100
            b, fn = _cost_buck(duty, nper)
            r = p.simulate(b, t_end=nper * pb.TB, dt=pb.TB / N, engine="pwl",
                           switch_fn=fn, on_cache=caches.append, **kw)
            m = caches[0].metrics()
            print(f"{name:5s} {duty:6s}: per period event_builds "
                  f"{m.event_builds / nper:.2f} refactors {m.event_refactors / nper:.2f}; "
                  f"restart factors {r.n_restart_factorizations}, nominal "
                  f"{caches[0].num_built_segments()}, splits/period {r.n_split_steps / nper:.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    pb.FLAG.clear()
    if cmd in ("table", "dcm"):
        pb.FLAG["discontinuity_restart"] = True
        if "--split" in sys.argv:
            pb.FLAG["enable_substep_state_correction"] = True
        if "--off" in sys.argv:
            pb.FLAG.clear()
    if cmd == "save":
        save(sys.argv[2])
    elif cmd == "check":
        sys.exit(1 if check(sys.argv[2]) else 0)
    elif cmd == "table":
        table()
    elif cmd == "gates":
        gates()
    elif cmd == "cost":
        cost()
    elif cmd == "perf":
        perf()
    else:
        dcm()
