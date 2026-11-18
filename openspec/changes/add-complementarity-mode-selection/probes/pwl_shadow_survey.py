"""Tasks 4.2 / 4b.3 on pwl: mode_selection='shadow' over the dsed survey's families -> the table.

    PYTHONPATH=<worktree>/python:<worktree>/benchmarks python3 pwl_shadow_survey.py > pwl_shadow_survey.out

Every circuit runs on the fixed-step engine (dt = T/200 for the converters, 10 us for the 50 Hz
bridges) with today's settle deciding, the guard OFF (the settle alone), and the LCP posed on the
same site (step / restart stage / split sub-step) beside it. For each disagreement the site is
solved under BOTH candidate modes and each checked against the engine's own law (ON: i >= -tol,
OFF: v <= V_on + tol) — "right" = consistent there. The default run (guard on) is then compared
with the guard-off run: bit-identical unless the guard repaired a wrong settle.
"""
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import pulsim as p  # noqa: E402
from tests.oracles import lcp_pulsim_bridge as B  # noqa: E402

T = B.T
GON, GOFF = 1e3, 1e-9


def pwm(b, name, D):
    return p.make_pwm_switch_fn(1 / T, D, b.switch_index_of(name), b.graph.num_switches, 0.0)


def fam(name, vth, t_end, dt):
    def make():
        f = B.FAMILIES[name]
        b = f.builder(vth)
        sws = [a[0] for k, *a in f.parts if k == "S"]
        fn = pwm(b, sws[0], 0.5 if name.startswith("boost") else 0.3) if sws else None
        return b, fn, t_end, dt
    return make


def buck_rest():
    b = B.FAMILIES["buck_rest"].builder(0.0)
    off = p.SwitchStateMask(b.graph.num_switches)
    return b, (lambda t: off), 1e-4, T / 200


def buck_l(delta, sign):
    def make():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 24.0)
        b.add_switch("S", "vin", "sw", GON, GOFF)
        b.add_diode("D1", "gnd", "sw", GON, GOFF, 0.0)
        b.add_inductor("L", "sw", "vout", 100e-6 * (1 + sign * delta))
        b.add_capacitor("C", "vout", "gnd", 47e-6)
        b.add_resistor("R", "vout", "gnd", 50.0)
        return b, pwm(b, "S", 0.3), 400 * T, T / 200
    return make


def half_wave_rl():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D1", "in", "out", 200.0, 1e-6)
    b.add_resistor("R1", "out", "nl", 10.0)
    b.add_inductor("L1", "nl", "gnd", 1e-3)
    return b, None, 0.04, 1e-5


def bank(n_bridge, ls=0.2e-3):
    def make():
        b = p.CircuitBuilder()
        for k in range(n_bridge):
            pk, nk = f"p{k}", f"n{k}"
            for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
                s, n = f"s{k}{ph}", f"x{k}{ph}"
                b.add_sine_voltage_source(f"V{k}{ph}", s, "gnd", 0.0, 100.0, 50.0,
                                          np.deg2rad(ang + 6.0 * k))
                b.add_inductor(f"L{k}{ph}", s, n, ls)
                b.add_diode(f"Dt{k}{ph}", n, pk, GON, GOFF, 0.0)
                b.add_diode(f"Db{k}{ph}", nk, n, GON, GOFF, 0.0)
            b.add_capacitor(f"C{k}", pk, nk, 200e-6)
            b.add_resistor(f"R{k}", pk, nk, 50.0)
            b.add_resistor(f"Rg{k}", nk, "gnd", 1e6)
        return b, None, 0.04, 1e-5
    return make


CASES = [
    ("buck at rest, gate off, V_th=0", buck_rest),
    ("DCM buck V_th=0, 400 T", fam("dcm_buck", 0.0, 400 * T, T / 200)),
    ("DCM buck V_th=1e-2, 400 T", fam("dcm_buck", 1e-2, 400 * T, T / 200)),
    ("boost R=50 V_th=0, 400 T", fam("boost_50", 0.0, 400 * T, T / 200)),
    ("boost R=500 V_th=0, 400 T", fam("boost_500", 0.0, 400 * T, T / 200)),
    ("half-wave RL", half_wave_rl),
    ("1-ph bridge V_f=0.7", fam("bridge1_vf07", 0.0, 0.04, 1e-5)),
    ("1-ph bridge V_f=0", fam("bridge1_vf0", 0.0, 0.04, 1e-5)),
    ("3-ph bridge overlap L_s=1mH", fam("bridge3_overlap", 0.0, 0.04, 1e-5)),
    ("3-ph bridge C filter V_f=0.7", fam("bridge3_cfilter", 0.0, 0.04, 1e-5)),
] + [(f"DCM buck L(1{'+' if s > 0 else '-'}{d:g})", buck_l(d, s))
     for d in (1e-5, 1e-4, 1e-2) for s in (1, -1)] + [
    ("5-bridge bank, L_s=0.2 mH", bank(5)),
    ("10-bridge bank, L_s=0.2 mH", bank(10)),
]

CONFIGS = [("pwl", {}), ("pwl+restart", {"discontinuity_restart": True}),
           ("pwl+split", {"discontinuity_restart": True, "enable_substep_state_correction": True})]


def verdict(d):
    s, l_ = d["settle_consistent"], d["lcp_consistent"]
    return {(1, 0): "settle", (0, 1): "LCP", (1, 1): "both", (0, 0): "neither"}.get((s, l_), "unknown")


def run(make, kw, mode, guard):
    b, fn, t_end, dt = make()
    k = dict(kw)
    if fn is not None:
        k["switch_fn"] = fn
    if mode:
        k["mode_selection"] = mode
    p._PWL_MODE_GUARD = guard
    t0 = time.perf_counter()
    try:
        r = p.simulate(b, t_end=t_end, dt=dt, engine="pwl", **k)
    finally:
        p._PWL_MODE_GUARD = True
    return r, time.perf_counter() - t0


def same(a, b):
    return (np.array_equal(np.asarray(a.times), np.asarray(b.times))
            and np.array_equal(np.asarray(a.states), np.asarray(b.states)))


def main():
    print("pulsim from", p.__file__)
    print(f"{'circuit':32s} {'config':11s} {'sites':>6s} {'disagr':>6s} {'LCP':>4s} {'settle':>6s} "
          f"{'both':>4s} {'neith':>5s} {'breach':>6s} {'repairs':>7s} {'maxpiv':>6s} {'ties':>4s} "
          f"{'wid':>4s} {'shadow=off':>10s} {'guard=off':>9s}")
    for label, make in CASES:
        for cfg, kw in CONFIGS:
            try:
                r_off, _ = run(make, kw, None, False)
                r_sh, _ = run(make, kw, "shadow", False)
                r_def, _ = run(make, kw, None, True)
            except Exception as e:   # noqa: BLE001 — the survey reports, it does not stop
                print(f"{label:32s} {cfg:11s} FAILS {type(e).__name__}: {str(e)[:110]}")
                continue
            ms = r_sh.diagnostics["mode_shadow"]
            g = r_def.diagnostics["mode_guard"]
            dis = ms["disagreements"]
            v = [verdict(d) for d in dis]
            print(f"{label:32s} {cfg:11s} {ms['events']:6d} {len(dis):6d} {v.count('LCP'):4d} "
                  f"{v.count('settle'):6d} {v.count('both'):4d} {v.count('neither'):5d} "
                  f"{len(list(r_off.event_iteration_breaches)):6d} {g['repairs']:7d} "
                  f"{ms['max_pivots_per_event']:6d} {ms['ties']:4d} {ms['widenings']:4d} "
                  f"{'yes' if same(r_off, r_sh) else 'NO':>10s} {'yes' if same(r_off, r_def) else 'NO':>9s}")
            if ms["errors"]:
                print(f"    errors: {ms['errors']} last: {ms['last_error'][:150]}")
            for d in dis[:3]:
                print(f"    t={d['t']:.9e} {d['site']:8s} settle={d['settle_mode']} lcp={d['lcp_mode']} "
                      f"prev={d['previous_mode']} rounds={d['settle_rounds']} -> {verdict(d)} "
                      f"(viol settle {d['settle_violation']:.3g} V, lcp {d['lcp_violation']:.3g} V)")
            for t_, dn, vi in list(zip(g["repair_times"], g["repair_diodes"], g["repair_violations"]))[:3]:
                print(f"    guard repair t={t_:.9e} {dn} violation {vi:.3g} V")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
