"""Tasks 4.2 / 4b.3 / 6.3 on pwl: mode_selection='shadow' and 'lcp' over the survey families.

    PYTHONPATH=<worktree>/python:<worktree>/benchmarks python3 pwl_shadow_survey.py > pwl_shadow_survey.out

Every circuit runs on the fixed-step engine (dt = T/200 for the converters, 10 us for the 50 Hz
bridges) in three configurations (plain trapezoid, the restart rule, restart + event split) and
four ways: the settle alone (guard off), 'shadow' (guard off: the LCP posed at the same sites
beside the settle), the default (settle + guard) and 'lcp'. Columns:

  sites    decision sites the shadow posed an LCP at (step / restart stage / split sub-step)
  disagr   sites where the settle and the LCP chose different modes; each is judged by the
           engine's own law on the site's solve under both modes (ON: i >= -tol, OFF:
           v <= V_on + tol): LCP / settle / both / neither = which candidate is consistent
  breach   steps the settle ended without a fixed point (guard off)
  repairs  the default guard's repairs (an inconsistent settled mode replaced by the LCP's)
  err_off, err_def, err_lcp   end value of the settle alone, the default and 'lcp' against
           trbdf2 (rtol 1e-10); the half-wave rectifier compares its time-weighted mean
  sh=off   'shadow' bit-identical to the settle alone; def=off: the default bit-identical too
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


def fam(name, vth, t_end, dt, node):
    def make():
        f = B.FAMILIES[name]
        b = f.builder(vth)
        sws = [a[0] for k, *a in f.parts if k == "S"]
        fn = pwm(b, sws[0], 0.5 if name.startswith("boost") else 0.3) if sws else None
        return b, fn, t_end, dt, node
    return make


def buck_rest():
    b = B.FAMILIES["buck_rest"].builder(0.0)
    off = p.SwitchStateMask(b.graph.num_switches)
    return b, (lambda t: off), 1e-4, T / 200, "sw"


def buck_l(delta, sign):
    def make():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 24.0)
        b.add_switch("S", "vin", "sw", GON, GOFF)
        b.add_diode("D1", "gnd", "sw", GON, GOFF, 0.0)
        b.add_inductor("L", "sw", "vout", 100e-6 * (1 + sign * delta))
        b.add_capacitor("C", "vout", "gnd", 47e-6)
        b.add_resistor("R", "vout", "gnd", 50.0)
        return b, pwm(b, "S", 0.3), 400 * T, T / 200, "vout"
    return make


def half_wave_rl():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "in", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D1", "in", "out", 200.0, 1e-6)
    b.add_resistor("R1", "out", "nl", 10.0)
    b.add_inductor("L1", "nl", "gnd", 1e-3)
    return b, None, 0.06, 1e-5, "mean:out"


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
        return b, None, 0.04, 1e-5, "p0-n0"
    return make


CASES = [
    ("buck at rest, gate off, V_th=0", buck_rest),
    ("DCM buck V_th=0, 400 T", fam("dcm_buck", 0.0, 400 * T, T / 200, "vout")),
    ("DCM buck V_th=1e-2, 400 T", fam("dcm_buck", 1e-2, 400 * T, T / 200, "vout")),
    ("boost R=50 V_th=0, 400 T", fam("boost_50", 0.0, 400 * T, T / 200, "out")),
    ("boost R=500 V_th=0, 400 T", fam("boost_500", 0.0, 400 * T, T / 200, "out")),
    ("half-wave RL", half_wave_rl),
    ("1-ph bridge V_f=0.7", fam("bridge1_vf07", 0.0, 0.2, 1e-5, "p-n")),
    ("1-ph bridge V_f=0", fam("bridge1_vf0", 0.0, 0.2, 1e-5, "p-n")),
    ("3-ph bridge overlap L_s=1mH", fam("bridge3_overlap", 0.0, 0.04, 1e-5, "p-n")),
    ("3-ph bridge C filter V_f=0.7", fam("bridge3_cfilter", 0.0, 0.2, 1e-5, "p-n")),
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


def end(r, node):
    if node.startswith("mean:"):    # time-weighted mean (adaptive grids: np.trapezoid)
        t, v = np.asarray(r.times), np.asarray(r.v(node[5:]))
        return float(np.trapezoid(v, t) / (t[-1] - t[0]))
    if "-" in node:
        a, c = node.split("-")
        return float(np.asarray(r.v(a))[-1] - np.asarray(r.v(c))[-1])
    return float(np.asarray(r.v(node))[-1])


def run(make, kw, mode, guard):
    b, fn, t_end, dt, _ = make()
    k = dict(kw)
    if fn is not None:
        k["switch_fn"] = fn
    if mode:
        k["mode_selection"] = mode
    p._PWL_MODE_GUARD = guard
    try:
        return p.simulate(b, t_end=t_end, dt=dt, engine="pwl", **k)
    finally:
        p._PWL_MODE_GUARD = True


def reference(make):
    b, fn, t_end, _, node = make()
    k = {"switch_fn": fn} if fn is not None else {}
    try:
        r = p.simulate(b, t_end=t_end, engine="trbdf2", rtol=1e-10, atol=1e-12, **k)
        return end(r, node)
    except Exception as e:   # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:60]}"


def same(a, b):
    return (np.array_equal(np.asarray(a.times), np.asarray(b.times))
            and np.array_equal(np.asarray(a.states), np.asarray(b.states)))


def main():
    print("pulsim from", p.__file__)
    print(f"{'circuit':32s} {'config':11s} {'sites':>6s} {'disagr':>6s} {'LCP':>4s} {'settle':>6s} "
          f"{'both':>4s} {'neith':>5s} {'breach':>6s} {'repairs':>7s} {'err_def':>9s} {'err_lcp':>9s} "
          f"{'maxpiv':>6s} {'sh=off':>6s} {'def=off':>7s} {'lcp=def':>7s}")
    for label, make in CASES:
        ref = ("skipped: trbdf2 does not finish it (max_steps / h_min, KNOWN_LIMITATIONS)"
               if ("bank" in label or "overlap" in label) else reference(make))
        node = make()[4]
        print(f"# {label}: trbdf2 rtol 1e-10 end {node} = {ref}")
        for cfg, kw in CONFIGS:
            t0 = time.perf_counter()
            try:
                r_off = run(make, kw, None, False)
                r_sh = run(make, kw, "shadow", False)
                r_def = run(make, kw, None, True)
            except Exception as e:   # noqa: BLE001 — the survey reports, it does not stop
                print(f"{label:32s} {cfg:11s} FAILS {type(e).__name__}: {str(e)[:110]}")
                continue
            try:
                r_lcp = run(make, kw, "lcp", True)
            except Exception as e:   # noqa: BLE001
                r_lcp = None
                lcp_err = f"{type(e).__name__}: {str(e)[:100]}"
            ms = r_sh.diagnostics["mode_shadow"]
            g = r_def.diagnostics["mode_guard"]
            dis = ms["disagreements"]
            v = [verdict(d) for d in dis]

            def err(r):
                if r is None or isinstance(ref, str):
                    return "     n/a"
                return f"{end(r, node) - ref:+9.2e}"
            print(f"{label:32s} {cfg:11s} {ms['events']:6d} {len(dis):6d} {v.count('LCP'):4d} "
                  f"{v.count('settle'):6d} {v.count('both'):4d} {v.count('neither'):5d} "
                  f"{len(list(r_off.event_iteration_breaches)):6d} {g['repairs']:7d} "
                  f"{err(r_off):>9s} {err(r_def):>9s} {err(r_lcp):>9s} "
                  f"{ms['max_pivots_per_event']:6d} {'yes' if same(r_off, r_sh) else 'NO':>6s} "
                  f"{'yes' if same(r_off, r_def) else 'NO':>7s} "
                  f"{('yes' if same(r_def, r_lcp) else 'no') if r_lcp is not None else '-':>7s}"
                  f"   ({time.perf_counter() - t0:.1f} s)")
            if r_lcp is None:
                print(f"    lcp FAILS {lcp_err}")
            else:
                ml = r_lcp.diagnostics["mode_selection"]
                print(f"    lcp: sites {ml['events']} settle_rounds {ml['settle_rounds_total']} "
                      f"mask_cycles {ml['mask_cycles']} inconsistent {ml['inconsistent']} "
                      f"max_piv {ml['max_pivots_per_event']} ties {ml['ties']} leak_pins {ml['leak_pins']} "
                      f"errors {ml['errors']} breaches {len(list(r_lcp.event_iteration_breaches))}")
            if ms["errors"]:
                print(f"    shadow errors: {ms['errors']} last: {ms['last_error'][:150]}")
            for d in dis[:3]:
                print(f"    t={d['t']:.9e} {d['site']:8s} settle={d['settle_mode']} lcp={d['lcp_mode']} "
                      f"prev={d['previous_mode']} rounds={d['settle_rounds']} -> {verdict(d)} "
                      f"(viol settle {d['settle_violation']:.3g} V, lcp {d['lcp_violation']:.3g} V)")
            for t_, dn, vi in list(zip(g["repair_times"], g["repair_diodes"], g["repair_violations"]))[:3]:
                print(f"    guard repair t={t_:.9e} {dn} violation {vi:.3g} V")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
