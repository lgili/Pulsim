"""Task 8.3, the cost gate: a 60-diode bank (ten three-phase bridges fed straight from their
sources, 6-degree phase steps, each feeding its own 200 uF || 50 ohm; one LCP of 60 ports) on dsed
with the settle alone (guard off), the default (settle + guard) and mode_selection='lcp'. The
variants with 0.2 mH line inductors, and the ten bridges paralleled onto one bus, chatter or fail
to settle on every dsed mode (listed at the end; DECISIONS #20).

    PYTHONPATH=<worktree>/python python3 bank60_cost.py [n_rounds] [--cost-only] > bank60_cost.out

Counters per mode: events, guard checks / repairs, LCP problems, pivots (total, max per event),
port-model problems (one solve for q plus one per free column), ties, widenings, enumerations.
Wall time per run is INTERLEAVED A/B over n_rounds (settle, guard, lcp, settle, ...) and reported as
the median; it is information, never a gate. Reference: trbdf2 rtol 1e-9 on v(p0) - v(n0).
"""
import statistics
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import pulsim as p  # noqa: E402
import pulsim._dsed_dispatch as DD  # noqa: E402

GON, GOFF = 1e3, 1e-9
N_BRIDGE, F, VPK, LS = 10, 50.0, 100.0, 0.2e-3
T_END = 0.04


def bank(vth=0.0, vf=0.0, n_bridge=N_BRIDGE, shared_bus=False, ls=0.0):
    """Ten three-phase bridges, bridge k's sources shifted by 6k degrees, each phase behind L_s.
    shared_bus=False: each bridge feeds its own 200 uF || 50 ohm (the 60 ports of one LCP, ten
    decoupled converters); True: all ten paralleled onto one 2 mF || 5 ohm bus."""
    b = p.CircuitBuilder()
    for k in range(n_bridge):
        pk, nk = ("p", "n") if shared_bus else (f"p{k}", f"n{k}")
        for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
            s, n = f"s{k}{ph}", f"x{k}{ph}"
            if not ls:
                s = n
            b.add_sine_voltage_source(f"V{k}{ph}", s, "gnd", 0.0, VPK, F,
                                      np.deg2rad(ang + 6.0 * k))
            if ls:
                b.add_inductor(f"L{k}{ph}", s, n, ls)
            b.add_diode(f"Dt{k}{ph}", n, pk, GON, GOFF, vth, vf)
            b.add_diode(f"Db{k}{ph}", nk, n, GON, GOFF, vth, vf)
        if not shared_bus:
            b.add_capacitor(f"C{k}", pk, nk, 200e-6)
            b.add_resistor(f"R{k}", pk, nk, 50.0)
            b.add_resistor(f"Rg{k}", nk, "gnd", 1e6)    # ties the floating DC side
    if shared_bus:
        b.add_capacitor("C", "p", "n", 2e-3)
        b.add_resistor("R", "p", "n", 5.0)
        b.add_resistor("Rg", "n", "gnd", 1e6)
    return b


def run(mode, guard, b_kw):
    DD.MODE_GUARD = guard
    try:
        kw = {"mode_selection": mode} if mode else {}
        t0 = time.perf_counter()
        r = p.simulate(bank(**b_kw), t_end=T_END, engine="dsed", **kw)
        return r, time.perf_counter() - t0
    finally:
        DD.MODE_GUARD = True


def vpn(r):
    try:
        return np.asarray(r.v("p0")) - np.asarray(r.v("n0"))
    except Exception:  # noqa: BLE001  (shared bus)
        return np.asarray(r.v("p")) - np.asarray(r.v("n"))


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    print("pulsim from", p.__file__)
    b_kw = {"vth": 0.0}
    print(f"== cost case: 60-diode bank {b_kw}, no line inductance, t_end = {T_END} s")
    t0 = time.perf_counter()
    rr = p.simulate(bank(**b_kw), t_end=T_END, engine="trbdf2", rtol=1e-9)
    print(f"   reference trbdf2 rtol 1e-9: v_pn0 end = {vpn(rr)[-1]:.9f} [{time.perf_counter() - t0:.1f} s]")
    sys.stdout.flush()
    ref = vpn(rr)[-1]
    modes = (("settle", None, False), ("guard", None, True), ("lcp", "lcp", True))
    times = {m[0]: [] for m in modes}
    res = {}
    for _ in range(n_rounds):
        for tag, mode, guard in modes:
            r, dt = run(mode, guard, b_kw)
            times[tag].append(dt)
            res[tag] = r
    for tag, _, _ in modes:
        r = res[tag]
        dg = r.diagnostics or {}
        g = dg.get("mode_guard", {})
        s = dg.get("mode_selection") or g.get("lcp", {})
        ev = max(r.n_events, 1)
        med = statistics.median(times[tag])
        print(f"   {tag:6s} events={r.n_events} v_pn0 end d={vpn(r)[-1] - ref:+.2e} "
              f"| guard checks={g.get('checks', 0)} repairs={g.get('repairs', 0)} "
              f"| lcp problems={s.get('lcp_count', 0)} pivots={s.get('lcp_pivots_total', 0)} "
              f"max/event={s.get('max_pivots_per_event', 0)} port_models={s.get('port_model_problems', 0)} "
              f"ties={s.get('ties', 0)} widenings={s.get('widenings', 0)} pins={s.get('leak_pins', 0)} "
              f"enum={s.get('enumerations', 0)} | masks={len(r.exact['masks'])} n_eig={r.exact['n_eig']} "
              f"| wall median {med * 1e3:.0f} ms "
              f"({med / ev * 1e6:.0f} us/event) min {min(times[tag]) * 1e3:.0f} max {max(times[tag]) * 1e3:.0f}")
    sys.stdout.flush()
    if "--cost-only" in sys.argv:
        return 0
    print("== variants that do not run (one run each, every mode; not cost cases)")
    for b_kw in ({"vth": 0.0, "ls": LS}, {"vth": 1e-2, "ls": LS}, {"vth": 0.0, "ls": LS, "shared_bus": True}):
        for tag, mode, guard in modes:
            try:
                r, dt = run(mode, guard, b_kw)
                print(f"   {b_kw} {tag}: runs, {r.n_events} events, v_pn0 end {vpn(r)[-1]:.6f} [{dt:.1f} s]")
            except Exception as e:  # noqa: BLE001
                print(f"   {b_kw} {tag}: {type(e).__name__}: {str(e)[:110]}")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
