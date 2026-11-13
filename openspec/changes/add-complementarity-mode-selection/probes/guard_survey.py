"""Slice 2, tasks 6 and 4: the survey circuits on dsed with today's settle (guard off), the default
(settle + guard) and mode_selection='lcp', against trbdf2 at a tight rtol.

    PYTHONPATH=<worktree>/python:<worktree>/python/tests/.. python3 guard_survey.py > guard_survey.out

Per circuit and mode: events, guard checks / repairs, LCP pivots / pins / ties, wall time; per
probed node the end value and the time-weighted mean over the second half (np.trapezoid: the grids
differ), each minus trbdf2's. 'bit-id' = the guarded default run equals the unguarded one exactly
(the guard changed nothing); otherwise the largest difference on a common grid.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import pulsim as p  # noqa: E402
import pulsim._dsed_dispatch as DD  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import shadow_survey as S  # noqa: E402

REF_RTOL = 1e-9

CANDIDATES = (["vout"], ["out"], ["vdc_p-vdc_n"], ["p-n"], ["v_out"], ["output"])


def nodes_for(r):
    """The first candidate node set the result can read ('a-b' is a difference: the bridges' DC
    side floats, and its common mode is a gauge on which the engines disagree, D10)."""
    for cand in CANDIDATES:
        try:
            for nd in cand:
                v(r, nd)
            return cand
        except Exception:  # noqa: BLE001
            continue
    return []


def tw(t, y):
    a = t[0] + 0.5 * (t[-1] - t[0])
    m = t >= a
    return np.trapezoid(y[m], t[m]) / (t[m][-1] - t[m][0])


def run(make, mode, guard=True, engine="dsed"):
    DD.MODE_GUARD = guard
    try:
        b, fn, t_end = make()
        kw = {} if fn is None else {"switch_fn": fn}
        if mode:
            kw["mode_selection"] = mode
        if engine == "trbdf2":
            kw["rtol"] = REF_RTOL
        t0 = time.perf_counter()
        r = p.simulate(b, t_end=t_end, engine=engine, **kw)
        return b, r, time.perf_counter() - t0
    finally:
        DD.MODE_GUARD = True


def v(r, nd):
    if "-" in nd:
        a, b = nd.split("-")
        return np.asarray(r.v(a), dtype=float) - np.asarray(r.v(b), dtype=float)
    return np.asarray(r.v(nd), dtype=float)


def main():
    print("pulsim from", p.__file__)
    for label, make in S.CASES:
        try:
            b, r_off, t_off = run(make, None, guard=False)
            _, r_g, t_g = run(make, None, guard=True)
            _, r_l, t_l = run(make, "lcp")
        except Exception as e:  # noqa: BLE001
            print(f"{label}: FAILS {type(e).__name__}: {str(e)[:160]}")
            continue
        try:
            _, r_ref, _ = run(make, None, engine="trbdf2")
        except Exception as e:  # noqa: BLE001
            r_ref = None
            print(f"{label}: trbdf2 reference FAILS {type(e).__name__}: {str(e)[:120]}")
        nds = nodes_for(r_off)
        same = (np.array_equal(np.asarray(r_off.times), np.asarray(r_g.times))
                and np.array_equal(np.asarray(r_off.states), np.asarray(r_g.states)))
        g = r_g.diagnostics.get("mode_guard", {})
        s = r_l.diagnostics.get("mode_selection", {})
        print(f"== {label}")
        print(f"   events settle/guard/lcp = {r_off.n_events}/{r_g.n_events}/{r_l.n_events}; "
              f"guard checks={g.get('checks', 0)} repairs={g.get('repairs', 0)} "
              f"confirmed={g.get('confirmed', 0)} unchecked={g.get('unchecked', 0)}; "
              f"lcp max_piv={s.get('max_pivots_per_event', 0)} pins={s.get('leak_pins', 0)} "
              f"ties={s.get('ties', 0)} widenings={s.get('widenings', 0)}; "
              f"t settle/guard/lcp = {t_off:.2f}/{t_g:.2f}/{t_l:.2f} s")
        if g.get("repairs"):
            print(f"   repairs at {['%.6e' % x for x in g['repair_times'][:4]]} "
                  f"diodes {sorted(set(g['repair_diodes']))}")
        if not same:
            for nd in nds:
                t1, t2 = np.asarray(r_off.times), np.asarray(r_g.times)
                tt = np.linspace(max(t1[0], t2[0]), min(t1[-1], t2[-1]), 50001)
                d = np.interp(tt, t1, v(r_off, nd)) - np.interp(tt, t2, v(r_g, nd))
                print(f"   guard vs settle: max |dv({nd})| = {np.abs(d).max():.3e}")
        else:
            print("   guard vs settle: bit-identical")
        for nd in nds:
            row = f"   v({nd}):"
            ref_end = ref_avg = None
            if r_ref is not None:
                tr = np.asarray(r_ref.times)
                ref_end, ref_avg = v(r_ref, nd)[-1], tw(tr, v(r_ref, nd))
                row += f" trbdf2 end={ref_end:.9f} mean={ref_avg:.9f} |"
            for tag, r in (("settle", r_off), ("guard", r_g), ("lcp", r_l)):
                t = np.asarray(r.times)
                e, a = v(r, nd)[-1], tw(t, v(r, nd))
                if ref_end is None:
                    row += f" {tag} end={e:.9f}"
                else:
                    row += f" {tag} d_end={e - ref_end:+.2e} d_mean={a - ref_avg:+.2e}"
            print(row)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
