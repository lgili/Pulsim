"""Task 8.2: cost per event on the D9 families (tests/oracles/lcp_pulsim_bridge.FAMILIES) on dsed.

    PYTHONPATH=<worktree>/python python3 family_cost.py [n_rounds] > family_cost.out

Per family, three runs of the same circuit and drive:
  'shadow'  the settle decides (today's rule) and the LCP is posed on the same held state at
            every event: settle rounds and LCP pivots on the SAME events;
  default   settle + guard (what a user gets);
  'lcp'     the LCP decides: pivots max / mean per event against design D9 (+1 allowed), and
            port-model problems per event (`port_model_problems`: one factor solve each for q;
            each pivot adds a column solve with the same factor).
Solves per event follow D9's count: one factor solve per LCP problem (for q) plus one column
solve per pivot; settle rounds are full re-solves. Pivots per event include tie re-solves (D9's
bracketed numbers); the max is the first LCP's. Wall per event is the whole run's wall / events
(exact stepping included), median of n_rounds INTERLEAVED runs: information, never a gate.
"""
import statistics
import sys
import time
import warnings

warnings.simplefilter("ignore")
import pulsim as p  # noqa: E402
from tests.oracles import lcp_pulsim_bridge as B  # noqa: E402

T = B.T
# family -> (t_end, D9 max pivots per event, first LCP)
D9 = {
    "buck_rest": (400 * T, 2), "dcm_buck": (400 * T, 2), "boost_50": (400 * T, 2),
    "boost_500": (400 * T, 2), "bridge1_vf07": (0.2, 6), "bridge1_vf0": (0.2, 6),
    "bridge3_overlap": (0.1, 4), "bridge3_cfilter": (0.1, 5),
}


def case(name):
    fam = B.FAMILIES[name]
    b = fam.builder(0.0)
    sws = [a[0] for k, *a in fam.parts if k == "S"]
    fn = None
    if name == "buck_rest":
        off = p.SwitchStateMask(b.graph.num_switches)
        fn = lambda t: off  # noqa: E731
    elif sws:
        fn = p.make_pwm_switch_fn(1e5, 0.5 if name.startswith("boost") else 0.3,
                                  b.switch_index_of(sws[0]), b.graph.num_switches, 0.0)
    return b, fn


def run(name, mode):
    b, fn = case(name)
    kw = {"switch_fn": fn} if fn else {}
    if mode:
        kw["mode_selection"] = mode
    t0 = time.perf_counter()
    r = p.simulate(b, t_end=D9[name][0], engine="dsed", **kw)
    return r, time.perf_counter() - t0


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("pulsim from", p.__file__)
    print(f"{'family':16s} {'events':>6s} | {'settle rnds/ev':>14s} {'lcp solves/ev':>13s} | "
          f"{'lcp piv max (D9)':>16s} {'piv/ev':>6s} {'solves/ev':>9s} | {'default us/ev':>13s} {'lcp us/ev':>9s}")
    for name in D9:
        try:
            sh, _ = run(name, "shadow")
            s = (sh.diagnostics or {}).get("mode_shadow", {})
            ev_s = max(s.get("events", 0), 1)
            wall = {"default": [], "lcp": []}
            res = {}
            for _ in range(n_rounds):
                for tag, mode in (("default", None), ("lcp", "lcp")):
                    r, dt = run(name, mode)
                    wall[tag].append(dt / max(r.n_events, 1))
                    res[tag] = r
        except Exception as e:  # noqa: BLE001
            print(f"{name:16s} {type(e).__name__}: {str(e).splitlines()[0][:100]}", flush=True)
            continue
        lc = (res["lcp"].diagnostics or {}).get("mode_selection", {})
        ev_l = max(lc.get("events", 0), 1)
        mx = lc.get("max_pivots_per_event", 0)
        ok = "ok" if mx <= D9[name][1] + 1 else "OVER"

        def solves(d, ev):
            return (d.get("port_model_problems", 0) + d.get("lcp_pivots_total", 0)) / ev
        print(f"{name:16s} {res['lcp'].n_events:6d} | "
              f"{s.get('settle_rounds_total', 0) / ev_s:14.2f} {solves(s, ev_s):13.2f} | "
              f"{mx:>6d} ({D9[name][1]}) {ok:4s} {lc.get('lcp_pivots_total', 0) / ev_l:6.2f} "
              f"{solves(lc, ev_l):9.2f} | "
              f"{statistics.median(wall['default']) * 1e6:13.1f} {statistics.median(wall['lcp']) * 1e6:9.1f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
