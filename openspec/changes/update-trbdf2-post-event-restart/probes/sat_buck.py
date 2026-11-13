"""Saturable inductor as a buck's inductor, gated PWM, trbdf2 vs pwl.

Before update-trbdf2-event-first-step: trbdf2 lost the whole inductor
current at a turn-off edge (v_out 7.68 V vs 12.85 V on pwl).
Usage: python sat_buck.py [engine ...]
"""
import sys
import numpy as np
import pulsim as p


def build():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_saturable_inductor("L", "sw", "out", 100e-6, 20.0, 10e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", 2.0)
    return b


T = 1e-5


def run(engine, t_end=20 * T, **kw):
    b = build()
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / T) - np.floor(t / T + 1e-9) < 0.5 - 1e-9))
        return m
    if engine == "pwl":
        kw.setdefault("dt", T / 400)
    return p.simulate(b, t_end=t_end, engine=engine, switch_fn=fn, **kw)


def tw(r, node="out", t0=10 * T):
    t = np.asarray(r.times)
    v = np.asarray(r.v(node))
    m = t > t0
    return float(np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0]))


if __name__ == "__main__":
    for eng in sys.argv[1:] or ["pwl", "trbdf2"]:
        try:
            r = run(eng)
            st = getattr(r, "_trbdf2_stats", {}) or {}
            print(eng, "tw v_out", tw(r), "end", float(np.asarray(r.v("out"))[-1]),
                  "max|v_sw|", float(np.abs(np.asarray(r.v("sw"))).max()),
                  {k: st.get(k) for k in ("n_accept", "n_reject", "n_solves",
                                          "n_gate_events", "n_diode_events")})
        except Exception as e:  # noqa: BLE001
            print(eng, "ERR", type(e).__name__, str(e)[:400])
