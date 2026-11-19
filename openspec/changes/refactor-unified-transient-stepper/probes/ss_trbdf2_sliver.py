"""steady_state(engine='trbdf2', dt=1e-7) on a 24 V buck: "no periodic orbit
found". Diagnosis (refactor-unified-transient-stepper DECISIONS 11).

Prints, for the base run and every monodromy probe, whether the accepted
step grid equals the base run's (it does, at three probe sizes: the step
sequence is NOT what breaks the map), the per-component column of the
monodromy (the last history slot's entry scales as 1/probe: a fixed jump,
not a derivative), the sliver steps of a period, and how far one period
moves each history slot from the linear-solve answer.

Run: PYTHONPATH=python python3 openspec/changes/refactor-unified-transient-stepper/probes/ss_trbdf2_sliver.py
"""
import warnings

import numpy as np

import pulsim as p
from pulsim.steady_state import _snapshot_like

warnings.simplefilter("ignore")
T, DT = 1e-5, 1e-7


def buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):   # edges 1e-9*T before k*T and k*T + T/2
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / T) - np.floor(t / T + 1e-9) < 0.5 - 1e-9))
        return m
    return b, fn


def main():
    b, fn = buck()
    for eng in ("pwl", "trbdf2"):
        try:
            r = p.steady_state(b, period=T, dt=DT, switch_fn=fn, engine=eng)
            print(f"{eng}: ok, residual {r.residual:.2e}")
        except ValueError as e:
            print(f"{eng}: ValueError {str(e)[:70]}...")

    eng = "trbdf2"
    s0 = p.simulate(b, t_end=T, dt=DT, switch_fn=fn, engine=eng).final_snapshot
    h0 = np.asarray(list(s0.history), float)

    def run(h):
        return p.simulate(b, t_end=float(s0.t) + T, dt=DT, switch_fn=fn,
                          engine=eng, resume_from=_snapshot_like(s0, h))
    base = run(h0)
    c = np.asarray(list(base.final_snapshot.history))
    tb = np.asarray(base.times)
    d = np.diff(tb)
    print("period steps", d.size, "; the two smallest:", np.sort(d)[:2],
          "at t - t0 =", (tb[np.argsort(d)[:2]] - s0.t))
    for scale in (1e-4, 1e-6, 1e-8):
        for j in range(h0.size):
            hj = h0.copy()
            hj[j] += scale * np.abs(c).max()
            r = run(hj)
            same = np.array_equal(np.asarray(r.times), tb)
            col = (np.asarray(list(r.final_snapshot.history)) - c) / (hj[j] - h0[j])
            print(f"probe {scale:g} slot {j}: same grid {same}, column "
                  f"{np.array2string(col, precision=4)}")

    r = p.steady_state(b, period=T, dt=DT, switch_fn=fn, engine=eng, verify=False)
    s = r.snapshot
    for it in range(2):
        nxt = p.simulate(b, t_end=float(s.t) + T, dt=DT, switch_fn=fn,
                         engine=eng, resume_from=s).final_snapshot
        mv = np.asarray(list(nxt.history)) - np.asarray(list(s.history))
        print(f"period {it} from the answer moves the slots by "
              f"{np.array2string(mv, precision=3)}")
        s = nxt


if __name__ == "__main__":
    main()
