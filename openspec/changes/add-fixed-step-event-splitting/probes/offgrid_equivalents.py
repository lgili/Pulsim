"""Off-grid edge error, flag off / restart rule / rule + split, on the
equivalents of the integration-branch circuits the rule made worse:

  buck   D = 0.4567 (edge oracle: make_pwm_switch_fn)
  fb     full bridge, 50 kHz, phase shift 0.4567*pi (off the grid), R-L
         between the legs (make_phase_shift_full_bridge_fn)
  vcsw   switch gated by a 0/10 V pulse through a 5 V threshold
         (delay 20.37 us), R-L load; `opaque`: switch_fn reads the pulse
         (no oracle, as a node-voltage-controlled switch); `sched`: the
         same gate as a schedule with next_edge_after

Error: state at t_end against trbdf2 rtol 1e-10 driven by the schedule.
Run: PYTHONPATH=python python3 offgrid_equivalents.py
"""
import math
import warnings

import numpy as np

import pulsim as p

warnings.simplefilter("ignore")
MODES = {"off": {}, "rule": {"discontinuity_restart": True},
         "split": {"discontinuity_restart": True,
                   "enable_substep_state_correction": True}}
GON, GOFF = 1e3, 1e-9


def ref(b, fn, t_end):
    return p.simulate(b, t_end=t_end, engine="trbdf2", switch_fn=fn,
                      rtol=1e-10, atol=1e-12)


def end_err(make, t_end, var, N, T):
    b, fn = make(False)
    r0 = ref(b, fn, t_end)
    x_ref = np.asarray(getattr(r0, var[0])(var[1]))[-1]
    out = {}
    for name, kw in MODES.items():
        errs = []
        for n in N:
            b, fn = make(False)
            r = p.simulate(b, t_end=t_end, dt=T / n, engine="pwl",
                           switch_fn=fn, **kw)
            errs.append(float(np.asarray(getattr(r, var[0])(var[1]))[-1] - x_ref))
        out[name] = errs
    return out


def buck(_):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "vout", 50e-6)
    b.add_capacitor("C", "vout", "gnd", 20e-6)
    b.add_resistor("RL", "vout", "gnd", 1.0)
    fn = p.make_pwm_switch_fn(100e3, 0.4567, b.switch_index_of("S"),
                              b.graph.num_switches, 0.0)
    return b, fn


def fb(_):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "dc", "gnd", 100.0)
    for leg in "ab":
        b.add_switch("S" + leg + "h", "dc", leg, GON, GOFF)
        b.add_switch("S" + leg + "l", leg, "gnd", GON, GOFF)
    b.add_resistor("R", "a", "m", 2.0)
    b.add_inductor("L", "m", "b", 50e-6)
    idx = [b.switch_index_of(s) for s in ("Sah", "Sal", "Sbh", "Sbl")]
    fn = p.make_phase_shift_full_bridge_fn(50e3, 0.4567 * math.pi, *idx,
                                           b.graph.num_switches, 0.0)
    return b, fn


TD, TW, TP = 20.37e-6, 30e-6, 50e-6


class _Gate:
    """0/10 V pulse through a 5 V threshold; next_edge_after = the oracle."""

    def __init__(self, n, k):
        self.n, self.k = n, k

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        u = (t - TD) % TP if t >= TD else -1.0
        m.set(self.k, 0.0 <= u < TW)
        return m

    def next_edge_after(self, t):
        if t < TD:
            return TD
        c = TD + math.floor((t - TD) / TP) * TP
        for e in (c + TW, c + TP, c + TP + TW):
            if e > t:
                return e
        return math.inf


def vcsw(opaque):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "out", GON, GOFF)
    b.add_resistor("R", "out", "m", 10.0)
    b.add_inductor("L", "m", "gnd", 100e-6)
    g = _Gate(b.graph.num_switches, b.switch_index_of("S"))
    return b, (g.__call__ if opaque else g)


if __name__ == "__main__":
    N = (50, 100, 200, 400)
    cases = [("buck D=0.4567 v_C", buck, 40.7 / 100e3, ("v", "vout"), 1 / 100e3),
             ("full bridge phi=0.4567pi i_L", fb, 20.1 / 50e3, ("i", "L"), 1 / 50e3),
             ("vcswitch sched i_L", lambda o: vcsw(False), 3 * TP, ("i", "L"), TP)]
    for title, mk, te, var, T in cases:
        res = end_err(mk, te, var, N, T)
        for m, e in res.items():
            f = [abs(e[i] / e[i + 1]) if e[i + 1] else math.inf for i in range(len(e) - 1)]
            print(f"{title:32s} {m:5s} T/{N[0]}..T/{N[-1]}: "
                  + " ".join(f"{v:+.2e}" for v in e)
                  + "  order " + " ".join(f"{math.log2(x):.2f}" for x in f))
    # opaque gate: no oracle, the split cannot see the edge
    b, g = vcsw(False)
    x_ref = np.asarray(ref(b, g, 3 * TP).i("L"))[-1]
    for m, kw in MODES.items():
        e = []
        for n in N:
            b, fn = vcsw(True)
            r = p.simulate(b, t_end=3 * TP, dt=TP / n, engine="pwl", switch_fn=fn, **kw)
            e.append(float(np.asarray(r.i("L"))[-1] - x_ref))
        print(f"{'vcswitch opaque i_L':32s} {m:5s} T/{N[0]}..T/{N[-1]}: "
              + " ".join(f"{v:+.2e}" for v in e))
