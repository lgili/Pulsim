"""Level-0 add_igbt on-state, gate ON, forced +/-10 A. Read-only probe.

Measured 2026-11-12 on docs/core-roadmap-openspec: |v_CE| = 0.1000 V in both
directions (R_on = 10 mOhm): no knee, no reverse blocking. The sign printed
follows add_current_source's terminal convention (a positive value leaves
the source at `from`), so the first line is the control: the device conducts.
"""
import warnings

import numpy as np

import pulsim as p

warnings.simplefilter("ignore")

for current in (10.0, -10.0):
    b = p.CircuitBuilder()
    b.add_current_source("I", "gnd", "c", current)
    b.add_igbt("Q", "c", "gnd")
    b.add_resistor("Rb", "c", "gnd", 1e6)
    n = b.graph.num_switches

    def gate_on(t, n=n):
        m = p.SwitchStateMask(n)      # a SIZE, every bit off
        m.set(0, True)
        return m

    r = p.simulate(b, t_end=1e-5, dt=1e-6, engine="pwl", switch_fn=gate_on)
    print("add_igbt gated ON, I=%+.0f A -> v_CE=%+.4f V" % (current, np.asarray(r.v("c"))[-1]))
