"""Reproducer for the TR-BDF2 track: a resumed run ticks at t_start + k*T.

`run_trbdf2_transient(observers=, observer_periods=)` schedules observer
j at t_start + k*observer_periods[j]. A control graph resumed between
two of its ticks (resume_from at 1.53 ms, period 100 us) then ticks at
1.63, 1.73 ms ... instead of 1.6, 1.7 ms: its duty moved by up to 0.9 %
against the uninterrupted run (measured 2026-11-18 on a closed-loop
buck, before simulate() refused it). Since G26 simulate() refuses that
resume on trbdf2 by name (feature 'resume_from'; 'auto' routes to pwl).
What would lift the refusal: a phase per observer (first tick instant),
e.g. `observer_first_ticks`, so the graph's own grid t0 + k*T is kept.

  PYTHONPATH=<repo>/python python trbdf2_resume_tick_phase.py
"""
import numpy as np

import pulsim as p


def main():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "u", "gnd", 1.0)
    b.add_resistor("R", "u", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-7)
    seen = []

    class Obs:
        period = 1e-4

        def tick(self, t, x):
            seen.append(t)

        def __call__(self, t, x):
            self.tick(t, x)

    p.simulate(b, 1.9e-3, t_start=1.53e-3, engine="trbdf2",
               step_observer=Obs())
    print("tick instants (ms):", np.round(np.asarray(seen) * 1e3, 6))
    print("a graph started at 0 with period 0.1 ms ticks at 1.6, 1.7, 1.8 ms")


if __name__ == "__main__":
    main()
