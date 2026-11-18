"""Overhead of the IR path (add-immutable-circuit-ir, final slice).

Measures, interleaved and repeated (min and median of REPS runs):

* ``simulate(builder)`` against ``builder.build()`` + ``simulate(circuit)``
  on a small circuit (RLC buck-like, 6 branches) and on the n = 1502
  ladder;
* ``circuit.with_value("Rload.R", x)`` per sweep point, against the
  ``simulate`` of that point.

Usage: python ir_overhead.py   (prints one table)
"""
from __future__ import annotations

import statistics
import time
import warnings

import pulsim as p

REPS = 15


def small():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_switch("S", "in", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("Rload", "out", "gnd", 5.0)
    return b, dict(t_end=2e-4, dt=1e-7, engine="pwl")


def ladder(N=500):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3)
    for k in range(N):
        b.add_inductor(f"L{k}", f"n{k}", f"m{k}", 1e-5)
        b.add_resistor(f"R{k}", f"m{k}", f"n{k+1}", 0.1)
        b.add_capacitor(f"C{k}", f"n{k+1}", "gnd", 1e-7)
    b.add_resistor("Rload", f"n{N}", "gnd", 50.0)
    return b, dict(t_end=2e-5, dt=1e-7, engine="pwl")


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    warnings.simplefilter("ignore")
    rows = []
    for name, make in (("small (6 br)", small), ("ladder (1502 br)", ladder)):
        b, kw = make()
        p.simulate(b, **kw)                      # warm caches / imports
        a, c = [], []
        for _ in range(REPS):                    # interleaved
            a.append(timed(lambda: p.simulate(b, **kw)))
            c.append(timed(lambda: p.simulate(b.build(), **kw)))
        circ = b.build()
        n = 200
        wv = timed(lambda: [circ.with_value("Rload.R", 1.0 + i) for i in range(n)]) / n
        bld = timed(lambda: [b.build() for _ in range(20)]) / 20
        rows.append((name, min(a), statistics.median(a), min(c),
                     statistics.median(c), bld, wv))
    print(f"{'circuit':18} {'sim(builder) ms min/med':>24} "
          f"{'build+sim(circuit) ms min/med':>30} {'ratio(med)':>10} "
          f"{'build() us':>11} {'with_value us':>14} {'wv/sim':>8}")
    for name, amin, amed, cmin, cmed, bld, wv in rows:
        print(f"{name:18} {amin*1e3:11.3f} / {amed*1e3:9.3f} "
              f"{cmin*1e3:14.3f} / {cmed*1e3:12.3f} {cmed/amed:10.3f} "
              f"{bld*1e6:11.1f} {wv*1e6:14.1f} {wv/amed:8.4f}")


if __name__ == "__main__":
    main()
