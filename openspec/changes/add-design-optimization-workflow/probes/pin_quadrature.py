"""Probe (DECISIONS 29): p_in of a DCM buck on the fixed-step engine.

    PYTHONPATH=python python3 openspec/changes/add-design-optimization-workflow/probes/pin_quadrature.py

The exact route's p_in, then on engine='pwl' at T/200, T/800 and T/3200:
p_in with the period's samples integrated straight across each gate edge
(the old reading) and with each side of the edge integrated separately
(the evaluator's reading), on the pss and the transient routes, and the
transient route with discontinuity_restart=True. Prints observations;
python/tests/test_design_pin_quadrature.py asserts the claims."""

import warnings

import pulsim as p
import pulsim.design as D
from pulsim.design import Evaluator, SteadySpec
from pulsim.stats import time_average

warnings.simplefilter("ignore")
T = 1e-5
DESIGN = dict(L=20e-6, R=50.0)


def buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "x", d["L"])
    b.add_resistor("RLs", "x", "vout", 0.01)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", d["R"])
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


def main():
    ex = Evaluator(buck, period=T, loads=("RL",),
                   steady=SteadySpec(method="pss")).evaluate(DESIGN)
    print(f"exact (pss/dsed): p_in {ex.kpis['p_in']:.9f} W, efficiency "
          f"{ex.kpis['efficiency']:.12f}")
    for method in ("pss", "transient"):
        for n in (200, 800, 3200):
            ev = Evaluator(buck, period=T, loads=("RL",), engine="pwl",
                           options={"dt": T / n},
                           steady=SteadySpec(method=method))
            src, sf = ev._build(DESIGN)
            run, a, z, b = ev._steady(src, sf, T, {"simulations": 0})
            col = int(b.pool.branch_var_id_for_source(
                [c for c in b.components() if c["name"] == "Vin"][0][
                    "branch_id"], b.graph))

            def y(st):
                return st[:, col]
            t0, i0 = D._with_right_limits(run, y)
            t1, i1 = D._with_right_limits(run, y,
                                          edges=D._gate_edges(sf, a, z))
            old = -24.0 * time_average(t0, i0, a, z)
            new = -24.0 * time_average(t1, i1, a, z)
            rec = ev.evaluate(DESIGN)
            print(f"{method:9s} T/{n:<5d} p_in straight {old:.9f} "
                  f"({old / ex.kpis['p_in'] - 1:+.2e}), two-sided {new:.9f} "
                  f"({new / ex.kpis['p_in'] - 1:+.2e}); p_out "
                  f"{rec.kpis['p_out'] / ex.kpis['p_out'] - 1:+.2e}, "
                  f"efficiency {rec.kpis['efficiency'] - ex.kpis['efficiency']:+.1e}, "
                  f"residual {rec.counters.get('energy_residual', float('nan')):+.2e}")
    for n in (200, 800):
        rec = Evaluator(buck, period=T, loads=("RL",), engine="pwl",
                        options={"dt": T / n, "discontinuity_restart": True},
                        steady=SteadySpec(method="transient")).evaluate(DESIGN)
        print(f"transient T/{n} discontinuity_restart: p_out "
              f"{rec.kpis['p_out'] / ex.kpis['p_out'] - 1:+.2e}")


if __name__ == "__main__":
    main()
