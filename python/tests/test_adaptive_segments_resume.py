"""``run_transient_adaptive`` chains segments without losing history.

Each segment used to restart ``simulate`` from the previous segment's
state alone (``initial_state``), which re-invents the trapezoidal
companion history as zero: a first-order kick at every boundary. On a
10 kHz buck with ``dt_max = dt_init`` (the fixed-step grid, 20
segments) the output read 7.962 V against 8.043 V fixed-step. Segments
now resume from ``final_snapshot`` and match the fixed run exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as p


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("M1", "vin", "sw", 1.0e3, 1.0e-4)
    b.add_diode("D1", "gnd", "sw", 20.0, 1e-5, 0.7)
    b.add_inductor("L1", "sw", "out", 220e-6)
    b.add_capacitor("C1", "out", "gnd", 220e-6)
    b.add_resistor("Rload", "out", "gnd", 8.0)
    b.add_resistor("Rbleed_sw", "sw", "gnd", 1e5)
    return b


def test_adaptive_on_the_fixed_grid_equals_the_fixed_run():
    assert p.__file__.startswith(str(__import__("pathlib").Path(__file__).parents[2]))
    dt, t_end = 2e-6, 4e-3

    def sw(b):
        return p.make_pwm_switch_fn(frequency=10e3, duty=0.3, switch_idx=0,
                                    num_switches=b.graph.num_switches)

    b1 = _buck()
    r1 = p.simulate(b1, t_end=t_end, dt=dt, switch_fn=sw(b1))
    b2 = _buck()
    r2 = p.run_transient_adaptive(b2, t_end=t_end, dt_init=dt, dt_max=dt, switch_fn=sw(b2))
    k = b1.node_id_of("out")
    v1 = np.asarray(r1.states)[:, k]
    v2 = np.asarray(r2.states)[:, k]
    assert len(r2.dt_history) == 20
    assert np.asarray(r2.times) == pytest.approx(np.asarray(r1.times), abs=1e-12)
    m1 = p.time_average(np.asarray(r1.times), v1, t_from=3.2e-3)
    m2 = p.time_average(np.asarray(r2.times), v2, t_from=3.2e-3)
    assert m1 == pytest.approx(8.0431, abs=2e-3)
    assert m2 == pytest.approx(m1, abs=1e-6)
    assert np.max(np.abs(v2 - v1)) < 1e-6
