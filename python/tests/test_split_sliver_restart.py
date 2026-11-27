"""F-Z-3: the restart rule after a split SLIVER (an event piece much shorter
than dt). A hard-switched capacitor (C = 1 nF, R_on = 1 mOhm: a 1 ps mode,
dt = 1 ns) gated a fraction of dt before a grid point: the step is cut at the
edge and the event's restart probe spans only gamma * (the piece), so the mode
was neither damped nor projected and rang +-2.25 A on under trapezoid (max
deviation 1.022 A at 1e-5 dt, 0.362 A at 1e-3 dt). The next grid step's restart
now probes it with the event before its start. Options explicit: split_snap
is reachable only through run_transient."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

TB, DT, GON, GOFF, V, R = 1e-5, 1e-9, 1e3, 1e-9, 100.0, 10.0
I_ON = V / (R + 1 / GON)


def _cap_cell(off, split=True):
    """Edges at (k + 0.5 - off) TB and (k + 1 - off) TB; off < 0: after."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_resistor("R", "in", "sw", R)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_capacitor("C", "sw", "gnd", 1e-9)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def sw(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / TB) - math.floor(t / TB + off) >= 0.5 - off))
        return m

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(DT)
    o = p.SimulationOptions(t_start=0.0, t_end=20 * TB + 1e-12, dt=DT)
    o.discontinuity_restart = True
    o.enable_substep_state_correction = split
    o.split_snap = 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.run_transient(cache, b.graph, b.pool, o, sw)
    t = np.asarray(res.times)
    i_sw = np.asarray(res.states)[:, b.node_id_of("sw")] * GON
    ph = (t / TB) % 1.0
    on = (ph > 0.52) & (ph < 0.98) & (t > 10 * TB)   # well inside the ON phase
    return np.abs(i_sw[on] - I_ON).max(), res


# The switch current is G_on * v_sw with v_sw ~ 10 mV computed on the 100 V
# source's scale: its round-off is eps * V * G_on = 2.2e-11 A; the bound
# allows 100x that (the rule alone, split off, gives 1.8e-15 A).
BOUND = 100 * np.finfo(float).eps * V * GON


@pytest.mark.parametrize("side", [1, -1], ids=["before", "after"])
@pytest.mark.parametrize("frac", [1e-7, 1e-5, 1e-3, 0.1, 0.5, 0.9])
def test_no_ringing_after_a_split_sliver(frac, side):
    dev, res = _cap_cell(side * frac * DT / TB)
    assert dev < BOUND, f"max deviation {dev:.3g} A after the edge"
    if frac >= 1e-5:   # 1e-7 dt is inside split_snap (1e-6): taken at the grid point
        assert res.n_split_steps > 0


def test_rule_alone_is_the_reference_class():
    dev, res = _cap_cell(1e-5 * DT / TB, split=False)
    assert dev < BOUND and res.n_split_steps == 0
