"""DECISIONS #30 (add-complementarity-mode-selection): mode decisions on a floating DC bus do not
depend on rounding.

A bridge with every diode OFF leaves its DC bus (C || R) tied to the rest only through the OFF
diodes' g_off and, here, an optional large tie. The port problem's factor rounded the island's
common mode to eps * (internal current) / G_tie, about the tie tolerance, and pivoting the
previous mode's port matrix onto the all-OFF candidate read the pair's split through a block of
condition ~R_island / R_on: the selected mode changed with the platform (GitHub CI, Linux).

The platform-independent stand-in for "same answer on Linux": perturb the inputs in their last
bits and require the same decisions.
"""
import numpy as np
import pytest

import pulsim as p

pytest.importorskip("scipy")
from tests.oracles import lcp_pulsim_bridge as B  # noqa: E402

BRIDGES = ["bridge1_vf0", "bridge1_vf07", "bridge3_cfilter"]


@pytest.mark.parametrize("name", BRIDGES)
def test_selected_mode_survives_last_bit_perturbations_of_the_state(name):
    fam = B.FAMILIES[name]
    _, rec = B.prototype_selections(fam)
    P = B.PulsimPorts(fam)
    h = 1e-7 * B.T_REF[name]
    rng = np.random.default_rng(7)
    for t, y, sw, prev, _ in rec:
        base = tuple(P.select(t, y, sw, prev, h)["mode"])
        for _ in range(20):
            yp = y * (1.0 + rng.choice([-1.0, 0.0, 1.0], size=y.size) * np.finfo(float).eps)
            assert tuple(P.select(t, yp, sw, prev, h)["mode"]) == base, t


def _bridge(vf, tie, k):
    rng = np.random.default_rng(k)

    def f(v):
        return v * (1.0 + rng.uniform(-1.0, 1.0) * 1e-15) if k else v
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "s", "gnd", 0.0, f(325.0 if tie else 100.0), 50.0)
    if tie:
        b.add_resistor("Rs", "s", "a", f(0.5))
        b.add_inductor("Ls", "a", "ac", f(1e-3))
    else:
        b.add_resistor("Rs", "s", "ac", f(1e-3))
    for nm, an, ca in (("D1", "ac", "p"), ("D3", "gnd", "p"), ("D2", "n", "ac"), ("D4", "n", "gnd")):
        b.add_diode(nm, an, ca, 1e3, 1e-9, 0.0, vf)
    b.add_capacitor("C", "p", "n", f(470e-6 if tie else 1e-3))
    b.add_resistor("R", "p", "n", f(100.0))
    if tie:
        b.add_resistor("Rtie", "n", "gnd", f(1e6))
    return b


def _modes(r):
    s = np.asarray(r.switch_states)
    ch = np.flatnonzero(np.any(s[1:] != s[:-1], axis=1)) + 1
    return tuple(map(tuple, s[ch].astype(int)))


@pytest.mark.parametrize("vf,tie", [(0.0, False), (0.7, False), (0.0, True)])
def test_dsed_mode_sequence_survives_last_bit_perturbations(vf, tie):
    runs = [p.simulate(_bridge(vf, tie, k), 0.04, engine="dsed") for k in range(21)]
    seqs = {_modes(r) for r in runs}
    assert len(seqs) == 1, sorted(len(s) for s in seqs)
    v = [np.asarray(r.v("p"))[-1] - np.asarray(r.v("n"))[-1] for r in runs]
    assert np.ptp(v) < 1e-9 * abs(v[0])
    # and the run is right: an independent reference (pwl at a fine step)
    ref = p.simulate(_bridge(vf, tie, 0), 0.04, engine="pwl", dt=1e-7)
    vr = np.asarray(ref.v("p"))[-1] - np.asarray(ref.v("n"))[-1]
    assert v[0] == pytest.approx(vr, rel=1e-8)   # pwl dt 1e-7 vs 5e-8: 5e-11
