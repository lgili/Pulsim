"""add-complementarity-mode-selection DECISIONS #31: the selector's probes (the judge and
`first_departure_`) stop at the next time event (a gate edge the switch function announces,
or a source corner), and a switch function with no edge oracle adds no cap (its
next_edge_after is a dt_max/10 poll boundary, not an edge).

The poll case is the bridge bank of test_mode_lcp
(`test_a_widened_pass_does_not_release_a_diode_before_its_own_turn_off`): capping its probes at
the poll boundary brought back the 0.128 V early release of #24. Here: the decisions on a
gated circuit do not depend on rounding. This is the rounding fuzz of the new decision; it
does not show the cap changing a decision (no surveyed circuit does: the survey stayed
bit-identical).
"""
import warnings

import numpy as np
import pytest

import pulsim as p


def _dcm_buck(k):
    """A DCM buck at 100 kHz, D = 0.3: the freewheel diode turns off inside every period, a
    few microseconds before the next gate edge; every component perturbed at 1e-15 (k > 0)."""
    rng = np.random.default_rng(k)

    def f(v):
        return v * (1.0 + rng.uniform(-1.0, 1.0) * 1e-15) if k else v
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", f(24.0))
    b.add_switch("HS", "in", "sw", 1e3, 1e-9)
    b.add_diode("DFW", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", f(5e-6))
    b.add_capacitor("C", "out", "gnd", f(20e-6))
    b.add_resistor("R", "out", "gnd", f(20.0))
    return b


def _modes(r):
    s = np.asarray(r.switch_states)
    ch = np.flatnonzero(np.any(s[1:] != s[:-1], axis=1)) + 1
    return tuple(map(tuple, s[ch].astype(int)))


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_gated_decisions_survive_last_bit_perturbations(mode):
    kw = {} if mode is None else {"mode_selection": mode}
    runs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(21):
            b = _dcm_buck(k)
            pwm = p.NativePwm2Switch(10e-6, 0.3, b.graph.num_switches, True)
            runs.append(p.simulate(b, t_end=1e-3, engine="dsed", switch_fn=pwm, **kw))
    assert len({_modes(r) for r in runs}) == 1
    v = [float(np.asarray(r.v("out"))[-1]) for r in runs]
    assert np.ptp(v) < 1e-9 * abs(v[0])      # measured 1.5e-13 V on 17.1 V
