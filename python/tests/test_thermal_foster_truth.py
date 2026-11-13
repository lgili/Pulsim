"""Foster networks are Foster, and T_j(t) is sampled at t.

Found auditing the examples (run_thermal_buck.py, notebooks 03/11/28):

* ``add_foster_network`` (and ``add_shared_heatsink`` given
  ``FosterStage`` values) put every stage capacitor to AMBIENT, which is
  a Cauer ladder: 10 W into (0.05 K/W, 0.1 ms), (0.3, 5 ms), (0.8, 0.5 s)
  read 0.016 K at 1 ms against the Foster 1.06 K.
* ``compute_temperature`` heated the first sample with P[0], so the
  trace led t by one step: 1.90 K at t = 0 for 20 W into 1 K/W, 1 s,
  dt = 0.1 s.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

STAGES = [p.FosterStage(R_th_K_per_W=0.05, tau_s=1e-4),
          p.FosterStage(R_th_K_per_W=0.30, tau_s=5e-3),
          p.FosterStage(R_th_K_per_W=0.80, tau_s=0.5)]


def _foster_rise(t, P):
    return sum(P * s.R_th_K_per_W * (1 - np.exp(-t / s.tau_s)) for s in STAGES)


def _step_response(build):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    tj = build(b)
    obs, bex = p.make_thermal_observer(b, junction_node=tj,
                                       power_fn=lambda t, x: 10.0,
                                       ambient_node="T_amb", T_amb_C=25.0)
    r = p.simulate(b, t_end=0.2, dt=1e-5, step_observer=obs, b_extra_fn=bex)
    t = np.asarray(r.times)
    return t, r.v(tj) - 25.0


@pytest.mark.parametrize("t_probe", [1e-3, 1e-2, 0.2])
def test_add_foster_network_is_a_foster_network(t_probe):
    t, rise = _step_response(lambda b: (p.add_foster_network(
        b, STAGES, junction_node="T_j", ambient_node="T_amb",
        T_amb_C=25.0), "T_j")[1])
    k = int(np.argmin(abs(t - t_probe)))
    assert rise[k] == pytest.approx(_foster_rise(t[k], 10.0), rel=5e-3)


def test_shared_heatsink_with_foster_stages_is_foster():
    def build(b):
        hs = p.add_shared_heatsink(
            b, [p.HeatsinkDevice("Q", junction_to_case=STAGES)],
            R_th_sink_to_amb_K_per_W=1e-6, ambient_node="T_amb",
            T_amb_C=25.0)
        return hs.junction_nodes["Q"]
    t, rise = _step_response(build)
    k = int(np.argmin(abs(t - 1e-3)))
    assert rise[k] == pytest.approx(_foster_rise(t[k], 10.0), rel=5e-3)


def test_compute_temperature_is_sampled_at_t():
    t = np.arange(0.0, 1.0001, 0.1)
    T = p.compute_temperature(t, np.full_like(t, 20.0),
                              [p.FosterStage(1.0, 1.0)], T_amb_C=0.0)
    np.testing.assert_allclose(T, 20 * (1 - np.exp(-t)), atol=1e-12)
