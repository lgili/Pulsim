"""``run_ac_sweep`` reports a point only once its window has settled.

Settling counted in test cycles is short at high frequency. An LC
low-pass (100 µH, 47 µF, 5 Ω; τ = 2RC = 0.47 ms) started from zero with
24 V applied, swept at 11 kHz with 8 settling cycles (0.73 ms), read
−6.09 dB against the analytic −26.65 dB: the start-up ring (volts) was
still in the window of a millivolt response. The sweep now continues
the same run window by window until two agree within ``settle_rtol``.
"""
from __future__ import annotations

import math

import pytest

import pulsim as p

L, C, R = 100e-6, 47e-6, 5.0


def _sweep(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_inductor("L1", "in", "out", L)
    b.add_capacitor("C1", "out", "gnd", C)
    b.add_resistor("RL", "out", "gnd", R)
    n = b.pool.state_size(b.graph)
    k = b.pool.branch_var_id_for_source(b.branch_id_of("Vin"), b.graph)

    def excite(t, eps, f):
        out = [0.0] * n
        out[k] = -eps * math.sin(2 * math.pi * f * t)
        return out

    return p.run_ac_sweep(b, freqs=[11e3], excite_fn=excite, output_idx=b.node_id_of("out"),
                          perturbation_amplitude=0.1, settling_cycles=8, cycles_per_point=20,
                          start_from_dc_op=False, verbose=False, **kw)


def test_cold_start_ring_is_waited_out():
    w = 2 * math.pi * 11e3
    exact = 20 * math.log10(abs(1 / (1 - w * w * L * C + 1j * w * L / R)))
    assert exact == pytest.approx(-26.647, abs=1e-3)
    assert _sweep().mag_dB[0] == pytest.approx(exact, abs=0.1)


def test_unsettled_point_warns():
    with pytest.warns(RuntimeWarning, match="did not settle"):
        _sweep(max_extra_windows=1, settle_rtol=1e-9)
