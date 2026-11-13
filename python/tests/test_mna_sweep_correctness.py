"""The kernel AC sweep answers only for the circuit it models.

Before this change ``run_mna_sweep(freqs=, v_in_branch_id=)`` returned a
curve for circuits it did not model, with no error (|H| at 100 Hz /
1 kHz / 10 kHz):

* coupled transformer, k = 0.99: 0 / 0 / 0, exact 0.0617 / 0.3885 / 0.4949
  (the hand-written mass matrix had no mutual terms);
* RC through a switch meant closed: 0 / 0 / 0 (all-OFF mask forced);
* Shockley diode, saturable inductor, junction capacitor: 1, 0.5, 0.9999
  flat (Newton branches dropped by the static assembler);
* any kernel exception was swallowed by a broad ``except`` and the call
  fell through to the impulse-FFT path.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

F = np.array([100.0, 1e3, 1e4])
W = 2 * np.pi * F


def _sweep(b, out, src="V", **kw):
    return p.run_mna_sweep(b, output_idx=b.node_id_of(out), freqs=F,
                           v_in_branch_id=b.branch_id_of(src), **kw)


def _close(H, exact, tol=1e-9):
    np.testing.assert_allclose(np.abs(H), np.abs(exact), rtol=tol)
    np.testing.assert_allclose(np.angle(H), np.angle(exact), rtol=tol, atol=tol)


def test_coupled_transformer_uses_the_mutual_inductance():
    Rs, RL, Lp, Ls, k = 10.0, 10.0, 1e-3, 1e-3, 0.99
    M = k * np.sqrt(Lp * Ls)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("Rs", "in", "a", Rs)
    b.add_transformer("T", "a", "gnd", "s", "gnd", Lp, Ls, k)
    b.add_resistor("RL", "s", "gnd", RL)
    H = 1j * W * M * RL / ((Rs + 1j * W * Lp) * (RL + 1j * W * Ls) + W ** 2 * M ** 2)
    np.testing.assert_allclose(np.abs(H), [0.0617, 0.3885, 0.4949], atol=1e-4)
    _close(_sweep(b, "s").H, H)   # was 0 / 0 / 0


def _rc_switch():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("R", "in", "a", 100.0)
    b.add_switch("S", "a", "o", 1e3, 1e-9)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    b.add_resistor("RL", "o", "gnd", 1e6)
    return b


def test_switch_state_is_the_callers_mask():
    b = _rc_switch()
    m = p.SwitchStateMask(b.graph.num_switches)
    m.set(b.switch_index_of("S"), True)
    Z = 1.0 / (1.0 / 1e6 + 1j * W * 1e-6)
    H = Z / (100.0 + 1.0 / 1e3 + Z)
    np.testing.assert_allclose(np.abs(H), [0.998, 0.8467, 0.1572], atol=1e-3)
    _close(_sweep(b, "o", mask=m).H, H)
    off = p.SwitchStateMask(b.graph.num_switches)
    assert abs(_sweep(b, "o", mask=off).H[0]) < 1e-5


def test_switches_without_a_mask_are_refused_by_name():
    with pytest.raises(ValueError, match=r"mask.*\bS\b|\bS\b.*mask"):
        _sweep(_rc_switch(), "o")


def _shockley():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.7)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_shockley_diode("D", "a", "gnd")
    return b, "a", "D", "ShockleyDiode"


def _saturable():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_saturable_inductor("L", "a", "o", 1e-3, 1.0)
    b.add_resistor("RL", "o", "gnd", 10.0)
    return b, "o", "L", "SaturableInductor"


def _junction_cap():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("R", "in", "o", 100.0)
    b.add_nonlinear_capacitor("C", "o", "gnd", 1e-6, 1.0)
    b.add_resistor("RL", "o", "gnd", 1e6)
    return b, "o", "C", "NonlinearCapacitor"


@pytest.mark.parametrize("make", [_shockley, _saturable, _junction_cap])
def test_newton_devices_are_refused_with_name_and_kind(make):
    b, out, name, kind = make()
    with pytest.raises(ValueError) as exc:
        _sweep(b, out)
    msg = str(exc.value)
    assert f"'{name}'" in msg and kind in msg, msg


def test_a_kernel_failure_reaches_the_caller(monkeypatch):
    from pulsim import _pulsim as k

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(k, "run_mna_sweep_kernel", boom)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("R", "in", "o", 100.0)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    with pytest.raises(RuntimeError, match="boom"):
        _sweep(b, "o", dt=1e-6, t_end=1e-2)


def test_vcvs_gain_is_flat():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("Ra", "in", "gnd", 10.0)
    b.add_vcvs("E", "in", "gnd", "s", "gnd", 2.0)
    b.add_resistor("RL", "s", "gnd", 10.0)
    _close(_sweep(b, "s").H, np.full(3, 2.0 + 0j))


def test_ideal_transformer_ratio():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("Rs", "in", "a", 1.0)
    b.add_ideal_transformer("X", "a", "gnd", "s", "gnd", 0.5)
    b.add_resistor("RL", "s", "gnd", 10.0)
    H = _sweep(b, "s").H
    # v_s = n v_p; RL reflects as RL/n^2 = 40 ohm: v_s = 0.5 * 40/41.
    _close(H, np.full(3, 0.5 * 40.0 / 41.0 + 0j))
    # Same answer from the time domain (impulse-FFT), independent path.
    ref = _sweep(b, "s", use_kernel=False, dt=1e-6, t_end=0.05).H
    np.testing.assert_allclose(np.abs(H), np.abs(ref), rtol=1e-3)


def test_three_winding_transformer_matches_the_time_domain():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("Rs", "in", "a", 10.0)
    k = [[1.0, 0.95, 0.95], [0.95, 1.0, 0.95], [0.95, 0.95, 1.0]]
    b.add_multi_winding_transformer("X", [("a", "gnd", 1e-3), ("s", "gnd", 1e-3),
                                          ("t", "gnd", 1e-3)], k)
    b.add_resistor("RL", "s", "gnd", 10.0)
    b.add_resistor("RL2", "t", "gnd", 10.0)
    H = _sweep(b, "s").H
    ref = _sweep(b, "s", use_kernel=False, dt=1e-6, t_end=0.05).H
    np.testing.assert_allclose(np.abs(H), np.abs(ref), rtol=1e-3)


def test_linear_rlc_sweep_is_unchanged():
    # Closed form of the series-R, shunt-(L || C) || RL divider.
    R, L, C, RL = 50.0, 1e-3, 1e-6, 1e3
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("R", "in", "o", R)
    b.add_inductor("L", "o", "m", L)
    b.add_resistor("RLs", "m", "gnd", 1.0)
    b.add_capacitor("C", "o", "gnd", C)
    b.add_resistor("RL", "o", "gnd", RL)
    Zl = 1.0 + 1j * W * L
    Zsh = 1.0 / (1.0 / Zl + 1j * W * C + 1.0 / RL)
    _close(_sweep(b, "o").H, Zsh / (R + Zsh), tol=1e-12)
