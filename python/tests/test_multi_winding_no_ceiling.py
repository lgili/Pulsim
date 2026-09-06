"""Multi-winding transformers without a winding ceiling — audit C.4.

`add_multi_winding_transformer` refused N > 6 with "N must be in
[2, 6]". The pool's coupling registry is an unbounded vector and the
builder expands any N into N·(N−1)/2 pair-wise couplings, so the
ceiling was an argument check with nothing behind it — and the
method was not bound in Python at all, despite the docs listing it.

What DOES need refusing is a set of couplings that no transformer can
have: the inductance matrix must be positive definite or the stored
energy ½·iᵀ·M·i goes negative and the circuit is an oscillator with no
physics behind it. Pair-wise k's let a user write that down.

The saturable transformer gains the same freedom: any number of
secondaries, each an ideal transformer on one gapped-core magnetising
branch.
"""

import numpy as np
import pytest

import pulsim as p


def test_eight_windings_are_accepted_and_ratio_correctly():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "src", "gnd", v_dc=0.0, v_amplitude=10.0,
                              frequency=10e3, phase=0.0)
    b.add_resistor("Rs", "src", "w1", 0.01)
    windings = [("w1", "gnd", 1e-3)] + [(f"w{k}", "gnd", 1e-3 * k * k) for k in range(2, 9)]
    k = [[0.999] * 8 for _ in range(8)]
    b.add_multi_winding_transformer("T8", windings, k)
    for j in range(2, 9):
        b.add_resistor(f"R{j}", f"w{j}", "gnd", 1e4)
    res = p.simulate(b, t_end=200e-6, dt=1e-8)
    t = np.asarray(res.times)
    m = t > 100e-6
    v1 = float(np.abs(np.asarray(res.v("w1"))[m]).max())
    assert v1 > 5.0
    for j in range(2, 9):
        vj = float(np.abs(np.asarray(res.v(f"w{j}"))[m]).max())
        assert vj / v1 == pytest.approx(j, rel=2e-2), (j, vj / v1)
    assert b.branch_id_of("T8.w7") >= 0


def test_unrealisable_couplings_are_refused_by_name():
    b = p.CircuitBuilder()
    w = [("a", "gnd", 1e-3), ("b", "gnd", 1e-3), ("c", "gnd", 1e-3)]
    with pytest.raises(Exception, match="not realisable"):
        b.add_multi_winding_transformer("T", w, [[1, 1, 1], [1, 1, 0.5], [1, 0.5, 1]])
    with pytest.raises(Exception, match=r"outside \[0, 1\]"):
        b.add_multi_winding_transformer("T", w, [[1, 1.2, 1], [1.2, 1, 1], [1, 1, 1]])
    with pytest.raises(Exception, match="at least 2"):
        b.add_multi_winding_transformer("T", [("a", "gnd", 1e-3)])


CORE = dict(Ae=76e-6, le=72e-3, lg=0.5e-3, B_sat=0.35)


def _three_winding(v_amp, f=100e3):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "src", "gnd", v_dc=0.0, v_amplitude=v_amp,
                              frequency=f, phase=0.0)
    b.add_resistor("Rs", "src", "p", 0.2)
    b.add_saturable_transformer_n("T", "p", "gnd", N_p=25, L_leak_p=2e-6,
                                  secondaries=[("s1", "gnd", 13, 0.0), ("s2", "gnd", 5, 0.0)],
                                  **CORE)
    b.add_resistor("R1", "s1", "gnd", 20.0)
    b.add_resistor("R2", "s2", "gnd", 20.0)
    res = p.simulate(b, t_end=4.0 / f, dt=1e-8)
    t = np.asarray(res.times)
    m = t > 2.0 / f
    pk = lambda n: float(np.abs(np.asarray(res.v(n))[m]).max())  # noqa: E731
    i_m = float(np.abs(np.asarray(res.i("T.m"))[m]).max())
    return pk("p"), pk("s1"), pk("s2"), i_m, res


def test_three_winding_saturable_transformer_ratios_below_the_knee():
    vp, v1, v2, i_m, res = _three_winding(3.0)
    assert i_m < 1.0                              # far below the ~6 A knee
    assert v1 / vp == pytest.approx(13 / 25, rel=2e-2)
    assert v2 / vp == pytest.approx(5 / 25, rel=2e-2)
    # Both ideal secondaries carry current by name.
    assert np.abs(np.asarray(res.i("T.s0"))).max() > 0.01
    assert np.abs(np.asarray(res.i("T.s1"))).max() > 0.01


def test_three_winding_saturable_transformer_collapses_past_the_knee():
    # 10 kHz so the flux swing, λ_pk = V/ω, reaches the core's
    # λ(B_sat) = 0.665 mWb·t: at 3 V that is 4.8e-5 Wb (far below), at
    # 100 V it is 1.6 mWb (2.4× past). A linear transformer would give
    # 33× the secondary voltage; the core does not — the magnetising
    # current runs past the knee and the secondary voltage stops
    # following.
    _, v1_low, _, i_low, _ = _three_winding(3.0, f=10e3)
    vp_hi, v1_hi, _, i_hi, _ = _three_winding(100.0, f=10e3)
    assert i_low < 1.0, i_low
    assert i_hi > 6.0, i_hi
    assert v1_hi < 0.6 * (100.0 / 3.0 * v1_low), (v1_hi, 100 / 3 * v1_low)


def test_yaml_multi_winding_and_n_secondaries():
    y = """
circuit:
  devices:
    - {type: sine_voltage_source, name: V, from: src, to: gnd, v_dc: 0.0, v_amplitude: 3.0, frequency: 100000.0}
    - {type: resistor, name: Rs, from: src, to: p, R: 0.2}
    - type: saturable_transformer
      name: T
      p_from: p
      p_to: gnd
      N_p: 25
      L_leak_p: 2.0e-6
      Ae: 76.0e-6
      le: 72.0e-3
      lg: 0.5e-3
      B_sat: 0.35
      secondaries:
        - {s_from: s1, s_to: gnd, N_s: 13}
        - {s_from: s2, s_to: gnd, N_s: 5}
    - {type: resistor, name: R1, from: s1, to: gnd, R: 20.0}
    - {type: resistor, name: R2, from: s2, to: gnd, R: 20.0}
    - type: multi_winding_transformer
      name: T8
      windings:
        - {from: a1, to: gnd, L: 1.0e-3}
        - {from: a2, to: gnd, L: 4.0e-3}
        - {from: a3, to: gnd, L: 9.0e-3}
      k: [[1, 0.99, 0.99], [0.99, 1, 0.99], [0.99, 0.99, 1]]
    - {type: resistor, name: Ra1, from: p, to: a1, R: 1.0}
    - {type: resistor, name: Ra2, from: a2, to: gnd, R: 1000.0}
    - {type: resistor, name: Ra3, from: a3, to: gnd, R: 1000.0}
simulation: {t_start: 0.0, t_end: 4.0e-5, dt: 1.0e-8}
"""
    loaded = p.load_yaml_string(y)
    res = p.simulate(loaded.builder, t_end=40e-6, dt=1e-8)
    t = np.asarray(res.times)
    m = t > 20e-6
    vp = float(np.abs(np.asarray(res.v("p"))[m]).max())
    v1 = float(np.abs(np.asarray(res.v("s1"))[m]).max())
    v2 = float(np.abs(np.asarray(res.v("s2"))[m]).max())
    assert v1 / vp == pytest.approx(13 / 25, rel=3e-2)
    assert v2 / vp == pytest.approx(5 / 25, rel=3e-2)
    va2 = float(np.abs(np.asarray(res.v("a2"))[m]).max())
    va3 = float(np.abs(np.asarray(res.v("a3"))[m]).max())
    assert va3 / va2 == pytest.approx(1.5, rel=3e-2)
