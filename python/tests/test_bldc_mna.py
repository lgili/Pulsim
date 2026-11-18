"""Non-sinusoidal PM flux on the MNA-native PMSM; the BLDC built on it
(add-mna-native-machines, group 3).

References are independent of the engine: the ideal trapezoid (closed
form), the PMSM without a harmonic list (bit identity), the power balance
of a generator into a resistive load at held speed, and a synthetic
back-EMF whose series is known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p
from pulsim.machines import pm_harmonics_from_waveform, trapezoid_harmonics

RS, LS, KE, PP = 0.3, 1e-3, 0.05, 4
W_M = 100.0                       # held mechanical speed [rad/s]
T_E = 2 * math.pi / (PP * W_M)    # electrical period [s]


def _bldc(b, *, load=None, n_harmonics=25):
    b.add_bldc_mna("M", "ua", "ub", "uc", "nn", "w", "th", R_s=RS, L_s=LS,
                   k_e=KE, pole_pairs=PP, J=1e-4, omega0=W_M,
                   n_harmonics=n_harmonics)
    b.add_voltage_source("Vw", "w", "gnd", W_M)
    for x in ("ua", "ub", "uc"):
        b.add_resistor(f"R{x}", x, "gnd", 1e6 if load is None else load)
    return b


def _ideal_trapezoid(theta_e):
    gamma = math.pi / 6
    x = np.mod(theta_e, 2 * math.pi)
    up = np.clip(np.minimum(x, math.pi - x) / gamma, -1.0, 1.0)
    dn = -np.clip(np.minimum(x - math.pi, 2 * math.pi - x) / gamma, -1.0, 1.0)
    return np.where(x < math.pi, up, dn)


def test_series_truncation_matches_the_design_figure():
    th = np.linspace(0, 2 * math.pi, 20000, endpoint=False)
    harm = trapezoid_harmonics(KE, PP)
    dl = sum(-h * psi * np.sin(h * th + phi) for h, psi, phi in harm)
    e = -PP * dl / KE                 # back-EMF per unit k_e·ω_m, sign flipped
    rms = math.sqrt(np.mean((e - _ideal_trapezoid(th)) ** 2))
    assert rms == pytest.approx(0.0040, abs=2e-4)     # design: 0.40 %


def test_open_circuit_back_emf_is_the_trapezoid():
    res = p.simulate(_bldc(p.CircuitBuilder()), t_end=3 * T_E, dt=T_E / 2000,
                     engine="pwl")
    t = np.asarray(res.times)
    sel = t >= 2 * T_E                # one electrical period, steady
    e = (np.asarray(res.v("ua")) - np.asarray(res.v("nn")))[sel]
    th_e = PP * np.asarray(res.v("th"))[sel]
    ideal = -KE * W_M * _ideal_trapezoid(th_e)
    rms = math.sqrt(p.time_average(t[sel], (e - ideal) ** 2))
    assert rms < 5e-3 * KE * W_M, rms / (KE * W_M)


def test_fundamental_only_is_bit_identical_to_the_plain_pmsm():
    def run(harm):
        b = p.CircuitBuilder()
        for k, n in enumerate(("ua", "ub", "uc")):
            b.add_sine_voltage_source(f"V{n}", n, "gnd", 0.0, 12.0, 30.0,
                                      -2 * math.pi / 3 * k)
        kw = {"harmonics": harm} if harm else {}
        b.add_pmsm_mna("M", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5,
                       L_d=1e-3, L_q=3e-3, psi_pm=0.05, pole_pairs=4,
                       J=1e-3, B=1e-4, **kw)
        r = p.simulate(b, t_end=0.02, dt=5e-5, engine="pwl")
        return np.asarray(r.states)
    assert np.array_equal(run(None), run([(1.0, 0.05, 0.0)]))


def test_generator_power_balance_at_held_speed():
    # Mechanical power in = copper loss in R_s and the load (steady state,
    # whole electrical periods, so the stored energy returns).
    R_L = 2.0
    res = p.simulate(_bldc(p.CircuitBuilder(), load=R_L), t_end=12 * T_E,
                     dt=T_E / 4000, engine="pwl")
    t = np.asarray(res.times)
    t0, t1 = 2 * T_E, 12 * T_E
    p_mech = W_M * p.time_average(t, -np.asarray(res.i("Vw")), t0, t1)
    i2 = sum(np.asarray(res.i(f"M_{x}")) ** 2 for x in "abc")
    p_cu = (RS + R_L) * p.time_average(t, i2, t0, t1)
    assert p_mech > 1.0
    assert p_mech == pytest.approx(p_cu, rel=1e-6)      # measured 5.3e-8
    # T_em trace from the same series: its mean is the shaft torque.
    # The machine brakes: T_em is negative.
    assert -p.time_average(t, res.signal("M.T_em"), t0, t1) == pytest.approx(
        p_mech / W_M, rel=1e-9)


def test_harmonics_from_a_waveform_recover_a_known_series():
    known = [(1.0, 0.02, 0.3), (3.0, 0.004, -1.1), (5.0, 0.001, 2.0)]
    th = np.linspace(0, 2 * math.pi, 256, endpoint=False) + 0.1
    w_e = 400.0
    emf = w_e * sum(-h * psi * np.sin(h * th + phi) for h, psi, phi in known)
    got = {int(h): (psi, phi) for h, psi, phi in
           pm_harmonics_from_waveform(th, emf, w_e, n_harmonics=7)}
    for h, psi, phi in known:
        assert got[int(h)][0] == pytest.approx(psi, rel=1e-12)
        d = (got[int(h)][1] - phi + math.pi) % (2 * math.pi) - math.pi
        assert abs(d) < 1e-12
    assert all(got[h][0] < 1e-15 for h in (2, 4, 6, 7) if h in got)


def test_invalid_harmonic_is_refused_by_name():
    with pytest.raises(ValueError, match="M9.*harmonic"):
        p.CircuitBuilder().add_pmsm_mna(
            "M9", "a", "b", "c", "n", "w", "th", R_s=0.5, L_d=1e-3,
            L_q=1e-3, psi_pm=0.05, pole_pairs=4, J=1e-3,
            harmonics=[(1.5, 0.05, 0.0)])
