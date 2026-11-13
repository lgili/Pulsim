"""add-exact-propagator-strategy D14 / task 16: sources through augmented states. The sine class
(one oscillator pair per frequency) is exact on dsed today; pinned here against the closed form.
PWM / pulse sources are not (DECISIONS: group 16 slice)."""

import warnings

import numpy as np

import pulsim as p

R, L, C, A, F = 1.0, 1e-3, 10e-6, 100.0, 50.0


def _closed_form(t):
    """Series R-L-C from rest driven by A sin(wt): i_L(t), v_C(t) (x' = M x + B A sin wt)."""
    w = 2 * np.pi * F
    Z = R + 1j * w * L + 1 / (1j * w * C)
    Ip = A / Z                                  # phasor of the forced current (sin reference)
    Vp = Ip / (1j * w * C)
    i_p = np.imag(Ip * np.exp(1j * w * t))
    v_p = np.imag(Vp * np.exp(1j * w * t))
    M = np.array([[-R / L, -1 / L], [1 / C, 0.0]])            # x = (i_L, v_C)
    lam, V = np.linalg.eig(M)
    c = np.linalg.solve(V, -np.array([i_p[0], v_p[0]], dtype=complex))
    xh = (V[:, :, None] * (c[:, None] * np.exp(lam[:, None] * t[None, :]))[None, :, :]).sum(axis=1).real
    return i_p + xh[0], v_p + xh[1]


def test_sine_driven_rlc_is_exact():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, A, F)
    b.add_resistor("R", "a", "b", R)
    b.add_inductor("L", "b", "c", L)
    b.add_capacitor("C", "c", "gnd", C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=0.06, engine="exact", integrator="rk45")
    t = np.asarray(r.times)
    i_ref, v_ref = _closed_form(t)
    v = np.asarray(r.v("c"))
    i = np.asarray(r.i("L"))
    # Measured: 1.0e-14 (v_C) and 2.7e-13 (i_L) relative; the gate is 1e-10.
    assert np.abs(v - v_ref).max() <= 1e-10 * np.abs(v_ref).max()
    assert np.abs(i - i_ref).max() <= 1e-10 * np.abs(i_ref).max()
    assert r.n_exact_steps == len(t) - 1                 # every step on the closed form
    assert [m["tier"] for m in r.exact["masks"]] == [1]  # eigen tier, oscillator augmented


def test_sine_into_rl_at_twenty_steps_per_period():
    # add-exact-propagator-strategy task 16.2: 100 V 50 Hz into 1 ohm + 1 mH at h = T/20, 5 periods,
    # within 1e-10 A of the closed form (measured 3.2e-13 A; prototype 1.6e-12).
    w, Rl, Ll = 2 * np.pi * 50.0, 1.0, 1e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_resistor("R", "a", "b", Rl)
    b.add_inductor("L", "b", "gnd", Ll)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=0.1, engine="exact", integrator="rk45", dt=1e-2)
    t = np.asarray(r.times)
    assert np.allclose(np.diff(t), 1e-3, rtol=0, atol=1e-15)   # h = T/20
    Z = Rl + 1j * w * Ll
    ph = np.angle(Z)
    ref = 100.0 / abs(Z) * (np.sin(w * t - ph) + np.sin(ph) * np.exp(-Rl / Ll * t))
    assert np.abs(np.asarray(r.i("L")) - ref).max() <= 1e-10
    assert r.n_exact_steps == 100
