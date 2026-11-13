"""add-exact-propagator-strategy D14 / task 16: sources through augmented states. The sine class
(one oscillator pair per frequency) is exact on dsed; PWM / pulse sources are exact too since
the group-16 corner slice: level + ramp between corners (phi_1 / phi_2 terms), each corner a time
event the scheduler lands on. Pinned against closed forms and matrix-exponential references."""

import warnings

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------------------------
# Piecewise-linear sources (task 16.1): references by exact affine propagation in numpy
# ---------------------------------------------------------------------------------------------
def _expm(M):
    """exp(M) by scaling and squaring of a 30-term Taylor series (small dense M; no scipy)."""
    nrm = np.abs(M).sum(axis=0).max()
    k = max(0, int(np.ceil(np.log2(max(nrm, 1e-300) / 0.25))))
    X = M / 2.0**k
    E, T = np.eye(len(M)), np.eye(len(M))
    for j in range(1, 30):
        T = T @ X / j
        E = E + T
    for _ in range(k):
        E = E @ E
    return E


def _propagate(A, B, src, t_samples, x0):
    """x' = A x + B u(t), u = src(t) -> (value, slope) piecewise linear, propagated exactly from
    sample to sample (the samples include every corner, so u is affine on each interval)."""
    n = len(A)
    M = np.zeros((n + 2, n + 2))
    M[:n, :n] = A
    M[:n, n] = B
    M[n, n + 1] = 1.0
    out, x = [x0.copy()], x0.copy()
    for ta, tb in zip(t_samples[:-1], t_samples[1:]):
        v, sl = src(ta, tb)
        x = (_expm(M * (tb - ta)) @ np.r_[x, v, sl])[:n]
        out.append(x)
    return np.array(out)


def _pulse(v0, v1, t0, rise, width, fall, period):
    """The PulseVoltageSource law: (value, slope) of the piece holding (ta, tb)."""
    def src(ta, tb):
        tm = 0.5 * (ta + tb)
        if tm < t0:
            return v0, 0.0
        e = (tm - t0) % period if period > 0 else tm - t0
        base = ta - tm                     # value at ta = value(tm) + slope * base
        if e < rise:
            sl = (v1 - v0) / rise
            return v0 + sl * (e + base), sl
        if e < rise + width:
            return v1, 0.0
        if e < rise + width + fall:
            sl = (v0 - v1) / fall
            return v1 + sl * (e - rise - width + base), sl
        return v0, 0.0
    return src


def _run(b, t_end, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine="exact", **kw)


def _assert_corners_sampled(t, corners):
    t = np.asarray(t)
    for c in corners:
        assert np.abs(t - c).min() <= 1e-14 * c, c   # a sample on it (k*T + offset, to a few ulps)


@pytest.mark.parametrize("integrator", ["auto", "rk45"])
def test_ramp_into_rc_is_exact_with_a_sample_at_its_corner(integrator):
    """0 -> 10 V over 1 ms (a pulse source's rise) into 1 kohm / 1 uF. Before the corner slice it
    ran 0 exact steps, 8.97e-8 V off, with no sample at 1 ms; measured now 2.7e-13 V."""
    R_, C_, TR = 1e3, 1e-6, 1e-3
    tau = R_ * C_
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", 0.0, 10.0, 0.0, 1.0, 0.0, TR, 0.0)
    b.add_resistor("R", "a", "o", R_)
    b.add_capacitor("C", "o", "gnd", C_)
    r = _run(b, 5e-3, integrator=integrator)
    t = np.asarray(r.times)
    a = 10.0 / TR
    t1 = np.minimum(t, TR)
    ref = a * (t1 - tau * (1 - np.exp(-t1 / tau)))
    vT = a * (TR - tau * (1 - np.exp(-TR / tau)))
    late = t > TR
    ref[late] = 10.0 + (vT - 10.0) * np.exp(-(t[late] - TR) / tau)
    assert np.abs(np.asarray(r.v("o")) - ref).max() <= 1e-10 * 10.0
    assert np.any(t == TR)                                   # the corner is a sample, exactly
    assert r.n_exact_steps == len(t) - 1
    assert all(s["exact"] for s in r.exact["segments"]) if integrator == "rk45" else True


def test_pulse_train_into_rlc_matches_the_matrix_exponential():
    """A trapezoidal pulse train (rise 2 us, width 20 us, fall 3 us, period 50 us, from 10 us)
    into a series 2 ohm / 100 uH / 10 uF: every corner a sample, every step exact, the states
    within 1e-10 relative of the exact affine propagation."""
    Rs, Ls, Cs = 2.0, 100e-6, 10e-6
    pls = (0.0, 10.0, 10e-6, 2e-6, 20e-6, 3e-6, 50e-6)
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", pls[0], pls[1], pls[2], pls[4], pls[6], pls[3], pls[5])
    b.add_resistor("R", "a", "m", Rs)
    b.add_inductor("L", "m", "c", Ls)
    b.add_capacitor("C", "c", "gnd", Cs)
    t_end = 400e-6
    r = _run(b, t_end, integrator="rk45")
    t = np.asarray(r.times)
    A = np.array([[-Rs / Ls, -1 / Ls], [1 / Cs, 0.0]])        # (i_L, v_C)
    ref = _propagate(A, np.array([1 / Ls, 0.0]), _pulse(*pls), t, np.zeros(2))
    i, v = np.asarray(r.i("L")), np.asarray(r.v("c"))
    assert np.abs(i - ref[:, 0]).max() <= 1e-10 * np.abs(ref[:, 0]).max()
    assert np.abs(v - ref[:, 1]).max() <= 1e-10 * np.abs(ref[:, 1]).max()
    corners = [pls[2] + k * pls[6] + e for k in range(8)
               for e in (0.0, pls[3], pls[3] + pls[4], pls[3] + pls[4] + pls[5]) if pls[2] + k * pls[6] + e < t_end]
    _assert_corners_sampled(t, corners)
    assert r.n_exact_steps == len(t) - 1
    assert all(s["exact"] for s in r.exact["segments"])


def test_pwm_source_buck_matches_the_matrix_exponential():
    """A PWM voltage source (24 V / 0 V, 100 kHz, D = 0.4) into 100 uH and 47 uF || 5 ohm: the
    switching node of a buck as a source. 20 periods within 1e-10 relative, a sample on every
    edge, every step exact."""
    Lb, Cb, Rb, f, D = 100e-6, 47e-6, 5.0, 1e5, 0.4
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("V", "sw", "gnd", 24.0, 0.0, f, D)
    b.add_inductor("L", "sw", "o", Lb)
    b.add_capacitor("C", "o", "gnd", Cb)
    b.add_resistor("R", "o", "gnd", Rb)
    t_end = 20 / f
    r = _run(b, t_end, integrator="rk45")
    t = np.asarray(r.times)
    A = np.array([[0.0, -1 / Lb], [1 / Cb, -1 / (Rb * Cb)]])

    def src(ta, tb):
        return (24.0 if ((0.5 * (ta + tb)) * f) % 1.0 < D else 0.0), 0.0
    ref = _propagate(A, np.array([1 / Lb, 0.0]), src, t, np.zeros(2))
    i, v = np.asarray(r.i("L")), np.asarray(r.v("o"))
    assert np.abs(i - ref[:, 0]).max() <= 1e-10 * np.abs(ref[:, 0]).max()
    assert np.abs(v - ref[:, 1]).max() <= 1e-10 * np.abs(ref[:, 1]).max()
    _assert_corners_sampled(t, [k / f + e for k in range(20) for e in (0.0, D / f)])
    assert r.n_exact_steps == len(t) - 1


def test_source_corner_coinciding_with_a_gate_edge_is_landed_once():
    """A 100 kHz PWM source (10 V / 0, D = 0.5) and a 50 kHz PWM switch whose edges fall on every
    other source corner: 10 ohm into 1 uF, the switch (1 mohm / 1 Gohm) to 10 ohm. One state,
    v_C' = ((v_s - v)/R1 - v/(R2 + R_sw))/C: every corner and edge is a sample, within 1e-10."""
    R1, R2, Cc, f = 10.0, 10.0, 1e-6, 1e5
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("V", "s", "gnd", 10.0, 0.0, f, 0.5)
    b.add_resistor("R1", "s", "a", R1)
    b.add_capacitor("C", "a", "gnd", Cc)
    b.add_switch("S", "a", "x", 1e3, 1e-9)
    b.add_resistor("R2", "x", "gnd", R2)
    fn = p.make_pwm_switch_fn(f / 2, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    t_end = 200e-6
    r = _run(b, t_end, integrator="rk45", switch_fn=fn)
    t = np.asarray(r.times)
    v = np.asarray(r.v("a"))
    x = [0.0]
    for ta, tb in zip(t[:-1], t[1:]):
        tm = 0.5 * (ta + tb)
        vs = 10.0 if (tm * f) % 1.0 < 0.5 else 0.0
        rsw = 1e-3 if (tm * f / 2) % 1.0 < 0.5 else 1e9
        a = -(1 / R1 + 1 / (R2 + rsw)) / Cc
        c = vs / (R1 * Cc)
        x.append(np.exp(a * (tb - ta)) * x[-1] + c * np.expm1(a * (tb - ta)) / a)
    assert np.abs(v - np.array(x)).max() <= 1e-10 * 10.0
    _assert_corners_sampled(t, [k * 0.5 / f for k in range(int(t_end * 2 * f))])
    assert len(np.unique(t)) == len(t)                       # no zero-length double landing


def test_pulse_with_instant_edges_into_rc_is_exact_at_every_sample():
    """Task 8.1 / 8.3 (audit item 11: trbdf2 lands this jump by rejection and floors near
    1.8e-4 V on 10 V at rtol 1e-6): 0 / 10 V, zero rise and fall, from 3.3 us, width 20 us,
    period 50 us, into 1 kohm / 10 nF. Piecewise-constant b, every jump a sample, 1e-10."""
    R_, C_ = 1e3, 10e-9
    pls = (0.0, 10.0, 3.3e-6, 0.0, 20e-6, 0.0, 50e-6)
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", pls[0], pls[1], pls[2], pls[4], pls[6], pls[3], pls[5])
    b.add_resistor("R", "a", "o", R_)
    b.add_capacitor("C", "o", "gnd", C_)
    t_end = 300e-6
    r = _run(b, t_end)
    t = np.asarray(r.times)
    a = -1.0 / (R_ * C_)
    ref = _propagate(np.array([[a]]), np.array([-a]), _pulse(*pls), t, np.zeros(1))[:, 0]
    assert np.abs(np.asarray(r.v("o")) - ref).max() <= 1e-10 * 10.0
    _assert_corners_sampled(t, [pls[2] + k * pls[6] + e for k in range(6) for e in (0.0, pls[4])])
    assert r.n_exact_steps == len(t) - 1


@pytest.mark.parametrize("rise,width,fall", [(0.0, 3e-7, 0.0), (2e-7, 3e-7, 1e-7)])
def test_delayed_single_shot_pulse_reads_the_new_piece_at_its_corner(rise, width, fall):
    """Reviewer finding: a single-shot pulse from t0 = 1 ms whose corners t0 + e round below
    t0 + e - t0 (300 ns, 200 ns, 600 ns all do) held the old piece for the step after the corner:
    a 300 ns, 10 V pulse into 1 kohm / 1 nF peaked at 7.27 V instead of 2.59 V. Within 1e-10."""
    R_, C_ = 1e3, 1e-9
    pls = (0.0, 10.0, 1e-3, rise, width, fall, 0.0)
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", pls[0], pls[1], pls[2], pls[4], pls[6], pls[3], pls[5])
    b.add_resistor("R", "a", "o", R_)
    b.add_capacitor("C", "o", "gnd", C_)
    r = _run(b, 1.003e-3)
    t = np.asarray(r.times)
    a = -1.0 / (R_ * C_)
    ref = _propagate(np.array([[a]]), np.array([-a]), _pulse(*pls), t, np.zeros(1))[:, 0]
    assert np.abs(np.asarray(r.v("o")) - ref).max() <= 1e-10 * 10.0
    _assert_corners_sampled(t, [pls[2] + e for e in (0.0, rise, rise + width, rise + width + fall)])


def test_ramp_into_rl_matches_the_closed_form():
    """Task 8.3: 0 -> 10 V over 1 ms into 10 ohm + 10 mH (tau = 1 ms), then held:
    i = (a/R)(t - tau(1 - e^{-t/tau})) on the ramp, an exponential to 1 A after; 1e-10."""
    R_, L_, TR = 10.0, 10e-3, 1e-3
    tau, a = L_ / R_, 10.0 / TR
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", 0.0, 10.0, 0.0, 1.0, 0.0, TR, 0.0)
    b.add_resistor("R", "a", "m", R_)
    b.add_inductor("L", "m", "gnd", L_)
    r = _run(b, 5e-3)
    t = np.asarray(r.times)
    t1 = np.minimum(t, TR)
    ref = (a / R_) * (t1 - tau * (1 - np.exp(-t1 / tau)))
    iT = (a / R_) * (TR - tau * (1 - np.exp(-TR / tau)))
    late = t > TR
    ref[late] = 1.0 + (iT - 1.0) * np.exp(-(t[late] - TR) / tau)
    assert np.abs(np.asarray(r.i("L")) - ref).max() <= 1e-10 * 1.0
    assert np.any(t == TR)
    assert r.n_exact_steps == len(t) - 1
