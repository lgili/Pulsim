"""v2.0 Phase 3, item 3: stiff + sine no longer grinds.

Item 2's exact stepping covered DC-driven modes; the RK45 fallback
still owned anything with a time-varying source, and an ordinary
sine-driven rectifier with an RC snubber ground it down:

    tau = 1e-6 :   157k steps, 0.6 s
    tau = 1e-8 :   1.9M steps, 8.3 s
    tau = 1e-10:   refused by the progress guard

The audit prescribed a Radau stiff member here. The measurement
argued for something stronger first: a sine-driven LTI mode is
autonomous too, once the state is augmented with the source's own
oscillator pair — u = (sin ωt, cos ωt), u̇ = [[0,ω],[−ω,0]]·u,
amplitudes and phases folded into the coupling columns. The
augmented system steps EXACTLY, any h, any stiffness.

The arbitration that validated it: the exact path, the dsed RK45
path, and the pwl engine at dt = 1e-8 all agree on 5.9262 V — and
the pwl engine at dt = 1e-7 is the outlier at 5.74 (3.2% off from
its own commutation resolution). The event-driven answer is again
the sharper one.

What still takes the numeric path, on purpose: PWM/pulse sources
(b(t) with steps is not a finite oscillator sum) and user
b_extra_fn callbacks. A stiff circuit in THOSE classes remains
Radau's future subject — code waits for a repro, per this project's
standing rule.
"""

import time
import warnings

import numpy as np
import pytest

import pulsim as p


def rect_snub(Rs, Cs):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 60.0)
    b.add_diode("D", "ac", "vout", 1e3, 1e-9, 0.7)
    b.add_resistor("Rs", "ac", "sn", Rs)
    b.add_capacitor("Cs", "sn", "vout", Cs)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("Rl", "vout", "gnd", 200.0)
    return b


def _tavg(t, v):
    t = np.asarray(t)
    v = np.asarray(v)
    return float(np.trapezoid(v, t) / (t[-1] - t[0]))


@pytest.mark.parametrize("Rs,Cs", [
    (10.0, 100e-9),      # tau = 1e-6 — was 157k steps
    (1.0, 10e-9),        # tau = 1e-8 — was 1.9M steps, 8.3 s
    (0.1, 1e-9),         # tau = 1e-10 — was REFUSED
], ids=["tau-1e-6", "tau-1e-8", "tau-1e-10"])
def test_stiff_sine_rectifier_is_fast_and_matches_pwl(Rs, Cs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        rd = p.simulate(rect_snub(Rs, Cs), t_end=5e-2,
                         engine="dsed")
        elapsed = time.perf_counter() - t0
        rp = p.simulate(rect_snub(Rs, Cs), t_end=5e-2, dt=1e-8)

    assert elapsed < 5.0, elapsed
    # State order [v_Cs, v_C]: compare the output cap's time average
    # against the pwl engine at a dt fine enough to be converged
    # (dt = 1e-7 is 3.2% off — the fixture that exposed it).
    avg_d = _tavg(rd.times, np.asarray(rd.states)[:, 1])
    avg_p = _tavg(rp.times, rp.v("vout"))
    assert avg_d == pytest.approx(avg_p, rel=2e-3)
    assert rd.n_events >= 6           # it commutates, still


def test_the_diodeless_sine_path_is_exact_too():
    """A plain sine RC through the whole engine: the recorded
    trajectory must sit on the closed-form response at every
    recorded instant — not merely averaged."""
    R, C = 1e3, 1e-6
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "in", "gnd", 2.0, 5.0, 60.0,
                                0.7)
    b.add_resistor("R1", "in", "n1", R)
    b.add_capacitor("C1", "n1", "gnd", C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=2e-2, engine="dsed")

    w = 2.0 * np.pi * 60.0
    tau = R * C
    th = np.arctan(w * tau)
    Vp = 5.0 / np.sqrt(1.0 + (w * tau) ** 2)
    t = np.asarray(r.times)
    ss = 2.0 + Vp * np.sin(w * t + 0.7 - th)
    ss0 = 2.0 + Vp * np.sin(0.7 - th)
    ref = ss - ss0 * np.exp(-t / tau)
    v = np.asarray(r.states)[:, 0]
    np.testing.assert_allclose(v, ref, rtol=0, atol=1e-8)
