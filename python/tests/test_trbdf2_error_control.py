"""TR-BDF2 error control (update-trbdf2-error-control).

Three parts, each pinned by a test that fails without it:
source breakpoints (a pulse jump is landed and the step does not
collapse at it, core review p13b.py), floors that scale with the step
ceiling instead of t_end - t_start (p17_span.py), and the factor 2 in
the local error estimate (p10_lte.py: true / estimate was 1.97).
Counters, not wall time.
"""

import numpy as np
import pytest

import pulsim as p

R, C = 1e3, 1e-9
T_ON, T_W = 3.3e-6, 50e-6


def _pulse_rc():
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, T_ON, T_W,
                               0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", R)
    b.add_capacitor("C", "a", "gnd", C)
    return b


def _pulse_rc_exact(t):
    tau = R * C
    v = np.zeros_like(t)
    m = t > T_ON
    v[m] = 10 * (1 - np.exp(-(t[m] - T_ON) / tau))
    m2 = t > T_ON + T_W
    v[m2] = 10 * (1 - np.exp(-T_W / tau)) * np.exp(-(t[m2] - T_ON - T_W) / tau)
    return v


def test_pulse_jumps_are_landed_and_the_step_does_not_collapse():
    """Before: 32 rejects, 8 accepted steps below 1e-10 s, 1.4e-14 s
    right after the 3.3 us jump. Now each jump is a sample time and no
    accepted step is shorter than 1e-10 s, except the committed restart
    probe right after each jump (h_probe, one per jump)."""
    res = p.simulate(_pulse_rc(), t_end=100e-6, engine="trbdf2")
    t = np.asarray(res.times)
    st = res._trbdf2_stats
    for te in (T_ON, T_ON + T_W):
        assert np.abs(t - te).min() < 1e-15
    assert st["n_source_breakpoints"] == 2
    h = np.diff(t)
    for te in (T_ON, T_ON + T_W):     # 1.2e-10 and 6.3e-11 s before
        before = (t[1:] <= te + 1e-15) & (t[1:] > te - 1e-6)
        assert h[before].min() > 1e-9
    assert int((h < 1e-10).sum()) <= 2       # the two restart probes
    assert st["n_reject"] <= 8               # 32 before


def test_pulse_rc_error_at_rtol_1e_6():
    """Max error against the exact response at rtol 1e-6: not above
    1.9e-4 V (1.824e-4 before; the jump error, not the tolerance)."""
    res = p.simulate(_pulse_rc(), t_end=100e-6, engine="trbdf2",
                     rtol=1e-6, atol=1e-9)
    t = np.asarray(res.times)
    v = np.asarray(res.v("a"))
    assert np.abs(v - _pulse_rc_exact(t)).max() < 1.9e-4


def test_ramp_corners_are_sample_times():
    """A pulse with 100 ns rise and fall: all four corners are landed
    (before, only the corners a step happened to hit were samples)."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, T_ON, T_W,
                               0.0, 1e-7, 1e-7)
    b.add_resistor("R", "in", "a", R)
    b.add_capacitor("C", "a", "gnd", C)
    res = p.simulate(b, t_end=100e-6, engine="trbdf2")
    t = np.asarray(res.times)
    for tc in (T_ON, T_ON + 1e-7, T_ON + 1e-7 + T_W, T_ON + 2e-7 + T_W):
        assert np.abs(t - tc).min() < 1e-15



def _pulse_i_rc():
    """The pulse RC driven by a 10 mA pulse CURRENT source into R || C:
    the same 10 V response as `_pulse_rc`."""
    b = p.CircuitBuilder()
    b.add_pulse_current_source("I", "a", "gnd", 0.0, 10e-3, T_ON, T_W,
                               0.0, 0.0, 0.0)
    b.add_resistor("R", "a", "gnd", R)
    b.add_capacitor("C", "a", "gnd", C)
    return b


def test_current_source_corners_are_landed_like_the_voltage_sources():
    """Integration seam: the waveform current sources' corners go through
    the same source-breakpoint landing (left limit, restart) as the
    pulse voltage source's: both jumps are samples and counted, and the
    error at rtol 1e-6 is the voltage-source bound."""
    res = p.simulate(_pulse_i_rc(), t_end=100e-6, engine="trbdf2",
                     rtol=1e-6, atol=1e-9)
    t = np.asarray(res.times)
    for te in (T_ON, T_ON + T_W):
        assert np.abs(t - te).min() < 1e-15
    assert res._trbdf2_stats["n_source_breakpoints"] == 2
    assert np.abs(np.asarray(res.v("a")) - _pulse_rc_exact(t)).max() < 1.9e-4
    # table knots too
    b = p.CircuitBuilder()
    knots = [0.0, 10e-6, 10.1e-6, 40e-6, 40.1e-6, 120e-6]   # last knot after t_end
    b.add_pwl_current_source("I", "a", "gnd", knots, [0, 0, 1e-2, 1e-2, 0, 0])
    b.add_resistor("R", "a", "gnd", R)
    b.add_capacitor("C", "a", "gnd", C)
    r2 = p.simulate(b, t_end=100e-6, engine="trbdf2")
    t2 = np.asarray(r2.times)
    for tk in knots[1:-1]:
        assert np.abs(t2 - tk).min() < 1e-15
    assert r2._trbdf2_stats["n_source_breakpoints"] == 4

VIN, L, CO, T = 24.0, 100e-6, 47e-6, 1e-5


def _buck(t_end, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", CO)
    b.add_resistor("RL", "vout", "gnd", 1.0)
    n = b.graph.num_switches
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        fr = t / T - np.floor(t / T + 1e-9)
        m.set(k, bool(fr < 0.5 - 1e-9))
        return m
    return p.simulate(b, t_end=t_end, engine="trbdf2", switch_fn=fn, **kw)


def test_floors_do_not_depend_on_t_end():
    """With the ceiling fixed (dt = T/20), a run to 40 T and one to
    400 T take the same steps up to 40 T. Before, h_min, the probe
    floor and the chatter floor were fractions of t_end - t_start, so
    the post-edge probe (and every step after it) moved with t_end."""
    a = _buck(40 * T, dt=T / 20)
    b = _buck(400 * T, dt=T / 20)
    assert a._trbdf2_stats["h_max_basis"] == 2
    ta, tb = np.asarray(a.times), np.asarray(b.times)
    # Up to the last period: the 40 T run then lands its final edge on
    # t_end, which the 400 T run crosses (a different, legitimate step).
    n = int(np.searchsorted(ta, 39 * T))
    np.testing.assert_array_equal(ta[:n], tb[:n])          # diverged at 24
    np.testing.assert_array_equal(np.asarray(a.v("vout"))[:n],
                                  np.asarray(b.v("vout"))[:n])


def _rc_decay(rtol):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", R)
    b.add_capacitor("C", "a", "gnd", C)
    res = p.simulate(b, t_end=5e-6, engine="trbdf2", rtol=rtol, atol=1e-12,
                     dt=1e-6)
    t = np.asarray(res.times)
    err = np.abs(np.asarray(res.v("a")) - 10 * (1 - np.exp(-t / (R * C))))
    return res._trbdf2_stats["n_accept"], err.max()


def test_estimate_is_the_true_local_error():
    """A smooth RC step response. The estimate was half the true local
    error, so a third-order controller took 2**(-1/3) of the steps it
    needed: rtol meant twice itself. With the factor the step count
    grows (by at most 2**(1/3) = 1.26) and the error shrinks.
    Measured at rtol 1e-5: 58 steps and 8.17e-4 V before, 66 steps
    and 5.22e-4 V after (+-5 % on the count)."""
    n5, e5 = _rc_decay(1e-5)
    assert n5 == pytest.approx(66, rel=0.05)
    assert e5 < 6e-4


def test_doubled_tolerances_restore_the_previous_density():
    """Span-bound ceiling (no periodic source below span/1000): the
    floors must equal the old span fractions exactly, so rtol and atol
    doubled undo the factor 2. Before this change the bridge took 1242
    steps, 123 rejects and 22 diode events at 1e-5 / 1e-8. A floor that
    differed only by roundoff ground this bridge to max_steps."""
    gon, goff = 1e3, 1e-9
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "s", "a", 0.5)
    b.add_inductor("Ls", "a", "ac1", 1e-3)
    b.add_diode("D1", "ac1", "dp", gon, goff)
    b.add_diode("D2", "gnd", "dp", gon, goff)
    b.add_diode("D3", "dn", "ac1", gon, goff)
    b.add_diode("D4", "dn", "gnd", gon, goff)
    b.add_capacitor("C", "dp", "dn", 470e-6)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    res = p.simulate(b, t_end=40e-3, engine="trbdf2", rtol=2e-5, atol=2e-8)
    st = res._trbdf2_stats
    assert st["h_max_basis"] == 0
    assert st["n_accept"] == pytest.approx(1242, rel=0.02)
    assert st["n_diode_events"] <= 30
