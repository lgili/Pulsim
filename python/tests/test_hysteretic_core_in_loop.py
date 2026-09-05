"""Jiles-Atherton hysteresis INSIDE the Newton loop — audit C.4.

The observer this replaces (`make_hysteretic_inductor_observer`) put
an air-core L_0 in series with a dummy source and modulated the source
with N·A·μ_0·dM/dt computed from the PREVIOUS step's current. Two
things were wrong with it, both measured before this landed on a
linear-χ stand-in through the identical mechanism (so the JA model
itself was not in question):

    L_M      q      injected sign   phase(I₁/V)   |I₁|/(V/R)   analytic
    50 µH   0.40    kernel's +v_M      +0.670°       1.0032     −1.126°
    50 µH   0.40    motors'  −v_M      −1.119°       1.0016     −1.126°

Current LEADING voltage and |I₁| above V/R on a passive R–L branch:
the magnetisation acted as a negative inductance. And with the sign
corrected, the lagged coupling is an explicit treatment of a stiff
inductive term:

    q = L_M / (dt·(R + 2L_0/dt)):   0.2 → 0.9995 A    0.6 → 5.2e4 A
                                    0.4 → 0.9990 A    0.8 → 1.2e50 A
                                                      2.0 → NaN

Every shipped use sat at q of hundreds; only the ±M_s clamp bounded
it. These tests are the physics anchors the old Python-vs-C++
agreement test could not provide, because both paths were wrong the
same way.
"""

import math

import numpy as np
import pytest

import pulsim as p

MU0 = 4e-7 * math.pi


def _core(b, name="L_core", **kw):
    params = kw.pop("params", p.reference_material("ferrite_n87"))
    return p.add_hysteretic_inductor(b, name=name, from_node=kw.pop("from_node", "n1"),
                                     to_node=kw.pop("to_node", "gnd"), params=params,
                                     N_turns=kw.pop("N_turns", 100), l_m=kw.pop("l_m", 0.05),
                                     A_core=kw.pop("A_core", 1e-4), **kw)


# 1 V at 50 Hz on 100 turns × 1 cm² is B_pk = V/(ω N A) = 0.32 T — a
# loop that reaches 0.6 M_s without slamming into saturation. (At
# 50 V the core saturates within a fraction of the cycle and the
# current is just V/R, which hides everything this file tests.)
# The source starts at its PEAK (phase π/2): energising at the zero
# crossing puts a flux offset on the core (inrush — the first-cycle
# peak is 20× the steady one) that decays with L/R ≈ 0.8 s here, and
# no window inside a 60 ms run is then a steady cycle.
def _ac_run(dt, V=1.0, f=50.0, R=0.1, t_end=0.06, phase=math.pi / 2, **core_kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, V, f, phase)
    b.add_resistor("Rs", "a", "n1", R)
    h = _core(b, **core_kw)
    res = p.simulate(b, t_end=t_end, dt=dt)
    t = np.asarray(res.times)
    i = np.asarray(res.i("L_core"))
    v = np.asarray(res.v("a"))
    assert np.all(np.isfinite(i))
    return t, i, v, h, res


def _fundamental(t, x, f, t_from):
    m = t >= t_from
    w = 2 * math.pi * f
    c = np.trapezoid(x[m] * np.exp(-1j * w * t[m]), t[m])
    return c, 2 * abs(c) / (t[m][-1] - t[m][0])


def test_current_lags_voltage_and_stays_below_v_over_r():
    """A passive R–L branch: current lags, and |I₁| < V/R. The old
    path failed both."""
    t, i, v, _, _ = _ac_run(2e-5)
    ci, I1 = _fundamental(t, i, 50.0, 0.04)
    cv, _ = _fundamental(t, v, 50.0, 0.04)
    phase = math.degrees(np.angle(ci / cv))
    assert -90.0 < phase < -20.0, phase          # strongly inductive
    assert I1 < 1.0 / 0.1, I1


def test_stable_where_the_observer_diverged():
    """The test circuit's own q is in the hundreds. The observer
    returned kiloamps or NaN here; the in-loop device returns the
    same waveform at every dt."""
    peaks = []
    for dt in (1e-4, 2e-5, 5e-6):
        t, i, v, _, _ = _ac_run(dt)
        peaks.append(float(np.abs(i[t > 0.04]).max()))
    assert peaks[0] < 1.0 / 0.1                  # never R-limited
    # Converging, not wandering.
    assert abs(peaks[1] - peaks[2]) < abs(peaks[0] - peaks[2]) + 1e-9
    assert abs(peaks[1] - peaks[2]) < 2e-2 * peaks[2], peaks


def test_converges_with_dt_toward_a_fine_reference():
    ref_t, ref_i, _, _, _ = _ac_run(2e-6, t_end=0.04)
    errs = []
    for dt in (4e-5, 2e-5, 1e-5):
        t, i, _, _, _ = _ac_run(dt, t_end=0.04)
        ii = np.interp(ref_t, t, i)
        m = ref_t > 0.02
        errs.append(float(np.abs(ii[m] - ref_i[m]).max() / np.abs(ref_i[m]).max()))
    assert errs[2] < errs[0], errs
    assert errs[2] < 2e-2, errs


def test_loop_energy_equals_the_electrical_energy_absorbed():
    """∮H dB · Ae · le per cycle (the hysteresis loss the B–H replay
    reports) must equal ∮ v_L·i dt over the same steady cycle — the
    branch is lossless apart from the loop."""
    t, i, v_a, h, res = _ac_run(5e-6, t_end=0.08)
    v_L = np.asarray(res.v("n1"))            # across the device (to gnd)
    m = (t >= 0.04) & (t <= 0.06)            # one full 50 Hz cycle
    W_elec = float(np.trapezoid(v_L[m] * i[m], t[m]))
    # The replay reports the LAST cycle's ∮ H dB; compare that cycle.
    m = (t >= 0.06) & (t <= 0.08)
    W_elec = float(np.trapezoid(v_L[m] * i[m], t[m]))
    loop = h.bh_loop(res, period=0.02)
    W_loop = float(loop.energy_per_cycle_per_m3) * h.A_core * h.l_m
    assert W_elec > 0.0
    assert abs(W_elec - W_loop) < 0.05 * W_elec, (W_elec, W_loop)


def test_the_anhysteretic_limit_is_lossless():
    """c = 1 removes the irreversible part: the loop closes to a curve
    and the steady-cycle electrical energy vanishes."""
    params = p.JilesAthertonParams(Ms=4.0e5, a=50.0, alpha=5e-5, c=1.0, k=30.0)
    t, i, v_a, h, res = _ac_run(5e-6, t_end=0.08, params=params)
    v_L = np.asarray(res.v("n1"))
    m = (t >= 0.04) & (t <= 0.06)
    W_cycle = float(np.trapezoid(v_L[m] * i[m], t[m]))
    W_scale = float(np.trapezoid(np.abs(v_L[m] * i[m]), t[m]))
    assert abs(W_cycle) < 1e-3 * W_scale, (W_cycle, W_scale)


def test_gap_lowers_the_inductance_and_raises_the_knee():
    t0, i0, _, h0, _ = _ac_run(2e-5)
    tg, ig, _, hg, _ = _ac_run(2e-5, l_gap=0.5e-3)
    # More current for the same drive through the gapped core.
    assert np.abs(ig[tg > 0.04]).max() > 3.0 * np.abs(i0[t0 > 0.04]).max()
    assert hg.L_0 < h0.L_0


def test_observer_factory_refuses_by_name():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "n1", "gnd", 1.0)
    h = _core(b)
    with pytest.raises(RuntimeError, match="INSIDE the Newton loop"):
        p.make_hysteretic_inductor_observer(b, h, dt=1e-5)


def test_handle_and_topology():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "n1", "gnd", 1.0)
    n_before = b.graph.num_branches
    h = _core(b)
    assert b.graph.num_branches == n_before + 1          # ONE branch, no dummy source
    assert h.branch_id == n_before and h.bemf_source_id == -1
    assert h.L_0 == pytest.approx(100 ** 2 * 1e-4 * MU0 / 0.05, rel=1e-9)


def test_variable_step_engine_agrees():
    t_r, i_r, _, _, _ = _ac_run(5e-6, t_end=0.04)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 1.0, 50.0, math.pi / 2)
    b.add_resistor("Rs", "a", "n1", 0.1)
    _core(b)
    res = p.simulate(b, t_end=0.04, dt=2e-4, engine="trbdf2", rtol=1e-6, atol=1e-9)
    t_v = np.asarray(res.times)
    i_v = np.asarray(res.i("L_core"))
    ii = np.interp(t_r, t_v, i_v)
    m = t_r > 0.02
    # 6 %: the branch direction is held fixed within a step, so the
    # step containing each reversal carries an O(h) error at the loop
    # tip, and the two engines take different steps there (h_max is
    # 1 % of the period here). The fixed engine's own convergence
    # with dt is pinned separately above.
    assert np.abs(ii[m] - i_r[m]).max() < 6e-2 * np.abs(i_r[m]).max()


def test_yaml_builds_the_in_loop_device():
    y = """
circuit:
  devices:
    - {type: sine_voltage_source, name: Vs, from: a, to: gnd, v_dc: 0.0, v_amplitude: 1.0, frequency: 50.0, phase: 1.5707963267948966}
    - {type: resistor, name: Rs, from: a, to: n1, R: 0.1}
    - type: hysteretic_inductor
      name: L_core
      from: n1
      to: gnd
      N_turns: 100
      l_m: 0.05
      A_core: 1.0e-4
      Ms: 4.0e5
      a: 50.0
      alpha: 5.0e-5
      c: 0.2
      k: 30.0
simulation: {t_start: 0.0, t_end: 0.06, dt: 2.0e-5}
"""
    loaded = p.load_yaml_string(y)
    res = p.simulate(loaded.builder, t_end=0.06, dt=2e-5)
    i_y = np.asarray(res.i("L_core"))
    t, i, _, _, _ = _ac_run(2e-5)
    assert np.allclose(i_y, i, atol=1e-9 * np.abs(i).max() + 1e-12)


def test_inrush_from_a_zero_crossing_and_from_remanence():
    """Energised at the voltage zero crossing the flux doubles and the
    first-cycle current is many times the steady peak; a remanent M0
    in the same direction makes it worse. Both are the physics the
    audit called unsimulable."""
    t, i, _, _, _ = _ac_run(2e-5, phase=0.0)
    first = float(np.abs(i[t < 0.02]).max())
    tp, ip, _, _, _ = _ac_run(2e-5, phase=math.pi / 2)
    steady = float(np.abs(ip[tp > 0.04]).max())
    assert first > 5.0 * steady, (first, steady)
    tr, ir, _, _, _ = _ac_run(2e-5, phase=0.0, M0=0.8 * 4.0e5)
    assert float(np.abs(ir[tr < 0.02]).max()) > 1.2 * first
