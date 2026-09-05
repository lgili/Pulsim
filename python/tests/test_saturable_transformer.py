"""Saturable transformer on a gapped core — audit C.4.

`add_transformer` is a pair of coupled inductors: linear by
construction, the magnetising inductance folded into L_p and M, no
core anywhere in it. Measured on the example flyback (100 µH / 25 µH /
k = 0.95) wound on a plausible gapped ferrite (ETD29-class, Ae = 76 mm²,
N_p = 25, B_sat = 0.35 T) and loaded harder and harder:

    R_L    duty   i_p,pk    B implied   B/B_sat
    5.00   0.50    6.52 A    0.343 T     0.98
    1.00   0.50   12.75 A    0.671 T     1.92
    0.25   0.70   21.61 A    1.137 T     3.25

At 1.14 T on a core that saturates at 0.35 T the model still returned
a tidy 9.41 V output and not a word. A real core at 1.14 T is air; the
primary is a few µH; the current runs away.

This device is the T-model — per-winding linear leakage, an ideal
transformer, and ONE nonlinear magnetising branch whose λ(i) comes
from the core's geometry and gap — so what saturates is the thing
that saturates. Below the knee it is the coupled pair exactly (the
C++ test pins that to 2.7e-7); these tests are the other half.
"""

import numpy as np
import pytest

import pulsim as p

CORE = dict(N_p=25, N_s=13, Ae=76e-6, le=72e-3, lg=0.5e-3, B_sat=0.35)
L_M = 111.4e-6                       # the core's L_unsat (C++ pins it)
N = CORE["N_s"] / CORE["N_p"]


def _flyback(saturable, R_L, L_leak_p=5e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("Q1", "sw", "gnd", R_on=1e-2, R_off=1e9)
    if saturable:
        b.add_saturable_transformer("T1", "vin", "sw", "sec_neg", "sec_anode",
                                    L_leak_p=L_leak_p, **CORE)
    else:
        # The equivalent coupled pair: L_p = L_lp + L_m, M = n L_m,
        # L_s = n² L_m.
        L_p = L_leak_p + L_M
        L_s = N * N * L_M
        k = N * L_M / np.sqrt(L_p * L_s)
        b.add_transformer("T1", "vin", "sw", "sec_neg", "sec_anode",
                          L_p=L_p, L_s=L_s, k=k)
    b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "sec_neg", 100e-6)
    b.add_resistor("R_L", "vout", "sec_neg", R_L)
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1e-6)
    return b


def _run_flyback(saturable, R_L, t_end=1e-3):
    b = _flyback(saturable, R_L)
    pwm = p.make_pwm_switch_fn(frequency=100e3, duty=0.5, switch_idx=0,
                               num_switches=b.graph.num_switches, phase=0.0)
    res = p.simulate(b, t_end=t_end, dt=1e-8, switch_fn=pwm)
    t = np.asarray(res.times)
    m = t > 0.8 * t_end
    i_m = np.asarray(res.i("T1.m" if saturable else "T1.p"))[m]
    v_out = float(np.asarray(res.v("vout"))[m].mean())
    return float(np.abs(i_m).max()), v_out


def test_below_the_knee_the_two_models_agree():
    """Nominal load: the magnetising current stays under the knee and
    the saturable flyback IS the linear flyback."""
    i_lin, v_lin = _run_flyback(False, 5.0)
    i_sat, v_sat = _run_flyback(True, 5.0)
    assert i_sat < 6.5                        # under the ~6 A knee
    assert abs(v_sat - v_lin) < 0.05 * v_lin, (v_sat, v_lin)


def test_past_the_knee_the_core_runs_away_and_the_output_collapses():
    """Overload. The linear model's current grows with the load like a
    resistor's; the core's magnetising current runs away and the
    output falls — because the core is air past B_sat and the stored
    energy per cycle stops growing."""
    i_lin, v_lin = _run_flyback(False, 1.0)
    i_sat, v_sat = _run_flyback(True, 1.0)
    assert i_sat > 1.4 * i_lin, (i_sat, i_lin)   # measured 25.5 vs 15.4 A
    assert v_sat < 0.8 * v_lin, (v_sat, v_lin)   # measured 11.6 vs 16.7 V


def test_the_magnetising_current_is_exposed_by_name():
    b = _flyback(True, 5.0)
    pwm = p.make_pwm_switch_fn(frequency=100e3, duty=0.5, switch_idx=0,
                               num_switches=b.graph.num_switches, phase=0.0)
    res = p.simulate(b, t_end=2e-4, dt=1e-8, switch_fn=pwm)
    i_m = np.asarray(res.i("T1.m"))
    i_s = np.asarray(res.i("T1"))
    i_lp = np.asarray(res.i("T1.lp"))
    assert np.all(np.isfinite(i_m)) and np.abs(i_m).max() > 1.0
    assert np.abs(i_s).max() > 1.0
    # KCL at the magnetising node: primary leakage current = i_m + i_p,
    # with the ideal primary current i_p = −n·i_s.
    assert np.allclose(i_lp, i_m - N * i_s, atol=1e-9 * np.abs(i_lp).max() + 1e-12)


def test_inrush_when_energised_at_the_voltage_zero_crossing():
    """The textbook transformer inrush: an unloaded winding switched
    on at the source's zero crossing sees the flux integrate to
    TWICE its steady-state peak in the first half-cycle. On a linear
    core that only doubles the magnetising current; on a real core it
    drives it far past the knee, and the first-cycle current peak is
    many times the steady-state peak. The linear model cannot show
    this at all — the whole reason the audit called inrush
    'unsimulable'.

    The offset decays with L_m/R_s, so the series resistance must be
    small against the period for the doubling to survive the first
    cycle: 1 kHz with R_s = 20 mΩ gives τ = 5.6 periods."""
    def run(saturable):
        b = p.CircuitBuilder()
        N_p, Ae = CORE["N_p"], CORE["N_s"] and CORE["Ae"]
        f = 1e3
        # Steady-state peak flux at 0.8·B_sat: λ_pk = V/ω = B_pk·N·Ae.
        B_pk = 0.8 * CORE["B_sat"]
        V_amp = B_pk * N_p * Ae * 2 * np.pi * f
        b.add_sine_voltage_source("V", "src", "gnd", v_dc=0.0, v_amplitude=V_amp,
                                  frequency=f, phase=0.0)   # starts at v = 0
        b.add_resistor("Rs", "src", "p", 0.02)
        if saturable:
            b.add_saturable_transformer("T", "p", "gnd", "s", "gnd", **CORE)
        else:
            b.add_transformer("T", "p", "gnd", "s", "gnd",
                              L_p=L_M, L_s=N * N * L_M, k=1.0)
        b.add_resistor("RL", "s", "gnd", 1e6)                # unloaded
        res = p.simulate(b, t_end=30e-3, dt=2e-7)
        t = np.asarray(res.times)
        i = np.asarray(res.i("T.m" if saturable else "T.p"))
        first = float(np.abs(i[t < 1.0 / f]).max())          # first cycle
        steady = float(np.abs(i[t > 25e-3]).max())            # after settling
        return first, steady

    first_lin, steady_lin = run(False)
    first_sat, steady_sat = run(True)
    # Linear: flux doubling doubles the current (a little less, since
    # R_s has begun to eat the offset within the first cycle).
    assert 1.6 < first_lin / steady_lin < 2.1, (first_lin, steady_lin)
    # Real core: the doubled flux is past B_sat, the current is off
    # the knee, and the first-cycle peak is many times steady state.
    assert first_sat > 4.0 * steady_sat, (first_sat, steady_sat)
    assert first_sat > 3.0 * first_lin, (first_sat, first_lin)


@pytest.mark.xfail(strict=True,
                   reason="TR-BDF2 Newton stalls at h_min after a hard "
                          "switching edge when any Newton device is present: "
                          "||dx|| sits at 1e-7 and the worst residual is on "
                          "T1.lp, a LINEAR inductor row scaled by 2L/h, so "
                          "the absolute residual criterion cannot be met at "
                          "a femtosecond landing step. Solver-level; tracked "
                          "separately. The device is right on the fixed "
                          "engine (every other test in this file).")
def test_engines_agree_below_saturation():
    """On the variable-step engine, with the RC snubber the parity
    test in test_engine_auto.py uses: an ideal switch interrupting the
    primary leakage current with nowhere for it to go is an
    unresolved fast mode that the adaptive engine refuses by name
    (max_steps), while the fixed engine numerically damps it. That is
    the circuit's fault, not the device's — and with the snubber in
    place what remains is the Newton stall named in the xfail."""
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 48.0)
        b.add_switch("Q1", "sw", "gnd", 1e3, 1e-9)
        b.add_resistor("Rsn", "sw", "sn1", 47.0)
        b.add_capacitor("Csn", "sn1", "gnd", 2.2e-9)
        b.add_saturable_transformer("T1", "vin", "sw", "sec_neg", "sec_anode",
                                    L_leak_p=5e-6, **CORE)
        b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7)
        b.add_capacitor("Cout", "vout", "sec_neg", 100e-6)
        b.add_resistor("R_L", "vout", "sec_neg", 5.0)
        b.add_resistor("Rgnd", "sec_neg", "gnd", 1e-6)
        return b

    T = 10e-6
    b = build()
    q = b.switch_index_of("Q1")
    n = b.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(q, (t % T) / T < 0.4)
        return m

    ref = p.simulate(build(), t_end=1e-3, dt=1e-8, switch_fn=sf, engine="pwl")
    var = p.simulate(build(), t_end=1e-3, switch_fn=sf, engine="trbdf2",
                     rtol=1e-6, atol=1e-9)
    t_r = np.asarray(ref.times); v_r = np.asarray(ref.v("vout"))
    t_v = np.asarray(var.times); v_v = np.asarray(var.v("vout"))
    m_r = t_r > 0.8e-3
    m_v = t_v > 0.8e-3
    dc_r = float(np.trapezoid(v_r[m_r], t_r[m_r]) / (t_r[m_r][-1] - t_r[m_r][0]))
    dc_v = float(np.trapezoid(v_v[m_v], t_v[m_v]) / (t_v[m_v][-1] - t_v[m_v][0]))
    assert dc_r > 3.0
    assert abs(dc_v - dc_r) < 0.03 * abs(dc_r), (dc_v, dc_r)


def test_yaml_round_trip():
    yaml = """
circuit:
  devices:
    - {type: sine_voltage_source, name: V, from: src, to: gnd, v_dc: 0.0, v_amplitude: 10.0, frequency: 1000.0}
    - {type: resistor, name: Rs, from: src, to: p, R: 0.5}
    - type: saturable_transformer
      name: T
      p_from: p
      p_to: gnd
      s_from: s
      s_to: gnd
      N_p: 25
      N_s: 13
      Ae: 76.0e-6
      le: 72.0e-3
      lg: 0.5e-3
      B_sat: 0.35
      L_leak_p: 5.0e-6
    - {type: resistor, name: RL, from: s, to: gnd, R: 5.0}
simulation: {t_start: 0.0, t_end: 2.0e-3, dt: 1.0e-7}
"""
    loaded = p.load_yaml_string(yaml)
    res = p.simulate(loaded.builder, t_end=2e-3, dt=1e-7)
    v_s = np.asarray(res.v("s"))
    assert np.all(np.isfinite(v_s))
    # The same circuit built directly must give the same waveform —
    # that is the round trip. (The absolute level is a voltage
    # divider between R_s and the 1 kHz magnetising reactance, not
    # 10·n, so it is not asserted as a number.)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "src", "gnd", v_dc=0.0, v_amplitude=10.0,
                              frequency=1000.0)
    b.add_resistor("Rs", "src", "p", 0.5)
    b.add_saturable_transformer("T", "p", "gnd", "s", "gnd", L_leak_p=5e-6, **CORE)
    b.add_resistor("RL", "s", "gnd", 5.0)
    ref = np.asarray(p.simulate(b, t_end=2e-3, dt=1e-7).v("s"))
    assert np.abs(v_s).max() > 1.0
    assert np.allclose(v_s, ref, atol=1e-9 * np.abs(ref).max() + 1e-12)


def test_refuses_bad_turns_and_leakage_by_name():
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match="positive integers"):
        b.add_saturable_transformer("T", "a", "gnd", "s", "gnd", N_p=2.5, N_s=1,
                                    Ae=76e-6, le=72e-3, lg=0.5e-3)
    with pytest.raises(Exception, match="leakage"):
        b.add_saturable_transformer("T", "a", "gnd", "s", "gnd", N_p=2, N_s=1,
                                    Ae=76e-6, le=72e-3, lg=0.5e-3, L_leak_p=-1e-6)
    with pytest.raises(Exception, match="mm"):
        b.add_saturable_transformer("T", "a", "gnd", "s", "gnd", N_p=2, N_s=1,
                                    Ae=76.0, le=72e-3, lg=0.5e-3)
