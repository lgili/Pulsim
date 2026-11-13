"""Realistic passives (openspec add-realistic-passives).

ESR / ESL / c_par composites, the R_ac ladder and the pi-section line are
networks of existing linear R, L, C. Every reference is the analytic
impedance or response of that network (verification), except the line
step, which pins the probe's measured artefact. Values are declared
examples, not parts.
"""
import math

import numpy as np
import pytest

import pulsim as p
from pulsim.passives import RacLadder, fit_rac_ladder, recommend_line_sections


def _impedance(b, node, freqs):
    """Exact small-signal impedance seen at ``node`` (to ground): a unit
    current injected into the node row, its voltage read back, by the
    kernel's complex MNA sweep."""
    from pulsim._pulsim import run_mna_sweep_kernel

    k = b.node_id_of(node)
    r = run_mna_sweep_kernel(b.graph, b.pool, p.SwitchStateMask(0),
                               list(freqs), k, k)
    return np.asarray(r.H)


def _amplitude(t, y, f, n_periods):
    """Least-squares sine amplitude of y at f over the last n periods."""
    t, y = np.asarray(t), np.asarray(y)
    keep = t >= t[-1] - n_periods / f
    w = 2 * np.pi * f * t[keep]
    a = np.column_stack([np.sin(w), np.cos(w), np.ones_like(w)])
    c = np.linalg.lstsq(a, y[keep], rcond=None)[0]
    return math.hypot(c[0], c[1])


# ---------------------------------------------------------------------------
# ESR / ESL / c_par
# ---------------------------------------------------------------------------

def _rc(add):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "s", "gnd", 0.0, 1.0, 1e-6, 1.0)
    b.add_resistor("R", "s", "a", 10.0)
    add(b)
    return b


def test_zero_parasitics_are_the_native_call_bit_for_bit():
    def wrapped(b):
        b.add_capacitor("C", "a", "m", 1e-6, 0.5, esr=0.0, esl=0.0)
        b.add_inductor("L", "m", "gnd", 1e-4, 0.1, esr=0.0, c_par=0.0)

    def native(b):
        p.CircuitBuilder.add_capacitor._pulsim_native(b, "C", "a", "m", 1e-6, 0.5)
        p.CircuitBuilder.add_inductor._pulsim_native(b, "L", "m", "gnd", 1e-4,
                                                     0.1)

    bw, bn = _rc(wrapped), _rc(native)
    assert bw.graph.num_nodes == bn.graph.num_nodes
    assert bw.components() == bn.components()
    rw = p.simulate(bw, t_end=50e-6, dt=1e-7, engine="pwl")
    rn = p.simulate(bn, t_end=50e-6, dt=1e-7, engine="pwl")
    np.testing.assert_array_equal(np.asarray(rw.states),
                                  np.asarray(rn.states))


def _cap_esr_esl():
    b = p.CircuitBuilder()
    b.add_capacitor("C", "a", "gnd", 100e-6, esr=10e-3, esl=10e-9)
    return b


def _z_cap(f, C=100e-6, esr=10e-3, esl=10e-9):
    w = 2 * np.pi * np.asarray(f)
    return esr + 1j * (w * esl - 1 / (w * C))


def test_capacitor_impedance_is_the_series_network():
    b = _cap_esr_esl()
    f = np.logspace(3, 7, 10)
    z = _impedance(b, "a", f)
    np.testing.assert_allclose(np.abs(z), np.abs(_z_cap(f)), rtol=1e-9)
    f0 = 1 / (2 * np.pi * math.sqrt(10e-9 * 100e-6))
    assert f0 == pytest.approx(159.15e3, rel=1e-4)
    z0 = _impedance(b, "a", [f0])
    assert abs(z0[0]) == pytest.approx(10e-3, rel=1e-9)
    assert np.all(np.abs(_impedance(b, "a", [0.99 * f0, 1.01 * f0]))
                  > abs(z0[0]))


@pytest.mark.parametrize("f", [20e3, 159.15e3, 800e3])
def test_capacitor_transient_matches_the_phasor(f):
    R_src = 0.1
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 1.0, f)
    b.add_resistor("R", "s", "a", R_src)
    b.add_capacitor("C", "a", "gnd", 100e-6, esr=10e-3, esl=10e-9)
    n_per = 200
    r = p.simulate(b, t_end=200e-6 + 10 / f, dt=1 / (f * n_per),
                   engine="pwl")
    z = _z_cap(f)
    expected = abs(z / (R_src + z))
    assert _amplitude(r.times, r.v("a"), f, 5) == pytest.approx(
        expected, rel=1e-3)


def test_c0_is_the_ideal_capacitor_voltage():
    b = p.CircuitBuilder()
    b.add_capacitor("C", "a", "gnd", 1e-6, 2.0, esr=1.0, esl=1e-9)
    b.add_resistor("R", "a", "gnd", 1e6)
    r = p.simulate(b, t_end=1e-9, dt=1e-10, engine="pwl")
    assert r.v("C.esl_n")[0] == pytest.approx(2.0, rel=1e-12)


def test_esr_step_in_the_ripple():
    # +-1000 V behind 1 kOhm into 100 uF + 50 mOhm: the current reverses
    # from -1 A to +1 A (to 5e-5); the capacitor voltage barely moves in
    # one step (trapezoid: h/2C*(i- + i+) = 0), so the terminal voltage
    # steps by esr * delta_i.
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "s", "gnd", -1000.0, 1000.0, 5e-9, 1.0)
    b.add_resistor("R", "s", "a", 1000.0)
    b.add_capacitor("C", "a", "gnd", 100e-6, esr=50e-3)
    r = p.simulate(b, t_end=10e-9, dt=1e-10, engine="pwl")
    v = np.asarray(r.v("a"))
    k = int(np.argmax(np.asarray(r.v("s")) > 0.0))   # first sample at +1000 V
    expected = 50e-3 * 2000.0 / (1000.0 + 50e-3)
    assert v[k] - v[k - 1] == pytest.approx(expected, rel=1e-6)


def test_inductor_self_resonance_from_c_par():
    from scipy.optimize import minimize_scalar

    L, C, esr = 100e-6, 10e-12, 0.1
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "gnd", L, esr=esr, c_par=C)
    f = np.logspace(5, 8, 7)
    w = 2 * np.pi * f
    z_ref = 1 / (1 / (esr + 1j * w * L) + 1j * w * C)
    np.testing.assert_allclose(_impedance(b, "a", f), z_ref, rtol=1e-9)

    def neg_z(mhz):
        return -abs(_impedance(b, "a", [mhz * 1e6])[0])

    peak = minimize_scalar(neg_z, bounds=(4.9, 5.2), method="bounded",
                           options={"xatol": 1e-9}).x
    assert peak == pytest.approx(5.033e6 / 1e6, rel=1e-3)
    f_ref = 1 / (2 * math.pi * math.sqrt(L * C))
    assert peak * 1e6 == pytest.approx(f_ref, rel=1e-6)


def test_esr_loss_is_reported_under_the_component():
    f, C, esr = 1e3, 100e-6, 50e-3
    zmag = abs(esr + 1 / (2j * np.pi * f * C))
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, math.sqrt(2) * zmag, f)
    b.add_capacitor("C", "a", "gnd", C, esr=esr)
    r = p.simulate(b, t_end=20e-3, dt=1e-7, engine="pwl")
    rows = p.device_loss_summary(b, r, t_from=10e-3, t_to=20e-3)
    row = next(x for x in rows if x["name"] == "C.esr")
    assert row["P_avg"] == pytest.approx(50.0e-3, rel=1e-6)


def test_engines_agree_on_an_esr_esl_capacitor():
    def build():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "s", "gnd", 0.0, 1.0, 50e3)
        b.add_resistor("R", "s", "a", 0.5)
        b.add_capacitor("C", "a", "gnd", 10e-6, esr=20e-3, esl=5e-9)
        b.add_inductor("L", "a", "gnd", 1e-3, esr=0.1, c_par=1e-9)
        return b

    fixed = p.simulate(build(), t_end=200e-6, dt=5e-9, engine="pwl")
    var = p.simulate(build(), t_end=200e-6, engine="trbdf2")
    assert var.engine_used == "trbdf2"
    v_ref = np.interp(var.times, fixed.times, fixed.v("a"))
    assert np.max(np.abs(np.asarray(var.v("a")) - v_ref)) <= 1e-3


@pytest.mark.parametrize("kw", [dict(esr=-1e-3), dict(esl=float("nan"))])
def test_bad_capacitor_parasitic_is_refused_by_name(kw):
    with pytest.raises(ValueError, match="add_capacitor\\('C9'\\)"):
        p.CircuitBuilder().add_capacitor("C9", "a", "gnd", 1e-6, **kw)


def test_bad_inductor_parasitic_is_refused_by_name():
    with pytest.raises(ValueError, match="add_inductor\\('L9'\\).*c_par"):
        p.CircuitBuilder().add_inductor("L9", "a", "gnd", 1e-6, c_par=-1.0)


# ---------------------------------------------------------------------------
# R_ac ladder
# ---------------------------------------------------------------------------

_ONE_CELL = RacLadder(R_dc=10e-3, L_hf=1e-6, R_k=(90e-3,), L_k=(143.24e-9,))


def test_ladder_resistance_against_frequency():
    expected = [10.009e-3, 55.00e-3, 99.11e-3]
    f = [1e3, 100e3, 1e6]
    np.testing.assert_allclose(_ONE_CELL.r_ac(f), expected, rtol=2e-4)
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "gnd", 1e-6 + 143.24e-9, rac=_ONE_CELL)
    z = _impedance(b, "a", f)
    np.testing.assert_allclose(z, _ONE_CELL.impedance(f), rtol=1e-9)


def test_ladder_harmonic_loss_in_the_time_domain():
    # 1 A peak at 1 kHz plus 1 A peak at 100 kHz: P = (R(1k) + R(100k))/2.
    f1, f2 = 1e3, 100e3
    z1, z2 = _ONE_CELL.impedance([f1, f2])
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V1", "a", "m", 0.0, abs(z1), f1,
                              float(np.angle(z1)))
    b.add_sine_voltage_source("V2", "m", "gnd", 0.0, abs(z2), f2,
                              float(np.angle(z2)))
    b.add_inductor("L", "a", "gnd", 1e-6 + 143.24e-9, rac=_ONE_CELL)
    r = p.simulate(b, t_end=3e-3, dt=20e-9, engine="pwl")
    t = np.asarray(r.times)
    keep = t >= 1e-3
    pw = np.asarray(r.v("a"))[keep] * np.asarray(r.i("L"))[keep]
    p_avg = np.trapezoid(pw, t[keep]) / (t[keep][-1] - t[keep][0])
    assert p_avg == pytest.approx(32.50e-3, rel=1e-3)


def test_fit_round_trip_recovers_the_ladder():
    f = np.logspace(3, 7, 60)
    corners = np.logspace(3, 7, 4)
    tau = 1 / (2 * np.pi * corners)
    r_k = np.array([5e-3, 20e-3, 40e-3, 80e-3])
    truth = RacLadder(R_dc=10e-3, L_hf=2e-6, R_k=tuple(r_k),
                      L_k=tuple(r_k * tau))
    L = 2e-6 + float(np.sum(r_k * tau))
    fit = fit_rac_ladder(f, truth.r_ac(f), L=L, n_cells=4)
    np.testing.assert_allclose(fit.R_k, truth.R_k, rtol=1e-6)
    np.testing.assert_allclose(fit.L_k, truth.L_k, rtol=1e-6)
    assert fit.R_dc == pytest.approx(10e-3, rel=1e-6)
    assert fit.L_hf == pytest.approx(2e-6, rel=1e-6)
    assert fit.residual <= 1e-6


def _round_wire(f, a=0.5e-3, rho=1.68e-8, mu=4e-7 * np.pi):
    """R_ac/R_dc of a round wire (Bessel form)."""
    from scipy.special import jv

    delta = np.sqrt(rho / (np.pi * np.asarray(f) * mu))
    k = (1 - 1j) / delta
    return np.real(k * a / 2 * jv(0, k * a) / jv(1, k * a))


def test_fit_on_the_skin_effect_curve():
    # Two decades around the skin corner (delta = radius): R_ac/R_dc goes
    # 1.0002 -> 1.853. Measured residuals: 1 cell 39 % (refused), 2 cells
    # 1.75 %, 4 cells (the default) 0.60 %, 6 cells 0.18 %.
    f_c = 1.68e-8 / (np.pi * 4e-7 * np.pi * (0.5e-3) ** 2)
    f = np.logspace(math.log10(f_c / 10), math.log10(f_c * 10), 80)
    r = 10e-3 * _round_wire(f)
    with pytest.raises(ValueError, match="residual"):
        fit_rac_ladder(f, r, L=1e-6, n_cells=1, name="Lw")
    assert fit_rac_ladder(f, r, L=1e-6, n_cells=2).residual <= 0.05
    fit = fit_rac_ladder(f, r, L=1e-6, name="Lw")
    assert fit.residual <= 0.01
    assert fit.band == (f[0], f[-1])


def test_fit_refuses_an_inductance_deficit_by_name():
    f = np.logspace(3, 6, 30)
    r = _ONE_CELL.r_ac(f)
    with pytest.raises(ValueError, match="'Lx'.*deficit"):
        fit_rac_ladder(f, r, L=10e-9, name="Lx")


# ---------------------------------------------------------------------------
# pi-section transmission line
# ---------------------------------------------------------------------------

Z0, TAU = 50.0, 100e-9


def test_pi_line_input_impedance():
    b = p.CircuitBuilder()
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            sections=50)
    f = 0.5 / (2 * np.pi * TAU)
    z = _impedance(b, "k", [f])[0]
    assert -Z0 / math.tan(0.5) == pytest.approx(-91.52, rel=1e-4)
    assert abs(z.real) < 1e-6 * abs(z)
    assert z.imag == pytest.approx(-Z0 / math.tan(0.5), rel=1e-3)


def _line_step(n_sec, t_rise=0.0):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "s", "gnd", 0.0, 1.0, 10e-9, 1.0, 0.0,
                               max(t_rise, 1e-12), 1e-12)
    b.add_resistor("Rs", "s", "k", Z0)
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            sections=n_sec)
    b.add_resistor("Rend", "m", "gnd", 1e9)
    r = p.simulate(b, t_end=4 * TAU, dt=25e-12, engine="pwl")
    t, v = np.asarray(r.times), np.asarray(r.v("m"))
    k = int(np.argmax(v >= 0.5))
    t50 = np.interp(0.5, [v[k - 1], v[k]], [t[k - 1], t[k]])
    return (t50 - 10e-9 - t_rise / 2) / TAU, v.max()


def test_pi_line_ideal_step_overshoots_27_percent():
    # The documented artefact (probes/pi_line_step.py): the exact line
    # gives 1.000 V with no overshoot; 50 lumped sections give 1.271 V.
    t50, peak = _line_step(50)
    assert peak == pytest.approx(1.271, rel=0.01)
    assert t50 == pytest.approx(1.015, rel=0.01)


def test_recommended_sections_keep_a_finite_edge_near_5_percent():
    n = recommend_line_sections(TAU, 20e-9)
    assert n == 55
    _, peak = _line_step(n, t_rise=20e-9)
    assert peak < 1.06


def test_line_from_per_metre_data_and_refusals():
    b = p.CircuitBuilder()
    # 250 nH/m, 100 pF/m, 2 m: Z0 = 50 ohm, delay = 10 ns.
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", L_per_m=250e-9,
                            C_per_m=100e-12, length=2.0, sections=4)
    assert sum(c["params"].get("C_farads", 0.0) for c in b.components()
               ) == pytest.approx(200e-12, rel=1e-12)
    with pytest.raises(p.UnsupportedError, match="model='pi'") as ei:
        b.add_transmission_line("T2", "k", "gnd", "m", "gnd", Z0=Z0,
                                delay=TAU, model="bergeron")
    assert (ei.value.feature, ei.value.device) == ("model", "T2")
    with pytest.raises(ValueError, match="'T3'.*Z0"):
        b.add_transmission_line("T3", "k", "gnd", "m", "gnd", Z0=-1.0,
                                delay=TAU, sections=4)
    with pytest.raises(ValueError, match="'T4'.*sections"):
        b.add_transmission_line("T4", "k", "gnd", "m", "gnd", Z0=Z0,
                                delay=TAU)
