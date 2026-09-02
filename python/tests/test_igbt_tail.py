"""The IGBT turn-off tail (audit C.1).

An IGBT is a PNP transistor driven by a MOSFET. When the gate
falls below threshold the MOS channel cuts off in nanoseconds, but
the minority carriers stored in the n- drift region can only
RECOMBINE — a tail current that keeps flowing with the full rail
already across the device. It is 40-70 % of an IGBT's turn-off
energy, and on a hard-switched inverter that is most of the
switching loss.

Measured on a clamped inductive turn-off (600 V, 100 A) before
this existed:

    t (us)   v(c) V    I_C (A)
     1.90       6.6   102.254
     2.00       6.6   102.372     gate falls here
     2.02     160.2     0.00000
     2.05     600.1     0.00000
     3.00     600.1     0.00000

    E_off = 44 uJ

I_C goes to EXACTLY zero within one step. A real 600 V / 100 A
part's datasheet E_off is 5-15 mJ, so the model understated
turn-off loss by more than two orders of magnitude — and reported
it with no warning.

The model splits the collector current:

    i_C  = (1 - k)*i_ss(v) + Q/tau
    dQ   = k*i_ss(v) - Q/tau
    dt

In equilibrium Q = k*i_ss*tau, so the two terms add back to
i_ss exactly and the DC curve is untouched by construction. At
turn-off i_ss collapses and i_C = Q/tau decays from k*I_C with
time constant tau. Both are datasheet numbers.
"""

import numpy as np
import pytest

import pulsim as p

TAU_TAIL = 1e-6     # 1 us decay
K_TAIL = 0.30       # tail starts at 30 % of the on-state current


def _turn_off(tau_tail=0.0, k_tail=0.0, t_end=8e-6, dt=2e-9):
    """Clamped inductive turn-off — the standard E_off fixture.
    The gate is held at 15 V until t = 2 us, then drops."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 600.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0,
                                2e-6, 0.0, 1e-8, 1e-8)
    b.add_inductor("Lload", "dc", "c", 500e-6, i0=100.0)
    b.add_diode("Dfw", "c", "dc", 1e3, 1e-9, 0.7)
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0,
                       tau_tail=tau_tail, k_tail=k_tail)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)   # 100 uOhm shunt
    b.add_capacitor("Cs", "c", "gnd", 2e-9)
    n = b.graph.num_switches
    res = p.simulate(b, t_end=t_end, dt=dt, engine="pwl",
                     switch_fn=lambda _t: p.SwitchStateMask(n))
    t = np.asarray(res.times)
    return t, np.asarray(res.v("c")), np.asarray(res.i("Rsense"))


def _e_off(t, v, i, t0=1.95e-6):
    m = t > t0
    return float(np.trapezoid(np.maximum(v[m] * i[m], 0.0), t[m]))


def _at(t, arr, when):
    return float(arr[int(np.argmin(np.abs(t - when)))])


# ---------------------------------------------------------------
# The gap, pinned.
# ---------------------------------------------------------------

def test_without_a_tail_the_current_vanishes_instantly():
    """The baseline this exists to fix."""
    t, _, i = _turn_off()
    assert _at(t, i, 1.99e-6) > 100.0
    assert abs(_at(t, i, 2.02e-6)) < 1e-3
    assert abs(_at(t, i, 3.0e-6)) < 1e-3


def test_the_tail_keeps_conducting_after_the_gate_is_gone():
    """The property."""
    t, _, i = _turn_off(TAU_TAIL, K_TAIL)
    i_on = _at(t, i, 1.99e-6)
    # Starts near k * I_C ...
    assert _at(t, i, 2.05e-6) == pytest.approx(K_TAIL * i_on,
                                                rel=0.25)
    # ... and is still conducting a lifetime later.
    assert _at(t, i, 3.0e-6) > 0.05 * i_on


def test_it_decays_with_the_time_constant_it_was_given():
    """One tau must take the tail down by 1/e. This is the check
    that says the ODE is being integrated, not merely that some
    current lingers."""
    t, _, i = _turn_off(TAU_TAIL, K_TAIL)
    t0 = 2.1e-6                       # clear of the switching edge
    i0 = _at(t, i, t0)
    i1 = _at(t, i, t0 + TAU_TAIL)
    assert i1 / i0 == pytest.approx(np.exp(-1.0), rel=0.15), (
        i0, i1, i1 / i0)


def test_the_tail_dominates_turn_off_energy():
    """Why it matters: 44 uJ becomes a datasheet-plausible number."""
    e_none = _e_off(*_turn_off())
    e_tail = _e_off(*_turn_off(TAU_TAIL, K_TAIL))
    assert e_tail > 20 * e_none, (e_none, e_tail)
    # A 600 V / 100 A part is 5-15 mJ; this fixture's numbers put
    # it in that neighbourhood rather than three decades below.
    assert 1e-3 < e_tail < 5e-2, e_tail


# ---------------------------------------------------------------
# What must NOT change.
# ---------------------------------------------------------------

def test_the_dc_curve_is_untouched():
    """In equilibrium the split adds back to the steady-state law
    exactly, so conduction loss cannot move when a tail is
    enabled. If this drifts, every already-validated conduction
    number drifts with it."""
    def dc(tau_tail, k_tail):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", 24.0)
        b.add_voltage_source("Vg", "g", "gnd", 15.0)
        b.add_resistor("Rin", "in", "c", 10.0)
        b.add_igbt_level1("Q1", "c", "gnd", "g", 1.5, 0.05, 5.0,
                           tau_tail=tau_tail, k_tail=k_tail)
        res = p.simulate(b, t_end=2e-4, dt=1e-7, engine="pwl")
        return float(np.asarray(res.v("c"))[-1])

    assert dc(TAU_TAIL, K_TAIL) == pytest.approx(dc(0.0, 0.0),
                                                  rel=1e-6)


def test_conduction_matches_once_the_charge_has_settled():
    """Once Q has filled, the split adds back to i_ss and the
    on-state is identical. Checked with a short lifetime so
    1.9 us is ~19 tau — see the next test for what happens
    before that."""
    tau = 1e-7
    t_a, v_a, i_a = _turn_off()
    t_b, v_b, i_b = _turn_off(tau, K_TAIL)
    assert _at(t_b, i_b, 1.9e-6) == pytest.approx(
        _at(t_a, i_a, 1.9e-6), rel=1e-3)
    assert _at(t_b, v_b, 1.9e-6) == pytest.approx(
        _at(t_a, v_a, 1.9e-6), rel=1e-3)


def test_the_charge_takes_a_lifetime_to_fill_after_turn_on():
    """Not a bug, and worth naming because it looks like one.

    An earlier version of the test above compared at 1.5 us with
    tau = 1 us and failed: v(c) read 6.87 V with a tail against
    6.60 V without. That is the charge still FILLING. Q builds
    with the same tau it decays with, so at 1.5 tau it has
    reached 1 - exp(-1.5) = 78 % of equilibrium, the stored term
    Q/tau is short of k*i_ss, and the device carries slightly
    less current — which shows up as a higher V_CE.

    That is qualitatively an IGBT's forward recovery: conductivity
    modulation takes time to establish, so V_CE(on) starts high
    and falls as the drift region floods. It falls out of the
    charge-split formulation rather than being a fitted
    forward-recovery model, so treat the SHAPE as meaningful and
    the magnitude as incidental.
    """
    t, v, _ = _turn_off(TAU_TAIL, K_TAIL)
    t0, v0 = 0.2e-6, None
    v0 = _at(t, v, t0)                  # 0.2 tau in: barely filled
    v1 = _at(t, v, 1.9e-6)              # 1.9 tau in: mostly filled
    assert v0 > v1, (v0, v1)            # V_CE falls as Q builds
    _, v_flat, _ = _turn_off()          # no tail: no transient
    assert _at(t, v_flat, t0) == pytest.approx(
        _at(t, v_flat, 1.9e-6), rel=0.02)


# ---------------------------------------------------------------
# Refusals.
# ---------------------------------------------------------------

def test_half_a_tail_is_refused():
    """Setting one knob alone models no tail at all — silently,
    which is the failure mode worth naming."""
    for kw in ({"tau_tail": 1e-6}, {"k_tail": 0.3}):
        b = p.CircuitBuilder()
        with pytest.raises(Exception, match="together"):
            b.add_igbt_level1("Q1", "c", "e", "g", **kw)


@pytest.mark.parametrize("kw,frag", [
    ({"tau_tail": -1e-6, "k_tail": 0.3}, "tau_tail"),
    ({"tau_tail": 1e-6, "k_tail": 1.0}, "k_tail"),
    ({"tau_tail": 1e-6, "k_tail": -0.1}, "k_tail"),
])
def test_bad_tail_parameters_are_refused_by_name(kw, frag):
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match=frag):
        b.add_igbt_level1("Q1", "c", "e", "g", **kw)
