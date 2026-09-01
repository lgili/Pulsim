"""The exponential (Shockley) diode (audit C.1).

Pulsim had two diodes and neither was exponential: `add_diode` is
binary PWL, and the smooth-blend nonlinear diode is a sigmoid
blended onto a STRAIGHT LINE, i = (v - V_F0)/R_d above the knee.
Both fix the forward drop by construction.

A real junction does not. V_F rises about 60 mV per decade of
current, so the same device that drops 0.53 V at 1 mA drops
0.77 V at 10 A. Over the decades a converter spans between light
and full load, a fixed-drop model is wrong by a couple of hundred
millivolts in the quantity that sets conduction loss -- and wrong
in OPPOSITE directions at the two ends, which is an error no
single fitted V_F0 can remove.
"""

import math

import numpy as np
import pytest

import pulsim as p

I_S, N, V_T = 1e-12, 1.0, 0.025852


def _v_forward(current, **kw):
    """Forward voltage carried by a forced current."""
    b = p.CircuitBuilder()
    # add_current_source(from, to) DRAINS `to`, so this pushes
    # `current` INTO node "a".
    b.add_current_source("I1", "a", "gnd", current)
    b.add_shockley_diode("D1", "a", "gnd", **kw)
    res = p.simulate(b, t_end=1e-9, dt=1e-9, engine="pwl")
    return float(np.asarray(res.v("a"))[-1])


def _analytic(i, I_S=I_S, n=N, V_T=V_T):
    return n * V_T * math.log(i / I_S + 1.0)


def test_it_follows_the_shockley_law_over_seven_decades():
    for i in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
        assert _v_forward(i) == pytest.approx(_analytic(i),
                                               rel=1e-4), i


def test_v_f_rises_sixty_millivolts_per_decade():
    """The property the fixed-drop models cannot have. For n = 1
    the slope is n*V_T*ln(10) = 59.5 mV/decade."""
    v1, v2 = _v_forward(1e-3), _v_forward(1e-2)
    assert (v2 - v1) == pytest.approx(V_T * math.log(10.0),
                                       rel=1e-3)


def test_the_working_range_is_the_exponential_not_a_tangent():
    """The mistake this model was FIRST written with, kept as a
    test because it is silent and plausible.

    The exponent has to be limited or Newton's first trial step
    (v ~ 50 V, e^1934) is +inf. SPICE does that with `pnjlim`,
    which limits the per-iteration STEP -- the converged answer
    still sits on the true curve. Continuing the curve by its own
    tangent above SPICE's `vcrit` instead LOOKS equivalent and is
    not: vcrit is only ~0.63 V, i.e. ~18 mA, and the tangent from
    there is a 1.41 ohm resistor. That model reported 14.7 V at
    10 A instead of 0.77 V, and converged happily while doing it.

    So: pin the physical range against the closed form, hard.
    """
    for i in (0.018, 0.1, 1.0, 10.0, 100.0, 1000.0):
        v = _v_forward(i)
        assert v == pytest.approx(_analytic(i), rel=1e-4), i
        assert v < 1.2, (i, v)      # a junction, not a resistor


def test_the_limiter_only_lives_above_any_real_current():
    """Where the tangent DOES take over, and that it is far away.
    Default i_lim = 1e6 A -- four orders past the largest
    converter anyone simulates."""
    v_lim = _analytic(1e6)
    assert 1.0 < v_lim < 1.15, v_lim
    # Just below it the law is still exactly exponential.
    assert _v_forward(9e5) == pytest.approx(_analytic(9e5),
                                             rel=1e-4)


def test_it_survives_a_voltage_newton_would_never_reach():
    """The reason the limiter exists at all: without it,
    exp(50/0.02585) = e^1934 is +inf, the Jacobian is +inf, the
    update is NaN and the run is over. A stiff forward-biased
    source must simply converge."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "s", "gnd", 50.0)
    b.add_resistor("R1", "s", "a", 5.0)
    b.add_shockley_diode("D1", "a", "gnd")
    res = p.simulate(b, t_end=1e-8, dt=1e-9, engine="pwl")
    v = float(np.asarray(res.v("a"))[-1])
    i = (50.0 - v) / 5.0
    assert v == pytest.approx(_analytic(i), rel=1e-3), (v, i)


def test_it_blocks_in_reverse():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", -100.0)
    b.add_shockley_diode("D1", "a", "gnd")
    res = p.simulate(b, t_end=1e-8, dt=1e-9, engine="pwl")
    i = abs(float(np.asarray(res.i("V1"))[-1]))
    assert i < 1e-6, i          # I_S plus G_min leakage only


def test_breakdown_conducts_past_the_knee_and_not_before():
    """A 5.1 V Zener: blocking at 4 V, conducting at 6 V."""
    def rev(v_rev):
        b = p.CircuitBuilder()
        b.add_voltage_source("V1", "s", "gnd", -v_rev)
        b.add_resistor("R1", "s", "a", 10.0)
        b.add_shockley_diode("D1", "a", "gnd", BV=5.1)
        res = p.simulate(b, t_end=1e-8, dt=1e-9, engine="pwl")
        return float(np.asarray(res.v("a"))[-1])

    assert rev(4.0) == pytest.approx(-4.0, abs=0.01)   # blocking
    v6 = rev(6.0)
    assert -6.0 < v6 < -5.1, v6                        # clamped


def test_raising_only_v_t_raises_the_drop():
    """The opposite of what everyone remembers, so it is pinned.

    At fixed current and fixed I_S, V_F = n*V_T*ln(i/I_S), so V_T
    and V_F move TOGETHER. Sweeping temperature through
    `thermal_voltage` alone therefore makes the diode drop MORE,
    not less. (An earlier version of this test asserted the
    familiar negative tempco and failed at 0.948 V vs 0.710 V --
    the expectation was wrong, and so was the docstring that had
    promised it.)
    """
    v_cold = _v_forward(1.0, V_T=p.thermal_voltage(298.15))
    v_hot_vt_only = _v_forward(1.0, V_T=p.thermal_voltage(398.15))
    assert v_hot_vt_only > v_cold
    assert v_hot_vt_only / v_cold == pytest.approx(
        398.15 / 298.15, rel=1e-3)


def test_the_negative_tempco_needs_i_s_as_well():
    """Where the real -2 mV/K comes from: I_S roughly doubles
    every 10 C, and that term overwhelms kT/q. Scale both and the
    familiar behaviour appears."""
    T_cold, T_hot = 298.15, 398.15
    v_cold = _v_forward(1.0, V_T=p.thermal_voltage(T_cold),
                        I_S=p.shockley_saturation_current_at(
                            I_S, T_cold))
    v_hot = _v_forward(1.0, V_T=p.thermal_voltage(T_hot),
                       I_S=p.shockley_saturation_current_at(
                           I_S, T_hot))
    assert v_hot < v_cold, (v_cold, v_hot)
    tempco = (v_hot - v_cold) / (T_hot - T_cold)
    assert -3e-3 < tempco < -1e-3, tempco   # ~ -2 mV/K


def test_the_closed_form_inverse_agrees_with_the_simulation():
    for i in (1e-3, 1.0, 50.0):
        assert p.shockley_voltage_for_current(i) == pytest.approx(
            _v_forward(i), rel=1e-4)


@pytest.mark.parametrize("kw,frag", [
    ({"I_S": 0.0}, "I_S"),
    ({"n": 0.0}, "emission"),
    ({"V_T": -1.0}, "V_T"),
    ({"BV": -5.1}, "magnitude"),
])
def test_unphysical_parameters_are_refused_by_name(kw, frag):
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match=frag):
        b.add_shockley_diode("D1", "a", "gnd", **kw)


def test_it_beats_a_fixed_drop_model_where_that_matters():
    """The motivating comparison, end to end. A fixed-V_F model
    fitted at one current is wrong in opposite directions at the
    two ends of a converter's load range."""
    fixed_vf = 0.7
    light, heavy = 1e-3, 10.0
    v_light, v_heavy = _v_forward(light), _v_forward(heavy)
    assert v_light < fixed_vf < v_heavy
    # And the spread is large enough to matter for loss: the
    # fixed model overstates light-load drop by >20% and
    # understates full-load drop.
    assert (fixed_vf - v_light) / v_light > 0.2
    assert v_heavy > fixed_vf
