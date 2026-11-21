"""Excess (anomalous) core loss element (add-permeance-magnetic-domain 4.2-4.5).

The law (design.md, models/magnetic_loss.hpp): F = c*u/(u^2 + eps^2)^(1/4)
for the flux rate u through the segment, Bertotti's c*sign(u)*|u|^(1/2)
made smooth below |u| ~ eps. Truths:

* sine flux rate of amplitude U: the ideal law's mean power is
  c*U^1.5*K, K = mean|cos|^1.5 = Gamma(5/4)/(sqrt(pi)*Gamma(7/4)) =
  0.5564178944...; the regularised law's is below it by
  (3/4)*(eps/U)^2 + O((eps/U)^2.5) (mpmath, 40 digits:
  probes/excess_loss_truth.py; the numbers are pinned below);
* the loss of a fixed flux amplitude scales as f^1.5;
* the fitted Bertotti split (fit_core_loss_elements) put back into a run
  reproduces the loss data it was fitted on (a round trip, not a
  validation).

The circuit is algebraic in u (a voltage source across a winding forces
u = v/N), so a solved sample is the law at that sample to Newton
tolerance and the loss error is the time quadrature's; the trapezoid of
a periodic analytic function converges geometrically once the step
resolves eps/(omega*U) (8.8e-9 at 2000 samples per period, eps/U = 1e-3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

K15 = math.gamma(1.25) / (math.sqrt(math.pi) * math.gamma(1.75))
# mpmath (probes/excess_loss_truth.py): mean power of the REGULARISED law,
# c = 0.01, eps = 0.01 V, U = 10 V (the 4.3 case)
P_REG_43 = 0.17595465881849556633
P_IDEAL_43 = 0.17595478773352083655


def law(u, c, eps):
    return c * u / np.sqrt(np.sqrt(u * u + eps * eps))


def drive(c=0.01, eps=0.01, *, V=100.0, f=1e3, N=10, perm=None, R_m=None):
    """Sine voltage across an N-turn winding on one segment: an optional
    permeance, an optional eddy resistance and the excess element in
    series (design order), closing on the reference."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "e", "gnd", 0.0, V, f)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N)
    node = "m1"
    if perm is not None:
        b.add_permeance("P", node, "m2", perm)
        node = "m2"
    if R_m is not None:
        b.add_magnetic_resistance("Rm", node, "m3", R_m)
        node = "m3"
    b.add_excess_loss("X", node, "gnd", c, eps)
    return b


def period_mean(t, y, T, n):
    """Mean over the last n whole periods: trapezoid on the samples."""
    return p.time_average(t, y, t_from=t[-1] - n * T, t_to=t[-1])


def test_excess_loss_follows_its_law_at_every_sample():
    for engine, kw in (("pwl", dict(dt=1e-6)), ("trbdf2", dict(rtol=1e-9, atol=1e-12))):
        b = drive(perm=2e-6)
        r = p.simulate(b, t_end=2e-3, engine=engine, **kw)
        u = r.v("X.e")
        F = r.i("X")
        # u is the winding's flux rate, v/N
        assert np.max(np.abs(u - r.v("e") / 10)) <= 1e-9 * 10
        # F is the MMF across the element, and its law at u
        assert np.max(np.abs(F - r.v("m2"))) <= 1e-9 * np.max(np.abs(F))
        assert np.max(np.abs(F - law(u, 0.01, 0.01))) <= 1e-9 * np.max(np.abs(F))


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_excess_loss_under_sinusoidal_flux_matches_the_closed_form(engine):
    # task 4.3: c = 0.01, eps = 0.01 V, N = 10, 100 V peak sine -> 0.1759 W
    f = 1e3
    T = 1 / f
    kw = (dict(dt=T / 2000) if engine == "pwl"
          else dict(rtol=1e-10, atol=1e-13, dt=T / 2000))
    b = drive()
    r = p.simulate(b, t_end=3 * T, engine=engine, **kw)
    t = np.asarray(r.times)
    P = period_mean(t, r.v("X.e") * r.i("X"), T, 2)
    assert abs(P - 0.1759) / 0.1759 < 5e-3                 # the design's row
    assert abs(P - P_REG_43) / P_REG_43 < 1e-6              # the regularised law
    # the ideal square-root law: (3/4)(eps/U)^2 = 7.5e-7 away (7.3e-7 exact)
    assert abs(P - P_IDEAL_43) / P_IDEAL_43 < 1e-6
    assert P_IDEAL_43 == pytest.approx(0.01 * 10 ** 1.5 * K15, rel=1e-11)
    # the source supplies it (no storage in the circuit)
    p_in = r.v("e") * r.i("W", port=1)
    assert abs(period_mean(t, p_in, T, 2) - P) <= 1e-9 * P
    # reported as core loss from the kernel's law (piecewise-linear product
    # integral: second order, (omega*h)^2/12-class at 2000 samples a period)
    rows = p.device_loss_summary(b, r, t_from=t[-1] - 2 * T, t_to=t[-1])
    row = next(x for x in rows if x["name"] == "X")
    assert row["kind"] == "excess_loss" and row["category"] == "core_loss"
    assert row["c"] == 0.01 and row["eps"] == 0.01
    assert abs(row["P_avg"] - P_REG_43) / P_REG_43 < 1e-5
    assert r.power("X") > 0.0      # res.power reads the same summary row


def test_regularisation_error_is_three_quarters_eps_over_u_squared():
    # the mean power of the law on a 4000-sample period, against the ideal
    # closed form: delta = (3/4)(eps/U)^2 - O((eps/U)^2.5)
    t = (np.arange(4000) + 0.5) / 4000 * 2 * math.pi
    for r_ in (1e-2, 1e-3):
        u = np.cos(t)
        P = np.mean(u * law(u, 1.0, r_))
        delta = (K15 - P) / K15
        assert 0.0 < delta < 0.75 * r_ ** 2
        assert delta > 0.75 * r_ ** 2 - 1.0 * r_ ** 2.5


def test_excess_loss_scales_as_f_to_the_1p5_over_two_decades():
    # fixed flux amplitude Phi = 1e-4 Wb-turn per turn: U = 2*pi*f*Phi; eps
    # at 1e-4 of the smallest U keeps the regularisation below 1e-8
    N, Phi, c = 10, 1e-4, 0.02
    fs = np.array([10.0, 31.6227766, 100.0, 316.227766, 1000.0])
    eps = 1e-4 * 2 * math.pi * fs[0] * Phi
    P = []
    for f in fs:
        V = N * 2 * math.pi * f * Phi
        b = drive(c, eps, V=V, f=f, N=N)
        T = 1 / f
        r = p.simulate(b, t_end=2 * T, dt=T / 4000, engine="pwl")
        P.append(period_mean(np.asarray(r.times), r.v("X.e") * r.i("X"), T, 1))
    slope = np.polyfit(np.log(fs), np.log(P), 1)[0]
    assert abs(slope - 1.5) < 1e-3
    # and each point is the closed form
    U = 2 * math.pi * fs * Phi
    np.testing.assert_allclose(P, c * U ** 1.5 * K15, rtol=1e-6)


def test_bertotti_split_round_trip_through_a_run():
    # synthetic loss data (an illustration, not a material) -> fitted R_m, c
    # -> a run of the eddy and excess elements at the data's (f, B) ->
    # the eddy + excess shares of the data (the hysteresis share belongs to
    # the JA law and is the fit's k_h f B^beta, not simulated here)
    k_h, beta, k_e, k_x = 40.0, 2.3, 0.6, 1.8
    f = np.repeat([25e3, 100e3], 2)
    B = np.tile([0.05, 0.2], 2)
    Pv = k_h * f * B ** beta + k_e * f ** 2 * B ** 2 + k_x * f ** 1.5 * B ** 1.5
    A, seg_l, N = 76e-6, 72e-3, 10
    fit = p.fit_core_loss_elements(f, B, Pv, geometry=(A, seg_l))
    for fk, Bk, Pk in zip(f, B, Pv):
        U = 2 * math.pi * fk * Bk * A          # flux-rate amplitude per turn
        b = drive(fit.c, 1e-4 * U, V=N * U, f=fk, N=N, perm=2e-6, R_m=fit.R_m)
        T = 1 / fk
        r = p.simulate(b, t_end=2 * T, dt=T / 4000, engine="pwl")
        t = np.asarray(r.times)
        u = r.v("X.e")
        P_exc = period_mean(t, u * r.i("X"), T, 1)
        P_edd = period_mean(t, (r.v("m2") - r.v("m3")) ** 2 / fit.R_m, T, 1)
        vol = A * seg_l
        want = (Pk - fit.k_h * fk * Bk ** fit.beta) * vol
        assert (P_exc + P_edd) == pytest.approx(want, rel=1e-6)
        assert P_exc == pytest.approx(k_x * fk ** 1.5 * Bk ** 1.5 * vol, rel=1e-6)


def test_excess_loss_ac_sweep_is_the_regularised_core():
    # linearised at the DC point (zero flux rate): F = (c/sqrt(eps))*u, the
    # magnetic resistance R_m = c/sqrt(eps) — the same small-signal circuit
    c, eps = 0.01, 1e-4
    def build(excess):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "e", "gnd", 0.0)
        b.add_winding("W", "e", "gnd", "m1", "gnd", 10)
        b.add_permeance("P", "m1", "m2", 2e-6)
        if excess:
            b.add_excess_loss("X", "m2", "gnd", c, eps)
        else:
            b.add_magnetic_resistance("Rm", "m2", "gnd", c / math.sqrt(eps))
        return b
    freqs = np.logspace(1, 5, 9)
    def sweep(b):
        return np.asarray(p.run_mna_sweep(b, output_idx=b.node_id_of("m2"), freqs=freqs,
                                          v_in_branch_id=b.branch_id_of("V")).H)
    H, ref = sweep(build(True)), sweep(build(False))
    assert np.all(np.abs(ref) > 0)
    np.testing.assert_allclose(H, ref, rtol=1e-9, atol=0)


def test_excess_loss_surfaces():
    b = drive(perm=2e-6)
    kinds = {d["name"]: d for d in b.components()}
    assert kinds["X.w"]["kind"] == "gyrator"
    assert kinds["X"]["kind"] == "excess_loss"
    assert kinds["X"]["params"]["c"] == 0.01 and kinds["X"]["params"]["eps"] == 0.01
    assert str(b.pool.kind_of(b.branch_id_of("X"))) == "StoredKind.ExcessLoss"
    # the parameters are validated by name before anything is added
    for c, eps, needle in ((0.0, 1e-3, "c must be positive"),
                           (0.01, 0.0, "eps must be positive"),
                           (0.01, 1e-200, "eps must be positive"),
                           (float("nan"), 1e-3, "c must be positive")):
        bb = p.CircuitBuilder()
        with pytest.raises(ValueError, match=f"add_excess_loss.*{needle}"):
            bb.add_excess_loss("X", "m1", "gnd", c, eps)
        assert bb.graph.num_branches == 0
    # a duplicate name is refused before the winding is added
    bb = p.CircuitBuilder()
    bb.add_resistor("X", "a", "gnd", 1.0)
    n0 = bb.graph.num_branches
    with pytest.raises(ValueError, match="add_excess_loss.*already used"):
        bb.add_excess_loss("X", "m1", "gnd", 0.01, 1e-3)
    assert bb.graph.num_branches == n0
    # the exact engine refuses it by name (a Newton kind)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(drive(perm=2e-6), t_end=1e-4, engine="dsed", dt=1e-6)
    assert ei.value.device == "X"
    # YAML authoring type builds the same circuit; the IR round-trips it
    y = ("circuit:\n  devices:\n"
         "    - {type: sine_voltage_source, name: V, from: e, to: gnd, v_dc: 0, "
         "v_amplitude: 100, frequency: 1000}\n"
         "    - {type: winding, name: W, e_pos: e, e_neg: gnd, m_pos: m1, "
         "m_neg: gnd, turns: 10}\n"
         "    - {type: permeance, name: P, from: m1, to: m2, P: 2.0e-6}\n"
         "    - {type: excess_loss, name: X, from: m2, to: gnd, c: 0.01, eps: 0.01}\n")
    lc = p.load_yaml_string(y)
    assert lc.builder.build().value_hash() == b.build().value_hash()
    c = b.build()
    text = c.to_yaml()
    assert "kind: excess_loss" in text
    assert p.load_yaml_string(text).builder.build().value_hash() == c.value_hash()
    with pytest.raises(Exception, match="eps"):
        p.load_yaml_string(y.replace(", eps: 0.01", ""))
    # the field table documents both parameters
    rows = {r["name"]: r for r in p.device_fields("excess_loss")}
    assert rows["c"]["unit"] == "A/V^0.5" and rows["eps"]["unit"] == "V"


def test_excess_loss_runs_on_esdirk43_and_the_interface_path():
    b = drive(perm=2e-6)
    kw = dict(t_end=2e-3, engine="trbdf2", rtol=1e-9, atol=1e-12)
    ref = p.simulate(b, **kw)
    for extra in (dict(integrator="esdirk43"), dict(device_path="interface")):
        r = p.simulate(drive(perm=2e-6), **kw, **extra)
        F = r.i("X")
        assert np.max(np.abs(F - law(r.v("X.e"), 0.01, 0.01))) <= 1e-9 * np.max(np.abs(F))
        # same physics: the loss over the last period within the runs' accuracy
        P = period_mean(np.asarray(r.times), r.v("X.e") * F, 1e-3, 1)
        P0 = period_mean(np.asarray(ref.times), ref.v("X.e") * ref.i("X"), 1e-3, 1)
        assert P == pytest.approx(P0, rel=1e-4)


def test_excess_loss_power_in_the_loop_recorder():
    b = drive(perm=2e-6)
    rec = p.Recorder("x", ["p(X)", "i(X)"], keep="every", windows=[(1e-3, 3e-3)])
    r = p.simulate(b, t_end=3e-3, dt=5e-7, engine="pwl", recorders=[rec])
    assert r.recorder_feed["mode"] == "in-loop"
    P = r.recorded["x"].windows["p(X)"]["avg"][0]
    assert abs(P - P_REG_43) / P_REG_43 < 1e-6


def test_excess_loss_on_a_triangular_flux_is_c_u_to_the_1p5():
    # a +/-V square voltage: |u| = V/N on both slopes, so the loss is
    # c*(V/N)^1.5 less the two 1e-4*T edges (at most 1.2e-4 of it)
    f, N, U, c = 50e3, 10, 1.52, 1.7
    T = 1 / f
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "e", "gnd", -N * U, N * U, 0.0, T / 2, T,
                               T * 1e-4, T * 1e-4)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N)
    b.add_permeance("P", "m1", "m2", 2e-6)
    b.add_excess_loss("X", "m2", "gnd", c, 1e-4 * U)
    r = p.simulate(b, t_end=3 * T, dt=T / 20000, engine="pwl")
    t = np.asarray(r.times)
    P = period_mean(t, r.v("X.e") * r.i("X"), T, 2)
    assert P == pytest.approx(c * U ** 1.5, rel=3e-4)
    assert P < c * U ** 1.5


def test_thermal_specs_on_an_excess_loss_is_refused_by_name():
    """The excess element dissipates (device_loss_summary books it under
    core_loss), but the thermal summary has no junction model for it:
    thermal_specs naming it is refused by name (release group G), never
    left out of the result silently."""
    b = drive(perm=2e-6)
    r = p.simulate(b, t_end=2e-4, engine="pwl", dt=1e-6)
    with pytest.raises(p.OptionsError, match="'X'") as ei:
        p.device_thermal_summary(
            b, r, thermal_specs={"X": {"stages": [p.FosterStage(1.0, 1e-3)]}})
    assert ei.value.option == "thermal_specs"
    assert "excess_loss" in str(ei.value)
