"""Regression: PMSM / BLDC / DC motor observers publish rotor state as
named result traces on :class:`SimulationResult` (GUI integration
findings T2.2).

**Bug history.** Before this change, ``pulsim.PMSM`` (and BLDC / DC)
only exposed the C++ topology metadata: ``neutral_node``, branch ids.
Rotor speed and angle lived inside ``make_pmsm_observer``'s closure
on the ``motor.mech.omega_rad_s`` / ``.theta_rad`` fields. Callers
needed to either:

  a) wrap the observer in a custom probe to record the values, or
  b) reach into ``motor.mech`` AFTER the simulation finished, which
     only ever gave them the LAST value — no time series.

``res.v(...)`` / ``res.i(...)`` couldn't reach motor state at all.
Every motor study (speed step, torque ripple, FOC current loops)
needs ω/θ/T_em/i_d/i_q traces.

**Fix.** :func:`make_pmsm_observer`, :func:`make_bldc_observer`, and
:func:`make_dc_motor_observer` now all return a
:class:`MotorObserverBundle` — a callable step_observer that ALSO
exposes per-step trace buffers. Existing call sites that did
``obs, b_extra = make_pmsm_observer(...)`` keep working (the bundle
iterates as a 2-tuple). Callers that keep the bundle reference can
read ``bundle.omega_rad_s`` / ``.theta_rad`` / ``.T_em`` / ``.i_d``
/ ``.i_q`` / ``.i_a`` / ``.i_b`` / ``.i_c`` after the run.

:func:`pulsim.simulate` auto-attaches the bundle's traces to the
result so ``res.signal("M1.omega")``, ``res.signal("M1.theta")``,
... all resolve.

These tests pin:

1. **Bundle is iterable for backward compat** — old tuple unpacking
   keeps working unchanged.
2. **Bundle records per-step ω/θ/T_em/i_d/i_q** for PMSM.
3. **Result auto-attaches the traces** — ``res.signal("M1.omega")``
   returns the same data as ``bundle.omega_rad_s``.
4. **DC motor bundle** records ω/θ/T_em/i_a (no dq).
5. **Custom name= prefix** — multiple motors in the same circuit can
   each carry distinct trace names.
6. **Unknown signal raises** :class:`NameNotFoundError` with fuzzy
   suggestions.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# Helpers — small open-loop drives for PMSM and DC motor
# ---------------------------------------------------------------------------
def _build_pmsm_open_loop():
    """A trivial open-loop PMSM: 3-phase symmetric voltage sources
    drive the stator at a fixed frequency. Just enough to make the
    rotor spin so the observer accumulates non-trivial traces.
    """
    b = p.CircuitBuilder()
    # 3-phase sinusoidal sources at 30 Hz, 12 V_peak. The PMSM
    # back-EMF starts at zero (rotor at standstill); the rotor will
    # accelerate to roughly synchronous speed = ω_e / pp.
    f_e = 30.0
    V_peak = 12.0
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(
            f"Vs_{('a','b','c')[k]}", node, "gnd",
            v_dc=0.0, v_amplitude=V_peak,
            frequency=f_e,
            phase=-2.0 * math.pi / 3.0 * k,
        )
    motor = p.add_pmsm(
        b, name="M1",
        phase_nodes=("ua", "ub", "uc"),
        neutral_node="nn",
        R_s=0.5, L_s=2e-3,
        psi_pm=0.05, pole_pairs=4,
        J=1e-3, B=1e-4, T_load=0.0,
    )
    return b, motor


def _build_dc_motor_step():
    b = p.CircuitBuilder()
    # DC bus = 24 V, motor across it.
    b.add_voltage_source("V_dc", "vp", "gnd", 24.0)
    motor = p.add_dc_motor(
        b, name="DC1",
        armature_pos="vp", armature_neg="gnd",
        R_a=0.5, L_a=1e-3,
        Ke=0.08, Kt=0.08,
        J=5e-4, B=1e-4, T_load=0.05,
    )
    return b, motor


# ---------------------------------------------------------------------------
# Bundle contract
# ---------------------------------------------------------------------------
def test_bundle_iterable_for_backcompat():
    """Legacy callers do
        obs, b_extra = make_pmsm_observer(...)
    The bundle's __iter__ keeps that working.
    """
    b, motor = _build_pmsm_open_loop()
    obs, b_extra = p.make_pmsm_observer(b, motor, dt=2e-5)
    assert callable(obs)
    assert callable(b_extra)
    # The unpacking should give us back the bundle itself as obs (it
    # IS the step_observer).
    assert isinstance(obs, p.MotorObserverBundle)
    # Indexing also works (tuple-like).
    bundle = p.make_pmsm_observer(b, motor, dt=2e-5)
    assert bundle[0] is bundle
    assert bundle[1] is bundle.b_extra_fn


def test_pmsm_bundle_records_traces_during_run():
    b, motor = _build_pmsm_open_loop()
    DT = 5e-5
    bundle = p.make_pmsm_observer(b, motor, dt=DT, name="M1")
    res = p.simulate(
        b, t_end=10e-3, dt=DT,
        step_observer=bundle,
        b_extra_fn=bundle.b_extra_fn,
    )

    # Buffers populated.
    n = len(bundle.times)
    assert n > 100, f"expected >100 observer samples, got {n}"
    assert len(bundle.omega_rad_s) == n
    assert len(bundle.theta_rad) == n
    assert len(bundle.T_em) == n
    assert len(bundle.i_d) == n
    assert len(bundle.i_q) == n

    # Rotor accelerated past 0 (open-loop drive should torque it up).
    omega = np.asarray(bundle.omega_rad_s)
    assert omega[0] == pytest.approx(0.0, abs=1e-9)
    assert abs(omega[-1]) > 1.0, (
        f"rotor should have accelerated to non-trivial ω; got "
        f"{omega[-1]:.3f} rad/s")
    # θ monotone-ish in same sign as ω.
    theta = np.asarray(bundle.theta_rad)
    assert np.sign(theta[-1]) == np.sign(omega[-1])

    _ = res  # silence


def test_simulation_result_signal_lookup():
    """After simulate(), res.signal('M1.omega') returns the same data
    as bundle.omega_rad_s."""
    b, motor = _build_pmsm_open_loop()
    DT = 5e-5
    bundle = p.make_pmsm_observer(b, motor, dt=DT, name="M1")
    res = p.simulate(
        b, t_end=10e-3, dt=DT,
        step_observer=bundle, b_extra_fn=bundle.b_extra_fn,
    )

    omega_via_signal = res.signal("M1.omega")
    omega_via_bundle = np.asarray(bundle.omega_rad_s)
    assert isinstance(omega_via_signal, np.ndarray)
    np.testing.assert_allclose(omega_via_signal, omega_via_bundle)

    # Every documented trace name should resolve.
    for suffix in ("t", "omega", "theta", "T_em",
                       "i_d", "i_q", "i_a", "i_b", "i_c"):
        arr = res.signal(f"M1.{suffix}")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == omega_via_signal.shape

    # .signals() lists them.
    names = res.signals()
    for n in ("M1.t", "M1.omega", "M1.theta", "M1.T_em",
                  "M1.i_d", "M1.i_q", "M1.i_a", "M1.i_b", "M1.i_c"):
        assert n in names


def test_dc_motor_bundle_records_armature_current():
    b, motor = _build_dc_motor_step()
    DT = 1e-4
    bundle = p.make_dc_motor_observer(b, motor, dt=DT, name="DC1")
    assert bundle.has_dq is False
    res = p.simulate(
        b, t_end=50e-3, dt=DT,
        step_observer=bundle, b_extra_fn=bundle.b_extra_fn,
    )

    # ω rises (after a small initial backward kick from the static
    # T_load while i_a is still ramping; that's physical, not a bug).
    omega = np.asarray(bundle.omega_rad_s)
    # First sample fires AFTER one step — small magnitude regardless
    # of sign (load torque acts before current builds up).
    assert abs(omega[0]) < 1.0
    assert omega[-1] > 50.0, (
        f"DC motor should have spun up under 24 V step; "
        f"ω_final={omega[-1]:.2f} rad/s")
    # i_a starts at locked-rotor (V/R ≈ 24/0.5 = 48 A region, but
    # L_a/R_a delays it) and falls toward steady state.
    i_a = np.asarray(bundle.i_a)
    assert i_a[0] >= 0.0
    # No-dq buffers stay empty.
    assert bundle.i_d == []
    assert bundle.i_q == []

    np.testing.assert_allclose(res.signal("DC1.omega"), omega)
    np.testing.assert_allclose(res.signal("DC1.i_a"), i_a)


def test_custom_name_prefix():
    """Two PMSMs in the same circuit must publish under distinct
    trace prefixes."""
    b = p.CircuitBuilder()
    # Two PMSMs in parallel on a shared 3-phase bus.
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(
            f"Vs_{('a','b','c')[k]}", node, "gnd",
            v_dc=0.0, v_amplitude=10.0, frequency=20.0,
            phase=-2.0 * math.pi / 3.0 * k,
        )
    m1 = p.add_pmsm(
        b, name="MA", phase_nodes=("ua", "ub", "uc"), neutral_node="nA",
        R_s=0.5, L_s=2e-3, psi_pm=0.04, pole_pairs=2,
        J=1e-3, B=1e-4)
    m2 = p.add_pmsm(
        b, name="MB", phase_nodes=("ua", "ub", "uc"), neutral_node="nB",
        R_s=0.5, L_s=2e-3, psi_pm=0.04, pole_pairs=2,
        J=1e-3, B=1e-4)

    DT = 5e-5
    bA = p.make_pmsm_observer(b, m1, dt=DT, name="MA")
    bB = p.make_pmsm_observer(b, m2, dt=DT, name="MB")

    def composed_obs(t, x):
        bA(t, x)
        bB(t, x)

    def composed_b_extra(t):
        a = bA.b_extra_fn(t)
        c = bB.b_extra_fn(t)
        # Element-wise sum (each bundle only writes its own rows).
        return [a[i] + c[i] for i in range(len(a))]

    # Pre-tag for auto-attach (composed_obs hides the bundles).
    composed_obs._inner_observers = [bA, bB]   # type: ignore[attr-defined]

    res = p.simulate(
        b, t_end=8e-3, dt=DT,
        step_observer=composed_obs,
        b_extra_fn=composed_b_extra,
    )

    # Both motor's traces present, distinct, non-empty.
    omega_A = res.signal("MA.omega")
    omega_B = res.signal("MB.omega")
    assert len(omega_A) > 0 and len(omega_B) > 0
    assert len(omega_A) == len(omega_B)


def test_unknown_signal_raises_with_suggestions():
    b, motor = _build_pmsm_open_loop()
    DT = 5e-5
    bundle = p.make_pmsm_observer(b, motor, dt=DT, name="M1")
    res = p.simulate(
        b, t_end=2e-3, dt=DT,
        step_observer=bundle, b_extra_fn=bundle.b_extra_fn,
    )
    with pytest.raises(p.NameNotFoundError) as exc:
        res.signal("M1.omegaa")     # typo
    # Should suggest "M1.omega".
    assert exc.value.kind == "signal"
    assert "M1.omega" in exc.value.suggestions
