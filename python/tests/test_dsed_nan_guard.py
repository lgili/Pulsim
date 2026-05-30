"""Regression: DSED schedulers must raise an actionable error when
any step produces NaN/Inf, instead of silently storing the NaN in the
result (GUI integration findings T1.3).

**Bug history.** Running the canonical 2-stage switched topology
(boost MOSFET 65 kHz + 3φ VSI 20 kHz + PMSM) under
``engine="dsed"`` previously collapsed the bus and motor nodes to
NaN and the simulation completed silently — the user only noticed
when downstream plots / metrics showed "all NaN" with no clue WHY.

Root causes that can produce NaN at the RK45 / BDF2 step level:

1. **Ill-conditioned A matrix.** Some switch-mask combos extract an
   LTI A so ill-conditioned that ``A·x`` overflows double precision
   on the first step; ``f(t, x)`` returns NaN; the RK45 stages
   propagate it to ``x_new``.
2. **Python ``b_extra_fn`` returning NaN.** A motor / control
   observer that divides by ω (rotor speed) hits ω=0 at standstill
   and returns ``inf`` or ``NaN``. The kernel adds that straight to
   the b vector; ``rhs = A·x + b`` is NaN; same cascade.
3. **switch_fn returning a topology whose extracted dynamics blow
   up** — a switch combination that creates an infinite-gain
   feedback loop in the algebraic block.

Pre-fix behaviour. RK45's PI controller treated ``err_norm = NaN``
as a step rejection (NaN comparisons are always false), shrank
``h``, retried — and after ``max_rejects=5`` consecutive NaN
rejections threw a generic ``"PIController: 5 consecutive rejections
(err=nan, h=...)"`` error. BDF2 is fixed-step and just committed
the NaN ``x_new`` straight to the result.

**Fix.** All three DSED schedulers (``PEDSimulator`` /
``PEDSimulatorBDF2`` / ``PEDSimulatorAuto``) now detect the
NaN/Inf at the per-step level:

* **RK45 / auto-RK45**: tracks a ``nan_streak_`` counter; on a NaN
  step, invalidates FSAL, shrinks h aggressively (× 0.1), and
  retries. After ``kNanMaxStreak = 3`` consecutive NaN steps, throws
  an actionable error pointing the caller at the common root causes
  + workarounds (try ``engine='pwl'`` — T1.2 auto-LM handles
  rank-deficient Jacobians).
* **BDF2 / auto-BDF2**: BDF2 is fixed-step so we can't shrink h.
  Throws immediately on the first NaN step with the same
  actionable hint.

This test pins the new behaviour using a trivial single-state RC
circuit driven by a ``b_extra_fn`` that returns NaN. The choice
matters: it isolates the NaN-detection logic from any A-matrix
extraction quirks (those throw at adapter.A_matrix() time and aren't
the focus of T1.3).
"""
from __future__ import annotations

import re

import pytest

import pulsim as p


def _build_rc():
    """Trivial RC plant: V_in=5V, R=1kΩ, C=1µF. State vector is
    just the capacitor voltage."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "n1", 1e3)
    b.add_capacitor("C1", "n1", "gnd", 1e-6)
    return b


def _make_nan_b_extra(b):
    """A b_extra_fn that returns a NaN-laden vector. Imitates a
    motor / control observer with a divide-by-zero."""
    n_state = b.pool.state_size(b.graph)

    def b_extra(_t):
        return [float("nan")] * n_state

    return b_extra


# ---------------------------------------------------------------------------
# Default scheduler (auto)
# ---------------------------------------------------------------------------
def test_dsed_auto_raises_on_nan_b_extra():
    """The default DSED dispatch (integrator='auto') must surface NaN
    as an actionable RuntimeError instead of silently committing it."""
    b = _build_rc()
    with pytest.raises(RuntimeError) as exc:
        p.simulate(
            b, t_end=1e-4, dt=1e-6,
            engine="dsed",
            b_extra_fn=_make_nan_b_extra(b),
        )
    msg = str(exc.value)
    # Error must mention NaN/Inf and provide an actionable hint.
    assert "NaN" in msg or "nan" in msg.lower(), (
        f"error should mention NaN: got {msg!r}")
    assert "engine='pwl'" in msg, (
        f"error should suggest engine='pwl' workaround: got {msg!r}")


# ---------------------------------------------------------------------------
# RK45 (variable-step)
# ---------------------------------------------------------------------------
def test_dsed_rk45_raises_on_nan_b_extra():
    b = _build_rc()
    with pytest.raises(RuntimeError) as exc:
        p.simulate(
            b, t_end=1e-4, dt=1e-6,
            engine="dsed", integrator="rk45",
            b_extra_fn=_make_nan_b_extra(b),
        )
    msg = str(exc.value)
    assert "NaN" in msg or "nan" in msg.lower()
    # RK45 path tracks consecutive NaN streak.
    assert re.search(r"consecutive|iterations|streak", msg, re.IGNORECASE), (
        f"RK45 NaN error should mention the streak: got {msg!r}")


# ---------------------------------------------------------------------------
# BDF2 (fixed-step)
# ---------------------------------------------------------------------------
def test_dsed_bdf2_raises_on_nan_b_extra():
    """BDF2 has no h-shrink retry — must throw immediately on first
    NaN with the same actionable hint."""
    b = _build_rc()
    with pytest.raises(RuntimeError) as exc:
        p.simulate(
            b, t_end=1e-4, dt=1e-6,
            engine="dsed", integrator="bdf2",
            h_bdf2=1e-6,
            b_extra_fn=_make_nan_b_extra(b),
        )
    msg = str(exc.value)
    assert "NaN" in msg or "nan" in msg.lower()
    assert "BDF2" in msg, f"BDF2 error should self-identify: got {msg!r}"


# ---------------------------------------------------------------------------
# Sanity: clean run still succeeds (no false positive)
# ---------------------------------------------------------------------------
def test_dsed_clean_rc_still_runs():
    """Make sure the NaN guard doesn't false-positive on a healthy
    simulation.

    DSED returns ``_PEDSimulationResult`` (not the kernel
    ``SimulationResult``), so the assertion walks ``res.states``
    directly. The cap voltage in this single-state plant lives in
    ``states[:, 0]``.
    """
    import math
    import numpy as np
    b = _build_rc()
    res = p.simulate(b, t_end=1e-3, dt=1e-5, engine="dsed")
    states = np.asarray(res.states)
    assert states.shape[0] > 10, f"expected >10 samples, got {states.shape}"
    # No NaN/Inf snuck through.
    assert np.all(np.isfinite(states)), (
        "clean RC simulation should not produce non-finite samples")
    v_cap = float(states[-1, 0])
    # Cap charged up; analytical limit V_inf=5V, after τ=1ms
    # ≈ 1 time-constant of an RC=1ms, so v ≈ 5·(1−1/e) ≈ 3.16 V.
    assert 1.0 < v_cap < 5.0, (
        f"V_cap={v_cap} outside expected RC charging range "
        f"(analytical: V_inf · (1 - exp(-t/τ)) ≈ "
        f"{5.0 * (1.0 - math.exp(-1.0)):.3f} V)")
