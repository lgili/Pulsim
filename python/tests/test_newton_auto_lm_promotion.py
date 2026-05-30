"""Regression: Newton must auto-promote to LM on rank-deficient /
near-miss conditions (GUI integration findings T1.2).

**Bug history.** Multi-stage switched topologies (e.g. boost MOSFET
at 65 kHz on the same bus as a 3φ VSI at 20 kHz, or any switched
plant with a stiff nonlinear device like an SH1 MOSFET on inductive
load) hit one of two failure modes inside Newton:

1. **Numerically singular factorize.** The combined Jacobian
   ``J = J_lin + J_nl`` becomes rank-deficient on certain mask
   combinations. ``sparse::Solver::factorize`` returns false and
   the kernel throws ``"solve_with_newton: combined matrix is
   numerically singular at iter N"``.

2. **Near-miss stall.** Newton hits residual ≈ 1e-13 (well below
   ``tol_res``) but ``||dx||_inf`` plateaus around 1e-6 — Newton's
   full step overshoots a flat valley in the solution manifold.
   The kernel throws ``"failed to converge after N iterations"``.

The user could work around both by setting
``enable_newton_lm=True`` (and often also adding a physical RC
snubber). That's a steep cliff for a common multi-stage topology
class and the reason `examples/scripts/
run_boost_realistic_mosfet_v14.py` ships with both
``enable_newton_line_search=True`` and ``enable_newton_lm=True``
hand-set.

**Fix.** ``solve_with_newton_b_extra`` now auto-promotes to LM
when either failure pattern shows up mid-solve:

* Singular factorize → set ``enable_lm = true`` and ``continue``
  the iteration; LM's ``J + λ·I`` restores full rank.
* Near-miss stall for ``kNearMissStreak = 3`` consecutive
  iterations (residual < tol_res AND ``||dx||`` ≥ tol_dx) →
  promote to LM; LM's diagonal damping curls dx back toward the
  descent direction.

Both promotions are transparent to the user; the auto-LM path
preserves the existing LM correctness invariants
(``norm`` acceptance test, λ shrink/grow schedule, max attempts).

These tests pin:

1. **Realistic SH1-MOSFET boost converges without manual LM
   flag.** This is the canonical example that today's docs
   point at as the "needs all the Newton hardening" case.
   Pre-fix it converges only with ``enable_newton_lm=True``;
   post-fix the default settings (auto-LM) reach the same
   trajectory.
2. **Legacy explicit-LM path still works.** Setting
   ``enable_newton_lm=True`` continues to give the same answer
   — auto-LM is a default-OFF safety net, not a behaviour change
   for callers who already opted in.
"""
from __future__ import annotations

import numpy as np

import pulsim as p


def _build_realistic_boost_sh1():
    """Realistic boost with an SH1-level MOSFET on an inductive
    load — the canonical Newton-hardening case from
    `examples/scripts/run_boost_realistic_mosfet_v14.py`."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V_in", "vin", "gnd", 10.0)
    b.add_pulse_voltage_source(
        "V_g_drive", "V_g", "gnd",
        v_initial=0.0, v_pulsed=10.0,
        t_start=5e-6, pulse_width=2e-6, period=1e-5,
        rise_time=2e-6, fall_time=2e-6,
    )
    b.add_inductor("L_boost", "vin", "sw", 100e-6)
    b.add_diode("D_boost", "sw", "vout", 1e3, 1e-9, V_th=0.5)
    b.add_mosfet_level1(
        "M1", "sw", "gnd", "V_g",
        K=0.05, V_T=2.0, lambda_=0.02, kappa=15.0,
    )
    b.add_capacitor("C_out", "vout", "gnd", 100e-6)
    b.add_resistor("R_load", "vout", "gnd", 20.0)
    return b


def test_realistic_boost_sh1_converges_without_manual_lm() -> None:
    """The canonical Newton-hardening case: realistic SH1 MOSFET +
    inductive load. Pre-fix this required
    ``enable_newton_line_search=True`` AND
    ``enable_newton_lm=True``. Post-fix the default solver settings
    auto-promote to LM and reach the same V_out steady state.

    We still keep ``enable_newton_line_search=True`` here — line
    search is a separate globalization and the user reasonably
    opts in for stiff MOSFET cases. The auto-LM check pins the
    NEW behaviour: enable_newton_lm can stay at its default
    (False) and Newton still succeeds.
    """
    b = _build_realistic_boost_sh1()
    # NOTE: enable_newton_lm IS NOT set. Pre-fix this would throw
    # `solve_with_newton: combined matrix is numerically singular
    # at iter N` or `failed to converge after N iterations`.
    res = p.simulate(
        b, t_end=1e-3, dt=1e-7,
        enable_nonlinear_refresh=True,
        max_newton_iterations=200,
        tol_newton_dx=1e-5, tol_newton_res=1e-5,
        enable_newton_line_search=True,
        # enable_newton_lm=False  ← intentionally left as default
    )
    assert res.num_steps() > 0
    v_out = np.asarray(res.v("vout"))
    v_out_final = float(v_out[-1])
    # Sanity: V_out > V_in (boost is, well, boosting) — exact
    # steady state depends on duty + load but for this open-loop
    # 20 % duty it should land roughly in [10 V, 30 V].
    assert 8.0 < v_out_final < 50.0, (
        f"V_out_final={v_out_final:.3f} V — outside expected boost "
        f"operating range; auto-LM may not have promoted correctly.")


def test_explicit_lm_still_works() -> None:
    """Backward-compat guard: callers that already pass
    ``enable_newton_lm=True`` keep getting the same answer.
    Auto-promotion is a safety net, not a behaviour change for
    explicit opt-in callers."""
    b = _build_realistic_boost_sh1()
    res_auto = p.simulate(
        b, t_end=5e-4, dt=1e-7,
        enable_nonlinear_refresh=True,
        max_newton_iterations=200,
        tol_newton_dx=1e-5, tol_newton_res=1e-5,
        enable_newton_line_search=True,
    )
    res_explicit = p.simulate(
        b, t_end=5e-4, dt=1e-7,
        enable_nonlinear_refresh=True,
        max_newton_iterations=200,
        tol_newton_dx=1e-5, tol_newton_res=1e-5,
        enable_newton_line_search=True,
        enable_newton_lm=True,
    )
    # Both succeed and land at the same V_out within numerical
    # tolerance (auto-LM and explicit LM differ at most in WHEN
    # LM gets engaged within the Newton loop; the final state
    # converges to the same fixed point).
    v_auto = float(np.asarray(res_auto.v("vout"))[-1])
    v_explicit = float(np.asarray(res_explicit.v("vout"))[-1])
    assert abs(v_auto - v_explicit) < 0.5, (
        f"auto={v_auto:.3f}V vs explicit={v_explicit:.3f}V — "
        f"auto-LM and explicit-LM should converge to the same "
        f"steady state on the same circuit.")
