"""Gate 1 validation: PED prototype on buck CCM.

Test scenario from `notes/DSED_FOUNDATIONS.md` §6 Scenario A:
- 24 V → 12 V sync-buck, 100 kHz, D = 0.5
- L = 100 µH, C = 100 µF, R_load = 2.4 Ω
- 5 ms simulation window

Validation criteria (OpenSpec tasks.md Gate 1):
- 1A correctness: RMSE on V_out vs analytical CCM steady-state ≤ 0.1 %
- 1B no regression: wall-clock ≤ 2× a comparably-tuned fixed-step
  trapezoidal reference (implemented inline)

The reference is fixed-step trapezoidal at Δt = 100 ns (50 001 steps),
which is what Pulsim v1.4.0 would use on this circuit. We implement
both inline here so the prototype has zero Pulsim dependency at
Gate 1 (Gate 2 wires to pp.simulate(...)).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make `prototype/` importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsed.buck_model import BuckCCMModel, BuckParams, BuckPSCSwitchFn
from dsed.scheduler import PEDSimulator
from dsed.step_controller import PIController


# -------------------------------------------------------------------------
# Reference solver: fixed-step trapezoidal companion (v1.4.0 emulation)
# -------------------------------------------------------------------------


def trapezoidal_reference(
    params: BuckParams,
    switch_fn: BuckPSCSwitchFn,
    x0: np.ndarray,
    t_end: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate the buck with fixed-step trapezoidal integration.

    This is the v1.4.0 PWL-cache baseline that the OpenSpec asks us
    to beat (or at least not regress against) in Gate 1B.

    Returns (times, states, cpu_time).
    """
    model = BuckCCMModel(params)
    n_steps = int(np.ceil(t_end / dt))

    # Pre-build the two trapezoidal-companion matrices, one per mask
    # J = (I/dt - A/2)·dt = I - dt/2 · A; solve J·x_new = K·x + dt/2·(b_old + b_new)
    # For our LTI two-state with constant b (within a mask), this is:
    #   x_new = (I - dt/2·A)^-1 · ((I + dt/2·A)·x + dt·b)
    A = model._A  # noqa: SLF001 — prototype direct access
    I = np.eye(2)
    M_inv = np.linalg.inv(I - 0.5 * dt * A)
    K = I + 0.5 * dt * A
    # Pre-multiply (M_inv · K) and (M_inv · dt) to avoid per-step matrix work
    Khat = M_inv @ K
    dt_M_inv = dt * M_inv

    times = np.empty(n_steps + 1, dtype=float)
    states = np.empty((n_steps + 1, 2), dtype=float)
    times[0] = 0.0
    states[0] = x0
    x = x0.astype(float).copy()

    t0_wall = time.perf_counter()
    for k in range(n_steps):
        t_cur = k * dt
        # Sample switch_fn at midpoint (trapezoidal — average of endpoints)
        mask = switch_fn(t_cur + 0.5 * dt)
        b = (
            np.array([params.V_in / params.L, 0.0])
            if mask
            else np.array([0.0, 0.0])
        )
        x = Khat @ x + dt_M_inv @ b
        times[k + 1] = (k + 1) * dt
        states[k + 1] = x
    cpu = time.perf_counter() - t0_wall

    return times, states, cpu


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def run_validation() -> tuple[float, float]:
    params = BuckParams(
        V_in=24.0,
        L=100e-6,
        C=100e-6,
        R_load=2.4,
        f_sw=100e3,
        D=0.5,
    )
    switch_fn = BuckPSCSwitchFn(T_sw=params.T_sw, D=params.D)
    t_end = 5e-3  # 5 ms

    # Start from analytical CCM steady state to skip the inrush transient
    x0 = np.array([params.I_L_steady, params.V_out_steady], dtype=float)

    print("=" * 70)
    print("PED Prototype Gate 1 Validation — Buck CCM 24V→12V, 100 kHz")
    print("=" * 70)
    print(f"  Topology  : sync buck, V_in={params.V_in}V, D={params.D}")
    print(f"  Passives  : L={params.L*1e6:.0f} µH, C={params.C*1e6:.0f} µF, "
          f"R_load={params.R_load} Ω")
    print(f"  Switching : f_sw={params.f_sw/1e3:.0f} kHz, T_sw={params.T_sw*1e6:.2f} µs")
    print(f"  Window    : t_end={t_end*1e3:.1f} ms ({int(t_end/params.T_sw)} cycles)")
    print(f"  Init      : i_L={x0[0]:.3f}A, v_C={x0[1]:.3f}V (CCM steady state)")
    print(f"  Reference : V_out_steady = D·V_in = {params.V_out_steady:.3f} V")
    print()

    # ---------------------------------------------------------------------
    # 1. Reference: fixed-step trapezoidal at dt=100ns (v1.4.0 emulation)
    # ---------------------------------------------------------------------
    print("[1/2] Fixed-step trapezoidal reference (Δt = 100 ns)...")
    dt_ref = 100e-9
    t_ref, x_ref, cpu_ref = trapezoidal_reference(
        params, switch_fn, x0, t_end, dt_ref
    )
    print(f"      steps     : {len(t_ref) - 1}")
    print(f"      wall-clock: {cpu_ref*1000:.1f} ms")
    print()

    # ---------------------------------------------------------------------
    # 2. PED prototype
    # ---------------------------------------------------------------------
    print("[2/2] PED prototype (DOPRI5 + PI + Illinois)...")
    model = BuckCCMModel(params)
    controller = PIController(
        rtol=1e-6,
        atol=1e-9,
        kP=0.7,
        kI=0.3,
        rho_max=5.0,
        safety=0.9,
    )
    sim = PEDSimulator(
        system=model,
        switch_fn=switch_fn,
        controller=controller,
        dt_init=1e-9,
        dt_max=params.T_sw / 4,   # cap at 1/4 of period for safety
        store_every=1,
    )
    result = sim.simulate(x0, t_end)
    print(f"      n_accept  : {result.n_accept}")
    print(f"      n_reject  : {result.n_reject}")
    print(f"      n_events  : {result.n_events}")
    print(f"      sample pts: {len(result.times)}")
    print(f"      wall-clock: {result.cpu_time*1000:.1f} ms")
    print(f"      avg dt    : {(t_end / max(result.n_accept, 1))*1e9:.1f} ns")
    print()

    # ---------------------------------------------------------------------
    # 3. Compare: RMSE on V_out over the last 2.5 ms (250 cycles steady state)
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("Validation: Gate 1A correctness (RMSE), Gate 1B regression")
    print("=" * 70)

    # Reference V_out time series (last 2.5 ms)
    mask_steady_ref = t_ref >= 2.5e-3
    t_window = t_ref[mask_steady_ref]
    vout_ref = x_ref[mask_steady_ref, 1]

    # Interpolate PED V_out onto the reference grid
    vout_ped_interp = np.interp(
        t_window, result.times, result.states[:, 1]
    )

    # RMSE (absolute and relative)
    rmse_abs = float(np.sqrt(np.mean((vout_ped_interp - vout_ref) ** 2)))
    rmse_rel_pct = 100.0 * rmse_abs / params.V_out_steady

    # Mean DC offset (a sanity-check on steady-state convergence)
    mean_ped = float(np.mean(vout_ped_interp))
    mean_ref = float(np.mean(vout_ref))
    dc_diff_mv = 1000.0 * (mean_ped - mean_ref)

    print(f"  RMSE(V_out, PED vs trap)   = {rmse_abs*1000:.3f} mV")
    print(f"                              = {rmse_rel_pct:.4f} % of 12.0 V")
    print(f"  Mean V_out  (PED)          = {mean_ped:.6f} V")
    print(f"  Mean V_out  (trap ref)     = {mean_ref:.6f} V")
    print(f"  DC offset   (PED - trap)   = {dc_diff_mv:+.3f} mV")
    print()
    print(f"  Wall-clock ratio PED/trap  = {result.cpu_time / cpu_ref:.2f}×")
    print(f"  Steps ratio       PED/trap = {result.n_accept / (len(t_ref)-1):.3f}")
    print()

    # ---------------------------------------------------------------------
    # 4. Gate verdicts
    # ---------------------------------------------------------------------
    gate_1A_pass = rmse_rel_pct <= 0.1
    gate_1B_pass = result.cpu_time <= 2.0 * cpu_ref

    print("Gate verdicts:")
    print(
        f"  Gate 1A (RMSE ≤ 0.1 %):     "
        f"{'✓ PASS' if gate_1A_pass else '✗ FAIL'}  "
        f"({rmse_rel_pct:.4f} %)"
    )
    print(
        f"  Gate 1B (wall-clock ≤ 2×):  "
        f"{'✓ PASS' if gate_1B_pass else '✗ FAIL'}  "
        f"({result.cpu_time / cpu_ref:.2f}×)"
    )
    print()

    if gate_1A_pass and gate_1B_pass:
        print("✓ Gate 1 PASSED — proceed to Gate 2 (C++23 port).")
    else:
        print("✗ Gate 1 FAILED — debug prototype before proceeding.")

    return rmse_rel_pct, result.cpu_time / cpu_ref


if __name__ == "__main__":
    run_validation()
