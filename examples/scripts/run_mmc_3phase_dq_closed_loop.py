#!/usr/bin/env python3
"""3-φ MMC DC/AC inverter with dq-frame current control.

The canonical industrial MMC application: an HVDC / motor-drive
inverter that regulates active and reactive current in the
synchronous (dq) frame, with six L1 multilevel arms providing the
power conversion stage.

Topology (per phase X ∈ {a, b, c}):

                   dc_pos (V_DC)
                         │
                  ┌──[arm_X_p]──┐
                  │             │  ← upper arm: m_X_p ∈ [0, 1]
                  │            L_b_p
                  │             │
   ac_X ──┬──── L_filt ── R_load ── star ─┐
          │                                │
         L_b_n   ┌─[arm_X_n]──┐            │
          │     │             │  ← lower arm: m_X_n ∈ [0, 1]
          └─────┘             │
                              │
                          dc_neg (gnd)

(plus the three arms repeated for phases b, c — symmetric.)

Modulation (half-bridge MMC convention):
    m_X_p(t) = 0.5 − v_X_ref(t) / V_DC      ← upper arm
    m_X_n(t) = 0.5 + v_X_ref(t) / V_DC      ← lower arm
so when ``v_X_ref > 0`` the lower arm inserts more SMs (pulls ac_X
toward dc_neg) and the upper inserts less. Sum stays at ``v_C / N``
so the DC bus is balanced by construction.

Control loop (every dt):

  1. Read i_a, i_b, i_c from the three filter inductors.
  2. ``abc → αβ`` (Clarke 2/3) ── ``αβ → dq`` (Park, θ = ω_grid · t).
  3. ``i_d`` → PI → ``v_d`` and ``i_q`` → PI → ``v_q``.
  4. ``dq → αβ`` → ``abc`` → per-phase voltage references.
  5. Per-arm modulation indices written to the six MmcArmMultilevel
     objects; the L1 observer dispatches the PS-PWM staircase
     accordingly.

Setpoint sequence:
   0–15 ms : i_d_ref = 10 A, i_q_ref = 0  (turn on, settle)
   15 ms   : i_d_ref steps to 20 A         (doubles active current)
   30 ms   : i_q_ref steps to -5 A         (adds capacitive reactive)

KPIs at the end of the run show mean d-axis and q-axis currents
tracking their setpoints. Plot panels: three phase currents, dq
currents vs references, six arm cap voltages.
"""

from __future__ import annotations

import math
from math import pi as PI, sqrt
from pathlib import Path

import numpy as np

import pulsim as p


# =============================================================================
# Plant
# =============================================================================

V_DC      = 400.0           # DC bus [V] — sized so a ~200 V peak AC
                            # output gives a healthy ±0.5 modulation
                            # swing on each arm (~4 SMs at N=8).
N_SM      = 8               # SMs per arm
C_SM      = 20e-3           # per-SM cap [F] → C_arm = 2.5 mF
V_C0      = V_DC            # matched
L_B       = 5e-3            # arm inductor [H]
F_CARRIER = 5000.0          # PS-PWM carrier per SM (f_switch = 40 kHz/arm)

L_FILT    = 5e-3            # output filter [H]
R_LOAD    = 5.0             # Y-phase load [Ω]

DT        = 5e-6            # 5 µs sim step (resolves the 40 kHz f_switch)
T_END     = 45e-3           # 45 ms = ~2.25 grid periods


# =============================================================================
# Controller
# =============================================================================

F_OUT     = 50.0            # 50 Hz output fundamental

# Setpoint schedule mirrors the 6-switch dq reference (5→10 A active,
# 0→-3 A reactive) so the comparison stays apples-to-apples.
I_D_REF_1 = 5.0
I_D_REF_2 = 10.0
I_Q_REF_1 = 0.0
I_Q_REF_2 = -3.0
T_STEP_D  = 15e-3
T_STEP_Q  = 30e-3

# PI tuning identical to the 6-switch dq reference. The L1 staircase
# at 40 kHz/arm has negligible extra phase lag relative to that test
# (which uses a 5 kHz carrier).
KP_I      = 2.0
KI_I      = 2000.0
M_MAX     = 0.95            # arm-modulation depth limit


# =============================================================================
# Math helpers (Clarke / Park, power-invariant 2/3)
# =============================================================================

def clarke(a: float, b: float, c: float) -> "tuple[float, float]":
    alpha = (2.0/3.0) * (a - 0.5*b - 0.5*c)
    beta  = (2.0/3.0) * ((sqrt(3)/2)*b - (sqrt(3)/2)*c)
    return alpha, beta


def park(alpha: float, beta: float, theta: float) -> "tuple[float, float]":
    c, s = math.cos(theta), math.sin(theta)
    return c*alpha + s*beta, -s*alpha + c*beta


def inv_park(d: float, q: float, theta: float) -> "tuple[float, float]":
    c, s = math.cos(theta), math.sin(theta)
    return c*d - s*q, s*d + c*q


def inv_clarke(alpha: float, beta: float) -> "tuple[float, float, float]":
    a = alpha
    b = -0.5*alpha + (sqrt(3)/2)*beta
    c = -0.5*alpha - (sqrt(3)/2)*beta
    return a, b, c


# =============================================================================
# Plant build
# =============================================================================

def build_plant():
    """Build the 6-arm MMC + filter + load. Returns (builder, arms,
    m_refs, iL_indices) where:
      * arms is a 6-list of MmcArmMultilevel in order
        (a_p, b_p, c_p, a_n, b_n, c_n)
      * m_refs is the mutable 6-list of per-arm modulation indices
        (the controller updates this; arms read it via their fn).
      * iL_indices is a 3-tuple of state-vector indices for the
        three filter-inductor currents.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "dc_n", V_DC)

    arms: "list[p.MmcArmMultilevel]" = []
    # m_refs ordered (a_p, b_p, c_p, a_n, b_n, c_n) — index = phase + 3*half.
    m_refs: "list[float]" = [0.5] * 6

    def m_ref_reader(idx: int):
        return lambda _t, _idx=idx: m_refs[_idx]

    params = p.MmcArmMultilevelParams(
        n_sm=N_SM, c_sm=C_SM, v_c0=V_C0, f_carrier=F_CARRIER,
    )

    # Upper arms + upper arm inductors.
    for k, ph in enumerate("abc"):
        arm_p = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=params, m_ref=m_ref_reader(k),
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"ac_{ph}", L_B)

    # Lower arm inductors + lower arms.
    for k, ph in enumerate("abc"):
        b.add_inductor(f"Lb_{ph}_n", f"ac_{ph}", f"mid_{ph}_n", L_B)
        arm_n = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="dc_n",
            params=params, m_ref=m_ref_reader(3 + k),
        )
        arms.append(arm_n)

    # Output filter + Y-connected RL load.
    iL_indices: "list[int]" = []
    for ph in "abc":
        # Capture the branch id of the filter inductor *before* adding it.
        l_filt_id = b.graph.num_branches
        b.add_inductor(f"Lfilt_{ph}", f"ac_{ph}", f"rphase_{ph}", L_FILT)
        b.add_resistor(f"R_{ph}", f"rphase_{ph}", "star", R_LOAD)
        iL_indices.append(
            b.pool.branch_var_id_for_inductor(l_filt_id, b.graph),
        )

    # Star tie to dc_n — weakly grounded so the MNA matrix is well-posed.
    b.add_resistor("R_star", "star", "dc_n", 1e6)

    return b, arms, m_refs, tuple(iL_indices)


# =============================================================================
# Run
# =============================================================================

def main() -> None:
    b, arms, m_refs, (iLa_idx, iLb_idx, iLc_idx) = build_plant()

    # dq current PI controllers (independent axes).
    pi_d = p.PIController(
        Kp=KP_I, Ki=KI_I,
        output_min=-V_DC/2, output_max=V_DC/2,
    )
    pi_q = p.PIController(
        Kp=KP_I, Ki=KI_I,
        output_min=-V_DC/2, output_max=V_DC/2,
    )

    def i_d_ref_fn(t):
        return I_D_REF_1 if t < T_STEP_D else I_D_REF_2

    def i_q_ref_fn(t):
        return I_Q_REF_1 if t < T_STEP_Q else I_Q_REF_2

    # Single composed observer: cascade arm observers under one
    # callable so we have a single step_observer / b_extra_fn pair.
    obs_arms, bex = p.make_mmc_arm_multilevel_observers(b, arms, dt=DT)

    # Logging.
    n_samples = int(round(T_END / DT)) + 1
    log = {
        "t":    np.zeros(n_samples),
        "i_a":  np.zeros(n_samples),
        "i_b":  np.zeros(n_samples),
        "i_c":  np.zeros(n_samples),
        "i_d":  np.zeros(n_samples),
        "i_q":  np.zeros(n_samples),
        "i_d_ref": np.zeros(n_samples),
        "i_q_ref": np.zeros(n_samples),
        "m_a_p": np.zeros(n_samples),
        "m_a_n": np.zeros(n_samples),
        "v_C": np.zeros((6, n_samples)),
    }
    counter = [0]

    def control_and_observe(t, x):
        i_a = float(x[iLa_idx])
        i_b = float(x[iLb_idx])
        i_c = float(x[iLc_idx])
        theta = 2.0 * PI * F_OUT * t

        i_alpha, i_beta = clarke(i_a, i_b, i_c)
        i_d, i_q = park(i_alpha, i_beta, theta)

        i_d_ref = i_d_ref_fn(t)
        i_q_ref = i_q_ref_fn(t)
        v_d = pi_d.update(setpoint=i_d_ref, measured=i_d, dt=DT)
        v_q = pi_q.update(setpoint=i_q_ref, measured=i_q, dt=DT)

        v_alpha, v_beta = inv_park(v_d, v_q, theta)
        v_a, v_b, v_c = inv_clarke(v_alpha, v_beta)

        # MMC half-bridge modulation: m_X_p + m_X_n = 1 by construction.
        # Clamp the half-swing to ±M_MAX·V_DC/2 to avoid m exiting [0, 1].
        half_swing_max = M_MAX * V_DC / 2.0
        v_a = max(-half_swing_max, min(half_swing_max, v_a))
        v_b = max(-half_swing_max, min(half_swing_max, v_b))
        v_c = max(-half_swing_max, min(half_swing_max, v_c))

        for k, v_x in enumerate((v_a, v_b, v_c)):
            m_p = 0.5 - v_x / V_DC
            m_n = 0.5 + v_x / V_DC
            m_refs[k] = m_p           # arms[0..2] = upper a/b/c
            m_refs[3 + k] = m_n       # arms[3..5] = lower a/b/c

        # Run the six-arm L1 update.
        obs_arms(t, x)

        # Log.
        idx = counter[0]
        if idx < n_samples:
            log["t"][idx]       = t
            log["i_a"][idx]     = i_a
            log["i_b"][idx]     = i_b
            log["i_c"][idx]     = i_c
            log["i_d"][idx]     = i_d
            log["i_q"][idx]     = i_q
            log["i_d_ref"][idx] = i_d_ref
            log["i_q_ref"][idx] = i_q_ref
            log["m_a_p"][idx]   = m_refs[0]
            log["m_a_n"][idx]   = m_refs[3]
            for armk in range(6):
                log["v_C"][armk, idx] = arms[armk].v_C
        counter[0] += 1

    print(f"3-φ MMC DC/AC inverter — dq current control:")
    print(f"  V_dc             = {V_DC:.0f} V")
    print(f"  N_SM per arm     = {N_SM}, f_switch = "
          f"{N_SM*F_CARRIER:.0f} Hz/arm")
    print(f"  C_SM, L_b        = {C_SM*1e3:.1f} mF, {L_B*1e3:.1f} mH")
    print(f"  Filter L         = {L_FILT*1e3:.1f} mH, R_load = "
          f"{R_LOAD:.1f} Ω/phase")
    print(f"  Output freq      = {F_OUT:.1f} Hz")
    print(f"  Setpoint i_d     : {I_D_REF_1} A → {I_D_REF_2} A "
          f"@ {T_STEP_D*1e3:.0f} ms")
    print(f"  Setpoint i_q     : {I_Q_REF_1} A → {I_Q_REF_2} A "
          f"@ {T_STEP_Q*1e3:.0f} ms")
    print(f"  PI(Kp={KP_I}, Ki={KI_I}) on each axis")
    print()
    print(f"Simulating {T_END*1e3:.0f} ms at dt = {DT*1e6:.0f} µs...")

    p.simulate(
        b, t_end=T_END, dt=DT,
        step_observer=control_and_observe,
        b_extra_fn=bex,
        start_from_dc_op=True,
    )

    # Trim to actual sample count.
    n = counter[0]
    for k in ("t", "i_a", "i_b", "i_c", "i_d", "i_q", "i_d_ref",
              "i_q_ref", "m_a_p", "m_a_n"):
        log[k] = log[k][:n]
    log["v_C"] = log["v_C"][:, :n]

    # KPIs — measured mean current in the steady-state windows.
    times_ms = log["t"] * 1e3
    pre_mask  = (times_ms >  5) & (times_ms < T_STEP_D*1e3)
    btwn_mask = (times_ms >  T_STEP_D*1e3 + 3) & (times_ms < T_STEP_Q*1e3)
    post_mask = times_ms >  T_STEP_Q*1e3 + 3

    print(f"\nKPIs:")
    print(f"  i_d  pre-step   (5–{T_STEP_D*1e3:.0f} ms): "
          f"mean = {log['i_d'][pre_mask].mean():.2f} A "
          f"(target {I_D_REF_1:.1f})")
    print(f"  i_d  mid window ({T_STEP_D*1e3+3:.0f}–"
          f"{T_STEP_Q*1e3:.0f} ms): "
          f"mean = {log['i_d'][btwn_mask].mean():.2f} A "
          f"(target {I_D_REF_2:.1f})")
    print(f"  i_q  post-step  (>{T_STEP_Q*1e3+3:.0f} ms): "
          f"mean = {log['i_q'][post_mask].mean():.2f} A "
          f"(target {I_Q_REF_2:.1f})")
    print(f"  v_C drift (per arm, start → end):")
    for k, name in enumerate(("a_p", "b_p", "c_p", "a_n", "b_n", "c_n")):
        print(f"    arm_{name}: {log['v_C'][k][0]:.2f} → "
              f"{log['v_C'][k][-1]:.2f} V  "
              f"({(log['v_C'][k][-1] - log['v_C'][k][0]) / log['v_C'][k][0] * 100:+.2f} %)")

    # Plot.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Panel 1: phase currents.
    axes[0].plot(times_ms, log["i_a"], label="i_a", color="tab:red", lw=0.7)
    axes[0].plot(times_ms, log["i_b"], label="i_b", color="tab:green", lw=0.7)
    axes[0].plot(times_ms, log["i_c"], label="i_c", color="tab:blue", lw=0.7)
    axes[0].set_ylabel("phase current [A]")
    axes[0].set_title(
        "3-φ MMC DC/AC inverter — dq current control with L1 PS-PWM "
        f"(N_SM={N_SM}, f_switch={N_SM*F_CARRIER:.0f} Hz/arm)"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].axvline(T_STEP_D*1e3, ls=":", color="k", alpha=0.4)
    axes[0].axvline(T_STEP_Q*1e3, ls=":", color="k", alpha=0.4)

    # Panel 2: dq currents vs references.
    axes[1].plot(times_ms, log["i_d"], label="i_d", color="tab:orange",
                     lw=0.9)
    axes[1].plot(times_ms, log["i_d_ref"], ls="--", color="tab:orange",
                     alpha=0.5, lw=1.1, label="i_d_ref")
    axes[1].plot(times_ms, log["i_q"], label="i_q", color="tab:purple",
                     lw=0.9)
    axes[1].plot(times_ms, log["i_q_ref"], ls="--", color="tab:purple",
                     alpha=0.5, lw=1.1, label="i_q_ref")
    axes[1].set_ylabel("dq current [A]")
    axes[1].set_title("Synchronous-frame currents (Park-transformed)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", ncol=2, fontsize=9)
    axes[1].axvline(T_STEP_D*1e3, ls=":", color="k", alpha=0.4)
    axes[1].axvline(T_STEP_Q*1e3, ls=":", color="k", alpha=0.4)

    # Panel 3: six arm capacitor voltages.
    arm_labels = ("a_p", "b_p", "c_p", "a_n", "b_n", "c_n")
    for k, label in enumerate(arm_labels):
        axes[2].plot(times_ms, log["v_C"][k], label=f"arm_{label}",
                         linewidth=0.7)
    axes[2].axhline(V_C0, color="k", ls="--", alpha=0.4,
                       label=f"v_c0 = {V_C0:.0f} V")
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("v_C [V]")
    axes[2].set_title("Per-arm capacitor-sum voltages")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="lower left", ncol=4, fontsize=8)

    fig.tight_layout()
    out = (Path(__file__).resolve().parent / "out" /
           "mmc_3phase_dq_closed_loop.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
