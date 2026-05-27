#!/usr/bin/env python3
"""IM indirect FOC closed-loop with V/f bootstrap → IFOC handoff.

Closed-loop IFOC drive on the 4 kW squirrel-cage IM:

  measured i_a/i_b/i_c ── Clarke ── Park(θ̂_e) ── i_d, i_q
  i_d_ref, i_q_ref ──→ PI(i_d), PI(i_q) ──→ v_d, v_q
  v_d, v_q ── Inverse Park(θ̂_e) ──→ v_α, v_β ──→ 3-φ b_extra

The angle ``θ̂_e`` is built from:
  θ̂_e = pp · θ_m_measured (sensored)   +   ∫ ω_sl_ref dt   (slip feed-forward)
       ω_sl_ref = i_q_ref / (T_r · i_d_ref)

This is the **textbook indirect FOC** strategy that PSIM and PLECS
ship as a drag-and-drop template.

Two-phase startup — why
-----------------------
Closing an IFOC loop from cold (zero rotor flux, zero angle) demands the
controller pick a synchronous frame BEFORE there is a rotor flux to align
with. The controller end up commanding voltage along an arbitrary frame
and the resulting current builds rotor flux somewhere "random" — the
plant and the controller never agree on where the d-axis lives.

Industrial drives solve this with a **V/f open-loop bootstrap**: spin the
machine up to a low base frequency on constant V/f (which DOES build
rotor flux because the stationary-frame model is unambiguous), then HAND
OFF to the closed IFOC loop with the slip-integrator seeded to match the
current rotor-flux angle. The handoff is glitchless because both loops
agree on what the synchronous angle is at the moment of switchover.

This demo implements that:

* **Phase 1 (0-150 ms)**: V/f open-loop ramp 0 → 25 Hz with low-frequency
  voltage boost. Builds up |ψ_r| from 0 to ~0.5 Wb. Mechanical speed
  rises with the synchronous frequency.
* **Phase 2 (150-300 ms)**: hold V/f at 25 Hz so the rotor flux locks in.
* **Phase 3 (300 ms onwards)**: close the IFOC loop. The slip integrator
  is seeded so θ̂_e at handoff equals atan2(ψ_β, ψ_α). PI(i_d), PI(i_q)
  with dq decoupling take over voltage commands.

Topology simplification
-----------------------
Instead of a full 6-switch 3-φ VSI + SVM + dead-time, we inject the
controller's v_α/v_β directly into 3 voltage sources via b_extra
(ideal-VFD approximation). This isolates the **control-law +
observer** behaviour from the PWM-switching artefacts.

A "full" version with 6-switch VSI + SVM is the next refinement.

What you should see
-------------------
* During V/f phase: |ψ_r| grows monotonically, ω_m accelerates linearly.
* At handoff (t ≈ 300 ms): smooth (no glitch) transition; PI integrators
  take over.
* In IFOC phase: i_d tracks ~i_d_ref (sets rotor flux); i_q produces real
  torque (~Lm²/Lr · i_d_ref · i_q_ref); ω_m continues to accelerate
  toward synchronous because T_load = 0.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# ---------- 4 kW reference machine -----------------------------------------

P_RATED = 4_000.0
V_LL    = 400.0
F_NOM   = 50.0
PP      = 2

# ---------- Controller references ------------------------------------------

I_D_REF = 8.0        # A — magnetizing (≈ 70% of nominal flux current)
I_Q_REF = 6.0        # A — torque-producing
V_LIMIT = math.sqrt(2.0) * V_LL / math.sqrt(3.0)   # nominal phase peak ≈ 326 V

# ---------- V/f bootstrap parameters --------------------------------------

F_BOOTSTRAP = 25.0           # Hz — base frequency at handoff
T_BOOTSTRAP = 0.150          # ramp 0 → F_BOOTSTRAP over this duration
T_HANDOFF_HOLD = 0.150       # hold at F_BOOTSTRAP for this long before IFOC
HANDOFF_TIME = T_BOOTSTRAP + T_HANDOFF_HOLD

# ---------- IFOC PI tuning (sized from plant) ------------------------------

OMEGA_CL = 2.0 * math.pi * 20.0   # closed-loop bandwidth 20 Hz (conservative)

# ---------- Simulation -----------------------------------------------------

DT      = 50e-6
T_END   = 1.0


def main():
    # Plant: 3 voltage sources (zeroed) + IM + small Rs per phase.
    b = p.CircuitBuilder()
    src_a_id = b.graph.num_branches
    b.add_voltage_source("Va", "a", "gnd", 0.0)
    src_b_id = b.graph.num_branches
    b.add_voltage_source("Vb", "b", "gnd", 0.0)
    src_c_id = b.graph.num_branches
    b.add_voltage_source("Vc", "c", "gnd", 0.0)
    kw = p.im_parameters_from_nameplate(
        P_rated_W=P_RATED, V_LL_V=V_LL, f_Hz=F_NOM, pole_pairs=PP)
    diag = kw.pop("_diagnostics")
    # Viscous damping B so the motor reaches a stable speed below sync.
    # Without damping, T_em > 0 → motor accelerates past sync indefinitely.
    # B chosen so steady-state ω ≈ 100 rad/s for the IFOC torque level.
    motor = p.add_induction_motor(
        b, name="IM",
        phase_nodes=("a", "b", "c"),
        neutral_node="n",
        T_load=0.0,
        B=0.10,            # Nm·s/rad
        **kw,
    )
    b.add_resistor("R_leak", "n", "gnd", 1e6)

    im_obs, im_b_extra = p.make_induction_motor_observer(b, motor, dt=DT)

    src_a_idx = b.pool.branch_var_id_for_source(src_a_id, b.graph)
    src_b_idx = b.pool.branch_var_id_for_source(src_b_id, b.graph)
    src_c_idx = b.pool.branch_var_id_for_source(src_c_id, b.graph)
    i_a_idx = b.pool.branch_var_id_for_inductor(motor.phase_branch_ids[0], b.graph)
    i_b_idx = b.pool.branch_var_id_for_inductor(motor.phase_branch_ids[1], b.graph)
    i_c_idx = b.pool.branch_var_id_for_inductor(motor.phase_branch_ids[2], b.graph)

    sqrt3 = math.sqrt(3.0)
    Tr = kw["L_r"] / kw["R_r"]
    L_s = kw["L_s"]
    L_m = kw["L_m"]
    R_s = kw["R_s"]
    sigma = 1.0 - (L_m * L_m) / (L_s * kw["L_r"])
    L_sigma_s = sigma * L_s

    # PI gains: cancel the plant pole (R_s / L_σs) and set the bandwidth.
    KP_I = L_sigma_s * OMEGA_CL
    KI_I = KP_I * R_s / L_sigma_s

    # V/f bootstrap: keep V/f constant from F_BOOTSTRAP onwards.
    V_AMP_NOM = V_LL * math.sqrt(2.0 / 3.0)
    V_BOOST = 0.05 * V_AMP_NOM

    state = {
        "theta_e_slip": 0.0,
        "pi_d_integ": 0.0,
        "pi_q_integ": 0.0,
        "v_alpha": 0.0,
        "v_beta": 0.0,
    }
    log_t = []
    log_id = []
    log_iq = []
    log_vd = []
    log_vq = []
    log_omega = []
    log_torque = []
    log_psi = []

    def vf_voltage(t):
        """3-φ sine voltage during the V/f bootstrap phase."""
        if t < T_BOOTSTRAP:
            f_now = F_BOOTSTRAP * (t / T_BOOTSTRAP)
        else:
            f_now = F_BOOTSTRAP
        v_amp = max(V_BOOST, V_AMP_NOM * (f_now / F_NOM))
        omega = 2.0 * math.pi * f_now
        v_a = v_amp * math.sin(omega * t)
        v_b = v_amp * math.sin(omega * t - 2.0 * math.pi / 3.0)
        v_c = v_amp * math.sin(omega * t + 2.0 * math.pi / 3.0)
        return v_a, v_b, v_c

    def step_observer(t, x):
        # Run the IM observer first (updates motor.mech + b_extra back-EMF).
        im_obs(t, x)

        # Measured stator currents.
        i_a = float(x[i_a_idx])
        i_b = float(x[i_b_idx])
        i_c = float(x[i_c_idx])
        # Amplitude-invariant Clarke 3φ → αβ.
        i_alpha = (2.0 / 3.0) * (i_a - 0.5 * i_b - 0.5 * i_c)
        i_beta  = (2.0 / 3.0) * (sqrt3 / 2.0) * (i_b - i_c)

        if t < HANDOFF_TIME:
            # V/f phase — controller idle, voltages set by vf_voltage()
            # directly in combined_b_extra.
            state["v_alpha"] = 0.0
            state["v_beta"]  = 0.0
            # Seed the slip integrator so theta_e at handoff matches the
            # actual rotor-flux angle: theta_e = pp·θ_m + theta_slip.
            # Set theta_slip = atan2(ψ_β, ψ_α) − pp·θ_m so they match.
            psi_norm = math.hypot(motor.psi_r_alpha_Wb, motor.psi_r_beta_Wb)
            if psi_norm > 0.01:
                state["theta_e_slip"] = (
                    math.atan2(motor.psi_r_beta_Wb, motor.psi_r_alpha_Wb)
                    - PP * motor.mech.theta_rad
                )
            i_d_meas = 0.0
            i_q_meas = 0.0
            v_d = 0.0
            v_q = 0.0
        else:
            # IFOC closed loop.
            omega_sl = I_Q_REF / (Tr * I_D_REF) if I_D_REF > 1e-6 else 0.0
            state["theta_e_slip"] += omega_sl * DT
            theta_e_hat = PP * motor.mech.theta_rad + state["theta_e_slip"]
            theta_e_hat = math.fmod(theta_e_hat, 2.0 * math.pi)
            if theta_e_hat < 0.0:
                theta_e_hat += 2.0 * math.pi

            # Park αβ → dq.
            c, s = math.cos(theta_e_hat), math.sin(theta_e_hat)
            i_d_meas =  c * i_alpha + s * i_beta
            i_q_meas = -s * i_alpha + c * i_beta

            # Synchronous electrical speed for dq decoupling.
            omega_e = PP * motor.mech.omega_rad_s + omega_sl

            # PI on i_d with feedforward decoupling (−ω_e·σ·L_s·i_q).
            err_d = I_D_REF - i_d_meas
            state["pi_d_integ"] += KI_I * DT * err_d
            v_d_ff = -omega_e * L_sigma_s * i_q_meas
            v_d = KP_I * err_d + state["pi_d_integ"] + v_d_ff
            if v_d > V_LIMIT:
                v_d = V_LIMIT
                state["pi_d_integ"] = v_d - KP_I * err_d - v_d_ff
            elif v_d < -V_LIMIT:
                v_d = -V_LIMIT
                state["pi_d_integ"] = v_d - KP_I * err_d - v_d_ff

            # PI on i_q with feedforward decoupling (+ω_e·L_s·i_d).
            err_q = I_Q_REF - i_q_meas
            state["pi_q_integ"] += KI_I * DT * err_q
            v_q_ff = omega_e * L_s * i_d_meas
            v_q = KP_I * err_q + state["pi_q_integ"] + v_q_ff
            if v_q > V_LIMIT:
                v_q = V_LIMIT
                state["pi_q_integ"] = v_q - KP_I * err_q - v_q_ff
            elif v_q < -V_LIMIT:
                v_q = -V_LIMIT
                state["pi_q_integ"] = v_q - KP_I * err_q - v_q_ff

            # Inverse Park dq → αβ.
            v_alpha = v_d * c - v_q * s
            v_beta  = v_d * s + v_q * c
            state["v_alpha"] = v_alpha
            state["v_beta"]  = v_beta

        # Diagnostic log every 5 ms.
        if not log_t or t - log_t[-1] >= 5e-3:
            log_t.append(t)
            log_id.append(i_d_meas)
            log_iq.append(i_q_meas)
            log_vd.append(v_d)
            log_vq.append(v_q)
            log_omega.append(motor.mech.omega_rad_s)
            log_torque.append(motor.last_T_em_Nm)
            log_psi.append(math.hypot(
                motor.psi_r_alpha_Wb, motor.psi_r_beta_Wb))

    def combined_b_extra(t):
        out = list(im_b_extra(t))
        if t < HANDOFF_TIME:
            # V/f-driven voltages
            v_a, v_b, v_c = vf_voltage(t)
        else:
            # IFOC-driven voltages: αβ → abc inverse Clarke.
            v_alpha = state["v_alpha"]
            v_beta = state["v_beta"]
            v_a = v_alpha
            v_b = -0.5 * v_alpha + (sqrt3 / 2.0) * v_beta
            v_c = -0.5 * v_alpha - (sqrt3 / 2.0) * v_beta
        out[src_a_idx] += v_a
        out[src_b_idx] += v_b
        out[src_c_idx] += v_c
        return out

    print("  IM IFOC indirect closed-loop (V/f bootstrap → IFOC handoff):")
    print("    Plant: 4 kW / 400 V / 50 Hz / 4-pole IM")
    print(f"    Tr (rotor TC) = {Tr*1e3:.2f} ms, σ = {sigma:.3f}, "
          f"L_σs = {L_sigma_s*1e3:.2f} mH")
    print(f"    Controller: PI(i_d={I_D_REF}A), PI(i_q={I_Q_REF}A), "
          f"Kp={KP_I:.2f}, Ki={KI_I:.0f}")
    print(f"                ω_cl = {OMEGA_CL:.0f} rad/s ({OMEGA_CL/(2*math.pi):.1f} Hz)")
    print(f"    V/f bootstrap: 0 → {F_BOOTSTRAP} Hz over {T_BOOTSTRAP*1e3:.0f} ms, "
          f"hold {T_HANDOFF_HOLD*1e3:.0f} ms, close loop at {HANDOFF_TIME*1e3:.0f} ms")
    print(f"    Sim {T_END}s @ dt={DT*1e6:.0f}µs = "
          f"{int(T_END/DT):,} steps")

    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=lambda t: p.SwitchStateMask(0),
        step_observer=step_observer,
        b_extra_fn=combined_b_extra,
        progress=True,
    )

    # KPI window: last 100 ms.
    t_arr = np.asarray(log_t)
    mask = t_arr > (T_END - 0.1)
    i_d_ss = float(np.mean(np.asarray(log_id)[mask]))
    i_q_ss = float(np.mean(np.asarray(log_iq)[mask]))
    omega_ss = float(np.mean(np.asarray(log_omega)[mask]))
    torque_ss = float(np.mean(np.asarray(log_torque)[mask]))
    psi_ss = float(np.mean(np.asarray(log_psi)[mask]))

    print("\n  KPI (steady-state, last 100 ms):")
    print(f"    i_d        = {i_d_ss:+.3f} A   (target {I_D_REF})")
    print(f"    i_q        = {i_q_ss:+.3f} A   (target {I_Q_REF})")
    print(f"    |ψ_r|      = {psi_ss:.3f} Wb")
    print(f"    ω_m        = {omega_ss:.2f} rad/s "
          f"({omega_ss*60/(2*math.pi):.0f} rpm)")
    print(f"    T_em       = {torque_ss:+.3f} Nm")
    # Expected torque (textbook IFOC steady-state):
    # T = (3/2)·pp·(Lm²/Lr)·i_d·i_q
    expected_torque = 1.5 * PP * (L_m * L_m / kw["L_r"]) * I_D_REF * I_Q_REF
    print(f"    Expected T = {expected_torque:+.3f} Nm "
          f"(= 1.5·pp·(Lm²/Lr)·i_d_ref·i_q_ref)")
    print(f"    ω_sync     = {diag['omega_sync_rad_s']:.2f} rad/s")
    print(f"    steps      = {res.num_steps():,}")

    # CSV
    try:
        import csv
        out_path = Path(__file__).with_name("im_ifoc_trace.csv")
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "i_d_A", "i_q_A", "v_d_V", "v_q_V",
                            "omega_m_rad_s", "T_em_Nm", "psi_r_Wb"])
            for i in range(len(log_t)):
                w.writerow([log_t[i], log_id[i], log_iq[i],
                                log_vd[i], log_vq[i],
                                log_omega[i], log_torque[i], log_psi[i]])
        print(f"    trace      → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
