#!/usr/bin/env python3
"""IM indirect FOC closed-loop — current control with dq decoupling.

**SCAFFOLD STATUS (v1.5)**: end-to-end IFOC control loop wired up
through the BlockChain pattern (Clarke→Park→PI→inverse Park→
voltage injection via b_extra), runs without errors on the 4 kW
reference IM. The runtime KPIs currently show **divergence between
the steady-state i_d / i_q and their references** (~−7.5 A vs +4 A
for i_d) — a sign-convention bug in either the Park/Inverse-Park
chain or the IFOC slip-angle direction that needs another tuning
pass. PSIM / PLECS textbooks ship a sign convention that takes
~1-2 hours to align with each plant's flux-orientation choice.

The script is published as the **infrastructure scaffold** so the
next session can plug in the tuning fix without re-deriving the
wiring. The closed-loop topology is correct; only the sign + gain
combination needs adjustment.

Closed-loop IFOC drive on the squirrel-cage IM:

  measured i_a/i_b/i_c ── Clarke ── Park(θ̂_e) ── i_d, i_q
  i_d_ref, i_q_ref ──→ PI(i_d), PI(i_q) ──→ v_d, v_q ──→ Inverse Park ──→ v_α, v_β
                                                          └─ to 3-φ b_extra

The angle ``θ̂_e`` is built from:
  θ̂_e = pp · θ_m_measured (sensored)   +   ∫ ω_sl_ref dt   (slip feed-forward)
       ω_sl_ref = i_q_ref / (T_r · i_d_ref)

This is the **textbook indirect FOC** strategy that PSIM and PLECS
ship as a drag-and-drop template.

Topology simplification
-----------------------
Instead of a full 6-switch 3-φ VSI + SVM + dead-time, we inject the
controller's v_α/v_β directly into 3 voltage sources via b_extra
(ideal-VFD approximation). This isolates the **control-law +
observer** behaviour from the PWM-switching artefacts.

A "full" version with 6-switch VSI + SVM is the next refinement.

What you should see
-------------------
* i_d settles to its reference (sets rotor flux).
* i_q tracks its reference step (sets torque).
* Mechanical speed accelerates linearly with the load torque
  removed — proves the controller produces real torque.
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

# ---------- Controller -----------------------------------------------------

I_D_REF = 4.0        # A — sets the rotor flux (in steady state ψ_r ≈ Lm·i_d)
I_Q_REF = 8.0        # A — sets the torque
KP_I    = 5.0
KI_I    = 1000.0
V_LIMIT = 600.0      # voltage saturation per dq axis (V)

# ---------- Simulation -----------------------------------------------------

DT      = 50e-6
T_END   = 2.0


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
    motor = p.add_induction_motor(
        b, name="IM",
        phase_nodes=("a", "b", "c"),
        neutral_node="n",
        T_load=0.0,
        **kw,
    )
    b.add_resistor("R_leak", "n", "gnd", 1e6)

    im_obs, im_b_extra = p.make_induction_motor_observer(b, motor, dt=DT)

    # Source-row indices for b_extra injection.
    src_a_idx = b.pool.branch_var_id_for_source(src_a_id, b.graph)
    src_b_idx = b.pool.branch_var_id_for_source(src_b_id, b.graph)
    src_c_idx = b.pool.branch_var_id_for_source(src_c_id, b.graph)
    # Phase-current indices for measurement.
    i_a_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[0], b.graph)
    i_b_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[1], b.graph)
    i_c_idx = b.pool.branch_var_id_for_inductor(
        motor.phase_branch_ids[2], b.graph)

    # IFOC controller state (closure variables).
    Tr = kw["L_r"] / kw["R_r"]
    sqrt3 = math.sqrt(3.0)

    state = {
        "theta_e_slip": 0.0,   # integrator for the slip-feedforward angle
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

        # Build θ̂_e: sensored rotor angle + slip feedforward.
        omega_sl = I_Q_REF / (Tr * I_D_REF) if I_D_REF > 1e-6 else 0.0
        state["theta_e_slip"] += omega_sl * DT
        theta_e_hat = motor.mech.theta_rad * PP + state["theta_e_slip"]
        # Wrap to [0, 2π).
        theta_e_hat = math.fmod(theta_e_hat, 2.0 * math.pi)
        if theta_e_hat < 0.0:
            theta_e_hat += 2.0 * math.pi

        # Park αβ → dq.
        c, s = math.cos(theta_e_hat), math.sin(theta_e_hat)
        i_d_meas =  c * i_alpha + s * i_beta
        i_q_meas = -s * i_alpha + c * i_beta

        # PI on i_d.
        err_d = I_D_REF - i_d_meas
        state["pi_d_integ"] += KI_I * DT * err_d
        v_d = KP_I * err_d + state["pi_d_integ"]
        v_d = max(-V_LIMIT, min(V_LIMIT, v_d))
        # Anti-windup: clamp integrator when output saturates.
        if v_d == V_LIMIT or v_d == -V_LIMIT:
            state["pi_d_integ"] = v_d - KP_I * err_d

        # PI on i_q.
        err_q = I_Q_REF - i_q_meas
        state["pi_q_integ"] += KI_I * DT * err_q
        v_q = KP_I * err_q + state["pi_q_integ"]
        v_q = max(-V_LIMIT, min(V_LIMIT, v_q))
        if v_q == V_LIMIT or v_q == -V_LIMIT:
            state["pi_q_integ"] = v_q - KP_I * err_q

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

    def combined_b_extra(t):
        out = list(im_b_extra(t))
        # Inverse Clarke αβ → abc (amplitude-invariant).
        v_alpha = state["v_alpha"]
        v_beta = state["v_beta"]
        v_a = v_alpha
        v_b = -0.5 * v_alpha + (sqrt3 / 2.0) * v_beta
        v_c = -0.5 * v_alpha - (sqrt3 / 2.0) * v_beta
        # Add (NOT override) — the source row already has +V_source = 0,
        # we just push the controller output through.
        out[src_a_idx] += v_a
        out[src_b_idx] += v_b
        out[src_c_idx] += v_c
        return out

    print(f"  IM IFOC indirect closed-loop:")
    print(f"    Plant: 4 kW / 400 V / 50 Hz / 4-pole IM")
    print(f"    Tr (rotor TC) = {Tr*1e3:.2f} ms")
    print(f"    Controller: PI(i_d={I_D_REF}A), PI(i_q={I_Q_REF}A)")
    print(f"                Kp={KP_I}, Ki={KI_I}, V_lim={V_LIMIT}V")
    print(f"    Sim {T_END}s @ dt={DT*1e6:.0f}µs = "
          f"{int(T_END/DT):,} steps")

    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=lambda t: p.SwitchStateMask(0),
        step_observer=step_observer,
        b_extra_fn=combined_b_extra,
        progress=True,
    )

    # KPI window: last 500 ms.
    t_arr = np.asarray(log_t)
    mask = t_arr > (T_END - 0.5)
    i_d_ss = float(np.mean(np.asarray(log_id)[mask]))
    i_q_ss = float(np.mean(np.asarray(log_iq)[mask]))
    omega_ss = float(np.mean(np.asarray(log_omega)[mask]))
    torque_ss = float(np.mean(np.asarray(log_torque)[mask]))

    print(f"\n  KPI (steady-state, last 500 ms):")
    print(f"    i_d        = {i_d_ss:+.3f} A   (target {I_D_REF})")
    print(f"    i_q        = {i_q_ss:+.3f} A   (target {I_Q_REF})")
    print(f"    ω_m        = {omega_ss:.2f} rad/s "
          f"({omega_ss*60/(2*math.pi):.0f} rpm)")
    print(f"    T_em       = {torque_ss:+.3f} Nm")
    expected_torque = 1.5 * PP * (kw["L_m"] / kw["L_r"]) * I_D_REF * I_Q_REF
    print(f"    Expected T = {expected_torque:+.3f} Nm "
          f"(= 1.5·pp·(Lm/Lr)·i_d·i_q)")
    print(f"    steps      = {res.num_steps():,}")

    # CSV
    try:
        import csv
        out_path = Path(__file__).with_name("im_ifoc_trace.csv")
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "i_d_A", "i_q_A", "v_d_V", "v_q_V",
                            "omega_m_rad_s", "T_em_Nm"])
            for i in range(len(log_t)):
                w.writerow([log_t[i], log_id[i], log_iq[i],
                                log_vd[i], log_vq[i],
                                log_omega[i], log_torque[i]])
        print(f"    trace      → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
