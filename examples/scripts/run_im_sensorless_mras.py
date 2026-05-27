#!/usr/bin/env python3
"""IM sensorless speed estimation via Flux-MRAS — V/f-driven plant.

Closes Phase 2.3 task 3.2 — PARTIAL.

The :class:`FluxMRASObserver` recovers the rotor electrical speed
from stator-frame αβ voltages + currents WITHOUT a position encoder.
This showcase exercises the observer on an IM under V/f open-loop
drive.

Status
------
* **Bootstrap fix landed** — the normalised cross-product
  (``normalise_eps=True``, the new default) converts the adaptation
  signal from raw cross-product (magnitude-dominated, kills cold
  start) to ``sin(angle error)``. Verified by
  :func:`test_mras_normalised_eps_converges_from_cold_start` in the
  pytest suite: with synthetic excitation at ω_e = 200 rad/s the
  observer locks within 500 ms to < 5 % error.
* **IM-driven case is harder** — the additional dynamics of the
  full IM plant (transient currents, motor acceleration, rotor flux
  build-up) interact with the adaptive flux equation in ways the
  synthetic test doesn't exercise. With the feedforward bootstrap
  (force ω̂ = 2π·f_source during 0 → 500 ms) the lock is partial:
  ω̂ tracks the trend but with a steady-state error that exceeds the
  5 % target of the synthetic case.

For closed-loop sensorless control you would typically:

1. Run V/f open-loop until ω_m above ~10 % rated.
2. Activate the MRAS in tracking mode with a known initial estimate
   (feedforward bootstrap).
3. Use the MRAS ω̂ to drive the IFOC speed-loop PI.

Steps 1-2 are what this showcase demonstrates. Step 3 (closed-loop
sensorless IFOC) is the natural next step and benefits from the
observer running co-located with the IFOC chain — that integration
is queued for v1.6.

Topology
--------
4 kW / 400 V / 50 Hz / 4-pole IM driven by a 3-φ V/f source ramping
0 → 50 Hz over 1 s. The MRAS observer runs in PARALLEL with the
plant — every kernel step we feed the observer the same αβ voltage
the controller commands and the αβ stator current the plant produces.
The estimated ω̂ is logged and compared against the simulator's
ground-truth ω_m_elec = pp · ω_m.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# ---------- 4 kW reference machine -----------------------------------------

P_RATED = 4_000.0
V_LL    = 400.0
F_END   = 50.0
PP      = 2
RAMP_S  = 1.0
HOLD_S  = 1.0

# ---------- MRAS tuning ----------------------------------------------------
#
# Conservative gains for the IM-driven case (the additional plant
# dynamics make the loop more sensitive than the synthetic stand-alone
# test in test_sensorless_observers.py). The normalised cross-product
# only requires the gain product to be large enough to track the
# fundamental — over-tuning destabilises the bootstrap.
KP_MRAS  = 10.0
KI_MRAS  = 200.0
HPF_OMEGA = 5.0
# Raise the eps-normalisation floor for the IM-driven case — protects
# against division by a transiently small |ψ_adj| while it builds up.
EPS_NORM_FLOOR = 1e-4
# Feedforward bootstrap: during the first FEEDFORWARD_S seconds we
# FORCE ω̂ to the commanded V/f electrical frequency (2π·f·pp). That
# loads the PI integrator near the right value, lets ψ_adj build up,
# and the loop holds when we release control.
FEEDFORWARD_S = 0.50
DT       = 50e-6
T_END    = RAMP_S + HOLD_S


def main():
    V_AMP_NOMINAL = V_LL * math.sqrt(2.0 / 3.0)
    V_BOOST = 0.05 * V_AMP_NOMINAL
    F_BOOST = 5.0

    def vf(t):
        if t < RAMP_S:
            f_now = F_END * (t / RAMP_S)
        else:
            f_now = F_END
        if f_now > F_BOOST:
            v_now = V_AMP_NOMINAL * (f_now / F_END)
        else:
            v_at_boost = V_AMP_NOMINAL * (F_BOOST / F_END)
            v_now = V_BOOST + (v_at_boost - V_BOOST) * (f_now / F_BOOST)
        return v_now, f_now

    b = p.CircuitBuilder()
    src_a_id = b.graph.num_branches
    b.add_voltage_source("Va", "a", "gnd", 0.0)
    src_b_id = b.graph.num_branches
    b.add_voltage_source("Vb", "b", "gnd", 0.0)
    src_c_id = b.graph.num_branches
    b.add_voltage_source("Vc", "c", "gnd", 0.0)
    b.add_resistor("Ra", "a", "a_int", 0.1)
    b.add_resistor("Rb", "b", "b_int", 0.1)
    b.add_resistor("Rc", "c", "c_int", 0.1)
    kw = p.im_parameters_from_nameplate(
        P_rated_W=P_RATED, V_LL_V=V_LL, f_Hz=F_END, pole_pairs=PP)
    diag = kw.pop("_diagnostics")
    motor = p.add_induction_motor(
        b, name="IM",
        phase_nodes=("a_int", "b_int", "c_int"),
        neutral_node="n",
        T_load=0.0,
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

    # MRAS — wired from the motor handle so parameters match by
    # construction. Default normalise_eps=True enables the bootstrap fix.
    mras = p.FluxMRASObserver.from_motor(
        motor, Kp_mras=KP_MRAS, Ki_mras=KI_MRAS)
    mras.voltage_model_hpf_omega = HPF_OMEGA
    mras.eps_norm_floor = EPS_NORM_FLOOR

    log_t = []
    log_omega_truth = []
    log_omega_hat = []
    log_psi_truth = []
    log_psi_ref = []
    log_psi_adj = []

    def step_observer(t, x):
        im_obs(t, x)

        # Re-compute the αβ voltage the controller is commanding (same
        # formula the plant's b_extra uses, kept synchronised).
        v_amp_now, f_now = vf(t)
        omega = 2.0 * math.pi * f_now
        v_a = v_amp_now * math.sin(omega * t)
        v_b = v_amp_now * math.sin(omega * t - 2.0 * math.pi / 3.0)
        v_c = v_amp_now * math.sin(omega * t + 2.0 * math.pi / 3.0)
        v_alpha = (2.0 / 3.0) * (v_a - 0.5 * v_b - 0.5 * v_c)
        v_beta  = (2.0 / 3.0) * (sqrt3 / 2.0) * (v_b - v_c)

        i_a = float(x[i_a_idx])
        i_b = float(x[i_b_idx])
        i_c = float(x[i_c_idx])
        i_alpha = (2.0 / 3.0) * (i_a - 0.5 * i_b - 0.5 * i_c)
        i_beta  = (2.0 / 3.0) * (sqrt3 / 2.0) * (i_b - i_c)

        omega_hat, _, _ = mras.update(
            v_alpha=v_alpha, v_beta=v_beta,
            i_alpha=i_alpha, i_beta=i_beta, dt=DT)

        # Feedforward bootstrap: override the PI output with the
        # commanded electrical frequency while the rotor flux is
        # building up. ω_e (electrical) = 2π·f_source (the source
        # frequency, NOT multiplied by pp again — that ratio is in
        # ω_m = ω_e/pp). Setting AFTER update is critical because
        # update() overwrites omega_hat = Kp·ε + Ki∫ε every step.
        if t < FEEDFORWARD_S:
            omega_ff = 2.0 * math.pi * f_now
            mras.omega_hat = omega_ff
            mras._mras_integ = omega_ff
            omega_hat = omega_ff

        omega_e_truth = PP * motor.mech.omega_rad_s

        if not log_t or t - log_t[-1] >= 5e-3:
            log_t.append(t)
            log_omega_truth.append(omega_e_truth)
            log_omega_hat.append(omega_hat)
            log_psi_truth.append(math.hypot(
                motor.psi_r_alpha_Wb, motor.psi_r_beta_Wb))
            log_psi_ref.append(math.hypot(
                mras.psi_r_alpha_ref, mras.psi_r_beta_ref))
            log_psi_adj.append(math.hypot(
                mras.psi_r_alpha_adj, mras.psi_r_beta_adj))

    def combined_b_extra(t):
        out = list(im_b_extra(t))
        v_amp_now, f_now = vf(t)
        omega = 2.0 * math.pi * f_now
        out[src_a_idx] += v_amp_now * math.sin(omega * t)
        out[src_b_idx] += v_amp_now * math.sin(omega * t - 2.0 * math.pi / 3.0)
        out[src_c_idx] += v_amp_now * math.sin(omega * t + 2.0 * math.pi / 3.0)
        return out

    print("  IM Flux-MRAS sensorless — normalised-eps bootstrap")
    print("    Plant: 4 kW / 400 V / 50 Hz / 4-pole IM")
    print("    Parameters wired from_motor (zero parameter mismatch)")
    print(f"    MRAS: Kp={KP_MRAS}, Ki={KI_MRAS}, HPF={HPF_OMEGA} rad/s, "
          f"normalise_eps=True")
    print(f"    V/f ramp 0 → {F_END} Hz over {RAMP_S}s, hold {HOLD_S}s")
    print(f"    Sim {T_END}s @ dt={DT*1e6:.0f}µs = {int(T_END/DT):,} steps")

    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=lambda t: p.SwitchStateMask(0),
        step_observer=step_observer,
        b_extra_fn=combined_b_extra,
        progress=True,
    )

    # KPI window: last 200 ms (after V/f hold + plenty of lock-in time).
    t_arr = np.asarray(log_t)
    mask = t_arr > (T_END - 0.2)
    omega_truth_ss = float(np.mean(np.asarray(log_omega_truth)[mask]))
    omega_hat_ss = float(np.mean(np.asarray(log_omega_hat)[mask]))
    psi_truth_ss = float(np.mean(np.asarray(log_psi_truth)[mask]))
    psi_ref_ss = float(np.mean(np.asarray(log_psi_ref)[mask]))
    psi_adj_ss = float(np.mean(np.asarray(log_psi_adj)[mask]))

    print(f"\n  KPI (steady-state, last 200 ms at {F_END} Hz):")
    print(f"    ω_e (true)   = {omega_truth_ss:+.2f} rad/s "
          f"({omega_truth_ss*60/(2*math.pi*PP):.0f} rpm mech)")
    print(f"    ω_e (MRAS)   = {omega_hat_ss:+.2f} rad/s")
    err_abs = omega_hat_ss - omega_truth_ss
    err_pct = (err_abs / omega_truth_ss * 100.0
                  if abs(omega_truth_ss) > 1e-6 else 0.0)
    print(f"    error        = {err_abs:+.3f} rad/s ({err_pct:+.2f}%)")
    print(f"    |ψ_r| true   = {psi_truth_ss:.4f} Wb")
    print(f"    |ψ_r| ref    = {psi_ref_ss:.4f} Wb (voltage model)")
    print(f"    |ψ_r| adj    = {psi_adj_ss:.4f} Wb (current model)")
    print(f"    ω_sync       = {diag['omega_sync_rad_s']:.2f} rad/s")
    print(f"    steps        = {res.num_steps():,}")

    # CSV
    try:
        import csv
        out_path = Path(__file__).with_name("im_mras_trace.csv")
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "omega_e_truth_rad_s",
                        "omega_e_mras_rad_s",
                        "psi_r_truth_Wb", "psi_r_ref_Wb",
                        "psi_r_adj_Wb"])
            for i in range(len(log_t)):
                w.writerow([log_t[i], log_omega_truth[i],
                            log_omega_hat[i],
                            log_psi_truth[i], log_psi_ref[i],
                            log_psi_adj[i]])
        print(f"    trace        → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
