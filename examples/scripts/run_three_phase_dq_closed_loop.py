#!/usr/bin/env python3
"""3-φ VSI with dq current control — kernel stress test #2.

   V_DC (600 V) → 6-switch inverter → L filter (5 mH/phase)
                                      → Y-connected R load (5 Ω/phase to neutral)

The controller reads i_a, i_b, i_c from the three filter inductors,
transforms to the synchronous dq frame (θ = ω·t at 50 Hz, no PLL —
this is island mode driving a passive load), and runs two PI loops:
i_d → active current setpoint, i_q → 0 (unity PF).

The dq output drives SPWM modulation references (m_a, m_b, m_c)
per leg via inverse Park + inverse Clarke. We use a custom
`switch_fn(t)` that synthesizes the carrier-vs-reference comparison
inline so the modulation can change every dt.

Stress points for the kernel:
  - **6-switch enumeration** (64 reachable cache masks; lazy build).
  - **Phase-shifted modulation refs** — per-leg duties change
    independently every step. Three independent PWM bits flipping
    out of phase.
  - **Two PI controllers** + abc-dq transform every step.
  - **Y-connected load** — neutral floats; small ohmic tie to gnd
    means the matrix is ill-conditioned. KLU stress.

Setpoint sequence:
  0–10 ms: i_d_ref = 5 A, i_q_ref = 0  (turn on, settle)
  10 ms:   i_d_ref steps to 10 A         (doubles the active current)
  20 ms:   i_q_ref steps to -3 A         (adds capacitive reactive current)
"""

from __future__ import annotations

import math
from math import pi as PI, sqrt
from pathlib import Path

import numpy as np

import pulsim as p


# =============================================================================
# Plant + grid + controller config
# =============================================================================
V_DC      = 600.0
F_OUT     = 50.0                 # 50 Hz output fundamental
F_CARRIER = 5000.0               # 5 kHz carrier (10× lower than usual,
                                  # makes the dq dynamics visible in 25 ms)
T_CARR    = 1.0 / F_CARRIER
DT        = 5.0e-6               # 5 µs → 200 samples/cycle
T_END     = 25.0e-3

R_PHASE   = 5.0
L_PHASE   = 5.0e-3

I_D_REF_1 = 5.0
I_D_REF_2 = 10.0
I_Q_REF_1 = 0.0
I_Q_REF_2 = -3.0
T_STEP_D  = 10.0e-3
T_STEP_Q  = 20.0e-3

# PI tuning for current loop (decoupled per axis — bandwidth ~500 Hz).
# Aggressive: pushes the kernel through fast duty changes every step.
KP_I = 2.0
KI_I = 2000.0
M_MAX = 0.95           # modulation depth limit


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vbus", "vbus", "gnd", V_DC)

    # 6 controlled switches (insertion order = bit order).
    for name, frm, to in [
        ("HS_A", "vbus", "mid_a"), ("LS_A", "mid_a", "gnd"),
        ("HS_B", "vbus", "mid_b"), ("LS_B", "mid_b", "gnd"),
        ("HS_C", "vbus", "mid_c"), ("LS_C", "mid_c", "gnd"),
    ]:
        b.add_switch(name, frm, to, g_on=1e3, g_off=1e-9)

    # 6 body diodes — auto-commutated.
    for name, anode, cathode in [
        ("D_HS_A", "mid_a", "vbus"), ("D_LS_A", "gnd", "mid_a"),
        ("D_HS_B", "mid_b", "vbus"), ("D_LS_B", "gnd", "mid_b"),
        ("D_HS_C", "mid_c", "vbus"), ("D_LS_C", "gnd", "mid_c"),
    ]:
        b.add_diode(name, anode, cathode, 1e3, 1e-9)

    # Y-connected RL load per phase.
    for phase in ("a", "b", "c"):
        b.add_resistor(f"R_{phase}",
                        f"mid_{phase}", f"r{phase}_lend", R_PHASE)
        b.add_inductor(f"L_{phase}",
                        f"r{phase}_lend", "n", L_PHASE)
    b.add_resistor("R_neutral", "n", "gnd", 1.0e-3)
    return b


def clarke(a: float, b: float, c: float) -> tuple[float, float]:
    """abc → αβ (power-invariant)."""
    alpha = (2.0/3.0) * (a - 0.5*b - 0.5*c)
    beta  = (2.0/3.0) * ((sqrt(3)/2)*b - (sqrt(3)/2)*c)
    return alpha, beta


def park(alpha: float, beta: float, theta: float) -> tuple[float, float]:
    """αβ → dq."""
    c, s = math.cos(theta), math.sin(theta)
    d =  c*alpha + s*beta
    q = -s*alpha + c*beta
    return d, q


def inv_park(d: float, q: float, theta: float) -> tuple[float, float]:
    """dq → αβ."""
    c, s = math.cos(theta), math.sin(theta)
    alpha = c*d - s*q
    beta  = s*d + c*q
    return alpha, beta


def inv_clarke(alpha: float, beta: float) -> tuple[float, float, float]:
    """αβ → abc (back-transform of the 2/3 Clarke)."""
    a = alpha
    b = -0.5*alpha + (sqrt(3)/2)*beta
    c = -0.5*alpha - (sqrt(3)/2)*beta
    return a, b, c


def main() -> None:
    builder = build_plant()
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")

    # Inductor branches: L_a, L_b, L_c are branches 13, 15, 17 in
    # insertion order (1 Vsrc + 6 switches + 6 diodes + 3 (R,L)
    # pairs + R_neutral). Use the API to be robust.
    # Branch order from CircuitBuilder: Vbus(0) → 6 switches(1..6)
    # → 6 diodes(7..12) → R_a(13), L_a(14), R_b(15), L_b(16),
    # R_c(17), L_c(18), R_neutral(19).
    iLa_idx = builder.pool.branch_var_id_for_inductor(14, builder.graph)
    iLb_idx = builder.pool.branch_var_id_for_inductor(16, builder.graph)
    iLc_idx = builder.pool.branch_var_id_for_inductor(18, builder.graph)
    print(f"  iLa_idx={iLa_idx}, iLb_idx={iLb_idx}, iLc_idx={iLc_idx}")

    # Two PI controllers (one per dq axis).
    pi_d = p.PIController(Kp=KP_I, Ki=KI_I,
                            output_min=-V_DC/2, output_max=V_DC/2)
    pi_q = p.PIController(Kp=KP_I, Ki=KI_I,
                            output_min=-V_DC/2, output_max=V_DC/2)

    # Mutable refs read by switch_fn.
    mod_refs = [0.0, 0.0, 0.0]   # m_a, m_b, m_c ∈ [-1, +1]

    def i_d_ref(t):
        return I_D_REF_1 if t < T_STEP_D else I_D_REF_2

    def i_q_ref(t):
        return I_Q_REF_1 if t < T_STEP_Q else I_Q_REF_2

    def observe(t: float, x) -> None:
        i_a = float(x[iLa_idx])
        i_b = float(x[iLb_idx])
        i_c = float(x[iLc_idx])
        theta = 2.0 * PI * F_OUT * t

        # abc → dq
        i_alpha, i_beta = clarke(i_a, i_b, i_c)
        i_d, i_q = park(i_alpha, i_beta, theta)

        # PI per axis (parallel, no cross-coupling decoupling for simplicity).
        v_d = pi_d.update(setpoint=i_d_ref(t), measured=i_d, dt=DT)
        v_q = pi_q.update(setpoint=i_q_ref(t), measured=i_q, dt=DT)

        # dq → abc (just math — no kernel impact)
        v_alpha, v_beta = inv_park(v_d, v_q, theta)
        m_a, m_b, m_c = inv_clarke(v_alpha, v_beta)

        # Normalize voltage reference → modulation index ∈ [-M_MAX, M_MAX].
        scale = 2.0 / V_DC
        mod_refs[0] = max(-M_MAX, min(M_MAX, m_a * scale))
        mod_refs[1] = max(-M_MAX, min(M_MAX, m_b * scale))
        mod_refs[2] = max(-M_MAX, min(M_MAX, m_c * scale))

    num_switches = builder.graph.num_switches

    def switch_fn(t: float):
        # Single carrier sawtooth from 0 to 1 over T_CARR.
        carrier = math.fmod(t * F_CARRIER, 1.0)
        # Each leg's duty = (1 + m) / 2 — symmetric PWM.
        duty_a = 0.5 + 0.5 * mod_refs[0]
        duty_b = 0.5 + 0.5 * mod_refs[1]
        duty_c = 0.5 + 0.5 * mod_refs[2]

        m = p.SwitchStateMask(num_switches)
        # Bit 0 = HS_A, bit 1 = LS_A, etc.
        if carrier < duty_a: m.set(0, True)
        else:                m.set(1, True)
        if carrier < duty_b: m.set(2, True)
        else:                m.set(3, True)
        if carrier < duty_c: m.set(4, True)
        else:                m.set(5, True)
        return m

    print(f"\n  3-φ dq current control:")
    print(f"    PI(Kp={KP_I}, Ki={KI_I}), carrier={F_CARRIER/1e3:.0f} kHz")
    print(f"    setpoint i_d: {I_D_REF_1} A → {I_D_REF_2} A at "
          f"t = {T_STEP_D*1e3:.0f} ms")
    print(f"    setpoint i_q: {I_Q_REF_1} A → {I_Q_REF_2} A at "
          f"t = {T_STEP_Q*1e3:.0f} ms")
    res = p.simulate(
        builder, t_end=T_END, dt=DT,
        switch_fn=switch_fn,
        step_observer=observe,
        max_event_iterations=12,
    )
    print(f"  samples: {res.num_steps()}")

    # Reconstruct dq from the recorded states for plotting.
    times = np.asarray(res.times) * 1e3   # ms
    i_a_arr = np.array([s[iLa_idx] for s in res.states])
    i_b_arr = np.array([s[iLb_idx] for s in res.states])
    i_c_arr = np.array([s[iLc_idx] for s in res.states])

    theta_arr = 2.0 * PI * F_OUT * np.asarray(res.times)
    cos_t, sin_t = np.cos(theta_arr), np.sin(theta_arr)
    i_alpha = (2.0/3.0) * (i_a_arr - 0.5*i_b_arr - 0.5*i_c_arr)
    i_beta  = (2.0/3.0) * ((sqrt(3)/2)*i_b_arr - (sqrt(3)/2)*i_c_arr)
    i_d_arr =  cos_t*i_alpha + sin_t*i_beta
    i_q_arr = -sin_t*i_alpha + cos_t*i_beta

    i_d_ref_arr = np.array([i_d_ref(t) for t in res.times])
    i_q_ref_arr = np.array([i_q_ref(t) for t in res.times])

    pre_d  = i_d_arr[(times > 5)  & (times < T_STEP_D*1e3)].mean()
    post_d = i_d_arr[(times > T_STEP_D*1e3 + 3) & (times < T_STEP_Q*1e3)].mean()
    post_q = i_q_arr[(times > T_STEP_Q*1e3 + 3)].mean()
    print(f"\n  KPI:")
    print(f"    i_d  pre-step  (5–{T_STEP_D*1e3:.0f} ms): mean = {pre_d:.2f} A "
          f"(target {I_D_REF_1:.1f})")
    print(f"    i_d  post-step ({T_STEP_D*1e3+3:.0f}–{T_STEP_Q*1e3:.0f} ms): "
          f"mean = {post_d:.2f} A (target {I_D_REF_2:.1f})")
    print(f"    i_q  post-step (>{T_STEP_Q*1e3+3:.0f} ms): "
          f"mean = {post_q:.2f} A (target {I_Q_REF_2:.1f})")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_abc, ax_dq) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_abc.plot(times, i_a_arr, label="i_a", color="tab:red", lw=0.5)
    ax_abc.plot(times, i_b_arr, label="i_b", color="tab:green", lw=0.5)
    ax_abc.plot(times, i_c_arr, label="i_c", color="tab:blue", lw=0.5)
    ax_abc.set_ylabel("phase current [A]"); ax_abc.grid(alpha=0.3)
    ax_abc.legend(loc="upper right", ncol=3)
    ax_abc.set_title("3-φ VSI dq current control — passive Y-connected RL load")

    ax_dq.plot(times, i_d_arr, label="i_d (measured)", color="tab:orange", lw=0.7)
    ax_dq.plot(times, i_d_ref_arr, ls="--", lw=0.7, color="tab:orange",
                alpha=0.5, label="i_d_ref")
    ax_dq.plot(times, i_q_arr, label="i_q (measured)", color="tab:purple", lw=0.7)
    ax_dq.plot(times, i_q_ref_arr, ls="--", lw=0.7, color="tab:purple",
                alpha=0.5, label="i_q_ref")
    ax_dq.set_xlabel("time [ms]"); ax_dq.set_ylabel("dq current [A]")
    ax_dq.grid(alpha=0.3); ax_dq.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "three_phase_dq_closed_loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
