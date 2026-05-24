#!/usr/bin/env python3
"""Three-phase DC/AC Modular Multilevel Converter — L0 average model.

Topology (thesis fig 2.7, Sousa 2022):

                       dc_pos
                         │
               ┌─────────┼─────────┐
               │         │         │
           [arm_a_p] [arm_b_p] [arm_c_p]   (upper arms)
               │         │         │
             L_a_p     L_b_p     L_c_p     (arm inductors)
               │         │         │
              ac_a      ac_b      ac_c     ──┐
               │         │         │         ├ Y-connected RL load
             L_a_n     L_b_n     L_c_n     ──┘  (R, L per phase)
               │         │         │
           [arm_a_n] [arm_b_n] [arm_c_n]   (lower arms)
               │         │         │
               └─────────┼─────────┘
                         │
                       dc_neg

Demonstrates Phase 20.4:
  * ``p.add_mmc_three_phase_dc_ac`` — six-arm topology in one call.
  * ``p.make_mmc_arms_observer`` — kernel co-sim hook.
  * Open-loop sinusoidal modulation with the standard formula
        m_b_p(t) = 0.5 + 0.5·M·sin(ωt + φ),
        m_b_n(t) = 1 − m_b_p(t)
    where ``M`` is the modulation depth.

Operating point (medium-voltage scale, intentionally modest to
make the run fast on commodity hardware):

  * V_dc       = 800 V        ; DC bus
  * N          = 10 SMs/arm   ; half-bridge
  * C_SM       = 1 mF         ; ⇒ C_arm = 100 µF
  * v_c0       = 800 V        ; matched to DC bus
  * L_b        = 1 mH         ; arm inductor
  * f_grid     = 50 Hz        ; AC fundamental
  * M          = 0.8          ; modulation depth
  * Y load     = 5 Ω + 5 mH   ; per phase
  * t_end      = 60 ms        ; 3 grid periods
  * dt         = 50 µs        ; 400 samples per period

Expected results:
  * Phase voltages: balanced 50 Hz, peak ≈ M·V_dc/2 ≈ 320 V.
  * Capacitor sums: stay near 800 V; each arm rides a small 50 Hz +
    2·50 Hz ripple per Sousa eq (2.43); peak-to-peak ≈ a few V.
  * Mean DC drift over 3 periods should be ≪ 1 % thanks to the
    balanced top/bottom modulation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# =============================================================================
# Operating point
# =============================================================================

V_DC   = 800.0           # DC bus voltage [V]
N_SM   = 10              # submodules per arm
C_SM   = 20e-3           # per-SM capacitance [F] — sized for ~5 % v_C
                         # ripple at 50 Hz per Sousa eq (2.43)
V_C0   = V_DC            # initial cap-sum voltage [V]
L_B    = 5e-3            # arm inductance [H] — limits circulating-current
F_AC   = 50.0            # AC fundamental [Hz]
M      = 0.8             # modulation depth ∈ [0, 1]
R_LOAD = 20.0            # per-phase load resistance [Ω]
L_LOAD = 5e-3            # per-phase load inductance [H]
T_END  = 80e-3           # simulation horizon [s] — 4 periods (~1 settled)
DT     = 50e-6           # time step [s]


# =============================================================================
# Modulation signals
# =============================================================================

OMEGA = 2.0 * math.pi * F_AC


def m_a_p(t: float) -> float:
    return 0.5 + 0.5 * M * math.sin(OMEGA * t)


def m_b_p(t: float) -> float:
    return 0.5 + 0.5 * M * math.sin(OMEGA * t - 2.0 * math.pi / 3.0)


def m_c_p(t: float) -> float:
    return 0.5 + 0.5 * M * math.sin(OMEGA * t + 2.0 * math.pi / 3.0)


# =============================================================================
# Build + simulate
# =============================================================================

def build_plant() -> "tuple[p.CircuitBuilder, p.MmcThreePhaseDcAc, str]":
    b = p.CircuitBuilder()

    # DC bus.
    b.add_voltage_source("Vdc", "dc_p", "dc_n", V_DC)

    # Six arms + six arm inductors.
    mmc = p.add_mmc_three_phase_dc_ac(
        b,
        dc_pos="dc_p", dc_neg="dc_n",
        ac_nodes=("ac_a", "ac_b", "ac_c"),
        n_sm=N_SM, c_sm=C_SM, l_b=L_B,
        sm_type="half_bridge",
        v_c0=V_C0,
        m_signals=(m_a_p, m_b_p, m_c_p),
    )

    # Y-connected RL load between each phase and a common star point.
    star = "load_star"
    for phase in ("a", "b", "c"):
        b.add_resistor(f"R_{phase}", f"ac_{phase}", f"r{phase}_mid", R_LOAD)
        b.add_inductor(f"L_{phase}", f"r{phase}_mid", star, L_LOAD)

    # Tie the star point to the DC bus midpoint via a huge resistor —
    # just to ensure MNA has a path to ground (the v-source already
    # references everything to gnd through dc_n).
    b.add_resistor("R_star_ref", star, "dc_n", 1e6)

    return b, mmc, star


def run() -> "dict[str, np.ndarray]":
    b, mmc, _ = build_plant()
    obs, bex = p.make_mmc_arms_observer(b, mmc.all_arms, dt=DT)

    # Log v_C of all six arms + the three AC node voltages.
    # ``mmc.all_arms`` is a list[MmcArmAverage] — we capture v_C from
    # each in the observer wrapper.
    n_samples = int(round(T_END / DT)) + 1
    time = np.zeros(n_samples)
    v_C_log = np.zeros((6, n_samples))
    # We can't easily read ac_a/b/c node voltages from the state
    # vector without knowing the node-index → state-index map; for
    # the showcase we'll reconstruct v_phase ≈ v_arm_p - V_dc/2 (i.e.
    # the modulated arm voltage minus the DC midpoint).
    v_arm_p_log = np.zeros((3, n_samples))

    counter = [0]

    def logging_obs(t, x):
        obs(t, x)
        i = counter[0]
        if i < n_samples:
            time[i] = t
            for k, arm in enumerate(mmc.all_arms):
                v_C_log[k, i] = arm.v_C
            for k, arm in enumerate(mmc.upper_arms):
                v_arm_p_log[k, i] = arm.v_b
        counter[0] += 1

    p.simulate(
        b, t_end=T_END, dt=DT,
        step_observer=logging_obs, b_extra_fn=bex,
        start_from_dc_op=True,
    )

    return {
        "t": time[:counter[0]],
        "v_C": v_C_log[:, :counter[0]],
        "v_arm_p": v_arm_p_log[:, :counter[0]],
    }


# =============================================================================
# Reporting
# =============================================================================

def report(data: "dict[str, np.ndarray]") -> None:
    t = data["t"]
    v_C = data["v_C"]
    v_arm_p = data["v_arm_p"]

    # Phase voltage seen by the load = v_arm_p - V_dc/2 + (no L_b drop
    # at steady state). The first half-period is transient.
    v_phase = v_arm_p - 0.5 * V_DC

    # Last grid period — the steady-state window we report on.
    last_period_mask = t >= (T_END - 1.0 / F_AC)
    n_last = int(np.sum(last_period_mask))

    print("=" * 72)
    print("MMC three-phase DC/AC — L0 average model")
    print("=" * 72)
    print(f"  V_dc           = {V_DC:.1f} V")
    print(f"  N_SM           = {N_SM}")
    print(f"  C_SM           = {C_SM*1e3:.1f} mF   "
          f"(C_arm = {C_SM/N_SM*1e6:.1f} µF)")
    print(f"  L_b            = {L_B*1e3:.1f} mH")
    print(f"  R_load, L_load = {R_LOAD:.1f} Ω, {L_LOAD*1e3:.1f} mH")
    print(f"  f_AC           = {F_AC:.1f} Hz")
    print(f"  Modulation M   = {M:.2f}")
    print(f"  t_end          = {T_END*1e3:.1f} ms  ({T_END * F_AC:.1f} periods)")
    print(f"  dt             = {DT*1e6:.1f} µs")
    print(f"  Samples        = {len(t)}  (last period: {n_last})")
    print()

    # v_C statistics — should stay near V_C0.
    print("Capacitor-sum voltage (last grid period):")
    print(f"  {'arm':<10} {'mean [V]':>12} {'pk-pk [V]':>12} "
          f"{'drift [V]':>12}")
    arm_names = ("a_p", "b_p", "c_p", "a_n", "b_n", "c_n")
    for k, name in enumerate(arm_names):
        v_last = v_C[k, last_period_mask]
        mean = float(v_last.mean())
        pkpk = float(v_last.max() - v_last.min())
        drift = mean - V_C0
        print(f"  arm_{name:<6} {mean:>12.3f} {pkpk:>12.4f} {drift:>+12.4f}")
    print()

    # Phase voltage amplitudes.
    print("AC-side modulated voltage v_arm_p − V_dc/2 (last period):")
    print(f"  Expected peak amplitude ≈ M · V_dc / 2 "
          f"= {M * V_DC / 2:.1f} V")
    for k, phase in enumerate("abc"):
        v_p_last = v_phase[k, last_period_mask]
        peak = float(np.abs(v_p_last).max())
        print(f"  phase_{phase}: peak |v_arm_p − V_dc/2| = {peak:.2f} V")
    print()


def main() -> None:
    print("Building circuit + running 60 ms simulation...")
    data = run()
    report(data)

    # Plot if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: capacitor-sum voltages.
    arm_labels = ("a_p", "b_p", "c_p", "a_n", "b_n", "c_n")
    for k, label in enumerate(arm_labels):
        axes[0].plot(data["t"] * 1e3, data["v_C"][k],
                         label=f"arm_{label}", linewidth=0.8)
    axes[0].axhline(V_C0, color="k", linestyle="--",
                       alpha=0.4, label=f"v_c0 = {V_C0} V")
    axes[0].set_ylabel("v_C [V]")
    axes[0].set_title("Arm capacitor-sum voltages")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Bottom: AC-side modulated voltages.
    v_phase = data["v_arm_p"] - 0.5 * V_DC
    for k, phase in enumerate("abc"):
        axes[1].plot(data["t"] * 1e3, v_phase[k],
                         label=f"phase_{phase}", linewidth=1.0)
    axes[1].set_xlabel("time [ms]")
    axes[1].set_ylabel("v_arm_p − V_dc/2 [V]")
    axes[1].set_title("AC-side modulated voltage (per phase)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    out = Path(__file__).resolve().parent / "out" / "mmc_three_phase_dc_ac.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
