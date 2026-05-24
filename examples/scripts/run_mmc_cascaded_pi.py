#!/usr/bin/env python3
"""Closed-loop MMC buck-mode regulator — cascaded PI feeding the L1
PS-PWM modulator.

  V_dc ── [MMC arm] ── L_filt ── v_out ── C_out ── R_load (→ gnd)
            ↑ multilevel switching cell (8 SMs, PS-PWM)            │
            └────────────────────────────────────────────── gnd ──┘

Control architecture:

  V_out_ref ──┐
              ▼
   outer PI ──┤  → I_L_ref
              │
  V_out_meas ─┘

  I_L_ref ────┐
              ▼
   inner PI ──┤  → v_arm_demand
              │
  I_L_meas ───┘

  m_ref = v_arm_demand / v_C  → L1 PS-PWM → s_b/N → arm voltage staircase

The L1 modulator quantizes the inner-loop ``m_ref`` into a discrete
``s_b ∈ {0..N}`` via phase-shifted PWM. The effective arm-side
switching frequency is ``N · f_carrier = 8 kHz`` here; the
``L_filt + C_out`` tank sits at ~500 Hz so the staircase is filtered
cleanly.

A setpoint step at t = 60 ms (400 → 500 V) demonstrates the loop's
disturbance-rejection bandwidth — the outer loop ramps the current
reference, the inner loop drives the modulation up to match, and the
output settles to the new target within ~10 ms.

**v_C drift is intentional and physically meaningful.** A half-bridge
arm in *buck-mode* DC operation carries only positive arm current
(no regenerative path), so the modulator average
``m_avg ≈ (V_dc − V_out) / v_C`` integrates against ``i_L > 0`` and
the cap charges throughout the run. The plot makes this drift
visible — it is exactly why real MMC topologies pair a *complementary
lower arm* that sinks current during the second half-cycle (or why
sustained AC operation works: ``m·i`` averages to zero when the two
are in quadrature). For this single-arm demo the inner PI tracks
``i_L`` cleanly and the outer PI keeps ``V_out`` on setpoint despite
``v_C`` rising — that's the central point.

Run knobs:
  * ``V_OUT_REF_PRE``  → ``V_OUT_REF_POST`` setpoint step at ``T_STEP``
    exercises the closed-loop bandwidth (~50 Hz outer / ~500 Hz inner).
  * ``N_SM`` & ``F_CARRIER`` set the multilevel resolution and the
    effective ``f_switch = N·F_CARRIER`` of the modulator.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# =============================================================================
# Plant
# =============================================================================

V_DC      = 1000.0           # DC bus [V]
N_SM      = 8                # SMs per arm
C_SM      = 5e-3             # per-SM cap [F] → C_arm = 625 µF
V_C0      = 1000.0           # matched to V_dc
F_CARRIER = 1000.0           # PS-PWM carrier per SM [Hz]

L_FILT    = 1e-3             # buck filter L [H]
C_OUT     = 100e-6           # buck filter C [F]
R_LOAD    = 8.0              # output load [Ω] → I_L ≈ 50 A at steady state

DT        = 10e-6            # 10 µs sim step
T_END     = 100e-3           # 100 ms total
T_STEP    = 60e-3            # setpoint step at 60 ms

V_OUT_REF_PRE  = 400.0
V_OUT_REF_POST = 500.0


# =============================================================================
# Controller
# =============================================================================

# NOTE on plant polarity. The arm voltage ``v_arm = m_ref · v_C`` is in
# series with the buck switch node, so ``V_mid = V_dc - v_arm`` and
# higher ``m_ref`` ⇒ *lower* ``V_mid`` ⇒ *lower* ``V_out``. We give the
# inner PI the conventional "buck duty" ``d`` ∈ [0, 1] semantics (more
# duty ⇒ more output) and convert at the modulator boundary:
#   ``m_ref = 1 - d`` (since ``v_arm = (1-d)·v_C`` puts ``V_mid = d·V_dc``).
# Both PI gains are then positive in the natural sense.

# Outer voltage loop — slow (~50 Hz crossover).
KP_OUTER  = 0.8              # A/V
KI_OUTER  = 80.0             # A/(V·s)
I_L_MIN   = 0.0
I_L_MAX   = 100.0

# Inner current loop — fast (~500 Hz crossover).
# Plant: di/dt ≈ V_dc · d / L  ⇒ DC gain ~1000 A per unit duty.
# Pick Kp small so a 50 A error doesn't saturate the loop.
KP_INNER  = 0.005            # duty / A
KI_INNER  = 5.0              # duty / (A·s)


def make_setpoint(t_step: float,
                  v_pre: float,
                  v_post: float):
    def setpoint(t: float) -> float:
        return v_pre if t < t_step else v_post
    return setpoint


# =============================================================================
# Build + simulate
# =============================================================================

def build_plant() -> "tuple[p.CircuitBuilder, p.MmcArmMultilevel, list[float]]":
    """Plant + a [mutable] holder of the m_ref command.

    The arm reads its modulation index from the holder (passed in as a
    callable). The closed-loop observer writes to the holder each step.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", V_DC)

    # m_ref starts at the open-loop steady-state estimate:
    #   m_avg ≈ (V_dc - V_out_ref) / V_C0 → for 1000 V/400 V → 0.6
    m_ref_initial = (V_DC - V_OUT_REF_PRE) / V_C0
    m_ref_holder = [m_ref_initial]

    params = p.MmcArmMultilevelParams(
        n_sm=N_SM, c_sm=C_SM, v_c0=V_C0, f_carrier=F_CARRIER,
    )
    arm = p.add_mmc_arm_multilevel(
        b, name="ARM", node_a="vdc", node_b="mid",
        params=params,
        m_ref=lambda _t: m_ref_holder[0],
    )

    b.add_inductor("L_filt", "mid",  "vout", L_FILT)
    b.add_capacitor("C_out",  "vout", "gnd",  C_OUT)
    b.add_resistor("R_load",  "vout", "gnd",  R_LOAD)

    return b, arm, m_ref_holder


def main() -> None:
    b, arm, m_ref_holder = build_plant()
    vout_idx = b.node_id_of("vout")
    iL_idx   = b.pool.branch_var_id_for_inductor(
        b.graph.num_branches - 3, b.graph,  # L_filt is 3 branches back
    )
    # Compute the L_filt branch ID more carefully.
    # branches were added in order: Vdc, arm_Varm, L_filt, C_out, R_load
    # So L_filt is branch index 2.
    iL_idx = b.pool.branch_var_id_for_inductor(2, b.graph)

    # Outer + inner PI.
    outer = p.PIController(
        Kp=KP_OUTER, Ki=KI_OUTER,
        output_min=I_L_MIN, output_max=I_L_MAX,
    )
    inner = p.PIController(
        Kp=KP_INNER, Ki=KI_INNER,
        output_min=0.0, output_max=1.0,    # duty ∈ [0, 1]
    )
    setpoint_fn = make_setpoint(T_STEP, V_OUT_REF_PRE, V_OUT_REF_POST)

    # The L1 observer drives the arm dynamics.
    obs_arm, bex = p.make_mmc_arm_multilevel_observer(b, arm, dt=DT)

    # Loggers for plotting / KPIs.
    n_samples = int(round(T_END / DT)) + 1
    log_t       = np.zeros(n_samples)
    log_v_out   = np.zeros(n_samples)
    log_i_L     = np.zeros(n_samples)
    log_i_L_ref = np.zeros(n_samples)
    log_m_ref   = np.zeros(n_samples)
    log_v_C     = np.zeros(n_samples)
    log_setpoint = np.zeros(n_samples)
    counter = [0]

    # Cascaded PI runs every dt to keep the demo simple (no
    # multi-rate logic). At a 10 µs step that's 100 kHz, which is
    # ~12× faster than the inner loop's target crossover — plenty.
    def control_and_observe(t, x):
        v_out = float(x[vout_idx])
        i_L   = float(x[iL_idx])
        sp    = setpoint_fn(t)

        # Outer: V_out → I_L_ref (positive plant gain).
        i_L_ref = outer.update(setpoint=sp, measured=v_out, dt=DT)

        # Inner: I_L → buck duty d ∈ [0, 1] (positive plant gain:
        # higher duty ⇒ higher V_mid ⇒ higher I_L).
        d = inner.update(setpoint=i_L_ref, measured=i_L, dt=DT)

        # Convert to MMC modulation index: m_ref = 1 - d, so the arm
        # voltage v_arm = m_ref · v_C ≈ (1-d)·V_dc gives V_mid ≈ d·V_dc
        # (matching the canonical buck-duty interpretation).
        m_ref_holder[0] = max(0.0, min(1.0, 1.0 - d))

        # Run the L1 step (advances v_C, stashes v_b).
        obs_arm(t, x)

        # Log.
        i = counter[0]
        if i < n_samples:
            log_t[i]        = t
            log_v_out[i]    = v_out
            log_i_L[i]      = i_L
            log_i_L_ref[i]  = i_L_ref
            log_m_ref[i]    = m_ref_holder[0]
            log_v_C[i]      = arm.v_C
            log_setpoint[i] = sp
        counter[0] += 1

    print(f"Closed-loop MMC buck:")
    print(f"  V_dc           = {V_DC:.0f} V")
    print(f"  N_SM           = {N_SM}, f_switch = {N_SM*F_CARRIER:.0f} Hz")
    print(f"  L_filt, C_out  = {L_FILT*1e3:.2f} mH, {C_OUT*1e6:.0f} µF "
          f"(ω_LC ≈ {1/(2*math.pi*math.sqrt(L_FILT*C_OUT)):.0f} Hz)")
    print(f"  R_load         = {R_LOAD:.1f} Ω")
    print(f"  Setpoint       = {V_OUT_REF_PRE} V → "
          f"{V_OUT_REF_POST} V @ {T_STEP*1e3:.0f} ms")
    print(f"  Outer PI       = Kp={KP_OUTER}, Ki={KI_OUTER}")
    print(f"  Inner PI       = Kp={KP_INNER}, Ki={KI_INNER}")
    print()
    print(f"Simulating {T_END*1e3:.0f} ms at dt={DT*1e6:.0f} µs...")

    p.simulate(
        b, t_end=T_END, dt=DT,
        step_observer=control_and_observe,
        b_extra_fn=bex,
        start_from_dc_op=True,
    )

    # Trim arrays to actual sample count.
    n = counter[0]
    log_t        = log_t[:n]
    log_v_out    = log_v_out[:n]
    log_i_L      = log_i_L[:n]
    log_i_L_ref  = log_i_L_ref[:n]
    log_m_ref    = log_m_ref[:n]
    log_v_C      = log_v_C[:n]
    log_setpoint = log_setpoint[:n]

    # =========================================================================
    # KPIs
    # =========================================================================

    # Pre-step steady state (40-60 ms).
    pre_mask = (log_t >= 40e-3) & (log_t < T_STEP)
    # Post-step steady state (90-100 ms).
    post_mask = log_t >= 90e-3
    pre_v_out  = log_v_out[pre_mask]
    post_v_out = log_v_out[post_mask]
    print(f"\nKPIs:")
    print(f"  V_out  pre-step  (40-60 ms): mean={pre_v_out.mean():.2f} V "
          f"(target {V_OUT_REF_PRE} V, error "
          f"{pre_v_out.mean()-V_OUT_REF_PRE:+.2f} V)")
    print(f"  V_out  post-step (90-100 ms): mean={post_v_out.mean():.2f} V "
          f"(target {V_OUT_REF_POST} V, error "
          f"{post_v_out.mean()-V_OUT_REF_POST:+.2f} V)")
    print(f"  Min / Max V_out during step transient: "
          f"{log_v_out.min():.2f} / {log_v_out.max():.2f} V")
    print(f"  v_C drift (start → end): "
          f"{log_v_C[0]:.2f} → {log_v_C[-1]:.2f} V "
          f"({(log_v_C[-1]-log_v_C[0])/log_v_C[0]*100:+.2f} %)")

    # =========================================================================
    # Plot
    # =========================================================================

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ts_ms = log_t * 1e3

    # Panel 1: V_out + setpoint.
    axes[0].plot(ts_ms, log_v_out, label="V_out", color="C0",
                     linewidth=1.2)
    axes[0].plot(ts_ms, log_setpoint, label="V_out_ref",
                     color="C3", linestyle="--", linewidth=1.2)
    axes[0].axvline(T_STEP*1e3, color="k", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("V_out [V]")
    axes[0].set_title(
        f"Closed-loop MMC buck — cascaded PI feeding L1 PS-PWM"
    )
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: I_L + I_L_ref.
    axes[1].plot(ts_ms, log_i_L, label="I_L", color="C0",
                     linewidth=1.2)
    axes[1].plot(ts_ms, log_i_L_ref, label="I_L_ref (outer PI)",
                     color="C3", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("I_L [A]")
    axes[1].set_title("Filter-inductor current (inner-loop tracking)")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: m_ref.
    axes[2].plot(ts_ms, log_m_ref, color="C2", linewidth=1.0)
    axes[2].set_ylabel("m_ref")
    axes[2].set_title("Inner-PI modulation command (drives the L1 PS-PWM)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: v_C drift.
    axes[3].plot(ts_ms, log_v_C, color="C1", linewidth=1.0)
    axes[3].axhline(V_C0, color="k", linestyle="--", alpha=0.4,
                       label=f"v_c0 = {V_C0:.0f} V")
    axes[3].set_xlabel("time [ms]")
    axes[3].set_ylabel("v_C [V]")
    axes[3].set_title("Arm-capacitor sum voltage")
    axes[3].legend(fontsize=9, loc="lower right")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "out" / "mmc_cascaded_pi.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
