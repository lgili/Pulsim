#!/usr/bin/env python3
"""IC500 IPM thermal model — OLD per-device-independent vs NEW Pulsim P1–P4.

The drive's current loss model (``losses.py``) estimates the inverter
IPM junction temperature as

    T_j = T_amb + R_th_ja · (P_IC500_total / 6)        # ← the OLD model

i.e. it applies the *full* junction-to-ambient resistance — which
includes the shared baseplate→heatsink path (R_th_ha = 3.5 K/W) — to
only **one sixth** of the module's dissipation. But the IKCM20L60GD is
six IGBTs + six diodes on **one** baseplate: the heatsink rise is driven
by the *total* power of all twelve dies, not 1/6. The old formula is
therefore optimistic about the shared-heatsink contribution, and it
cannot see (a) inter-die coupling or (b) the V_ce(T)/V_f(T) feedback
that makes conduction loss climb with temperature.

This script rebuilds the IPM thermal estimate with the Pulsim thermal
upgrade and quantifies the gap:

  * P1  shared baseplate — six IGBTs + six diodes coupled through one sink
  * P2  electro-thermal feedback — V_ce(T) raises conduction loss with T_j
  * P3  offset+slope conduction (V_ce0 + r_ce·I)  used for the loss anchor
  * P4  (datasheet E_sw curve — see losses.py; here E_sw is a scalar)

Run (needs the P1–P4 thermal API):
    PYTHONPATH=build/python python3 \
        projects/inverters/pfc_vsi_drive/thermal_comparison.py

NOTE on parameters: the per-die R_th and the shared-heatsink R_th_sa are
stated below from the IKCM20L60GD datasheet + a representative forced-air
heatsink. The point is the *modelling* difference and its magnitude; the
absolute numbers move with the heatsink you actually choose — which is
exactly the design knob the new model exposes.
"""

from __future__ import annotations

import math

import pulsim as p


# --- IKCM20L60GD IPM datasheet (Infineon CIPOS, 600 V / 20 A) ---------------
V_CE0, R_CE = 0.9, 0.040     # V_ce_sat = 0.9 + 0.040·I → 1.7 V @ 20 A
V_F0,  R_D  = 0.9, 0.030     # V_f      = 0.9 + 0.030·I → 1.5 V @ 20 A
E_ON, E_OFF, E_REC = 300e-6, 300e-6, 150e-6
V_REF, I_REF = 300.0, 20.0   # E_sw datasheet reference point
A_COND_IGBT = +0.004         # net IGBT conduction tempco [1/°C] (~+0.4 %/°C)
A_COND_DIODE = -0.002        # diode V_f negative tempco
A_SW = +0.005                # switching-energy tempco

# Thermal structure (datasheet per-die + a real shared heatsink) -------------
R_JC_IGBT, R_JC_DIODE = 2.0, 3.0   # K/W, junction → IPM case (per die)
R_CS = 0.3                          # K/W, case → heatsink (shared TIM/baseplate)
R_SA_SHARED = 0.5                   # K/W, heatsink → ambient (forced air)
R_TH_JA_OLD = 5.8                   # K/W, the project's lumped R_th_ja

T_AMB = 40.0
T_J_MAX = 150.0
F_SW = 10.0e3
V_DC = 380.0
M, PF = 0.85, 0.90        # modulation depth, power factor


# --- 3-phase 2-level VSI per-device loss (textbook SPWM averages) -----------
def vsi_losses(I_pk: float, T_j: float = 25.0):
    """Per-IGBT and per-diode dissipation [W] at peak phase current I_pk
    and junction temperature T_j (conduction coefficients drift with T)."""
    c = PF  # cos(phi)
    # Conduction current shares (Infineon AN2008-03 / Erickson §22).
    I_avg_T = I_pk * (1.0 / (2.0 * math.pi) + M * c / 8.0)
    I_rms_T = I_pk * math.sqrt(1.0 / 8.0 + M * c / (3.0 * math.pi))
    I_avg_D = I_pk * (1.0 / (2.0 * math.pi) - M * c / 8.0)
    I_rms_D = I_pk * math.sqrt(max(1.0 / 8.0 - M * c / (3.0 * math.pi), 0.0))
    dT = T_j - 25.0

    vce0 = V_CE0 * (1.0 + A_COND_IGBT * dT)
    rce = R_CE * (1.0 + A_COND_IGBT * dT)
    vf0 = V_F0 * (1.0 + A_COND_DIODE * dT)
    rd = R_D * (1.0 + A_COND_DIODE * dT)
    esw = (1.0 + A_SW * dT)

    P_cond_T = vce0 * I_avg_T + rce * I_rms_T ** 2
    P_cond_D = vf0 * I_avg_D + rd * I_rms_D ** 2
    # Switching: average of |i| over the conducting half = I_pk/π.
    I_sw_avg = I_pk / math.pi
    P_sw_T = (E_ON + E_OFF) * esw * F_SW * (V_DC / V_REF) * (I_sw_avg / I_REF)
    P_rec_D = E_REC * esw * F_SW * (V_DC / V_REF) * (I_sw_avg / I_REF)
    return (P_cond_T, P_sw_T), (P_cond_D, P_rec_D)


def t_j_old(I_pk: float) -> float:
    """The project's model: lumped R_th_ja applied to P_total/6, fixed
    25 °C coefficients."""
    (PcT, PsT), (PcD, PrD) = vsi_losses(I_pk, T_j=25.0)
    P_total = 6.0 * (PcT + PsT) + 6.0 * (PcD + PrD)
    return T_AMB + R_TH_JA_OLD * (P_total / 6.0)


def electrothermal_new(I_pk: float, *, R_sa: float = R_SA_SHARED,
                       a_cond_igbt: float = A_COND_IGBT):
    """P1+P2: six IGBTs + six diodes on one baseplate, conduction loss
    drifting with T_j. Returns the hottest-junction result dict."""
    (PcT, PsT), (PcD, PrD) = vsi_losses(I_pk, T_j=25.0)
    igbts = [p.HeatsinkDevice(f"Q{i}",
                              [p.FosterStage(R_th_K_per_W=R_JC_IGBT, tau_s=0.05)],
                              R_th_case_to_sink_K_per_W=R_CS) for i in range(6)]
    diodes = [p.HeatsinkDevice(f"D{i}",
                               [p.FosterStage(R_th_K_per_W=R_JC_DIODE, tau_s=0.03)],
                               R_th_case_to_sink_K_per_W=R_CS) for i in range(6)]
    # TempCoLoss anchored at 25 °C; the conduction part carries the tempco,
    # the switching part its own.
    q_model = p.TempCoLoss(P_cond_ref_W=PcT, P_sw_ref_W=PsT,
                           a_cond_per_C=a_cond_igbt, a_sw_per_C=A_SW)
    d_model = p.TempCoLoss(P_cond_ref_W=PcD, P_sw_ref_W=PrD,
                           a_cond_per_C=A_COND_DIODE, a_sw_per_C=A_SW)
    models = {**{f"Q{i}": q_model for i in range(6)},
              **{f"D{i}": d_model for i in range(6)}}
    return p.electrothermal_steady_state(
        igbts + diodes, models,
        R_th_sink_to_amb_K_per_W=R_sa, T_amb_C=T_AMB)


def i_pk_for_power(P_out_W: float) -> float:
    """Peak phase current for a given 3φ output power (V_phase from m·V_dc)."""
    V_ph_rms = M * V_DC / (2.0 * math.sqrt(2.0))
    I_rms = P_out_W / (3.0 * V_ph_rms * PF)
    return I_rms * math.sqrt(2.0)


def main() -> int:
    print("=== IC500 IPM thermal: OLD (lumped P/6) vs NEW (shared + electro-thermal) ===")
    print(f"T_amb={T_AMB} °C  V_dc={V_DC} V  f_sw={F_SW/1e3:.0f} kHz  "
          f"R_sa_shared={R_SA_SHARED} K/W  T_j_max={T_J_MAX} °C\n")

    # Rated operating point (~1.1 kW output).
    P_rated = 1100.0
    I_pk = i_pk_for_power(P_rated)
    (PcT, PsT), (PcD, PrD) = vsi_losses(I_pk, 25.0)
    P_ipm = 6 * (PcT + PsT) + 6 * (PcD + PrD)
    old = t_j_old(I_pk)
    new = electrothermal_new(I_pk)
    hottest = max(d["T_j_C"] for d in new["devices"].values())

    print(f"At P_out ≈ {P_rated:.0f} W  (I_phase_pk ≈ {I_pk:.1f} A):")
    print(f"  IPM total dissipation        : {P_ipm:6.2f} W")
    print(f"  T_sink (shared baseplate)    : {new['T_sink_C']:6.1f} °C")
    print(f"  OLD model  T_j (lumped P/6)  : {old:6.1f} °C")
    print(f"  NEW model  T_j (hottest IGBT): {hottest:6.1f} °C   "
          f"(Δ = {hottest - old:+.1f} °C vs OLD)")
    print(f"  electro-thermal feedback gain ρ : {new['feedback_gain']:.3f}  "
          f"(runaway at ρ≥1)\n")

    # Power sweep — where does each model hit T_j_max?
    print("  P_out [W] | I_pk [A] | T_j OLD | T_j NEW |  Δ (NEW−OLD)")
    print("  " + "-" * 56)
    p_old_max = p_new_max = None
    for P_out in range(1000, 6501, 500):
        ipk = i_pk_for_power(float(P_out))
        to = t_j_old(ipk)
        rn = electrothermal_new(ipk)
        tn = max(d["T_j_C"] for d in rn["devices"].values())
        if p_new_max is None and tn > T_J_MAX:
            p_new_max = P_out
        if p_old_max is None and to > T_J_MAX:
            p_old_max = P_out
        print(f"  {P_out:8d}  | {ipk:7.1f} | {to:6.1f}  | {tn:6.1f} |  {tn-to:+5.1f}")

    print()
    print(f"  Max P_out before T_j_max — OLD model : "
          f"{'>6500' if p_old_max is None else p_old_max} W")
    print(f"  Max P_out before T_j_max — NEW model : "
          f"{'>6500' if p_new_max is None else p_new_max} W")
    if p_old_max and p_new_max and p_new_max < p_old_max:
        print(f"  → the OLD model over-promises the power headroom by "
              f"~{p_old_max - p_new_max} W. At this design point the shared "
              f"R_ha/6 ≈ matches the coupling, but the V_ce(T) feedback "
              f"(loss climbing with temperature) is invisible to it.")

    # Degraded-cooling what-if — the unique value of P2: a worse heatsink
    # plus a hotter conduction tempco can drive the loss↔temperature loop
    # into RUNAWAY. The OLD model (fixed 25 °C coefficients) can never
    # show this; it just reports an ever-rising-but-finite T_j.
    print("\n  Degraded cooling what-if  (R_sa = 2.2 K/W, IGBT tempco +0.9 %/°C):")
    for P_out in (2000, 3000, 4000, 5000):
        ipk = i_pk_for_power(float(P_out))
        rn = electrothermal_new(ipk, R_sa=2.2, a_cond_igbt=0.009)
        if rn.get("runaway"):
            print(f"    P_out={P_out:5d} W → THERMAL RUNAWAY "
                  f"(feedback gain ρ={rn['feedback_gain']:.2f} ≥ 1) — "
                  f"OLD model would report a finite T_j here and miss it.")
        else:
            tn = max(d["T_j_C"] for d in rn["devices"].values())
            flag = "  ⚠ exceeds T_j_max (device fails)" if tn > T_J_MAX else ""
            print(f"    P_out={P_out:5d} W → T_j={tn:6.0f} °C  "
                  f"(ρ={rn['feedback_gain']:.2f}, margin {1-rn['feedback_gain']:.2f})"
                  f"{flag}")
    print("    (T_j past T_j_max is the linear electro-thermal trend toward "
          "the ρ→1 singularity — the device is destroyed well before those "
          "values; the point is that the feedback gain is climbing to runaway.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
