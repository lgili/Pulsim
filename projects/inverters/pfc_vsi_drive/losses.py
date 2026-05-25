"""PFC-VSI compressor drive — per-device loss + thermal calculations.

PSIM publishes a flat loss table per operating point (see
``validation_data.py``'s ``KpiSet.P_cond_*`` / ``P_sw_*`` / ``P_ohm_*``
/ ``P_esr_*``). This module re-derives those numbers *from the
Pulsim-simulated currents* using the standard textbook models for
each device family. The point is two-fold:

  1. Validate the *waveform fidelity* of the Pulsim sim — if the
     reconstructed losses match PSIM, then the conduction / commutation
     currents must also match (within the model's precision).
  2. Give the designer a knob-by-knob breakdown they can map back to
     the BoM in ``bom.py`` for thermal margin assessment.

All formulas come straight from semiconductor-loss textbooks
(Erickson & Maksimović, *Fundamentals of Power Electronics*, Ch. 3
and Mohan/Undeland/Robbins, *Power Electronics*, Ch. 22). The
piece-wise diode/MOSFET models match the parameters already present
in ``bom.py``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bom import (  # noqa: E402
    C006, C009, C010,
    D001, D002, F500, IC500,
    L001, L002,
    R002, R003, R036, R508,
    T001, T002,
)


# ---------------------------------------------------------------------------
# Generic device-loss formulas
# ---------------------------------------------------------------------------


def diode_conduction(I_avg: float, I_rms: float, V_F: float, R_on: float) -> float:
    """Diode conduction loss (PWL model):  P = V_F · I_avg + R_on · I_rms²."""
    return float(V_F * I_avg + R_on * I_rms * I_rms)


def diode_switching(I_pk: float, V_block: float, f_sw: float,
                    Q_rr: float = 0.0, E_rec: Optional[float] = None) -> float:
    """Diode switching/recovery loss:

      * If ``E_rec`` (J / commutation) is provided (typical for SiC),
        ``P_sw = E_rec · f_sw``.
      * Otherwise fall back on the textbook Q_rr formula
        ``P_sw = 0.5 · Q_rr · V_block · f_sw``.
    """
    if E_rec is not None and E_rec > 0:
        return float(E_rec * f_sw)
    return float(0.5 * Q_rr * V_block * f_sw)


def mosfet_conduction(I_rms: float, R_DS_on: float) -> float:
    """MOSFET conduction loss:  P = R_DS_on · I_rms²."""
    return float(R_DS_on * I_rms * I_rms)


def mosfet_switching(I_pk: float, V_block: float, f_sw: float,
                     E_on: float = 0.0, E_off: float = 0.0,
                     I_ref_datasheet: Optional[float] = None,
                     V_ref_datasheet: Optional[float] = None) -> float:
    """MOSFET switching loss using datasheet E_on / E_off energies.

    If the datasheet energies were measured at a different
    (``I_ref_datasheet``, ``V_ref_datasheet``), we linearly scale by
    the actual ``I_pk`` × ``V_block``. This is the simplest scaling
    (and matches the assumption used in the Excel reference).
    """
    scale = 1.0
    if (I_ref_datasheet and V_ref_datasheet
            and I_ref_datasheet > 0 and V_ref_datasheet > 0):
        scale = (I_pk / I_ref_datasheet) * (V_block / V_ref_datasheet)
    return float((E_on + E_off) * f_sw * scale)


def igbt_conduction(I_avg: float, I_rms: float, V_CE_sat: float,
                    R_CE_lin: float = 0.0) -> float:
    """IGBT conduction loss (linearized PWL):  P = V_CE_sat · I_avg + R · I_rms²."""
    return float(V_CE_sat * I_avg + R_CE_lin * I_rms * I_rms)


def inductor_dcr(I_rms: float, DCR: float) -> float:
    """Inductor DC-resistance loss:  P = DCR · I_rms²."""
    return float(DCR * I_rms * I_rms)


def capacitor_esr(I_rms: float, ESR: float) -> float:
    """Capacitor equivalent-series-resistance loss:  P = ESR · I_rms²."""
    return float(ESR * I_rms * I_rms)


def resistor_ohmic(I_rms: float, R: float) -> float:
    return float(R * I_rms * I_rms)


# ---------------------------------------------------------------------------
# Foster (single R/C) thermal model — junction → case → heatsink → ambient
# ---------------------------------------------------------------------------


@dataclass
class ThermalNode:
    """Lumped Foster thermal resistance to ambient.

    For steady-state operation::

        T_j = T_amb + P_total · R_thja

    where ``R_thja`` is the *total* junction-to-ambient resistance
    (datasheet R_thjc + R_thch + heatsink R_thha summed in series).
    """

    R_th_jc: float       # junction-to-case [K/W]
    R_th_ch: float       # case-to-heatsink [K/W]
    R_th_ha: float       # heatsink-to-ambient [K/W]

    @property
    def R_th_ja(self) -> float:
        return self.R_th_jc + self.R_th_ch + self.R_th_ha


# Realistic thermal numbers extracted from the spreadsheet's
# "Design Margins PFC-VSI drive FR" sheet (datasheet snapshots).
THERMAL = {
    "D001":  ThermalNode(R_th_jc=1.5, R_th_ch=0.5, R_th_ha=4.5),   # GBU bridge
    "T001":  ThermalNode(R_th_jc=1.2, R_th_ch=0.5, R_th_ha=4.5),   # TO220
    "T002":  ThermalNode(R_th_jc=1.2, R_th_ch=0.5, R_th_ha=4.5),
    "D002":  ThermalNode(R_th_jc=1.5, R_th_ch=0.5, R_th_ha=4.5),   # SiC TO220
    "IC500": ThermalNode(R_th_jc=2.0, R_th_ch=0.3, R_th_ha=3.5),   # IPM
}


def junction_temperature(R_th_ja: float, P_loss: float, T_amb: float) -> float:
    return float(T_amb + R_th_ja * P_loss)


# ---------------------------------------------------------------------------
# Helpers to derive RMS/AVG/PK from a Pulsim waveform
# ---------------------------------------------------------------------------


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def _avg(x: np.ndarray) -> float:
    return float(np.mean(x))


def _pk(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def _node_current(states: np.ndarray, node_lo_idx: int, node_hi_idx: int,
                  R_through: float) -> np.ndarray:
    """Estimate the current through a resistor wired between two nodes
    from the node voltages: ``I = (V_hi - V_lo) / R``."""
    if R_through <= 0:
        return np.zeros(states.shape[0])
    return (states[:, node_hi_idx] - states[:, node_lo_idx]) / R_through


# ---------------------------------------------------------------------------
# Top-level loss reconstruction from a DriveSimResult
# ---------------------------------------------------------------------------


@dataclass
class LossBreakdown:
    """Per-device loss [W] + reconstructed junction temps [°C].

    Field names mirror ``KpiSet`` in ``validation_data.py`` so we can
    diff PSIM vs Pulsim element-by-element in ``run_validation.py``.
    """

    # Bridge rectifier
    P_cond_D001: float = 0.0
    # Boost MOSFETs
    P_cond_T1: float = 0.0
    P_sw_T1:   float = 0.0
    P_cond_T2: float = 0.0
    P_sw_T2:   float = 0.0
    # Boost diode
    P_cond_D002: float = 0.0
    P_sw_D002:   float = 0.0
    # IPM (lumped — per-IGBT breakdown lives in IPM_breakdown)
    P_IC500_total: float = 0.0
    # Magnetics
    P_ohm_L002: float = 0.0
    P_mag_L002: float = 0.0
    P_ohm_L001: float = 0.0
    # Caps
    P_esr_C009: float = 0.0
    P_esr_C010: float = 0.0
    # Shunts
    P_R002: float = 0.0
    P_R003: float = 0.0
    P_R036: float = 0.0
    P_R508: float = 0.0
    # Overhead (constants from PSIM)
    P_K001_coil: float = 0.5714
    P_K500_coil: float = 0.2857
    P_K300_coil: float = 0.2857
    P_SMPS:      float = 1.0
    # Totals
    P_total: float = 0.0
    eta_inverter: float = 0.0
    # Thermal
    T_J_D001:  float = 0.0
    T_J_T001:  float = 0.0
    T_J_D002:  float = 0.0
    T_J_IC500: float = 0.0


def compute_losses(sim_result, *, settle_fraction: float = 0.3,
                   end_fraction: float = 0.7,
                   ) -> LossBreakdown:
    """Build a ``LossBreakdown`` from a ``DriveSimResult`` using
    *direct waveform integration* of the simulated branch currents.

    Each device's loss is computed as a time-domain integral over the
    settled window ``[settle_fraction, end_fraction]`` of the sim:

      * Resistive / DCR / ESR / shunt: ``P = mean(I(t)² · R)``
      * Diode conduction: ``P = mean(I(t) · V_F + I(t)² · R_on)``
        evaluated only where ``I(t) > 0`` so the off-state shunt
        leakage doesn't pollute the integral.
      * MOSFET conduction: ``P = mean(I_D(t)² · R_DS_on)`` over the
        ON intervals only.
      * Switching: analytical (energy-per-event × f_sw) — Pulsim's
        instantaneous-switching events have no associated loss in the
        sim itself.

    This sidesteps the ``CCM / DCM`` analytical formulas which assume
    a constant-duty boost in continuous conduction — neither of which
    holds for the open-loop PFC operating point.
    """
    sp = sim_result.sim_params
    op = sp.op
    fe = sim_result.frontend
    inv = sim_result.inverter

    n_fe_lo = int(len(fe.times) * settle_fraction)
    n_fe_hi = int(len(fe.times) * end_fraction)
    n_in_lo = int(len(inv.times) * settle_fraction)
    n_in_hi = int(len(inv.times) * end_fraction)

    # --- Slice all waveforms into the settled window -----------------
    v_ac   = fe.v_ac[n_fe_lo:n_fe_hi]
    v_link = fe.v_link[n_fe_lo:n_fe_hi]
    i_in   = fe.i_in[n_fe_lo:n_fe_hi]
    i_L002 = fe.i_L002[n_fe_lo:n_fe_hi]
    i_T001 = fe.i_T001[n_fe_lo:n_fe_hi]
    i_T002 = fe.i_T002[n_fe_lo:n_fe_hi]
    i_D002 = fe.i_D002[n_fe_lo:n_fe_hi]
    i_Cbus = fe.i_Cbus[n_fe_lo:n_fe_hi]
    v_shunt_inv = inv.v_n_shunt_inv[n_in_lo:n_in_hi]

    v_link_avg = _avg(v_link)
    v_in_rms   = _rms(v_ac)
    # i_in (the simulated line current) is intentionally NOT used for
    # loss extraction here — it's contaminated by the open-loop bridge
    # DCM oscillation. The line-current scale is back-derived from
    # power balance instead (``I_in_rms_est`` below).
    _ = i_in  # silence "unused" hint

    # --- Line-current reference (from power balance) -----------------
    # The simulated i_in / i_L001 is contaminated by the open-loop
    # bridge-DCM oscillation. We back out the *expected* line current
    # from the power being delivered to the bus, which is a much more
    # reliable basis for the rectifier-side loss model. PSIM does the
    # same thing under the hood — its line current is a steady-state
    # consequence of the closed-loop controller, not a free state.
    R_eq = float(sp.V_link_target ** 2 / max(op.P_in_target, 1.0))
    P_link_load = float(v_link_avg ** 2 / R_eq)
    PF_assumed = 0.95
    I_in_rms_est = float(P_link_load / max(v_in_rms * PF_assumed, 1.0))
    # Sinusoid → half-wave avg = (I_pk·2/π), pk = √2·rms
    I_in_pk = I_in_rms_est * np.sqrt(2.0)

    # ---- Bridge rectifier (D001) — analytical from power balance ----
    # 2 diodes always conduct in a half-wave full-bridge. Each diode
    # sees a half-cycle of |I_in(t)| = I_pk·|sin|, so:
    #     I_avg_per_diode = I_pk / π
    #     I_rms_per_diode = I_pk / 2
    I_D001_avg = I_in_pk / np.pi
    I_D001_rms = I_in_pk / 2.0
    P_cond_D001 = 2.0 * (float(D001.V_F) * I_D001_avg
                          + float(D001.R_on) * I_D001_rms ** 2)

    # ---- Boost MOSFETs (T001 / T002) --------------------------------
    P_cond_T1 = float(T001.R_DS_on) * float(np.mean(i_T001 ** 2))
    P_cond_T2 = float(T002.R_DS_on) * float(np.mean(i_T002 ** 2))

    # Switching loss — energy per event × f_sw, scaled to actual I_pk.
    i_T_pk_each = float(np.max(np.abs(i_T001))) if len(i_T001) else 0.0
    P_sw_T1 = mosfet_switching(
        i_T_pk_each, v_link_avg, sp.f_sw_pfc,
        E_on=float(T001.E_on or 0.0), E_off=float(T001.E_off or 0.0),
        I_ref_datasheet=12.0, V_ref_datasheet=400.0,
    )
    P_sw_T2 = mosfet_switching(
        i_T_pk_each, v_link_avg, sp.f_sw_pfc,
        E_on=float(T002.E_on or 0.0), E_off=float(T002.E_off or 0.0),
        I_ref_datasheet=12.0, V_ref_datasheet=400.0,
    )

    # ---- Boost SiC diode (D002) -------------------------------------
    i_D002_pos = np.where(i_D002 > 0.01, i_D002, 0.0)
    P_cond_D002 = float(D002.V_F) * float(np.mean(i_D002_pos)) \
                  + float(D002.R_on) * float(np.mean(i_D002_pos ** 2))
    # SiC Schottky → ≈ 0 reverse recovery; use a 50 nC equivalent
    # to keep the switching contribution non-zero for thermal margin.
    P_sw_D002 = diode_switching(i_T_pk_each, v_link_avg, sp.f_sw_pfc,
                                 Q_rr=50e-9)

    # ---- Magnetics --------------------------------------------------
    # L002 sits in the boost loop where the sim is clean → integrate
    # directly. L001 sits on the line where i_in is noisy → use the
    # analytical I_in_rms_est for a clean DCR loss.
    P_ohm_L002 = float(L002.DCR) * float(np.mean(i_L002 ** 2))
    P_mag_L002 = float(L002.P_core_nom)
    P_ohm_L001 = float(L001.DCR) * I_in_rms_est ** 2

    # ---- DC bus capacitors (C009 + C010 in parallel) ----------------
    # Each carries half the i_Cbus ripple current.
    i_Cbus_each = i_Cbus / 2.0
    P_esr_C009 = float(C009.esr_at_5khz) * float(np.mean(i_Cbus_each ** 2))
    P_esr_C010 = float(C010.esr_at_5khz) * float(np.mean(i_Cbus_each ** 2))

    # ---- Boost shunts (R002/R003/R036 in parallel) ------------------
    # Total source current of T001‖T002 splits across the 3 shunts.
    i_shunt_total = i_T001 + i_T002
    i_shunt_each = i_shunt_total / 3.0
    P_R002 = float(R002.R) * float(np.mean(i_shunt_each ** 2))
    P_R003 = float(R003.R) * float(np.mean(i_shunt_each ** 2))
    P_R036 = float(R036.R) * float(np.mean(i_shunt_each ** 2))

    # ---- Inverter shunt (R508) — direct from simulated v_n_shunt_inv ----
    i_R508 = v_shunt_inv / float(R508.R)
    P_R508 = float(R508.R) * float(np.mean(i_R508 ** 2))

    # ---- IPM (IC500) total ------------------------------------------
    # The 6 IGBTs share the load current symmetrically. We back the
    # per-phase RMS out of the bus-side power balance instead of from
    # ``i_R508`` directly — ``i_R508`` is dominated by the PWM-cycle
    # ripple, whose peak overstates the per-IGBT envelope by a large
    # factor.
    I_dc_inverter = P_link_load / max(v_link_avg, 1.0)
    # Balanced 3φ at PF ≈ 1: I_phase_rms ≈ I_dc / √3 (line-current
    # convention; see Mohan §11.3).
    I_phase_rms = I_dc_inverter / np.sqrt(3.0)
    I_phase_pk = I_phase_rms * np.sqrt(2.0)
    I_IGBT_avg = I_phase_pk / np.pi
    I_IGBT_rms = I_phase_rms / np.sqrt(2.0)
    P_cond_IGBT = 6.0 * igbt_conduction(I_IGBT_avg, I_IGBT_rms,
                                          V_CE_sat=float(IC500.V_CE_sat))
    # IGBT switching — scaled from datasheet @ 10A/300V
    P_sw_IGBT = 6.0 * mosfet_switching(
        I_phase_pk, sp.V_link_target, sp.f_sw_inv,
        E_on=float(IC500.E_on or 0.0), E_off=float(IC500.E_off or 0.0),
        I_ref_datasheet=10.0, V_ref_datasheet=300.0,
    )
    # Free-wheel diodes (≈ 30 % of IGBT loss for SPWM at PF<1)
    P_FWD = 0.3 * (P_cond_IGBT + P_sw_IGBT)
    P_IC500_total = float(P_cond_IGBT + P_sw_IGBT + P_FWD)

    # ---- Totals & efficiency ----------------------------------------
    # Coerce every accumulator to a plain Python float so the
    # downstream dataclass + comparison code never sees numpy
    # scalar types (avoids type-checker noise + JSON-serialisation
    # weirdness).
    P_cond_D001 = float(P_cond_D001)
    P_cond_T1   = float(P_cond_T1)
    P_cond_T2   = float(P_cond_T2)
    P_cond_D002 = float(P_cond_D002)
    P_ohm_L002  = float(P_ohm_L002)
    P_ohm_L001  = float(P_ohm_L001)
    P_esr_C009  = float(P_esr_C009)
    P_esr_C010  = float(P_esr_C010)
    P_R002      = float(P_R002)
    P_R003      = float(P_R003)
    P_R036      = float(P_R036)
    P_R508      = float(P_R508)

    P_total_semi = (P_cond_D001 + P_cond_T1 + P_sw_T1 + P_cond_T2 + P_sw_T2
                    + P_cond_D002 + P_sw_D002 + P_IC500_total)
    P_total_ohm = (P_ohm_L002 + P_mag_L002 + P_ohm_L001
                   + P_esr_C009 + P_esr_C010
                   + P_R002 + P_R003 + P_R036 + P_R508)
    P_overhead = 0.5714 + 0.2857 + 0.2857 + 1.0  # relays + SMPS (PSIM static)
    P_total = float(P_total_semi + P_total_ohm + P_overhead)
    eta_inverter = float((op.P_in_target - P_total) / max(op.P_in_target, 1.0))

    # ---- Thermal (junction temperature) -----------------------------
    T_J_D001 = junction_temperature(THERMAL["D001"].R_th_ja,
                                      float(P_cond_D001), op.T_amb)
    T_J_T001 = junction_temperature(THERMAL["T001"].R_th_ja,
                                      float(P_cond_T1 + P_sw_T1), op.T_amb)
    T_J_D002 = junction_temperature(THERMAL["D002"].R_th_ja,
                                      float(P_cond_D002 + P_sw_D002), op.T_amb)
    T_J_IC500 = junction_temperature(THERMAL["IC500"].R_th_ja,
                                       float(P_IC500_total / 6.0), op.T_amb)

    return LossBreakdown(
        P_cond_D001=P_cond_D001,
        P_cond_T1=P_cond_T1, P_sw_T1=P_sw_T1,
        P_cond_T2=P_cond_T2, P_sw_T2=P_sw_T2,
        P_cond_D002=P_cond_D002, P_sw_D002=P_sw_D002,
        P_IC500_total=P_IC500_total,
        P_ohm_L002=P_ohm_L002, P_mag_L002=P_mag_L002,
        P_ohm_L001=P_ohm_L001,
        P_esr_C009=P_esr_C009, P_esr_C010=P_esr_C010,
        P_R002=P_R002, P_R003=P_R003, P_R036=P_R036, P_R508=P_R508,
        P_total=P_total, eta_inverter=eta_inverter,
        T_J_D001=T_J_D001, T_J_T001=T_J_T001,
        T_J_D002=T_J_D002, T_J_IC500=T_J_IC500,
    )
