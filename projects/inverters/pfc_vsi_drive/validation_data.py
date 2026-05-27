"""Reference PSIM validation data — PFC-VSI compressor drive — Compressor.

Source: Internal design deliverable
  ``the reference PSIM simulation``
  (project ID PFC compressor drive,
   "Design Deliverable — Power Stage Simulation").

Three operating points were simulated in PSIM at the schematic
(production) level. This module re-publishes the numerical results as
Python data so the Pulsim port (see
``pfc_vsi_drive_pulsim_validation.py``) can target them
quantitatively, OP by OP, KPI by KPI.

The original PSIM signal names are preserved verbatim (e.g.
``S2.I_T001``) so anyone reading both the spreadsheet and this
module can cross-reference without ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    """A single test condition the PSIM model was run at."""

    id: str                  # e.g. "OP_2.2"
    label: str               # human-readable
    V_ac: float              # input line voltage [V_rms]
    f_line: float            # line frequency [Hz]
    P_in_target: float       # nameplate input power [W]
    T_amb: float             # ambient temperature [°C]
    speed_rpm: float         # mechanical speed of the compressor [rpm]
    airflow_mps: float = 2.0 # forced ventilation [m/s]


OP_22 = OperatingPoint(
    id="OP_2.2",
    label="115 V / 60 Hz / 1000 W / 40 °C (low-line, nominal load)",
    V_ac=115.0, f_line=60.0, P_in_target=1000.0,
    T_amb=40.0, speed_rpm=3500.0,
)
OP_23 = OperatingPoint(
    id="OP_2.3",
    label="220 V / 50 Hz / 1090 W / 50 °C (high-line, nominal load)",
    V_ac=220.0, f_line=50.0, P_in_target=1090.0,
    T_amb=50.0, speed_rpm=3500.0,
)
OP_24 = OperatingPoint(
    id="OP_2.4",
    label="220 V / 50 Hz / 1023 W / 50 °C (high-line, current-limited)",
    V_ac=220.0, f_line=50.0, P_in_target=1023.0,
    T_amb=50.0, speed_rpm=3500.0,
    # Spec nameplate is 1400 W "max load" but PSIM only delivers
    # 1023 W (= KPI_24.P_in) at this OP — the boost current loop
    # saturates well short of the rated value. We use PSIM's
    # *measured* power as the target so our cascade pumps the same
    # amount and the loss budget tracks PSIM directly.
)
OPERATING_POINTS = (OP_22, OP_23, OP_24)


# ---------------------------------------------------------------------------
# KPI matrix
# ---------------------------------------------------------------------------
#
# For each operating point we record the values the PSIM run produced.
# Naming convention copies the spreadsheet's ``Variable`` column verbatim
# so a reader can locate the cell that fed each entry.
#
# Three kinds of statistic are reported in the spreadsheet:
#
#   RMS  — root-mean-square over the simulation window
#   PK   — peak absolute value
#   AVG  — average over the simulation window
#
# When the spreadsheet leaves a cell blank we publish ``None`` here
# rather than fabricate a value. The OP_2.2 RMS column is dense; the
# PK column is sparse and only populated for the components whose
# datasheet limit is peak-based.
#
# All numbers are SI: V, A, W, °C, RPM.


@dataclass
class KpiSet:
    """Per-OP collection of PSIM-reported numbers."""

    op: OperatingPoint

    # --- High-level signals ---
    V_in_rms: Optional[float] = None
    V_link_avg: Optional[float] = None    # DC bus voltage (target ~378 V)
    I_in_rms: Optional[float] = None
    P_in: Optional[float] = None
    PF: Optional[float] = None             # power factor at AC input

    # --- Boost (PFC) stage currents ---
    # I_a2 / I_b2 :  boost-leg branch currents (2 parallel MOSFET legs)
    I_a2_rms: Optional[float] = None
    I_b2_rms: Optional[float] = None
    # S2.ID001        : rectifier bridge current (RMS, PK)
    I_D001_rms: Optional[float] = None
    I_D001_pk:  Optional[float] = None
    I_D001a_rms: Optional[float] = None
    I_D001b_rms: Optional[float] = None
    # I(S2.C006)      : input filter cap RMS
    I_C006_rms: Optional[float] = None
    # I(S2.L002)      : boost inductor RMS
    I_L002_rms: Optional[float] = None
    # S2.I_T001 / T002: boost MOSFET RMS, PK
    I_T001_rms: Optional[float] = None
    I_T001_pk:  Optional[float] = None
    I_T002_rms: Optional[float] = None
    I_T002_pk:  Optional[float] = None
    # I(S2.R003)      : boost shunt RMS
    I_R003_rms: Optional[float] = None
    # S2.I_SHUNT      : combined shunt sense current
    I_shunt_rms: Optional[float] = None
    I_shunt_pk:  Optional[float] = None
    # S2.ID002 / Psw  : boost output diode RMS / PK (SiC, low Qrr)
    I_D002_avg: Optional[float] = None
    I_D002_pk:  Optional[float] = None

    # --- DC link capacitor currents (two 220 µF in parallel) ---
    I_Cbus_rms: Optional[float] = None    # combined
    I_C009_rms: Optional[float] = None
    I_C010_rms: Optional[float] = None

    # --- Inverter (S1) ---
    speed_rpm: Optional[float] = None     # achieved compressor speed
    # Body diode + IGBT collector currents per leg (PSIM names)
    I_Q800_avg: Optional[float] = None
    I_Q800_pk:  Optional[float] = None
    I_Q803_avg: Optional[float] = None
    I_Q803_pk:  Optional[float] = None
    I_D_Q800_avg: Optional[float] = None
    I_D_Q803_avg: Optional[float] = None
    I_F500_rms: Optional[float] = None     # fuse current
    I_F500_pk:  Optional[float] = None
    I_R508_rms: Optional[float] = None     # inverter shunt

    # --- Losses [W] per device (matches Excel "Losses Calculations" block) ---
    P_cond_D001: Optional[float] = None
    P_cond_T1:   Optional[float] = None    # = P_cond_T001
    P_sw_T1:     Optional[float] = None    # = P_sw_T001
    P_cond_T2:   Optional[float] = None    # = P_cond_T002
    P_sw_T2:     Optional[float] = None    # = P_sw_T002
    P_cond_D002: Optional[float] = None
    P_sw_D002:   Optional[float] = None
    P_IC500_total: Optional[float] = None  # IPM total dissipation
    P_ohm_L002:  Optional[float] = None    # boost L DCR
    P_mag_L002:  Optional[float] = None    # boost L core loss
    P_ohm_L001:  Optional[float] = None    # input choke DCR
    P_esr_C009:  Optional[float] = None
    P_esr_C010:  Optional[float] = None
    P_R002:      Optional[float] = None
    P_R003:      Optional[float] = None
    P_R036:      Optional[float] = None
    P_R508:      Optional[float] = None
    P_K001_coil: Optional[float] = None    # relay coil
    P_K500_coil: Optional[float] = None
    P_K300_coil: Optional[float] = None
    P_SMPS:      Optional[float] = None    # auxiliary SMPS overhead
    P_total:     Optional[float] = None    # sum of all the above
    eta_inverter: Optional[float] = None   # P_motor / P_in  (–)

    # --- Junction / case / heatsink temperatures [°C] ---
    T_J_D001:  Optional[float] = None
    T_C_D001:  Optional[float] = None
    T_HS_D001: Optional[float] = None
    T_J_T001:  Optional[float] = None
    T_C_T001:  Optional[float] = None
    T_HS_T001: Optional[float] = None
    T_J_D002:  Optional[float] = None
    T_C_D002:  Optional[float] = None
    T_HS_D002: Optional[float] = None
    T_J_IGBT_IC500: Optional[float] = None
    T_J_D_IC500:    Optional[float] = None
    T_C_IC500:      Optional[float] = None
    T_HS_IC500:     Optional[float] = None


# ---------------------------------------------------------------------------
# OP 2.2 — 115 V / 1000 W / 40 °C
# ---------------------------------------------------------------------------

KPI_22 = KpiSet(
    op=OP_22,
    V_in_rms=114.66,
    I_in_rms=9.041,        # Iin RMS
    I_a2_rms=3.838,
    I_b2_rms=3.741,
    I_D001_rms=9.015,
    I_D001_pk=16.3,
    I_C006_rms=0.3122,
    I_L002_rms=9.003,
    I_T001_rms=3.652,
    I_T001_pk=8.07,
    I_T002_rms=3.652,
    I_T002_pk=8.07,
    I_R003_rms=3.001,
    I_shunt_rms=9.003,
    I_shunt_pk=16.6,
    I_D002_pk=16.6,
    I_Cbus_rms=5.161,
    I_C009_rms=2.581,
    I_C010_rms=2.581,
    I_Q800_pk=5.77,
    I_Q803_pk=5.77,
    I_D_Q800_avg=None,
    I_D_Q803_avg=None,
    I_F500_rms=3.382,
    I_F500_pk=5.8,
    I_R508_rms=3.382,
    # Losses (W)
    P_ohm_L002=4.053,
    P_mag_L002=3.3,
    P_ohm_L001=8.991,
    P_esr_C009=1.130,
    P_esr_C010=1.130,
    P_R002=0.5404,
    P_R003=0.5404,
    P_R036=0.5404,
    P_R508=0.1144,
    P_total=70.870,
    eta_inverter=0.9302,
    # Temperatures peak limits (datasheet)
    T_J_D001=147.0,
    T_J_T001=178.0,
    T_J_D002=128.0,
)


# ---------------------------------------------------------------------------
# OP 2.3 — 220 V / 1090 W / 50 °C
# ---------------------------------------------------------------------------

KPI_23 = KpiSet(
    op=OP_23,
    V_in_rms=219.35,
    V_link_avg=378.98,
    I_in_rms=5.057,
    P_in=1014.61,
    PF=0.97933,
    I_a2_rms=3.952,
    I_b2_rms=4.012,
    I_D001_rms=4.984,
    I_D001a_rms=3.987,
    I_D001b_rms=3.991,
    I_C006_rms=0.3106,
    I_L002_rms=4.948,
    I_T001_rms=1.419,
    I_T002_rms=1.419,
    I_R003_rms=1.649,
    I_shunt_rms=4.948,
    I_D002_avg=2.539,
    I_Cbus_rms=3.890,
    I_C009_rms=1.945,
    I_C010_rms=1.945,
    speed_rpm=3883.45,
    I_Q800_avg=1.010,
    I_Q803_avg=1.522,
    I_D_Q800_avg=-0.2182,
    I_D_Q803_avg=-0.5600,
    I_F500_rms=3.514,
    I_R508_rms=3.514,
    # Losses (W)
    P_cond_D001=15.581,
    P_cond_T1=4.265,
    P_sw_T1=4.333,
    P_cond_T2=4.265,
    P_sw_T2=4.333,
    P_cond_D002=3.545,
    P_sw_D002=0.1294,
    P_IC500_total=11.938,
    P_ohm_L002=1.224,
    P_mag_L002=3.3,
    P_ohm_L001=2.813,
    P_esr_C009=0.8515,
    P_esr_C010=0.8515,
    P_R002=0.1632,
    P_R003=0.1632,
    P_R036=0.1632,
    P_R508=0.1234,
    P_K001_coil=0.5714,
    P_K500_coil=0.2857,
    P_K300_coil=0.2857,
    P_SMPS=1.0,
    P_total=41.787,
    eta_inverter=0.95916,
    # Temperatures (°C)
    T_J_D001=102.11,
    T_C_D001=86.52,
    T_HS_D001=76.30,
    T_J_T001=115.89,
    T_C_T001=79.78,
    T_HS_T001=63.73,
    T_J_D002=76.14,
    T_C_D002=60.71,
    T_HS_D002=53.85,
    T_J_IGBT_IC500=67.00,
    T_J_D_IC500=64.78,
    T_C_IC500=64.23,
    T_HS_IC500=53.85,
)


# ---------------------------------------------------------------------------
# OP 2.4 — 220 V / 1400 W / 50 °C  (max load)
# ---------------------------------------------------------------------------

KPI_24 = KpiSet(
    op=OP_24,
    V_in_rms=219.35,
    V_link_avg=377.90,
    I_in_rms=6.784,
    P_in=1023.28,
    PF=0.92268,
    I_a2_rms=4.838,
    I_b2_rms=4.929,
    I_D001_rms=6.723,
    I_D001a_rms=2.105,
    I_D001b_rms=2.105,
    I_C006_rms=0.3106,
    I_L002_rms=4.948,
    I_T001_rms=1.419,
    I_T002_rms=1.419,
    I_R003_rms=1.649,
    I_shunt_rms=4.948,
    I_D002_avg=2.657,
    I_Cbus_rms=3.890,
    I_C009_rms=1.945,
    I_C010_rms=1.945,
    speed_rpm=3798.43,
    I_Q800_avg=1.084,
    I_Q803_avg=1.477,
    I_D_Q800_avg=-0.2170,
    I_D_Q803_avg=-0.6061,
    I_F500_rms=3.514,
    I_R508_rms=3.514,
    # Losses (W)
    P_cond_D001=7.735,
    P_cond_T1=0.6438,
    P_sw_T1=1.664,
    P_cond_T2=0.6438,
    P_sw_T2=1.664,
    P_cond_D002=3.066,
    P_sw_D002=0.0249,
    P_IC500_total=12.720,
    P_ohm_L002=1.224,
    P_mag_L002=3.3,
    P_ohm_L001=2.813,
    P_esr_C009=0.8515,
    P_esr_C010=0.8515,
    P_R002=0.1632,
    P_R003=0.1632,
    P_R036=0.1632,
    P_R508=0.1234,
    P_K001_coil=0.5714,
    P_K500_coil=0.2857,
    P_K300_coil=0.2857,
    P_SMPS=1.0,
    P_total=41.787,
    eta_inverter=0.95916,
    # Temperatures (°C)
    T_J_D001=80.83,
    T_C_D001=73.10,
    T_HS_D001=68.02,
    T_J_T001=70.37,
    T_C_T001=60.68,
    T_HS_T001=56.37,
    T_J_D002=80.40,
    T_C_D002=67.42,
    T_HS_D002=61.65,
    T_J_IGBT_IC500=78.81,
    T_J_D_IC500=76.43,
    T_C_IC500=75.82,
    T_HS_IC500=64.75,
)


# Single-point KPI lookup dict.
KPI_BY_OP_ID: Dict[str, KpiSet] = {
    "OP_2.2": KPI_22,
    "OP_2.3": KPI_23,
    "OP_2.4": KPI_24,
}

ALL_KPI = (KPI_22, KPI_23, KPI_24)
