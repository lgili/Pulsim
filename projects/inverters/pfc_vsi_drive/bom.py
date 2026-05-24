"""PFC-VSI compressor drive — Bill of Materials (BoM) for the power stage.

Each entry mirrors a real part on the reference prototype, with the
parameters the Pulsim builder needs to mirror PSIM's behaviour.

Sources:
  * the reference PSIM simulation, sheet
    ``1-Modeling`` (component values) + ``4-Design Margins``
    (datasheet snapshots).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diode:
    """Piecewise-linear ideal diode model parameters."""

    ref: str
    pn: str
    manufacturer: str
    V_blocking_max: float       # datasheet VRRM [V]
    I_avg_max:      float       # IF(av) [A]
    I_fsm:          Optional[float]   # peak forward surge [A]
    V_F:            float = 0.7       # forward drop @ rated current [V]
    R_on:           float = 50e-3     # on-state slope [Ω]
    R_off:          float = 1e9       # off-state shunt [Ω]
    T_j_max:        float = 150.0     # max junction temp [°C]


@dataclass(frozen=True)
class Mosfet:
    """Conduction + switching parameters for a Si MOSFET."""

    ref: str
    pn: str
    manufacturer: str
    V_DS_max:    float
    I_D_cont:    float
    R_DS_on:     float       # on resistance at 25 °C [Ω]
    V_GS_th:     float = 4.0
    Q_g:         Optional[float] = None   # total gate charge [nC]
    E_on:        Optional[float] = None   # turn-on energy [J] @ rated
    E_off:       Optional[float] = None   # turn-off energy [J]
    V_F_body:    float = 0.9              # body diode V_F
    T_j_max:     float = 150.0


@dataclass(frozen=True)
class IgbtIpm:
    """Three-phase IGBT IPM (e.g. Infineon CIPOS family)."""

    ref: str
    pn: str
    manufacturer: str
    V_CES_max:    float        # collector-emitter blocking [V]
    I_C_cont:     float        # continuous collector current [A]
    V_CE_sat:     float = 1.7  # IGBT V_CE(sat) [V]
    V_F_diode:    float = 1.5  # body diode V_F [V]
    E_on:         Optional[float] = None
    E_off:        Optional[float] = None
    E_rec:        Optional[float] = None  # body-diode reverse recovery [J]
    T_j_max:      float = 150.0


@dataclass(frozen=True)
class Inductor:
    """Power inductor."""

    ref: str
    L:     float          # inductance [H]
    DCR:   float          # DC resistance [Ω]
    P_core_nom: float = 0.0  # nominal core loss [W] (optional)


@dataclass(frozen=True)
class CapacitorElectro:
    """Electrolytic capacitor with frequency-dependent ESR."""

    ref: str
    C:      float        # capacitance [F]
    V_rated: float
    # ESR at three reference frequencies (from the Excel model sheet).
    esr_at_100hz:  float = 0.0
    esr_at_5khz:   float = 0.0
    esr_at_65khz:  float = 0.0
    T_max:  float = 105.0  # operating temp [°C]


@dataclass(frozen=True)
class Resistor:
    """Discrete resistor — typically a shunt or snubber."""

    ref: str
    R:        float
    P_rated:  float
    T_max:    float = 155.0


@dataclass(frozen=True)
class Fuse:
    """Fast-acting fuse — modelled as a wire resistance + I²t budget."""

    ref: str
    I_rated:    float
    V_rated:    float
    R_cold:     float = 5e-3     # cold resistance [Ω]
    I_t_rating: Optional[float] = None  # I²t [A²·s]


# ---------------------------------------------------------------------------
# Concrete BoM — values from the spreadsheet
# ---------------------------------------------------------------------------


# Rectifier
D001 = Diode(
    ref="D001", pn="GBU2506", manufacturer="MCC",
    V_blocking_max=600.0, I_avg_max=25.0, I_fsm=300.0,
    V_F=1.0, R_on=20e-3,    # GBU2506 typ V_F @ 12.5A ≈ 1.05V
    T_j_max=150.0,
)

# Input choke (line side)
L001 = Inductor(ref="L001", L=1.0e-3, DCR=0.11, P_core_nom=0.5)

# Input filter cap (small, X-cap)
C006 = CapacitorElectro(
    ref="C006", C=470e-9, V_rated=275.0,
    esr_at_100hz=0.2, esr_at_5khz=0.05, esr_at_65khz=0.02,
)

# Boost inductor (post-rectifier)
L002 = Inductor(ref="L002", L=1.4e-3, DCR=0.05, P_core_nom=3.3)

# Boost MOSFETs (parallel pair)
T001 = Mosfet(
    ref="T001", pn="STx Mosfet TO220-FP 600 V 0.168Ω",
    manufacturer="ST",
    V_DS_max=650.0, I_D_cont=12.0, R_DS_on=168e-3,
    Q_g=42e-9, E_on=120e-6, E_off=40e-6, V_F_body=0.9,
    T_j_max=150.0,
)
T002 = Mosfet(
    ref="T002", pn="STx Mosfet TO220-FP 600 V 0.168Ω",
    manufacturer="ST",
    V_DS_max=650.0, I_D_cont=12.0, R_DS_on=168e-3,
    Q_g=42e-9, E_on=120e-6, E_off=40e-6, V_F_body=0.9,
    T_j_max=150.0,
)

# Boost output diode (SiC Schottky — fast, near-zero Qrr)
D002 = Diode(
    ref="D002", pn="STPSC10H065", manufacturer="ST",
    V_blocking_max=650.0, I_avg_max=10.0, I_fsm=80.0,
    V_F=1.4, R_on=80e-3,     # STPSC10H065 typ V_F @ 10A ≈ 1.4V
    T_j_max=175.0,
)

# DC bus electrolytics (parallel pair, 220 µF / 450 V each)
C009 = CapacitorElectro(
    ref="C009", C=220e-6, V_rated=450.0,
    esr_at_100hz=1.447,  # from spreadsheet
    esr_at_5khz=0.029,
    esr_at_65khz=0.00223,
    T_max=105.0,
)
C010 = CapacitorElectro(
    ref="C010", C=220e-6, V_rated=450.0,
    esr_at_100hz=1.447, esr_at_5khz=0.029, esr_at_65khz=0.00223,
    T_max=105.0,
)

# Inverter shunt + boost shunts
R508 = Resistor(ref="R508", R=10e-3, P_rated=0.75)
R002 = Resistor(ref="R002", R=60e-3, P_rated=2.0)
R003 = Resistor(ref="R003", R=60e-3, P_rated=2.0)
R036 = Resistor(ref="R036", R=60e-3, P_rated=2.0)

# Fuse on inverter return path
F500 = Fuse(ref="F500", I_rated=8.0, V_rated=250.0, R_cold=8e-3)

# Three-phase IGBT IPM (drives the compressor)
IC500 = IgbtIpm(
    ref="IC500", pn="IKCM20L60GD", manufacturer="Infineon",
    V_CES_max=600.0, I_C_cont=20.0,
    V_CE_sat=1.7, V_F_diode=1.5,
    E_on=300e-6, E_off=300e-6, E_rec=150e-6,
    T_j_max=150.0,
)

# Compressor mechanical parameters (rough — refined by validation)
COMPRESSOR_PARAMS = dict(
    speed_rpm_nom=3500.0,
    pole_pairs=2,        # typical 4-pole BLDC for small fractional-HP
    Rs=1.5,              # stator resistance per phase [Ω]
    Ld=8e-3, Lq=8e-3,    # synchronous inductances [H]
    Ke=0.08,             # back-EMF constant [V/(rad/s)]
    J=2e-4,              # moment of inertia [kg·m²]
    B=1e-4,              # viscous friction [N·m/(rad/s)]
    T_load_at_3500rpm=1.0,    # load torque @ rated [N·m]
)
