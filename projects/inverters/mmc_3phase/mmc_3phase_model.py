"""3-φ MMC DC/AC inverter — analytical helpers + plant builders.

This module is imported by the notebook(s) in this folder:

  * ``01_mmc_validation_gean.ipynb`` — model the MMC, design open-loop
    modulation, simulate with pulsim, and validate against the
    experimental run from Section 4.1 of Sousa's thesis
    (UFSC PhD, 2022).

The thesis prototype runs at:

  * ``S = 15 kVA``, ``V_cc = 640 V`` (DC bus), ``V̂ = 272 V`` (AC peak)
  * ``N = 5`` SMs per arm, ``V_CSM = 128 V`` (each cap target)
  * ``C_SM = 470 µF`` per submodule
  * ``L_b ≈ 1 mH`` (typical, used as a tuning knob below)
  * RL load (Y-connected): ``R_load = 9.75 Ω``, ``L_load = 2.8 mH``
    (R nominal 9.2 Ω + parasitic; L nominal 1.9 mH + 0.9 mH parasitic)
  * ``T_d = T_m = 5 µs`` (dead-time and minimum pulse width)
  * ``f_grid = 60 Hz``
  * ``M = 0.85`` (modulation depth ⇒ V̂ = M·V_cc/2 = 272 V)
  * Modulation: In-Phase Disposition (IPD)

The thesis paper reports the following key metrics (Tabela 4.2,
open-loop run, 1 µs sim step):

  * ``THD(i_a) = 0.706 %`` (with dead-time, sim 1)
  * ``RMS(i_ca) = 4.55 A``  (circulating-current, with dead-time)
  * ``RMS(CA(i_cc)) = 1.14 A``  (DC-bus current AC component)

Pulsim's L1 PS-PWM modulator is NOT IPD, so the *instantaneous*
switching waveforms differ from the thesis figures. However the
*averaged* quantities — cap-voltage means and ripples, AC port-current
RMS, DC-bus mean — should match within a few percent. The notebook
walks through this comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt
from typing import Callable

import numpy as np

import pulsim as p


# =============================================================================
# Operating point — values from Tabela 4.1 / Section 4.1 of the thesis
# =============================================================================


@dataclass(frozen=True)
class GeanThesisParams:
    """Open-loop operating point from Section 4.1 of Sousa (2022)."""

    # DC + AC ratings
    V_dc: float = 640.0          # DC bus voltage [V]
    V_ac_peak: float = 272.0     # target AC phase-voltage peak [V]
    f_grid: float = 60.0         # AC fundamental [Hz]

    # MMC topology
    n_sm: int = 5                # SMs per arm
    c_sm: float = 470e-6         # per-SM capacitance [F]
    l_b: float = 1.0e-3          # arm inductance [H] (tuning knob —
                                 #  thesis prototype value not pinned
                                 #  in the open-loop section)
    r_b: float = 0.675           # arm-side parasitic resistance [Ω]
                                 #  (from Section 4.1 — used by sim 1
                                 #  to match the experimental damping)

    # AC-side RL load (Y-connected, per phase)
    r_load: float = 9.75         # [Ω]  (9.2 Ω nominal + parasitic)
    l_load: float = 2.8e-3       # [H]  (1.9 mH nominal + 0.9 mH parasitic)

    # IGBT non-idealities
    t_dead: float = 5e-6         # dead-time [s]
    t_min: float = 5e-6          # minimum pulse width [s]

    # Modulation
    m_depth: float = 0.85        # M = V̂ / (V_dc / 2)
    f_carrier: float = 1800.0    # carrier per SM [Hz]
    modulation_scheme: str = "ipd"  # "ipd" (thesis) or "ps_pwm"

    # Initial conditions
    v_c0: float | None = None    # capacitor-sum IC; default = V_dc.

    @property
    def v_c_init(self) -> float:
        return self.V_dc if self.v_c0 is None else float(self.v_c0)

    @property
    def omega_grid(self) -> float:
        return 2.0 * pi * self.f_grid


# =============================================================================
# Reference signals — IPD-equivalent sinusoidal modulation for HB MMC
# =============================================================================


def make_phase_mref_fns(
    params: GeanThesisParams,
) -> "tuple[Callable[[float], float], Callable[[float], float], Callable[[float], float]]":
    """Return three modulation references ``(m_a_p_ref, m_b_p_ref, m_c_p_ref)``
    for the *upper* arms of an MMC inverter in open-loop operation.

    Convention (matches our 3-φ MMC integration tests):

      * Upper-arm ``m_X_p = 0.5 − v_X_ac / V_dc``
      * Lower-arm ``m_X_n = 0.5 + v_X_ac / V_dc``  (1 − m_X_p for HB)

    where ``v_X_ac`` is the desired AC phase voltage relative to the
    bus midpoint. For a balanced 3-φ sinusoidal output:

      ``v_X_ac(t) = (M · V_dc / 2) · cos(ω·t − φ_X)``

    with ``φ_a = 0``, ``φ_b = 2π/3``, ``φ_c = 4π/3``.
    """
    omega = params.omega_grid
    v_peak = 0.5 * params.m_depth * params.V_dc  # = M · V_dc / 2

    def m_a_p(t: float) -> float:
        v_a = v_peak * cos(omega * t)
        return 0.5 - v_a / params.V_dc

    def m_b_p(t: float) -> float:
        v_b = v_peak * cos(omega * t - 2.0 * pi / 3.0)
        return 0.5 - v_b / params.V_dc

    def m_c_p(t: float) -> float:
        v_c = v_peak * cos(omega * t + 2.0 * pi / 3.0)
        return 0.5 - v_c / params.V_dc

    return m_a_p, m_b_p, m_c_p


# =============================================================================
# Plant builders — three layers of fidelity
# =============================================================================


@dataclass
class MmcPlant:
    """Bundle of (builder, six-arm list, filter-inductor indices)
    returned by the plant builders below."""

    builder: object
    arms: list[object] = field(default_factory=list)
    iL_indices: tuple[int, int, int] = (0, 0, 0)


def build_l1_plant(params: GeanThesisParams) -> MmcPlant:
    """3-φ MMC inverter using L1 PS-PWM multilevel arms (no dead-time)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "dc_n", params.V_dc)

    m_a, m_b_, m_c = make_phase_mref_fns(params)

    arm_params = p.MmcArmMultilevelParams(
        n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
        f_carrier=params.f_carrier,
        modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
    )

    arms: list[object] = []
    # Upper arms + upper arm inductors.
    upper_refs = (m_a, m_b_, m_c)
    for k, ph in enumerate("abc"):
        arm_p = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=arm_params, m_ref=upper_refs[k],
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"rb_{ph}_p", params.l_b)
        b.add_resistor(f"Rb_{ph}_p", f"rb_{ph}_p", f"ac_{ph}", params.r_b)

    # Lower arm inductors + lower arms (complement modulation).
    def _complement(f):
        return lambda t, _f=f: 1.0 - float(_f(t))

    lower_refs = tuple(_complement(f) for f in upper_refs)
    for k, ph in enumerate("abc"):
        b.add_resistor(f"Rb_{ph}_n", f"ac_{ph}", f"rb_{ph}_n", params.r_b)
        b.add_inductor(f"Lb_{ph}_n", f"rb_{ph}_n", f"mid_{ph}_n", params.l_b)
        arm_n = p.add_mmc_arm_multilevel(
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="dc_n",
            params=arm_params, m_ref=lower_refs[k],
        )
        arms.append(arm_n)

    # Y-connected RL load.
    iL_indices: list[int] = []
    for ph in "abc":
        l_id = b.graph.num_branches
        b.add_inductor(f"Lload_{ph}", f"ac_{ph}", f"rload_{ph}", params.l_load)
        b.add_resistor(f"R_{ph}", f"rload_{ph}", "star", params.r_load)
        iL_indices.append(
            b.pool.branch_var_id_for_inductor(l_id, b.graph),
        )
    # Weak star tie for MNA conditioning.
    b.add_resistor("R_star", "star", "dc_n", 1e6)

    return MmcPlant(builder=b, arms=arms,
                     iL_indices=(iL_indices[0], iL_indices[1], iL_indices[2]))


def build_l2_plant(params: GeanThesisParams) -> MmcPlant:
    """3-φ MMC inverter using L2 SM-equivalent arms (dead-time aware)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "dc_n", params.V_dc)

    m_a, m_b_, m_c = make_phase_mref_fns(params)

    arm_params = p.MmcArmEquivalentParams(
        n_sm=params.n_sm, c_sm=params.c_sm, v_c0=params.v_c_init,
        f_carrier=params.f_carrier,
        t_dead=params.t_dead, t_min=params.t_min,
        modulation_scheme=params.modulation_scheme,  # type: ignore[arg-type]
    )

    arms: list[object] = []
    upper_refs = (m_a, m_b_, m_c)
    for k, ph in enumerate("abc"):
        arm_p = p.add_mmc_arm_equivalent(
            b, name=f"A_{ph}_p",
            node_a="dc_p", node_b=f"mid_{ph}_p",
            params=arm_params, m_ref=upper_refs[k],
        )
        arms.append(arm_p)
        b.add_inductor(f"Lb_{ph}_p", f"mid_{ph}_p", f"rb_{ph}_p", params.l_b)
        b.add_resistor(f"Rb_{ph}_p", f"rb_{ph}_p", f"ac_{ph}", params.r_b)

    def _complement(f):
        return lambda t, _f=f: 1.0 - float(_f(t))

    lower_refs = tuple(_complement(f) for f in upper_refs)
    for k, ph in enumerate("abc"):
        b.add_resistor(f"Rb_{ph}_n", f"ac_{ph}", f"rb_{ph}_n", params.r_b)
        b.add_inductor(f"Lb_{ph}_n", f"rb_{ph}_n", f"mid_{ph}_n", params.l_b)
        arm_n = p.add_mmc_arm_equivalent(
            b, name=f"A_{ph}_n",
            node_a=f"mid_{ph}_n", node_b="dc_n",
            params=arm_params, m_ref=lower_refs[k],
        )
        arms.append(arm_n)

    iL_indices: list[int] = []
    for ph in "abc":
        l_id = b.graph.num_branches
        b.add_inductor(f"Lload_{ph}", f"ac_{ph}", f"rload_{ph}", params.l_load)
        b.add_resistor(f"R_{ph}", f"rload_{ph}", "star", params.r_load)
        iL_indices.append(
            b.pool.branch_var_id_for_inductor(l_id, b.graph),
        )
    b.add_resistor("R_star", "star", "dc_n", 1e6)

    return MmcPlant(builder=b, arms=arms,
                     iL_indices=(iL_indices[0], iL_indices[1], iL_indices[2]))


# =============================================================================
# Run drivers + metrics
# =============================================================================


@dataclass
class MmcRunResult:
    """Output of :func:`run_mmc_open_loop`."""

    t: np.ndarray
    i_a: np.ndarray
    i_b: np.ndarray
    i_c: np.ndarray
    v_b_a_p: np.ndarray              # arm-generated voltage, phase a upper
    v_C: np.ndarray                  # shape (6, n_samples) — per-arm cap sums
    arm_names: tuple[str, ...] = (
        "a_p", "b_p", "c_p", "a_n", "b_n", "c_n",
    )


def run_mmc_open_loop(
    plant: MmcPlant,
    *,
    t_end: float = 50e-3,
    dt: float = 5e-6,
    layer: str = "l1",
) -> MmcRunResult:
    """Run a plant produced by :func:`build_l1_plant` or
    :func:`build_l2_plant` for ``t_end`` seconds at ``dt`` step.

    Args:
        plant: Output of one of the ``build_*_plant`` helpers.
        t_end: Simulation horizon [s].
        dt: Fixed time step [s].
        layer: ``"l1"`` or ``"l2"`` — which observer factory to use.

    Returns:
        :class:`MmcRunResult` with the logged time series.
    """
    if layer == "l1":
        obs, bex = p.make_mmc_arm_multilevel_observers(
            plant.builder, plant.arms, dt=dt,  # type: ignore[arg-type]
        )
    elif layer == "l2":
        obs, bex = p.make_mmc_arm_equivalent_observers(
            plant.builder, plant.arms, dt=dt,  # type: ignore[arg-type]
        )
    else:
        raise ValueError(f"layer must be 'l1' or 'l2' (got {layer!r})")

    iLa, iLb, iLc = plant.iL_indices
    n_samples = int(round(t_end / dt)) + 1
    log_t   = np.zeros(n_samples)
    log_ia  = np.zeros(n_samples)
    log_ib  = np.zeros(n_samples)
    log_ic  = np.zeros(n_samples)
    log_vba = np.zeros(n_samples)
    log_vC  = np.zeros((6, n_samples))
    counter = [0]

    def log_obs(t, x):
        obs(t, x)
        i = counter[0]
        if i < n_samples:
            log_t[i]   = t
            log_ia[i]  = x[iLa]
            log_ib[i]  = x[iLb]
            log_ic[i]  = x[iLc]
            arms_list = plant.arms  # type: ignore[assignment]
            log_vba[i] = arms_list[0].v_b  # type: ignore[attr-defined]
            for k in range(6):
                log_vC[k, i] = arms_list[k].v_C  # type: ignore[attr-defined]
        counter[0] += 1

    p.simulate(
        plant.builder, t_end=t_end, dt=dt,  # type: ignore[arg-type]
        step_observer=log_obs, b_extra_fn=bex,
        start_from_dc_op=True,
    )

    n = counter[0]
    return MmcRunResult(
        t=log_t[:n], i_a=log_ia[:n], i_b=log_ib[:n], i_c=log_ic[:n],
        v_b_a_p=log_vba[:n], v_C=log_vC[:, :n],
    )


def thd(signal: np.ndarray, fs: float, f0: float, n_harm: int = 50) -> float:
    """Total harmonic distortion of ``signal`` at fundamental ``f0`` [%].

    Computes ``THD = sqrt(sum H_k²) / H_1 × 100 %`` over ``2..n_harm``.
    Uses a Hann window + rfft.
    """
    sig = np.asarray(signal, dtype=np.float64)
    sig = sig - sig.mean()
    n = len(sig)
    win = np.hanning(n)
    spec = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    # Find the bin closest to the fundamental.
    k1 = int(round(f0 / (fs / n)))
    if k1 < 1:
        return float("nan")
    fund = abs(spec[k1])
    harmonics_sq = 0.0
    for k in range(2, n_harm + 1):
        ki = k * k1
        if ki < len(spec):
            harmonics_sq += abs(spec[ki]) ** 2
    return float(100.0 * sqrt(harmonics_sq) / fund) if fund > 0 else float("nan")


def circulating_current(arm_p_branch_currents: np.ndarray,
                        arm_n_branch_currents: np.ndarray) -> np.ndarray:
    """Circulating current = (i_arm_p + i_arm_n) / 2 (Sousa eq 2.22).

    Both inputs are per-phase arm currents; the DC component of the
    sum equals the DC-port contribution; the AC component is the
    circulating part. Average of the two arms removes the AC port
    current entirely.
    """
    return 0.5 * (arm_p_branch_currents + arm_n_branch_currents)


def rms(signal: np.ndarray) -> float:
    sig = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(sig**2)))


def rms_ac(signal: np.ndarray) -> float:
    """RMS of the AC component (signal minus its DC mean)."""
    sig = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean((sig - sig.mean()) ** 2)))
