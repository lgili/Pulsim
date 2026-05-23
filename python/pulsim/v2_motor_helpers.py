"""Pulsim v2 — high-level motor-control wiring helpers.

The v2 chain executor (``MixedDomainBlockChain``) exposes every
primitive needed for vector control (Clarke, Park, SVM, PI, PWM,
inverse Park…), but assembling a complete drive from those blocks
takes 15-20 lines per motor that all look the same. These helpers
condense the boilerplate into one call.

Currently provides:

* :func:`wire_pmsm_foc` — assembles a complete current-controlled
  PMSM FOC drive: stator-current → Clarke/Park → dq PI loops →
  inverse Park → SVM → three PWM gate channels. Returns the channel
  names + switch wiring info the user needs to connect to the
  inverter's switch_fn.

This is the v2 equivalent of v1's ``runtime_circuit.add_pmsm_foc``
device, but split into chain wiring (this helper) + power-stage
(use :func:`add_three_phase_vsi` + your PMSM motor block).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


__all__ = [
    "FocWiringResult",
    "wire_pmsm_foc",
]


@dataclass
class FocWiringResult:
    """Names of channels created by :func:`wire_pmsm_foc`. Use these
    to feed downstream blocks (PWM gates → switch_fn) and to read
    diagnostics (i_d / i_q for tuning, v_d / v_q for saturation
    detection).
    """
    # Output gate channels for the three high-side switches.
    gate_hs_channels: List[str] = field(default_factory=list)
    # Output gate channels for the three low-side switches (complement
    # of the HS gates, no dead-time inserted).
    gate_ls_channels: List[str] = field(default_factory=list)
    # Channel names of the measured Park currents (for plotting /
    # tuning).
    i_d_channel: str = ""
    i_q_channel: str = ""
    # Channel names of the dq voltage references (PI controller
    # outputs). Useful for detecting saturation.
    v_d_channel: str = ""
    v_q_channel: str = ""
    # Channel names of the per-phase modulator duty cycles (post-SVM,
    # before PWM comparator). Useful for understanding what the SVM
    # actually generated.
    duty_a_channel: str = ""
    duty_b_channel: str = ""
    duty_c_channel: str = ""


def wire_pmsm_foc(chain,
                     *,
                     i_a_channel: str,
                     i_b_channel: str,
                     i_c_channel: str,
                     theta_elec_channel: str,
                     v_dc: float,
                     f_pwm: float,
                     Kp_i: float,
                     Ki_i: float,
                     i_d_ref: "float | str" = 0.0,
                     i_q_ref: "float | str" = 0.0,
                     output_prefix: str = "foc",
                     v_sat: Optional[float] = None,
                     ) -> FocWiringResult:
    """Add the complete FOC control chain (Clarke→Park→PI→ipark→SVM→
    PWM) for a 3-phase PMSM.

    Topology (Block diagram)::

        i_a, i_b, i_c          ┌─ Clarke ─┐
                  ─────────────┤          ├──── i_α, i_β
                               └──────────┘
                                    │
                               ┌──── ▼ ────┐
                  θ_elec  ──── ┤    Park   ├──── i_d, i_q
                               └───────────┘
                                    │
              i_d_ref ──►(  )──►PI──►v_d  ─────►┐
                                                │ Inverse Park
              i_q_ref ──►(  )──►PI──►v_q  ─────►┘   (using θ_elec)
                                                │
                                                ▼
                                          ┌─────────┐
                                          │   SVM   │  v_α/v_β → d_a/d_b/d_c
                                          └─────────┘
                                                │
                                  ┌─────────────┼─────────────┐
                                  ▼             ▼             ▼
                              PWM(d_a)      PWM(d_b)      PWM(d_c)
                                  │             │             │
                              HS gate_a    HS gate_b    HS gate_c
                              LS gate_a    LS gate_b    LS gate_c
                              (1 - gate_a)  (...)        (...)

    Parameters
    ----------
    chain
        :class:`MixedDomainBlockChain` to append the blocks to. Must
        already have ``i_a/b/c_channel`` and ``theta_elec_channel``
        populated by upstream blocks (typically Clarke is on stator
        currents and theta_elec from a position sensor or
        observer).
    i_a_channel, i_b_channel, i_c_channel
        Names of the chain channels carrying the three phase currents.
    theta_elec_channel
        Name of the chain channel carrying the electrical rotor angle
        (rad). For a PMSM with ``pole_pairs`` P, this is P times the
        mechanical angle.
    v_dc
        DC bus voltage (V) — used to set the SVM normalisation and to
        clip the PI outputs at ±V_DC/2.
    f_pwm
        PWM carrier frequency (Hz).
    Kp_i, Ki_i
        Per-axis current-loop PI gains. Same gains used for d and q
        axes (the standard assumption for a balanced PMSM with
        Ld ≈ Lq).
    i_d_ref, i_q_ref
        Current references. Either a constant float (most common)
        or a channel name (e.g. from an outer speed loop). Default
        ``i_d_ref=0.0`` (zero d-axis current, max torque per amp
        for surface PMSM) and ``i_q_ref=0.0`` (no torque — the user
        almost always overrides this).
    output_prefix
        Prefix for the auto-created channel names. Default ``"foc"``;
        gate channels become ``foc.gate_hs_a``, ``foc.gate_ls_a``,
        etc. Internal channels: ``foc.i_d``, ``foc.i_q``, ``foc.v_d``,
        ``foc.v_q``, ``foc.duty_a``, …
    v_sat
        Optional saturation limit for the PI output (default
        ``V_DC / 2``). Same value used for both axes.

    Returns
    -------
    FocWiringResult
        Names of all created channels. Pass
        ``.gate_hs_channels`` (+ ``.gate_ls_channels`` if you want
        explicit complementary control) to
        :meth:`chain.make_multi_pwm_switch_fn` to wire the gates
        to the 3-phase VSI's switch indices.

    Example
    -------
    ::

        # 1. Power stage
        vsi = p.add_three_phase_vsi(b, "INV",
                  vdc_pos="vdc", vdc_neg="gnd",
                  out_a="ua", out_b="ub", out_c="uc")
        # ... motor / load wiring ...

        # 2. Sensors (e.g. shunt resistors → i_a/i_b/i_c, encoder
        #    → theta_elec). Add to chain.

        # 3. FOC chain in one call.
        foc = p.wire_pmsm_foc(chain,
                  i_a_channel="i_a", i_b_channel="i_b", i_c_channel="i_c",
                  theta_elec_channel="theta_elec",
                  v_dc=400.0, f_pwm=10e3,
                  Kp_i=0.5, Ki_i=2000.0,
                  i_q_ref=5.0)

        # 4. Wire FOC gates to inverter switches.
        observer = chain.make_step_observer(b, dt=DT)
        sw_fn = chain.make_multi_pwm_switch_fn(
            foc.gate_hs_channels,
            num_switches=b.graph.num_switches,
            switch_indices=vsi.high_side_switch_indices)
        # (Low-side complementary; wire via chain.make_multi_pwm_switch_fn
        # on .gate_ls_channels OR via a dead-time half-bridge helper.)
    """
    # Late import to avoid circular dependency at module load.
    from .v2_control import (  # type: ignore[import-not-found]
        ClarkeTransform, ParkTransform, InverseParkTransform,
        SpaceVectorModulator, PIController, PwmGenerator)

    pfx = output_prefix
    v_sat_eff = float(v_sat if v_sat is not None else v_dc / 2.0)

    # Resolve setpoint references — accept either a float (constant)
    # or a channel name (from an outer loop).
    def _setpoint(ref):
        if isinstance(ref, str):
            return f"channel:{ref}"
        return float(ref)

    # ---------- Clarke ----------
    chain.add(f"{pfx}.clarke", ClarkeTransform(),
                inputs=dict(a=f"channel:{i_a_channel}",
                              b=f"channel:{i_b_channel}",
                              c=f"channel:{i_c_channel}"),
                output=(f"{pfx}.i_alpha",
                          f"{pfx}.i_beta",
                          f"{pfx}.i_zero"))

    # ---------- Park ----------
    chain.add(f"{pfx}.park", ParkTransform(),
                inputs=dict(alpha=f"channel:{pfx}.i_alpha",
                              beta=f"channel:{pfx}.i_beta",
                              theta=f"channel:{theta_elec_channel}"),
                output=(f"{pfx}.i_d", f"{pfx}.i_q"))

    # ---------- dq PI loops ----------
    chain.add(f"{pfx}.pi_d",
                PIController(Kp=Kp_i, Ki=Ki_i,
                                output_min=-v_sat_eff,
                                output_max=+v_sat_eff),
                inputs=dict(setpoint=_setpoint(i_d_ref),
                              measured=f"channel:{pfx}.i_d",
                              dt="dt"),
                output=f"{pfx}.v_d")
    chain.add(f"{pfx}.pi_q",
                PIController(Kp=Kp_i, Ki=Ki_i,
                                output_min=-v_sat_eff,
                                output_max=+v_sat_eff),
                inputs=dict(setpoint=_setpoint(i_q_ref),
                              measured=f"channel:{pfx}.i_q",
                              dt="dt"),
                output=f"{pfx}.v_q")

    # ---------- Inverse Park ----------
    chain.add(f"{pfx}.ipark", InverseParkTransform(),
                inputs=dict(d=f"channel:{pfx}.v_d",
                              q=f"channel:{pfx}.v_q",
                              theta=f"channel:{theta_elec_channel}"),
                output=(f"{pfx}.v_alpha", f"{pfx}.v_beta"))

    # ---------- SVM ----------
    chain.add(f"{pfx}.svm",
                SpaceVectorModulator(v_dc=v_dc),
                inputs=dict(v_alpha=f"channel:{pfx}.v_alpha",
                              v_beta=f"channel:{pfx}.v_beta"),
                output=(f"{pfx}.duty_a",
                          f"{pfx}.duty_b",
                          f"{pfx}.duty_c"))

    # ---------- 3 PWM generators (high-side gates) ----------
    for leg in ("a", "b", "c"):
        chain.add(f"{pfx}.pwm_{leg}",
                    PwmGenerator(frequency=f_pwm),
                    inputs=dict(duty=f"channel:{pfx}.duty_{leg}",
                                  t="time"),
                    output=f"{pfx}.gate_hs_{leg}")

    # ---------- Complementary low-side gates (no dead-time) ----------
    # Use a tiny lambda-like block: 1 − HS. The block-chain executor
    # supports arbitrary user blocks with .reset() + .update(x).
    class _Complement:
        def reset(self) -> None:
            pass

        def update(self, x) -> float:
            return 1.0 - float(x)

    for leg in ("a", "b", "c"):
        chain.add(f"{pfx}.ls_{leg}", _Complement(),
                    inputs=dict(x=f"channel:{pfx}.gate_hs_{leg}"),
                    output=f"{pfx}.gate_ls_{leg}")

    return FocWiringResult(
        gate_hs_channels=[f"{pfx}.gate_hs_a",
                            f"{pfx}.gate_hs_b",
                            f"{pfx}.gate_hs_c"],
        gate_ls_channels=[f"{pfx}.gate_ls_a",
                            f"{pfx}.gate_ls_b",
                            f"{pfx}.gate_ls_c"],
        i_d_channel=f"{pfx}.i_d",
        i_q_channel=f"{pfx}.i_q",
        v_d_channel=f"{pfx}.v_d",
        v_q_channel=f"{pfx}.v_q",
        duty_a_channel=f"{pfx}.duty_a",
        duty_b_channel=f"{pfx}.duty_b",
        duty_c_channel=f"{pfx}.duty_c",
    )
