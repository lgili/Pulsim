"""Synchronous buck converter as a linear ODE between events.

State vector x = (i_L, v_C) ∈ R²
- i_L : inductor current (A)
- v_C : output capacitor voltage (V), also = V_out

Mask: (HS_state, LS_state) — complementary in CCM
- (True,  False) "HS_on"  : inductor sees V_in - v_C
- (False, True ) "HS_off" : inductor sees -v_C (low-side switch returns inductor current to ground)

CCM equations:

  L · di_L/dt = v_L
  C · dv_C/dt = i_L - v_C / R_load

with v_L = V_in - v_C  (HS on)  or  v_L = -v_C  (HS off).

In matrix form (state = [i_L, v_C]):

  ẋ = A · x + b

  A = [[0,        -1/L],
       [1/C,    -1/(R·C)]]

  b_HSon  = [V_in/L, 0]ᵀ
  b_HSoff = [0,      0]ᵀ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BuckParams:
    """Synchronous buck operating point + passives."""
    V_in: float = 24.0        # input DC voltage (V)
    L: float = 100e-6         # inductor (H)
    C: float = 100e-6         # output capacitor (F)
    R_load: float = 2.4       # output load resistance (Ω)
    f_sw: float = 100e3       # switching frequency (Hz)
    D: float = 0.5            # duty ratio (HS on-time / period)

    @property
    def V_out_steady(self) -> float:
        """Analytical CCM steady-state V_out."""
        return self.D * self.V_in

    @property
    def I_L_steady(self) -> float:
        """Analytical CCM steady-state mean inductor current."""
        return self.V_out_steady / self.R_load

    @property
    def T_sw(self) -> float:
        return 1.0 / self.f_sw


class BuckCCMModel:
    """Two-state linear ODE for sync-buck in CCM.

    Mask convention: ``True`` means HS is ON (LS is OFF, complementary).
    """

    def __init__(self, params: BuckParams) -> None:
        self.p = params
        # Both masks share the same A matrix; only b changes
        # (low-side and high-side LC topology is identical, only the
        # voltage driving the inductor changes).
        self._A = np.array(
            [
                [0.0, -1.0 / params.L],
                [1.0 / params.C, -1.0 / (params.R_load * params.C)],
            ],
            dtype=float,
        )
        self._b_HSon = np.array([params.V_in / params.L, 0.0], dtype=float)
        self._b_HSoff = np.array([0.0, 0.0], dtype=float)

        # Current state
        self._mask: bool = True
        self._A_cur = self._A
        self._b_cur = self._b_HSon

    @property
    def current_mask(self) -> bool:
        return self._mask

    def set_mask(self, mask: bool) -> None:
        """Set the active mask. ``True`` = HS on, ``False`` = HS off."""
        self._mask = bool(mask)
        self._b_cur = self._b_HSon if self._mask else self._b_HSoff

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Compute ẋ = A·x + b at the current mask. ``t`` unused (LTI)."""
        return self._A_cur @ x + self._b_cur


class BuckPSCSwitchFn:
    """Switch function for a buck converter with PSC-PWM-like
    fixed-duty modulation.

    ``T_sw`` is the period (s); within each period, HS is on for
    ``D·T_sw`` and off for ``(1-D)·T_sw``. Exposes
    ``next_edge_after(t)`` so the scheduler can predict gate edges
    analytically (no root-finding needed).
    """

    def __init__(self, T_sw: float, D: float) -> None:
        self.T_sw = T_sw
        self.D = D

    def __call__(self, t: float) -> bool:
        """Return mask (True=HS on) at time t."""
        phase = (t / self.T_sw) % 1.0
        return phase < self.D

    def next_edge_after(self, t: float) -> float:
        """Return the next gate-edge time strictly greater than ``t``."""
        # Edges occur at t = k·T_sw and t = k·T_sw + D·T_sw
        k = int(np.floor(t / self.T_sw))
        candidates = [
            k * self.T_sw + self.D * self.T_sw,
            (k + 1) * self.T_sw,
            (k + 1) * self.T_sw + self.D * self.T_sw,
        ]
        # Return the first candidate strictly greater than t (with tiny tolerance)
        eps = 1e-15
        for c in candidates:
            if c > t + eps:
                return c
        # Fallback (should never reach): two periods ahead
        return (k + 2) * self.T_sw
