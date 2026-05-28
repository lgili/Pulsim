"""Three-mode synchronous-buck model for Discontinuous Conduction Mode (DCM).

In **DCM** (light-load buck) each switching cycle goes through
three modes rather than the two of CCM:

  * **Mode A — HS on** : V_in drives the inductor; i_L ramps UP
        L · di_L/dt = V_in - v_C
        C · dv_C/dt = i_L - v_C / R_load

  * **Mode B — HS off, LS conducting (synchronous rectifier)** :
        L · di_L/dt = -v_C
        C · dv_C/dt = i_L - v_C / R_load
    This mode exits at the gate edge OR when i_L crosses zero
    (whichever comes first). In SR mode the channel conducts
    bidirectionally; the body-diode model below catches when
    we should "open" instead.

  * **Mode C — zero-current freewheel** :
    Body diode of the low-side switch blocks reverse current,
    so when i_L hits zero in Mode B at light load, the inductor
    is effectively *disconnected*. State reduces to ONLY the
    cap discharge:
        L · di_L/dt = 0  (algebraically clamped)
        C · dv_C/dt = -v_C / R_load
    Mode C exits at the next HS gate edge (start of new cycle).

Mode-transition events:

  A → B  : HS gate falling edge (scheduled).
  B → C  : i_L = 0 crossing (Illinois root-find on the
            current-zero-cross predicate).
  C → A  : HS gate rising edge (scheduled).

Analytical DCM regulator equation (Erickson & Maksimovic,
*Fundamentals of Power Electronics*, 3rd ed., §5.2.3 eq. 5.44):

    K = 2·L / (R_load · T_sw)
    K_crit = 1 - D                     (= D'  for a buck)
    DCM is sustained when K < K_crit.

    M(D, K) = V_out / V_in = 2 / (1 + sqrt(1 + 4·K/D²))

(CCM equivalent for comparison: M_ccm = D, so V_out_ccm = D·V_in.)

For the default DCM params (V_in=24 V, D=0.5, L=100 µH, C=100 µF,
R_load=240 Ω, f_sw=100 kHz):

    K = 2·100e-6 / (240·10e-6) = 0.0833
    K_crit = 0.5 → K << K_crit → deep DCM
    M = 2 / (1 + sqrt(1 + 4·0.0833/0.25)) = 2 / (1 + sqrt(2.333)) ≈ 0.7915
    V_out_steady ≈ 19.0 V   (vs 12 V if it were running in CCM)

Note that in DCM the buck output rises ABOVE the CCM ratio
M_ccm = D, because the inductor delivers its full energy packet
in a shorter time, producing a higher mean V_out for the same R.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BuckMode(Enum):
    """Three operating modes for a sync-buck in DCM."""
    HS_ON = "A"             # High-side switch conducting
    LS_CONDUCTING = "B"     # Low-side conducting (i_L > 0)
    ZERO_CURRENT = "C"      # Body diode blocking, i_L = 0


@dataclass
class BuckDCMParams:
    """Sync-buck operating point + passives, sized for DCM.

    Defaults give deep DCM (V_out ≈ 19V vs V_out_ccm = 12V at D=0.5).
    The Gate 1 CCM scenario was R_load=2.4Ω; raising it 100× to 240Ω
    pushes the operating point firmly into DCM (K = 0.083 ≪ K_crit = 0.5).
    """
    V_in: float = 24.0
    L: float = 100e-6
    C: float = 100e-6
    R_load: float = 240.0   # 100× CCM → deep DCM at D=0.5
    f_sw: float = 100e3
    D: float = 0.5

    @property
    def T_sw(self) -> float:
        return 1.0 / self.f_sw

    @property
    def K(self) -> float:
        """Erickson's DCM design constant: K = 2·L / (R_load · T_sw).

        DCM is sustained when ``K < K_crit = 1 - D`` for a buck
        (Erickson & Maksimovic eq. 5.46).
        """
        return 2.0 * self.L / (self.R_load * self.T_sw)

    @property
    def K_crit(self) -> float:
        """DCM/CCM boundary value: K_crit = 1 - D for a buck."""
        return 1.0 - self.D

    @property
    def M_dcm(self) -> float:
        """Analytical DCM buck conversion ratio (Erickson eq. 5.44):

            M(D, K) = 2 / (1 + sqrt(1 + 4·K/D²))
        """
        return 2.0 / (1.0 + np.sqrt(1.0 + 4.0 * self.K / (self.D ** 2)))

    @property
    def M_ccm(self) -> float:
        """CCM conversion ratio (for cross-check): M_ccm = D."""
        return self.D

    @property
    def V_out_steady(self) -> float:
        """Analytical DCM steady-state V_out."""
        return self.M_dcm * self.V_in

    @property
    def V_out_ccm_steady(self) -> float:
        """Reference CCM steady-state V_out (= D·V_in) — for cross-check."""
        return self.M_ccm * self.V_in

    @property
    def is_dcm_at_steady_state(self) -> bool:
        """Operating-point check per Erickson eq. 5.46:
        ``K < 1 - D = K_crit`` → DCM (else CCM).
        """
        return self.K < self.K_crit


class BuckDCMModel:
    """3-mode sync-buck DCM model.

    State: x = (i_L, v_C) ∈ R²
    Mode is mutated externally by the PED scheduler's event handler.
    """

    def __init__(self, params: BuckDCMParams) -> None:
        self.p = params
        # A matrix is the SAME for HS_ON and LS_CONDUCTING:
        # the C-R sub-system has the same dynamics independent of
        # which switch is providing the V_in or 0 to the inductor.
        self._A = np.array(
            [
                [0.0, -1.0 / params.L],
                [1.0 / params.C, -1.0 / (params.R_load * params.C)],
            ],
            dtype=float,
        )
        # b vectors differ per mode
        self._b_HSon = np.array([params.V_in / params.L, 0.0], dtype=float)
        self._b_LSconducting = np.array([0.0, 0.0], dtype=float)
        # Mode C: A reduced — i_L = 0 algebraically; only v_C dynamics
        self._A_zero = np.array(
            [
                [0.0, 0.0],
                [0.0, -1.0 / (params.R_load * params.C)],
            ],
            dtype=float,
        )

        self._mode: BuckMode = BuckMode.HS_ON

    @property
    def current_mask(self) -> BuckMode:
        """Mode getter — required by the PED scheduler interface."""
        return self._mode

    def set_mask(self, m: BuckMode) -> None:
        """Mode setter — called by the PED event handler."""
        self._mode = m

    def rhs(self, _t: float, x: np.ndarray) -> np.ndarray:
        """RHS dispatched on current mode (LTI per mode → ``t`` unused)."""
        del _t  # signature-compatible; silence "unused" lints
        if self._mode == BuckMode.HS_ON:
            return self._A @ x + self._b_HSon
        elif self._mode == BuckMode.LS_CONDUCTING:
            return self._A @ x + self._b_LSconducting
        else:  # ZERO_CURRENT
            # i_L is held at 0; only v_C decays through R_load
            return self._A_zero @ x


class BuckDCMSwitchFn:
    """Switch + mode schedule callable for the PED scheduler.

    Has two responsibilities:

    1. ``__call__(t) -> BuckMode`` — what mode SHOULD be active
       at time t, ASSUMING the gate-edge schedule alone (does not
       know about ZCD events; those are injected by the scheduler
       through ``register_zcd_transition()``).

    2. ``next_edge_after(t) -> float`` — next *gate* edge time
       (does not include ZCD events; those are predicate-driven
       through the EventPredictor).

    The PED scheduler combines both: it uses ``next_edge_after``
    for the fast path AND the EventPredictor's i_L-zero predicate
    for the B → C transition. When a ZCD fires, the scheduler calls
    ``register_zcd_transition(t)`` to mark that the next
    ``__call__`` should return ZERO_CURRENT until the next gate
    rising edge.
    """

    def __init__(self, T_sw: float, D: float) -> None:
        self.T_sw = T_sw
        self.D = D
        # Tracks zcd-induced mode override per cycle.
        # Maps cycle index → t at which ZCD fired.
        self._zcd_history: dict[int, float] = {}

    def _cycle_idx(self, t: float) -> int:
        return int(np.floor(t / self.T_sw))

    def _phase(self, t: float) -> float:
        return (t / self.T_sw) % 1.0

    def __call__(self, t: float) -> BuckMode:
        phase = self._phase(t)
        cycle = self._cycle_idx(t)
        if phase < self.D:
            return BuckMode.HS_ON
        # past HS-off boundary in this cycle
        if cycle in self._zcd_history and t > self._zcd_history[cycle]:
            return BuckMode.ZERO_CURRENT
        return BuckMode.LS_CONDUCTING

    def register_zcd_transition(self, t: float) -> None:
        """Called by the PED event handler when a ZCD fires."""
        cycle = self._cycle_idx(t)
        self._zcd_history[cycle] = t

    def next_edge_after(self, t: float) -> float:
        """Next GATE-edge time (does not include ZCD events)."""
        k = self._cycle_idx(t)
        candidates = [
            k * self.T_sw + self.D * self.T_sw,  # HS-off
            (k + 1) * self.T_sw,                  # HS-on (next cycle)
            (k + 1) * self.T_sw + self.D * self.T_sw,
        ]
        eps = 1e-15
        for c in candidates:
            if c > t + eps:
                return c
        return (k + 2) * self.T_sw


def make_zcd_predicate():
    """Build the inductor-current zero-crossing predicate.

    Fires when i_L crosses zero from above (only when in
    LS_CONDUCTING mode — the predicate caller is responsible
    for checking the active mode before treating the event as
    a B→C transition).
    """
    def predicate(_t: float, x: np.ndarray) -> float:
        # x[0] = i_L. Returns positive while i_L > 0, negative below.
        # ``t`` is unused — predicate is purely state-driven.
        del _t  # signature-compatible; silence "unused" lints
        return float(x[0])
    return predicate
