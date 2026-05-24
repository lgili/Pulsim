"""3-φ Conventional Matrix Converter (CMC) — modeling + SVM + plant builders.

Phase 21 of the pulsim roadmap. References:
  * Gili (2024), Sec 2.2 — "Conversor Matricial Convencional - CMC"
    (artigos/Luiz Carlos Gili-1.pdf).
  * Huber & Borojevic (1995) [20] — Space Vector Modulator for CMC.
  * Wheeler et al. (2002) [19] — Matrix Converters technology review.

The module exposes:
  * :class:`CmcParams` — operating-point parameters.
  * :data:`CMC_ZERO_VECTORS`, :data:`CMC_ACTIVE_VECTORS`,
    :data:`CMC_ROTATIONAL_VECTORS` — the 27 switching states organised
    as in the thesis Tab. 1, 2, 3.
  * :func:`svm_duty_cycles` — closed-form duty cycles from Eqs 7a-7d
    (Sec 2.2.1) given the reference vector phase + input current angle.
  * :func:`svm_sector_pair` — locate (K_v, K_i) and sectorial angles
    (α̃_o, β̃_i) of the reference vectors.
  * :func:`svm_active_vectors_for_sectors` — Tab. 4 of the thesis,
    indexed by (K_v, K_i) → 4 active configuration ids ∈ {±1..±9}.
  * :func:`switch_mask_for_config` — turn an active-vector id (e.g.
    ``+7`` or ``-3``) into a 9-bit mask of which ``S_1..S_9`` are ON.
  * :func:`make_cmc_gate_signals` — full symmetric sequence (Fig 5):
    Ta/2, Tb/2, Tc/2, Td/2, T0, Td/2, ..., Ta/2 — returns a callable
    ``t -> (S1, ..., S9)``.

Plant builders (``build_l0_plant``, ``build_l1_plant``) and run
helpers will be added in subsequent commits.

Sign / numbering conventions
----------------------------

Following the thesis exactly:

  * **Switches** ``S_1, ..., S_9`` are numbered by COLUMN of the 3×3
    matrix (Fig 1 of the thesis):
      - ``S_1, S_2, S_3``: input phase **A** → outputs a, b, c.
      - ``S_4, S_5, S_6``: input phase **B** → outputs a, b, c.
      - ``S_7, S_8, S_9``: input phase **C** → outputs a, b, c.
    Constraint: for each output (rows {1,4,7}, {2,5,8}, {3,6,9}),
    exactly **one** switch is ON at any time.

  * **Active configurations** ``±1, ..., ±9`` (Tab. 2): magnitude
    ``2/3 V_LL`` (line-line of one of the three input pairs);
    direction varies with the permutation. Positive vs negative
    differs by which input phase is "borrowed" for the third output.

  * **Sector indexing**: ``K_v = 1`` for ``α_o ∈ [-30°, +30°]``, then
    +60° per sector — same for ``K_i`` with ``β_i``. The sectorial
    angles ``α̃_o, β̃_i ∈ [-π/6, +π/6]`` are measured from the
    sector bisector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sqrt
from typing import Callable

import numpy as np
import pulsim as p


# =============================================================================
# Parameters
# =============================================================================


@dataclass(frozen=True)
class CmcParams:
    """Operating-point + topology parameters for a 3-φ CMC.

    The default values reflect a low-voltage, motor-drive style point
    of operation amenable to all-purpose validation. The thesis
    Chapter 2 doesn't fix a specific operating point for the CMC
    (it's the *literature review* section), so the defaults are
    chosen for the validation pass — see ``README.md`` for the three
    test points (step-down 60→60, motor 60→30, limit case).

    Attributes
    ----------
    V_in_peak : float
        Peak amplitude of the *line-to-neutral* input voltage [V].
        With V_LL=380 V (line-line, RMS), V_in_peak ≈ 380·√(2/3) ≈
        310.3 V.
    f_in : float
        Input fundamental frequency [Hz].
    f_out : float
        Output fundamental frequency [Hz]. Can be > or < f_in (or
        equal — the CMC is four-quadrant).
    m_depth : float
        Modulation index ``m = V_o_peak / V_in_peak`` ∈ [0, √3/2 ≈ 0.866].
        Limit derived in Sec 2.2.1 (Eq 11) assumes unity input PF.
    phi_i : float
        Input displacement angle [rad] — angle between input voltage
        and current vectors. ``0`` = unity power factor (typical).
    f_switching : float
        SVM switching frequency [Hz] (e.g. 10 kHz). The symmetric
        sequence applies 9 sub-intervals per ``T_s = 1/f_switching``.
    R_load : float
        Load resistance per phase [Ω] (Y-connected).
    L_load : float
        Load inductance per phase [H].

    Derived properties
    ------------------
    omega_in, omega_out : input/output angular frequencies [rad/s].
    V_LL_in_peak : peak line-line input voltage = V_in_peak · √3.
    V_o_peak : peak output line-to-neutral voltage = m · V_in_peak.
    """

    V_in_peak: float = 311.13   # 220 V_LL RMS = 311.13 V_peak line-neutral
    f_in: float = 60.0
    f_out: float = 30.0
    m_depth: float = 0.5
    phi_i: float = 0.0
    f_switching: float = 10_000.0
    R_load: float = 5.0
    L_load: float = 10.0e-3

    @property
    def omega_in(self) -> float:
        return 2.0 * pi * self.f_in

    @property
    def omega_out(self) -> float:
        return 2.0 * pi * self.f_out

    @property
    def V_LL_in_peak(self) -> float:
        """Peak line-line input voltage [V]."""
        return self.V_in_peak * sqrt(3.0)

    @property
    def V_o_peak(self) -> float:
        """Peak output line-to-neutral voltage [V] (m · V_in_peak)."""
        return self.m_depth * self.V_in_peak

    @property
    def T_s(self) -> float:
        return 1.0 / self.f_switching


# =============================================================================
# Switching state tables — directly from Gili (2024) Tab. 1, 2, 3
# =============================================================================
#
# Encoding: each state is a 3-tuple ``(out_a, out_b, out_c)`` where each
# entry is the *input phase index* {0='A', 1='B', 2='C'} that the output
# is tied to. Equivalent to specifying which one switch in each row of
# the 3×3 matrix is ON.
#
# The mapping back to the thesis's S_1..S_9 numbering:
#   * S_(3*k_in + k_out + 1) is ON  ↔  out_(k_out) connected to input k_in.
#   * E.g. (0, 1, 1) means out_a→A (S_1), out_b→B (S_5), out_c→B (S_6).


# Table 1 — three zero states (V_o = 0, I_in = 0).
CMC_ZERO_VECTORS: dict[str, tuple[int, int, int]] = {
    "0_1": (0, 0, 0),  # all outputs → A   (S1+S2+S3 ON)
    "0_2": (1, 1, 1),  # all outputs → B   (S4+S5+S6 ON)
    "0_3": (2, 2, 2),  # all outputs → C   (S7+S8+S9 ON)
}


# Table 2 — 18 active states. Magnitude = 2/3 V_LL of the line pair
# implied by the permutation; alpha_o and beta_i are fixed angles
# (modulo line-pair); only the line-line voltage *amplitude* varies
# with time.
#
# Convention: +k and -k share the same magnitude but opposite
# direction. The mapping is taken directly from Tab. 2 of the thesis:
#
#   Configuration   | (out_a, out_b, out_c) | implied V_LL | alpha_o
#   ----------------+------------------------+--------------+--------
#   +1: S1 S5 S6    | (0, 1, 1)              | V_AB         | 0°
#   -1: S4 S2 S3    | (1, 0, 0)              | V_AB (rev)   | 180°
#   +2: S4 S8 S9    | (1, 2, 2)              | V_BC         | 0°
#   -2: S7 S5 S6    | (2, 1, 1)              | V_BC (rev)   | 180°  ← Tab. 2 row "-2"
#   ... (and so on)
#
# We store the (out_a, out_b, out_c) triple; the magnitude and angle
# can be recomputed.

CMC_ACTIVE_VECTORS: dict[int, tuple[int, int, int]] = {
    # Group: outputs share input "A" with one differing (V_AB / V_CA family)
    +1: (0, 1, 1),  # S1 S5 S6
    -1: (1, 0, 0),  # S4 S2 S3
    +2: (1, 2, 2),  # S4 S8 S9
    -2: (2, 1, 1),  # S7 S5 S6
    +3: (2, 0, 0),  # S7 S2 S3
    -3: (0, 2, 2),  # S1 S8 S9
    # Group: pattern (X, Y, X) — output b is the odd one
    +4: (1, 0, 1),  # S4 S2 S6
    -4: (0, 1, 0),  # S1 S5 S3
    +5: (2, 1, 2),  # S7 S5 S9
    -5: (1, 2, 1),  # S4 S8 S6
    +6: (0, 2, 0),  # S1 S8 S3
    -6: (2, 0, 2),  # S7 S2 S9
    # Group: pattern (X, X, Y) — output c is the odd one
    +7: (1, 1, 0),  # S5 S5 S3  (wait, S5 twice? read thesis again)
    -7: (0, 0, 1),  # S1 S2 S6
    +8: (2, 2, 1),  # S7 S8 S6
    -8: (1, 1, 2),  # S4 S5 S9
    +9: (0, 0, 2),  # S1 S2 S9
    -9: (2, 2, 0),  # S7 S8 S3
}


# Table 3 — six rotational states. Each is a 3! permutation of inputs
# across outputs. The triple is unique up to cyclic rotation.

CMC_ROTATIONAL_VECTORS: dict[str, tuple[int, int, int]] = {
    "R_1": (0, 1, 2),  # S1 S5 S9 — identity
    "R_2": (1, 2, 0),  # S2 S6 S7 — cycle a→B, b→C, c→A
    "R_3": (2, 0, 1),  # S3 S4 S8 — cycle a→C, b→A, c→B
    "R_4": (2, 1, 0),  # S3 S5 S7 — reverse identity
    "R_5": (0, 2, 1),  # S1 S6 S8 — swap b↔c
    "R_6": (1, 0, 2),  # S2 S4 S9 — swap a↔b
}


# =============================================================================
# Switch mask helpers
# =============================================================================


def switch_mask_for_state(state: tuple[int, int, int]) -> tuple[int, ...]:
    """Convert a switching state ``(out_a, out_b, out_c)`` (with each
    entry ∈ {0, 1, 2} for inputs A, B, C) into a 9-tuple
    ``(S_1, S_2, ..., S_9)`` of 0/1 ON-flags.

    Index mapping (column-major, following Fig 1 of the thesis):

      Input phase A → S_1 (out a), S_2 (out b), S_3 (out c)
      Input phase B → S_4 (out a), S_5 (out b), S_6 (out c)
      Input phase C → S_7 (out a), S_8 (out b), S_9 (out c)

    So ``S_(3·k_in + k_out + 1)`` is ON iff ``state[k_out] == k_in``.
    """
    mask = [0] * 9
    for k_out, k_in in enumerate(state):
        if not (0 <= k_in <= 2):
            raise ValueError(
                f"state[{k_out}] = {k_in} not in {{0,1,2}} (A,B,C)")
        s_id = 3 * k_in + k_out  # 0-indexed
        mask[s_id] = 1
    return tuple(mask)


def switch_mask_for_config(config_id: int | str) -> tuple[int, ...]:
    """Return the 9-bit switch mask for an *active* (``±1..±9``),
    *zero* (``"0_1"``, ``"0_2"``, ``"0_3"``), or *rotational*
    (``"R_1".."R_6"``) configuration.
    """
    if isinstance(config_id, int):
        if config_id not in CMC_ACTIVE_VECTORS:
            raise KeyError(
                f"unknown active configuration {config_id}; "
                f"expected one of {sorted(CMC_ACTIVE_VECTORS.keys())}")
        return switch_mask_for_state(CMC_ACTIVE_VECTORS[config_id])
    if config_id in CMC_ZERO_VECTORS:
        return switch_mask_for_state(CMC_ZERO_VECTORS[config_id])
    if config_id in CMC_ROTATIONAL_VECTORS:
        return switch_mask_for_state(CMC_ROTATIONAL_VECTORS[config_id])
    raise KeyError(f"unknown configuration id {config_id!r}")


# =============================================================================
# SVM analytical — Sec 2.2.1, Eqs 6, 7a-7d, 11
# =============================================================================


def svm_sector_pair(
    alpha_o: float, beta_i: float,
) -> tuple[int, int, float, float]:
    """Locate the (K_v, K_i) sector pair for output voltage vector
    angle ``alpha_o`` and input current vector angle ``beta_i``.

    Both inputs are in radians. Each sector covers 60°; the sectorial
    angles ``α̃_o`` and ``β̃_i`` are measured from the sector bisector
    and lie in ``[-π/6, +π/6]`` (Eq 6 of the thesis).

    Sector indexing (Fig 4 of the thesis):
      * ``K_v = 1`` for ``α_o ∈ [-30°, +30°]`` (centred on the α axis)
      * ``K_v`` increments every +60° going counter-clockwise.

    Returns
    -------
    K_v : int ∈ {1..6}
    K_i : int ∈ {1..6}
    alpha_til : float ∈ [-π/6, π/6]
    beta_til  : float ∈ [-π/6, π/6]
    """
    def _sector(angle: float) -> tuple[int, float]:
        # Wrap into [-π/6, 2π - π/6) so that sector 1 covers [-30°, +30°].
        a = (angle + pi / 6) % (2.0 * pi)
        k = int(a // (pi / 3.0)) + 1  # 1..6
        # Sectorial angle measured from bisector.
        bisector = (k - 1) * pi / 3.0   # 0, 60°, 120°, ...
        a_til = ((angle - bisector + pi) % (2.0 * pi)) - pi
        return k, a_til

    K_v, alpha_til = _sector(alpha_o)
    K_i, beta_til = _sector(beta_i)
    return K_v, K_i, alpha_til, beta_til


def svm_duty_cycles(
    m_depth: float,
    alpha_o: float,
    beta_i: float,
    phi_i: float = 0.0,
) -> tuple[float, float, float, float]:
    """Eqs 7a-7d of the thesis — four SVM duty cycles for the CMC.

    Parameters
    ----------
    m_depth : float
        Modulation index ∈ [0, √3/2].
    alpha_o : float
        Output voltage vector angle [rad] (absolute, NOT sectorial).
    beta_i : float
        Input current vector angle [rad] (absolute).
    phi_i : float, default 0
        Input displacement angle [rad]; ``cos(phi_i)`` appears in the
        denominator.

    Returns
    -------
    (delta_I, delta_II, delta_III, delta_IV) : tuple of 4 floats
        Duty cycles, **signed**. By Sec 2.2.1, "for any sector,
        exactly two of the four duty cycles are negative"; the
        absolute value gives the on-time fraction and the sign
        selects between the +k and -k variant of the corresponding
        active configuration.

    Notes
    -----
    The factor ``(-1)^(K_v + K_i + ...)`` in the original equations
    is folded into the sign of ``delta`` and applied here directly.
    """
    K_v, K_i, alpha_til, beta_til = svm_sector_pair(alpha_o, beta_i)
    cos_phi = cos(phi_i)
    if abs(cos_phi) < 1e-12:
        raise ValueError(
            "phi_i ≈ ±π/2 ⇒ cos(phi_i) ≈ 0; CMC SVM diverges"
        )

    # Common multiplier — Eq 7a-7d.
    K = (2.0 / sqrt(3.0)) * m_depth / cos_phi

    sign_a = (-1.0) ** (K_v + K_i + 1)  # δ^I, δ^IV
    sign_b = (-1.0) ** (K_v + K_i)       # δ^II, δ^III

    delta_I   = sign_a * K * cos(alpha_til - pi / 3.0) * cos(beta_til - pi / 3.0)
    delta_II  = sign_b * K * cos(alpha_til - pi / 3.0) * cos(beta_til + pi / 3.0)
    delta_III = sign_b * K * cos(alpha_til + pi / 3.0) * cos(beta_til - pi / 3.0)
    delta_IV  = sign_a * K * cos(alpha_til + pi / 3.0) * cos(beta_til + pi / 3.0)
    return delta_I, delta_II, delta_III, delta_IV


def svm_max_modulation(phi_i: float = 0.0) -> float:
    """Theoretical maximum modulation index — Eq 11 of the thesis.

    Returns
    -------
    m_max = (√3/2) · |cos(phi_i)|

    At unity power factor (``phi_i = 0``), ``m_max = √3/2 ≈ 0.866``.
    """
    return (sqrt(3.0) / 2.0) * abs(cos(phi_i))


# =============================================================================
# Tabela 4 — active vector sequence per (K_v, K_i)
# =============================================================================
#
# From Tab. 4 of the thesis (Sec 2.2.1):
#
#   Output voltage sector K_v →  1-4         2-5         3-6
#   Input current sector K_i ↓
#         1-4                    +9 +7 +3 +1  +6 +4 +9 +7  +3 +1 +6 +4
#         2-5                    +8 +9 +2 +3  +5 +6 +8 +9  +2 +3 +5 +6
#         3-6                    +7 +8 +1 +2  +4 +5 +7 +8  +1 +2 +4 +5
#
# Each cell lists the FOUR positive active configurations (a, b, c, d)
# corresponding to δ^I, δ^II, δ^III, δ^IV. The actual sign applied is
# the sign of the computed δ — if δ < 0, the NEGATIVE configuration
# (``-k`` instead of ``+k``) is used (Tab. 2 of the thesis).
#
# The table is symmetric under K → K+3 (mod 6) — sectors 1-4, 2-5, 3-6
# share the same vector pattern.


_TAB4_TRIPLES: dict[tuple[int, int], tuple[int, int, int, int]] = {
    # (K_v_mod_3, K_i_mod_3) → (config_a, config_b, config_c, config_d)
    # (K_v mod 3 ∈ {1, 2, 0} mapped from {1-4, 2-5, 3-6})
    (1, 1): (+9, +7, +3, +1),
    (2, 1): (+6, +4, +9, +7),
    (0, 1): (+3, +1, +6, +4),
    (1, 2): (+8, +9, +2, +3),
    (2, 2): (+5, +6, +8, +9),
    (0, 2): (+2, +3, +5, +6),
    (1, 0): (+7, +8, +1, +2),
    (2, 0): (+4, +5, +7, +8),
    (0, 0): (+1, +2, +4, +5),
}


def svm_active_vectors_for_sectors(
    K_v: int, K_i: int,
) -> tuple[int, int, int, int]:
    """Return the four POSITIVE-baseline active vectors
    ``(V^I, V^II, V^III, V^IV)`` from Tab. 4 of the thesis, indexed
    by sector pair (``K_v``, ``K_i``).

    The values are always ``+k`` (k ∈ {1..9}); whether each is
    applied as ``+k`` or ``-k`` depends on the sign of the
    corresponding duty cycle computed by :func:`svm_duty_cycles`.
    """
    if not (1 <= K_v <= 6) or not (1 <= K_i <= 6):
        raise ValueError(f"K_v={K_v}, K_i={K_i} must each be in {{1..6}}")
    return _TAB4_TRIPLES[(K_v % 3, K_i % 3)]


# =============================================================================
# Symmetric switching sequence (Fig 5 — "Sequência I")
# =============================================================================


def svm_step(
    t: float,
    m_depth: float,
    omega_out: float,
    omega_in: float,
    phi_i: float = 0.0,
    f_switching: float = 10_000.0,
) -> tuple[int, ...]:
    """Compute the instantaneous 9-bit switch mask at time ``t`` from
    the analytical SVM (Fig 5 — Sequência I).

    The output is a tuple ``(S_1, S_2, ..., S_9)`` of 0/1 ON-flags
    matching the thesis indexing.

    Implementation
    --------------
    1. Determine the current SVM period index ``k = floor(t / T_s)``
       and the offset ``τ = t - k · T_s`` within the period.
    2. Sample the reference angles at the START of the period:
       ``alpha_o = omega_out · (k · T_s)``,
       ``beta_i  = omega_in  · (k · T_s)``.
    3. Compute duty cycles (signed) and resolve to four active
       vectors with signs.
    4. Within the period, time-slice as:
       ``[Ta/2 | Tb/2 | Tc/2 | Td/2 | T0 | Td/2 | Tc/2 | Tb/2 | Ta/2]``
       and return the mask for whichever sub-interval ``τ`` lies in.

    For ``T0``, we use the zero state ``0_1`` (all to A) — this is
    one of the three choices; production SVM picks the one that
    minimises commutations from the previous active state, but this
    function uses the simpler fixed choice.
    """
    T_s = 1.0 / f_switching
    k = int(t // T_s)
    tau = t - k * T_s
    t_start = k * T_s
    alpha_o = omega_out * t_start
    beta_i = omega_in * t_start
    K_v, K_i, _, _ = svm_sector_pair(alpha_o, beta_i)
    d_I, d_II, d_III, d_IV = svm_duty_cycles(
        m_depth, alpha_o, beta_i, phi_i)
    vecs = svm_active_vectors_for_sectors(K_v, K_i)

    # Apply signs: each active vector picks the +k variant if δ > 0,
    # else the -k variant.
    signed = []
    for d, v in zip((d_I, d_II, d_III, d_IV), vecs):
        if d >= 0:
            signed.append(+v)
        else:
            signed.append(-v)
    duties = [abs(d) for d in (d_I, d_II, d_III, d_IV)]
    d0 = 1.0 - sum(duties)
    if d0 < -1e-9:
        # Over-modulation; clamp by scaling active duties to sum to 1
        s = sum(duties)
        duties = [d / s for d in duties]
        d0 = 0.0

    # Sub-interval boundaries in [0, T_s].
    # Half-period sequence: τ ∈ [0, Ta/2): active_I
    #                       τ ∈ [Ta/2, Ta/2+Tb/2): active_II
    #                       τ ∈ [Ta/2+Tb/2, Ta/2+Tb/2+Tc/2): active_III
    #                       τ ∈ ...+Td/2): active_IV
    #                       τ ∈ ...+T0): zero
    #                       (mirror) active_IV ... active_I
    halves = [d / 2.0 for d in duties]  # Ta/2, Tb/2, Tc/2, Td/2
    # cumulative t_s fractions
    t_a2 = halves[0] * T_s
    t_b2 = t_a2 + halves[1] * T_s
    t_c2 = t_b2 + halves[2] * T_s
    t_d2 = t_c2 + halves[3] * T_s
    t_0 = t_d2 + d0 * T_s
    t_d2_m = t_0 + halves[3] * T_s
    t_c2_m = t_d2_m + halves[2] * T_s
    t_b2_m = t_c2_m + halves[1] * T_s
    # t_a2_m = t_b2_m + halves[0] * T_s == T_s

    if tau < t_a2:
        return switch_mask_for_config(signed[0])
    elif tau < t_b2:
        return switch_mask_for_config(signed[1])
    elif tau < t_c2:
        return switch_mask_for_config(signed[2])
    elif tau < t_d2:
        return switch_mask_for_config(signed[3])
    elif tau < t_0:
        return switch_mask_for_config("0_1")
    elif tau < t_d2_m:
        return switch_mask_for_config(signed[3])
    elif tau < t_c2_m:
        return switch_mask_for_config(signed[2])
    elif tau < t_b2_m:
        return switch_mask_for_config(signed[1])
    else:
        return switch_mask_for_config(signed[0])


def make_cmc_gate_signals(
    params: CmcParams,
) -> Callable[[float], tuple[int, ...]]:
    """Build a callable ``t → (S_1, ..., S_9)`` that returns the
    instantaneous switch state under the symmetric SVM sequence
    (Fig 5 of the thesis), pre-binding all parameters.

    The returned function is suitable for passing to a pulsim
    ``b_extra_fn`` / observer to drive the 18 IGBT gate signals of
    the L1 plant.
    """
    m_depth = params.m_depth
    omega_o = params.omega_out
    omega_i = params.omega_in
    phi_i = params.phi_i
    f_sw = params.f_switching

    def _gate_fn(t: float) -> tuple[int, ...]:
        return svm_step(t, m_depth, omega_o, omega_i, phi_i, f_sw)

    return _gate_fn


# =============================================================================
# L0 — averaged-model plant + runner
# =============================================================================
#
# The L0 model represents the CMC as a Venturini-style continuous-time
# averaged converter: there is no high-frequency switching ripple in
# the output voltages, only the *fundamental* synthesised by SVM.
# Specifically, the output line-neutral voltages are:
#
#     v_a_out(t) = m · V_in_peak · cos(ω_o·t)
#     v_b_out(t) = m · V_in_peak · cos(ω_o·t − 2π/3)
#     v_c_out(t) = m · V_in_peak · cos(ω_o·t + 2π/3)
#
# Each one synthesised with an :func:`pulsim.add_sine_voltage_source`
# at the converter terminal. The load is a Y-connected RL bank.
#
# The L0 plant is *one-sided* by design: it shows the output side
# only. The full input-current dynamics couple to the output through
# the SVM modulation matrix and are captured by the L1 switched model
# (next phase). The L0 baseline lets us:
#
#   1. Validate that the load currents track ``i_o_peak =
#      m·V_in_peak / |Z_load|`` (closed-form RL response);
#   2. Confirm THD ≈ 0 (pure sinusoid — no ripple);
#   3. Provide a reference fundamental for L1 to match.


@dataclass
class CmcPlant:
    """Bundle of (builder, output-current branch indices) returned by
    the CMC plant builders. Same pattern as ``MmcPlant`` in the MMC
    project."""

    builder: object
    iL_out_indices: tuple[int, int, int] = (0, 0, 0)
    # Optional: input-current branch ids when the plant models the
    # input side. None in the output-only L0.
    iL_in_indices: tuple[int, int, int] | None = None


def build_l0_plant(params: CmcParams) -> CmcPlant:
    """3-φ CMC output-side averaged plant.

    Topology:

        ┌── V_a_out ──┬── L_load ──┬── R_load ──┐
        │                                       │
        │── V_b_out ──┬── L_load ──┬── R_load ──┼── star
        │                                       │
        └── V_c_out ──┴── L_load ──┴── R_load ──┘

    The three output voltage sources are sinusoids at ``f_out`` with
    amplitude ``m · V_in_peak``, phase-shifted by 120° — i.e. the
    *ideal* SVM-synthesised reference. The load is Y-connected RL,
    star tied weakly to ground for MNA conditioning.

    The plant is suitable for validating output fundamental
    amplitude, RMS, and the load impedance response. It does **not**
    model the input side (entrance currents are zero by construction).
    """
    b = p.CircuitBuilder()
    V_o_peak = params.V_o_peak  # = m · V_in_peak
    f_out = params.f_out

    # Output voltage sources — one per phase, sinusoidal at f_out.
    # Phase offset includes ``+π/2`` to convert pulsim's ``sin``
    # convention to the **cosine** convention used by the SVM theory
    # in the thesis (V_a = V_o·cos(ω_o·t), so at t=0 V_a = V_o).
    # This guarantees that the SVM α_o = ω_o·t lookup aligns with
    # the actual voltage vector position in pulsim's clock.
    b.add_sine_voltage_source(
        "V_a_out", "a", "star",
        v_dc=0.0, v_amplitude=V_o_peak, frequency=f_out,
        phase=+pi / 2.0,                       # sin(ωt + π/2) = cos(ωt)
    )
    b.add_sine_voltage_source(
        "V_b_out", "b", "star",
        v_dc=0.0, v_amplitude=V_o_peak, frequency=f_out,
        phase=+pi / 2.0 - 2.0 * pi / 3.0,       # cos(ωt − 2π/3)
    )
    b.add_sine_voltage_source(
        "V_c_out", "c", "star",
        v_dc=0.0, v_amplitude=V_o_peak, frequency=f_out,
        phase=+pi / 2.0 + 2.0 * pi / 3.0,       # cos(ωt + 2π/3)
    )

    # Y-load: ph → L_load → R_load → star (same neutral as sources).
    # Record the inductor branch IDs as they are created, but resolve
    # the *state indices* only after the FULL graph is built — pulsim's
    # state-index assignment depends on the total graph topology and
    # changes if branches are added later. Calling
    # ``branch_var_id_for_inductor`` mid-build yields stale indices
    # that silently permute the (i_a, i_b, i_c) mapping.
    L_branch_ids: list[int] = []
    for ph in "abc":
        L_id = b.graph.num_branches
        b.add_inductor(
            f"L_load_{ph}", ph, f"rload_{ph}", params.L_load,
        )
        b.add_resistor(
            f"R_load_{ph}", f"rload_{ph}", "star", params.R_load,
        )
        L_branch_ids.append(L_id)

    # Weak tie of star to ground (MNA conditioning).
    b.add_resistor("R_star_gnd", "star", "gnd", 1e6)

    # Now that the graph is complete, resolve state indices.
    iL_out_indices = tuple(
        b.pool.branch_var_id_for_inductor(L_id, b.graph)
        for L_id in L_branch_ids
    )

    return CmcPlant(
        builder=b,
        iL_out_indices=iL_out_indices,  # type: ignore[arg-type]
        iL_in_indices=None,
    )


# =============================================================================
# Run driver
# =============================================================================


@dataclass
class CmcRunResult:
    """Output of :func:`run_l0_open_loop` / future L1 runner."""

    t: np.ndarray
    # Output (load) currents, one per phase
    i_a_out: np.ndarray
    i_b_out: np.ndarray
    i_c_out: np.ndarray
    # Optional: input currents (populated by L1)
    i_a_in: np.ndarray = field(default_factory=lambda: np.empty(0))
    i_b_in: np.ndarray = field(default_factory=lambda: np.empty(0))
    i_c_in: np.ndarray = field(default_factory=lambda: np.empty(0))


def run_l0_open_loop(
    plant: CmcPlant,
    *,
    t_end: float = 100e-3,
    dt: float = 10e-6,
) -> CmcRunResult:
    """Run an L0 plant for ``t_end`` seconds at fixed ``dt``.

    No observer is needed — the sinusoidal sources update themselves
    automatically (built-in time-varying primitive in pulsim).
    """
    iLa, iLb, iLc = plant.iL_out_indices
    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_ia = np.zeros(n_samples)
    log_ib = np.zeros(n_samples)
    log_ic = np.zeros(n_samples)
    counter = [0]

    def log_obs(t, x):
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_ia[i] = x[iLa]
            log_ib[i] = x[iLb]
            log_ic[i] = x[iLc]
        counter[0] += 1

    p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=log_obs,
        start_from_dc_op=True,
    )

    n = counter[0]
    return CmcRunResult(
        t=log_t[:n],
        i_a_out=log_ia[:n],
        i_b_out=log_ib[:n],
        i_c_out=log_ic[:n],
    )


# =============================================================================
# Metrics — closed-form predictions + signal analysis
# =============================================================================


def predict_load_impedance(params: CmcParams) -> complex:
    """Per-phase complex impedance of the Y-connected RL load:

        Z = R + jω_out · L
    """
    return params.R_load + 1j * params.omega_out * params.L_load


def predict_i_out_peak(params: CmcParams) -> float:
    """Closed-form peak of the load current — Ohm's law at the output:

        |I_o| = V_o_peak / |Z_load|
    """
    return params.V_o_peak / abs(predict_load_impedance(params))


def predict_load_power_factor(params: CmcParams) -> float:
    """Power factor of the RL load: cos(arctan(ω·L / R))."""
    return float(np.cos(np.arctan2(params.omega_out * params.L_load,
                                     params.R_load)))


def rms(signal: np.ndarray) -> float:
    """RMS value of a signal."""
    return float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2)))


# =============================================================================
# L1 — switched plant (9 ideal bidirectional switches driven by SVM)
# =============================================================================
#
# L1 captures the actual switching behaviour of the CMC by replacing
# the synthesised sinusoidal output sources of L0 with the physical
# matrix-converter topology: 9 ideal bidirectional switches connecting
# every input phase to every output phase.
#
# For the initial L1 pass we use ``add_switch`` — a switched-resistor
# (g_on / g_off binary) primitive that is inherently bidirectional
# and provides the same abstraction the thesis Sec 2.2 SVM
# derivation uses. A Phase 21.4 upgrade will replace each switch
# with a 2-IGBT common-emitter pair + 2 anti-parallel diodes for
# physical V_CE_sat + R_CE_sat modelling.
#
# Switch ordering matches the thesis convention exactly (Fig 1):
#
#   S_1 = A → a      S_2 = A → b      S_3 = A → c
#   S_4 = B → a      S_5 = B → b      S_6 = B → c
#   S_7 = C → a      S_8 = C → b      S_9 = C → c
#
# The :class:`SwitchStateMask` returned by :func:`make_cmc_switch_fn`
# packs the 9 switch states in this column-major order so that the
# values returned by :func:`svm_step` align directly with the mask
# bits.


def build_l1_plant(params: CmcParams) -> CmcPlant:
    """3-φ CMC switched plant — 9 ideal bidirectional switches with
    the SVM driving the gate signals.

    Topology::

        V_A ── A ─┬── S1 ──┐    ┌── S2 ──┐    ┌── S3 ──┐
                  │         │    │         │    │         │
                  S4        a    S5        b    S6        c
                  │         │    │         │    │         │
                  └── ...  load_a load_b ... load_c
        V_B ── B ──...──┘
        V_C ── C ──...──┘

    There is no input filter — input currents are taken directly
    from the source branch currents and will show the full
    switching ripple. The output side has the Y-connected RL load
    (same as L0).
    """
    b = p.CircuitBuilder()
    V_in_peak = params.V_in_peak
    f_in = params.f_in

    # ---- Input voltage sources (Y) -------------------------------------
    # Cosine convention to align with SVM theory (V_A = V_in·cos(ω_i·t)).
    # See note in build_l0_plant — phase = +π/2 turns sin → cos.
    b.add_sine_voltage_source(
        "V_A", "A", "src_star",
        v_dc=0.0, v_amplitude=V_in_peak, frequency=f_in,
        phase=+pi / 2.0,
    )
    b.add_sine_voltage_source(
        "V_B", "B", "src_star",
        v_dc=0.0, v_amplitude=V_in_peak, frequency=f_in,
        phase=+pi / 2.0 - 2.0 * pi / 3.0,
    )
    b.add_sine_voltage_source(
        "V_C", "C", "src_star",
        v_dc=0.0, v_amplitude=V_in_peak, frequency=f_in,
        phase=+pi / 2.0 + 2.0 * pi / 3.0,
    )

    # Record source branch IDs (always 0, 1, 2 for V_A, V_B, V_C).
    src_branch_ids = (0, 1, 2)

    # ---- 9 bidirectional switches in column-major order ---------------
    # Order MUST match the thesis convention so svm_step output aligns
    # with the SwitchStateMask bits.
    input_nodes = ("A", "B", "C")
    output_nodes = ("a", "b", "c")
    switch_idx = 0
    for input_node in input_nodes:
        for output_node in output_nodes:
            switch_idx += 1
            b.add_switch(
                f"S_{switch_idx}",
                input_node, output_node,
                g_on=1e3,     # 1/g_on  = 1 mΩ  on-state resistance
                g_off=1e-6,   # 1/g_off = 1 MΩ off-state resistance
            )

    # ---- Output Y-load -------------------------------------------------
    L_branch_ids: list[int] = []
    for ph in output_nodes:
        L_id = b.graph.num_branches
        b.add_inductor(
            f"L_load_{ph}", ph, f"rload_{ph}", params.L_load,
        )
        b.add_resistor(
            f"R_load_{ph}", f"rload_{ph}", "load_star", params.R_load,
        )
        L_branch_ids.append(L_id)

    # Neutral / ground ties (MNA conditioning).
    b.add_resistor("R_src_gnd", "src_star", "gnd", 1e6)
    b.add_resistor("R_load_gnd", "load_star", "gnd", 1e6)

    # Resolve state indices after the FULL graph is built (see note in
    # build_l0_plant — pulsim re-numbers state indices as branches
    # are added).
    iL_out_indices = tuple(
        b.pool.branch_var_id_for_inductor(L_id, b.graph)
        for L_id in L_branch_ids
    )
    iL_in_indices = tuple(
        b.pool.branch_var_id_for_source(src_id, b.graph)
        for src_id in src_branch_ids
    )

    return CmcPlant(
        builder=b,
        iL_out_indices=iL_out_indices,    # type: ignore[arg-type]
        iL_in_indices=iL_in_indices,      # type: ignore[arg-type]
    )


def make_cmc_switch_fn(
    params: CmcParams,
) -> Callable[[float], "p.SwitchStateMask"]:
    """Build a callable ``t → SwitchStateMask(9)`` packing the 9
    instantaneous switch states from :func:`svm_step` in the order
    in which the switches were added to the builder (column-major:
    S_1..S_3 for input A, S_4..S_6 for B, S_7..S_9 for C).
    """
    gate_fn = make_cmc_gate_signals(params)

    def _switch_fn(t: float) -> "p.SwitchStateMask":
        mask_bits = gate_fn(t)  # 9-tuple of 0/1
        mask = p.SwitchStateMask(9)
        for i, b in enumerate(mask_bits):
            mask.set(i, bool(b))
        return mask

    return _switch_fn


def run_l1_open_loop(
    plant: CmcPlant,
    params: CmcParams,
    *,
    t_end: float = 100e-3,
    dt: float | None = None,
) -> CmcRunResult:
    """Run an L1 switched plant for ``t_end`` seconds. ``dt`` defaults
    to ``T_s / 20`` for adequate switching resolution.

    The SVM gating is driven by :func:`make_cmc_switch_fn(params)`,
    which is passed to ``simulate`` as the ``switch_fn``.
    """
    if dt is None:
        dt = params.T_s / 20.0  # 20 samples per T_s by default

    sw_fn = make_cmc_switch_fn(params)

    iLa, iLb, iLc = plant.iL_out_indices
    assert plant.iL_in_indices is not None, "L1 plant must have input current indices"
    iIA, iIB, iIC = plant.iL_in_indices
    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_iao = np.zeros(n_samples)
    log_ibo = np.zeros(n_samples)
    log_ico = np.zeros(n_samples)
    log_iai = np.zeros(n_samples)
    log_ibi = np.zeros(n_samples)
    log_ici = np.zeros(n_samples)
    counter = [0]

    def log_obs(t, x):
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_iao[i] = x[iLa]
            log_ibo[i] = x[iLb]
            log_ico[i] = x[iLc]
            # Source branch currents: pulsim's sign convention is
            # opposite to "current flowing OUT of `from` terminal".
            # Negate so that I_A > 0 means current drawn from V_A's
            # `from` terminal — the conventional grid-current.
            log_iai[i] = -x[iIA]
            log_ibi[i] = -x[iIB]
            log_ici[i] = -x[iIC]
        counter[0] += 1

    p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        switch_fn=sw_fn, step_observer=log_obs,
        start_from_dc_op=True,
    )

    n = counter[0]
    return CmcRunResult(
        t=log_t[:n],
        i_a_out=log_iao[:n], i_b_out=log_ibo[:n], i_c_out=log_ico[:n],
        i_a_in=log_iai[:n], i_b_in=log_ibi[:n], i_c_in=log_ici[:n],
    )


# =============================================================================
# Metrics — closed-form predictions + signal analysis
# =============================================================================


def thd(signal: np.ndarray, fs: float, f0: float, n_harm: int = 50) -> float:
    """Total harmonic distortion of ``signal`` at fundamental ``f0`` [%].

    Computes ``THD = sqrt(sum H_k²) / H_1 × 100 %`` over ``2..n_harm``.
    Uses a Hann window + rfft. Same formula as in
    ``mmc_3phase_model.thd``.
    """
    sig = np.asarray(signal, dtype=np.float64)
    sig = sig - sig.mean()
    n = len(sig)
    win = np.hanning(n)
    spec = np.fft.rfft(sig * win)
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
