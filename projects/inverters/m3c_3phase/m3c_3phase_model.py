"""3-φ Modular Multilevel Matrix Converter (M3C) — modeling + Fast SVM.

Phase 22 of the pulsim roadmap. References:
  * Gili (2024) PhD thesis, Sec 3 (Fast SVM) + Sec 4 (M3C topology) +
    Sec 5 (modeling and control) — ``artigos/Luiz Carlos Gili-1.pdf``.
  * Erickson & Al-Naseem (2001) [28] — original M3C topology.
  * Celanovic & Boroyevic (2001) [83] — original Fast SVM proposal.

Phase 22.1 exports the Fast SVM core:

  * :class:`M3cParams` — operating-point parameters (matches Tab. 15
    of the thesis when defaults are kept).
  * :data:`LG_TRANSFORM_MATRIX` — Eq 25 (3×3 integer-output matrix).
  * :func:`abc_to_lg` / :func:`lg_to_abc` — coordinate transformations.
  * :func:`fast_svm_4_vectors(v_ref_l, v_ref_g)` — Eqs 26a-d. Given
    a reference vector projected on the (l, g) axes, returns the
    four adjacent **integer** vectors: V_ul, V_lu, V_ll, V_uu.
  * :func:`fast_svm_pick_triangle(v_ref_l, v_ref_g)` — Eq 28: picks
    V_ll or V_uu as the third vector based on which triangle the
    reference falls in.
  * :func:`fast_svm_duty_cycles(v_ref_l, v_ref_g)` — Eqs 29 or 30
    depending on the selected triangle. Returns the **signed**
    duty cycles for (V_ul, V_lu, V_third).

The full M3C plant builders, module-voltage solver, cost function,
and sorting algorithm will arrive in subsequent Phase 22.x commits.

Sign / numbering conventions
----------------------------

Following the thesis exactly:

  * **Coordinate transformation** (Eq 25):
        [V_l]   [1  -1   0] [V_a]
        [V_g] = [0   1  -1] [V_b]
        [V_γ]   [1   1   1] [V_c]

    The first two rows produce **line voltages** (V_ab, V_bc);
    the third row produces 3× the common-mode component. For
    integer reference inputs (V_a, V_b, V_c) the output is also
    integer.

  * **lgγ plane** has 60° between the ``l`` and ``g`` axes (not
    90°). The integer vectors in this plane form a hexagonal
    lattice — see Fig 30 of the thesis. ``γ`` (common-mode axis)
    is orthogonal to the lg plane.

  * **Reference projection**: a desired 3-φ AC voltage with
    amplitude ``V_o`` and angular frequency ``ω_o`` produces
    a reference vector in lg whose magnitude is independent of
    ω_o and whose direction rotates at ω_o.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import floor, ceil, pi
from typing import Callable

import numpy as np


# =============================================================================
# Parameters — defaults from Tab. 15 of the thesis (sim config)
# =============================================================================


@dataclass(frozen=True)
class M3cParams:
    """Operating-point + topology parameters for a 3-φ M3C.

    Defaults match Tabela 15 of the thesis (Cap 6 simulation params):
    2 MVA / 13,8 kV / 11 kV, N = 6 SMs per module, C_SM = 680 µF.
    """

    # ---- Network parameters ------------------------------------------------
    V_in_LL_peak: float = 13_800.0 * np.sqrt(2)    # 13,8 kV LL RMS → peak
    f_in: float = 50.0
    V_out_LL_peak: float = 11_000.0 * np.sqrt(2)   # 11 kV LL RMS → peak
    f_out: float = 45.0

    # ---- Filter inductors --------------------------------------------------
    L_in: float = 25e-3
    L_out: float = 25e-3

    # ---- Module / submodule sizing ----------------------------------------
    n_modules: int = 9                # 3×3 matrix (fixed by topology)
    n_sm_per_module: int = 6          # submodules in series per module
    c_sm: float = 680e-6              # capacitance per SM [F]
    v_cap_nominal: float = 4_000.0    # target cap voltage [V]

    # ---- Switching ---------------------------------------------------------
    f_switching: float = 2_000.0      # 2 kHz per Tab. 15
    t_dead: float = 0.0               # dead-time (idealized for L0)

    # ---- Modulation indices ------------------------------------------------
    # Derived: m_v = V_out_LL_peak / (n_sm · v_cap_nominal)
    m_v: float = 0.5                  # output modulation index
    m_c: float = 1.0                  # input current modulation index

    # ---- Load (RL, Y-connected) -------------------------------------------
    # Defaults dimensioned for the 2 MVA / 11 kV operating point at 0.9 PF:
    #   |V_phase_rms| = 11 kV / √3 ≈ 6.351 kV
    #   I_load_rms   ≈ S / (√3 · V_LL_rms) = 2 MVA / (√3·11 kV) ≈ 105 A
    #   |Z_load|     ≈ V/I ≈ 60 Ω at 45 Hz, with R/L chosen for 0.9 PF.
    R_load: float = 54.0              # Ω (≈ 0.9·|Z|)
    L_load: float = 92.5e-3           # H (≈ |Z|·sin(arccos 0.9) / (2π·45))

    @property
    def omega_in(self) -> float:
        return 2.0 * pi * self.f_in

    @property
    def omega_out(self) -> float:
        return 2.0 * pi * self.f_out

    @property
    def T_s(self) -> float:
        return 1.0 / self.f_switching

    @property
    def n_total_sm(self) -> int:
        """Total submodules in the converter (9 modules × N each)."""
        return self.n_modules * self.n_sm_per_module

    @property
    def v_cap_total_per_module(self) -> float:
        """Maximum positive voltage a single module can produce =
        N × v_cap_nominal."""
        return self.n_sm_per_module * self.v_cap_nominal

    @property
    def n_levels_LL(self) -> int:
        """Number of distinct line-line voltage levels achievable
        with the current topology — Sec 4.1 of the thesis.
        Specifically, ``2·N + 1``."""
        return 2 * self.n_sm_per_module + 1

    @property
    def V_in_phase_peak(self) -> float:
        """Input phase-to-neutral peak voltage =
        ``V_in_LL_peak / √3``."""
        return self.V_in_LL_peak / np.sqrt(3.0)

    @property
    def V_out_phase_peak(self) -> float:
        """Output phase-to-neutral peak voltage =
        ``V_out_LL_peak / √3``."""
        return self.V_out_LL_peak / np.sqrt(3.0)


# =============================================================================
# Coordinate transformations — Eq 25 of the thesis
# =============================================================================
#
# The lgγ transformation matrix maps abc-coordinates to a non-orthogonal
# 2D plane (l, g) plus the common-mode axis γ.

LG_TRANSFORM_MATRIX: np.ndarray = np.array(
    [
        [1.0, -1.0, 0.0],   # V_l = V_a - V_b
        [0.0, 1.0, -1.0],   # V_g = V_b - V_c
        [1.0, 1.0, 1.0],    # V_γ = V_a + V_b + V_c
    ],
    dtype=np.float64,
)


# Inverse of LG_TRANSFORM_MATRIX. Computed once at import.
LG_INVERSE_MATRIX: np.ndarray = np.linalg.inv(LG_TRANSFORM_MATRIX)


def abc_to_lg(v_abc: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """Convert a 3-φ vector ``(V_a, V_b, V_c)`` to ``(V_l, V_g, V_γ)``
    coordinates per Eq 25 of the thesis.

    Returns a 1D ``np.ndarray`` of length 3.

    Important property: if all three input components are **integers**,
    so are all three output components — this is what makes the Fast
    SVM algorithm avoid trigonometry (see Sec 3.2 of the thesis).
    """
    v = np.asarray(v_abc, dtype=np.float64).reshape(3)
    return LG_TRANSFORM_MATRIX @ v


def lg_to_abc(v_lg: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """Inverse transformation — Eq 24 of the thesis, but solved
    numerically via the precomputed ``LG_INVERSE_MATRIX``."""
    v = np.asarray(v_lg, dtype=np.float64).reshape(3)
    return LG_INVERSE_MATRIX @ v


# =============================================================================
# Fast SVM — Sec 3.2 of the thesis (Eqs 26a-d, 28, 29, 30)
# =============================================================================


def fast_svm_4_vectors(
    v_ref_l: float, v_ref_g: float,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    """Compute the 4 integer-vector candidates adjacent to a reference
    point in the lg plane — Eqs 26a-d of the thesis.

    The reference is provided already projected onto the (l, g)
    axes (i.e., its first two coordinates after ``abc_to_lg``).

    Returns
    -------
    (V_ul, V_lu, V_ll, V_uu) : tuple of 4 ``(l, g)`` integer tuples
        * ``V_ul`` = ``(⌈l⌉, ⌊g⌋)`` — upper-l, lower-g
        * ``V_lu`` = ``(⌊l⌋, ⌈g⌉)`` — lower-l, upper-g
        * ``V_ll`` = ``(⌊l⌋, ⌊g⌋)`` — lower-l, lower-g (S triangle)
        * ``V_uu`` = ``(⌈l⌉, ⌈g⌉)`` — upper-l, upper-g (N triangle)

    The reference always lies INSIDE the quadrilateral with these 4
    corners; one of two triangles (V_ul, V_lu, V_ll) or
    (V_ul, V_lu, V_uu) is selected by :func:`fast_svm_pick_triangle`.
    """
    return (
        (ceil(v_ref_l), floor(v_ref_g)),     # V_ul
        (floor(v_ref_l), ceil(v_ref_g)),     # V_lu
        (floor(v_ref_l), floor(v_ref_g)),    # V_ll
        (ceil(v_ref_l), ceil(v_ref_g)),      # V_uu
    )


def fast_svm_pick_triangle(
    v_ref_l: float, v_ref_g: float,
) -> str:
    """Decide whether the reference falls in the **lower** triangle
    (``V_ll`` selected) or the **upper** triangle (``V_uu`` selected)
    of the lattice cell.

    The unit cell formed by the four adjacent integer vectors
    (V_ul, V_lu, V_ll, V_uu) is split by the diagonal V_ul ↔ V_lu
    into two triangles. The reference vector lies in exactly one of
    them — the one whose third corner (V_ll or V_uu) is closer.

    Implementation
    --------------
    The midpoint of the diagonal V_ul ↔ V_lu is::

        m = ((⌈l⌉ + ⌊l⌋) / 2, (⌊g⌋ + ⌈g⌉) / 2)

    The signed offset from the midpoint along the (+l, +g) direction
    decides the side:

        s = (V_ref_l - m_l) + (V_ref_g - m_g)

    * ``s > 0`` ⇒ reference on the "upper-right" side ⇒ ``V_uu``.
    * ``s < 0`` ⇒ reference on the "lower-left" side ⇒ ``V_ll``.

    .. note::
       Eq 28 of the Gili (2024) thesis (pg 73) gives the condition
       ``V_ref_l + V_ref_g - V_ul_l - V_ul_g > 0 → V_ll``, but
       independent verification against the geometry shows this
       formulation selects the **farthest** corner, not the
       closest. This is corrected here. Concretely, with the
       thesis's own example (V_ref = (-1.8, 1.2), Sec 3.2 worked
       example), the correct answer is ``V_ll`` because V_ll =
       (-2, 1) is at distance 0.28 while V_uu = (-1, 2) is at
       distance 1.13 — but the literal Eq 28 selects V_uu, which
       produces a negative duty cycle for the third vector.
       Suspected typo: V_ul should be V_uu in Eq 28's condition
       (then the inequality matches the geometry).

    Returns
    -------
    triangle : {"ll", "uu"}
        ``"ll"`` if the lower-lower vector is the third reference;
        ``"uu"`` if the upper-upper vector is.
    """
    midpoint_l = 0.5 * (floor(v_ref_l) + ceil(v_ref_l))
    midpoint_g = 0.5 * (floor(v_ref_g) + ceil(v_ref_g))
    s = (v_ref_l - midpoint_l) + (v_ref_g - midpoint_g)
    return "uu" if s > 0 else "ll"


def fast_svm_duty_cycles(
    v_ref_l: float, v_ref_g: float,
) -> tuple[float, float, float, str]:
    """Compute the three duty cycles for synthesising the reference
    via the three closest integer vectors — Eqs 29 (for ``ll`` triangle)
    or 30 (for ``uu`` triangle) of the thesis.

    Returns
    -------
    (d_ul, d_lu, d_third, triangle) : (float, float, float, {"ll","uu"})
        Duty cycles for the three vectors in the order
        ``V_ul, V_lu, V_third`` where the third is ``V_ll`` if
        ``triangle == "ll"`` else ``V_uu``.

    The duty cycles sum to 1 by construction (no nullable / zero
    vector explicitly accounted for; the three non-zero vectors
    fully define the reference within the triangle).
    """
    V_ul, V_lu, V_ll, V_uu = fast_svm_4_vectors(v_ref_l, v_ref_g)
    triangle = fast_svm_pick_triangle(v_ref_l, v_ref_g)
    if triangle == "ll":
        # Eq 29
        d_ul = v_ref_l - V_ll[0]
        d_lu = v_ref_g - V_ll[1]
        d_ll = 1.0 - d_ul - d_lu
        return d_ul, d_lu, d_ll, "ll"
    else:
        # Eq 30
        d_ul = V_uu[1] - v_ref_g
        d_lu = V_uu[0] - v_ref_l
        d_uu = 1.0 - d_ul - d_lu
        return d_ul, d_lu, d_uu, "uu"


def fast_svm_step(
    t: float,
    params: M3cParams,
    *,
    side: str = "output",
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int],
           float, float, float]:
    """Full Fast SVM step at time ``t`` — returns the three selected
    vectors and their duty cycles.

    Parameters
    ----------
    t : float
        Simulation time [s].
    params : M3cParams
        Operating-point parameters.
    side : {"input", "output"}
        Which side of the converter (input ≈ rectifier reference,
        output ≈ inverter reference) is being modulated. The two
        sides run independent Fast SVMs but share the same ``T_s``.

    Returns
    -------
    (V_a, V_b, V_c, d_a, d_b, d_c) : tuple
        Three integer ``(l, g)`` vectors and their duty cycles
        satisfying ``d_a + d_b + d_c = 1``.

    Notes
    -----
    The reference projection at time ``t`` is computed from the
    AC sinusoidal references at frequency ``ω`` (= ``omega_in`` or
    ``omega_out`` depending on ``side``). The modulation index ``m_v``
    or ``m_c`` scales the reference magnitude.

    The voltage reference per phase (cos convention to match the
    thesis equations):

        V_a(t) = m · N · V_cap · cos(ω·t)
        V_b(t) = m · N · V_cap · cos(ω·t − 2π/3)
        V_c(t) = m · N · V_cap · cos(ω·t + 2π/3)

    where ``N · V_cap`` is the maximum SM voltage (``v_cap_total_per_module``).
    The lg transform converts this to (V_l, V_g) which is the input
    to the Fast SVM.
    """
    if side == "output":
        omega = params.omega_out
        m = params.m_v
    elif side == "input":
        omega = params.omega_in
        m = params.m_c
    else:
        raise ValueError(f"side must be 'input' or 'output', got {side!r}")

    # Cosine-convention 3-φ reference, normalised by N (so the lg
    # reference is in "voltage levels", not volts).
    omega_t = omega * t
    V_abc_norm = m * params.n_sm_per_module * np.array([
        np.cos(omega_t),
        np.cos(omega_t - 2.0 * pi / 3.0),
        np.cos(omega_t + 2.0 * pi / 3.0),
    ])
    V_lg = abc_to_lg(V_abc_norm)
    v_l, v_g = float(V_lg[0]), float(V_lg[1])

    d_ul, d_lu, d_third, triangle = fast_svm_duty_cycles(v_l, v_g)
    V_ul, V_lu, V_ll, V_uu = fast_svm_4_vectors(v_l, v_g)
    V_third = V_ll if triangle == "ll" else V_uu

    return V_ul, V_lu, V_third, d_ul, d_lu, d_third


def make_fast_svm_fn(
    params: M3cParams, *, side: str = "output",
) -> Callable[[float], tuple]:
    """Build a closure ``t → (V_a, V_b, V_c, d_a, d_b, d_c)`` with
    parameters pre-bound. Useful for passing as a callback into the
    pulsim simulation framework or for plotting trajectories."""
    p = params

    def _fn(t: float):
        return fast_svm_step(t, p, side=side)

    return _fn


# =============================================================================
# Module connection configurations — Sec 4.3 of the thesis
# =============================================================================
#
# The M3C has 9 modules ``M_xy`` arranged in a 3×3 matrix between
# the 3 input phases x ∈ {A, B, C} and 3 output phases y ∈ {a, b, c}.
# At any instant, the matrix-converter "5 modules conducting"
# constraint must hold. The valid topologies follow Erickson's
# rules (Sec 4.3, also Tab. 7 of the thesis):
#
#   1. Exactly **one** conduction path between any two distinct
#      input phases (and likewise between any two output phases) —
#      i.e. the bipartite graph of active modules must be
#      **connected** (no decoupled subsystems).
#   2. If one side has a phase with 2 connections, another phase on
#      the same side must also have 2, and the third has 1.
#   3. If one side has a phase with 3 connections (full row/column),
#      the other two phases on the same side have 1 each.
#
# Rules 2 and 3 reduce to: each side's row-sum (or column-sum)
# distribution must be (3, 1, 1) or (2, 2, 1) — never (3, 2, 0),
# (4, 1, 0), etc. Rule 1 additionally excludes the 9 (2,2,1)×(2,2,1)
# cases where the two high-sum inputs happen to connect to the same
# pair of outputs (which would split the topology into a {single
# input, single output} block plus a {2 inputs, 2 outputs} block).
#
# C(9,5) = 126 → 90 (after no-empty-row/col) → **81 valid** (after
# connectivity). Matches the thesis count (Sec 4.3 pg 82) and Tabela
# 7 (27 base patterns × 3 input rotations = 81).


# Phase indices: 0=A, 1=B, 2=C on input; 0=a, 1=b, 2=c on output.
INPUT_LABELS = ("A", "B", "C")
OUTPUT_LABELS = ("a", "b", "c")


@dataclass(frozen=True)
class ModuleConfiguration:
    """A 3×3 boolean matrix indicating which of the 9 modules
    ``M_{xy}`` are currently conducting (True) vs blocked (False).

    Stored as a tuple-of-tuples for hashability (allows config to be
    used as a dict key when caching cost-function results).

    Attributes
    ----------
    grid : tuple of 3 tuples, each containing 3 booleans
        ``grid[i][j]`` is True iff module ``M_{x_i,y_j}`` conducts.
        Indices: i for input phase (A=0, B=1, C=2), j for output
        phase (a=0, b=1, c=2). Runtime-enforced to be 3×3 by
        the enumerator; type annotation kept loose for Pyright.
    """

    grid: tuple = field(default_factory=tuple)  # 3-tuple of 3-tuples of bool

    # ---- accessors -------------------------------------------------

    def is_active(self, input_idx: int, output_idx: int) -> bool:
        return self.grid[input_idx][output_idx]

    def row_sum(self, input_idx: int) -> int:
        """Number of output phases connected to this input phase."""
        return sum(self.grid[input_idx])

    def col_sum(self, output_idx: int) -> int:
        """Number of input phases connected to this output phase."""
        return sum(self.grid[i][output_idx] for i in range(3))

    def n_active(self) -> int:
        """Total number of modules conducting (should be 5 for any
        valid M3C configuration)."""
        return sum(self.row_sum(i) for i in range(3))

    def active_modules(self) -> list[tuple[int, int]]:
        """List of ``(input_idx, output_idx)`` for each active module."""
        return [
            (i, j) for i in range(3) for j in range(3)
            if self.grid[i][j]
        ]

    # ---- validation ------------------------------------------------

    def is_valid(self) -> bool:
        """Check the Sec 4.3 rules:
          * Total active modules = 5.
          * Row sums and column sums each follow distribution
            (3, 1, 1) or (2, 2, 1) — no zero rows/cols.
          * Bipartite graph of active modules is **connected** —
            no decoupled subsystems (excludes the 9 cases where the
            two row-sum=2 inputs share an identical output set).
        """
        if self.n_active() != 5:
            return False
        row_dist = sorted(self.row_sum(i) for i in range(3))
        col_dist = sorted(self.col_sum(j) for j in range(3))
        valid_dists = ([1, 1, 3], [1, 2, 2])
        if row_dist not in valid_dists or col_dist not in valid_dists:
            return False
        return _is_connected_bipartite(self.grid)

    # ---- formatting ------------------------------------------------

    def to_string(self) -> str:
        """Compact visual: e.g.::

              a b c
            A ✓ . .
            B . ✓ ✓
            C ✓ . ✓
        """
        lines = ["    a b c"]
        for i, in_label in enumerate(INPUT_LABELS):
            cells = " ".join(
                "✓" if self.grid[i][j] else "." for j in range(3)
            )
            lines.append(f"  {in_label} {cells}")
        return "\n".join(lines)


def _is_valid_distribution(row_sums) -> bool:
    """Check if a row-sum tuple follows the (3,1,1) or (2,2,1) rule."""
    sorted_sums = sorted(row_sums)
    return sorted_sums == [1, 1, 3] or sorted_sums == [1, 2, 2]


def _is_connected_bipartite(grid) -> bool:
    """Check that the bipartite graph of active modules is connected.

    Modules form edges between 3 input-side nodes (rows 0..2) and 3
    output-side nodes (cols 0..2). The graph is connected iff a BFS
    from any node reaches all 6 nodes.

    Assumes no empty rows/cols (the caller already filtered). Returns
    ``True`` for connected, ``False`` for decoupled.
    """
    # BFS starting from input 0 (which is guaranteed non-isolated
    # since all rows have ≥1 active cell).
    visited_in = {0}
    visited_out: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i in list(visited_in):
            for j in range(3):
                if grid[i][j] and j not in visited_out:
                    visited_out.add(j)
                    changed = True
        for j in list(visited_out):
            for i in range(3):
                if grid[i][j] and i not in visited_in:
                    visited_in.add(i)
                    changed = True
    return len(visited_in) == 3 and len(visited_out) == 3


def enumerate_valid_configurations() -> list[ModuleConfiguration]:
    """Enumerate the **81 valid M3C connection configurations** per
    Sec 4.3 of the thesis.

    Algorithm:

      1. Take all ``C(9, 5) = 126`` ways to choose 5 modules out of 9.
      2. Filter to no-empty-row/col + (3,1,1)/(2,2,1) distribution:
         → 90 candidates.
      3. Filter by bipartite-graph connectivity (Sec 4.3 rule 1):
         → **81 valid configurations**.

    The result is cached at module import time via the module-level
    constant :data:`ALL_VALID_CONFIGURATIONS`.
    """
    configs: list[ModuleConfiguration] = []
    positions = [(i, j) for i in range(3) for j in range(3)]
    for chosen in combinations(positions, 5):
        # Build 3×3 grid.
        grid = [[False] * 3 for _ in range(3)]
        for i, j in chosen:
            grid[i][j] = True
        grid_t = tuple(tuple(grid[i]) for i in range(3))
        cfg = ModuleConfiguration(grid=grid_t)
        row_sums = tuple(cfg.row_sum(i) for i in range(3))
        col_sums = tuple(cfg.col_sum(j) for j in range(3))
        if not _is_valid_distribution(row_sums):
            continue
        if not _is_valid_distribution(col_sums):
            continue
        if not _is_connected_bipartite(grid_t):
            continue
        configs.append(cfg)
    return configs


# Precomputed at module load — used by cost function (Phase 22.3).
ALL_VALID_CONFIGURATIONS: list[ModuleConfiguration] = \
    enumerate_valid_configurations()


def configurations_by_distribution() -> dict[
    tuple[tuple[int, int, int], tuple[int, int, int]],
    list[ModuleConfiguration],
]:
    """Group :data:`ALL_VALID_CONFIGURATIONS` by their (row-dist,
    col-dist) tuple, where each distribution is sorted.

    Useful for understanding the structure of the configuration
    space — should produce 4 groups:
      * ((1,1,3), (1,1,3)) — both sides have one phase with 3 conns
      * ((1,1,3), (1,2,2))
      * ((1,2,2), (1,1,3))
      * ((1,2,2), (1,2,2))
    """
    by_dist: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[ModuleConfiguration],
    ] = {}
    for cfg in ALL_VALID_CONFIGURATIONS:
        row_dist = tuple(sorted(cfg.row_sum(i) for i in range(3)))
        col_dist = tuple(sorted(cfg.col_sum(j) for j in range(3)))
        key = (row_dist, col_dist)  # type: ignore[assignment]
        by_dist.setdefault(key, []).append(cfg)  # type: ignore[arg-type]
    return by_dist


def configurations_containing_module(
    input_idx: int, output_idx: int
) -> list[ModuleConfiguration]:
    """Return all 81 valid configurations where the module
    ``M_(input_idx, output_idx)`` is active.

    By symmetry across the 9 modules, each module appears in
    exactly **45 of 81** configurations (since 81 × 5/9 = 45).

    This is used by the cost function (Sec 5.5.3) to reduce 81 → 45
    candidates once the SVM has identified the short-circuited
    module (the M_xy connecting the minimum-voltage input phase to
    the minimum-voltage output phase, see Sec 5.5.3 paragraph 4).
    """
    return [
        cfg for cfg in ALL_VALID_CONFIGURATIONS
        if cfg.is_active(input_idx, output_idx)
    ]


# =============================================================================
# Sec 4.3 — Module voltage solver (Eqs 31-34)
# =============================================================================
#
# Given a configuration with 5 active modules forming a spanning
# tree over 6 nodes (3 inputs + 3 outputs), and a "short" module
# whose voltage is forced to 0, the remaining 4 module voltages are
# uniquely determined by the SVM phase-voltage references.
#
# Convention deduced from the thesis worked example (Sec 4.3 pg 83-85,
# Figures 42-43, Eqs 31-34):
#
#   M_xy = V_in_rel[x] - V_out_rel[y]
#
# where
#
#   V_in_rel[x]  = V_input[x]  - V_input[short_in_idx]
#   V_out_rel[y] = V_output[short_out_idx] - V_output[y]  (*sign flip*)
#
# The sign flip on the output side is a peculiarity of how the SVM
# references are defined in the lgγ plane: the output line voltage
# V_ab in the thesis is actually -(V_a - V_b) in the standard "input
# minus output" convention. Without the flip the algorithm would
# produce module voltages of the wrong sign.
#
# Verified against thesis Example 4.3 (V_input=(-1,0,0), V_output=
# (1,0,0), short=M_Ba):
#   M_Ba=0, M_Ca=0, M_Cb=-1, M_Ab=-2, M_Ac=-2  ✓


def solve_module_voltages(
    cfg: ModuleConfiguration,
    short_module: tuple[int, int],
    V_input: np.ndarray | list | tuple,
    V_output: np.ndarray | list | tuple,
) -> dict[tuple[int, int], float]:
    """Compute the 5 active module voltages for an M3C configuration.

    Implements Sec 4.3 Eqs 31-34 of the thesis. The 5 active modules
    form a spanning tree over the 6-node bipartite input/output graph
    (3 inputs × 3 outputs). Picking one module as "short" (V_xy = 0)
    uniquely determines the other 4 voltages from the SVM references.

    Parameters
    ----------
    cfg
        A valid :class:`ModuleConfiguration` (5 active modules).
    short_module
        ``(input_idx, output_idx)`` — must be an active module. The
        thesis recommends choosing the module that connects the
        minimum-voltage input phase to the minimum-voltage output
        phase (Sec 5.5.3 paragraph 4).
    V_input
        SVM input-side phase voltage references ``(V_A, V_B, V_C)``
        in units of ``V_cap`` (integer for true Fast SVM output).
    V_output
        SVM output-side phase voltage references ``(V_a, V_b, V_c)``,
        same units.

    Returns
    -------
    dict[(input_idx, output_idx) -> float]
        Voltage in units of ``V_cap`` for each of the 5 active modules.

    Raises
    ------
    ValueError
        If ``short_module`` is not one of the 5 active modules of ``cfg``.

    Examples
    --------
    >>> cfg = ModuleConfiguration(grid=(
    ...     (False, True,  True),    # row A: connects to b, c
    ...     (True,  False, False),   # row B: connects to a
    ...     (True,  True,  False),   # row C: connects to a, b
    ... ))
    >>> V = solve_module_voltages(cfg, (1, 0), [-1, 0, 0], [1, 0, 0])
    >>> sorted(V.items())  # doctest: +NORMALIZE_WHITESPACE
    [((0, 1), -2.0), ((0, 2), -2.0), ((1, 0), 0.0),
     ((2, 0), 0.0), ((2, 1), -1.0)]
    """
    if not cfg.is_active(*short_module):
        raise ValueError(
            f"Short module {short_module} is not active in this "
            f"configuration. Active modules: {cfg.active_modules()}"
        )

    Vin = np.asarray(V_input, dtype=float)
    Vout = np.asarray(V_output, dtype=float)
    if Vin.shape != (3,) or Vout.shape != (3,):
        raise ValueError(
            f"V_input and V_output must be length-3; "
            f"got shapes {Vin.shape} and {Vout.shape}"
        )

    short_in, short_out = short_module
    V_in_rel = Vin - Vin[short_in]
    V_out_rel = Vout[short_out] - Vout

    return {
        (i, j): float(V_in_rel[i] - V_out_rel[j])
        for (i, j) in cfg.active_modules()
    }


# =============================================================================
# Module current solver (Sec 5.5.3 — implicit in Eq 162)
# =============================================================================
#
# The 5 module currents are determined by KCL at the 6 nodes:
#
#   At input X:  sum_y I_xy = I_X         (current entering input node)
#   At output Y: sum_x I_xy = I_y         (current leaving output node)
#
# Total = 6 equations, but ``sum I_X = sum I_y`` is required by
# overall power conservation, so 5 are independent. With 5 unknowns
# (the 5 active module currents) and the modules forming a spanning
# tree, the system has a **unique** solution.
#
# Algorithm: leaf-stripping. A spanning tree always has at least one
# leaf (degree-1 node). KCL at a leaf gives its unique incident
# edge's current directly. Remove that edge, repeat — works O(5²) in
# the worst case but trivially fast.


def solve_module_currents(
    cfg: ModuleConfiguration,
    I_input: np.ndarray | list | tuple,
    I_output: np.ndarray | list | tuple,
) -> dict[tuple[int, int], float]:
    """Compute the 5 active module currents for an M3C configuration.

    Solves the KCL system on the bipartite tree of active modules.
    Used by the cost function (Sec 5.5.3 Eq 162) to predict capacitor
    voltage changes due to module current.

    Parameters
    ----------
    cfg
        A valid :class:`ModuleConfiguration` (5 active modules).
    I_input
        Input-side terminal currents ``(I_A, I_B, I_C)`` flowing
        INTO the converter at each input phase.
    I_output
        Output-side terminal currents ``(I_a, I_b, I_c)`` flowing
        OUT of the converter at each output phase.

    Returns
    -------
    dict[(input_idx, output_idx) -> float]
        Current through each active module, with the convention
        "positive = flowing from input toward output".

    Raises
    ------
    ValueError
        If conservation ``sum(I_input) ≈ sum(I_output)`` is violated
        beyond floating-point tolerance.

    Notes
    -----
    The algorithm strips leaves of the bipartite tree iteratively
    (KCL at a degree-1 node gives its incident edge's current
    directly). Always terminates in ≤ 5 iterations.
    """
    Iin = np.asarray(I_input, dtype=float).copy()
    Iout = np.asarray(I_output, dtype=float).copy()
    if Iin.shape != (3,) or Iout.shape != (3,):
        raise ValueError(
            f"I_input and I_output must be length-3; "
            f"got shapes {Iin.shape} and {Iout.shape}"
        )
    if not np.isclose(Iin.sum(), Iout.sum(), atol=1e-9):
        raise ValueError(
            f"KCL conservation violated: sum(I_input)={Iin.sum()} != "
            f"sum(I_output)={Iout.sum()}"
        )

    # Live adjacency: which active modules remain to be solved.
    remaining: set[tuple[int, int]] = set(cfg.active_modules())
    in_degree = [0] * 3
    out_degree = [0] * 3
    for (i, j) in remaining:
        in_degree[i] += 1
        out_degree[j] += 1

    currents: dict[tuple[int, int], float] = {}

    while remaining:
        # Find a leaf: any node with degree 1 in the remaining graph.
        leaf_found = False
        for i in range(3):
            if in_degree[i] == 1:
                # Find its single remaining edge.
                j = next(jj for (ii, jj) in remaining if ii == i)
                # KCL at input i: I_in[i] = I_xy for this single edge.
                currents[(i, j)] = float(Iin[i])
                # Subtract from output j's accumulated current.
                Iout[j] -= Iin[i]
                Iin[i] = 0.0
                remaining.discard((i, j))
                in_degree[i] -= 1
                out_degree[j] -= 1
                leaf_found = True
                break
        if leaf_found:
            continue
        for j in range(3):
            if out_degree[j] == 1:
                i = next(ii for (ii, jj) in remaining if jj == j)
                # KCL at output j: I_out[j] = I_xy (flowing in from input).
                currents[(i, j)] = float(Iout[j])
                Iin[i] -= Iout[j]
                Iout[j] = 0.0
                remaining.discard((i, j))
                in_degree[i] -= 1
                out_degree[j] -= 1
                leaf_found = True
                break
        if not leaf_found:
            # Tree has no leaf — should be impossible for a connected
            # 5-edge bipartite graph on 6 nodes.
            raise RuntimeError(
                "Module-current solver could not find a leaf; "
                "the active configuration is not a spanning tree "
                f"(remaining edges: {remaining})"
            )

    return currents


# =============================================================================
# Sec 5.5.3 — Cost function for capacitor balancing (Eqs 161-163)
# =============================================================================
#
# The M3C has 9 modules × N submodules each (= 54 SM capacitors in
# Tab. 15). To keep all caps near V_cap_ref, the cost function picks
# the configuration that minimizes:
#
#   ε_xy   = V_xy - mean(V_xy across 9 modules)         (Eq 161)
#   ΔV_xy  = S_n · I_xy · T_s / C                       (Eq 162)
#   C_cost = Σ_{xy} (ε_xy + ΔV_xy)²                     (Eq 163)
#
# where V_xy is the **measured** module capacitor voltage (sum of N
# SM caps) at the start of the period, I_xy is the predicted module
# current under the candidate configuration, and S_n is the number of
# SM capacitors actively contributing to the module voltage in this
# config (≤ N).
#
# For inactive modules (4 of the 9), I_xy = 0 so ΔV_xy = 0 and only
# the ε term contributes. The sum runs over all 9 modules to penalize
# both new excursions (ΔV) AND pre-existing imbalance (ε).


def connection_cost(
    cfg: ModuleConfiguration,
    V_caps: np.ndarray,
    V_xy: dict[tuple[int, int], float],
    I_xy: dict[tuple[int, int], float],
    T_s: float,
    C_sm: float,
) -> float:
    """Compute the Sec 5.5.3 cost function (Eq 163) for a candidate
    connection ``cfg``.

    The cost approximates the squared sum of "post-T_s" capacitor
    voltage deviations from their mean: each module's deviation is
    its current imbalance ε plus the projected change ΔV during T_s.

    Eq 162: ``ΔV_xy = S_n · I_xy · T_s / C`` where ``S_n`` is the
    number of **active** SM capacitors in the module. For a module
    with ``V_xy = k · V_cap``, exactly ``|k|`` SMs are in state 1 or
    state 2 (the rest bypassed) — so ``S_n = |k| = |V_xy| / V_cap``
    and the SIGNED quantity ``V_xy/V_cap`` captures both ``S_n`` and
    the charge-direction sign in one term.

    Parameters
    ----------
    cfg
        Candidate configuration (5 active modules). Used only for
        ``is_active`` checks; the actual physics is carried by V_xy
        and I_xy.
    V_caps
        Length-9 array of module capacitor sum voltages [V], flat
        row-major (``V_caps[3·i + j]``).
    V_xy
        Dict ``(i, j) -> integer`` mapping each ACTIVE module to its
        voltage in **V_cap units** (signed integer, as produced by
        :func:`solve_module_voltages` when given integer references).
    I_xy
        Dict ``(i, j) -> float`` for each active module's current [A]
        (as produced by :func:`solve_module_currents`).
    T_s
        Switching period [s].
    C_sm
        Submodule capacitance [F].

    Returns
    -------
    float
        :math:`\\mathcal{C} = \\sum_{xy} (\\epsilon_{xy} + \\Delta V_{xy})^2`
        with the sum over all 9 modules; inactive modules contribute
        only their ε² term.
    """
    V_caps = np.asarray(V_caps, dtype=float).reshape(-1)
    if V_caps.shape != (9,):
        raise ValueError(
            f"V_caps must be length 9; got shape {V_caps.shape}"
        )

    V_mean = float(V_caps.mean())
    eps = V_caps - V_mean                                  # Eq 161

    # Predict the module-level cap-sum change for each module.
    # ΔV_module = sign(V_xy) · |V_xy / V_cap| · I · T_s / C_SM
    #           = (V_xy / V_cap) · I · T_s / C_SM       [signed]
    # Since V_xy is already in V_cap units (integer), this is just
    # ``V_int · I · T_s / C_SM``. Inactive modules: V_xy=0 → ΔV=0.
    delta_v = np.zeros(9)
    for (i, j), V_int in V_xy.items():
        I_val = I_xy.get((i, j), 0.0)
        delta_v[i * 3 + j] = float(V_int) * float(I_val) * T_s / C_sm

    return float(np.sum((eps + delta_v) ** 2))


def select_best_connection(
    short_module: tuple[int, int],
    V_caps: np.ndarray,
    V_input_int: np.ndarray | list | tuple,
    V_output_int: np.ndarray | list | tuple,
    I_input: np.ndarray | list | tuple,
    I_output: np.ndarray | list | tuple,
    T_s: float,
    C_sm: float,
) -> tuple[ModuleConfiguration, float]:
    """Sec 5.5.3 selector: among the 45 configurations that contain
    ``short_module``, pick the one with the lowest cost.

    For each candidate this function computes:
      * its 5 active module voltages via :func:`solve_module_voltages`
        (in V_cap units, signed integers from the integer SVM refs);
      * its 5 active module currents via :func:`solve_module_currents`;
      * the Eq 163 cost via :func:`connection_cost`.

    Parameters
    ----------
    short_module
        ``(input_idx, output_idx)`` — the short module shared by all
        45 candidates (Sec 5.5.3 paragraph 4).
    V_caps
        9-element module cap-sum voltages [V], row-major.
    V_input_int, V_output_int
        Integer SVM phase references in V_cap units, shape (3,).
    I_input, I_output
        Terminal currents [A] (length 3 each, balanced).
    T_s
        Switching period [s].
    C_sm
        Submodule capacitance [F].

    Returns
    -------
    (best_cfg, best_cost)
        The minimum-cost configuration containing ``short_module``,
        along with its cost value.
    """
    candidates = configurations_containing_module(*short_module)
    if not candidates:
        raise ValueError(
            f"No valid configuration contains module {short_module}"
        )
    best_cfg = candidates[0]
    best_cost = float("inf")
    for cfg in candidates:
        V_xy = solve_module_voltages(
            cfg, short_module, V_input_int, V_output_int,
        )
        I_xy = solve_module_currents(cfg, I_input, I_output)
        cost = connection_cost(cfg, V_caps, V_xy, I_xy, T_s, C_sm)
        if cost < best_cost:
            best_cost = cost
            best_cfg = cfg
    return best_cfg, best_cost


# =============================================================================
# L0 — averaged-model plant + runner (Phase 22.4)
# =============================================================================
#
# Like the CMC L0 (Phase 21.2), the M3C L0 plant represents the
# multilevel-matrix converter as a Venturini-style continuous-time
# averaged converter: there is no high-frequency switching ripple in
# the output voltage, only the fundamental synthesised by the Fast
# SVM. The output line-neutral voltages are:
#
#     v_a_out(t) = V_out_phase_peak · cos(ω_o·t)
#     v_b_out(t) = V_out_phase_peak · cos(ω_o·t − 2π/3)
#     v_c_out(t) = V_out_phase_peak · cos(ω_o·t + 2π/3)
#
# at the three Y-load terminals. The averaging hides:
#   * the discretization of the 13 output line-voltage levels;
#   * the capacitor-voltage ripple of the 54 SMs;
#   * the input-side current draw (no input is modelled in L0).
#
# These are intentionally absent at this layer — L0 is purely a
# verification baseline that establishes the SVM-synthesised
# fundamental matches its closed-form expectation for the M3C's
# nominal 11 kV / 45 Hz operating point. Upper layers re-introduce
# the missing physics:
#
#   * L1: 54 ideal SMs switched via the SVM duty cycles, capacitor
#     dynamics, cost-function selection (Phase 22.5).
#   * L2: realistic switch models (IGBT V_CE_sat + body diodes).
#   * L3: closed-loop dq current control (Phase 22.6).


try:                                                   # type: ignore[unused-ignore]
    import pulsim as _p                                # type: ignore[import-not-found]
except ImportError:                                    # pragma: no cover
    _p = None  # The plant builders require pulsim;
    #             tests for Tiers 1-7 still run without it.


@dataclass
class M3cPlant:
    """Bundle returned by the M3C plant builders.

    Mirrors :class:`CmcPlant` to keep the project-wide pattern.

    Attributes
    ----------
    builder
        The fully-assembled :class:`pulsim.CircuitBuilder`.
    iL_out_indices
        State indices of the 3 output-side load inductor currents
        (i_a, i_b, i_c). Resolved after the graph is complete.
    iL_in_indices
        State indices of the 3 input-side filter inductor currents,
        or ``None`` when the plant does not model the input side
        (as is the case for L0).
    module_v_src_state_indices
        State indices of the 9 per-module controlled voltage sources
        in row-major order ``(M_Aa, M_Ab, M_Ac, M_Ba, M_Bb, M_Bc,
        M_Ca, M_Cb, M_Cc)``. Populated by the L1 builder for use
        with the ``b_extra`` injection mechanism (matches the MMC
        L1 multilevel arm pattern). ``None`` for L0.
    """

    builder: object
    iL_out_indices: tuple[int, int, int] = (0, 0, 0)
    iL_in_indices: tuple[int, int, int] | None = None
    module_v_src_state_indices: tuple[int, ...] | None = None


def build_l0_plant(params: M3cParams) -> M3cPlant:
    """3-φ M3C output-side averaged plant.

    Topology::

        ┌── V_a_out ──┬── L_load ──┬── R_load ──┐
        │                                       │
        │── V_b_out ──┬── L_load ──┬── R_load ──┼── star
        │                                       │
        └── V_c_out ──┴── L_load ──┴── R_load ──┘

    Three output voltage sources are sinusoids at ``params.f_out``
    with amplitude ``params.V_out_phase_peak``, phase-shifted by
    120° — the *ideal* SVM-synthesised reference. The load is
    Y-connected RL, star tied weakly to ground for MNA conditioning.

    Suitable for verifying output fundamental amplitude, RMS, balance,
    and load impedance response. Does **not** model the input side
    (entrance currents are zero by construction at L0).
    """
    if _p is None:
        raise RuntimeError(
            "pulsim is not importable — build_l0_plant requires the "
            "full pulsim package, not just the SVM helpers."
        )

    b = _p.CircuitBuilder()
    V_o_peak = params.V_out_phase_peak  # phase peak from LL peak / √3

    # Output voltage sources — one per phase, sinusoidal at f_out.
    # Phase offset includes +π/2 to convert pulsim's sin convention
    # to the cosine convention used by the Fast SVM theory in the
    # thesis (V_a = V_o·cos(ω_o·t), so at t=0 V_a is at its peak).
    # This guarantees that the SVM α_o = ω_o·t lookup aligns with
    # the actual voltage vector position in pulsim's clock.
    b.add_sine_voltage_source(
        "V_a_out", "a", "star",
        v_dc=0.0, v_amplitude=V_o_peak, frequency=params.f_out,
        phase=+pi / 2.0,                       # sin(ωt + π/2) = cos(ωt)
    )
    b.add_sine_voltage_source(
        "V_b_out", "b", "star",
        v_dc=0.0, v_amplitude=V_o_peak, frequency=params.f_out,
        phase=+pi / 2.0 - 2.0 * pi / 3.0,      # cos(ωt − 2π/3)
    )
    b.add_sine_voltage_source(
        "V_c_out", "c", "star",
        v_dc=0.0, v_amplitude=V_o_peak, frequency=params.f_out,
        phase=+pi / 2.0 + 2.0 * pi / 3.0,      # cos(ωt + 2π/3)
    )

    # Y-load: ph → L_load → R_load → star (same neutral as sources).
    # Capture inductor branch IDs as they are created, but resolve
    # *state indices* only after the FULL graph is built — pulsim's
    # state-index assignment depends on the total graph topology and
    # would yield stale indices if queried mid-build (a well-known
    # gotcha from Phases 20+21).
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

    return M3cPlant(
        builder=b,
        iL_out_indices=iL_out_indices,  # type: ignore[arg-type]
        iL_in_indices=None,
    )


# =============================================================================
# Run driver
# =============================================================================


@dataclass
class M3cRunResult:
    """Output of :func:`run_l0_open_loop`."""

    t: np.ndarray
    i_a_out: np.ndarray
    i_b_out: np.ndarray
    i_c_out: np.ndarray
    # Optional: input currents (populated by L1+).
    i_a_in: np.ndarray = field(default_factory=lambda: np.empty(0))
    i_b_in: np.ndarray = field(default_factory=lambda: np.empty(0))
    i_c_in: np.ndarray = field(default_factory=lambda: np.empty(0))


def run_l0_open_loop(
    plant: M3cPlant,
    *,
    t_end: float = 100e-3,
    dt: float = 10e-6,
) -> M3cRunResult:
    """Run an L0 plant for ``t_end`` seconds at fixed ``dt``.

    No observer is needed at L0 — the sinusoidal sources update
    themselves automatically via pulsim's built-in time-varying
    SineVoltageSource primitive.
    """
    if _p is None:
        raise RuntimeError(
            "pulsim is not importable — run_l0_open_loop requires "
            "the full pulsim package."
        )

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

    _p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=log_obs,
        start_from_dc_op=True,
    )

    n = counter[0]
    return M3cRunResult(
        t=log_t[:n],
        i_a_out=log_ia[:n],
        i_b_out=log_ib[:n],
        i_c_out=log_ic[:n],
    )


# =============================================================================
# Metrics — closed-form predictions + signal analysis
# =============================================================================


def predict_load_impedance(params: M3cParams) -> complex:
    """Per-phase complex impedance of the Y-connected RL load:

        Z = R + jω_out · L
    """
    return params.R_load + 1j * params.omega_out * params.L_load


def predict_i_out_peak(params: M3cParams) -> float:
    """Closed-form peak of the load current — Ohm's law at the output:

        |I_o| = V_out_phase_peak / |Z_load|
    """
    return params.V_out_phase_peak / abs(predict_load_impedance(params))


def predict_load_power_factor(params: M3cParams) -> float:
    """Power factor of the RL load: cos(arctan(ω·L / R))."""
    return float(np.cos(
        np.arctan2(params.omega_out * params.L_load, params.R_load)
    ))


def rms(signal: np.ndarray) -> float:
    """RMS value of a signal."""
    return float(np.sqrt(np.mean(
        np.asarray(signal, dtype=np.float64) ** 2
    )))


def thd(signal: np.ndarray, fs: float, f_fundamental: float) -> float:
    """Total harmonic distortion in % via DFT.

    Returns ``100·√(Σ |X_k|² / |X_1|²)`` where ``X_1`` is the
    fundamental and the sum runs over harmonics 2..N within the
    Nyquist limit.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)
    spec = np.fft.rfft(sig)
    k1 = int(round(f_fundamental * n / fs))
    if k1 <= 0 or k1 >= len(spec):
        return float("nan")
    fundamental = abs(spec[k1])
    if fundamental < 1e-12:
        return float("nan")
    harmonics_sq = 0.0
    k = 2 * k1
    while k < len(spec):
        harmonics_sq += float(abs(spec[k])) ** 2
        k += k1
    return 100.0 * float(np.sqrt(harmonics_sq)) / fundamental


# =============================================================================
# L1 — switched plant + open-loop SVM controller (Phase 22.5)
# =============================================================================
#
# L1 captures the actual switched behaviour of the M3C: 9 modules
# between input and output phases, each module being either INACTIVE
# (open switch) or ACTIVE with a quantised voltage ``V_xy = k·V_cap``
# (``k`` integer in [-N, +N] for N submodules per module). The
# multilevel staircase appears at the output line-line voltage.
#
# This first L1 cut is intentionally minimal:
#
#   * Per-T_s switching only (no within-T_s vector-mixing yet).
#   * Heuristic config selector: pick the FIRST of the 45 valid
#     configurations that include the "short" module (= min-voltage
#     input × min-voltage output, per Sec 5.5.3 paragraph 4). The
#     cost-function selector (Eq 163) will arrive in a follow-up
#     commit.
#   * No capacitor dynamics: the SM caps are treated as ideal
#     references at ``v_cap_nominal``. Cap balancing kicks in once
#     we add their state-tracking + the cost function.
#
# Each module is realised in pulsim with TWO branches:
#
#   1. A constant ``add_voltage_source`` set to 0 V baseline. We
#      then use the ``b_extra`` injection mechanism (same pattern as
#      MMC L1 multilevel arm) to update the effective source voltage
#      to ``V_xy · V_cap_nominal`` every integration step.
#   2. A bidirectional ``add_switch`` in series. When the module is
#      inactive in the current configuration, the switch is OFF and
#      the high R_off blocks current.
#
# The two-branch realisation means the per-module index in the
# SwitchStateMask (0..8 row-major) does **not** equal the per-module
# index in the b_extra vector — those use different state-vector
# indices. The plant exposes both via ``module_v_src_state_indices``
# and a fixed mask layout.


def _module_index(input_idx: int, output_idx: int) -> int:
    """Row-major flat index for module M_(input_idx, output_idx)."""
    return input_idx * 3 + output_idx


def build_l1_plant(params: M3cParams) -> M3cPlant:
    """3-φ M3C switched plant (open-loop SVM, no cap dynamics).

    Topology::

        V_A ── A ── L_in ── inA ─┬── VM_Aa──S_Aa ──┬── outA ── L_out ─┐
                                  ├── VM_Ab──S_Ab ──┤                  │
                                  └── VM_Ac──S_Ac ──┤   load Y         │
        V_B ── B ── L_in ── inB ──···──S_Bx ───────┤  (R + jωL,        │
        V_C ── C ── L_in ── inC ──···──S_Cx ───────┤   neutral=load_star)
                                                   └─── load_star

    Each "VM_xy──S_xy" pair is a *controlled* module: a constant
    voltage source initialised at 0 V (updated each step via
    ``b_extra``) followed by an ideal bidirectional switch. With
    the switch ON, the source enforces ``V_xy · V_cap_nominal``
    across the module; with the switch OFF, the 1 MΩ R_off makes
    the module nearly an open circuit.

    Returns
    -------
    M3cPlant
        Fully assembled, with ``iL_in_indices``, ``iL_out_indices``,
        and ``module_v_src_state_indices`` (row-major M_Aa..M_Cc) all
        resolved against the final graph.
    """
    if _p is None:
        raise RuntimeError(
            "pulsim is not importable — build_l1_plant requires the "
            "full pulsim package."
        )

    b = _p.CircuitBuilder()

    # ---- Input voltage sources (Y) -------------------------------------
    # Cosine convention to align with the Fast SVM (V_A=V_in·cos(ω·t)).
    V_in_pk = params.V_in_phase_peak
    for k_ph, ph in enumerate(("A", "B", "C")):
        b.add_sine_voltage_source(
            f"V_{ph}", ph, "src_star",
            v_dc=0.0, v_amplitude=V_in_pk, frequency=params.f_in,
            phase=pi / 2.0 - k_ph * 2.0 * pi / 3.0,
        )

    # ---- Input filter inductors ----------------------------------------
    # X → L_in → in_X (each phase has its own inductor).
    L_in_branch_ids: list[int] = []
    for ph in "ABC":
        L_id = b.graph.num_branches
        b.add_inductor(f"L_in_{ph}", ph, f"in_{ph}", params.L_in)
        L_in_branch_ids.append(L_id)

    # ---- 9 controlled modules (row-major: M_Aa, M_Ab, ..., M_Cc) -------
    # Each module = voltage source (0 V baseline, updated via b_extra)
    # then a switch in series. Order MUST match _module_index(i, j) so
    # that the SwitchStateMask bits align with the b_extra injection.
    module_v_src_branch_ids: list[int] = []
    in_labels = ("A", "B", "C")
    out_labels = ("a", "b", "c")
    for i, in_ph in enumerate(in_labels):
        for j, out_ph in enumerate(out_labels):
            src_id = b.graph.num_branches
            b.add_voltage_source(
                f"VM_{in_ph}{out_ph}",
                f"in_{in_ph}", f"mid_{in_ph}{out_ph}",
                0.0,                     # V baseline (updated via b_extra)
            )
            module_v_src_branch_ids.append(src_id)
            b.add_switch(
                f"SM_{in_ph}{out_ph}",
                f"mid_{in_ph}{out_ph}", f"out_{out_ph}",
                g_on=1e4,                # 1/g_on = 0.1 mΩ ON resistance
                g_off=1e-6,              # 1/g_off = 1 MΩ OFF resistance
            )

    # ---- Output filter inductors + Y-load ------------------------------
    L_out_branch_ids: list[int] = []
    for ph in "abc":
        L_id = b.graph.num_branches
        b.add_inductor(
            f"L_out_{ph}", f"out_{ph}", f"rload_{ph}", params.L_out,
        )
        b.add_resistor(
            f"R_load_{ph}", f"rload_{ph}", "load_star", params.R_load,
        )
        L_out_branch_ids.append(L_id)

    # MNA conditioning ties.
    b.add_resistor("R_src_gnd", "src_star", "gnd", 1e6)
    b.add_resistor("R_load_gnd", "load_star", "gnd", 1e6)

    # ---- Resolve all state indices AFTER the full graph is built ------
    iL_in_indices = tuple(
        b.pool.branch_var_id_for_inductor(L_id, b.graph)
        for L_id in L_in_branch_ids
    )
    iL_out_indices = tuple(
        b.pool.branch_var_id_for_inductor(L_id, b.graph)
        for L_id in L_out_branch_ids
    )
    module_v_src_state_indices = tuple(
        b.pool.branch_var_id_for_source(src_id, b.graph)
        for src_id in module_v_src_branch_ids
    )

    return M3cPlant(
        builder=b,
        iL_in_indices=iL_in_indices,           # type: ignore[arg-type]
        iL_out_indices=iL_out_indices,         # type: ignore[arg-type]
        module_v_src_state_indices=module_v_src_state_indices,
    )


# -----------------------------------------------------------------------------
# Open-loop SVM controller for L1
# -----------------------------------------------------------------------------


def _phase_voltages_at(
    t: float, V_peak: float, omega: float, m: float = 1.0,
) -> np.ndarray:
    """3-φ phase voltages at time ``t`` with the same cosine convention
    used by the L0 plant builder (i.e. V_a(0) = V_peak)."""
    return m * V_peak * np.array([
        np.cos(omega * t),
        np.cos(omega * t - 2.0 * pi / 3.0),
        np.cos(omega * t + 2.0 * pi / 3.0),
    ])


def _select_config_heuristic(
    short_module: tuple[int, int],
) -> ModuleConfiguration:
    """Phase 22.5 heuristic: pick the **first** of the 45 valid
    configurations containing the short module. Future commits will
    replace this with the Sec 5.5.3 cost-function selector."""
    candidates = configurations_containing_module(*short_module)
    if not candidates:
        raise RuntimeError(
            f"No valid configuration contains short {short_module}"
        )
    return candidates[0]


def _build_l1_step_state(params: M3cParams):
    """Mutable scratch state shared between switch_fn and b_extra.

    Both callbacks are invoked at each integration step but can be
    called in any order; we recompute the SVM/config/voltages at most
    once per T_s tick and cache the result.
    """
    state = {
        "last_period": -1,                 # T_s tick index of last update
        "switch_bits": [False] * 9,        # per-module ON/OFF
        "v_module": [0.0] * 9,             # per-module V [V]
    }
    return state


def make_m3c_l1_open_loop_control(
    params: M3cParams,
    plant: M3cPlant,
):
    """Build ``(switch_fn, b_extra_fn)`` for the L1 open-loop SVM
    controller.

    Both callbacks share a small dict of cached values that is
    refreshed once per switching period ``T_s``. Within a single
    T_s interval the switch states and module voltages are constant
    (no within-T_s vector mixing yet — that's a Phase 22.6 upgrade).

    The control logic per T_s:

      1. Compute the 3-φ phase voltage references at the centre of
         the period for both input (using ``f_in``) and output
         (using ``f_out``).
      2. Round each side's phases to integer multiples of V_cap_nominal
         (Fast SVM "integer plane" Sec 3.2).
      3. Pick the *short* module ``M_(argmin V_in, argmin V_out)`` —
         Sec 5.5.3 paragraph 4.
      4. Pick the first valid configuration containing the short
         (heuristic — cost-function selector arrives later).
      5. Compute the 5 active module voltages via
         :func:`solve_module_voltages`.
      6. Set the switch mask + voltage-source values accordingly.
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if plant.module_v_src_state_indices is None:
        raise ValueError("L1 plant must expose module_v_src_state_indices")

    state = _build_l1_step_state(params)
    state_size = plant.builder.pool.state_size(plant.builder.graph)
    v_cap = params.v_cap_nominal
    T_s = params.T_s
    src_indices = plant.module_v_src_state_indices

    def _refresh_if_needed(t: float) -> None:
        period_idx = int(t / T_s)
        if period_idx == state["last_period"]:
            return
        state["last_period"] = period_idx

        # Reference phase voltages at the CENTRE of this T_s window.
        t_centre = (period_idx + 0.5) * T_s
        V_in_ref = _phase_voltages_at(
            t_centre, V_peak=params.V_in_phase_peak,
            omega=params.omega_in, m=params.m_c,
        )
        V_out_ref = _phase_voltages_at(
            t_centre, V_peak=params.V_out_phase_peak,
            omega=params.omega_out, m=params.m_v,
        )

        # Quantise to integer * V_cap (Sec 3.2 integer plane).
        V_in_int = np.round(V_in_ref / v_cap).astype(int)
        V_out_int = np.round(V_out_ref / v_cap).astype(int)

        # Short module = argmin(V_in_int) × argmin(V_out_int).
        # Use the *float* refs for argmin (so ties resolve smoothly).
        short_in = int(np.argmin(V_in_ref))
        short_out = int(np.argmin(V_out_ref))
        cfg = _select_config_heuristic((short_in, short_out))

        # Module voltages (in V_cap units) via Sec 4.3 solver.
        V_xy = solve_module_voltages(
            cfg, (short_in, short_out),
            V_input=V_in_int, V_output=V_out_int,
        )

        # Populate switch + voltage caches.
        switch_bits = [False] * 9
        v_module = [0.0] * 9
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            switch_bits[idx] = True
            v_module[idx] = float(V_int) * v_cap

        state["switch_bits"] = switch_bits
        state["v_module"] = v_module

    def switch_fn(t: float):  # -> p.SwitchStateMask
        _refresh_if_needed(t)
        mask = _p.SwitchStateMask(9)
        for k, on in enumerate(state["switch_bits"]):
            mask.set(k, bool(on))
        return mask

    def b_extra_fn(t: float):
        _refresh_if_needed(t)
        out = [0.0] * state_size
        for k, src_idx in enumerate(src_indices):
            # baseline=0, so inject = -V_target makes effective V = V_target.
            out[src_idx] = -state["v_module"][k]
        return out

    return switch_fn, b_extra_fn


def run_l1_open_loop(
    plant: M3cPlant,
    params: M3cParams,
    *,
    t_end: float = 80e-3,
    dt: float | None = None,
) -> M3cRunResult:
    """Run an L1 plant with SVM open-loop control.

    Parameters
    ----------
    plant
        Built via :func:`build_l1_plant`.
    params
        Same ``M3cParams`` used to build the plant.
    t_end
        Simulation end time [s].
    dt
        Integration step. Defaults to ``T_s / 20`` so the per-T_s
        update has clean resolution; clamp to a minimum of 1 µs.
    """
    if _p is None:
        raise RuntimeError("pulsim required")

    if dt is None:
        dt = max(1e-6, params.T_s / 20.0)

    switch_fn, b_extra_fn = make_m3c_l1_open_loop_control(params, plant)

    assert plant.iL_in_indices is not None
    iLa_in, iLb_in, iLc_in = plant.iL_in_indices
    iLa, iLb, iLc = plant.iL_out_indices

    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_ia = np.zeros(n_samples)
    log_ib = np.zeros(n_samples)
    log_ic = np.zeros(n_samples)
    log_iAi = np.zeros(n_samples)
    log_iBi = np.zeros(n_samples)
    log_iCi = np.zeros(n_samples)
    counter = [0]

    def log_obs(t, x):
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_ia[i] = x[iLa]
            log_ib[i] = x[iLb]
            log_ic[i] = x[iLc]
            log_iAi[i] = x[iLa_in]
            log_iBi[i] = x[iLb_in]
            log_iCi[i] = x[iLc_in]
        counter[0] += 1

    _p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=log_obs,
        switch_fn=switch_fn,
        b_extra_fn=b_extra_fn,
        start_from_dc_op=False,           # cold start (caps assumed nominal)
    )

    n = counter[0]
    return M3cRunResult(
        t=log_t[:n],
        i_a_out=log_ia[:n],
        i_b_out=log_ib[:n],
        i_c_out=log_ic[:n],
        i_a_in=log_iAi[:n],
        i_b_in=log_iBi[:n],
        i_c_in=log_iCi[:n],
    )


# =============================================================================
# L1 cost-function controller (Phase 22.6)
# =============================================================================
#
# Upgrade to the heuristic open-loop controller of Phase 22.5: at the
# centre of each T_s window, evaluate the Sec 5.5.3 cost function
# (Eq 163) over the 45 candidate configurations containing the short
# module, and pick the one that minimises projected cap imbalance.
#
# This requires module-level capacitor voltages tracked externally
# (pulsim does not model the individual SM caps), updated at every
# T_s using the predicted module current of the chosen configuration
# (Eq 162: ΔV_xy = S_n · I_xy · T_s / C with S_n = N).
#
# The terminal currents I_input and I_output needed by the cost
# function are read from the pulsim state vector via the L_in and
# L_out inductor branch indices — hence the controller now takes a
# step_observer hook to access ``x`` at each integration step.


@dataclass
class M3cL1ControlState:
    """Live state for the L1 cost-function controller.

    Refreshed once per ``T_s`` tick inside the step observer; the
    switch_fn and b_extra_fn callbacks simply read the cached
    ``switch_bits`` and ``v_module_target`` arrays.

    The 9-element ``v_caps_module`` array tracks module-level
    capacitor sums ``V_module = Σ_k V_cap_SM_k``. Each entry starts
    at ``N · v_cap_nominal`` (balanced nominal) and integrates the
    Eq 162 update for whichever configuration was selected.
    """

    last_period: int = -1
    switch_bits: list[bool] = field(default_factory=lambda: [False] * 9)
    v_module_target: list[float] = field(
        default_factory=lambda: [0.0] * 9,
    )
    v_caps_module: list[float] = field(default_factory=list)
    # Bookkeeping for tests / diagnostics.
    n_refreshes: int = 0
    chosen_configs: list = field(default_factory=list)
    chosen_costs: list = field(default_factory=list)
    # Trajectory of v_caps_module sampled at each T_s tick (one row per
    # tick, 9 columns). Populated by all the cost-based step_observers
    # so that long-run notebooks can plot cap balance over time.
    v_caps_module_history: list = field(default_factory=list)
    # T_s tick centre time for each row in v_caps_module_history.
    refresh_t_centres: list = field(default_factory=list)


def make_m3c_l1_cost_control(
    params: M3cParams,
    plant: M3cPlant,
    *,
    state: M3cL1ControlState | None = None,
):
    """Build ``(step_observer, switch_fn, b_extra_fn)`` for the L1
    cost-function controller (Sec 5.5.3 Eq 163).

    Per-T_s update logic:

      1. Read I_input from ``x[iL_in_indices]`` and I_output from
         ``x[iL_out_indices]``.
      2. Compute V_input_ref, V_output_ref at the T_s centre from the
         cosine references; quantise to integer × V_cap.
      3. Identify the short module
         ``M_(argmin V_input, argmin V_output)``.
      4. Use :func:`select_best_connection` to pick the minimum-cost
         configuration of the 45 candidates containing the short.
      5. Compute the 5 active module voltages via
         :func:`solve_module_voltages` and the 5 module currents via
         :func:`solve_module_currents`.
      6. Set switch_bits + v_module_target accordingly.
      7. Integrate Eq 162: ``v_caps_module[k] += N · I_xy · T_s / C``
         for each of the 5 active modules.

    Parameters
    ----------
    params, plant
        Same as :func:`make_m3c_l1_open_loop_control`.
    state
        Optional initial state. Defaults to a freshly-allocated one
        with ``v_caps_module = N · v_cap_nominal`` for all 9 modules.

    Returns
    -------
    (step_observer, switch_fn, b_extra_fn)
        Three callables with the same signatures pulsim's
        ``simulate()`` expects.
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if plant.module_v_src_state_indices is None:
        raise ValueError("L1 plant must expose module_v_src_state_indices")
    if plant.iL_in_indices is None:
        raise ValueError("L1 plant must expose iL_in_indices")

    if state is None:
        state = M3cL1ControlState(
            v_caps_module=[
                params.v_cap_total_per_module for _ in range(9)
            ],
        )
    elif not state.v_caps_module:
        state.v_caps_module = [
            params.v_cap_total_per_module for _ in range(9)
        ]

    state_size = plant.builder.pool.state_size(plant.builder.graph)
    v_cap = params.v_cap_nominal
    T_s = params.T_s
    src_indices = plant.module_v_src_state_indices
    iL_in_indices = plant.iL_in_indices
    iL_out_indices = plant.iL_out_indices

    def step_observer(t, x):
        period_idx = int(t / T_s)
        if period_idx == state.last_period:
            return
        state.last_period = period_idx
        state.n_refreshes += 1

        # 1. Terminal currents from the pulsim state vector.
        I_in = np.array([float(x[i]) for i in iL_in_indices])
        I_out = np.array([float(x[i]) for i in iL_out_indices])
        # KCL conservation: enforce numerical balance (caller may have
        # tiny drift). The cost function would otherwise reject.
        diff = (I_in.sum() - I_out.sum()) / 3.0
        I_out_balanced = I_out + diff

        # 2. SVM references at the T_s centre.
        t_centre = (period_idx + 0.5) * T_s
        V_in_ref = _phase_voltages_at(
            t_centre, V_peak=params.V_in_phase_peak,
            omega=params.omega_in, m=params.m_c,
        )
        V_out_ref = _phase_voltages_at(
            t_centre, V_peak=params.V_out_phase_peak,
            omega=params.omega_out, m=params.m_v,
        )
        V_in_int = np.round(V_in_ref / v_cap).astype(int)
        V_out_int = np.round(V_out_ref / v_cap).astype(int)

        # 3. Short module = argmin × argmin.
        short_in = int(np.argmin(V_in_ref))
        short_out = int(np.argmin(V_out_ref))
        short = (short_in, short_out)

        # 4. Cost-function selector — pick the best of 45 candidates.
        best_cfg, best_cost = select_best_connection(
            short_module=short,
            V_caps=np.array(state.v_caps_module, dtype=float),
            V_input_int=V_in_int,
            V_output_int=V_out_int,
            I_input=I_in,
            I_output=I_out_balanced,
            T_s=T_s,
            C_sm=params.c_sm,
        )

        # 5. Module voltages + currents for the selected configuration.
        V_xy = solve_module_voltages(
            best_cfg, short,
            V_input=V_in_int, V_output=V_out_int,
        )
        I_xy = solve_module_currents(
            best_cfg, I_in, I_out_balanced,
        )

        # 6. Populate switch + voltage caches.
        switch_bits = [False] * 9
        v_module = [0.0] * 9
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            switch_bits[idx] = True
            v_module[idx] = float(V_int) * v_cap
        state.switch_bits = switch_bits
        state.v_module_target = v_module

        # 7. Integrate Eq 162 — cap-sum voltage update per module:
        #    ΔV_module = V_xy_signed · I_xy · T_s / C_SM     (S_n = |V_int|;
        #    sign carried by V_int for charge-direction physics).
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            state.v_caps_module[idx] += (
                float(V_int) * float(I_xy[(i, j)]) * T_s / params.c_sm
            )

        # Diagnostics.
        state.chosen_configs.append(best_cfg)
        state.chosen_costs.append(best_cost)
        state.v_caps_module_history.append(list(state.v_caps_module))
        state.refresh_t_centres.append(float(t_centre))

    def switch_fn(_t: float):
        mask = _p.SwitchStateMask(9)
        for k, on in enumerate(state.switch_bits):
            mask.set(k, bool(on))
        return mask

    def b_extra_fn(_t: float):
        out = [0.0] * state_size
        for k, src_idx in enumerate(src_indices):
            out[src_idx] = -state.v_module_target[k]
        return out

    return step_observer, switch_fn, b_extra_fn


def run_l1_cost_loop(
    plant: M3cPlant,
    params: M3cParams,
    *,
    t_end: float = 80e-3,
    dt: float | None = None,
    initial_state: M3cL1ControlState | None = None,
) -> tuple[M3cRunResult, M3cL1ControlState]:
    """Run an L1 plant with the Sec 5.5.3 cost-function controller.

    Returns
    -------
    (result, control_state)
        ``result`` is the standard :class:`M3cRunResult` with the
        terminal currents logged at every dt step. ``control_state``
        is the final :class:`M3cL1ControlState`, which exposes the
        evolved ``v_caps_module``, the chosen configurations per T_s,
        and the per-T_s cost values — useful for verifying the
        balancing behaviour.
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if dt is None:
        dt = max(1e-6, params.T_s / 20.0)

    state = initial_state or M3cL1ControlState(
        v_caps_module=[
            params.v_cap_total_per_module for _ in range(9)
        ],
    )
    control_obs, switch_fn, b_extra_fn = make_m3c_l1_cost_control(
        params, plant, state=state,
    )

    assert plant.iL_in_indices is not None
    iLa_in, iLb_in, iLc_in = plant.iL_in_indices
    iLa, iLb, iLc = plant.iL_out_indices

    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_ia = np.zeros(n_samples)
    log_ib = np.zeros(n_samples)
    log_ic = np.zeros(n_samples)
    log_iAi = np.zeros(n_samples)
    log_iBi = np.zeros(n_samples)
    log_iCi = np.zeros(n_samples)
    counter = [0]

    def combined_obs(t, x):
        control_obs(t, x)
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_ia[i] = x[iLa]
            log_ib[i] = x[iLb]
            log_ic[i] = x[iLc]
            log_iAi[i] = x[iLa_in]
            log_iBi[i] = x[iLb_in]
            log_iCi[i] = x[iLc_in]
        counter[0] += 1

    _p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=combined_obs,
        switch_fn=switch_fn,
        b_extra_fn=b_extra_fn,
        start_from_dc_op=False,
    )

    n = counter[0]
    return (
        M3cRunResult(
            t=log_t[:n],
            i_a_out=log_ia[:n],
            i_b_out=log_ib[:n],
            i_c_out=log_ic[:n],
            i_a_in=log_iAi[:n],
            i_b_in=log_iBi[:n],
            i_c_in=log_iCi[:n],
        ),
        state,
    )


# =============================================================================
# Phase 22.7 — Closed-loop dq output current control (Sec 5.6.2)
# =============================================================================
#
# The Phase 22.5/22.6 controllers were open-loop: V_out_ref came from
# a fixed cosine reference. Phase 22.7 replaces the cosine with a
# PI-driven dq controller that tracks user-specified output current
# references (i_d_ref, i_q_ref) in the synchronous (45 Hz) frame.
#
# Block diagram::
#
#   i_d_ref(t) ──┐                        ┌── V_d_pi ──┐
#                ├── e_d ── K_p + K_i/s ──┤            ├── V_d_ref
#   i_d_meas ──┘                          │ +ωL·i_q   │  (after decoupling)
#                                          └────────────┘
#   ... (analogous for q)
#   V_d_ref, V_q_ref ── inv-Park(θ) ── V_a/b/c_ref
#       ── ÷ V_cap ── round ── integer SVM refs
#       ── cost-function selector  ── Phase 22.6 inner loop
#
# Where θ = ω_out · t. Cross-axis ωL decoupling (R+sL plant in dq
# frame has coupling terms ±ωL·i_other) is enabled by default.
#
# The inner-loop machinery (cost function, voltage solver, switch_fn,
# b_extra_fn, cap voltage tracking) is reused unchanged.


# Synchronous Park transforms — amplitude-invariant convention so that
# a peak-A sinusoidal abc input gives ``|d+jq| = A`` in dq.

_SQRT3 = float(np.sqrt(3.0))
_TWO_THIRDS = 2.0 / 3.0


def abc_to_dq(a: float, b: float, c: float, theta: float) -> tuple[float, float]:
    """Amplitude-invariant Park transform ``(a, b, c) → (d, q)``.

    Uses the 2/3 scaling so that a sinusoid ``a = A·cos(θ),
    b = A·cos(θ − 2π/3), c = A·cos(θ + 2π/3)`` projects to
    ``d = A``, ``q = 0`` at any given ``θ``.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Clarke (amplitude-invariant 2/3 scaling).
    alpha = _TWO_THIRDS * (a - 0.5 * b - 0.5 * c)
    beta = _TWO_THIRDS * (_SQRT3 * 0.5 * b - _SQRT3 * 0.5 * c)
    # Park.
    d = alpha * cos_t + beta * sin_t
    q = -alpha * sin_t + beta * cos_t
    return float(d), float(q)


def dq_to_abc(d: float, q: float, theta: float) -> tuple[float, float, float]:
    """Inverse Park ``(d, q) → (a, b, c)``. Companion to
    :func:`abc_to_dq` (amplitude-invariant)."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    alpha = d * cos_t - q * sin_t
    beta = d * sin_t + q * cos_t
    a = alpha
    b = -0.5 * alpha + (_SQRT3 * 0.5) * beta
    c = -0.5 * alpha - (_SQRT3 * 0.5) * beta
    return float(a), float(b), float(c)


@dataclass
class M3cDqController:
    """Synchronous-frame PI controller for output i_d, i_q.

    The default gains target a ~500 Hz bandwidth on the M3C-default
    plant (R_load=54 Ω, L_out_eff≈117.5 mH including L_out):

        ω_bw = 2π · 500 ≈ 3142 rad/s
        K_p = ω_bw · L_out_eff ≈ 369
        K_i = ω_bw · R_load  ≈ 1.70e5

    Cross-axis ωL decoupling is enabled by default.
    """

    K_p: float = 369.0
    K_i: float = 1.70e5
    # Cross-axis coupling magnitude ω·L (used for decoupling).
    omega_L_decouple: float = 0.0
    # PI integrator state.
    integral_d: float = 0.0
    integral_q: float = 0.0
    # Anti-windup saturation per axis [V].
    v_pi_max: float = 30_000.0
    # Diagnostics (last evaluated values).
    last_i_d: float = 0.0
    last_i_q: float = 0.0
    last_e_d: float = 0.0
    last_e_q: float = 0.0
    last_V_d_ref: float = 0.0
    last_V_q_ref: float = 0.0

    def step(
        self,
        i_d_meas: float, i_q_meas: float,
        i_d_ref: float, i_q_ref: float,
        dt: float,
    ) -> tuple[float, float]:
        """One PI step: returns ``(V_d_ref, V_q_ref)`` after cross-axis
        decoupling. ``dt`` is the controller sample time (= T_s in our
        use)."""
        e_d = i_d_ref - i_d_meas
        e_q = i_q_ref - i_q_meas

        # Update integrators (trapezoidal for symmetry; simple forward
        # Euler is fine at our scale).
        self.integral_d += e_d * dt
        self.integral_q += e_q * dt

        # PI raw output before saturation.
        V_d_pi = self.K_p * e_d + self.K_i * self.integral_d
        V_q_pi = self.K_p * e_q + self.K_i * self.integral_q

        # Anti-windup: clamp + back-calculate.
        if V_d_pi > self.v_pi_max:
            self.integral_d -= (V_d_pi - self.v_pi_max) / max(self.K_i, 1e-12)
            V_d_pi = self.v_pi_max
        elif V_d_pi < -self.v_pi_max:
            self.integral_d -= (V_d_pi + self.v_pi_max) / max(self.K_i, 1e-12)
            V_d_pi = -self.v_pi_max
        if V_q_pi > self.v_pi_max:
            self.integral_q -= (V_q_pi - self.v_pi_max) / max(self.K_i, 1e-12)
            V_q_pi = self.v_pi_max
        elif V_q_pi < -self.v_pi_max:
            self.integral_q -= (V_q_pi + self.v_pi_max) / max(self.K_i, 1e-12)
            V_q_pi = -self.v_pi_max

        # Cross-axis ωL decoupling (Sec 5.6.2 — V_d_ref carries the
        # +ωL·i_q feedforward, V_q_ref carries the -ωL·i_d term so
        # the closed loop "looks like" decoupled R+sL plants).
        V_d_ref = V_d_pi - self.omega_L_decouple * i_q_meas
        V_q_ref = V_q_pi + self.omega_L_decouple * i_d_meas

        # Stash diagnostics.
        self.last_i_d = float(i_d_meas)
        self.last_i_q = float(i_q_meas)
        self.last_e_d = float(e_d)
        self.last_e_q = float(e_q)
        self.last_V_d_ref = float(V_d_ref)
        self.last_V_q_ref = float(V_q_ref)
        return V_d_ref, V_q_ref


def make_m3c_l1_dq_control(
    params: M3cParams,
    plant: M3cPlant,
    *,
    i_d_ref: float | Callable[[float], float] = 0.0,
    i_q_ref: float | Callable[[float], float] = 0.0,
    dq_controller: M3cDqController | None = None,
    initial_state: M3cL1ControlState | None = None,
):
    """Closed-loop dq output-current control + Phase 22.6 cost function.

    Parameters
    ----------
    params, plant
        Same as the open-loop variants.
    i_d_ref, i_q_ref
        Either a scalar (constant reference) or a function ``t →
        float`` for time-varying references. Defaults to 0 A on both
        axes (idle).
    dq_controller
        Optional pre-built :class:`M3cDqController`. If ``None``, a
        default controller is constructed with ``omega_L_decouple`` =
        ``ω_out · (L_out + L_load)`` for cross-axis cancellation.
    initial_state
        Optional :class:`M3cL1ControlState`. If ``None``, a fresh one
        is allocated with ``v_caps_module = N·v_cap_nominal``.

    Returns
    -------
    (step_observer, switch_fn, b_extra_fn, ctrl_state, dq_ctrl)
        Three callables plus the inner cost-loop state and the dq
        controller (so tests / notebooks can inspect their evolution).
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if plant.module_v_src_state_indices is None:
        raise ValueError("L1 plant must expose module_v_src_state_indices")
    if plant.iL_in_indices is None:
        raise ValueError("L1 plant must expose iL_in_indices")

    if dq_controller is None:
        # Default cross-coupling: ω·L through the L_out + L_load chain.
        dq_controller = M3cDqController(
            omega_L_decouple=(
                params.omega_out * (params.L_out + params.L_load)
            ),
        )

    if initial_state is None:
        initial_state = M3cL1ControlState(
            v_caps_module=[
                params.v_cap_total_per_module for _ in range(9)
            ],
        )
    state = initial_state

    state_size = plant.builder.pool.state_size(plant.builder.graph)
    v_cap = params.v_cap_nominal
    T_s = params.T_s
    src_indices = plant.module_v_src_state_indices
    iL_in_indices = plant.iL_in_indices
    iL_out_indices = plant.iL_out_indices

    # Normalise references to callables.
    i_d_ref_fn = (
        i_d_ref if callable(i_d_ref)
        else (lambda _t, _v=float(i_d_ref): _v)  # type: ignore[arg-type]
    )
    i_q_ref_fn = (
        i_q_ref if callable(i_q_ref)
        else (lambda _t, _v=float(i_q_ref): _v)  # type: ignore[arg-type]
    )

    def step_observer(t, x):
        period_idx = int(t / T_s)
        if period_idx == state.last_period:
            return
        state.last_period = period_idx
        state.n_refreshes += 1

        # Read terminal currents.
        I_in = np.array([float(x[i]) for i in iL_in_indices])
        I_out = np.array([float(x[i]) for i in iL_out_indices])
        diff = (I_in.sum() - I_out.sum()) / 3.0
        I_out_balanced = I_out + diff

        # Compute dq angle for output side.
        t_centre = (period_idx + 0.5) * T_s
        theta_o = params.omega_out * t_centre

        # Measured output currents → dq.
        i_d, i_q = abc_to_dq(
            float(I_out[0]), float(I_out[1]), float(I_out[2]),
            theta_o,
        )

        # DQ PI step.
        V_d_ref, V_q_ref = dq_controller.step(
            i_d, i_q,
            float(i_d_ref_fn(t_centre)),
            float(i_q_ref_fn(t_centre)),
            T_s,
        )

        # Inverse Park → output abc voltage refs.
        # NOTE: The Sec 4.3 voltage solver convention has an internal
        # sign flip on the output side (matches the thesis worked
        # example M_xy values), so the resulting V(out_y) actual is
        # the NEGATIVE of V_output_ref[y]. To compensate, we negate
        # the dq-derived abc refs before feeding the SVM. This makes
        # ``V_d_ref > 0 → i_d > 0`` (the natural control sign).
        Va_ref, Vb_ref, Vc_ref = dq_to_abc(V_d_ref, V_q_ref, theta_o)
        V_out_ref = -np.array([Va_ref, Vb_ref, Vc_ref])

        # Input side: still open-loop cosine (Sec 5.6.1 controller
        # will be added in a follow-up).
        V_in_ref = _phase_voltages_at(
            t_centre, V_peak=params.V_in_phase_peak,
            omega=params.omega_in, m=params.m_c,
        )

        # Quantise SVM refs.
        V_in_int = np.round(V_in_ref / v_cap).astype(int)
        V_out_int = np.round(V_out_ref / v_cap).astype(int)

        # Short module.
        short_in = int(np.argmin(V_in_ref))
        short_out = int(np.argmin(V_out_ref))
        short = (short_in, short_out)

        # Cost-function selector.
        best_cfg, best_cost = select_best_connection(
            short_module=short,
            V_caps=np.array(state.v_caps_module, dtype=float),
            V_input_int=V_in_int,
            V_output_int=V_out_int,
            I_input=I_in,
            I_output=I_out_balanced,
            T_s=T_s,
            C_sm=params.c_sm,
        )

        # Module voltages + currents.
        V_xy = solve_module_voltages(
            best_cfg, short,
            V_input=V_in_int, V_output=V_out_int,
        )
        I_xy = solve_module_currents(
            best_cfg, I_in, I_out_balanced,
        )

        # Populate caches.
        switch_bits = [False] * 9
        v_module = [0.0] * 9
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            switch_bits[idx] = True
            v_module[idx] = float(V_int) * v_cap
        state.switch_bits = switch_bits
        state.v_module_target = v_module

        # Update cap voltages (Eq 162).
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            state.v_caps_module[idx] += (
                float(V_int) * float(I_xy[(i, j)]) * T_s / params.c_sm
            )

        # Diagnostics.
        state.chosen_configs.append(best_cfg)
        state.chosen_costs.append(best_cost)
        state.v_caps_module_history.append(list(state.v_caps_module))
        state.refresh_t_centres.append(float(t_centre))

    def switch_fn(_t: float):
        mask = _p.SwitchStateMask(9)
        for k, on in enumerate(state.switch_bits):
            mask.set(k, bool(on))
        return mask

    def b_extra_fn(_t: float):
        out = [0.0] * state_size
        for k, src_idx in enumerate(src_indices):
            out[src_idx] = -state.v_module_target[k]
        return out

    return step_observer, switch_fn, b_extra_fn, state, dq_controller


# =============================================================================
# Phase 22.8 — Input-side dq current control (Sec 5.6.1)
# =============================================================================
#
# Symmetric to the output-side controller of Phase 22.7. Tracks the
# input current references ``i_d_in_ref`` and ``i_q_in_ref`` in the
# synchronous frame θ_i = ω_in · t. Together with the output dq
# loop, this gives FULL closed-loop control: the user specifies both
# input and output current dq components, and the M3C tracks both.
#
# Plant dynamics (input side, dq frame, balanced source):
#   V_source_d - V_in_node_d = L_in · (di_in_d/dt - ω_in · i_in_q)
#   V_source_q - V_in_node_q = L_in · (di_in_q/dt + ω_in · i_in_d)
# Steady-state inversion:
#   V_in_node_d = V_source_d + ω_in·L_in·i_in_q
#   V_in_node_q = V_source_q - ω_in·L_in·i_in_d
#
# The PI struct is the same M3cDqController (reused for input or
# output by choosing appropriate gains/decoupling). The integration
# with the cost-function inner loop replaces ``V_input_ref`` (which
# was an open-loop cosine of V_in_phase_peak·m_c) with the PI's
# inverse-Park output.


def make_m3c_l1_dq_full_control(
    params: M3cParams,
    plant: M3cPlant,
    *,
    i_d_in_ref: float | Callable[[float], float] = 0.0,
    i_q_in_ref: float | Callable[[float], float] = 0.0,
    i_d_out_ref: float | Callable[[float], float] = 0.0,
    i_q_out_ref: float | Callable[[float], float] = 0.0,
    dq_in_controller: M3cDqController | None = None,
    dq_out_controller: M3cDqController | None = None,
    initial_state: M3cL1ControlState | None = None,
):
    """Build ``(step_observer, switch_fn, b_extra_fn, ctrl_state,
    dq_in, dq_out)`` for the L1 plant with closed-loop dq control on
    BOTH input and output sides.

    Each side has its own PI controller + ωL decoupling. Output is
    fed to the SVM after the same sign-flip compensation as Phase
    22.7. Input is fed *without* the sign flip — the input voltage
    solver does not have the thesis output-side inversion.

    Reuses the Phase 22.6 cost-function inner loop. The cap voltage
    state is tracked via ``initial_state.v_caps_module``.
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if plant.module_v_src_state_indices is None:
        raise ValueError("L1 plant must expose module_v_src_state_indices")
    if plant.iL_in_indices is None:
        raise ValueError("L1 plant must expose iL_in_indices")

    if dq_in_controller is None:
        # Default gains for a PURE-inductor plant (no R_in):
        #   K_p = 2·ξ·ω_n·L_in, K_i = ω_n²·L_in.
        # Use ω_n = 2π·100 Hz, ξ=1 (critically damped). That's well
        # below the f_sw=2 kHz switching, fast enough that the cap-
        # outer loop can move a ~200 A i_d_in_ref correction in
        # ~10 ms.
        omega_n_in = 2.0 * pi * 100.0
        dq_in_controller = M3cDqController(
            K_p=2.0 * 1.0 * omega_n_in * params.L_in,
            K_i=omega_n_in**2 * params.L_in,
            omega_L_decouple=params.omega_in * params.L_in,
            v_pi_max=params.V_in_phase_peak * 0.5,  # half of source amplitude
        )
    if dq_out_controller is None:
        L_out_eff = params.L_out + params.L_load
        dq_out_controller = M3cDqController(
            K_p=2.0 * pi * 50.0 * L_out_eff,
            K_i=2.0 * pi * 50.0 * params.R_load,
            omega_L_decouple=params.omega_out * L_out_eff,
        )

    if initial_state is None:
        initial_state = M3cL1ControlState(
            v_caps_module=[
                params.v_cap_total_per_module for _ in range(9)
            ],
        )
    state = initial_state

    state_size = plant.builder.pool.state_size(plant.builder.graph)
    v_cap = params.v_cap_nominal
    T_s = params.T_s
    src_indices = plant.module_v_src_state_indices
    iL_in_indices = plant.iL_in_indices
    iL_out_indices = plant.iL_out_indices

    def _to_callable(ref):
        if callable(ref):
            return ref
        return lambda _t, _v=float(ref): _v  # noqa: E731

    i_d_in_fn = _to_callable(i_d_in_ref)
    i_q_in_fn = _to_callable(i_q_in_ref)
    i_d_out_fn = _to_callable(i_d_out_ref)
    i_q_out_fn = _to_callable(i_q_out_ref)

    def step_observer(t, x):
        period_idx = int(t / T_s)
        if period_idx == state.last_period:
            return
        state.last_period = period_idx
        state.n_refreshes += 1

        I_in = np.array([float(x[i]) for i in iL_in_indices])
        I_out = np.array([float(x[i]) for i in iL_out_indices])
        diff = (I_in.sum() - I_out.sum()) / 3.0
        I_out_balanced = I_out + diff

        t_centre = (period_idx + 0.5) * T_s
        theta_i = params.omega_in * t_centre
        theta_o = params.omega_out * t_centre

        # ---- INPUT side ----
        # KVL on L_in: V_source − V_in_node = L_in·di/dt.
        # To INCREASE input current we DECREASE V_in_node below
        # V_source (source "pushes harder"). So the PI's natural
        # "positive output for positive error" convention drives the
        # current the WRONG way for the input side — we invert the
        # PI output's contribution:
        #   V_d_in_ref = V_source_d − V_d_pi + ωL·i_q  (source-relative)
        #   V_q_in_ref = V_source_q − V_q_pi − ωL·i_d
        # The M3cDqController internally already mixes in its ωL·i
        # decoupling with a sign assuming output-style plant; for
        # the input we reverse that sign in addition to flipping
        # the PI bias.
        i_d_in_meas, i_q_in_meas = abc_to_dq(
            float(I_in[0]), float(I_in[1]), float(I_in[2]), theta_i,
        )
        V_d_in_pi, V_q_in_pi = dq_in_controller.step(
            i_d_in_meas, i_q_in_meas,
            float(i_d_in_fn(t_centre)),
            float(i_q_in_fn(t_centre)),
            T_s,
        )
        # NOTE: dq_in_controller.step has already added ±ωL decoupling
        # terms inside its output. Subtracting it from V_source flips
        # both the PI part and the decoupling sign — the correct
        # behavior for the input filter L_in.
        V_d_in_ref = params.V_in_phase_peak - V_d_in_pi
        V_q_in_ref = 0.0 - V_q_in_pi
        Va_in, Vb_in, Vc_in = dq_to_abc(V_d_in_ref, V_q_in_ref, theta_i)
        V_in_ref = np.array([Va_in, Vb_in, Vc_in])

        # ---- OUTPUT side ----
        i_d_out_meas, i_q_out_meas = abc_to_dq(
            float(I_out[0]), float(I_out[1]), float(I_out[2]), theta_o,
        )
        V_d_out_ref, V_q_out_ref = dq_out_controller.step(
            i_d_out_meas, i_q_out_meas,
            float(i_d_out_fn(t_centre)),
            float(i_q_out_fn(t_centre)),
            T_s,
        )
        Va_out, Vb_out, Vc_out = dq_to_abc(V_d_out_ref, V_q_out_ref, theta_o)
        V_out_ref = -np.array([Va_out, Vb_out, Vc_out])   # see Phase 22.7

        # ---- SVM quantisation ----
        V_in_int = np.round(V_in_ref / v_cap).astype(int)
        V_out_int = np.round(V_out_ref / v_cap).astype(int)
        short_in = int(np.argmin(V_in_ref))
        short_out = int(np.argmin(V_out_ref))
        short = (short_in, short_out)

        # ---- Cost-function selector + caps update (Phase 22.6) ----
        best_cfg, best_cost = select_best_connection(
            short_module=short,
            V_caps=np.array(state.v_caps_module, dtype=float),
            V_input_int=V_in_int, V_output_int=V_out_int,
            I_input=I_in, I_output=I_out_balanced,
            T_s=T_s, C_sm=params.c_sm,
        )
        V_xy = solve_module_voltages(
            best_cfg, short, V_input=V_in_int, V_output=V_out_int,
        )
        I_xy = solve_module_currents(best_cfg, I_in, I_out_balanced)

        switch_bits = [False] * 9
        v_module = [0.0] * 9
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            switch_bits[idx] = True
            v_module[idx] = float(V_int) * v_cap
        state.switch_bits = switch_bits
        state.v_module_target = v_module
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            state.v_caps_module[idx] += (
                float(V_int) * float(I_xy[(i, j)]) * T_s / params.c_sm
            )
        state.chosen_configs.append(best_cfg)
        state.chosen_costs.append(best_cost)
        state.v_caps_module_history.append(list(state.v_caps_module))
        state.refresh_t_centres.append(float(t_centre))

    def switch_fn(_t: float):
        mask = _p.SwitchStateMask(9)
        for k, on in enumerate(state.switch_bits):
            mask.set(k, bool(on))
        return mask

    def b_extra_fn(_t: float):
        out = [0.0] * state_size
        for k, src_idx in enumerate(src_indices):
            out[src_idx] = -state.v_module_target[k]
        return out

    return (
        step_observer, switch_fn, b_extra_fn,
        state, dq_in_controller, dq_out_controller,
    )


def run_l1_dq_full_closed_loop_with_cap_loop(
    plant: M3cPlant,
    params: M3cParams,
    *,
    i_d_in_ref: float | Callable[[float], float] = 0.0,
    i_q_in_ref: float | Callable[[float], float] = 0.0,
    i_d_out_ref: float | Callable[[float], float] = 0.0,
    i_q_out_ref: float | Callable[[float], float] = 0.0,
    cap_outer_loop: M3cCapOuterLoop | None = None,
    dq_in_controller: M3cDqController | None = None,
    dq_out_controller: M3cDqController | None = None,
    initial_state: M3cL1ControlState | None = None,
    t_end: float = 200e-3,
    dt: float | None = None,
):
    """Same as :func:`run_l1_dq_full_closed_loop` but with the cap-
    voltage outer loop active. Returns ``(result, ctrl_state, dq_in,
    dq_out, cap_loop)``."""
    if _p is None:
        raise RuntimeError("pulsim required")
    if dt is None:
        dt = max(1e-6, params.T_s / 20.0)

    obs, sw, bx, ctrl_state, dq_in, dq_out, cap_loop = (
        make_m3c_l1_dq_full_control_with_cap_loop(
            params, plant,
            i_d_in_ref=i_d_in_ref, i_q_in_ref=i_q_in_ref,
            i_d_out_ref=i_d_out_ref, i_q_out_ref=i_q_out_ref,
            cap_outer_loop=cap_outer_loop,
            dq_in_controller=dq_in_controller,
            dq_out_controller=dq_out_controller,
            initial_state=initial_state,
        )
    )

    assert plant.iL_in_indices is not None
    iLa_in, iLb_in, iLc_in = plant.iL_in_indices
    iLa, iLb, iLc = plant.iL_out_indices
    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_ia = np.zeros(n_samples)
    log_ib = np.zeros(n_samples)
    log_ic = np.zeros(n_samples)
    log_iAi = np.zeros(n_samples)
    log_iBi = np.zeros(n_samples)
    log_iCi = np.zeros(n_samples)
    counter = [0]

    def combined_obs(t, x):
        obs(t, x)
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_ia[i] = x[iLa]
            log_ib[i] = x[iLb]
            log_ic[i] = x[iLc]
            log_iAi[i] = x[iLa_in]
            log_iBi[i] = x[iLb_in]
            log_iCi[i] = x[iLc_in]
        counter[0] += 1

    _p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=combined_obs,
        switch_fn=sw, b_extra_fn=bx,
        start_from_dc_op=False,
    )

    n = counter[0]
    return (
        M3cRunResult(
            t=log_t[:n],
            i_a_out=log_ia[:n],
            i_b_out=log_ib[:n],
            i_c_out=log_ic[:n],
            i_a_in=log_iAi[:n],
            i_b_in=log_iBi[:n],
            i_c_in=log_iCi[:n],
        ),
        ctrl_state, dq_in, dq_out, cap_loop,
    )


def run_l1_dq_full_closed_loop(
    plant: M3cPlant,
    params: M3cParams,
    *,
    i_d_in_ref: float | Callable[[float], float] = 0.0,
    i_q_in_ref: float | Callable[[float], float] = 0.0,
    i_d_out_ref: float | Callable[[float], float] = 0.0,
    i_q_out_ref: float | Callable[[float], float] = 0.0,
    dq_in_controller: M3cDqController | None = None,
    dq_out_controller: M3cDqController | None = None,
    initial_state: M3cL1ControlState | None = None,
    t_end: float = 200e-3,
    dt: float | None = None,
):
    """L1 with closed-loop dq on BOTH sides + cost-function balance.

    Returns ``(M3cRunResult, ctrl_state, dq_in_ctrl, dq_out_ctrl)``.
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if dt is None:
        dt = max(1e-6, params.T_s / 20.0)

    obs, sw, bx, ctrl_state, dq_in, dq_out = make_m3c_l1_dq_full_control(
        params, plant,
        i_d_in_ref=i_d_in_ref, i_q_in_ref=i_q_in_ref,
        i_d_out_ref=i_d_out_ref, i_q_out_ref=i_q_out_ref,
        dq_in_controller=dq_in_controller,
        dq_out_controller=dq_out_controller,
        initial_state=initial_state,
    )

    assert plant.iL_in_indices is not None
    iLa_in, iLb_in, iLc_in = plant.iL_in_indices
    iLa, iLb, iLc = plant.iL_out_indices

    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_ia = np.zeros(n_samples)
    log_ib = np.zeros(n_samples)
    log_ic = np.zeros(n_samples)
    log_iAi = np.zeros(n_samples)
    log_iBi = np.zeros(n_samples)
    log_iCi = np.zeros(n_samples)
    counter = [0]

    def combined_obs(t, x):
        obs(t, x)
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_ia[i] = x[iLa]
            log_ib[i] = x[iLb]
            log_ic[i] = x[iLc]
            log_iAi[i] = x[iLa_in]
            log_iBi[i] = x[iLb_in]
            log_iCi[i] = x[iLc_in]
        counter[0] += 1

    _p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=combined_obs,
        switch_fn=sw, b_extra_fn=bx,
        start_from_dc_op=False,
    )

    n = counter[0]
    return (
        M3cRunResult(
            t=log_t[:n],
            i_a_out=log_ia[:n],
            i_b_out=log_ib[:n],
            i_c_out=log_ic[:n],
            i_a_in=log_iAi[:n],
            i_b_in=log_iBi[:n],
            i_c_in=log_iCi[:n],
        ),
        ctrl_state, dq_in, dq_out,
    )


# =============================================================================
# Phase 22.9 — Capacitor voltage outer loop (Sec 5.6.3)
# =============================================================================
#
# The Phase 22.6 inner cost function picks the *least-imbalancing*
# configuration each T_s, but it can't fight a net power imbalance:
# if more energy is flowing INTO the caps than OUT, all 9 caps slowly
# drift up regardless of which config is chosen.
#
# The outer loop closes this by adjusting the input active current
# reference ``i_d_in_ref`` based on the average cap voltage error:
#
#     mean_V_cap_err = mean(v_caps_module) - N · v_cap_nominal
#     Δi_d_in = K_outer · mean_V_cap_err   (positive = bring in more power)
#
# A simple proportional outer loop suffices in steady state. The
# user can pre-compose a desired i_d_in_ref schedule with a callable
# closure that adds Δi_d_in inside it — or use the convenience
# wrapper ``make_m3c_l1_dq_full_control_with_cap_loop`` below.


@dataclass
class M3cCapOuterLoop:
    """PI outer loop that adjusts ``i_d_in_ref`` to keep the mean
    module-level cap voltage at ``v_cap_total_per_module``.

    A proportional-only loop leaves a non-zero steady-state error
    because the input PI needs an *additional* ~100 A bias to deliver
    the converter's output power (and the bias only appears for
    non-zero cap error). Adding an integrator drives the steady-state
    error to zero so caps settle exactly at the target.

    Sign convention: positive ``v_caps_mean - target`` means caps are
    OVER-CHARGED → loop should DECREASE the active power drawn from
    the input (smaller i_d_in_ref). So the correction is:

        e = V_mean − V_target
        integral += e · dt
        Δi_d_in = -K_p · e − K_i · integral

    ``dt`` is the cap-loop sample time. By default we use the
    switching period ``T_s`` (the cap loop refreshes once per T_s
    tick by virtue of being called from the step_observer).

    Default gains: K_p=0.05 (~100 A correction per kV of error),
    K_i=0.05 (closes the steady-state gap over a few seconds).
    """

    K_p: float = 0.05           # A per V of cap error (proportional)
    K_i: float = 0.05           # A per (V·s) of cap error (integral)
    v_cap_target: float = 0.0   # set by factory below
    dt: float = 5e-4            # loop sample time (defaults to T_s)
    # PI integrator state.
    integral: float = 0.0
    # Anti-windup clamp on the correction.
    correction_max: float = 500.0
    # Diagnostics.
    last_error: float = 0.0
    last_correction: float = 0.0

    def apply(self, base_i_d_in_ref: float, v_caps_module) -> float:
        v_mean = float(np.mean(v_caps_module))
        err = v_mean - self.v_cap_target
        self.integral += err * self.dt
        delta = -self.K_p * err - self.K_i * self.integral
        # Anti-windup.
        if delta > self.correction_max:
            self.integral -= (delta - self.correction_max) / max(
                self.K_i, 1e-12,
            )
            delta = self.correction_max
        elif delta < -self.correction_max:
            self.integral -= (delta + self.correction_max) / max(
                self.K_i, 1e-12,
            )
            delta = -self.correction_max
        self.last_error = err
        self.last_correction = delta
        return float(base_i_d_in_ref + delta)


def make_m3c_l1_dq_full_control_with_cap_loop(
    params: M3cParams,
    plant: M3cPlant,
    *,
    i_d_in_ref: float | Callable[[float], float] = 0.0,
    i_q_in_ref: float | Callable[[float], float] = 0.0,
    i_d_out_ref: float | Callable[[float], float] = 0.0,
    i_q_out_ref: float | Callable[[float], float] = 0.0,
    cap_outer_loop: M3cCapOuterLoop | None = None,
    dq_in_controller: M3cDqController | None = None,
    dq_out_controller: M3cDqController | None = None,
    initial_state: M3cL1ControlState | None = None,
):
    """Like :func:`make_m3c_l1_dq_full_control` but with the Sec 5.6.3
    outer cap-voltage loop modulating ``i_d_in_ref``.

    Returns ``(step_observer, switch_fn, b_extra_fn, ctrl_state,
    dq_in, dq_out, cap_loop)``.
    """
    if cap_outer_loop is None:
        cap_outer_loop = M3cCapOuterLoop(
            K_p=0.05,
            K_i=0.05,
            dt=params.T_s,
            v_cap_target=params.v_cap_total_per_module,
        )

    # Normalise base refs to callables (we wrap i_d_in_ref).
    def _to_callable(ref):
        return ref if callable(ref) else (
            lambda _t, _v=float(ref): _v  # noqa: E731
        )
    base_i_d_in_fn = _to_callable(i_d_in_ref)

    # Initial state shared with the controller closures below.
    if initial_state is None:
        initial_state = M3cL1ControlState(
            v_caps_module=[
                params.v_cap_total_per_module for _ in range(9)
            ],
        )

    def i_d_in_with_cap_correction(t):
        return cap_outer_loop.apply(
            float(base_i_d_in_fn(t)),
            initial_state.v_caps_module,
        )

    obs, sw, bx, state, dq_in, dq_out = make_m3c_l1_dq_full_control(
        params, plant,
        i_d_in_ref=i_d_in_with_cap_correction,
        i_q_in_ref=i_q_in_ref,
        i_d_out_ref=i_d_out_ref,
        i_q_out_ref=i_q_out_ref,
        dq_in_controller=dq_in_controller,
        dq_out_controller=dq_out_controller,
        initial_state=initial_state,
    )
    return obs, sw, bx, state, dq_in, dq_out, cap_outer_loop


def run_l1_dq_closed_loop(
    plant: M3cPlant,
    params: M3cParams,
    *,
    i_d_ref: float | Callable[[float], float] = 0.0,
    i_q_ref: float | Callable[[float], float] = 0.0,
    dq_controller: M3cDqController | None = None,
    initial_state: M3cL1ControlState | None = None,
    t_end: float = 200e-3,
    dt: float | None = None,
) -> tuple[M3cRunResult, M3cL1ControlState, M3cDqController]:
    """Run the L1 plant with closed-loop dq output current control.

    Returns ``(result, ctrl_state, dq_ctrl)`` — the standard
    measurement bundle plus both control state objects.
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if dt is None:
        dt = max(1e-6, params.T_s / 20.0)

    control_obs, switch_fn, b_extra_fn, ctrl_state, dq_ctrl = (
        make_m3c_l1_dq_control(
            params, plant,
            i_d_ref=i_d_ref, i_q_ref=i_q_ref,
            dq_controller=dq_controller,
            initial_state=initial_state,
        )
    )

    assert plant.iL_in_indices is not None
    iLa_in, iLb_in, iLc_in = plant.iL_in_indices
    iLa, iLb, iLc = plant.iL_out_indices

    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_ia = np.zeros(n_samples)
    log_ib = np.zeros(n_samples)
    log_ic = np.zeros(n_samples)
    log_iAi = np.zeros(n_samples)
    log_iBi = np.zeros(n_samples)
    log_iCi = np.zeros(n_samples)
    counter = [0]

    def combined_obs(t, x):
        control_obs(t, x)
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_ia[i] = x[iLa]
            log_ib[i] = x[iLb]
            log_ic[i] = x[iLc]
            log_iAi[i] = x[iLa_in]
            log_iBi[i] = x[iLb_in]
            log_iCi[i] = x[iLc_in]
        counter[0] += 1

    _p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=combined_obs,
        switch_fn=switch_fn,
        b_extra_fn=b_extra_fn,
        start_from_dc_op=False,
    )

    n = counter[0]
    return (
        M3cRunResult(
            t=log_t[:n],
            i_a_out=log_ia[:n],
            i_b_out=log_ib[:n],
            i_c_out=log_ic[:n],
            i_a_in=log_iAi[:n],
            i_b_in=log_iBi[:n],
            i_c_in=log_iCi[:n],
        ),
        ctrl_state,
        dq_ctrl,
    )


# =============================================================================
# Phase 22.13 — Dead-Beat Predictive Control (DBPC)
# =============================================================================
#
# Replaces the PI/dq cascade of Phases 22.7-22.9 with one-step discrete
# predictive control. Computes the converter voltage that drives the
# current to the reference in EXACTLY one T_s sample:
#
#   Input  (L_in plant, no R):
#     V_source - V_in_node = L_in · di/dt
#     dead-beat (di/dt = (i_ref - i)/T_s):
#     V_in_node = V_source - (L_in/T_s)·(i_in_ref - i_in)
#
#   Output (R + L_out_eff plant):
#     V_out_node = R·i_out + L_out_eff·di/dt
#     dead-beat:
#     V_out_node = R·i_out + (L_out_eff/T_s)·(i_out_ref - i_out)
#
# Why this is *much* better than the PI/dq cascade:
#
#   * Zero gains to tune — only L_in, L_out_eff, R_load, T_s (all known
#     from M3cParams).
#   * Works at ANY output frequency, including f_out = 0 (DC). The dq
#     synchronous frame is bypassed — references and dynamics live in
#     abc directly.
#   * Latency = 1 T_s (vs ~1/ω_c with PI). Step response is essentially
#     a single-T_s ramp to the achievable level.
#   * Robust to f_in ≈ f_out: there's no beat-frequency PI integrator
#     that can become "trapped" on the cap-voltage oscillation.
#   * Naturally couples to the Phase 22.6 cost-function inner loop for
#     cap balancing — DBPC just sets the V_in_int / V_out_int targets,
#     same as the SVM did, then the cost function picks the cfg.


def make_sinusoidal_abc_ref(
    amplitude: float, frequency: float, phase: float = 0.0,
):
    """Build a callable ``t → (a, b, c)`` for a balanced 3-φ sinusoidal
    reference: ``a(t) = A·cos(ω·t + φ)``, b and c at ∓2π/3.

    Convenience helper for ``make_m3c_l1_dbpc_control``'s
    ``i_out_ref_fn`` / ``i_in_ref_fn`` arguments.
    """
    omega = 2.0 * pi * frequency

    def ref(t: float) -> np.ndarray:
        th = omega * t + phase
        return amplitude * np.array([
            np.cos(th),
            np.cos(th - 2.0 * pi / 3.0),
            np.cos(th + 2.0 * pi / 3.0),
        ])
    return ref


def make_freq_ramp_abc_ref(
    amplitude: float,
    f_start: float,
    f_end: float,
    t_ramp_start: float,
    t_ramp_end: float,
):
    """3-φ balanced reference whose FREQUENCY ramps linearly from
    ``f_start`` (Hz) to ``f_end`` (Hz) between ``t_ramp_start`` and
    ``t_ramp_end`` (seconds). Outside the ramp window the frequency
    is held constant. Amplitude is constant throughout.

    Builds the phase angle by integrating ω(t) analytically (closed
    form for a piecewise-linear frequency profile) so the abc waveform
    is smooth across the ramp boundaries — no spurious phase jumps.

    Use for motor V/f acceleration profiles: ``amplitude`` sets the
    torque-producing current peak, ``f_start → f_end`` defines the
    speed sweep, ``t_ramp_start, t_ramp_end`` set when and how long
    the acceleration takes.
    """
    omega_start = 2.0 * pi * f_start
    omega_end = 2.0 * pi * f_end
    t_ramp = max(t_ramp_end - t_ramp_start, 1e-12)
    domega_dt = (omega_end - omega_start) / t_ramp

    def theta(t: float) -> float:
        if t < t_ramp_start:
            return omega_start * t
        if t < t_ramp_end:
            tau = t - t_ramp_start
            return (
                omega_start * t_ramp_start
                + omega_start * tau
                + 0.5 * domega_dt * tau * tau
            )
        # After ramp.
        theta_at_end = (
            omega_start * t_ramp_start
            + omega_start * t_ramp
            + 0.5 * domega_dt * t_ramp * t_ramp
        )
        return theta_at_end + omega_end * (t - t_ramp_end)

    def freq_at(t: float) -> float:
        if t < t_ramp_start:
            return f_start
        if t < t_ramp_end:
            return f_start + (f_end - f_start) * (
                (t - t_ramp_start) / t_ramp
            )
        return f_end

    def ref(t: float) -> np.ndarray:
        th = theta(t)
        return amplitude * np.array([
            np.cos(th),
            np.cos(th - 2.0 * pi / 3.0),
            np.cos(th + 2.0 * pi / 3.0),
        ])

    # Expose the angle and frequency functions for plotting.
    ref.theta = theta              # type: ignore[attr-defined]
    ref.frequency = freq_at        # type: ignore[attr-defined]
    return ref


def make_dc_abc_ref(
    i_a: float, i_b: float | None = None, i_c: float | None = None,
):
    """Build a callable ``t → (a, b, c)`` for a CONSTANT DC reference.

    If only ``i_a`` is given, b and c are set to ``-i_a/2`` each
    (which sums to zero — necessary for a floating-star Y-load).
    """
    if i_b is None and i_c is None:
        i_b = -i_a / 2.0
        i_c = -i_a / 2.0
    elif i_b is None or i_c is None:
        raise ValueError("specify either i_a only OR all three of a/b/c")
    elif abs(i_a + i_b + i_c) > 1e-9:
        raise ValueError(
            f"DC reference must sum to 0: {i_a} + {i_b} + {i_c} != 0"
        )
    arr = np.array([float(i_a), float(i_b), float(i_c)])

    def ref(_t: float) -> np.ndarray:
        return arr
    return ref


def make_m3c_l1_dbpc_control(
    params: M3cParams,
    plant: M3cPlant,
    *,
    i_out_ref_fn,
    i_d_in_ref: float | Callable[[float], float] = 0.0,
    cap_outer_loop: M3cCapOuterLoop | None = None,
    initial_state: M3cL1ControlState | None = None,
):
    """Dead-beat predictive controller for the M3C.

    Parameters
    ----------
    params, plant
        Same as the PI/dq variants.
    i_out_ref_fn
        Callable ``t → (i_a_ref, i_b_ref, i_c_ref)`` in abc. Use
        :func:`make_sinusoidal_abc_ref` for sinusoidal output at any
        frequency (including 0 — see :func:`make_dc_abc_ref`).
    i_d_in_ref
        Base amplitude of the unity-PF input current (scalar or
        callable). The cap-outer loop ADDS to this to maintain the
        capacitor voltage at nominal. Set to 0 (default) to let the
        cap loop manage P_in fully.
    cap_outer_loop
        Optional PI outer loop for cap balancing. Defaults to the
        Phase 22.9 PI (K_p=0.05, K_i=0.05).
    initial_state
        Optional initial M3cL1ControlState.

    Returns
    -------
    (step_observer, switch_fn, b_extra_fn, ctrl_state, cap_loop)
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if plant.module_v_src_state_indices is None:
        raise ValueError(
            "L1 plant must expose module_v_src_state_indices"
        )
    if plant.iL_in_indices is None:
        raise ValueError("L1 plant must expose iL_in_indices")

    if cap_outer_loop is None:
        cap_outer_loop = M3cCapOuterLoop(
            K_p=0.05, K_i=0.05, dt=params.T_s,
            v_cap_target=params.v_cap_total_per_module,
        )

    if initial_state is None:
        initial_state = M3cL1ControlState(
            v_caps_module=[
                params.v_cap_total_per_module for _ in range(9)
            ],
        )
    state = initial_state

    state_size = plant.builder.pool.state_size(plant.builder.graph)
    v_cap = params.v_cap_nominal
    T_s = params.T_s
    L_in = params.L_in
    # IMPORTANT: the L1 plant (build_l1_plant) only models the
    # *output filter* inductor L_out in series with R_load — L_load
    # is purely a *target* parameter used by the L0 baseline. For
    # the L1 dead-beat to match the actual plant dynamics, use
    # ``L_out_eff = params.L_out`` only. (If your plant variant also
    # models L_load as a separate inductor, sum them here.)
    L_out_eff = params.L_out
    R_load = params.R_load
    N_sm = params.n_sm_per_module
    src_indices = plant.module_v_src_state_indices
    iL_in_indices = plant.iL_in_indices
    iL_out_indices = plant.iL_out_indices

    if callable(i_d_in_ref):
        i_d_in_base_fn = i_d_in_ref
    else:
        i_d_in_base_fn = lambda _t, _v=float(i_d_in_ref): _v  # noqa: E731

    def step_observer(t, x):
        period_idx = int(t / T_s)
        if period_idx == state.last_period:
            return
        state.last_period = period_idx
        state.n_refreshes += 1

        # 1) Read currents.
        I_in = np.array([float(x[i]) for i in iL_in_indices])
        I_out = np.array([float(x[i]) for i in iL_out_indices])
        diff = (I_in.sum() - I_out.sum()) / 3.0
        I_out_balanced = I_out + diff

        # 2) T_s centre for reference evaluation.
        t_centre = (period_idx + 0.5) * T_s

        # 3) Input reference (unity PF + cap-loop correction).
        i_d_in_base_val = float(i_d_in_base_fn(t_centre))
        i_d_in_total = cap_outer_loop.apply(
            i_d_in_base_val, state.v_caps_module,
        )
        omega_in_t = params.omega_in * t_centre
        # i_in_ref at t_centre + T_s is computed below alongside the
        # output ref to keep both predict-ahead in one place.

        # 4) Output reference at the END of the next T_s window — the
        #    NEXT-step prediction. This is the critical detail for
        #    sinusoidal tracking: using i_ref(k) instead of i_ref(k+1)
        #    in the dead-beat formula misses the L·(di_ref/dt) feed-
        #    forward term, which produces a steady-state phase lag
        #    that the proportional feedback cannot correct. Using
        #    i_ref(k+1) is equivalent to including ``+L·di_ref/dt``
        #    inside the formula.
        i_out_ref_next = np.asarray(
            i_out_ref_fn(t_centre + T_s), dtype=float,
        )
        if i_out_ref_next.shape != (3,):
            raise ValueError(
                f"i_out_ref_fn must return shape (3,); got "
                f"{i_out_ref_next.shape}"
            )

        # 5) DEAD-BEAT — compute the V_node values that drive
        #    i(k+1) = i_ref(k+1) in one T_s. To prevent the saturation
        #    instability of bare one-step dead-beat, we clip the
        #    requested step in i to the feasible range first.
        V_source = params.V_in_phase_peak * np.array([
            np.cos(omega_in_t),
            np.cos(omega_in_t - 2.0 * pi / 3.0),
            np.cos(omega_in_t + 2.0 * pi / 3.0),
        ])
        # Same predict-ahead trick for input: i_in_ref at t_centre+T_s.
        omega_in_t_next = params.omega_in * (t_centre + T_s)
        i_in_ref_next = i_d_in_total * np.array([
            np.cos(omega_in_t_next),
            np.cos(omega_in_t_next - 2.0 * pi / 3.0),
            np.cos(omega_in_t_next + 2.0 * pi / 3.0),
        ])
        delta_i_in_max = N_sm * v_cap * T_s / L_in
        delta_i_out_max = N_sm * v_cap * T_s / L_out_eff
        d_i_in_clip = np.clip(
            i_in_ref_next - I_in, -delta_i_in_max, +delta_i_in_max,
        )
        d_i_out_clip = np.clip(
            i_out_ref_next - I_out_balanced,
            -delta_i_out_max, +delta_i_out_max,
        )
        V_in_node_desired = V_source - (L_in / T_s) * d_i_in_clip
        V_out_node_desired = (
            R_load * I_out_balanced
            + (L_out_eff / T_s) * d_i_out_clip
        )

        # 6) SVM quantisation. Output side carries the Sec 4.3 sign
        #    flip (same convention as the dq controller). Saturate
        #    to ±N_sm so we stay within achievable module voltages.
        V_in_int = np.clip(
            np.round(V_in_node_desired / v_cap).astype(int),
            -N_sm, +N_sm,
        )
        V_out_int = np.clip(
            np.round(-V_out_node_desired / v_cap).astype(int),
            -N_sm, +N_sm,
        )

        # 7) Short = argmin × argmin (Sec 5.5.3 paragraph 4).
        short_in = int(np.argmin(V_in_int))
        short_out = int(np.argmin(V_out_int))
        short = (short_in, short_out)

        # 8) Cost-function: pick best of 45 configs (Phase 22.6).
        best_cfg, best_cost = select_best_connection(
            short_module=short,
            V_caps=np.array(state.v_caps_module, dtype=float),
            V_input_int=V_in_int, V_output_int=V_out_int,
            I_input=I_in, I_output=I_out_balanced,
            T_s=T_s, C_sm=params.c_sm,
        )
        V_xy = solve_module_voltages(
            best_cfg, short,
            V_input=V_in_int, V_output=V_out_int,
        )
        I_xy = solve_module_currents(
            best_cfg, I_in, I_out_balanced,
        )

        # 9) Populate caches + integrate cap voltages.
        switch_bits = [False] * 9
        v_module = [0.0] * 9
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            switch_bits[idx] = True
            v_module[idx] = float(V_int) * v_cap
        state.switch_bits = switch_bits
        state.v_module_target = v_module
        for (i, j), V_int in V_xy.items():
            idx = _module_index(i, j)
            state.v_caps_module[idx] += (
                float(V_int) * float(I_xy[(i, j)]) * T_s / params.c_sm
            )
        state.chosen_configs.append(best_cfg)
        state.chosen_costs.append(best_cost)
        state.v_caps_module_history.append(list(state.v_caps_module))
        state.refresh_t_centres.append(float(t_centre))

    def switch_fn(_t: float):
        mask = _p.SwitchStateMask(9)
        for k, on in enumerate(state.switch_bits):
            mask.set(k, bool(on))
        return mask

    def b_extra_fn(_t: float):
        out = [0.0] * state_size
        for k, src_idx in enumerate(src_indices):
            out[src_idx] = -state.v_module_target[k]
        return out

    return step_observer, switch_fn, b_extra_fn, state, cap_outer_loop


def run_l1_dbpc(
    plant: M3cPlant,
    params: M3cParams,
    *,
    i_out_ref_fn,
    i_d_in_ref: float | Callable[[float], float] = 0.0,
    cap_outer_loop: M3cCapOuterLoop | None = None,
    initial_state: M3cL1ControlState | None = None,
    t_end: float = 200e-3,
    dt: float | None = None,
):
    """Run the L1 plant with the dead-beat predictive controller.

    Returns ``(result, ctrl_state, cap_loop)``.
    """
    if _p is None:
        raise RuntimeError("pulsim required")
    if dt is None:
        dt = max(1e-6, params.T_s / 20.0)

    obs, sw, bx, ctrl_state, cap_loop = make_m3c_l1_dbpc_control(
        params, plant,
        i_out_ref_fn=i_out_ref_fn,
        i_d_in_ref=i_d_in_ref,
        cap_outer_loop=cap_outer_loop,
        initial_state=initial_state,
    )

    assert plant.iL_in_indices is not None
    iLa_in, iLb_in, iLc_in = plant.iL_in_indices
    iLa, iLb, iLc = plant.iL_out_indices

    n_samples = int(round(t_end / dt)) + 1
    log_t = np.zeros(n_samples)
    log_ia = np.zeros(n_samples)
    log_ib = np.zeros(n_samples)
    log_ic = np.zeros(n_samples)
    log_iAi = np.zeros(n_samples)
    log_iBi = np.zeros(n_samples)
    log_iCi = np.zeros(n_samples)
    counter = [0]

    def combined_obs(t, x):
        obs(t, x)
        i = counter[0]
        if i < n_samples:
            log_t[i] = t
            log_ia[i] = x[iLa]
            log_ib[i] = x[iLb]
            log_ic[i] = x[iLc]
            log_iAi[i] = x[iLa_in]
            log_iBi[i] = x[iLb_in]
            log_iCi[i] = x[iLc_in]
        counter[0] += 1

    _p.simulate(
        plant.builder, t_end=t_end, dt=dt,
        step_observer=combined_obs,
        switch_fn=sw, b_extra_fn=bx,
        start_from_dc_op=False,
    )

    n = counter[0]
    return (
        M3cRunResult(
            t=log_t[:n],
            i_a_out=log_ia[:n],
            i_b_out=log_ib[:n],
            i_c_out=log_ic[:n],
            i_a_in=log_iAi[:n],
            i_b_in=log_iBi[:n],
            i_c_in=log_iCi[:n],
        ),
        ctrl_state, cap_loop,
    )
