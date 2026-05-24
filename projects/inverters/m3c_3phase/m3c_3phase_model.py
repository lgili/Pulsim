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
    I_input: np.ndarray | list | tuple,
    I_output: np.ndarray | list | tuple,
    T_s: float,
    C_sm: float,
    n_sm_per_module: int,
) -> float:
    """Compute the Sec 5.5.3 cost function (Eq 163) for a candidate
    connection ``cfg``.

    The cost approximates the squared sum of "post-T_s" capacitor
    voltage deviations from their mean: each module's deviation is
    its current imbalance ε plus the projected change ΔV during T_s.

    Parameters
    ----------
    cfg
        Candidate configuration (5 active modules).
    V_caps
        Length-9 array of measured module capacitor voltages [V],
        flattened as ``V_caps[i*3 + j] = V_(input_i, output_j)``.
    I_input, I_output
        Terminal currents [A] (same convention as
        :func:`solve_module_currents`).
    T_s
        Switching period [s].
    C_sm
        Submodule capacitance [F].
    n_sm_per_module
        Number of submodules per module (== ``params.n_sm_per_module``).
        Determines :math:`S_n` in Eq 162 — for simplicity we assume
        all N SMs of each active module are actively contributing
        (the upper-bound worst case for ΔV).

    Returns
    -------
    float
        :math:`\\mathcal{C} = \\sum_{xy} (\\epsilon_{xy} + \\Delta V_{xy})^2`
    """
    V_caps = np.asarray(V_caps, dtype=float).reshape(-1)
    if V_caps.shape != (9,):
        raise ValueError(
            f"V_caps must be length 9; got shape {V_caps.shape}"
        )

    V_mean = float(V_caps.mean())
    eps = V_caps - V_mean   # Eq 161 — shape (9,)

    # Predict module currents for active modules. Inactive ones
    # contribute I_xy = 0 → ΔV_xy = 0.
    currents = solve_module_currents(cfg, I_input, I_output)
    delta_v = np.zeros(9)
    for (i, j), I_xy in currents.items():
        # Eq 162: ΔV = S_n · I · T_s / C. S_n = n_sm_per_module for
        # active modules in this simplified model.
        delta_v[i * 3 + j] = (
            n_sm_per_module * I_xy * T_s / C_sm
        )

    return float(np.sum((eps + delta_v) ** 2))


def select_best_connection(
    short_module: tuple[int, int],
    V_caps: np.ndarray,
    I_input: np.ndarray | list | tuple,
    I_output: np.ndarray | list | tuple,
    T_s: float,
    C_sm: float,
    n_sm_per_module: int,
) -> tuple[ModuleConfiguration, float]:
    """Sec 5.5.3 selector: among the 45 configurations that contain
    ``short_module``, pick the one with the lowest cost.

    Parameters
    ----------
    short_module
        The fixed connection (typically min-input × min-output phase
        per Sec 5.5.3) that all 45 candidates must include.
    V_caps, I_input, I_output, T_s, C_sm, n_sm_per_module
        See :func:`connection_cost`.

    Returns
    -------
    (best_cfg, best_cost)
        The configuration that minimizes the cost function and its
        cost value.
    """
    candidates = configurations_containing_module(*short_module)
    if not candidates:
        raise ValueError(
            f"No valid configuration contains module {short_module}"
        )
    best_cfg = candidates[0]
    best_cost = float("inf")
    for cfg in candidates:
        cost = connection_cost(
            cfg, V_caps, I_input, I_output, T_s, C_sm, n_sm_per_module
        )
        if cost < best_cost:
            best_cost = cost
            best_cfg = cfg
    return best_cfg, best_cost
