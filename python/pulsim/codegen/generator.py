"""Codegen generator: linearize → discretize → emit C99.

The pipeline:

1. **Linearize** the circuit at the DC operating point via the C++
   `Simulator::linearize_around` (from `add-frequency-domain-analysis`).
   That gives us continuous-time matrices `(E, A, B, C, D)` in
   descriptor form `E·dx/dt = A·x + B·u`.

2. **Reduce** `E` (often singular due to algebraic constraints in MNA)
   to a regular state-space form by partitioning the equations into
   differential and algebraic. For Pulsim's typical RLC + V-source
   circuits, the algebraic part is just the V-source branch
   constraint; we eliminate it by Gauss elimination over the
   E-singular rows.

3. **Discretize** the reduced `(A, B)` continuous-time pair into
   `(A_d, B_d)` discrete-time matrices via the matrix exponential
   `A_d = exp(A·dt)`, `B_d = (A_d − I) · A⁻¹ · B`.

4. **Stability check**: every eigenvalue of the discrete `A_d` must
   sit inside the unit circle. Equivalently, `|λ_max(A·dt)| ≤ ln(2)`
   for a typical safety budget.

5. **Emit C99**: pre-compute the discrete matrices as `static const
   float[N][N]` and write a `pulsim_model_step(state, u, y)` that
   does one matrix-vector product per call.

The output is a self-contained C99 module with no malloc, no globals,
and no platform dependencies — drops into any embedded build that has
a `float` and a basic libm.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


__all__ = [
    "CodegenSummary",
    "generate",
    "discretize_state_space",
    "stability_radius",
]


@dataclass
class CodegenSummary:
    """Summary of a code-generation run. Captures the design-time
    state-space matrices, the stability margin, and a coarse ROM/RAM
    estimate from the generated source size."""
    out_dir: Path
    state_size: int
    input_size: int
    output_size: int
    A_d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    B_d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    C: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    D: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    stability_radius: float = 0.0
    rom_estimate_bytes: int = 0
    ram_estimate_bytes: int = 0
    target: str = "c99"
    files_written: list[str] = field(default_factory=list)


def stability_radius(A_d: np.ndarray) -> float:
    """Spectral radius of the discrete-time matrix. Stable iff < 1."""
    if A_d.size == 0:
        return 0.0
    eigs = np.linalg.eigvals(A_d)
    return float(np.max(np.abs(eigs)))


def discretize_state_space(
    A: np.ndarray,
    B: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize the continuous-time pair (A, B) at fixed step dt.

    Uses `scipy.linalg.expm` for `A_d = exp(A·dt)`. Computes `B_d` via
    the augmented-matrix trick (van Loan):

        ⎡A  B⎤            ⎡A_d  B_d⎤
        ⎢    ⎥ ·dt → exp ⎢        ⎥
        ⎣0  0⎦            ⎣ 0   I  ⎦

    so we get `B_d` even when `A` is singular (a common case for
    integrator-only systems).
    """
    from scipy.linalg import expm

    n = A.shape[0]
    m = B.shape[1] if B.ndim == 2 else 1
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, m))

    B_2d = B.reshape(n, m) if B.ndim == 1 else B
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = A
    aug[:n, n:] = B_2d
    aug_d = expm(aug * dt)
    A_d = aug_d[:n, :n]
    B_d = aug_d[:n, n:]
    return A_d, B_d


def _reduce_descriptor_form(
    E: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a possibly-singular descriptor form `E·dx/dt = A·x + B·u`
    to a regular state-space `dx_red/dt = A_red · x_red + B_red · u`.

    Strategy: identify the rows of `E` that are zero (algebraic
    constraints), use them to solve for the dependent state variables,
    and substitute back. For Pulsim's MNA-trapezoidal output, the
    algebraic rows correspond to V-source branch equations; this is a
    well-defined Gauss elimination.

    If `E` is non-singular (no algebraic constraints), we just return
    `A_eff = E⁻¹ · A`, `B_eff = E⁻¹ · B`.
    """
    n = E.shape[0]
    # Identify all-zero rows of E (within numerical tolerance).
    row_norms = np.linalg.norm(E, axis=1)
    tol = max(1e-12, row_norms.max() * 1e-12) if row_norms.size > 0 else 1e-12
    alg_rows = np.where(row_norms < tol)[0]
    diff_rows = np.where(row_norms >= tol)[0]

    if alg_rows.size == 0:
        # E full-rank → invert directly. C / D are unchanged since the
        # state vector is unchanged.
        E_inv = np.linalg.inv(E)
        return E_inv @ A, E_inv @ B, C, D, np.eye(n)

    # When the state is reduced to `diff_rows`, C must also drop the
    # columns corresponding to the eliminated algebraic state. Outputs
    # remain valid for the differential state only — the algebraic
    # outputs aren't reconstructed by the discrete-time model.

    # Partition the state into (x_d, x_a) where x_a are eliminated by
    # the algebraic constraints. The constraints have form
    #   0 = A[alg, :] · x + B[alg, :] · u
    # Solve for x[alg_rows]:
    #   A_aa · x_a = -A_da · x_d - B_a · u
    A_da = A[np.ix_(alg_rows, diff_rows)]   # algebraic rows × diff cols
    A_aa = A[np.ix_(alg_rows, alg_rows)]    # algebraic block
    B_a  = B[alg_rows, :]
    if A_aa.size == 0 or np.linalg.matrix_rank(A_aa) < A_aa.shape[0]:
        # Non-invertible algebraic block — fall back to the full
        # pseudo-inverse approach below.
        A_eff = np.linalg.pinv(E) @ A
        B_eff = np.linalg.pinv(E) @ B
        return A_eff, B_eff, C, D, np.eye(n)

    A_aa_inv = np.linalg.inv(A_aa)
    # x_a = -A_aa_inv · (A[alg, diff] · x_d + B_a · u)
    # Substitute into the differential equations:
    E_dd = E[np.ix_(diff_rows, diff_rows)]
    A_dd = A[np.ix_(diff_rows, diff_rows)]
    A_d_a = A[np.ix_(diff_rows, alg_rows)]
    B_d  = B[diff_rows, :]
    A_red = A_dd - A_d_a @ A_aa_inv @ A_da
    B_red = B_d  - A_d_a @ A_aa_inv @ B_a
    E_red = E_dd

    # If E_red is singular too (rare), pinv as fallback.
    try:
        E_red_inv = np.linalg.inv(E_red)
    except np.linalg.LinAlgError:
        E_red_inv = np.linalg.pinv(E_red)
    A_eff = E_red_inv @ A_red
    B_eff = E_red_inv @ B_red

    # Build a permutation matrix that maps the reduced state back to
    # the full state ordering, so the C-side can populate the original
    # state vector layout from the reduced one.
    P = np.eye(n)[diff_rows, :]
    # Drop the C columns that referenced eliminated states.
    C_red_cols = C[:, diff_rows]
    return A_eff, B_eff, C_red_cols, D, P


def _to_dense(matrix) -> np.ndarray:
    """Convert a Pulsim sparse-matrix binding object to dense numpy.
    The C++ side returns scipy.sparse.csc_matrix-like objects via
    pybind11/eigen.h; we coerce to dense to feed numpy / scipy.linalg.
    """
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    if hasattr(matrix, "todense"):
        return np.asarray(matrix.todense(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _emit_c99(
    out_dir: Path,
    A_d: np.ndarray,
    B_d: np.ndarray,
    C_mat: np.ndarray,
    D_mat: np.ndarray,
    dt: float,
    state_size: int,
    input_size: int,
    output_size: int,
) -> list[str]:
    """Emit `model.h`, `model.c`, and `model_test.c` (PIL harness)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _fmt_matrix(name: str, M: np.ndarray) -> str:
        rows, cols = M.shape if M.ndim == 2 else (M.shape[0], 1)
        if rows == 0 or cols == 0:
            return f"static const float {name}[1][1] = {{{{0.0f}}}};\n"
        lines = [f"static const float {name}[{rows}][{cols}] = {{"]
        for i in range(rows):
            row = ", ".join(f"{M[i, j]:.10e}f" for j in range(cols))
            lines.append(f"    {{{row}}},")
        lines.append("};")
        return "\n".join(lines) + "\n"

    header = textwrap.dedent(f"""\
        /* Generated by pulsim.codegen — do not edit by hand. */
        /* Discrete-time state-space at dt = {dt:.6e} s. */
        #ifndef PULSIM_GEN_MODEL_H
        #define PULSIM_GEN_MODEL_H

        #include <stddef.h>

        #ifdef __cplusplus
        extern "C" {{
        #endif

        #define PULSIM_STATE_SIZE  {state_size}
        #define PULSIM_INPUT_SIZE  {input_size}
        #define PULSIM_OUTPUT_SIZE {output_size}

        typedef struct PulsimModel {{
            float x[PULSIM_STATE_SIZE];
        }} PulsimModel;

        void pulsim_model_init(PulsimModel* m);
        void pulsim_model_step(PulsimModel* m,
                               const float* u,
                               float* y);

        #ifdef __cplusplus
        }}
        #endif

        #endif
        """)

    body_lines = []
    body_lines.append("/* Generated by pulsim.codegen — do not edit. */")
    body_lines.append('#include "model.h"')
    body_lines.append("")
    body_lines.append(_fmt_matrix("A_d", A_d))
    body_lines.append(_fmt_matrix("B_d", B_d if B_d.ndim == 2 else B_d.reshape(-1, 1)))
    body_lines.append(_fmt_matrix("C_m", C_mat))
    body_lines.append(_fmt_matrix("D_m", D_mat if D_mat.ndim == 2 else D_mat.reshape(-1, 1)))

    body_lines.append("")
    body_lines.append("void pulsim_model_init(PulsimModel* m) {")
    body_lines.append("    for (size_t i = 0; i < PULSIM_STATE_SIZE; ++i) m->x[i] = 0.0f;")
    body_lines.append("}")
    body_lines.append("")
    body_lines.append("void pulsim_model_step(PulsimModel* m, const float* u, float* y) {")
    body_lines.append("    float x_next[PULSIM_STATE_SIZE];")
    body_lines.append("    for (size_t i = 0; i < PULSIM_STATE_SIZE; ++i) {")
    body_lines.append("        float acc = 0.0f;")
    body_lines.append("        for (size_t j = 0; j < PULSIM_STATE_SIZE; ++j) acc += A_d[i][j] * m->x[j];")
    body_lines.append("        for (size_t j = 0; j < PULSIM_INPUT_SIZE; ++j)  acc += B_d[i][j] * u[j];")
    body_lines.append("        x_next[i] = acc;")
    body_lines.append("    }")
    body_lines.append("    for (size_t i = 0; i < PULSIM_OUTPUT_SIZE; ++i) {")
    body_lines.append("        float acc = 0.0f;")
    body_lines.append("        for (size_t j = 0; j < PULSIM_STATE_SIZE; ++j) acc += C_m[i][j] * m->x[j];")
    body_lines.append("        for (size_t j = 0; j < PULSIM_INPUT_SIZE; ++j)  acc += D_m[i][j] * u[j];")
    body_lines.append("        y[i] = acc;")
    body_lines.append("    }")
    body_lines.append("    for (size_t i = 0; i < PULSIM_STATE_SIZE; ++i) m->x[i] = x_next[i];")
    body_lines.append("}")
    body = "\n".join(body_lines) + "\n"

    test_runner = textwrap.dedent(f"""\
        /* PIL harness for pulsim_model_step. Reads (input_trace, n_steps)
         * via argv, runs the generated model, and writes the y trace
         * back to stdout as one line per step. */
        #include "model.h"
        #include <stdio.h>
        #include <stdlib.h>
        int main(int argc, char** argv) {{
            if (argc < 2) {{
                fprintf(stderr, "usage: %s <n_steps> [u_const]\\n", argv[0]);
                return 1;
            }}
            int n_steps = atoi(argv[1]);
            float u_const = (argc >= 3) ? (float)atof(argv[2]) : 0.0f;
            PulsimModel m;
            pulsim_model_init(&m);
            float u[PULSIM_INPUT_SIZE];
            for (size_t i = 0; i < PULSIM_INPUT_SIZE; ++i) u[i] = u_const;
            float y[PULSIM_OUTPUT_SIZE];
            for (int k = 0; k < n_steps; ++k) {{
                pulsim_model_step(&m, u, y);
                for (size_t i = 0; i < PULSIM_OUTPUT_SIZE; ++i) {{
                    if (i) putchar(',');
                    printf("%.10e", y[i]);
                }}
                putchar('\\n');
            }}
            return 0;
        }}
        """)

    written = []
    for name, content in [("model.h", header), ("model.c", body), ("model_test.c", test_runner)]:
        path = out_dir / name
        path.write_text(content)
        written.append(str(path))
    return written


def _import_pulsim_runtime():
    from .. import _pulsim
    return _pulsim


def generate(
    circuit,
    *,
    dt: float,
    out_dir: str | Path,
    target: str = "c99",
    t_op: float = 0.0,
    x_op=None,
) -> CodegenSummary:
    """End-to-end codegen.

    Args:
        circuit: a `pulsim.Circuit` (PWL-admissible — the model uses
            `Simulator::linearize_around` under the hood).
        dt: fixed-step discretization period (s).
        out_dir: directory the generated `.c` / `.h` are written to.
        target: code generator target — `"c99"` is the only one shipped
            today (ARM Cortex-M7 / Zynq targets are deferred).
        t_op: time at which to take the operating-point linearization.
        x_op: optional pre-computed DC OP state; otherwise the simulator
            runs DC OP automatically.

    Returns:
        `CodegenSummary` with the discrete matrices, stability margin,
        and a coarse ROM/RAM estimate.
    """
    if target != "c99":
        raise ValueError(
            f"codegen target {target!r} not yet supported; "
            "only 'c99' ships today (ARM / Zynq targets are deferred)")

    pl = _import_pulsim_runtime()
    out_path = Path(out_dir)

    # Linearize at (x_op, t_op).
    sim = pl.Simulator(circuit, pl.SimulationOptions())
    if x_op is None:
        dc = sim.dc_operating_point()
        if not dc.success:
            raise RuntimeError(
                "codegen: DC operating point failed — circuit must be "
                "PWL-admissible and the DC solver must converge")
        x_op = dc.newton_result.solution
    sys = sim.linearize_around(x_op, t_op)
    if not sys.ok:
        raise RuntimeError(
            f"codegen: linearization failed ({sys.failure_reason!r})")

    E = _to_dense(sys.E)
    A = _to_dense(sys.A)
    B = _to_dense(sys.B)
    C = _to_dense(sys.C)
    D = _to_dense(sys.D)

    # Reduce descriptor form to regular state-space.
    A_red, B_red, C_red, D_red, _P = _reduce_descriptor_form(E, A, B, C, D)

    # Discretize.
    A_d, B_d = discretize_state_space(A_red, B_red, dt)

    # Stability check.
    rho = stability_radius(A_d)
    if rho >= 1.0:
        raise RuntimeError(
            f"codegen: discrete state matrix is unstable "
            f"(spectral radius {rho:.6f} ≥ 1.0). Choose a smaller dt or "
            "check the circuit for sign-flipped passive elements.")

    # Emit C99.
    state_size  = A_d.shape[0]
    input_size  = B_d.shape[1] if B_d.ndim == 2 else 1
    output_size = C_red.shape[0]
    written = _emit_c99(
        out_path, A_d, B_d, C_red, D_red, dt,
        state_size, input_size, output_size)

    # Coarse ROM estimate: 4 bytes per matrix entry × number of entries
    # (matrices are `static const float[][]`). RAM = state vector +
    # one scratch x_next array.
    rom_bytes = 4 * (A_d.size + B_d.size + C_red.size + D_red.size)
    ram_bytes = 4 * 2 * state_size      # x + x_next

    return CodegenSummary(
        out_dir=out_path,
        state_size=state_size,
        input_size=input_size,
        output_size=output_size,
        A_d=A_d, B_d=B_d, C=C_red, D=D_red,
        stability_radius=rho,
        rom_estimate_bytes=rom_bytes,
        ram_estimate_bytes=ram_bytes,
        target=target,
        files_written=written,
    )
