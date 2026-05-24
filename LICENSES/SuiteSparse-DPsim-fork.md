# Provenance: dpsim-simulator/SuiteSparse fork (KLU partial-refactor)

## Source

- **Repository**: https://github.com/dpsim-simulator/SuiteSparse
- **Branch**: `master`
- **Pinned commit**: `6cf768091962336466808e7f02d476842e4c5281`
- **Commit message**: "updated macros and clarified function arguments"
- **Commit date**: 2023-03-08
- **Vendored on**: 2026-05-24 (this commit on Pulsim feat/pwl-rank1-partial-refactor)

## What this fork adds vs upstream

Upstream Davis SuiteSparse (https://github.com/DrTimothyAldenDavis/SuiteSparse)
ships KLU as a circuit-MNA-optimized sparse LU solver. The DPsim fork
extends KLU with **path-based partial refactorization** primitives:

| New public function | Purpose |
|---|---|
| `klu_compute_path` | Precomputes the elimination-tree path for a set of varying matrix entries, populating `klu_numeric->path`, `block_path`, `variable_block`, `variable_offdiag_*` |
| `klu_partial_factorization_path` | Hot-path partial refactor — re-eliminates only the columns on the precomputed path. O(path) per call vs O(nnz·log n) for full `klu_refactor` |
| `klu_partial_refactorization_restart` | Variant that takes a starting column hint (consumes `klu_determine_start` output) |
| `klu_analyze_partial` | Alternative analyzer that orders the matrix specifically for partial refactor (not yet used by Pulsim — V8.2 candidate) |

Plus new fields on `klu_numeric`:
- `int *path`, `int pathLen`
- `int *block_path`
- `int *variable_block`, `int n_variable_blocks`
- `int *variable_offdiag_orig_entry`, `int *variable_offdiag_perm_entry`,
  `int variable_offdiag_length`

Plus new error code:
- `KLU_PATH_INVALID (-6)` — `klu_compute_path` wasn't called

## License

Each vendored sub-component keeps its upstream license:

| Sub-component | License |
|---|---|
| KLU/Source/*.c, KLU/Include/klu.h | **LGPL-2.1+** (incl. the new path-based files added by the fork — Schumacher/Dinkelbach 2021 contributed under the same LGPL terms as the surrounding code) |
| BTF/Source/*.c, BTF/Include/btf.h | LGPL-2.1+ |
| AMD/Source/*.c, AMD/Include/amd.h | BSD-3-Clause |
| COLAMD/Source/*.c, COLAMD/Include/colamd.h | BSD-3-Clause |
| SuiteSparse_config/* | BSD-3-Clause |
| `CMakeLists.txt` (root) — by Sergiu Deitsch 2016-2021 | Apache-2.0 |

The LGPL-2.1 license text is in [LGPL-2.1.txt](LGPL-2.1.txt) (501 lines).

LGPL-2.1+ allows static linking from MIT code (Pulsim) provided the
LGPL terms are honoured. See [README.md](README.md) "What you MUST
do if you redistribute Pulsim binaries".

## Why we pin to a 2-year-old commit

The DPsim fork has been quiet since 2023-03-08 (no new commits at the
time of vendoring). That's a feature: the partial-refactor patches
have been stable for over 2 years with no breaking changes. We pin
to that commit for reproducibility. If upstream eventually adds new
features or fixes we want, we re-pin in a follow-up PR.

## Algorithmic references

The path-based partial refactorization algorithm implemented in this
fork's KLU patches is described in:

- Chan, Brandwajn & Tinney, "Partial Matrix Refactorization,"
  *IEEE Trans. Power Systems* 1(1), 1986, pp. 193-199. The original
  algorithm (for power-flow studies).
- Chen, Brandwajn & Chan, "Partial refactorization with unrestricted
  topology changes," 1995. Generalization to broader matrix-change
  classes.
- Abusalah, Saad, Mahseredjian, Karaagac, Kocar, "Accelerated
  Sparse Matrix-Based Computation of Electromagnetic Transients,"
  *IEEE Open Access J. Power Electron.* 7, 2020.
- Dinkelbach, Schumacher, Razik, Benigni, Monti, "Factorisation Path
  Based Refactorisation for High-Performance LU Decomposition in
  Real-Time Power System Simulation," *Energies* 14(23):7989, 2021.
  **Primary algorithmic reference** — describes the exact algorithm
  implemented in the fork, plus benchmarks on 11k-220k-node EMT
  matrices.

## How to verify the vendored source

```bash
# Re-download the pinned commit (about 3 MB):
git clone https://github.com/dpsim-simulator/SuiteSparse.git /tmp/klu_check
cd /tmp/klu_check
git checkout 6cf768091962336466808e7f02d476842e4c5281

# Confirm the path-based files exist:
ls KLU/Source/klu_compute_path.c \
   KLU/Source/klu_partial_factorization_path.c \
   KLU/Source/klu_partial_refactorization_restart.c
# All three should print without error.

# Confirm the new public API in klu.h:
grep -E "klu_compute_path|klu_partial_factorization_path|KLU_PATH_INVALID" \
     KLU/Include/klu.h
# Should show the function prototypes + error code.
```

If those checks fail, the fork was modified after our 2026-05-24
capture and the vendoring claims in this file need updating.
