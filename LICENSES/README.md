# Third-party licenses bundled with Pulsim

Pulsim itself is **MIT-licensed** (see [`../LICENSE`](../LICENSE) at the
repo root). This directory contains license texts and provenance records
for third-party code that Pulsim builds against or vendors via CMake
`FetchContent`.

## Bundled / vendored components

| Component | License | Where it's pulled from | Why we vendor |
|---|---|---|---|
| **SuiteSparse KLU** (incl. AMD, COLAMD, BTF, SuiteSparse_config) — DPsim fork | KLU + BTF: **LGPL-2.1+**; AMD + COLAMD: BSD-3; SuiteSparse_config: BSD-3 (license texts in [LGPL-2.1.txt](LGPL-2.1.txt)) | [dpsim-simulator/SuiteSparse](https://github.com/dpsim-simulator/SuiteSparse) @ commit `6cf76809...` (master, 2023-03-08) — see [SuiteSparse-DPsim-fork.md](SuiteSparse-DPsim-fork.md) | Extends KLU with path-based partial refactorization primitives (`klu_compute_path`, `klu_partial_factorization_path`, `klu_partial_refactorization_restart`) per Schumacher/Dinkelbach 2021. These are NOT in upstream Davis SuiteSparse. Used by the rank-1 PWL cache fast-path (`openspec/specs/pwl-rank1-update/`). |

## Fetched-but-unmodified upstream dependencies

These are pulled via CMake `FetchContent` but built against unmodified.
Their licenses apply to the bundled binaries / static libs in any
distribution of Pulsim:

| Component | License | Source |
|---|---|---|
| Eigen 3.4+ | MPL-2 | https://gitlab.com/libeigen/eigen |
| yaml-cpp 0.8.0 | MIT | https://github.com/jbeder/yaml-cpp |
| Catch2 v3.8.0 (test-only) | BSL-1.0 | https://github.com/catchorg/Catch2 |

## What you MUST do if you redistribute Pulsim binaries

The LGPL-2.1+ terms attached to the vendored KLU + BTF require, when
distributing a binary that statically links them:

1. **Ship this `LICENSES/` directory verbatim** (or an equivalent
   compilation that includes the LGPL-2.1 text and the upstream URL).
2. **Provide a way for users to relink** their own KLU + BTF objects —
   typically satisfied by Pulsim being open-source itself (any user can
   rebuild from source against a different KLU).
3. **Acknowledge KLU's authorship** in user-facing documentation. The
   conventional form is: *"Pulsim uses KLU by Timothy A. Davis (Texas
   A&M University) and the path-based partial refactorization
   extensions by Lukas Schumacher and Markus Dinkelbach (DPsim
   project)."*

For pure source distributions (e.g. cloning the repo), the LGPL
requirements are inherently satisfied — anyone can rebuild against any
KLU they choose, including upstream Davis SuiteSparse, by passing
`-DPULSIM_ENABLE_KLU=OFF` (falls back to `Eigen::SparseLU`) or by
manually overriding the FetchContent source.

## License compatibility — short version

| What you're combining | Allowed? | Notes |
|---|:-:|---|
| Pulsim (MIT) + KLU (LGPL-2.1+) statically linked | ✅ | Provided LGPL terms above are honoured. |
| Pulsim (MIT) + Eigen (MPL-2) statically linked | ✅ | MPL is "weak copyleft"; static linking is fine for unmodified Eigen. |
| Pulsim binary in a closed-source product | ⚠️ | Possible, but the LGPL relinking requirement applies to the bundled KLU/BTF objects. Most commercial vendors satisfy this by either (a) shipping Pulsim as a dynamic library, or (b) including object files for the LGPL'd portions in the distribution. |
| Pulsim + a GPL-only library | ❌ | We don't link to any GPL code. Adding one would force the combined binary to be GPL. |

This summary is informational, not legal advice. For commercial
redistribution consult a lawyer.

## Why we DON'T vendor upstream Davis SuiteSparse

Two reasons:

1. **Functional gap.** Upstream KLU does NOT expose the elimination
   tree needed for path-based partial refactorization. The DPsim fork
   added the necessary public functions and `klu_numeric` fields in
   2021. Without those, Pulsim's rank-1 cache update path would have
   to delegate to `klu_refactor` (full numeric refactor with cached
   symbolic) — the V8 MVP behaviour, which we shipped in v1.2.0 and
   are now upgrading.

2. **License continuity.** The DPsim fork preserves upstream KLU's
   LGPL-2.1+ license — no relicensing surprises. The fork's CMake
   glue (by Sergiu Deitsch, 2016-2021) is Apache-2.0; that affects
   only the build system, not the linked binary.
