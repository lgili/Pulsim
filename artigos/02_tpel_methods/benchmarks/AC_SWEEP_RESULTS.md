# AC sweep complex sparse LU — 2-backend microbenchmark results

Microbenchmark feeding the TPEL §VI.B AC-sweep table. Captured to
quantify the per-frequency cost of the v1.4.0 in-house
`PulsimComplexSparseLuSolver` vs the v1.3.0 baseline
(`Eigen::SparseLU<std::complex<Real>>`).

The v1.4.0 win is **not a per-frequency speedup** — both backends
implement well-tuned direct sparse LU and land within 2× of each
other. The win is **"zero third-party sparse LU in the production
path"** (software-supply-chain argument): every algorithm in the
production AC-sweep code path is now in-house, auditable, and
shippable under the project's MIT licence.

`Backend::Eigen` is retained explicitly as the IEEE TPEL §VI.B
paper-comparison baseline.

## Backends

Both backends consume the same complex MNA matrix
`M(ω) = j·ω·E + J` per frequency and drive the standard
`analyze + factorize + solve` lifecycle. The matrix is rebuilt at
every frequency (the `j·ω·diag(C)` term changes with ω while the
real part J is constant).

| # | Backend | Notes |
|---|---------|-------|
| **A** | `Backend::Eigen`  | `Eigen::SparseLU<std::complex<Real>, COLAMDOrdering<Index>>`. The v1.3.0 production path; v1.4.0+ retained as paper-comparison baseline. Mature 30-year-old algorithm, excellent constants. |
| **B** | `Backend::Pulsim` | `PulsimComplexSparseLuSolver` (= `PulsimSparseLuSolverT<std::complex<Real>>`). v1.4.0 in-house production path. Same algorithms as the v1.3.0 real-scalar solver (RCM + Liu/Davis etree + Gilbert-Peierls left-looking + threshold partial pivoting) lifted to complex via the `Scalar` template parameter. |

## How to reproduce

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pulsim_benchmarks -j

# Default — writes ac_sweep_microbench.csv to ./bench-results/
./build/core/pulsim_benchmarks "[ac_sweep][microbench]"

# Or steer the CSV into the artigos results dir
PULSIM_BENCH_RESULTS_DIR=$(pwd)/artigos/02_tpel_methods/benchmarks/results \
    ./build/core/pulsim_benchmarks "[ac_sweep][microbench]"
```

Each row in the CSV is one matrix size `n`, with both backends timed
across the same 100 log-spaced frequencies (1 Hz → 1 MHz) on the
same fixture.

## Fixture

Synthetic MNA-shaped complex matrix family parameterised on `n`:

* Tri-diagonal real part `J_real`: 2 Ω on the diagonal, −1 Ω off-
  diagonal between consecutive nodes. Plus a 1×10⁶ anchor on
  `(0, 0)` to break the rank-1 nullspace.
* Frequency-dependent capacitive contribution: `j·ω·1 nF` on every
  diagonal entry. At ω = 2π × 1 Hz the imaginary part is
  ≈ 6.28×10⁻⁹ (J-dominated); at ω = 2π × 1 MHz it's ≈ 6.28×10⁻³
  (still well below the diagonal R, so the matrix stays diagonally
  dominant — no fallback path).
* Pattern matches `|J| + |J^T|` (already symmetric here). Bandwidth
  1, post-RCM fill matches Eigen's COLAMD on this family.

This isolates the **complex sparse LU solver kernel itself** from
the full converter MNA stamping pipeline. A buck / NPC / MMC
follow-up bench would land on the more realistic but harder-to-
isolate measurement; the v1.4.0 supply-chain argument doesn't need
it.

## Captured run

Captured on **macOS 26.5 / Apple Silicon (ARM64) / AppleClang
17.0.0 / Pulsim feat/add-pulsim-complex-sparse-lu / Eigen 3.4 /
Release build (-O3 -DNDEBUG)**.

100 log-spaced frequencies from 1 Hz to 1 MHz per row.

| n   | nnz  | µs/freq (Eigen) | µs/freq (Pulsim) | **Pulsim ÷ Eigen** | Parity (|x_P − x_E|) |
|----:|-----:|----------------:|-----------------:|-------------------:|---------------------:|
| 8   | 22   | 8.98            | 5.48             | **0.61× (Pulsim faster)** | 1.06×10⁻²² |
| 16  | 46   | 8.16            | 7.64             | 0.94×              | 5.29×10⁻²² |
| 32  | 94   | 13.98           | 14.56            | 1.04×              | 2.86×10⁻²¹ |
| 64  | 190  | 28.49           | 33.36            | 1.17×              | 1.27×10⁻²¹ |
| 128 | 382  | 46.13           | 91.52            | 1.98× (Eigen faster) | 4.34×10⁻²¹ |

**Parity within 4.34×10⁻²¹ on every size** — well below the 1×10⁻¹⁰
solve tolerance the test gate enforces. Different column orderings
(Eigen's COLAMD vs the in-house RCM) produce different round-off
patterns but agree to ~1e-21 on the actual solve output. The two
backends are numerically interchangeable.

## Interpretation

Three observations, in order of weight for the TPEL paper:

1. **Rough parity across the relevant size range (n ≤ 32).** For
   typical SMPS state-vector sizes (n ≈ 8–32 for buck / boost /
   flyback / NPC-3L; n ≈ 100–200 for MMC arms), the in-house solver
   is within 1.2× of Eigen either way. AC-sweep wall-time is
   dominated by `n · n_freq` factorisation cost, and at SMPS-typical
   `(n, n_freq) = (32, 100)` this is ~1.5 ms either way — sub-
   perceptible.

2. **Eigen pulls ahead at large n (≥ 64).** The in-house solver's
   factorisation path is `O(n²)` in the worst case (the dense
   left-looking workspace), while Eigen's SparseLU uses
   reachability-based sparse triangular solves (Davis 2006 §3 fast
   path) that scale as `O(nnz · log n)`. The crossover here is
   ~ n = 32. For MMC-arm-scale matrices (n ≈ 200) the Eigen path
   would likely be ~2-3× faster. **A v1.4.0
   reachability-based fast path would close the gap**; that
   optimisation is out of scope for v1.4.0 since the headline is
   correctness + supply-chain, not performance.

3. **Pulsim wins at small n (n = 8) by ~1.6×.** Lower constants
   per-call when the matrix is small enough that the dense
   workspace is cheap. Matches the v1.3.0 real-scalar result —
   the in-house solver is tuned for SMPS-scale matrices and
   loses asymptotically to Eigen on larger ones.

## Honest limitations of this microbench

- **Synthetic tri-diagonal fixture, not a real converter.** Real AC
  sweeps run on the full MNA matrix from `dc_assemble` + the
  descriptor mass matrix E built from L/C devices, with all the
  voltage-source asymmetries and diode constraint rows that produce.
  The integration tests in `core/tests/analysis/test_mna_sweep.cpp`
  (RC low-pass within 0.1 dB / 1°; series RLC peak within 1.5 %)
  cover the full pipeline at the correctness level; the per-converter
  perf characterisation (buck + NPC + MMC arm) would land in a
  follow-up `add-ac-sweep-per-converter-bench` proposal.
- **Per-frequency cost only; no symbolic reuse.** Both backends
  re-`analyze + factorize` at every frequency. The symbolic pattern
  is actually identical across frequencies (only `j·ω` changes),
  which is exactly the use-case for the symbolic-reuse / sliding-
  solver pattern from v1.3.0's PWL cache. A follow-up
  `add-ac-sweep-symbolic-reuse` proposal would lift that win into
  the AC-sweep path; expected speedup ~2-5× for large `n_freq`. The
  v1.4.0 bench captures the simpler "drop-in swap" cost so the
  before/after comparison is apples-to-apples.
- **n caps at 128.** Memory pressure of the dense workspace makes
  the bench impractical to push beyond ≈ 200 on a laptop. The
  asymptotic claim "Eigen wins for MMC-arm-scale matrices" is
  extrapolation from the slope visible in the table, not direct
  measurement above n = 128.
- **Single-threaded; no SIMD vectorisation of the complex inner
  loop.** Both backends single-threaded, scalar complex
  multiply/add. Multi-threaded factorisation is out of scope.

## How this maps to the TPEL paper

The §VI.B table summarises the per-frequency cost decomposition.
The interpretation paragraphs feed the discussion on "why the AC
sweep migration to the in-house solver is a software-supply-chain
move, not a perf move" — which is the v1.4.0 contribution. The
"Honest limitations" §VI.D explicitly calls out the n ≥ 64 gap as
the motivation for a v1.4.0 reachability-based fast path
(future work).

## Schedule

* This benchmark (v1.4.0, add-pulsim-complex-sparse-lu): **captured
  2026-05-24**. Numbers above.
* Symbolic-reuse optimisation for AC sweep (`add-ac-sweep-symbolic-
  reuse`): deferred. Expected ~2-5× speedup at typical `n_freq` =
  100.
* Reachability-based sparse triangular solve in
  `PulsimSparseLuSolverT`: deferred to a future proposal. Targets
  the n ≥ 64 gap visible above; expected to bring the in-house
  solver within ~1.2× of Eigen at n = 128 and ~parity at n = 200.
* TPEL submit target: **Q1 2027**.
