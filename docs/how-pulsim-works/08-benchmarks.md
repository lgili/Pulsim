# 8. Benchmarks

!!! info "Status: outline / next iteration"
    Source material:
    `artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`,
    `artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv`,
    and `core/tests/benchmarks/test_bench_pwl_rank1.cpp`.
    Plots will read directly from the CSV.

What the v1.3.0 improvements actually buy you, measured on
reproducible fixtures. The headline: **2.7-2.9× speedup at
$n_{\mathrm{state}} \ge 14$ over the baseline cache**, with the
win decomposed into the sliding-solver amortisation and the
path-based partial refactor on top.

## Planned sections

1. **Methodology**: the 3-backend microbench
   (`test_bench_pwl_rank1.cpp`). Same N-switch chain fixture,
   2000 single-bit Gray-code transitions, three routes through
   the cache:
   - (A) `solve(mask, ...)` — per-mask cache, lazy build
   - (B) `solve_rank1(...)` with `Backend::Eigen` — sliding
     solver, but Eigen::SparseLU doesn't support partial_refactor
     so every flip is a full numeric factorise
   - (C) `solve_rank1(...)` with `Backend::Pulsim` — path-based
     partial refactor
   B-vs-A isolates the amortised-symbolic win; C-vs-B isolates
   the path-based win.

2. **The captured table** (preview from RANK1_RESULTS.md):
   $n_{\mathrm{state}} \in \{6, 8, 10, 12, 14, 18, 22, 26\}$.
   At $n_{\mathrm{state}} = 14$: A = 10.0 µs/solve, B = 5.4
   µs/solve, C = 3.6 µs/solve → 2.81× headline. Zero fallbacks
   across all 1999 single-bit flips per N.

3. **The decomposition**: B/A ≈ 1.7× (sliding-solver
   amortisation), C/B ≈ 1.5-1.8× (path-based on top),
   product ≈ 2.7-2.9× headline. Showing the multiplicative
   stacking is the §VII discussion of the TPEL paper.

4. **The small-$n$ crossover**: at $n_{\mathrm{state}} \le 10$
   the per-mask cache beats path-based because path-construction
   overhead dominates. Honest acknowledgement that
   path-based is a "medium-to-large $n$" optimisation.

5. **Pivot-threshold tuning lessons**: original `PIVOT_RATIO_TOL
   = 1.1` (strict) caused fallbacks on circuit MNA's wide pivot-
   magnitude swings. Switching to KLU-style `PIVOT_THRESH = 1e-3`
   (threshold pivoting per Demmel 1997 §3.4) achieved zero
   fallbacks. This is the kind of tuning that has to come from
   benchmark data, not from theory.

6. **Limitations**:
   - Synthetic N-switch-chain fixture, not a real converter.
     Real-converter benchmark is deferred to
     `add-pwl-rank1-runtime-integration` (Python wiring needed
     first).
   - $n_{\mathrm{state}}$ caps at 26 on this fixture. Asymptotic
     "speedup at the MMC scale $n \approx 200$" relies on
     extrapolation of the flat-vs-linear per-call-cost scaling.
   - Single-bit-flip workload only. Multi-bit transitions fall
     back to full factorise.
   - Single-threaded. No GPU or multi-thread claims.

## Planned figures

- **Fig 8.1** — Captured speedup vs $n_{\mathrm{state}}$ for
  all three columns (A, B, C). Three line plots overlaid;
  shaded region for the small-$n$ crossover zone.
- **Fig 8.2** — Decomposition stacked bar at $n = 14, 18, 22, 26$.
  B/A and C/B bars stacked multiplicatively to reach C/A.
- **Fig 8.3** — Per-call cost (µs/solve) vs $n_{\mathrm{state}}$
  for all three columns. Log-y if needed. Shows the asymptotic
  scaling: A grows linearly with $n$, C stays roughly flat.
  This is the figure that justifies "the speedup grows with $n$".
- **Fig 8.4** — Pivot-fallback rate heatmap across the 8 N
  values × pivot-threshold sweep ($10^{-5}, 10^{-4}, 10^{-3},
  10^{-2}, 10^{-1}$). Confirms `1e-3` is the sweet spot.

## Reproducing the benchmark

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pulsim_benchmarks -j

PULSIM_BENCH_RESULTS_DIR=$(pwd)/artigos/02_tpel_methods/benchmarks/results \
    ./build/core/pulsim_benchmarks "[rank1][microbench]"
```

The CSV at
`artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv`
is committed in the repo; chapter figures pull from there
directly.

## Cross-references

- [`RANK1_RESULTS.md`](https://github.com/lgili/Pulsim/blob/main/artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md)
  has the full discussion (this chapter is a docs-side
  re-presentation of the same data).
- [Chapter 7 — Path-Based Partial Refactor](07-rank1-partial-refactor.md)
  is what's being measured here.
- [Chapter 9 — Architecture Walkthrough](09-architecture-walkthrough.md)
  explains where in the layer stack the benchmark hooks in.
