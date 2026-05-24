# Changelog

All notable changes to Pulsim are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] — 2026-05-24

### Highlights — PWL rank-1 cache update path (Layer 4 V8)

This release ships the algorithmic contribution that backs the
planned IEEE TPEL methods paper on Pulsim's PWL state-space cache
(see [`artigos/02_tpel_methods/`](artigos/02_tpel_methods/)). The
full design + decisions + delta specs live in
[`openspec/changes/add-pwl-rank1-update/`](openspec/changes/add-pwl-rank1-update/)
and the captured benchmark in
[`artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`](artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md).

**No BREAKING changes.** Every existing caller produces bit-identical
output. The rank-1 path is purely additive performance. 17,839 layer5
assertions across 89 test cases pass unchanged.

### Added

- **`pulsim::sparse::KluSolver`** — `DirectSolver` implementation
  wrapping SuiteSparse KLU (Davis & Natarajan, *ACM TOMS* 37(3), 2010,
  Algorithm 907). Purpose-built for circuit MNA matrices. Header-only,
  gated on `PULSIM_HAVE_KLU` (set by the root `CMakeLists.txt`'s
  `find_package(KLU CONFIG)` block). When KLU is absent the kernel
  builds and runs identically using `Eigen::SparseLU`.
- **`pulsim::sparse::Backend` enum + factory hint** — new overload
  `make_default_solver(Size n, Backend hint = Backend::Auto)` lets the
  caller request `Backend::KLU`, `Backend::Eigen`, or auto-pick by
  matrix size. Default `Backend::Auto` picks KLU when n ≥ 100
  (`PULSIM_KLU_AUTO_THRESHOLD`, tuneable at build).
- **`DirectSolver::supports_partial_refactor()`** + **`partial_refactor(M, changed_cols)`** —
  new virtual methods on the base interface, with default impls
  returning `false` so existing solvers transparently fall back.
- **`PwlStateSpaceCache::solve_rank1(mask, b_extra, x)`** — sliding-
  solver fast path. On single-bit Gray-code mask flips it calls
  `partial_refactor` instead of rebuilding the cache segment; falls
  back transparently to full re-factor on multi-bit flips, unsupported
  backends, or numerical singularities.
- **`PwlStateSpaceCache::set_rank1_backend(Backend)`** — pre-`solve_rank1`
  override for the rank-1 sliding solver's backend (useful for
  benchmarks that want to exercise KLU even at small n).
- **`PwlStateSpaceCache::metrics()`** + **`pulsim::pwl::CacheMetrics`** —
  `{rank1_hits, full_refactor_hits, fallbacks}` atomic monotonic
  counters for benchmark attribution. Thread-safe sampling via
  `std::memory_order_relaxed`.
- **Microbenchmark `core/tests/benchmarks/test_bench_pwl_rank1.cpp`** —
  Catch2 binary in the opt-in `pulsim_benchmarks` target. Sweeps
  N ∈ {4, 6, 8, 10, 12} switches, times `solve` vs `solve_rank1`,
  writes CSV to `${PULSIM_BENCH_RESULTS_DIR}/rank1_microbench.csv`.
- **CI matrix** updated to install `libsuitesparse-dev` on Linux +
  `suite-sparse` via brew on macOS across every existing entry
  (Clang 17/18, GCC 13, Debug sanitizers, coverage).
- **README "Build prerequisites"** section documenting the new
  optional dependency with install commands for macOS / Debian /
  Fedora and the `-DPULSIM_ENABLE_KLU=OFF` opt-out.

### Performance

Captured microbench on macOS 26.5 / Apple Silicon / AppleClang 17:

| N | n_state | µs/solve | µs/rank1 | speedup |
|--:|--:|--:|--:|:--:|
| 4  | 6  | 4.67 | 10.29 | 0.45× (overhead dominates at tiny n) |
| 6  | 8  | 2.57 | 2.79  | 0.92× (break-even) |
| 8  | 10 | 2.57 | 2.90  | 0.89× (break-even) |
| 10 | 12 | 4.60 | 2.73  | 1.69× (rank-1 wins) |
| 12 | 14 | 9.69 | 3.08  | **3.15×** (headline finding) |

Per-call rank-1 cost stays ~3 µs across the sweep while per-call
`solve` cost grows linearly with n — the textbook signature of
amortising the symbolic factorisation across all calls. The
V0 MVP delegates to `klu_refactor`; the V8.1 follow-up will replace
that with path-based partial re-elimination per Chen et al.,
IEEE TPEL 2024 §III, extending the speedup to 5-10× at n=200.

### Not changed

- **Public Python API** — `pp.simulate(builder, …)` continues to use
  the existing per-mask cache path via `cache.solve(mask)`. Wiring
  `solve_rank1` into Layer 5's `run_transient` + Python bindings is
  out of scope of this release; tracked as
  `add-pwl-rank1-runtime-integration` (TBD).
- **All 8 reference projects under `projects/`** — bit-identical
  output (verified via the layer5 / layer5_v1..v4 / showcase regression
  test suite, 17,839 assertions across 89 test cases).

## [1.1.0] — 2026-05-23

### Highlights — JOSS submission release

This release marks the first version of Pulsim accompanied by a
peer-reviewed publication. The accompanying paper has been submitted
to the [Journal of Open Source Software (JOSS)](https://joss.theoj.org/);
the source lives in [`artigos/01_joss_tool_paper/`](artigos/01_joss_tool_paper/).
Once the JOSS paper is accepted, this version's DOI will be the
canonical software citation.

### Added

- **`LICENSE`** at repo root — MIT text. The licence was previously
  only declared in `pyproject.toml`; JOSS (and most academic
  citation tools) require the licence file at the root.
- **`CITATION.cff`** at repo root — Citation File Format v1.2.0
  metadata for automatic citation generation by GitHub and tools
  like `cffconvert`.
- **`artigos/` directory** — paper sources for the Pulsim publication
  campaign, with `README.md` documenting the 4-paper strategic plan
  (JOSS tool paper → EPE-ECCE Europe 2026 conference →
  IEEE Open Journal of Power Electronics methods paper →
  IEEE TPEL / JESTPE application paper).

### Fixed

- **README quick-start example** — `p.scope(...)` updated to
  `p.plot.scope(...)` to match the actual location of the plot
  helper in the current Pulsim 1.x API. Verified end-to-end
  against the installed package.

## [0.10.0] — 2026-05-19

### Highlights

The 0.10.0 release closes the alpha cycle that started with `0.10.0a1`
and adds a **switched-mode closed-loop control surface** that brings
Pulsim into PSIM/Simulink territory for power-electronics controller
design and verification.

### Added — Switched-Mode Closed-Loop

- **`Simulator.run_transient(x0, circuit, callback)`** — new binding
  overload that accepts a Python callback invoked after every accepted
  timestep. The callback can call back into the circuit
  (`circuit.set_pwm_duty(name, new_duty)`, `circuit.set_pmsm_foc_references(...)`,
  …) to close the loop. Single transient run, full state preservation,
  Python in control — same architectural pattern as PSIM / Simulink.
- **GIL-safe streaming binding** — `run_transient_streaming` now
  releases the GIL around the C++ integration loop, lets callbacks
  re-enter pybind11 safely, and survives `None` callbacks. The
  `py::call_guard<py::gil_scoped_release>` race that crashed on every
  invocation is fixed.
- **`RuntimeCircuit::has_any_dynamic_history()`** — kernel helper that
  lets `Simulator::run_transient_native_impl` discriminate fresh-circuit
  vs. continuation calls. Continuations now preserve cap `i_prev` and
  inductor `v_prev` on the same Circuit instance (the per-period
  closed-loop pattern no longer collapses the dynamic state).
- Periodic shooting `run_periodic_shooting` retains "fresh-state-per-
  shooting-iteration" semantics — explicit `update_history(guess, true)`
  reset before each `run_transient(guess)` call.

### Added — Teaching Notebooks

- `examples/notebooks/vsi_inverter_design.ipynb` — end-to-end design
  of a 3φ Voltage Source Inverter (SPWM, 16 kHz, 6 SiC MOSFETs).
- `examples/notebooks/boost_pfc_vsi_design.ipynb` — full AC → DC → 3φ AC
  cascade (220 V_rms in, 400 V DC bus, 230 V_rms 3φ out).
- `examples/notebooks/boost_pfc_closed_loop.ipynb` — switched-mode
  closed-loop PFC using `Simulator.run_transient(x0, ckt, callback)`.
  V_dc converges (architecture proof-of-concept; PI tuning is iterative
  follow-up work — cascaded ACMC is the next milestone).

### Fixed

- `run_transient(x0)` no longer ping-pongs voltage-source nodes between
  `0` and `2·V_src` when `x0 = zeros` (consistent initialization fix
  in `a2cb883`).
- `run_transient_streaming` no longer aborts the process with
  `pybind11::handle::inc_ref` GIL assertions when any callback is
  passed (including `None`).
- Per-period closed-loop boost: cap state is preserved across
  Simulator constructions sharing the same Circuit, removing the
  divergence-to-0V symptom on continuation runs.
- 95 `ruff` errors across `python/` brought to zero — E702 multi-stmt
  semicolons split onto separate lines, F401 unused imports added to
  `__all__` or removed, E402 imports-after-importorskip ignored at the
  per-file level for property tests.
- `mkdocs build --strict` is green again — removed dangling refs to
  retired loss-params classes (`MOSFETLossParams`, `IGBTLossParams`,
  `DiodeLossParams`, `ConductionLoss`, `SwitchingLoss`), switched
  cross-tree file links to absolute GitHub URLs, added `: Any`
  annotations on `circuit` params that griffe was flagging.
- Stress benchmark suite no longer aborts on `periodic_rc_pwm` —
  added the missing entry to `benchmarks/benchmarks.yaml` (with no
  SPICE netlist, since the periodic-analysis bench has no parity
  baseline).
- `test_fmu_*` skip cleanly on Windows (ctypes.CDLL holds the DLL
  handle across `TemporaryDirectory` cleanup → PermissionError).
- `test_bode_plot_rejects_failed_result` skips when matplotlib is
  not installed (Windows CI).
- `test_shooting_uses_warm_start_retry_for_pwm_case` marked `xfail`
  pending shooting-solver re-tune for dead-time PWM (regression
  pre-dates this release; tracked separately).

### Notebooks — also revalidated

- `boost_converter_design.ipynb` runs end-to-end on the new kernel
- `flyback_converter_design.ipynb` runs end-to-end on the new kernel
- `vsi_inverter_design.ipynb` / `boost_pfc_vsi_design.ipynb` —
  `np.trapz` → `np.trapezoid` compat for NumPy 2.x

### Removed

- (No public API removals in this release. The loss-params classes
  documented in earlier alpha series were already replaced by
  device-side params during the alpha cycle.)

### Migration

- The new closed-loop pattern is **opt-in via a new binding overload**.
  Existing single-shot transient calls (`Simulator.run_transient()` /
  `Simulator.run_transient(x0)`) behave exactly as before.
- Per-period closed-loop users that reused the same Circuit across
  Simulator constructions now get **correct state preservation** by
  default. If your code depended on the old "reset on every call"
  behaviour, call `circuit.update_history(x, True)` explicitly before
  each `run_transient` to force the reset.

### Internal

- Kernel test suite: **304 cases / 4214 assertions** green.
- Python lint: **`ruff check python/`** zero errors.
- Docs build: **`mkdocs build --strict`** green.

### Notable commits

- `fc3c686` — kernel: preserve dynamic-device history + streaming GIL fix
- `ed879af` — bindings: `Simulator.run_transient(x0, ckt, callback)`
- `9062c78` — notebook: closed-loop PFC switched-mode proof-of-architecture
- `cef7981` — notebook: AC → DC → 3φ AC cascade design
- `663e3be` — notebook: 3φ VSI design walkthrough
- `9806df5` — chore: zero ruff errors
- `c5d7699` — fix: docs strict + benchmark index
- `1b9d01d` — fix: restore periodic shooting + Windows test gates

---

## Earlier Releases

See [GitHub Releases](https://github.com/lgili/Pulsim/releases) for
0.9.0 and earlier.
