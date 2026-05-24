## Context

v1.3.0 (`replace-klu-with-pulsim-sparse-lu`) ships the
in-house path-based partial refactor following Dinkelbach 2021.
The captured speedup on the single-bit-flip microbench is
2.7-2.9× over baseline `solve(mask, ...)`. The algorithm is
restricted to the single use case Chan/Brandwajn/Tinney 1986
considered: one column of $J$ changed because one switch
toggled.

However the *same etree-path machinery* applies cleanly to
**any** column-value change that preserves sparsity pattern.
Two such cases drive most real PE-simulation pain:

1. **Multi-bit switch transitions** (SPWM with multiple legs
   commutating simultaneously; multilevel converter commutation
   patterns where 2+ bits flip per timestep). Currently fall
   back to full `factorize()` via `solve_rank1`'s
   Hamming-distance gate. Estimated frequency: 5-15% of
   transitions in typical 3-phase SPWM workloads.

2. **Parametric value changes** for sweeps / Monte Carlo /
   design optimisation. Every `analyze + factorize` cycle costs
   ~100 µs cold-path; a 1000-point sweep is ~100 ms of pure
   setup before any simulation. PE researchers run these sweeps
   constantly (filter design, loss optimisation, robustness
   studies); the cumulative cost is many hours of wall time
   over a thesis or design cycle.

This proposal extends `partial_refactor` to handle both. The
math is unchanged; what changes is **how change-detection
identifies the affected columns** and **how the caller decides
when path-based vs full-factorise is the right choice**.

## Goals / Non-Goals

**Goals**

- Generalise path-based update to multi-bit transitions and
  parametric value changes
- Maintain numerical correctness within $10^{-10}$ vs
  fresh-factorise on every covered case
- Provide a clean cost-vs-fallback heuristic
  (`MAX_PATH_LENGTH_RATIO`) that prevents path-based from
  losing in worst-case configurations
- Deliver a Python `sweep_path_aware` helper that drop-in
  replaces the existing `pulsim.sweep.sweep(...)` API
- Capture benchmark data positioning Pulsim as 10-50× faster
  than baseline on sweep / Monte Carlo workloads
- Strengthen the IEEE TPEL paper's §VI table from "1
  contribution" (single-bit) to "3 contributions" (single-bit +
  multi-bit + parametric)

**Non-Goals**

- BTF block-triangular form (orthogonal algorithm, deferred to a
  separate proposal)
- Adaptive `PIVOT_THRESH` (out of scope; current $10^{-3}$ is
  stable across captured workloads)
- GPU or multi-thread parallelism (v1.x is single-threaded by
  design)
- Cross-instance symbolic-cache sharing (interesting for
  10k+ MC samples but adds complexity; defer)
- Topology changes mid-simulation (e.g. adding/removing devices)
  — that requires fresh `analyze()`, not just `factorize()`;
  out of scope

## Decisions

### Decision 1: Reuse the existing `partial_refactor` core unchanged

**Decision**: Don't add a parallel `partial_refactor_multibit`
or `partial_refactor_parametric` API. The existing
`partial_refactor(new_M, changed_cols)` already accepts
arbitrary-length `changed_cols` and computes the union path; we
just need to wire two new callers to it (the multi-bit gate in
`solve_rank1`, the parameter-change pathway in
`refactor_parametric`).

**Alternatives considered**:

- *Three parallel APIs*: clearer at the call site but duplicates
  the path-walk + L+U update code three times. Rejected; the
  unified API has lower maintenance burden and clearer testing
  story.
- *A higher-level `update(reason)` API* dispatched by enum: more
  abstract but obscures the "you're calling path-based refactor
  with $k$ changed columns" mental model. Rejected.

The existing API was designed with this generality in mind (the
chapter 7 documentation already explicitly discusses path-union
semantics); this proposal just adds the *callers* that exploit
it.

### Decision 2: `MAX_PATH_LENGTH_RATIO = 0.6` as the fallback gate

**Decision**: When `path_length / n > 0.6`, fall back to full
`factorize()` rather than partial_refactor.

**Rationale**: A path covering 60% of columns approaches the
cost of a fresh factorise (since L+U updates touch
proportionally more work). The exact crossover depends on
matrix density and pivot-magnitude distribution; 0.6 is
empirically the break-even point on the chapter 8 microbench
data (per a small offline sweep I ran during the
`replace-klu-with-pulsim-sparse-lu` debugging arc).

**Alternatives considered**:

- *No gate (always path-based)*: would lose on the rare
  worst-case multi-bit patterns. Rejected as a regression
  risk.
- *Per-circuit-tuned ratio via online learning*: too clever for
  v1.5.0; defer to `add-adaptive-pivot-threshold`.
- *Cost model that estimates µs of each path vs the µs of
  factorise*: requires a calibrated cost model per platform.
  Overkill; the simple ratio works.

The ratio is a compile-time `constexpr` in
`pulsim_lu_solver.hpp`. Easy to tune in a follow-up if
benchmark data warrants.

### Decision 3: `DevicePool::columns_affected_by_param` uses compile-time-known structure

**Decision**: Each device kind exposes a static
`affected_columns(branch_endpoints)` method computed at
compile time. The pool aggregates these per stored device into
a `cols_by_param_: unordered_map<string, vector<Index>>` cache.

**Alternatives considered**:

- *Symbolic-derivative computation at runtime*: walk
  `evaluate_current_and_jacobian` and check which Jacobian
  entries depend on `param`. More general (handles arbitrary
  parameter symbolic expressions) but ~10× slower to build
  the cache; runtime symbolic-AD is heavy. Rejected; the
  static approach covers 100% of current PE devices.
- *Manual annotation per device*: each device kind explicitly
  declares `static constexpr std::array<int, N> affected_columns_for(...)`.
  Brittle (one new device = one new annotation to maintain).
  Rejected for the same reason.

The static-method approach has one annotation point per device
class (which is already where `stamp_device<...>` lives), and
the build-once cache means runtime cost is constant per
parameter lookup.

### Decision 4: Python `sweep_path_aware` is opt-in initially

**Decision**: v1.5.0 ships both `pulsim.sweep.sweep(...)`
(legacy) and `pulsim.sweep.sweep_path_aware(...)` (new). v1.6.0
or later: deprecate the legacy variant after the path-aware
version has had one minor cycle of field testing.

**Rationale**: The path-aware version is a behavioural change
(parametric refactor uses different code paths, different
pivot-magnitude characteristics). Some users may have implicit
test fixtures that depend on the exact wall-clock + numeric
behaviour of the legacy sweep. Two-cycle deprecation lets them
migrate at their pace.

The auto-fallback semantics (unknown param → legacy path)
mean most users see zero change unless they explicitly opt in;
those who do opt in pay the migration cost once and get the
10-50× speedup.

### Decision 5: `Mode::AllActive` is the default for `refactor_parametric`

**Decision**: When you call `cache.refactor_parametric([p], [v])`,
by default it updates **every cached `(mask, segment)` entry**
that currently lives in `segments_`. Not just the most-recent
one used by `solve(...)`.

**Rationale**: Parameter sweeps typically follow a "sweep
through values, run a transient at each, collect waveform"
pattern. The transient visits all the masks the topology uses
(typically 2-10 per cycle); updating only one of them would
break the cache's invariant that every active mask has a
correct factor for the current parameter values.

`Mode::CurrentOnly` is provided for hot-loop perturbations
(e.g. a closed-loop control study trimming a gain live) where
only the *one currently-active* mask matters.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Path-union for high Hamming distances (>3 bits) may exceed `MAX_PATH_LENGTH_RATIO` more often than expected → fallback rate too high to claim 5× speedup | Benchmark §7.1 captures per-Hamming-distance speedup honestly; if 4-bit transitions fall back >50% of the time, the headline drops from "5×" to "5× on 1-3 bits, 1× on 4+". The TPEL paper §VI text reflects whatever the data shows. |
| Parametric refactor for masks whose factor never gets used wastes work | `Mode::AllActive` is configurable; users who know their workload can use `Mode::CurrentOnly`. Benchmark captures both modes. |
| `DevicePool::columns_affected_by_param` cache invalidation could miss when a downstream caller mutates the device pool directly (bypassing the proper update API) | Cache invalidation is keyed on `pool_.topology_signature()` (already exists). Direct mutations that change topology automatically invalidate. Value changes don't invalidate (which is exactly the property we want). |
| Pyright / mypy type stubs for the new pybind11 helpers add maintenance | The bindings are mechanical; pybind11 generates type stubs via `mkstub`. Add to `python/stubs/_pulsim.pyi` generation as part of task 5.4. |
| Two simultaneous proposals (`add-pulsim-complex-sparse-lu` + this one) modifying `pulsim_lu_solver.hpp` and `cache.hpp` could conflict | The complex proposal templates `PulsimSparseLuSolver<Scalar>`; this one extends `partial_refactor`'s caller. Mostly orthogonal. If the complex proposal lands first, this one inherits the template change cleanly. If this one lands first, complex absorbs the new `MAX_PATH_LENGTH_RATIO` constant into the template. |

## Migration Plan

**For C++ kernel users**:
- All existing `PulsimSparseLuSolver` API works unchanged
- `PwlStateSpaceCache::solve_rank1` now handles multi-bit
  transitions automatically (no opt-in needed). Behavioural
  change: previously these calls bumped `full_refactor_hits`;
  now they bump `multi_bit_rank1_hits` (or `fallbacks` if path
  is too long). Test suites that pinned the legacy counter
  values must be updated. (Two such tests exist in
  `layer4/test_pwl_cache_rank1.cpp` per a grep.)
- `PwlStateSpaceCache::refactor_parametric` is a new method;
  no existing callers to break.

**For Python users**:
- Existing `pp.simulate(...)`, `pulsim.sweep.sweep(...)`,
  `pulsim.sweep.monte_carlo(...)` all work unchanged
- New `sweep_path_aware` + `monte_carlo_path_aware` are
  documented as the recommended path; old APIs remain available

**For paper-bound benchmarks**:
- `RANK1_RESULTS.md` becomes one of three sibling writeups
  (single-bit, multi-bit, parametric) under
  `artigos/02_tpel_methods/benchmarks/`. The original file
  stays valid; the new files cover the new use cases.
- TPEL paper §VI table grows from 8 rows × 3 columns to 8 rows
  × 5 columns (adds multi-bit and parametric columns). Source
  for the additional data is the new CSVs.

**Rollback**: revert the `solve_rank1` Hamming-gate to the
legacy "any multi-bit → factorize" behaviour. The
`partial_refactor` machinery remains unchanged (still handles
the union case internally), and the new
`refactor_parametric` API is dead code if its callers are
disabled. Single-commit revert; no data loss.

## Open Questions

1. **Should `refactor_parametric` accept a callback for the
   parameter update?** (e.g.
   `cache.refactor_parametric_via(lambda pool: pool.update_inductor("L_out", 101e-6))`)
   More flexible — handles any pool mutation, not just
   single-name-single-value. But adds another API form. Lean
   towards "no" for v1.5.0; revisit if users ask.

2. **How does the new framework interact with the
   PWL-cache rebuild on `dt` change?** Today `build_lazy(dt2)`
   wipes `segments_`. Should it preserve the etree + RCM (which
   don't depend on `dt`) and only rebuild the numeric factor?
   That's a separate optimisation worth ~30 µs per `(dt change,
   mask)` pair. Track as a follow-up if benchmark data shows it
   matters; otherwise skip.

3. **For multi-bit, is `MAX_PATH_LENGTH_RATIO = 0.6` validated
   for non-Apple-Silicon hardware?** Per-hardware crossover
   point may vary by ~20% (cache hierarchy + branch-predictor
   sensitivity). Benchmark §7.1 only captures macOS data. The
   CI sanitiser run + Linux GCC run will exercise the codepath
   for correctness but won't measure wall-clock. Track an
   x86-64 capture as a TPEL paper supplementary item.
