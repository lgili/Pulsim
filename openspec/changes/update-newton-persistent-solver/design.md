## Context

`solve_with_newton_b_extra` (`pwl/nonlinear_solve.hpp:71`) is a free
function. It is called once per event iteration by the fixed engine
(`run_transient.hpp:989`, `:1008`, `:1647`), once per stage by TR-BDF2
(`trbdf2_transient.hpp:716`), and by the DC paths (`dc_strategy.hpp`,
`dc_assemble.hpp`, `continuation.hpp`). It owns nothing between calls,
so each iteration rebuilds everything: the merged matrix, the vectors
and a complete `Eigen::SparseLU` (analysis + factorisation).

The linear path is different: each cached segment holds an in-house
`PulsimSparseLuSolver` (`pwl/cache.hpp:1664`). The Newton path therefore
uses another factorisation than the rest of the kernel.

Constraints: the DC cascade and its absolute `dx` rule are not touched;
device laws are not touched; thread safety of concurrent GIL-released
runs must hold (review finding ZA-1: no shared mutable workspace on a
cache or solver instance).

## Goals / Non-Goals

- Goals: one symbolic analysis per pattern per run; no merged-matrix
  copy and no vector allocation per iteration; counters for iterations,
  analyses and factorisations; a bit-identical first stage.
- Non-Goals: chord Newton, a rate monitor, a different line search or
  predictor (`raw/numerics.md` item 12); the roundoff acceptance rule
  (`add-newton-roundoff-acceptance`); the device interface
  (`add-unified-device-interface`); the DC paths; fewer Newton
  iterations.

## Decisions

- **Workspace owned by the run, passed by pointer.**
  `solve_with_newton_b_extra` gains a trailing `NewtonWorkspace* ws =
  nullptr`. `nullptr` keeps today's code path, so the DC callers are
  unchanged. Each engine loop creates one workspace on its stack.
  - Alternative: a thread-local workspace. Rejected: its lifetime is
    the thread, not the run, so a stale analysis could outlive a
    circuit.
  - Alternative: store it on `PwlSegment` or the cache. Rejected:
    caches are shared between concurrent runs.
- **Combined Jacobian written in place.** The workspace builds the
  union pattern of `seg.J` and `J_nl` once and two slot maps. Per
  iteration, `J[k] = L[k] + N[k]` where `L` and `N` are the two operands
  scattered onto the union pattern with absent entries stored as
  `+0.0`. That is the arithmetic Eigen performs entry by entry for
  `J_combined = seg.J; J_combined += J_nl` (`a + b`, `a + 0`, `0 + b`),
  so values and pattern are identical.
  - Alternative: keep `J_combined += J_nl` on a persistent matrix.
    Rejected: Eigen evaluates the sum into a new matrix.
- **`J_nl` keeps its pattern.** The refreshers reset `J_nl` with
  `setZero()`, which discards the pattern and makes every stamp an
  insert. A helper zeroes the values and keeps the pattern; it replaces
  `setZero()` at exactly the sites that call it today
  (`nonlinear_refresh_mosfet_level1.hpp:68`, `:87`, `:108`,
  `nonlinear_refresh_saturable_inductor.hpp:137`,
  `nonlinear_solve.hpp:529`, and the engine wrappers). The stamps of
  every shipped refresher are unconditional in position (only
  active-node guards), so the pattern after the first refresh is the
  pattern of today.
- **Analysis reuse is guarded by a pattern check, not assumed.** After
  each refresh the workspace compares the index arrays of `seg.J` and
  `J_nl` with the ones the slot maps were built from (pointer and size
  first, `memcmp` on mismatch). A difference rebuilds the maps and
  re-analyses, and increments `newton_analyses`. Invalidation cases:
  static (`dt = 0`) versus dynamic assembly; a user refresh whose stamp
  positions depend on `x`.
- **Stage A1 keeps `Eigen::SparseLU`.** This is the decision that makes
  A1 bit-identical. `analyzePattern` depends only on the pattern;
  `factorize` on the analysis and the values. A fresh object and a
  reused analysis are therefore expected to give the same factors bit
  for bit. Task 3.2 proves this on the Newton goldens before the reuse
  is switched on; if it fails, A1 keeps a fresh analysis per iteration
  and delivers only the copy and allocation removal.
  - Alternative: go straight to the in-house LU (the roadmap wording).
    Rejected as the first step: different pivot rule (in-house: largest
    magnitude, ties to the diagonal; Eigen: supernodal with its own
    threshold), so different rounding, so no bit-identical oracle for
    the refactoring itself.
- **Stage A2 switches the backend through the existing
  `sparse::Backend` hint** on the workspace (`Eigen` → `Pulsim`). No new
  Python option. Gate: iteration counts equal to A1 on every Newton
  golden, states within 1e-9 relative, then the goldens are regenerated.
  A2 needs the allocation-free `factorize` of
  `update-trbdf2-factor-reuse` to reach zero allocations per iteration;
  without it A2 still removes Eigen's internal ones.
- **Stage B prefers `partial_refactor` over Woodbury.** The changed
  columns are the columns `J_nl` touches, a fixed set per circuit.
  `partial_refactor` exists, reuses the pivot order and falls back to a
  full factorisation on a pivot fault. Woodbury on the cached linear
  factor (`Z = J_lin⁻¹·U` cached per segment) avoids touching the
  factor at all but loses accuracy when the device conductance
  dominates the entry it updates (a conducting diode against `g_off`).
  The stage B baseline task measures both on the rectifier ladder
  before committing.
- **LM stays outside the fast path.** The LM branch (`:255-262`) adds a
  diagonal and so has another pattern; it keeps a fresh solver per
  attempt and is counted in `newton_lm_attempts`. It is the recovery
  path, not the profile.
- **Residual evaluation keeps its order.** `f = J_lin·x + b_constant +
  b_extra + f_nl` is evaluated with the product in a hoisted buffer
  first and the three additions in the same left-to-right order.

## Risks / Trade-offs

- Eigen might carry numeric state across `factorize` calls → task 3.2
  is a hard gate, with the fallback named above.
- The wrapper chain of stateful devices: every wrapper must keep
  stamping into the same `J_nl` (owner's note: a wrapper that calls the
  raw refresh deletes earlier stamps). The zero-keep-pattern helper
  must sit at the outermost refresh only → test with a circuit that has
  two stateful device kinds.
- A2 moves the Newton path to a factorisation with another pivot rule
  and a 1e-14 pivot tolerance → the tolerance gate (iteration counts
  equal) catches a conditioning regression; the stall and chatter
  characterisation tests stay green.
- Stage B depends on the pivot order staying valid across iterations →
  `newton_partial_refactors` and the fallback count are reported; a
  fallback ratio above 20 % on the probe circuits stops stage B.
- Estimates are from a time split → each stage has its own measurement
  gate.

## Migration Plan

1. Counters (no behaviour change), baseline.
2. A1 behind the `ws != nullptr` path; engines pass a workspace once
   the goldens are identical. Rollback: pass `nullptr`.
3. A2 after the owner accepts the tolerance gate; goldens of Newton
   circuits regenerated in the same commit. Rollback: backend hint back
   to `Eigen`.
4. Stage B only after A2 is merged and measured.

## Open Questions

- The roadmap lists this change under "bit-identical". Only A1 is. Does
  the owner accept A2 (states within 1e-9 relative, Newton goldens
  regenerated) inside Phase 1, or should Phase 1 stop at A1?
- Is "iteration counts identical and states within 1e-9 relative" the
  right gate for A2 and B, or should it be tied to `tol_newton_dx`?
- Should the LM branch get a persistent solver too, or is counting it
  enough until a profile shows it?
