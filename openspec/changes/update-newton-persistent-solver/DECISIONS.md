# Decisions taken while implementing (stage A1 only)

Format: question → option taken → why → how to reverse.

1. **Scope** → stage A1 only (workspace, in-place Jacobian, persistent
   analysis). A2 (in-house LU, needs the owner's tolerance sign-off) and
   B (low-rank update, after A2) are not started.

2. **Hard gate (task 3.2)** → passed, so A1 is implemented. Measured
   twice:
   - Synthetic (`probes/gate_sparselu.cpp`, build line in its header):
     `Eigen::SparseLU` (COLAMD, the backend
     `make_default_solver()` gives) on random patterns n = 50 / 400 /
     3000, diagonal weights 10 / 1 / 0.05, 4 seeds, 8 value sets per
     pattern: 288 factorisations through one reused `analyzePattern`
     against a fresh object each — solutions, row and column
     permutations byte-equal, 0 differences.
   - End to end: the golden probe's Newton cases (Shockley buck,
     Shockley rectifier n = 51 and 483, MOSFET Level-1, saturable
     inductor, transformer) `np.array_equal` before → after, and the
     C++ test "a run workspace reuses one analysis and gives the bytes
     of the fresh path".
   Eigen's `factorize` reinitialises its row permutation and work
   memory on every call, and the column order and elimination tree
   depend only on the pattern, which is why the reuse is exact.

3. **Zero-keep-pattern helper for `J_nl` (task 2.3)** → not done. The
   refreshers call `setZero()` on `J_nl` and re-insert; keeping the
   pattern would mean changing every refresher and the engine wrappers
   (`nonlinear_refresh_*.hpp`, the wrapper chain in
   `run_transient.hpp`), most of them outside this track's files, and
   it does not change the analysis count (the pattern check sees the
   same index arrays either way). → Its cost stays: the refresher's
   inserts allocate each iteration (counted in the 20.4 Eigen mallocs
   per iteration below). → Add `zero_keep_pattern(J_nl)` and use
   `coeffRef` in the refreshers.

4. **Out-parameter overload for `x` (task 2.4, last item)** → not
   added; `x` is still returned by value (one copy per call, not per
   iteration). → No caller asked for it and the engines assign the
   result straight back. → Add `void solve_with_newton_b_extra(...,
   Vector& x)`.

5. **Call without a workspace** → a call-local workspace, so every
   caller (DC paths, continuation, sweeps) gets the in-place Jacobian
   and the analysis reused across the iterations of that call. → One
   code path; the gate shows the reuse is exact. → Keep the old body
   behind `ws == nullptr`.

6. **Counters (task 1.2)** → `NewtonWorkspace::{solves, iterations,
   analyses, factorizations, lm_attempts}`, read by the C++ tests. Not
   surfaced in `SimulationResult` or the Python bindings: that needs
   new fields in `solver/result.hpp`, `python/bindings.cpp` and the
   engines' result filling, all other tracks' files, for counters no
   user asked for. Iteration-count equality (task 3.4) is shown by the
   byte-equal states (a different iteration count changes the bits)
   and by the C++ test. → Add `newton_*` fields to the result stats and
   copy the workspace counters at the end of each run.

7. **Allocation test (task 3.6)** → not met, measured instead with the
   probe (`operator new` + Eigen malloc guard, Shockley-diode buck, one
   43-iteration call): Eigen mallocs per iteration 47.7 → 20.4,
   `operator new` per call 15 → 9. The remainder is
   `Eigen::SparseLU::factorize` (`memInit` on every call) plus the
   refreshers' inserts (decision 3); separating the two needs an
   allocator hook inside Eigen. → Task 3.6 stays open.

8. **LM path** → unchanged apart from the `lm_attempts` counter: it
   still builds and analyses a fresh solver per attempt (λ·I changes
   the diagonal pattern). Recorded in KNOWN_LIMITATIONS.

9. **Workspace failure state** → if `analyze` fails the workspace drops
   its solver before throwing, so the next call analyses again; a failed
   `factorize` keeps the analysis (the pattern is unchanged and the
   next factorisation is a pure function of the values).

10. **Acceptance 3.7 missed on the Shockley buck.** ns/step, three
    interleaved runs of the pre-track build, the build before A1 and
    A1 (best of each):

    | Case | Baseline | Before A1 | A1 | A1 ÷ baseline | Target |
    |---|---|---|---|---|---|
    | Shockley buck, pwl | 6034 | 5916 | 3465 | 0.57 | ≤ 0.5 ✗ |
    | Shockley buck, trbdf2 | 20338 | 19535 | 11594 | 0.57 | — |
    | Shockley rectifier n = 51, pwl | 22355 | 21989 | 12237 | 0.55 | — |
    | Shockley rectifier n = 483, pwl | 245789 | 239615 | 137618 | 0.56 | (n = 1503: ≤ 0.75) |
    | MOSFET Level-1, pwl | 6657 | 6527 | 3891 | 0.58 | — |
    | saturable inductor, pwl | 6487 | 6476 | 3549 | 0.55 | — |
    | buck / transformer (no Newton) | 152 / 168 | 153 / 175 | 156 / 171 | ≈ 1 | no regression |

    The remaining per-iteration cost is the numeric `SparseLU`
    factorisation (it re-initialises its memory each call) and the
    refreshers' re-insert of `J_nl` (decision 3). → Report; A2 (the
    in-house LU with an allocation-free `factorize`) is the proposal's
    next lever. The n = 1503 rectifier is not in the golden set; n = 483
    stands in for it.
