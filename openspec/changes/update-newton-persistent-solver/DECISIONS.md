# Decisions taken while implementing (stage A1; stage 2: A2, B)

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

## Stage 2 (2026-11-13)

Decided by the maintainer on 2026-11-13: take the rounding-changing
stages now, behind these gates, re-baselining the Newton goldens once.
Gates: Newton iteration count of every call identical to the stage-1
build on every Newton case of the golden probe; step counts and
switch-state sequences identical; states within 1e-9 relative (per
variable, scaled by its range over the run) on the cases whose rounding
may change, `np.array_equal` elsewhere; the absolute dx rule and the DC
cascade untouched.

11. **Counters on the result** → `SimulationResult.newton`
    (`solver::NewtonCounters`: the five workspace counters,
    `partial_refactors`, and `iterations_per_solve`, one `uint16` per
    Newton call). The engines move the workspace's counters in at the
    end of the run (one line in each of `run_transient.hpp` and
    `trbdf2_transient.hpp`). → The gate needs per-call iteration counts,
    and a `uint16` per call is small next to the stored states. → Drop
    the field and the two lines.

12. **`J_nl` keeps its pattern (task 2.3)** → `sparse::zero_keep_pattern`
    (values set to `+0`, pattern kept when compressed) replaces
    `J_nl.setZero()` in the diode, MOSFET/IGBT Level-1 and saturable
    refreshers, in the five refresh wrappers of each engine (ten one-line
    hunks in `run_transient.hpp` / `trbdf2_transient.hpp`), and at the
    start of each Newton call (`J_nl` is resized only when `n` changes).
    Bit-identical: a stamp `+= v` on a kept `+0` gives the value a fresh
    insert gives, the positions are unconditional, and the union pattern
    of `J` is the same. Golden compare against the pre-stage-2 goldens:
    ALL IDENTICAL, with the Eigen backend still in place. A refresher that
    stamps nothing on a call leaves explicit zeros (KNOWN_LIMITATIONS).

13. **Out-parameter overload (task 2.4)** → `solve_with_newton_in_place(
    ..., Vector& x, ...)`; `solve_with_newton_b_extra` copies `x_init`
    and delegates. The engines keep the by-value form, which leaves
    `x_init` untouched on a throw (their failure handling was written
    against that; the in-place form leaves the last iterate). → The copy
    is one per call, not per iteration, and their call sites are other
    tracks' files.

14. **Allocations (tasks 3.6, 4.4)** → zero. `test_zero_alloc` "Newton
    iteration on a run workspace is allocation-free": 200 calls (> 400
    iterations) on a diode buck after three warm-up calls, operator new
    0 and the Eigen malloc guard silent (was 20.4 Eigen mallocs per
    iteration with A1). Needed besides the backend and task 2.3: the
    call-local workspace and the "compress a caller's uncompressed
    `seg.J`" copy are now built only when used (an Eigen sparse matrix
    allocates on default construction), and `partial_refactor` reuses
    the factorize work vectors. The run's `iterations_per_solve` grows
    amortised (the test reserves it).

15. **Gate A2 → passed, and bit-identical.** Golden probe `gate` mode
    against goldens re-recorded from the stage-1 numerics with the
    per-call trace (themselves `compare` ALL IDENTICAL to the stage-1
    goldens):

    | Case | iterations per call | largest relative state difference |
    |---|---|---|
    | Shockley buck, pwl / trbdf2 | identical (27540 / 31087) | 0 / 0 |
    | Shockley rectifier n = 51, pwl / trbdf2 | identical (11521 / 8012) | 0 / 0 |
    | Shockley rectifier n = 483, pwl | identical (1265) | 0 |
    | MOSFET Level-1, pwl | identical (13054) | 0 |
    | saturable inductor, pwl | identical (65645) | 0 |
    | the 12 cases without Newton (incl. the `mosl1_trbdf2` refusal) | — | byte-identical |

    The in-house LU (COLAMD, partial pivoting, ties to the diagonal)
    chooses the same pivots as `Eigen::SparseLU` on these Jacobians and
    forms each factor entry with the same operations, so nothing changed
    and no golden needed re-baselining. The C++ test on a two-diode buck
    (200 calls, both masks) also shows identical iteration counts and a
    relative difference below 1e-9. This is a property of these
    circuits, not a guarantee: a Jacobian where the two pivot rules
    disagree will round differently, which is why the stage stays listed
    as a numerical change.

16. **Acceptance 4.5** → met. ns/step, interleaved stage-1 build vs A2,
    3 rounds × best of 3 (`.scratch/ab.sh`, shared machine):

    | Case | pre-track baseline | stage 1 (A1) | A2 | A2 ÷ baseline | target |
    |---|---|---|---|---|---|
    | Shockley buck, pwl | 6034 | 3478 | 859 | 0.14 | ≤ 0.3 ✓ |
    | Shockley buck, trbdf2 | 20338 | 11410 | 3186 | 0.16 | — |
    | Shockley rectifier n = 51, pwl | 22355 | 12153 | 4860 | 0.22 | — |
    | Shockley rectifier n = 483, pwl | 245789 | 134789 | 59458 | 0.24 | (n = 1503: ≤ 0.6) |
    | MOSFET Level-1, pwl | 6657 | 3824 | 989 | 0.15 | — |
    | saturable inductor, pwl | 6487 | 3555 | 834 | 0.13 | — |
    | buck / transformer (no Newton) | 152 / 168 | 160 / 169 | 152 / 171 | ≈ 1 | no regression ✓ |

    Most of the gain over A1 is the allocation and supernodal-setup
    overhead of `Eigen::SparseLU::factorize` on small matrices.

17. **Stage B (low-rank update)** → measured, **not landed**. Decided by
    the maintainer on 2026-11-13: take the rounding-changing stages now,
    behind these gates, re-baselining the Newton goldens once. Neither
    variant passes both the gate and the speed bar (≥ 1.3× over A2 on
    the small and the large case), so tasks 5.x stay open. The
    prototype is kept as `probes/stage_b_lowrank_prototype.patch`
    (applies to A2's `pwl/nonlinear_solve.hpp`; variant chosen by the
    env var `PULSIM_NEWTON_LOWRANK=p|w`).

    - *Partial* (`p`): `partial_refactor` on J_nl's columns while the
      linear values are unchanged, behind the 0.6·n path gate; full
      factorisation otherwise.
    - *Woodbury* (`w`): a cached in-house factor of J_lin,
      `W = J_lin⁻¹·E_R`, and a k×k dense solve per iteration.

    Gate against the stage-1 goldens with the Newton trace:

    | Case | partial: trace / max rel | partial: refactors | Woodbury: trace / max rel |
    |---|---|---|---|
    | Shockley buck, pwl | identical / 3.4e-16 | 27209 of 27540 | **differs** (27540 → 27524) / 1.5e-12 |
    | Shockley buck, trbdf2 | **differs** (31087 → 31086) / 1.5e-10 | 26984 of 31086 | **aborts** (residual 1.3e17 at h_min) |
    | Shockley rectifier n = 51, pwl | identical / 0 | 0 (fell back every call) | identical / 3.7e-13 |
    | Shockley rectifier n = 51, trbdf2 | identical / 0 | 0 | identical / 1.5e-10 |
    | Shockley rectifier n = 483, pwl | identical / 0 | 0 | identical / 4.6e-14 |
    | MOSFET Level-1, pwl | identical / 1.5e-15 | 13053 of 13054 | identical / 4.7e-15 |
    | saturable inductor, pwl | identical / 0 | 65644 of 65645 | **aborts** (no convergence at t = 1e-7) |

    ns/step, interleaved (3 rounds × best of 3, full-power mode, re-measured
    with the stage-1 build in the same rounds):

    | Case | stage 1 | A2 | partial | A2 ÷ partial | Woodbury (own A/B) | A2 ÷ Woodbury |
    |---|---|---|---|---|---|---|
    | Shockley buck, pwl | 4062 | 1029 | 891 | 1.15 | 866 (A2 959) | 1.11 |
    | Shockley buck, trbdf2 | 15230 | 3643 | 3899 | 0.93 | — | — |
    | Shockley rectifier n = 51, pwl | 17156 | 6137 | 6735 | 0.91 | 3203 (A2 5481) | 1.71 |
    | Shockley rectifier n = 483, pwl | 178613 | 89079 | 89319 | 1.00 | 43434 (A2 81128) | 1.87 |
    | MOSFET Level-1, pwl | 5381 | 1013 | 803 | 1.26 | 1078 (A2 1002) | 0.93 |
    | saturable inductor, pwl | 5049 | 1006 | 897 | 1.12 | — | — |

    Reading: on the small circuits A2 already factors a 10–20 unknown
    matrix in well under a microsecond, so skipping the linear columns
    saves at most ~15 %. On the rectifier the diode columns span nearly
    every node, the partial path never runs (the prototype does not
    split its fallbacks by cause; the 0.6·n path gate is the likely
    refusal), and Woodbury's rank equals the diode count. Woodbury is
    the only variant with a large-case gain (1.87× at n = 483), but it
    changes the iteration trace on the buck and diverges on the buck
    under TR-BDF2 and on the saturable inductor: `I + W[C,:]·Ñ` is
    ill-conditioned when a device's conductance spans many decades,
    and without a conditioning test and a fallback it is unsafe.
    What would make B worth revisiting: a Woodbury guarded by a
    condition estimate of the k×k capacitance matrix with fallback to
    the full factor, used only above a size threshold (n ≳ 200); the
    small-circuit cost is now the residual/refresh, not the factor.
