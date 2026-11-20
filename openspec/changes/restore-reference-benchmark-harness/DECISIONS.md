# Decisions taken during implementation (2026-11-13)

The design's Open Questions were unanswered; each takes the option marked
"proposed", else the most conservative one.

| # | Question | Option taken | Why | How to reverse |
|---|---|---|---|---|
| OQ1 | Reader in the harness or rewrite the 126 files; rename the header? | Reader in the harness (`benchmarks/tools/v1_reader.py`); header `schema: pulsim-v1` kept | Proposed; renaming touches 126 files and their ngspice pairing for no behaviour | Change the `schema` check in `v1_reader.load` together with a bulk rename |
| OQ2 | Map `pi_controller` → `pwm_generator` now? | Mapped through `bind_pi_to_switch` (5 circuits run) | Proposed. The PI is updated once per PWM period; v1 updated it every step, so these are not v1-equivalent and none is pinned by a golden | Raise `Unmapped` for `pi_controller` in `_build` |
| OQ3 | The 70 legacy v1 baselines | Kept as the informational "drift from v1" column of `benchmark_runner.py` | Proposed | Delete `benchmarks/baselines/` and the `validation.baseline` lookups |
| OQ4 | v1 scenario names in `benchmarks.yaml` | File kept unchanged as history; the reference harness uses the triage manifest and the engine list | The conservative half of the proposed option: `benchmark_runner.py` / `benchmark_ngspice.py` select scenarios by those names, so replacing them breaks both | Rewrite the `scenarios:` block and the two runners |
| OQ5 | KPI tools and `kpi_baselines/` | Neither retired nor ported | No option marked proposed; leaving them untouched is the conservative choice | — |
| OQ6 | Golden platform and caps | Generated on macOS arm64; caps 1e-9 (linear) / 1e-6 (Newton) used as the initial `rtol` | Proposed. Task 5.3 (set each `rtol` from the three CI platforms) is open: the `reference-smoke` job is non-blocking until it has run there | Lower `rtol` per golden after the first three-platform run |
| OQ7 | Nightly failure policy; pin ngspice | One issue opened / commented (by title); ngspice not pinned, its version written to the artifact | Proposed | Edit `.github/workflows/nightly-reference.yml` |
| OQ8 | First stored sample of algebraic nodes | Documented (`KNOWN_LIMITATIONS.md`) and excluded from comparisons (`exclude_t0`) | Proposed | Drop `exclude_t0` once consistent initial conditions land |

Implementation choices not covered by the design:

1. **No kernel `RunCounters` yet (tasks 4.1, 4.2 open).** The change had to
   land without a C++ rebuild on a shared machine. The counter gate uses
   the counters the kernel already reports: `pwl` step / retry / event
   iteration / commutation counts and the PWL cache metrics (via
   `on_cache`), the `trbdf2` stats dict, the `dsed` accept / reject /
   event counts. Factorisation and Newton-iteration totals of the fixed
   and Newton paths are still missing. Reverse / complete: add
   `RunCounters` to `solver/result.hpp` and extend `counters.count`.
2. **dsed is gated with `integrator='rk45'`** and its analytic ceilings
   are floored at 1e-6: `fix-dsed-auto-integrator-selection` changes what
   `integrator='auto'` does, so nothing here pins it.
3. **Analytic cases live in one module** (`benchmarks/analytic/__init__.py`,
   a `CASES` registry) instead of 20 files: one import, one table.
4. **Ceiling policy is checked in the file** (`ceiling ≤ 2 × measured`
   with the floors, else a `reason`) plus a comparison with
   `git show HEAD:` when git is available. A pull-request CI checkout is
   a merge commit whose HEAD already contains the change, so the in-file
   rule is the one CI can enforce.
5. **Golden `atol` = `rtol` × the state's own scale** (its largest
   magnitude over the 8 checkpoints, at least 1e-6): a roundoff-level
   value in a large-scale state is not a regression.
6. **`uic: false` circuits start from the DC operating point** passed as
   `initial_state`, and only when it gives a capacitor voltage or inductor
   current a non-zero value; otherwise every engine starts at rest, which
   is the same state. `dsed` refuses a full state vector, so those cells
   are `refused`.
7. **v1 pwm `phase` follows the v1 kernel** (radians, advancing the
   carrier; on-time `duty·T − dead_time`, git 990995e4
   `core/include/pulsim/v1/sources.hpp`). Several files look as if the
   author meant a fraction of a period (`phase: 0.5` for two interleaved
   phases); the reader keeps the kernel semantics and the ngspice nightly
   is the arbiter.
8. **Harness tests carry the pytest marker `reference`.** They run in
   plain `pytest python/tests`; the general CI Python job deselects them
   and the `reference-smoke` job runs them on three platforms, non-blocking
   until one week green (design decision 8 / task 6.1).
9. **Two cells are recorded as `known` analytic defects instead of a
   ceiling:** `hard_switched_rc` on `pwl` (trapezoidal ringing, max error
   100 % of span with `tau_on = 1 ps ≪ dt`) and on `dsed` (rms 0.44), and
   `halfwave_rl` on `dsed` (diode configuration does not settle). The test
   asserts they still fail, so a fix surfaces as a failure asking for a
   ceiling. A ceiling at 2 × a wrong answer would certify it.
10. **Disagreements carry no issue numbers yet (task 3.2 open).** The 76
    `disagree` / `error` cells are `known:` entries in the triage manifest;
    nothing was posted to GitHub from this session.
11. **Wall-clock assertions (task 4.4 open), inventory:**
    `test_dsed_diode_events.py:169` (`elapsed < 5 s`),
    `test_dsed_stiff_sine.py:71` (`elapsed < 5 s`),
    `test_dsed_without_scipy.py:173` (`elapsed < 5 s`) and `:225`
    (speed-up ratio `> 2`), `test_sampled_ac.py:124` (time ratio `< 8`),
    `test_shipped_yaml_examples.py:94` (`elapsed < budget`),
    `test_steady_state.py:112` (`t_ss < 1 s`),
    `test_subsystems.py:294,299` (ratio `< 24`, `< 0.05 s`; being replaced
    by another change), `test_doc_snippets.py:375` (runtime budget). Not
    converted here: each needs its own count, and some files are being
    edited elsewhere.

## Close-out (2026-11-20, branch feat/core-benchmark-harness-closeout over 77ce4a28)

12. **Wall-clock assertions (task 4.4), every one of the inventory in 11.**
    Re-inventoried with `grep -nE 'perf_counter|time\.time\(\)|monotonic|elapsed|timeit'`
    over `python/tests` on 77ce4a28: nine of the ten were already converted
    by 41794994 ("test: no test asserts wall time"); one was still live.

    | site | old assertion | now |
    |---|---|---|
    | `test_dsed_diode_events.py` DCM buck | `elapsed < 5 s` | `rd.n_accept < 100_000` (10,158 measured; 41794994) |
    | `test_dsed_stiff_sine.py` 3 cases | `elapsed < 5 s` | `rd.n_accept < 100_000` (~50.1k measured; 41794994) |
    | `test_dsed_without_scipy.py` 5 ms buck | `elapsed < 5 s` | native result type + `0 < n_accept <= 10_000` (41794994) |
    | `test_dsed_without_scipy.py` speed-up | wall ratio `> 2` | step ratio `5 · n_accept(dsed) <= n_steps(pwl)` (41794994) |
    | `test_sampled_ac.py` cost vs points | time ratio `< 8` (later `< 2`) | `n_period_runs` equal for 10 and 400 points (41794994) |
    | `test_shipped_yaml_examples.py` | `elapsed < 20 / 60 s` | rows `<= MAX_STEPS + 1` (THIS change: 41794994 only said "asserted above"; it was implied by the grid, now asserted) |
    | `test_steady_state.py` brute force | `t_ss < 1 s` | `n_period_runs < 20` (27 under the regression below; 41794994) |
    | `test_subsystems.py` build growth | time ratio `< 24`, `< 0.05 s` | REMOVED (41794994): the growth is printed only. The C++ builder exposes no operation counter, so linearity of `add_*` / `branch_id_of` is unguarded by any assertion; restoring it needs a C++-side counter or test (debt, not this change) |
    | `test_doc_snippets.py` runtime budget | `total wall <= 60 s` | THIS change: rows returned by `pulsim.simulate` in all executed docs `<= 2,900,000` (2,175,414 measured, 33 % margin); wall time printed per doc |
    | `test_sweep_path_aware.py:269`, `test_design_steady_and_batch.py:159` | `wall_time_s > 0`, `wall_s >= 0` | kept: field-presence checks, cannot fail under load |

    Each converted gate was shown to FAIL on a deliberate regression in a
    scratch copy (`.scratch/regress`, package + tests copied, not committed):
    `steady_state(verify_periods=20)` default (was 8) → `assert 27 < 20`
    fails; `frequency_response` running one extra period per frequency →
    `assert 43 == 433` fails; tutorial 06 at `dt=5e-9` (was 2e-8) →
    `3975414 <= 2900000` fails. The unmodified copies pass.

13. **Tasks 4.1–4.3 reconciled with the tree at 77ce4a28.** Probe: buck_ccm,
    shockley_op, saturable_rl on three engines, printing `res.stats` and
    `res.counters`. `result.counters` exists but is not the design's
    `RunCounters`: it is the pwl fixed-loop dict (`switch_fn_calls`,
    `segment_searches`, `rollback_snapshots`, stage-2 / gate / mode keys;
    zeros on other engines). The engine-independent counters are
    `result.stats` (`record::RunStats`, `core/include/pulsim/record/grid.hpp`):
    `n_steps, n_accept, n_reject, n_solves, n_factorizations,
    n_newton_iterations`, event and integrator counts. Gaps against design 6:
    no `n_symbolic_analyses` / `n_segment_builds` / `n_event_refactors` on the
    result (only `CacheMetrics` through `on_cache`); pwl leaves `stats.n_steps`
    and its linear factorisation total at 0 (buck_ccm: all zero; shockley_op:
    `n_solves 100, n_factorizations 124`); trbdf2 reports `n_factorizations 0`
    on linear buck_ccm with `n_solves 2289`. The counter gate
    (`benchmarks/tools/counters.py`) reads `CacheMetrics` + `len(res.times)` +
    event lists on pwl, `_trbdf2_stats` on trbdf2, the dsed fields — not
    `res.stats`. 4.1 / 4.2 stay open (kernel work); 4.3 is met but for the
    scratch-build proof of the extra factorisation.
