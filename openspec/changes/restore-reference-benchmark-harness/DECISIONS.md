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
