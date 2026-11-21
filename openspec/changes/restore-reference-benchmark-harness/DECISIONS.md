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
    | `test_shipped_yaml_examples.py` | `elapsed < 20 / 60 s` | rows: the end-time and uniform-`dt` asserts force exactly `MAX_STEPS + 1` (41794994) |
    | `test_steady_state.py` brute force | `t_ss < 1 s` | `n_period_runs < 20` (27 under the regression below; 41794994) |
    | `test_subsystems.py` build growth | time ratio `< 24`, `< 0.05 s` | REMOVED (41794994): the growth is printed only. The C++ builder exposes no operation counter, so linearity of `add_*` / `branch_id_of` is unguarded by any assertion; restoring it needs a C++-side counter or test (debt, not this change) |
    | `test_doc_snippets.py` runtime budget | `total wall <= 60 s` | THIS change: rows returned by `pulsim.simulate` in all executed docs `<= 2,900,000` (2,175,414 measured, 33 % margin); wall time printed per doc. Limit: only top-level `pulsim.simulate` is counted; a snippet using `store_every`, `sweep`, `steady_state` or another helper costs work the count does not see |
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

14. **v1 reader unit tests (task 1.2): `python/tests/test_v1_reader.py`.**
    One hand-checked circuit per mapped type (resistor + dc source: the
    bridge re-derived from the file's own values, 6.6667 / 4.0000 V within
    1e-6; capacitor and inductor with `ic`; snubber_rc; sine, pulse and pwm
    sources; current source; binary diode both ways; Shockley diode (KCL root
    by bisection); static switch open / closed; vcswitch under a pwm control
    node; mosfet; igbt; ideal transformer; coupled inductor; pi_controller →
    pwm_generator loop) and six refusals by name (saturable_inductor,
    `is_nmos: false`, mosfet loss block, `pmos`, `lookup_table`, a PI that
    drives no switch). 24 pass, 1 strict xfail. Two engine facts set the
    tolerances, neither is the reader's: the first stored sample of an
    algebraic node is unsolved (OQ8, excluded), and a stored state starts with
    zero history current, which lags the fixed-engine trace by dt/2 (seen as
    exactly 5·e^{-dt/2τ} at the first step), so a transient truth is compared
    within scale·dt/τ. Findings against the v1 kernel (git 990995e4):
    - **R1 (debt, not fixed): MOSFET K.** The reader passes `K = kp`; v1's law
      is `id = kp (vov vds − vds²/2)(1 + λ vds)` (`components/mosfet.hpp:77-83`),
      i.e. 2.0's `K = kp/2`. `mosfet_nmos_dc` reads ≈ 0.025 V where the v1 law
      gives 0.0502 V (the file's own comment, re-derived). Affects
      `mosfet_nmos_dc`, `buck_mosfet_nonlinear`, `ll08_buck_mosfet`. Not fixed
      here: it moves two goldens, one counter cell, the ll08 triage `known_err`
      and the reference / scoreboard reports — a number-changing integration
      item. `test_nmos_triode_dc_point_is_the_v1_law` is a strict xfail that
      turns red when it is fixed.
    - R2: v1's IGBT is `ic = g_on·vce` when on — `v_ce_sat` cancels
      (`components/igbt.hpp:63-66`). The reader keeps the file's stated drop
      (`add_igbt_level1`, `V_CE_sat`), as decision 7 kept the files' pwm phase
      intent; ngspice is the arbiter (`buck_igbt` only).
    - R3: v1's parser ignored a diode's `is` / `n` (always binary,
      `yaml_parser.cpp:2479`); the reader maps them to the Shockley diode (12
      diodes in 6 files).
    - R4: v1's parser read a switch's `initial_state` only; all 11 files write
      `initial_closed`, so v1 ran them open. The reader honours `initial_closed`.
    - R5: a mosfet's `g_off` (3 files) is dropped: the Level-1 law has none.
    - Defaults aligned with v1 (`kp` 0.1, `lambda` 0.01, igbt `g_on` 1e4);
      every file sets these keys, so no number moves.
    - The unit tests use non-default values for every key they exercise
      (diode 50 / 1e-6 S, switch 1e4 / 1e-8 S, vcswitch threshold 4 V on a
      ramped control so the threshold sets the on window, Shockley n 1.5,
      igbt V_T 4 V with the gate 0.3 V above it): a reader that dropped the key
      would fail (review finding).
    - Decision 7 is superseded: the reader reads a pwm `phase` as a fraction of
      the period (module docstring, batch 6b), not radians.

15. **"The benchmark runner is broken on 2.0" / "the YAMLs do not load on
    2.0" — the facts (named debt BR-1, now closed except as listed).**
    - The claim "`benchmark_runner.py` imports the deleted
      `pulsim_python_backend`" is true of `origin/main` only (lines 22-24 there).
      On `feat/core-integration` (77ce4a28) and on
      `origin/feat/core-soft-switching-validation` the module imports nothing
      of the kind (this change ported it to `v1_reader` + `pulsim.simulate`,
      task 1.4); `test_benchmark_runner_imports_without_the_retired_backend`
      pins it.
    - What WAS broken on 77ce4a28, command
      `PYTHONPATH=python python3 benchmarks/tools/benchmark_runner.py --quiet
      --output-dir .scratch/bench_out`: (a) the whole run died with
      `ImportError: attempted relative import with no known parent package` at
      `from .kpi import (...)` — the first circuit with a `kpi:` block, because
      `benchmarks/` (home of the `kpi` package) is not on `sys.path` when the
      runner is a script (it is under pytest); (b) every scenario compared the
      unsolved t = 0 sample of algebraic nodes (`bridge_resistive_pure` failed
      with max_error 6.67 = V(a)); (c) the v1 solver scenarios
      (`gmres_trbdf2`, `jfnk_trbdf2`, `trbdf2`, `direct_bdf1`, `rosenbrockw`)
      ran the pwl default under their own names (rc_step: three identical
      errors). Fixed: `benchmarks/` on `sys.path`; t0 dropped from the
      comparison (OQ8; it also hides a t = 0 error of a state observable);
      `v2_refusal()` skips a benchmarks.yaml SCENARIO whose override selects
      another integrator (30 of 120 scenario rows), an iterative solver or
      Newton-Krylov, as `refused: …`; an unmappable component or periodic
      mode is `skipped` + `refused: <name>` instead of `failed`. The refusal
      is applied in `run_benchmarks` to the scenario override only — not to a
      file's own `integrator:` (29 files say bdf1; they still run as pwl, as
      before) and not inside `run_pulsim`, which `benchmark_ngspice.py` and
      `local_limit_suite.py` call directly (a first version refused there and
      the reviewer measured 29 ngspice parity rows turning `failed`). Debt
      BR-2: the ngspice nightly therefore still runs `gmres_trbdf2` /
      `trbdf2` rows as the pwl default under those names. The switching-loss
      KPI call site (`window_s` on the soft-switching branch) is untouched.
      The `sys.path` fix (a) is not covered by a test (the pytest run already
      has `benchmarks/` on the path); it was seen working by the CLI run below.
    - After the fix, the default manifest end to end (120 scenario rows):
      47 passed, 47 skipped (`refused:` — 30 solver scenarios, the rest
      unmappable components / periodic modes), 26 failed. Of the failures: 2 baselines missing
      (`current_source_rc`, `pulse_voltage_rl`), ~20 `max_error` against v1
      CSV baselines on switching circuits (10–48 V: a pointwise max of a
      waveform whose edges moved by a step is the full swing — the reason the
      harness grades time-weighted errors), 1 `run_transient` failure
      (`buck_igbt`, step at 7.8 µs not taken in 64 sub-steps), 1
      `observable 'out'` (`bode_rlc_low_pass`). Many "passed" rows have no validation block
      (status = it ran). Test:
      `test_benchmark_harness.py::test_benchmark_runner_runs_v1_files_against_their_stored_expectations`
      (rc_step analytic, bridge baseline CSV, transformer_step_up; two refused
      scenarios; a refused component). It failed before the t0 fix (bridge
      6.67 > 1e-6).
    - **Gate drive in a 2.0 YAML (claim "a 2.0 YAML file has no way to
      specify the gate drive")** — `core/include/pulsim/yaml/loader.hpp`:
      the top-level keys are only `circuit` and `simulation` (loader.hpp:38,
      1968). The PWL switch types `switch`, `mosfet`,
      `mosfet_with_body_diode`, `igbt` (loader.hpp:955-1000) take two power
      terminals and no gate, schedule or controller key: their state comes
      only from `simulate(switch_fn=…)` (a `SwitchSchedule` /
      `make_pwm_switch_fn` / a bound controller) in Python —
      `test_shipped_yaml_examples.py` runs every shipped YAML with all gates
      OFF for that reason. TRUE for those. NOT true for the gate-node devices:
      `mosfet_level1` (loader.hpp:596), `igbt_level1` (:757) and
      `power_mosfet` (:1372) take a `gate` node, and `pwm_voltage_source`
      (:565) / `pulse_voltage_source` (:584) can drive it — an open-loop PWM
      expressible in YAML, at the price of a Newton device. No closed-loop
      controller (PI / PWM block) exists in the YAML schema; the `chain:`
      wiring (`python/pulsim/yaml_chain.py`) covers machines and observers,
      not gate drive.

16. **Task 3.2 reworded: cause ids, not issues.** No GitHub issue is filed
    from this change (outward-facing). Each of the 28 `known:` entries of
    `v2_triage.yaml` carries `cause: [id, ...]`; the ids are the bullets of
    KNOWN_LIMITATIONS.md "Triage causes": TC-EDGE-GRID (10 circuits),
    TC-CLOSED-LOOP (4), TC-PWL-ORDER (9), TC-GATE-NODE (ll08, with
    TC-PWL-ORDER), TC-OPEN (5: transformer_leakage, ll06_buck_vcswitch,
    high_q_rlc_ringdown, ll11, ll13). The review moved two: `ll06` (edge
    error 0.0211 > 0.02, edges ~99 steps apart: not edge timing) to TC-OPEN,
    `boost_pfc_open_loop` (chatter on pwl, `pwl+split` agrees 1.06e-5) to
    TC-PWL-ORDER, whose definition now names restart OR split;
    `t_type_3level_halfbridge` stays TC-EDGE-GRID (REFERENCE_STATUS edge
    diagnostic: edge err 8.0e-4 / 1.4e-6, timing ≤ 1 step). The assignment follows the evidence already in the manifest
    comments and REFERENCE_STATUS.md: TC-PWL-ORDER where the `pwl` cell
    disagrees and the `pwl+restart` cell agrees or is much smaller;
    TC-EDGE-GRID where the edge diagnostic is within tolerance with timing
    ≤ 2 steps (the "late-gate-edge class"). TC-OPEN is a stated
    "not isolated", not a cause. The test fails on a known cell without a
    cause, an id that does not resolve, and an orphan id.
