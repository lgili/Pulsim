# Decisions — add-compiled-control-graph

Taken by the implementer under the owner's delegation (2026-11-13).
Each entry: question → option taken → evidence → how to reverse.

## G1. Where the graph lives

- **Question.** New `core/control/*` headers, or the existing
  `blockchain::BlockChain`?
- **Taken.** Inside `BlockChain` (`core/include/pulsim/blockchain/chain.hpp`),
  as the proposal allows for tasks 1–4 ("ships on today's structure").
  Its Python API (`CxxBlockChain.add_*`, `make_step_observer`,
  `record_channel`, `make_chain_switch_fn`) is unchanged, so the 30+
  Python helpers that build a `CxxBlockChain` (motors, observers,
  thermal, YAML) keep working. The C++ API did change: every
  `make_*_step` factory takes the chain first, `channels()` and
  `register_node_ref` are gone, and `bind_nodes` resolves named node
  inputs of the blocks added after it (no Python caller used it).
- **Evidence.** `chain_bit_identity.py` (probes/): chains A and B
  byte-identical before/after; every chain test in the suite passes.
- **Reverse.** Move `chain.hpp` to `core/control/graph.hpp` when the
  Phase 3 hooks (G-hooks below) land; the class is self-contained.

## G2. Direct-feedthrough table

- **Question.** Which inputs order the schedule?
- **Taken.** Every input of every pre-existing kernel block is direct
  feedthrough, including `Integrator`, `FirstOrderLowPass`, `PI`, `PID`,
  `PLL`, the motors and the observers. `DelayBlock` is the only block
  whose input is not.
- **Evidence.** These kernels compute their output from THIS
  evaluation's input in the same call (`Integrator::update` returns
  `y + gain*x*dt`, `FirstOrderLowPass` returns the updated state), so
  their output does depend on the input in the same instant. The
  design's "Integrator: no" describes a different integrator (output
  = state before the update); marking today's one "no" would let a
  loop through it run with the integrator reading a stale input, a
  silent reordering. A false "yes" can only produce a loop error,
  never a wrong number.
- **Reverse.** Give a block an `update` closure (phase 2) and read its
  input with `chain.in(ref, /*feedthrough=*/false)` — as `DelayBlock`
  does; that changes its numbers (output becomes the pre-update state).

## G3. A signal written by two blocks; a signal no block writes

- **Taken.** Two writers: `ControlGraphBuildError` naming the signal
  and both blocks (the insertion-order chain let the later one win,
  readers in between saw the other). No writer: legal, the slot holds
  0 or what `set_channel` put there (signals fed from Python).
- **Reverse.** Drop the check in `BlockChain::compile`.

## G4. Reporting a reordering

- **Taken.** A `UserWarning` once per build listing every
  consumer/producer pair inserted in the wrong order (each was a
  one-step delay before), plus `CxxBlockChain.reorders()`.
- **Reverse.** Make it an error (strict mode) or silence it.

## G5. Block state stays in each block (no flat `xd` / `xc` yet)

- **Taken.** Signals are flat slots (task 2.1); block state stays in the
  block's POD, owned by its closure.
- **Evidence.** Nothing reads graph state from outside yet: the solver
  snapshot (6.1) and the periodic-steady-state shooting are the first
  consumers, and the deque-based blocks (moving average, delay) do not
  fit fixed slices without changing their numbers.
- **Reverse.** Give each block `xd` / `xc` slices when 6.1 lands.

## G5b. The Python evaluator fallback keeps insertion order (until task 5)

- **Question.** A chain with a block the kernel cannot run (or
  `use_kernel=False`) runs on the Python evaluator, in insertion order
  with the old implicit one-step delays. Schedule it too?
- **Taken.** No: task 5 removes the fallback (an unknown kind becomes a
  build error), so scheduling it would be work thrown away. Until then
  the same chain can give different numbers on the two paths (found by
  the step-3 review).
- **Reverse.** n/a (removed in task 5).

## G6. Which state a tick reads on the fixed engine

- **Question.** The `pwl` kernel calls the observer with `(t_n, x_{n-1})`
  before step n: the label is the step end, the state is its start.
  Which instant is the tick?
- **Taken.** The state's own instant: a graph with rates runs at
  `tx = t_n - dt` (the first call, `(t_start, x0)`, at `t_start`), so a
  tick at τ reads x(τ) and its output drives the step that starts at τ
  (zero-order hold from τ). `dt` is inferred from the second call
  (`run_transient_with_chain` sets it). A chain with no `Rate` keeps the
  old protocol (every call, the label time) bit for bit.
- **Evidence.** The discrete P loop (`test_control_graph_rates.py`)
  lands on the exact sampled-data recursion (the closed form samples
  v(kT) and holds u from kT), second order in dt with the restart rule
  (G9). The label time would sample x(τ - dt) for a tick at τ.
- **Reverse.** `BlockChain::observe_fixed_step`: use `t` for `tx`.

## G7. Rate inheritance and the legacy group

- **Taken.** Group 0 is the legacy every-call group (no `Rate` anywhere:
  unchanged behaviour, `dt` from `make_step_observer(dt)`). With rates:
  a block's own `rate=`, else (stateless kinds only) the fastest rate
  among its producers, else the chain's `rate=`, else
  `ControlGraphBuildError` naming the block. Groups due at the same
  instant run in one pass of the schedule.
- **Evidence.** Tick counts 200 and 40/400 equal the Phase 0 schedule
  on `pwl` and `trbdf2` (characterisation 1.3); `n_block_evals` =
  Σ ticks × blocks per group.
- **Reverse.** `_assign_groups` in `blockchain.py`.

## G8. compute_delay and offsets per engine

- **Taken.** `compute_delay` defaults to 0. `"one_sample"` publishes the
  previous tick's outputs at each tick (all engines). A time delay and
  a group `offset` run on `pwl` (served at the first step at or after
  the instant) and are refused on `trbdf2` (`UnsupportedError`,
  features `compute_delay` / `control_rate_offset`): its tick scheduler
  lands steps at `t_start + k*T` only.
- **Hook needed** (forbidden file `solver/trbdf2_transient.hpp`): the
  observer-period list must take an offset per source and a second
  landing instant per tick (`tick + Td`), i.e. `observer_periods` →
  `{period, offset, extra_landings}`.
- **Reverse.** Drop the two checks in `ControlPlan.for_variable_step`.

## G9. Driven sources: b_extra with declared jumps

- **Taken.** `drive_source` sinks are summed into a `b_extra_fn`
  (`CxxChainBExtra`, C++) merged with the caller's above the engine
  dispatch. It declares its jumps (`next_jump_after`: each group's
  publication instant moved to the step it is served at), so with
  `discontinuity_restart=True` the fixed engine restarts on every tick.
- **Evidence.** Without the restart a ZOH jump is averaged by the
  trapezoidal history over one step, an O(dt) error that is the
  engine's documented first-order-after-a-discontinuity behaviour
  (known-limitations): P loop v[10] off by 6.148e-5 at dt = 1e-6 and
  6.138e-6 at 1e-7. With the rule: 7.2e-9 at 1e-6 and 7.3e-11 at 1e-7
  (second order; trbdf2 7.8e-9 at rtol 1e-10, 3.7e-10 at 1e-12). The
  chain fast path `run_transient_with_chain` dropped the declaration
  before this change (fixed: it reads it as `run_transient` does).
  The restart is not turned on implicitly: it changes every other
  discontinuity of the run too.
- **Reverse.** Return the plain `make_b_extra_fn` (no declaration).

## G10. Which law a ported block runs

- **Question.** Where the kernel already had a block of the same name
  with a different law (`PIDController`, `Integrator`, `Differentiator`,
  `MovingAverageFilter`, `Sum`, `PwmGenerator`, `FirstOrderLowPass`),
  which one does a chain run?
- **Taken.** The Python class's: it is the documented API and the
  Python evaluator ran it whenever the kernel mapping failed, so the
  same chain gave two answers. Ports keep Python's operand order
  (`-ffp-contract=off` is already on) and `min`/`max` argument order;
  `test_control_graph_blocks.py` checks every mapped block byte for
  byte on 10^4 random inputs (incl. dt = 0 and a zero divisor), except
  `SpaceVectorModulator`, the Clarke / Park transforms and the PLL
  (kernel blocks, not yet compared).
- **Evidence.** Before: `Integrator` differed on 9559 of 10^4 inputs
  (up to 2.3e-2), `FirstOrderLowPass` at dt = 0 and its `state=`,
  `PIDController` structurally, `PwmGenerator` read `phase` as a period
  fraction with the opposite sign (interleaved legs collapsed); `PIController`, `Gain`, `Subtract`, `Limiter` were
  already identical. The kernel `PIDController` / `Differentiator` /
  moving-window filter stay for C++ users (`add_pid_controller`, ...).
- **Reverse.** Map the class back to the old `add_*` in
  `_compile_to_cxx`.

## G11. Library blocks implemented in Python

- **Taken.** `SlidingModeObserver` and `FluxMRASObserver` (Python
  classes in `pulsim.observers`) run through the `PythonBlock` adapter
  automatically; user classes must be wrapped explicitly (an unknown
  kind is a build error, so a typo or a forgotten port cannot fall back
  silently). Their kernel twins (`add_sliding_mode_observer`, ...) are
  not proven bit-identical, so they are not substituted.
- **Reverse.** Drop them from `_PY_LIBRARY` in `blockchain.py`.

## G12. Rate-group edge cases (final review)

- **Question.** What a rated chain does when it is reused, resumed,
  under-sampled or holds a PWM.
- **Taken.** Time going back between observer calls (a second
  `simulate()`) restarts the tick grid, keeping block state (as a
  chain without rates always did; before, the groups never ticked
  again). A period below the fixed step and a `PwmGenerator` without
  its own rate in a rated chain are `ControlGraphBuildError`s;
  `controller_period=` with a rated chain is a `ValueError`. With
  `resume_from=` the grid restarts at the resume instant (hook 6.1).
- **Evidence.** `test_control_graph_blocks.py` (reuse, period, PWM,
  controller_period cases).
- **Reverse.** `BlockChain::rearm_if_restarted` / `check_periods`
  (`chain.hpp`), the PWM check in `_assign_groups`, the
  `controller_period` check in `compose_control`.

## G13. Label time for the consolidated helpers (task 9.1)

- **Question.** Moving `bind_pi_to_switch` / `add_c_block` onto the
  graph's state-time ticks (G6) shifts every closed-loop result by one
  step. Keep the old numbers or take the shift?
- **Taken.** A chain-level `label_time` mode reproducing the old
  scheduler exactly (`t >= due - 1e-9*dt`, `due < t_end - 1e-9*dt`,
  `x_{n-1}` at label `t_n`); the helpers build label-time graphs. The
  law is an advance of one step after tick 0 (docs/control-graph.md).
- **Evidence.** `.scratch` probe of 93 arrays (one / two-rate / paired
  loops, callable set-point, manual composition, duty callable,
  non-multiple period, on pwl, pwl+restart and trbdf2): bit-identical
  to 7b195243 except the final sample of C-block runs (G15). Test pins
  `w_10` against the label closed form.
- **Reverse.** Drop `label_time=True` in `_pi_loop_graph` /
  `add_c_block`; every closed-loop number moves by O(dt).

## G14. The helper's `pi` object is not updated

- **Taken.** The kernel PI starts from `pi`'s state; the Python object
  stays as passed. Histories are views of the graph's recordings.
- **Evidence.** No test, example or library caller reads `pi` after a
  run; writing it back would need a Python call per tick.
- **Reverse.** A `PythonBlock(pi)` (kept for non-pulsim controllers).

## G15. C blocks: one graph per builder, no tick at t_end

- **Taken.** One graph per builder holds every `CBlock` (integration batch 10:
  reached through the C-block handles, not a `builder._c_graph` attribute, so
  a `Circuit` built from the builder carries it in its ControlDecls), each in the group
  of its `dt`, compiled at each `simulate()` with the outputs the blocks
  hold. Compiled steps are called through their C ABI in C++. The old
  throttle also fired at `t_end` (11 fires where 10 are due); the graph
  does not: the final sample of such a run changes (6.4e-4 V on the
  probe's RC loop), nothing else.
- **Reverse.** `set_horizon(inf)` for the C-block graph.

## G16. Label-time graphs declare no jumps

- **Taken.** `next_jump_after` = inf in label mode, as the helpers
  declared none: `discontinuity_restart=True` results are unchanged.
- **Evidence.** Declaring them changes every restart run that holds a
  C block or a closed loop; the label hold starts at `t_{n-1}`, a grid
  point, so it would be exact with the declaration.
- **Reverse.** Return the served instants `tau - dt` in label mode.

## G17. The helpers' observers carry their cadence: closed-loop reference cells run on trbdf2

- **Question.** `benchmarks/tools/v1_reader.py` composes a
  `bind_pi_to_switch` loop by hand (`step_observer=loop.step_observer`).
  Before 9.1 that observer had no cadence and trbdf2 refused it, so the
  six `closed_loop/*` reference circuits had no converged reference and
  every cell was `error`.
- **Taken.** The loop's observer is now a graph observer
  (`_rate_ticks`), so trbdf2 runs it. New outcomes, frozen in
  `v2_triage.yaml`: `cl_buck_pi`, `cl_buck_aggressive_pi`, `ll16` (pwl)
  agree (0.0121, 0.0066, 0.0074); `ll14` disagrees (pwl 0.287,
  restart/split 0.212, trbdf2 0.16), `ll15` (pwl 0.27, restart/split
  0.0654), `ll16` trbdf2 0.149; `cl_boost_pi` trbdf2 grinds to
  max_steps (5.6 ms of 30 ms), so it keeps no reference.
- **Evidence.** The baseline run (7b195243) refused on trbdf2 with "a
  step_observer with no cadence"; the circuits' results did not change,
  they were never compared.
- **Reverse.** Strip `_rate_ticks` from the helper observer (the refusal
  returns). The disagreements are not diagnosed in this slice.

## G18. Clarke, Park, SVM and PLL map to their Python classes

- **Question.** The kernel `add_svm` / `add_clarke` / `add_park` /
  `add_pll` are different expressions from the classes the chain maps:
  Clarke `(2a - b - c)/3` against `(2/3)(a - b/2 - c/2)`, SVM with
  `0.8660254037844386` and `1/v_dc` (and `v_dc <= 0` -> duty 0.5
  against the class's `v_dc = 1`), PLL starting from `omega = 0` and
  wrapping to `[-pi, pi)` against `2 pi f_nominal` and `[0, 2 pi)`.
- **Taken.** Ported kinds `svm`, `clarke`, `inverse_clarke`, `park`,
  `inverse_park`, `pll` on `block_laws.h` (G19), bit-identical on 10^4
  random inputs each (`test_control_graph_blocks.py`); `PLL` accepts two
  or three outputs (the third is `omega/(2 pi)`). The `add_*` C++ API
  keeps its laws for C++ users.
- **Evidence.** Closed-loop numbers change only for chains holding
  these blocks: `run_foc_motor_drive.py` and `run_pmsm_foc.py` (both
  still within their example claims), rounding-level for Clarke/Park/SVM
  and from the first tick for a PLL (the old one started at 0 Hz).
  Closed forms: balanced abc through Clarke-Park is constant dq within
  1e-12; SVM duties (0.6875, 0.3125, 0.3125) at alpha = v_dc/4; the PLL
  locks a 50 Hz unit vector from 60 Hz to 1e-3 Hz and 1e-3 rad in 1 s.
- **Reverse.** Map the classes back to `add_svm` etc.

## G19. One law table: block_laws.h for the kernel and the C export

- **Taken.** `core/include/pulsim/blockchain/block_laws.h` (C99, no
  allocation) holds one function per block kind; the kernel blocks call
  it and `export_c` copies it verbatim (embedded in the extension by
  CMake, `_pulsim.block_laws_source()`, in chunks below MSVC's 16 KiB
  literal limit). Not routed through it: `DelayBlock` (a FIFO, no
  arithmetic), `SignalDemux` (a copy).
- **Export choices.** Constants as C99 hex floats (exact). The
  single-rate shim keeps its state in a static struct (no `malloc`), so
  one loaded library is one block instance. A non-finite `SignalMux`
  selector holds the output in C (the kernel raises). Refused by name:
  `PythonBlock` and the Python library blocks, every `CBlock` (a
  compiled object has no source), a graph with no `Rate` (continuous
  time), a rate with a `compute_delay` (a circuit-sink property).
- **Evidence.** `test_control_graph_export_c.py`: 31 block kinds tick by
  tick over 10^4 ticks and a PI + limiter + PWM graph through
  `add_c_block(lib=...)` over 10^4 ticks, both byte-identical; compiled
  with `-std=c99 -Wall -Wextra -Werror -pedantic -ffp-contract=off`.
- **Reverse.** Drop the export; the kernel keeps the header.

## G20. The C-block compile cache is per user

- **Taken.** `~/Library/Caches/pulsim/cblock` (macOS),
  `$XDG_CACHE_HOME` or `~/.cache` + `/pulsim/cblock` (Linux),
  `%LOCALAPPDATA%\pulsim\cblock` (Windows), mode 0700; a cached
  library another user owns, or that others can write, is refused.
- **Evidence.** Before: `<tempdir>/pulsim_cblock_cache`, shared on a
  multi-user host, so another user could plant `cblk_<hash>.so` and
  have it loaded.
- **Reverse.** None wanted.

## G21. C blocks on trbdf2; refused on the exact strategy (task 9.3)

- **Taken.** `engine="trbdf2"` no longer refuses C blocks: the plan
  already ticks the C-block graph on the variable-step engine, which
  lands on the ticks, so the block runs the state-time law. `auto` keeps
  routing circuits with C blocks to `pwl` (no auto run changes engine).
  `exact` / `dsed` refuse by name (`c_blocks`; cells under both names).
- **Evidence.** P loop of docs/control-graph.md: trbdf2 meets
  `v[k]` (k = 1, 5, 10) within 1e-7 at rtol 1e-9; pwl at dt = 0.1 us
  and trbdf2 agree within 1e-4 at 1.95 ms (`test_c_block_engines.py`).
  What the exact strategy misses: a `b_extra` input and an observer.
- **Reverse.** Re-add the blocker for `engine="trbdf2"`.

## G22. Resume: graphs keep their ticks and states across runs

- **Taken.** No snapshot field: a graph already keeps its block states
  and tick counters across `simulate()` calls, and
  `rearm_if_restarted` restarts the grid only when time goes back, so a
  `resume_from` run continues it. The C-block graph is now compiled once
  per circuit layout (it was rebuilt per run, which restarted its grid
  at the resumed `t_start`); `control_tick_counts` / `n_fires` report
  the last run. Resuming exactly on a tick instant is not exact on pwl
  (label time samples `x(t - h)`, the resumed run has only `x(t)`).
- **Evidence.** `test_control_graph_resume.py`: a closed loop resumed at
  1.53 ms and a C block resumed at 0.73 ms reproduce the uninterrupted
  duty history / tick counts exactly and the states within the
  open-loop resume bound (1e-9 relative, the snapshot's own).
- **Hook.** A graph used with a *different* builder or after `reset()`
  has nothing to carry; a snapshot-borne control state needs the
  `SolverSnapshot` field listed under the 6.1 hook below.
- **Revised (integration batch 10 review).** Carry-over by default made
  a `Circuit` with a stateful C block answer differently on each run
  (1.383 V, then 1.785 V). Question: what does a run start from?
  Options: (a) carry over unless `resume_from` (before); (b) every run
  without `resume_from` starts each block from its declared initial
  state; (c) a snapshot of the block states, restored per run. Taken:
  (b), in `_c_graph_observer` above the engine dispatch (so every
  engine): native libraries re-run `term` then `init`
  (`_load_native_step` `reinit`; an export's shim `init` resets its
  static struct), Python steps get a deep copy of their initial state
  dict, buffers, signals and the tick grid are cleared
  (`BlockChain::reset`). (c) needs a state size from each library, which
  the ABI does not give. Carry-over is explicit: `resume_from` only.
  Every fresh run resets, the first included (two circuits on one
  export_c library share its static state; review of the merge).
  A serial `sweep(circuit)` with C blocks is allowed now (points are
  independent); `parallel=True` stays refused (one state on threads).
  `resume_from` continues the state the last run left, so resume from
  that run's snapshot (a snapshot-borne block state is the 6.1 hook).
  Chains and closed loops the caller passes (`step_observer=`,
  `closed_loops=`) are the caller's objects, not the circuit's: they
  keep the documented carry-over. Evidence: `test_control_graph_rerun.py`
  (Python step, inline C, export_c library; pwl and trbdf2: two runs
  byte-identical; a resumed run continues the block). Reverse: drop
  the `fresh` reset in `_c_graph_observer`.

## G23. The loop gate declares its edges; the closed-loop cells diagnosed

- **Question.** Since G17 the six `closed_loop/*` circuits ran on
  trbdf2 and three disagreed with it (frozen undiagnosed); `cl_boost_pi`
  had no reference (trbdf2 hit `max_steps`). Which side is wrong?
- **Independent reference.** `probes/closed_loop_reference.py`, outside
  pulsim: expm flow per PWL mode (diode crossings by bisection on it) or
  Radau rtol 1e-12 for the Shockley diode, the PI law re-typed, ticks at
  `kT` on `x(kT)`. trbdf2 rtol 1e-10 meets it within 1.1e-7 of the span
  at every tick on all five non-suspect circuits; pwl converges to it at
  first order. trbdf2 is right; nothing in the control law is wrong.
- **Taken.** (1) `CxxChainPwmGate.next_edge_after(t)`: period starts and
  each leg's OFF threshold under the duties held now; the adapter hands it
  to every engine and `make_combined_switch_fn` keeps it (trbdf2
  invalidates its edge cache at a tick; pwl drops it after every
  observer call, which a loop's tick is). trbdf2 lands the edges
  instead of learning them by sampling, and pwl's event split splits at them. (2) `ll16`
  `reference: suspect` (not a flyback; KNOWN_LIMITATIONS). (3) `ll14` /
  `ll15` frozen `disagree` (trbdf2 / pwl+split on `V(sw)`) with a named engine limitation (no
  post-edge sample at an edge that is also a tick). (4) Label time kept
  (G13): it moves no number, and its cost is now measured below.
- **Evidence** (max |v(kT) - ref| / span; before -> after; the V(sw) /
  state figures in the last column are cells against trbdf2 rtol 1e-8,
  the harness reference, not against the independent one):

  | circuit | trbdf2 default | pwl+split | strict cells after |
  |---|---|---|---|
  | cl_buck_pi | 4.2e-5 -> 4.2e-5 | 4.5e-2 -> 1.4e-3 | agree (all) |
  | cl_buck_aggressive_pi | 6.0e-5 -> 6.9e-5 | 2.9e-2 -> 1.9e-2 | agree (all; converges pointwise, no statistical measure needed) |
  | cl_boost_pi | max_steps -> 5.7e-5 | not run (plain pwl 2.0e-1) -> 4.2e-4 | trbdf2, pwl+split agree; pwl 0.058, pwl+restart 0.058 disagree (gate on the grid) |
  | ll14 | 2.6e-6 -> 1.1e-5 | 4.0e-2 -> 3.0e-4 | all disagree; trbdf2, pwl+split on V(sw) only (states 2.6e-5 / 1.4e-3); pwl, pwl+restart I(L1) 0.13 too |
  | ll15 | 1.7e-6 -> 3.9e-5 | 3.0e-2 -> 9.9e-6 | all disagree; trbdf2, pwl+split on V(sw) only; pwl states 6.6e-3 |
  | ll16 | — | — | unverified (suspect) |

  `cl_boost_pi`'s grind: its PI starts at duty 0, the first pulse is
  0.28 ns, and trbdf2's learned narrowest pulse capped every later step
  at 0.56 ns (`probes/trbdf2_sticky_pulse_cap.py`: 519,746 accepted steps
  in 0.3 ms with the edges hidden, 1,590 with them). `ll15` trbdf2 went
  `agree` (3.9e-4) -> `disagree` (1.24e-3): before, reference and cell
  both sampled the edges and the reference was already off by 7.8e-3
  (rtol 1e-8 vs 1e-10); now both land them and the rising edge's missing
  post-edge sample (`probes/trbdf2_tick_edge_sample.py`) sets the gap.
  Only trbdf2 and pwl+split cells move; plain pwl and pwl+restart never
  read the oracle (bit-identical).
- **Modulator sink (task 6.2 slice).** The gate IS the sink: it reads the
  duty register (what the channel's sinks see) and declares its edges,
  so no new class. Order of the cl_buck_pi loop on pwl+split against the
  reference, `dt = T/20 ... T/640`: the helper (label time) 1.39e-3,
  6.87e-4, 3.41e-4, 1.70e-4, 8.49e-5, 4.24e-5 (order 1.00); a state-time
  graph with the same gate 1.80e-5, 2.16e-5, 1.14e-5, 3.07e-6, 1.44e-6,
  1.42e-6 (77x smaller at T/20, no clean order: the split takes an edge
  within 1 % of `dt` of a grid point at the point, and a CCM variant
  scatters the same way). Second order needs a state-time helper, a
  number-changing switch for the release, not this slice.
- **`engine='auto'` (measured, default unchanged).** The six circuits
  through the harness: without `dt` auto takes trbdf2 (the helper's
  observer carries its cadence) and gives exactly the trbdf2 cell; with
  `dt` it takes pwl and gives the pwl cell. Twelve closed-loop
  examples (`examples/scripts/run_*closed_loop*`, auto-tune, motor
  drives): auto takes pwl and every claim holds; forced to trbdf2 the
  eleven with a raw `step_observer` are refused (`run_buck_with_helper`
  runs: 4.9844, claim holds) with a typed `UnsupportedError` (an observer with no
  cadence). Routing them to trbdf2 needs `controller_period` on each
  script first, a separate change.
- **Reverse.** Drop `next_edge` for `ChainPwmGate` in
  `python/switch_fn_adapter.hpp` and the `SwitchSchedule` wrap in
  `make_combined_switch_fn`: trbdf2 and pwl+split sample the gate again.

## G24. Reviewer leftovers: cache parents, Windows, stacklevels, one instance

- **Cache parents.** `_cache_dir` tightens and checks `pulsim/` as well
  as `pulsim/cblock/`: a group-writable or foreign `pulsim/` let another
  user swap `cblock/` for theirs (a symlinked `pulsim/` is refused too,
  as `cblock/` already was). `~/Library/Caches` and `~/.cache` belong
  to the profile and are left alone.
- **Windows.** No POSIX owner or mode, and a real ACL check needs
  pywin32 or a page of ctypes. Taken: the cache must sit inside
  `%USERPROFILE%` (per-user ACL by default); a `LOCALAPPDATA` pointing
  elsewhere is refused by name. Reverse: an ACL check that admits a
  private directory outside the profile.
- **Stacklevels.** Measured where each warning pointed: the helpers'
  closure warnings pointed into `control.py` (3 -> 4), the `sim_dt`
  clamp into `c_block.py` (none -> 2), and the fixed-period warning one
  frame above the user for a closed loop (`_check_fixed_periods` now
  takes the level: 4 from `for_fixed_step`, 5 from `FixedStepTicks`).
  A test pins each to the caller's file.
- **Multi-rate call order.** Tested both ways: due together,
  `step_rate0` then `step_rate1` reproduce the kernel bit for bit when
  the slow rate reads the fast one, and differ when the fast reads the
  slow (the documented limit), so the doc's rule is now pinned.
- **One static state.** The shim exports `pulsim_cblock_static_state`
  (`extern const int`, so a C++ compile keeps it); a second
  `add_c_block` of a library carrying it on the same builder is refused
  (one state stepped twice per tick). Across builders it is allowed: a
  result keeps its builder, so refusing while an earlier handle lives
  broke `[simulate(make(k)) for k in ...]` sweeps (review); each
  `add_c_block`'s `init` resets the shared state, as before. Libraries
  with their own state from `init` are unaffected. Reverse: emit a
  heap-allocating shim, which the no-`malloc` rule excludes today.

- **Review of G23 (fresh reader).** Since the loop's gate declares its
  edges, `make_combined_switch_fn` of it has edges, and shooting and
  sensitivities, which refused it only as edgeless, accepted it with a
  frozen duty. Taken: a C++ member `SwitchSchedule.follows_controller`
  (read-only in Python), set on the wrapped loop gate and propagated by
  every combination, so `make_combined_switch_fn` keeps its
  `SwitchSchedule` return type (a Python attribute on a `py::object`
  return lost it); sensitivities refuse a closed-loop gate by name like
  shooting, and `sweep(parallel=True)` reads the member too.
  Also corrected: `ll14` / `ll15` pwl and pwl+restart miss the states
  too (not `V(sw)` only); `cl_boost_pi` pwl+split before was not run;
  the probe solves its own DC start (2.3880597 A / 23.880597 V,
  600.558046 A / 11.160915 V, equal to the engines').

## G25. One Park / PLL law; address wrappers; a libm-robust export test

- **Question.** `blocks.hpp` kept a second Park / inverse Park / PLL
  law (std::cos / std::sin; the PLL wrapped to [-pi, pi) and summed
  `Ki * vq * dt`) beside `block_laws.h`, used by `add_park` / `add_pll`
  and `CxxParkTransform` / `CxxPLL`; `block_laws.h` took `&cos`, whose
  address is unspecified in C++.
- **Taken.** The structs call `pl_park` / `pl_inverse_park` / `pl_pll`
  (one law; the raw `add_pll` / `CxxPLL` angle is now in [0, 2pi), as
  `pulsim.control.PLL`); the volatile pointers name `pl_cos_call` /
  `pl_sin_call`, wrappers whose address is defined in C and C++.
  Clarke and SVM keep their kernel variants (no libm call; G17).
- **Export test.** The kernel's cos / sin / fmod are the extension's
  libm, the export's are the C compiler's (MinGW against MSVC's UCRT on
  Windows). Every signal outside Park / PLL is compared byte for byte;
  the Park / PLL signals byte for byte when a probe compiled by the same
  compiler gives Python's cos / sin / fmod on 10^4 inputs, else within
  1e-10 of their scale. No skip.
- **Reverse.** Restore the struct bodies (git) and `= cos` / `= sin`.

## G26. Controller state per sweep point and in the snapshot

- **Question.** A serial `sweep` reused the caller's controllers: point
  k continued point k-1's integrator (`closed_loops=[loop]`, R = 10, 8,
  10, 8 ohm: v_out 5.1487, 6.6073, 7.7160, 7.9842 V at 1 ms; points 0
  and 2 differ). `resume_from` continued whatever the last run left in
  the graphs and C blocks, not the snapshot's run (the G22 hook).
- **Taken (sweep).** Inside `sweep` / `monte_carlo`
  (`_control_state.sweep_points`) each point restarts the caller's
  graphs from the state they had when the sweep began
  (`CxxBlockChain.save_state` / `restore_state` plus the Python-held
  parts). Python controller objects in the keyword arguments
  (`step_observer`, `switch_fn`, `b_extra_fn`, `closed_loops`) are
  deep-copied per point (one memo: an observer and its gate stay
  paired); the caller's object is untouched. Refused by name
  (`UnsupportedError`, feature `controller_state`): a closure over
  mutable state (cannot be copied), a graph on parallel points (one
  graph stepped by several threads). The old `ValueError` for parallel
  stateful kwargs is now that refusal. A plain `simulate()` keeps G22
  (the caller's graphs carry over; C blocks restart).
- **Taken (snapshot).** `SolverSnapshot` holds an opaque
  `std::shared_ptr<void> control_state` (`result.hpp`); Python stores
  `[(chain, GraphState)]` in it after the trbdf2 and pwl runs.
  `resume_from` restores it, so another run in between does not leak.
  A graph's state: every kernel block's POD (`BlockChain::capture`, a
  copy and a restorer per adapter; `uncaptured_blocks()` lists a block
  with a reset and no capture), signals, tick grid, recordings; a
  `PythonBlock` object: a deep copy of its `__dict__`; a Python C block:
  its state dict and buffers; a native C block: the optional ABI pair
  `pulsim_cblock_save` / `pulsim_cblock_restore` (every `export_c`
  shim emits it over its static struct; restore returns 0, or nonzero
  to refuse). Refused by name: an opaque
  native block (no pair), an object whose state does not deep-copy, a
  pickled snapshot of a run with controllers (the state does not
  pickle; the last field of the pickle tuple marks it — field 13 of
  the version-3 tuple since integration batch 12), a snapshot of other
  controllers.
- **trbdf2 resume.** The kernel ticks observers at `t_start + k*T`; a
  graph resumed between two ticks would tick off its grid (1.63, 1.73
  ms instead of 1.6, 1.7 ms; duty up to 0.9 % off). Refused on trbdf2
  when any rate group's next tick (`next_due_per_group`) is off the
  `t_start + k*period` grid (feature `resume_from`; `auto` names it as
  the route blocker).
  Reproducer for the TR-BDF2 track: `probes/trbdf2_resume_tick_phase.py`
  (needs a per-observer first tick). Resuming on a tick is allowed.
- **Also refused (review).** A graph block running a function whose
  closure cells, defaults or the module globals it reads may hold
  mutable state, followed into the functions and bound methods they
  hold (its state is not copied; the same test refuses such a
  `step_observer` in a sweep; only `pulsim._pulsim` natives count as
  stateless),
  and a resume from a snapshot that carries no state for a controller
  that has already run. A sweep mixing a graph and a controller object
  deep-copies only the object. The export_c restore returns nonzero on
  a size mismatch (refused by name).
- **Limits.** A call refused after the control plan is composed (a
  route blocker, a configuration error) has already put the snapshot's
  or sweep start's state into the caller's graphs. Every run with
  controllers copies their state into the final snapshot (recordings
  included). A restored object's `__dict__` holds copies: outside
  references to its old sub-objects no longer follow it.
- **Evidence.** `test_controller_state_isolation.py` (helper loop in
  both sweep forms, chain as `step_observer`, controller object on
  threads, refusals; resume on pwl at 1.53 ms bit-identical duties and
  rel 1e-9 states, trbdf2 at 1.5 ms rel 1e-6);
  `test_control_graph_rerun.py` (Python and `export_c` blocks resumed
  from an older snapshot, pwl and trbdf2; an opaque inline block
  refused). The old strict xfail is gone.
- **Reverse.** Drop `sweep_points` / `point_kwargs` from `sweep.py`,
  `save_into` / the snapshot branch of `start_run`.

## G27. One run at a time on an export_c library's static state

- **Question.** Two circuits that load one `export_c` library share its
  static struct. G22 resets it per run and G26 restores it on resume,
  so sequential runs are clean; two runs at once on user threads would
  step one struct twice per tick.
- **Taken.** The run's control plan holds the static-state libraries
  of its circuit and of the C blocks inside the caller's graphs
  (`c_block.static_state_hold`, re-entrant per thread,
  released when the plan is dropped as `simulate()` returns or
  unwinds; a `simulate` wrapper would have moved every warning's
  stacklevel); a run on another thread that needs one is refused by
  name (`RuntimeError`). Parallel sweeps already refuse C blocks.
  Limits: after an exception the hold lives as long as its traceback;
  a nested run on the same thread (from an observer) is not refused.
- **Evidence.** `test_a_static_library_runs_on_one_thread_at_a_time`:
  a run started from inside another's observer on a second thread is
  refused; after the first run returns the same call runs.
- **Reverse.** Drop `ControlPlan.static_hold`.

## G28. A discrete Gate: a graph signal drives switches on its ticks

- **Taken.** `MixedDomainBlockChain.gate(builder, {channel: switch})`
  returns a native schedule (`CxxChainSignalGate`): switch on while the
  published signal is > 0.5. Its `next_edge_after(t)` is the graph's
  `next_jump_after(t)` (the publication instants), so pwl+split splits
  at them and trbdf2 lands them; it marks `follows_controller`, so
  shooting refuses it as it refuses the loop gate. It needs the built
  graph (`make_step_observer` first; `ValueError` otherwise).
- **Evidence.** `test_control_graph_gate.py`: a 100 kHz comparator
  regulates a buck to 12 V; on trbdf2 every change of Q is recorded by
  a step that starts on a tick k*T (within 1e-12 s); pwl+split and
  trbdf2 end within 1e-3 of each other.
- **Waits.** A comparator crossing located between ticks is 7.2 (a
  state event on the unified device interface).
- **Reverse.** Remove `gate`, `ChainSignalGate` and its adapter branch.

## G29. One Clarke / inverse Clarke / SVM law (G25's leftover)

- **Taken.** `ClarkeTransform`, `InverseClarkeTransform` and
  `SpaceVectorModulator` in `blocks.hpp` (raw `add_clarke` /
  `add_inverse_clarke` / `add_svm`, `CxxClarkeTransform`,
  `CxxSpaceVectorModulator`) call `pl_clarke` / `pl_inverse_clarke` /
  `pl_svm`, the law the Python classes and the export already use
  (no libm call but `sqrt`, correctly rounded). Numbers move by ulps;
  `v_dc <= 0` now normalises by 1 (the Python class) instead of
  returning duty 0.5.
- **Evidence.** `test_raw_kernel_clarke_and_svm_are_the_one_law`: 10^4
  random inputs at v_dc = 400, 0, -5, byte-identical to the Python
  classes (Clarke and SVM; the inverse Clarke struct is the same
  one-line call, not reachable without a graph run). Golden probe identical (no golden circuit uses them).
- **Reverse.** Restore the struct bodies (git).

## G30. control_tick="state": second order for pwl+split closed loops

- **Question.** pwl+split converges at first order on the closed-loop
  circuits: label time samples `x(kT - dt)` (G17 / docs/control-graph.md)
  and the split moves an edge within `0.01*dt` of a step boundary onto
  it.
- **Taken.** `simulate(control_tick="state")`, default `"label"` (every
  number unchanged; golden identical). It runs the label-time graphs
  (closed-loop helpers, C blocks) in state time for that run (tick k on
  `x(kT)`, held from `kT`; their publication instants are declared, so
  the restart rule restarts on them) and sets
  `SimulationOptions.split_snap` (new, default 0.01, validated in
  [1e-12, 0.25]) to 1e-6. State time alone left the order ragged (1.4e-6
  floor on cl_buck_pi, order 0.0-1.9); the snap was the rest.
- **Evidence.** pwl+split, max |v(kT) - ref| / span against
  `probes/closed_loop_reference.py`, dt = T/d (harness dt = T/20; ll15
  T/10, ll16 T/26.7):

  | circuit | label T/20 | state T/20 | state T/40 .. T/640 orders |
  |---|---:|---:|---|
  | cl_buck_pi | 1.39e-3 | 8.35e-6 | 2.07 2.06 2.01 2.00 2.00 |
  | cl_buck_aggressive_pi | 1.95e-2 | 8.25e-5 | 2.43 1.81 2.37 1.99 1.99 |
  | cl_boost_pi | 4.16e-4 | 4.05e-5 | 1.72 2.21 2.39 2.37 2.07 |
  | ll14_buck_closed_loop | 3.04e-4 | 7.77e-5 | 1.08 1.58 1.82 1.88 2.01 |
  | ll16_flyback (G31) | 2.63e-4 | 3.27e-5 | 1.77 1.83 1.96 2.04 (to T/320) |
  | ll15_boost_closed_loop | 1.01e-5 | 1.01e-5 | same as label (duty pinned at 0.85) |

  Label time stays at order 1.00 (cl_buck_pi 1.39e-3 -> 4.2e-5 at
  T/640). Cost: wall time within the noise of the label run (0.40 vs
  0.42 s, ll14 at T/640; 0.80 vs 0.59 s, cl_boost_pi).
  `test_control_tick_state.py`: the pwl+split duty history sits 10x
  closer to trbdf2 rtol 1e-10 in state time; the default is the label
  run, byte for byte.
- **Limit.** The one `snap` also merges diode crossings; at 1e-6*dt
  two estimates of one crossing may split into two sub-steps (no change
  measured on diode_bridge_rectifier / half_bridge_transformer_dc).
- **Why not default.** It moves every closed-loop number (frozen cells,
  examples, the golden probe); the brief keeps defaults.
- **Reverse.** Drop the kwarg and the field.

## G31. ll16 is a flyback again; a structural lint of the circuits

- **Question.** `ll16_flyback_closed_loop` was not a flyback: `Lsec`
  was coupled to nothing (V(vout) = 0 on every engine, an open loop)
  and `Lm_core` / `Lleak_prim` sat in parallel with the switch, so each
  turn-off cut 604 A into roff = 1e7 ohm (G23 marked it `suspect`).
- **Taken.** `Lm_core` (160 uH) across an ideal 4:1 transformer
  (`transformer` T1, secondary dot reversed), `uic: true` (the DC point
  shorts Lm through the switch). The 2 uH leakage is folded into Lm: it
  had no clamp to take its current at turn-off. The independent tool
  (`probes/closed_loop_reference.py`, topology `flyback`: Lm, ideal
  transformer, Shockley diode by Radau) gives the reference: trbdf2
  rtol 1e-10 within 9.5e-8 of span on v(out) at every tick, pwl+split
  2.9e-4. Back on the strict gate (no `suspect`): every cell disagrees
  on V(sec) / V(drain) and is frozen. In the DCM idle interval the drain
  is set by the residual Lm current through roff (no drain
  capacitance), so it follows the solver tolerance (trbdf2 default vs
  rtol 1e-8: 0.31 of span on V(drain), 1.7e-5 on V(vout)). States agree
  (V(vout) / I(Lm): trbdf2 1.7e-5 / 5.7e-5, pwl+split 8.1e-5 / 5.4e-4).
  As authored the 12 V set-point is out of reach (duty capped at 0.5
  with a 4:1 ratio gives Vout + Vd = 12 V in CCM) and the 2 ms run ends
  in the start-up (6.7 V); kept, it is the author's loop.
- **Lint.** `benchmarks/tools/circuit_lint.py`, two rules on the
  netlist graph: `source_free` (an inductor whose biconnected block has
  no source, switch, winding or `ic`: it never carries current) and
  `inductor_cutset` (one switch open, the others closed, diodes on: a
  node group off ground that meets only inductors, whose current each
  turn-off cuts). On the 126 circuits it found only the old ll16 (both
  rules); a first variant with every switch open flagged 47 synchronous
  converters and was dropped. `test_circuit_lint.py` keeps the set
  clean.
- **Reverse.** Restore the old netlist and the `suspect` entry.

## G32. pwl graphs tick on the kernel's event queue (control_tick="state")

- **Question.** Task 6.1: trbdf2 and dsed ticked a graph's rate groups
  from the queue (add-unified-event-queue); pwl still let each graph
  decide its own ticks inside its observer.
- **Taken.** With `control_tick="state"` on pwl every rate group of every
  run graph (closed-loop helpers, C blocks, a chain as `step_observer`)
  is a kernel tick source (`run_transient(tick_fns=)`, after the Python
  sources): `CxxBlockChain.make_queue_tick(g)` (C++, no GIL) marks group
  g, and the graph's observer, run after the queue's ticks in the same
  kernel call, runs the marked groups in one pass
  (`set_queue_driven`). The pwl observer call labelled t_n carries
  x(t_{n-1}), so a source ticks at origin `t0 + offset + dt`: the call
  whose t_{n-1} >= tau, i.e. state time. Origin and next k come from the
  graph's own grid (`grid_t0`, `group_next_k`), so a resume continues it;
  the snapshot's `tick_origin` / `tick_next_k` list them after the
  Python sources. `run_transient(tick_fns=)` now takes a `CxxChainTick`
  as well as a callable. Default (`"label"`) unchanged: graphs tick
  themselves.
- **Evidence.** Byte-identical to the graph's own state-time ticking on
  the cl_buck_pi loop (pwl+split and plain, duty history, v(out), tick
  counts) and on a two-rate chain with offsets (recorded ticks, both
  channels, v(out)); label runs byte-identical to the tree before.
  `test_control_graph_queue_resume.py::test_state_mode_ticks_the_graph_from_the_kernel_queue`.
- **Split one step after a tick.** The queue never takes an instant at
  or after its t_end, so a state-time tick at tau = t_end - dt is not
  taken by the stopping run; the graph sees it due on its own grid at that
  last call and keeps the call's state (as G33's label stash), and the
  resumed run's priming call (its queue fires there, from the graph's
  next k) serves it with that state. A graph reused by a new run whose
  time goes back registers its sources from t_start (it rearms).
- **Reverse.** Drop `_queue_groups` in `_control_plan.py` and the
  chain's `queued_`.

## G33. Controllers' state in the pickled snapshot; a label resume on a tick

- **Question.** G26 left a pickled snapshot of a run with controllers
  refusing the resume (field 12 a marker). Separately, a pwl label-time
  closed loop resumed exactly on a tick served that tick from the resumed
  run's priming call with x(T1); the run that did not stop serves it at
  label T1 with x(T1 - dt) (duty 7.3e-5 off at tick 37 of cl_buck_pi).
- **Taken (pickle).** Field 12 of the version-5 tuple (still 17 fields)
  is either the old bool marker or `("pulsim.controllers", 1, [(bytes,
  parts)])`: each graph's kernel state as bytes
  (`CxxBlockChain.dump_state` / `load_state`; a captured block state is
  its bytes when trivially copyable, else its `fields()` — DelayBlock,
  TransferFunction, LookupTable1D, MovingAverageFilter, Mechanical), its
  Python-held parts (object `__dict__` copies, C-block state / buffers)
  as they are. Written only when every part pickles, else the marker
  (refused by name as before). A resume matches graphs by position and
  checks each by structure (block names, channels, groups: ValueError ->
  `UnsupportedError` "other controllers"); new controller objects built
  the same way take the state. Bytes are for the machine and build that
  wrote them (no endianness / layout tag beyond a magic).
- **Taken (label resume).** A label-time pass that leaves a due tick out
  only because of the horizon keeps that call's state; the next call at
  the same label (the resumed run's priming call) serves the tick with it.
  Part of the graph state (and of the bytes). Only a resume uses it: a
  plain run (G22 carry-over, even one starting at the last t_end) drops
  it, so a hand-chained run keeps its numbers. A pickle written by this
  build fails to load on an older build (field 12 is not a bool there).
- **Evidence.** cl_buck_pi loop, uninterrupted to 60T vs split at 37T (on
  a tick), 37.05T (one pwl step after one) and 37.3T (between): pwl label and pwl state (split), in-memory
  and pickled into fresh controllers: duty history, times and v(out)
  bytes identical. trbdf2 rtol 1e-8: pickled resume = in-memory resume
  byte for byte; against the run that did not stop 5.7e-9 (on) / 4.9e-9
  (between) in duty, 4.9e-8 / 3.1e-8 V in v(end) (its steps depend on
  the span). dsed has no snapshot (hook request below).
  `test_control_graph_queue_resume.py`.
- **Reverse.** Field 12 back to `bool(control_state)`; drop the stash.

## Closed-loop reference, per engine (2026-11-18)

max |v(kT) - ref| / span against `probes/closed_loop_reference.py`:
trbdf2 1e-10 1.8e-8, 1e-8 3.5e-7, default 1.1e-5; pwl 3.6e-2; pwl+split
1.4e-3; pwl+split state (queue) 8.4e-6; dsed 5.5e-11 (cl_buck_pi).
ll14_buck_closed_loop: trbdf2 1e-10 6.9e-9, default 7.6e-7; pwl 4.1e-2;
pwl+split 3.0e-4; state 7.8e-5; dsed refuses (Shockley diode).
cl_boost_pi: trbdf2 1e-10 1.2e-7, default 1.4e-6; pwl 2.0e-1; pwl+split
4.2e-4; state 4.0e-5; dsed 8.4e-8. cl_buck_aggressive_pi: trbdf2 1e-10
2.9e-8, default 4.3e-5; pwl 1.4e-2; pwl+split 1.9e-2; state 8.3e-5; dsed
1.4e-11. ll15_boost_closed_loop: trbdf2 1e-10 2.7e-8; pwl+split and
state 9.9e-6; dsed refuses. ll16_flyback: trbdf2 1e-10 5.9e-8; pwl+split
2.9e-4; state 3.1e-4 (the tree before this change gives the same 3.104e-4
with the probe's dt; G30's 3.27e-5 is not reproduced by this probe run —
not chased); dsed refuses. The state column equals the tree before (graphs
then ticked themselves): the queue path changed no number.

## Hooks for task groups 6–7 (not in this slice)

- **6.1 tick events.** Done for the snapshot (G26). The graph's ticks
  still reach `pwl` through the observer and `trbdf2` through
  `observer_periods`. On the event queue
  (`origin/feat/core-p3-event-queue`, `events/event_queue.hpp`) each
  rate group becomes `add_periodic(Kind::ControlTick, Rate{period,
  offset})`; on resume `offset = next_due() - t_start`, which lifts the
  G26 trbdf2 refusal. `next_due()` is in `BlockChain` now.
- **6.2 continuous-rate blocks.** Their states must join the MNA
  unknowns: a stamping hook in `run_transient` / `trbdf2_transient`
  (`(x, h) -> rows`), not available from the observer.
- **7.x tick and comparator events on the unified queue.**
  `core/include/pulsim/events/*` (forbidden) needs an event source
  interface `next_event_after(t)` that the graph implements; today the
  ticks reach `pwl` through the observer and `trbdf2` through
  `observer_periods`, and comparator crossings are not located.
- **Hook requests (2026-11-18, routed around, loops frozen).**
  (1) dsed: no `final_snapshot` / `resume_from` (the exact engine owns
  `dsed/scheduler.hpp`): a dsed closed loop cannot be resumed at all.
  (2) `run_transient`: the queue's last instant is `< t_end`; graphs
  keep the last call's state themselves (G32 / G33), the Python tick
  sources cannot: in label time their tick at a split instant presumably
  runs at the resumed priming call with x(T1) (not measured).
  (3) 6.2 / 7.1 / 7.3: continuous blocks need a stamping hook
  `(x, h) -> rows` in both loops. (4) 7.2: a comparator gate as a state
  event needs a located queue source (`add_located`) fed by a crossing
  locator over the step's state interpolant, i.e. a loop hook.
- **Hand-off (2026-11-18, after G32 / G33).** Open: 6.2 and group 7
  (hook requests (3) / (4): loop hooks, frozen loops); a dsed snapshot
  (request (1)); 10.1 docs (control guide: `control_tick='state'` on the
  queue, pickled controller state). Verified on this tip (macOS): ruff,
  full pytest (5197 passed + the 3 stub tests after regenerating the
  stub), `pytest -m reference` 368 passed, `reference_status` exit 0 with
  the table unchanged, openspec strict, stub `--check`, the cl_buck_pi
  label / state runs and a two-rate chain byte-identical to the tree
  before. The full ctest had not finished building when this was written:
  run it before pushing.
