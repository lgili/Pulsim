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

- **Taken.** `builder._c_graph` holds every `CBlock`, each in the group
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

## Hooks for task groups 6–7 (not in this slice)

- **6.1 graph state in the solver snapshot.** `SolverSnapshot`
  (`core/include/pulsim/solver/run_transient.hpp`, forbidden) needs an
  opaque `std::vector<Real> control_state` plus a save/restore callback
  pair `(BlockChain::save_state(), restore_state())`; each block then
  needs `xd` slices (G5).
- **6.2 continuous-rate blocks.** Their states must join the MNA
  unknowns: a stamping hook in `run_transient` / `trbdf2_transient`
  (`(x, h) -> rows`), not available from the observer.
- **7.x tick and comparator events on the unified queue.**
  `core/include/pulsim/events/*` (forbidden) needs an event source
  interface `next_event_after(t)` that the graph implements; today the
  ticks reach `pwl` through the observer and `trbdf2` through
  `observer_periods`, and comparator crossings are not located.
