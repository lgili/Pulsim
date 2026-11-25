## Why

A controller can be written in five ways today, and none of them has a
sampling instant, an evaluation order or a coupling to the circuit that
the kernel defines (paths relative to the worktree; `core/…` =
`core/include/pulsim/…`):

| Representation | Where | How it runs |
|---|---|---|
| Python closures (`step_observer`, `switch_fn`, `b_extra_fn`) | user code | called by the engine through pybind |
| `ClosedLoop` (`bind_pi_to_switch`, `bind_pi_to_duty_callable`) | `python/pulsim/control.py:1180, 1225, 1369` | two closures sharing a duty cell; throttle at `:1350-1355` |
| Python `MixedDomainBlockChain` | `python/pulsim/blockchain.py:128` | Python evaluator, or compiled to the C++ chain |
| C++ `BlockChain` | `core/blockchain/chain.hpp:138` | `vector<std::function>` over a string-keyed `unordered_map` |
| C blocks | `python/pulsim/c_block.py`, `python/pulsim/cblock_abi.h` | Python observer with its own zero-order hold (`c_block.py:387-390`) |

Read on this branch:

- **No schedule.** `BlockChain::step` runs the blocks in insertion
  order (`chain.hpp:183-190`); the header calls the list "topologically-
  ordered" (`chain.hpp:5`). A channel read before it is written returns
  0 with no error (`chain.hpp:109-112`), so a chain built in the wrong
  order runs with a one-step delay or a zero input. There is no
  algebraic-loop detection.
- **No rates.** Every block runs on every solver step with the solver's
  `dt` (`chain.hpp:293-297`). A sample rate exists only as three
  different throttle idioms (`control.py:1350-1355`,
  `c_block.py:387-390`, the chain's observer).
- **One step of lag.** The observer is called with the previous state
  before the step (`core/solver/run_transient.hpp:866-872`) and its
  outputs enter through `b_extra` (`:905`): the coupling is explicit.
  There are no continuous control states.
- **A silent second implementation.** When a block has no C++
  counterpart the chain falls back to the Python evaluator with a
  warning (`blockchain.py:268-281`). `control.py` has 28 block classes,
  `core/blockchain/blocks.hpp` has 18 structs; the ten without one are
  `Comparator`, `RateLimiter`, `SampleHold`, `TransferFunction`,
  `StateMachine`, `OpAmp`, `DelayBlock`, `SignalMux`, `SignalDemux`,
  `LookupTable1D` (the audit counted 29 and 11, including `ClosedLoop`).

Measured (audit `raw/arch.md` defects 4–6 and capability matrix, probes
`arch/probe2.py`, `probe6.py`, `probe7.py`; owner note
*v2-roadmap-2026-08* for the duty figures):

- Fixed engine: 198 controller ticks where 200 were due
  (`control.py:1214-1218`); realised duty off by 12,386 ppm at
  dt = 2e-6 and 157 ppm at dt = 2e-8 (one million steps), against
  0.4 ppm when the ticks are scheduled.
- TR-BDF2 with a 2 kHz PI beside a 20 kHz monitor: the PI ticks 400
  times instead of 40 and the output moves from 13.03 V to 12.09 V.
- `controller_period` on the fixed engine: 2001 observer calls where 20
  were asked for.
- C blocks are refused on `trbdf2` and run open-loop behind a warning on
  `dsed`.
- A Python `step_observer` costs 944 ns per step and a `b_extra_fn`
  852 ns per step, on a 393 ns step (`raw/perf.md` rows 16–18).

`fix-control-tick-semantics` (Phase 0) repairs the three measured
defects in place. This change replaces the structure that produced
them.

## What Changes

- **`ControlGraph` in the kernel**: typed blocks, signals as integer
  slots in one flat array (no strings at run time), block state in two
  flat arrays (discrete and continuous) that are part of the solver
  snapshot.
- **Rates**: every block group has a `Rate` — `Discrete{period,
  offset}`, `Continuous`, or `Triggered{event}`. Stateless blocks
  inherit the rate of their inputs; stateful blocks declare one.
- **A sorted schedule**: topological order of the direct-feedthrough
  graph, two-phase execution (all outputs, then all state updates), and
  an error that names the blocks of any algebraic loop.
- **Discrete blocks fire on tick events** of the unified event queue,
  at `offset + k·period` exactly, on every engine; a group may declare a
  computational delay (zero, a time, or one sample).
- **Continuous control states are integrated by the circuit solver**:
  the continuous subgraph contributes rows `d/dt x_c − f_c(x_c, probes)
  = 0` and algebraic outputs through the unified device interface, so
  there is no lag and the exact propagator covers linear compensators.
- **Outputs**: modulators (duty register, analytic edges),
  controlled sources, and comparator gates (a state event when the rate
  is continuous, a tick event otherwise).
- **One block library**: the ten missing blocks are ported; the Python
  evaluator and its fallback are removed; an unsupported block is an
  error.
- **Consolidation**: `MixedDomainBlockChain` becomes the Python builder
  of a `ControlGraph`; `bind_pi_to_switch` builds a three-block graph;
  a C block is a graph block with a rate (same ABI); a Python callable
  is a `PythonBlock` with a declared rate, flagged opaque.
- **C export**: one C99 translation unit per graph (state struct, init,
  one step function per discrete rate, no allocation), plus a shim on
  the C-block ABI for single-rate graphs so the export can be loaded
  back and compared bit for bit.
- **BREAKING:** a chain whose results depended on insertion order (an
  unintended one-step delay) changes; the release note lists how to
  keep the delay explicitly (`DelayBlock`).
- **BREAKING:** `make_step_observer(use_kernel=False)` and the Python
  evaluator are removed; `step_observer` / `b_extra_fn` closures without
  a rate are deprecated in favour of `PythonBlock`.

## Impact

- Affected specs: `control-graph` (new).
- Affected code: new `core/control/{graph,rate,schedule,blocks,
  modulator,export_c}.hpp`; `core/blockchain/*` folded in and retired;
  `python/bindings.cpp` (graph bindings; `run_transient_with_chain`
  removed); `python/pulsim/{blockchain,control,c_block,yaml_chain,
  fast_block}.py`; `python/pulsim/__init__.py` (the two loop/observer
  composition blocks at `:2215-2283` and `:2420-2455` disappear);
  docs (control guide, AI_GUIDE), stubs.
- Depends on:
  - `fix-control-tick-semantics` (Phase 0): the behaviour this change
    must preserve while replacing the mechanism.
  - `add-unified-event-queue` (Phase 3): tick events, declarative
    modulators with analytic `next_edge`, state events for comparators.
  - `add-unified-device-interface` (Phase 3): the continuous subgraph as
    a device (`f`, `q`, AD Jacobian).
  - `add-kernel-simulation-object` (Phase 3): the graph as part of
    `Simulation`, its state in the snapshot.
  - `add-probes-and-recorders` (Phase 3): circuit probes as graph
    inputs, signals as recorded outputs.
  - `add-controlled-source-set` (Phase 4): VCCS / CCVS / CCCS outputs.
- Ships on today's structure (tasks 1–4): rates, index slots, the
  sorted two-phase schedule with loop detection, and the ten ported
  blocks inside the existing C++ chain, with ticks delivered by the
  Phase 0 scheduling. C export of discrete graphs (task 8) needs only
  those.
- Must wait for Phase 3: tick events on every engine, modulators,
  comparator state events, continuous states in the solver.
- Used by: `add-periodic-steady-state-shooting` (controller states in
  the periodic state), `add-parameter-sensitivities` (controller gains
  as parameters).
- That discrete blocks at exact sample hits, continuous control states
  in the solver and code generation from the same graph are what PLECS
  offers is the audit's statement from general knowledge; it was not
  measured. "Close to what a HIL target needs" is the audit's
  inference, not a prototype.
