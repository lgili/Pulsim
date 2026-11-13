# Decisions — add-compiled-control-graph

Taken by the implementer under the owner's delegation (2026-11-13).
Each entry: question → option taken → evidence → how to reverse.

## G1. Where the graph lives

- **Question.** New `core/control/*` headers, or the existing
  `blockchain::BlockChain`?
- **Taken.** Inside `BlockChain` (`core/include/pulsim/blockchain/chain.hpp`),
  as the proposal allows for tasks 1–4 ("ships on today's structure").
  Its public API (`add_*`, `make_step_observer`, `record_channel`,
  `make_chain_switch_fn`) is unchanged, so the 30+ Python helpers that
  build a `CxxBlockChain` (motors, observers, thermal, YAML) keep
  working.
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
