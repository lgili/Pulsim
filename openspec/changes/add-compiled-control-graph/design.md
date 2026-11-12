## Context

Control in Pulsim grew as callbacks: an observer reads the previous
state, a closure decides the gates, a `b_extra` closure injects sources.
The C++ `BlockChain` moved the arithmetic out of Python but kept the
callback shape (per step, insertion order, string-keyed channels).
Owner notes (*v2-roadmap-2026-08*): a throttle on `t − t_last < T`
rejects exactly scheduled ticks (129 of 200), and a controller that
moves the duty invalidates the edge cache — both are symptoms of a
controller the kernel cannot see into.

Constraints: refusals instead of silent fallbacks; the GIL must not be
taken inside the step loop; `-ffp-contract=off` is part of the build and
of the bit-identity claims; the C-block ABI
(`python/pulsim/cblock_abi.h`) is public.

## Goals / Non-Goals

- Goals: one representation with a defined order, rate and coupling;
  identical tick instants on every engine; continuous control states
  solved with the circuit; controller state in snapshots; C export of
  discrete graphs.
- Non-Goals: an iterative algebraic-loop solver; automatic
  discretisation of continuous blocks for export; fixed-point code
  generation; a graphical editor; triggered subsystems beyond a single
  event trigger per group; multi-threaded graph execution.

## Decisions

### D1. Data model

```
ControlGraph
  signals : Real[n_sig]              one slot per block output
  xd      : Real[n_xd]               discrete states
  xc      : Real[n_xc]               continuous states
  blocks  : { kind, params, in[], out[], xd_slice, xc_slice, group }
  groups  : { rate, compute_delay, schedule[] }
Input   = Signal(slot) | Const(value) | Probe(id) | Time
Rate    = Discrete{period, offset} | Continuous | Triggered{event}
```

A probe is a circuit quantity from `add-probes-and-recorders` (node
voltage, branch current, device output), affine in the solver state for
linear probes. Names exist only in the builder and in the result; the
run loop uses indices.

### D2. Schedule and algebraic loops

Each block kind declares, per input, whether its output depends on that
input in the same evaluation (direct feedthrough). `Gain`, `Sum`,
`Limiter`, transforms: yes. `DelayBlock`, `Integrator`, `SampleHold`,
the state output of a filter: no.

- Order: Kahn's algorithm on the direct-feedthrough graph of the whole
  graph, ties broken by block index (deterministic).
- Execution of a group at an instant: phase 1 calls `output()` of its
  blocks in that order; phase 2 calls `update()` (state transition).
- When several groups fire at the same instant they are merged into one
  pass of the global order, so a fast group reading a slow group's
  signal sees the value of that instant.
- A strongly connected component of the direct-feedthrough graph with
  more than one block (Tarjan), or a self-loop, is an algebraic loop:
  the build fails, naming the blocks and suggesting a `DelayBlock`.
- Rate transitions need no block: a slot holds its last value (slow to
  fast is a zero-order hold; fast to slow samples at the slow tick).

- Alternatives: insertion order (today) — order-dependent results;
  iterating algebraic loops to a fixed point — hides modelling errors
  and has no convergence guarantee.

### D3. Discrete rates and ticks

A `Discrete{period, offset}` group fires at `t_k = offset + k·period`,
`k` an integer counter (no accumulated floating-point time). The event
queue lands a step on `t_k` on every engine. At `t_k`:

1. probes are read from the state at `t_k` (the left limit if another
   event coincides);
2. the group executes (D2);
3. its outputs take effect according to `compute_delay`:
   - `0`: at `t_k`;
   - a time `T_d < period`: at `t_k + T_d`, a second time event;
   - `one_sample`: at `t_{k+1}`.

`dt` passed to a block is `period`, exactly. The number of ticks in
`[t_start, t_end)` is the same on every engine.

### D4. Continuous rate

The blocks of the `Continuous` group form a subgraph with states `x_c`,
inputs from probes and outputs to controlled sources and comparators.
It is compiled into one residual contribution through the unified
device interface:

```
d/dt x_c − f_c(x_c, y_probe) = 0                (one row per state)
u_out − h(x_c, y_probe) = 0                     (one row per controlled source)
```

evaluated on the AD scalar, so the Jacobian with respect to the solver
state is exact and the coupling is implicit: no lag. A linear continuous
subgraph on a piecewise-LTI circuit keeps the system LTI per mask, so
the exact propagator advances compensator states and circuit states
together. A `Limiter` or `LookupTable1D` in the continuous group is
piecewise affine; its breakpoints are state events (an affine region in
the sense of `add-exact-propagator-strategy`). Other nonlinear
continuous blocks make the step an implicit one.

- Alternatives: integrate `x_c` explicitly between solver steps
  (today's coupling) — O(dt) lag error and a stability limit of its
  own; a separate ODE solver for the controller — two step-size
  controllers to reconcile.

### D5. Outputs to the circuit

| Sink | Semantics |
|---|---|
| `Modulator{carrier, duty slot, latch, dead time}` | the graph writes the duty register; the modulator, owned by the event queue, computes edges analytically (`next_edge`); `latch` ∈ carrier start, peak, both, immediate |
| `ControlledSource` | a signal sets a source value; discrete: constant between ticks (exact under zero-order hold); continuous: the algebraic row of D4 |
| `Gate` from a comparator | continuous rate: a state event `g = u − threshold` with hysteresis, located by the queue; discrete rate: changes at ticks |

### D6. Block library and the escape hatches

The POD blocks of `blockchain/blocks.hpp` become the library; the ten
missing ones are ported with the Python classes as the reference for
their tests. Two opaque kinds remain:

- `CBlock`: the existing ABI (`pulsim_cblock_step(in, n_in, out, n_out,
  t, dt, state)`), with a rate; it now runs on every engine because its
  hold is the slot and its instant is a tick event.
- `PythonBlock`: a callable with a declared rate; the GIL is taken at
  its ticks only. It is flagged opaque: no export, no periodic steady
  state, a diagnostics entry in the result.

`step_observer` and `b_extra_fn` closures keep working for one release
through an adapter that wraps them in a `PythonBlock`; a closure with
no rate on a variable-step engine is refused, as today.

### D7. C export

`graph.export_c(name)` writes one C99 file and a header:

```c
typedef struct { double xd[N_XD]; double sig[N_SIG]; } name_state;
void name_init(name_state *s);
void name_step_r0(const double *in, double *out, name_state *s);   /* one per discrete rate */
extern const double name_rate_period[], name_rate_offset[];
```

- Straight-line code in schedule order, block bodies generated from the
  same expressions as the kernel blocks (one definition, two
  renderings), no allocation, `math.h` only, a `NAME_REAL` macro
  (default `double`).
- For a single-rate graph a shim implements `pulsim_cblock_step`, so
  `add_c_block(lib=…)` can load the export. Compiled with
  `-ffp-contract=off` it must reproduce the in-kernel graph bit for bit;
  that round trip is the export's test.
- Export is refused, naming the block, for `PythonBlock`, a `CBlock`
  without source, and any block in the `Continuous` group.

### D8. State, snapshots and analyses

`xd`, `xc`, held signals and pending delayed outputs are part of the
solver snapshot (exact resume). `add-periodic-steady-state-shooting`
reads and sets them; `add-parameter-sensitivities` treats block
parameters as parameters.

## Applicability rules

Accepted: any graph whose direct-feedthrough graph is acyclic, whose
stateful blocks have a rate and whose blocks are all in the library or
one of the two opaque kinds.

Refused at build time, by name: an algebraic loop; a stateful block
with no rate; an unknown block kind (no Python fallback); a period
≤ 0 or an offset outside `[0, period)`; a `Continuous` block kind that
has no continuous form (`MovingAverageFilter`, `DelayBlock`,
`SampleHold`, `StateMachine`).

Refused at run time: a continuous-rate comparator on an engine with no
state-event location; a `PythonBlock` in an analysis that needs to read
state.

## Risks / Trade-offs

- Results change for chains that relied on insertion order → the build
  reports every pair whose order differs from insertion, once, with the
  slot names; a `DelayBlock` restores the old behaviour explicitly.
- Two renderings of each block (kernel and C) can diverge → both are
  produced from one expression per block, and the round-trip test runs
  on every block kind.
- The continuous subgraph as a device adds unknowns to the factor →
  typical compensators add fewer than ten; it is counted in the perf
  gate.
- Deprecating closures touches most examples → the adapter keeps them
  running for one release with a deprecation warning that names the
  replacement.
- `compute_delay` semantics differ between products → the default is
  `0`, documented, with `one_sample` named after what a microcontroller
  does when the duty is written at the next carrier period.

## Migration Plan

1. Characterisation: tick counts, order dependence, the measured
   defects as tests (expected to fail where Phase 0 has not landed).
2. Index slots and typed inputs inside the current `BlockChain` (no
   behaviour change).
3. Direct-feedthrough table, sorted two-phase schedule, loop error.
4. Rates on groups; ticks from the Phase 0 scheduling; the three
   throttle idioms removed.
5. Port the ten blocks; remove the Python evaluator and the fallback.
6. C export and the round-trip test.
7. After Phase 3: tick events from the unified queue on every engine;
   modulators; comparator state events; continuous subgraph as a
   device; graph state in the snapshot.
8. `ClosedLoop`, `MixedDomainBlockChain`, C blocks rebuilt on the
   graph; closures deprecated.

Rollback: steps 2–6 are confined to `blockchain/`; step 7 is behind the
`Simulation` object's control option.

## Open Questions

1. Default `compute_delay`: zero (ideal, proposed) or one sample (what
   most digital controllers do)?
2. Remove `step_observer` / `b_extra_fn` after one release, or keep
   them permanently as `PythonBlock` sugar?
3. Rate inheritance for stateless blocks (proposed) or an explicit rate
   on every block?
4. Should `export_c` also emit a multi-rate scheduler stub (a tick
   function with counters), or only the per-rate step functions?
5. Is `float` export (`NAME_REAL=float`) a requirement for the first
   release of the export, given that bit-identity can only be tested in
   `double`?
6. YAML: should the netlist carry the control graph (`control:`
   section) in this change, or with `add-immutable-circuit-ir`?
