# Decisions — add-immutable-circuit-ir

Delegated design decisions taken while implementing the first slice
(groups 0.2, 1 without the Simulation part, 2, 6, 10, 5, 3). Format:
question → option taken → evidence → how to reverse.

Scope of the slice (maintainer): task 0.1 (`add-kernel-simulation-object`)
is not a precondition; `simulate()` keeps its entry points and calls
`build()` / `to_builder()` internally; task 1.3's
`sim::Simulation takes ir::Circuit` waits for that change.

## 1. What is the lowered form?

- **Option:** a clone of the builder taken at `build()`
  (`Circuit::Data::lowered`, a `CircuitBuilder`), not a re-lowering of
  the IR tables into a fresh `Graph` + `DevicePool`.
- **Evidence:** the oracle of step 1 is bit-identity of every circuit on
  every engine and equal `components()` / unknown indices / switch order.
  A clone reproduces the builder's insertion order by construction —
  composites (MOSFET body diodes, transformer windings, PMSM phases,
  power-MOSFET sub-elements, subsystem-prefixed names) included — because
  nothing is replayed. A re-lowering pass would have to re-implement the
  side tables of 27 kinds (gate nodes, control nodes, couplings, PMSM
  machines, …) and keep them in step with every `add_*`. Measured: every
  shipped example and harness circuit has equal `components()`,
  `state_var_names()`, switch order and `initial_state()`
  (`python/tests/test_circuit_ir.py`).
- **Reverse:** replace `Data::lowered` by a lowering pass over
  `devices()`; `lowered()` keeps its signature.

## 2. How is a builder copied?

- **Option:** `Graph` gains an explicit `clone()` (its copy constructor
  stays deleted); the builder keeps its graph in a private `GraphSlot_`
  whose copy constructor calls `clone()`, so the builder's own copy
  constructor is the defaulted one — declared **private** — and
  `CircuitBuilder::clone()` is the only way to copy. A member added to
  the builder later is copied without anyone listing it.
- **Evidence:** pybind still sees a non-copyable builder
  (`std::is_copy_constructible` is false), so no binding changes
  behaviour; the move-only `static_assert` on `Graph` stays.
- **Reverse:** delete `clone()` and `GraphSlot_`.

## 3. What is a `Device`?

- **Option:** one `ir::Device` per lowered branch, in branch order: name
  (empty for anonymous branches), kind = index in `pwl::kKindCaps`, pins
  (from, to, then the gate / control nodes the pool keeps), fields from
  the kind's row of `ir::kKindFields`.
- **Evidence:** `components()` — the existing user-facing enumeration —
  is per branch; composite grouping needs the scopes of group 8.
- **Reverse:** group branches by composite once scopes exist.

## 4. The field table and the value hash

- **Option:** `core/include/pulsim/ir/fields.hpp`: one row per
  `DevicePool` kind (the capability table's order; a `static_assert`
  fails when a kind has no row), each field a name, an optional
  authoring alias (`R_ohms`, `C_farads`, `L_henries`, `lambda_`), a
  bound and a getter / setter on the pool's parameter variant. Only
  scalar values are fields. Resistances are stored as conductances; the
  `R` / `R_on` / `R_off` setters write `1 / R` exactly as the builder
  does, so the stored bits equal a rebuild. Table-valued parameters
  (flux tables, charge tables, Jiles–Atherton cores) are shared
  immutable objects; the PWL current-source knots enter `value_hash` by
  content. (Tables entered by identity until decision 16, which hashes
  them by content.)
- **Evidence:** the per-kind test (6.4) sets one field of every kind with
  `with_value` and compares against a rebuild with the new literal.
- **Reverse:** when `add-unified-device-interface` lands its field
  tables, generate `kKindFields` from them.

## 5. Values the pool keeps twice

- **Option:** `DevicePool::sync_derived_values(branch)` re-derives the
  copies after a field changes: a PMSM's machine record and its three
  phase entries; the `L_p` / `L_s` of every coupling naming an inductor.
  A power-MOSFET composite (channel and sub-elements) is refused by
  `set_value` / `with_value`: its element values are derived from the
  datasheet record (temperature coefficients), so editing one element
  would leave the record stale.
- **Evidence:** `test_circuit_ir.cpp` "set_value: equals rebuilding"
  (a coupled winding's coupling follows), the per-kind table test.
- **Reverse:** move the derivation into the unified device interface.

## 6. Errors

- **Option:** an unknown device, an unknown field, a malformed
  `"device.field"` and an out-of-bound value raise `ValueError`
  (`std::invalid_argument`) naming the device and the field; the
  unknown-field message lists the kind's fields.
- **Reverse:** a typed `NetlistError` subclass when one exists.

## 7. Bindings

- **Option:** `python/ir_bindings.cpp`, a separate translation unit
  like `dsed_bindings.cpp`; `bindings.cpp` gains three lines. Methods
  added to `CircuitBuilder` (`build`, `set_value`, `get_value`) are
  attached there to the already-registered class.
- **Evidence:** keeps the merge with the branch-form switch work (which
  edits `bindings.cpp`) mechanical; compiles in parallel.
- **Reverse:** move the block into `bindings.cpp`.

## 8. Where the preflight's ties go (group 2)

- **Question:** the preflight used to append `R_auto_iso_*` resistors
  to the caller's builder. Where do they go now?
- **Option:** `Circuit::with_preflight(opts)` runs the pure analysis
  on the circuit; when it finds nothing (or `auto_regularize` is off)
  it returns the receiver itself — same data, no copy. Otherwise it
  runs the builder's own fixed-point `run_preflight` on a clone and
  returns `(circuit, report, ties)`. `simulate()` and `compute_dc_op()`
  run that circuit; `result.plan` / `result.circuit` expose it. A
  tie's resistance is the option's value, not `1/G`, so
  `with_ties(ties)` re-adds it bit for bit.
- **Evidence:** regularised results equal the in-place preflight on
  pwl, trbdf2 and dsed (`test_regularised_results_equal_the_in_place_preflight`);
  `with_ties(r.ties)` equals `r.circuit` by value hash (C++ test).
  Callers that relied on the mutation, all fixed here: the hostile-
  circuit dsed test, `electrothermal` / `mmc_thevenin` (wrote to the
  caller's builder, now only the run's pool), the MMC driver's
  identity check (gets the caller's builder), `compute_dc_op`.
- **Reverse:** call `builder.run_preflight(opts)` in place (still
  public) before `simulate()`.

## 9. Named parameters before expressions (group 6)

- **Question:** the design's field value is `variant<Real, ExprId>`;
  expressions are group 7, not in this slice. What does a field hold?
- **Option:** `Value = variant<Real, ParamId>`: a literal or one named
  parameter (the one-identifier expression). Parameters live in the
  builder (`param(name, value, unit, lo, hi)`), fields are bound with
  `bind_param("dev.field", name)`; Python `add_*` binds the argument a
  `Parameter` was passed for (argument name = field name or alias,
  read from the signature or the pybind11 docstring). Which fields a
  parameter drives is structure (`structure_hash`); parameter values
  are values. `set_value` / `with_value` on a bound field makes it a
  literal again.
- **Evidence:** `test_with_params_equals_rebuilding` (value hash and
  pwl states bit-identical to rebuilding with the new values);
  `test_with_value_equals_rebuilding` for one field of all 26 scalar
  kinds; refusals name parameter, device and field.
- **Limits:** a composite's parameter argument (`add_transformer`'s
  `L_p`, an `add_power_mosfet` datasheet value) is refused by
  `bind_param` — the composite's name is not a device; the device is
  added before the binding is refused (the error says which argument).
  C++ `add_*` takes numbers only; C++ callers bind with `bind_param`.
  An overloaded binding (`add_nonlinear_capacitor`) is not wrapped (one
  signature cannot stand for its overloads in the stub check) and,
  since a `Parameter` is not a float, refuses one with `TypeError`
  instead of binding nothing; bind its field with `bind_param`.
  The wrappers carry the native signature (`__signature__`); the stub
  is generated from the native methods (`PULSIM_STUBGEN_NATIVE`).
- **Reverse:** group 7 replaces `ParamId` by `ExprId`; a bare
  identifier parses to the same binding.

## 10. The four `update_*` methods

- **Option:** deprecated in Python (`DeprecationWarning` naming
  `with_value("dev.F", value)` and `set_value`); internal callers
  (`sweep_path_aware`, `monte_carlo_path_aware`,
  `periodic_steady_state`'s input sensitivity) use `set_value`, which
  writes the same field setter (`G = 1/R`) and, beyond `update_*`,
  checks the bound and re-derives a coupled inductor's `L_p` / `L_s`.
  The C++ `DevicePool::update_*` used by `refactor_parametric` stay.
- **Reverse:** drop the wrappers in `_circuit_ir.install`.

## 11. Review after group 2 (fresh-context reviewer), and what changed

- A Circuit dropped the builder's Python side tables: Shockley loss /
  thermal parameters now come from the device's fields
  (`shockley_params` reads `get_value`; the `_shockley_params` copy and
  its recording wrapper are gone — group 3's Shockley item);
  `builder.build()` refuses a builder with C blocks (`UnsupportedError`)
  until they are `ControlDecl`s (group 3); `simulate` snapshots with
  the unchecked `_snapshot()` and keeps the builder's side state.
- `simulate(circuit, mmc_arms=...)` always failed the arms' builder
  identity check: refused up front, naming the builder.
- `set_value` accepted values `add_*` refuses: `ir::check_device` runs
  the model's `validate` and the IGBT tail rules on a copy, and refuses
  a field the device's law does not read.
- `Graph::id()`'s lazy cache could be written by two threads on a
  regularised circuit: `make_data_` fills it.
- `Circuit.components()` called the class attribute by name, so a
  Python patch of `CircuitBuilder.components` would run on the shared
  lowered builder: the native function is captured at module init.

## 12. Where the symbolic plan lives (group 10)

- **Question:** "held by structure hash": a process-wide map from
  `structure_hash` to an analysis, or a plan carried by the circuits?
- **Option:** carried. `Circuit::Data` holds a
  `shared_ptr<pwl::SymbolicPlan>`; `with_params` / `with_value` hand
  the receiver's plan to the derived circuit; `build()`, `with_ties`
  and a regularising `with_preflight` start a new one. A run's cache
  gets it through `PwlStateSpaceCache::share_symbolic`, adopts the
  stored analysis after the pattern check (`adopt_symbolic`), or
  analyses and offers its own. A plan whose pattern does not match is
  dropped for that cache (it analyses privately) — never a wrong
  factor.
- **Evidence:** the 100-point sweeps (BASELINE.md "after") count ONE
  symbolic analysis in the circuit form against 100 in the factory
  form, with bit-identical final states; `test_circuit_sweep.py`
  gates the count (30-point grid: 30 → 1) and that another structure
  does not share.
- **Why not a global map:** it is static mutable state shared by
  unrelated circuits (the thread-safety audit of this slice forbids
  it), and it would need eviction. The carried plan dies with its
  circuits.
- **Thread safety:** the plan is the one mutable object an immutable
  circuit reaches: a mutex guards its two slots, written once each.
  `SymbolicAnalysis` is read-only after `export_symbolic` (adoption
  copies it).
- **Reverse:** drop `share_symbolic` calls; caches analyse privately.

## 13. The kernel path of a circuit sweep (task 5.2)

- **Question:** the per-point work of `sweep(circuit, ...)` held the GIL
  (8 threads: ladder 1.1×). Move it into the kernel how, without losing
  what `simulate()` does around a run (refusals, warnings, result
  attributes)?
- **Measured cause:** not Python. `simulate()`'s own Python is ~0.03 ms
  of a 1.96 ms ladder point; `with_value` (0.46 ms), `to_builder`
  (0.23 ms) and `with_preflight` (0.14 ms) are C++ that ran with the
  GIL held (BASELINE.md "Task 5.2").
- **Option:** (1) those bindings release the GIL (helps every caller,
  the Python path of the optimiser included); (2) `_run_pwl_points`
  (ir_bindings.cpp): for a table of points (rows) over names
  (parameters or `"device.field"`), each point is derived, regularised,
  cloned, cached (sharing the symbolic plan) and run by `n_threads`
  native threads with the GIL released for the whole batch. The run
  recipe is not re-derived: point 0 runs through `simulate()` with a
  context variable set, and `simulate()` records at its pwl run site
  what it handed the kernel (options, dt, switch_fn, initial state,
  start_from_dc_op, the refresh flag). Each kernel point is then
  finished by `simulate()` with the result injected (a second context
  variable): its warnings, `result.plan`, `result.circuit` and every
  attribute come from the same code as a Python-path point. Two small
  additive hunks in `simulate()`.
- **Preflight once per structure:** `analyze_preflight` reads topology,
  branch kinds and device kinds only (`pwl/preflight.hpp`), never a
  value; `with_params` cannot change the structure (a parameter cannot
  drive a branch-form `R_on` / `g_off`). So the base is regularised
  once and every point is derived from the regularised base (ties are
  appended after every device with the same names and resistance in
  both orders: bit-identical). `with_value` CAN change the structure —
  a branch-form `R_on` or `g_off` to or from 0 is part of
  `structure_hash()` — and such a point (structure hash differs from
  the base's) is derived from the unregularised base and regularised
  by itself (`counters["preflights"]` counts it).
- **Eligibility:** pwl engine (what point 0 actually ran), no Python in
  the step loop (`step_loop_python_callbacks == ()`), no b_extra /
  observer / should_continue / live stream / resume / MMC arms /
  on_cache / progress, `enable_nonlinear_refresh` not False, backend
  not `"process"`. Anything else keeps the Python path, unchanged. A
  point that fails in the kernel is re-run through `simulate()`, which
  raises or records it exactly as the Python path does.
- **Thread safety (read):** nothing reachable from the base is written.
  Derivations clone the lowered builder (`Graph::clone` reads the id
  cache `make_data_` filled; no other `mutable` member, `static` or
  `thread_local` in builder, graph, pool, IR or preflight headers); the
  `SymbolicPlan` is mutex-guarded and written once per regime (point 0
  fills it before the batch); each point's builder, cache and result
  are its own. `adapt_switch_fn` hands threads a copy of a native
  `SwitchSchedule` (const, no mutable state) or raw pointers to a native
  PWM object the caller keeps alive; a Python callable never reaches
  the batch (eligibility).
- **Evidence:** `test_circuit_sweep.py`: Python `run_transient` calls =
  1 for 12 points, symbolic analyses = 1, preflights = 1, every point
  bit-identical (serial and 4 threads); a regularised circuit with a
  capacitor IC (ties, plan, value hash equal); a structure change
  (preflights = 2); a Python hook keeps the Python path; a failing
  point records the same message. Speedups in BASELINE.md.
- **Reverse:** drop `_run_circuit_points` in sweep.py (the circuit form
  falls back to `_run_jobs` on threads); the GIL releases stay.

## 14. Control declarations (group 3.2)

- **Question:** C blocks (`add_c_block`) and control-graph-driven
  sources (`ControlGraph.drive_source`) live as Python attributes of the
  builder instance (`_c_blocks`, `_graph_driven`), which a Circuit
  dropped: `build()` refused C blocks and `simulate(builder)` copied the
  builder's `__dict__` onto the run. Model them in the IR?
- **Option:** no model — an opaque `ir::ControlDecl {name, kind,
  content_hash, handle}` stored by the builder (so every derivation and
  `to_builder()` copies it) and appended by `Circuit::with_controls`
  (shares the symbolic plan: a control never changes the matrix
  structure). The handle is a Python object held by a shared pointer
  whose deleter takes the GIL (a derived circuit can die on a sweep
  worker). `name` and `kind` enter `structure_hash`, `content_hash` the
  `value_hash`; a circuit without controls hashes as before. The
  content hash is FNV-1a 64 of the block's name, ports and rate (pinned
  in a test; `hash()` of a str differs per process); a C block's step
  function is opaque, so two blocks differing only in code hash equal —
  they are not serialisable either (group 4 refuses them by name).
- **Why not delete `_c_blocks` / `_graph_driven` now:** they are set in
  `c_block.py` and `blockchain.py`, which another branch is moving onto
  the control graph. The patched `build()` converts both lists into
  declarations, every reader (`simulate`, the trbdf2 / dsed blockers,
  `sensitivity`, `periodic_steady_state`) reads them through
  `c_blocks_of` / `graph_driven_of` (circuit or builder), and the
  grep test `test_no_builder_side_state.py` allow-lists exactly those
  two assignments. When the consolidation lands it registers
  `ControlDecl`s directly and the allow-list goes.
- **Initial conditions:** the IR's `InitialCondition` is per device (a
  capacitor's voltage across it, an inductor's current through it);
  the KVL-loop refusal stays where the integration put it — in
  `initial_state()`, which `Circuit.initial_state()` and every run call —
  so `build()` still accepts such a builder (an existing test pins
  that; refusing at `build()` was tried and reverted).
- **Reverse:** `build()` refuses C blocks again; delete `with_controls`.

## 15. Review after group 3 (fresh-context reviewer), and what changed

- **C-block state is per handle, shared by derived circuits.** A C
  block keeps its tick counter and outputs in its handle
  (`c_block.py`), so a second run of the same handle starts with the
  counter of the first (the block never fires again): pre-existing for
  `simulate(builder)` twice, newly reachable through a circuit sweep.
  `sweep(circuit)` / `monte_carlo(circuit)` with C blocks are refused by
  name (`UnsupportedError`, feature `c_blocks`); the per-run reset
  belongs in `c_block.py` (another branch) — reported to the
  coordinator.
- **The recipe leaked into a kpi's simulate().** Point 0's recipe was
  recorded while its `kpi_fn` ran, so a `simulate()` inside the kpi
  overwrote dt / t_end for every kernel point. The context variables
  are now set around the `simulate()` call only (test).
- **`compute_dc_op(strategy="settle")` doubled every C block**: it still
  copied the builder's `__dict__` onto a run that already carried the
  blocks as controls. The copy is gone; the grep test now also flags
  `builder.__dict__.update(...)` / `vars(builder)`.
- **Missed reader:** `design.py`'s Evaluator reads
  `getattr(builder, "_graph_driven")`, which a `Circuit` returned by
  `build(design)` does not have. `design.py` belongs to another agent
  this slice; it should call `_circuit_ir.graph_driven_of`. Reported.
- Controls enter the structure stream behind a tag and a count; the
  backend label is `"kernel"` only when a point ran in the kernel; a
  thread that cannot be started leaves its points to the others.

## 16. The YAML round trip writes the lowered form (group 4)

- **Option:** `core/include/pulsim/yaml/emitter.hpp` (free functions
  `ir::to_yaml(circuit)` / `ir::from_yaml(text)`; Python
  `Circuit.to_yaml()` / `Circuit.from_yaml()`). The document is the
  LOWERED circuit: nodes in unknown order, one entry per registry-kind
  instance in branch order (a PMSM's three phase branches as one entry),
  then the side records (couplings, branch-form order, parameters,
  bindings, initial conditions, aliases, power-MOSFET records, control
  declarations). `from_yaml` adds every branch first (so a sensed branch
  named later resolves), then the pool entries in branch order, so the
  pool's counters (source, inductor and branch-form slots) come back in
  the same order. Additions only: a `derived` flag on `FieldSpec`
  (reciprocal rows), `friend struct ir::YamlIo` on the builder.
- **Schema table:** a kind's keys are its non-derived rows of
  `kKindFields` plus its pins and its table / law / waveform keys; its
  name is `kKindYaml`, checked against `kKindCaps` by a `static_assert`,
  and a kind without a reader fails to compile (`static_assert` in the
  visitor). The authoring loader (`yaml/loader.hpp`, composite types
  such as `mosfet` with a body diode) is unchanged: composites do not
  survive lowering one to one, so the round-trip format is a second,
  exact dialect, not the authoring one.
- **Stored members, not authoring values.** A resistor is written as
  `G`, a branch switch as `g_on` plus its `R_on` in `branch_form`:
  writing `R` would read back as `1/(1/G)`, which differs from `G` by an
  ulp for some values. Defaults (a value-initialised `Params`) are
  omitted bit for bit.
- **Numbers shortest-round-trip, not `%.17g`.** `std::to_chars` writes
  the shortest text that reads back to the same double (never more than
  17 significant digits; `0.1`, not `0.10000000000000001`), and
  `std::from_chars` reads it; both are locale-free. The guarantee asked
  of 17 digits (exact round trip, fixed point after one round) holds and
  is tested; the text is what a person would type.
- **Tables hash by content.** `value_hash` read a flux / charge table's
  address, so the same circuit hashed differently per process and after
  a round trip. It now hashes the knots (and every Jiles-Atherton
  scalar). Changes the value_hash of circuits with such tables only.
- **Not serialised:** a control with a handle (a C block: a Python
  callable) is refused by name; graph-driven declarations (no handle)
  round-trip with their content hash. `from_yaml` validates syntax,
  kinds, keys, pin counts, node and branch names, and the models' own
  `validate` (the pool's add path); it does not re-apply authoring
  bounds (a branch-form `g_off = 0` is legal there and would fail the
  field bound).
- **Pickle / clone:** pickle is the YAML text; `clone()` returns another
  handle to the same data. `sweep(circuit, backend="process")` is
  therefore allowed (its workers do not share the symbolic plan).
- **Evidence:** `test_circuit_yaml.py` — one cell per kind (the
  engine-conformance cells, plus the table / vdmos / power-MOSFET and
  branch-form variants) on pwl, trbdf2 and dsed: equal hashes, fixed
  point, bit-identical `times` / `states` or the same refusal; the 25
  shipped examples; the 111 v1 harness circuits that the reader loads
  (15 use unmapped v1 blocks). Pinned: `structure_hash` 0x57BF94FE7FA50967,
  `value_hash` 0x746D6010F3DC01E0 for the V-R-C cell (C++ and Python).
- **Reverse:** when `add-unified-device-interface` lands generated field
  tables, `kKindYaml` and the reader's visitor come from them.

## 17. Expressions are derived parameters (group 7)

- **Option:** an expression defines a PARAMETER (`ir::Parameter::expr`,
  compiled to a postfix `ir::Expr` kept beside it), never a field
  directly. Fields keep binding to parameters exactly as in group 6; a
  YAML or SPICE field expression becomes a derived parameter named
  `<device>:<field>` bound to that field. One mechanism, one hash rule
  (the text is structure), one round-trip path.
- **No cycles by construction in the builder:** `param_expr` resolves
  only parameters declared before it, so `set_param` re-evaluates the
  derived parameters after the changed one in one forward pass. Files
  (YAML `circuit.parameters`) may list them in any order: the loader
  orders them (`ir::dependency_order`) and refuses a cycle naming it.
- **Grammar without unit suffixes.** `10k` is refused inside an
  expression (`bad number "10k" (no unit suffixes)`): a suffix grammar
  differs between SPICE dialects and would make `1m` ambiguous (milli /
  mega). SPICE plain values keep their suffixes; only `{...}` uses this
  grammar.
- **Evidence:** fuzz of 10^4 random expressions (all operators and
  functions, domains kept real) against a Python reference: worst
  relative error 0 (bound 1e-15) — the evaluator does the same IEEE
  operations in the same order. A circuit with `2*Lbase*1.5` and
  `Vin/4` simulates bit for bit like its literal twin.
- **Reverse:** scopes (group 8) widen name lookup to the enclosing
  scopes; the grammar and the program are unchanged.

## 18. Final review (fresh-context reviewer, groups 4 and 7), and what changed

- **with_params checked derived bounds on half-updated values** (D =
  Ton / T moved with both was refused or not depending on dict order):
  `CircuitBuilder::set_params` sets every literal, then evaluates and
  checks once; `Circuit::with_params` calls it (signature unchanged).
- **from_yaml UB on hostile integers** (`body: 1e300` cast to int):
  integer keys are parsed as integers with a range.
- **Untyped yaml-cpp exceptions / zombie-node access** on a map where a
  list belongs, a scalar device, a phase that is not a map, a power
  MOSFET without its diode records: shape checks, and any remaining
  `YAML::Exception` or unknown-name `out_of_range` becomes
  `invalid_argument` ("from_yaml: ..."). `branch_form` must name a
  switch or a diode.
- **The loader's number test** (from_chars) read `.inf` / `+5` as
  expressions; it now asks yaml-cpp's own decode, so every value that
  loaded as a number still does and creates no parameter.
- **`pi`**: a parameter named `pi` now shadows the constant (it was the
  other way round, and SPICE's scope disagreed with the builder).
- **Deep parameter chains**: `dependency_order` is iterative (a 20 000
  link chain loads); `_eval_expr` checks names / values lengths.
- **Accepted, documented:** `spice_to_builder` declares top-level
  `.param` lines as parameters (CHANGELOG migration note). `from_yaml`
  takes field values as written (no authoring bounds; see 16).

## 19. Needs recorded by the optimisation slice 5 (Evaluator(circuit=...))

What the design evaluator needed from the IR and used instead
(add-design-optimization-workflow DECISIONS 20):

* **A picklable `Circuit`** (task 4.3): `evaluate_many(backend='process')`
  ships the evaluator to worker processes; the circuit form was refused
  there by name and the factory form was the process path. Since the
  merge of batch 10 into this branch a `Circuit` pickles, so the
  circuit form runs on the process backend (test: equal to threads);
  a part beside it that does not pickle (a SwitchSchedule as
  `switch_fn`) is still refused by name.
* **Builder side state in the IR** (task 3): `_graph_driven` (a control
  graph's `drive_source`) is a Python attribute of the builder and does
  not survive `build()`, so `simulate(circuit)`,
  `periodic_steady_state(circuit)` and the evaluator cannot refuse a
  graph-driven source given as a `Circuit`.
* Not needed: a GIL-free batch `with_params` or a cheaper clone --
  `with_value` costs 2.5 us and `to_builder` 1 us per design, against
  ~105 us of Python per evaluation.

(Numbered 13 on the optimisation branch; renumbered at integration batch 10
behind this branch's 13–18. Since 14 the graph-driven sources survive
`build()` as ControlDecls; since 16 a `Circuit` pickles (its YAML text).)

## 20. Scopes (group 8)

- **Option:** a scope is a node of a tree the builder keeps
  (`ir::Scope{name, parent, kind, path}`, `open_scope(path, kind)` /
  `close_scope()`, Python `with b.scope(path, kind=...)`), and the
  circuit carries it (`Circuit.scopes()`). `define_subsystem(...)
  .instantiate` opens one per instance (`kind` = the definition's name);
  YAML `circuit.subsystems` opens one per block. Parameters declared
  while a scope is open belong to it (`Parameter::scope`) and are stored
  under its path (`leg_a/sm17/Ron`).
- **Membership is by name, not by tag.** A device or node belongs to
  the deepest scope whose `path + "/"` prefixes its flattened name
  (`scope_of`). The flattened names are the old prefix names (task 8.2:
  unchanged), unique by construction, so this is total and
  deterministic, and the builder needs no per-element scope column: the
  lowered builder, every index and every stamp are those of the
  hand-prefixed twin. A root element hand-named `leg_a/x` while a scope
  `leg_a` exists is attributed to it — the same ambiguity the prefix
  scheme always had.
- **Names are still made by the proxy / the YAML block**, with the same
  join (`path + "/" + local`) as `builder.scoped_name(local)`; the C++
  builder does not rewrite names inside `add_*`. Doing it there would
  apply the rule a second time to every composite that passes a name
  it already resolved (power-MOSFET member names, sensed branches),
  which the audit of `circuit_builder.hpp` could not rule out; task 11.1
  deletes the proxy's string code a release later, when the device
  interface migration owns the composites.
- **Lookup is lexical, outward only.** An expression in scope S reads
  `S/name`, then each enclosing scope, then the root: an inner name
  shadows an outer one; an outer scope cannot read into an instance
  (`a/b` is a division in the grammar). Consequences: a cycle across
  scopes cannot be written (a parent never sees a child), and within
  one scope the builder's declared-before rule (Python) or the block's
  dependency order (YAML, a cycle refused naming its members) holds. A
  child `Ron = "2*Ron"` reads itself and is refused as a cycle, in YAML
  and in `builder.param` alike; write the outer value under another
  name. `bind_param` / `set_param` look a name up the same way (the open
  scope's own first, then outward, then the full name).
- **One definition, N instances.** The Python `Subsystem` is the
  definition (body + ports + defaults), each instance a scope whose
  `kind` names it. The IR stores the instances, not the definition
  (design D4: `to_yaml` writes nested instances; `kind` is kept so a
  later change can factor them). A path whose last segment already
  instantiated a definition is refused (`already used by an instance`);
  a plain path segment can be re-entered.
- **Hashes:** the tree (paths and kinds, in creation order) and each
  scoped parameter's scope id enter `structure_hash` only when a circuit
  has scopes: a circuit without scopes hashes as before (V-R-C pins
  hold). A scoped circuit and its flattened twin have the same nodes,
  devices, `components()` and results; they differ in `structure_hash`
  (and so `value_hash`, which folds it in), in `scopes()`, and in
  whatever parameters the scoped one declares.
- **YAML (lowered dialect):** `scopes: [{path, kind}]` before
  `parameters`, parents first; a scoped parameter carries `scope:`.
  `from_yaml` refuses a scope listed twice or before its parent.
- **Evidence:** `test_circuit_scopes.py` — a 100-cell arm (302
  branches, two levels: `leg_a/sm{i}`) built through `define_subsystem`
  with scoped parameters and shadowing decoys at the root equals its
  hand-prefixed twin in `components()` and simulates bit for bit on
  pwl, trbdf2 and dsed;
  `with_params` addresses one instance or the arm; the tree survives
  YAML and pickle; every authoring device type and every shipped
  example wrapped in a YAML subsystem block renames every node and
  device (none missed) and the examples simulate bit for bit.
  `test_subsystems.py` unchanged. The twin test requires all three
  engines to run (dsed: 21 samples, trbdf2: 1004, pwl: 201).
- **Limits.** A YAML block's `control:` (CCVS / CCCS) names a branch of
  the block: ports carry nodes, not branches, so a block cannot sense an
  outer branch. A block name is one path segment (`a/b` is refused; nest
  blocks). `with builder.scope(p)` composes with `instantiate` (the
  instance lands at `p/<path>`); inside a body, a scope opened by hand
  and left open refuses `instantiate` through the body's proxy.
- **Reverse:** delete `open_scope` / `close_scope` / `scopes_` and the
  `scopes:` key; the proxy's names never changed.

## 21. Slice 2 leftovers and batch 11 review (#5, #8)

- **One schema table.** `ir::kKindYaml` / `kKindFields` (compile-time
  checked against `StoredKind`) serve the emitter and, through
  `YamlIo::read_entry`, the authoring loader: a `type:` naming a
  registry kind without an authoring reader (`shockley_diode`,
  `lauritzen_diode`, `nonlinear_capacitor`, `pmsm_mna`) is read with the
  emitter's entry. Test: every registry kind loads in a netlist.
- **Bounds at load.** `from_yaml` applies the `kKindFields` bounds that
  `with_value` applies, naming device and field. Not every `add_*`
  applies them (`add_resistor` takes R < 0, `add_pwm_voltage_source`
  takes duty > 1), so such a circuit writes a document `from_yaml`
  refuses; a pickle (text this library wrote) reads with the bounds off.
  Making every `add_*` enforce the table is the cleaner end state and a
  behaviour change for users; not done here.
- **No NaN / inf in a document** (batch 11 review #5). `to_yaml` refuses
  a non-finite value naming the entry and the key; `from_yaml` refuses
  `.nan`, `.inf`, `nan`, `inf`. The one legitimate infinity, an ideal
  branch-form switch's `g_on = 1/R_on`, is no longer written: the reader
  derives it from `branch_form.R_on` exactly as `add_ideal_switch` /
  `add_diode_branch` do (same `value_hash`, tested).
- **Numbers stay shortest-round-trip** (`std::to_chars`), not `%.17g`:
  the same exact round trip (tested), shorter text a person would type.
- **Lazy tables.** `nodes()`, `devices()`, hashes are filled once on
  first use (`std::call_once`, into a local then moved, so a throw leaves
  nothing half-filled); a derivation pays only its clone.
- **Libm in tests** (batch 11 review #8). Random expressions of `+ - * /`,
  unary minus, `abs`, `sqrt`, `min`, `max` are compared bit for bit
  (IEEE-exact on every platform); every libm function (`exp log log10
  sin cos tan atan sinh cosh tanh pow ^ atan2`) is compared on
  bit-identical arguments to 4 ulp of the Python reference
  (`LIBM_ULPS`), which covers glibc, macOS and MSVC.

## 22. Domains (group 9)

- **Option:** a node carries `ir::Domain` (electrical, thermal,
  mechanical; magnetic reserved for add-permeance-magnetic-domain and
  refused). It is declared (`builder.node(name, domain)`, YAML
  `{name, domain}`) or implied by a typed pin (PMSM `omega` rad/s,
  `theta` rad); every other node is electrical. All domains stay in the
  one MNA system: the tag never reaches a stamp, so a tagged circuit and
  its untagged twin give bit-identical samples (tested on pwl, trbdf2;
  a DC machine with its shaft tagged likewise).
- **Pin table, coarse on purpose.** Two classes, not a per-kind table:
  electrical-only kinds (switch, diode, MOSFET / IGBT level 1, the
  nonlinear diodes, saturable inductor, nonlinear capacitor, PMSM
  phases) and domain-generic kinds whose pins come in pairs
  (`from`/`to`, then each control pair), each pair within one domain. A
  VCVS sensing a temperature and driving an electrical node is legal by
  construction. A violation is refused at `build()` naming device, pin
  and node (`device "D1": pin from (electrical) is connected to node
  "tj", which is thermal (declared)`). A device whose physics spans two
  domains in one pair (none shipped) would need its own row.
- **Preflight per domain:** `with_preflight` labels each finding with
  its subnet's domain; a floating thermal network reads
  `thermal-domain: ...` (`finding.domain`).
- **Units:** `Circuit.unit_of(node)` (V, degC, rad/s, rad). Result
  traces carry no unit yet (known-limitations); task 9.2's "units on
  probes" is met at the circuit, not at the result.
- **Hashes:** a declared non-electrical tag is structure (sorted by
  node index, hashed only when present): an untagged circuit and one
  that declares nodes electrical hash as before (V-R-C pins hold).
  Implied tags follow from the kinds already hashed.
- **Helpers (9.3, 9.4):** the PMSM's speed/angle nodes are implied by
  its pins (no code in the helper); `add_foster_network`,
  `add_cauer_thermal_network` and `add_shared_heatsink` declare their
  nodes thermal, only on a real `CircuitBuilder` (a subsystem proxy gets
  names as before). Done here, not in separate PRs: each is a few lines
  and the oracle (PMSM and electro-thermal suites unchanged) runs with
  the check.
- **Evidence:** `test_circuit_domains.py` (11 tests) and the C++ `[ir]`
  case; the full suite with the helpers tagging.
- **Reverse:** delete `node(name, Domain)`, `resolve_domains_` and the
  YAML key; nothing else reads the tag.

## 23. Overhead of the IR path (final measurement)

`probes/ir_overhead.py`, Apple M-series, Release, interleaved, 15 reps
(min / median):

| circuit | simulate(builder) | build() + simulate(circuit) | build() | with_value per point |
|---|---|---|---|---|
| small, 6 branches, 2000 steps | 0.278 / 0.294 ms | 0.278 / 0.292 ms | 1.6 us | 2.7 us (0.9 % of a run) |
| ladder, 1502 branches, 200 steps | 4.697 / 8.830 ms | 4.506 / 7.112 ms | 328 us | 190-337 us (4-7 % of the min run) |

- `build() + simulate(circuit)` costs what `simulate(builder)` costs (equal minimums;
  the medians are noise on a shared machine).
- Lazy tables cut `with_value` on the ladder from 657 us to one builder
  clone (190 us measured alone; `to_builder` 294 us with the Python
  wrapper). The clone copies five name tables (graph node and branch
  names, `node_map_`, `branch_names_`, `branch_index_`) and the
  per-node branch lists, ~0.13 us per branch.
- That is above "a few percent" only for very short runs (200 steps of
  a 1502-branch ladder). Removing it means sharing the immutable name
  tables between a circuit and its derivations (copy-on-write in
  `CircuitBuilder`): a builder refactor, left for the device-interface
  migration that owns the builder's storage. Recorded as the one
  overhead finding of this change.
