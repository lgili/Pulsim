Every numbered section is one or more PRs and ends with the
verification set: `ruff check`, `pytest python/tests`, `ctest`,
`openspec validate --all --strict`, the stub check, and
`core/tests/perf/test_zero_alloc.cpp`. "Goldens" = the fixed-engine
checkpoint hashes and the differential runner of
`restore-reference-benchmark-harness` (the full suite stands in until it
lands).

## 0. Preconditions and baseline
- [ ] 0.1 `add-kernel-simulation-object` section 10 merged (`Simulation{Circuit, RunOptions}` with `Circuit` as a view)
- [x] 0.2 Record: `components()` dumps of every shipped example and harness circuit; build + analyse + assemble time of a 100-point sweep on the audit's buck and on the n = 1502 ladder; the list of builder private attributes set from Python (4) and read with `getattr` (7 sites) — `probes/components_baseline.json` (152 circuits, 136 load), `probes/sweep_probe.py`, BASELINE.md

## 1. Immutable snapshot of the builder
- [x] 1.1 `core/include/pulsim/ir/circuit.hpp`: ids, `Value` (literal only for now), `Node`, `Device`, `Scope` (root only), `InitialCondition`, `Alias`, `Circuit` handle over shared const data — the root scope is implicit (no `Scope` table until group 8)
- [x] 1.2 `CircuitBuilder::build()`; `Circuit::lowered()` reproducing `Graph` + `DevicePool` in the builder's insertion order
- [ ] 1.3 `sim::Simulation` takes `ir::Circuit`; `simulate()` calls `build()`; binding `pulsim.Circuit` — binding and `simulate(circuit)` done; the `Simulation` part waits for 0.1 (maintainer's slice decision, DECISIONS.md)
- [x] 1.4 Tests: `components()`, unknown indices, switch order and `initial_state()` equal between the builder and `build().lowered()` for every example and harness circuit
- [x] 1.5 Oracle: suite and goldens bit-identical; zero-alloc tests green
- [x] 1.6 Docs: `docs/mental-model.md` (builder → circuit → lowered); stubs

## 2. Stop mutating the caller's circuit (**BREAKING**)
- [x] 2.1 `Circuit::with_ties`; the preflight regularises the run's own circuit; `result.circuit` and `result.plan.ties` expose it
- [x] 2.2 Update the warning text (no longer "the ties persist on the builder")
- [x] 2.3 Tests: `builder.build().value_hash()` equal before and after `simulate()`; a second run with `auto_regularize=False` raises the original singular error; results of regularised runs unchanged
- [x] 2.4 Docs: CHANGELOG migration note; `AI_GUIDE.md` preflight section

## 3. Python side state into the IR
- [ ] 3.1 Initial conditions (`c0=`, `i0=`, `set_initial`) and aliases stored in the builder / IR; delete `_pulsim_ic`, `_pulsim_aliases`
- [ ] 3.2 C-block declarations as `ControlDecl`; delete `_c_blocks`
- [ ] 3.3 Shockley parameters as device fields; delete `_shockley_params`
- [ ] 3.4 Oracle: `test_initial_conditions_across_engines.py`, the alias tests, the C-block tests, `res.i` of Shockley diodes unchanged; a grep-based test that no module sets a private attribute on a builder

## 4. YAML round trip, clone, pickle, hashes
- [ ] 4.1 Schema table shared by loader and emitter (generated from the device field tables when `add-unified-device-interface` section 12 has landed; otherwise kept in `yaml/schema.hpp`), covering every registry kind including `shockley_diode`, `lauritzen_diode`, `nonlinear_capacitor`, `pmsm_mna` and the flux-table inductor
- [ ] 4.2 `yaml/emitter.hpp`: `Circuit::to_yaml()` (17 significant digits, defaults omitted, metadata kept); `Circuit::from_yaml()`
- [ ] 4.3 `structure_hash()`, `value_hash()`; `clone()`; pickle through YAML
- [ ] 4.4 Tests: for every shipped example, every harness circuit and one circuit per kind, `from_yaml(to_yaml(c))` has the same `value_hash` and simulates bit-identically; `to_yaml` is a fixed point after one round; a registry kind without a schema entry fails; provenance metadata survives
- [ ] 4.5 Oracle: `test_shipped_yaml_examples.py`, `test_yaml_composite_devices.py`, `test_yaml_integrator_option.py` green
- [ ] 4.6 Docs: YAML reference (new kinds), `AI_GUIDE.md`

## 5. Sweeps over a circuit
- [x] 5.1 `sweep(circuit, params=...)` and `monte_carlo(circuit, ...)` beside the `builder_factory` forms; the circuit is shared by the worker threads
- [ ] 5.2 Oracle: results equal the factory form; `test_sweep_path_aware.py` green; an 8-job thread sweep scales as audit row 23 (within the noise of the shared machine)

## 6. Named parameters and generic updates
- [x] 6.1 `Parameter`, `builder.param(name, value, unit)`; `Value` overloads on every `add_*`
- [x] 6.2 `Circuit::with_params`, `Circuit::with_value("dev.field", v)`; bounds checks from the parameter and from the field table; errors name device and field
- [x] 6.3 Deprecate `update_resistor_R`, `update_inductor_L`, `update_capacitor_C`, `update_voltage_source_V` (warning naming `with_value`)
- [x] 6.4 Tests: for one field of every kind, `with_value` equals rebuilding with the new literal, bit for bit; the receiver's `value_hash` is unchanged
- [x] 6.5 Docs: parameters page; CHANGELOG

## 7. Expressions
- [ ] 7.1 `ir/expr.hpp`: recursive-descent parser to a postfix program; evaluator; dependency ordering; cycle, unknown-name and non-finite errors naming device, field and text
- [ ] 7.2 YAML: `circuit.parameters`; string-valued fields as expressions; enumerated fields excluded through the schema table
- [ ] 7.3 SPICE importer keeps `.param` as IR parameters; `{2*L}` evaluates
- [ ] 7.4 Tests: fuzz against a Python reference evaluator (10⁴ random expressions, relative 1e-15); a circuit with expressions equals the literal circuit bit for bit; `to_yaml` writes the expression text, not its value
- [ ] 7.5 Docs: expression grammar; YAML reference; SPICE import notes

## 8. Scopes
- [ ] 8.1 `Scope` tree; `builder.scope(name, kind)`; scoped lookup of parameters
- [ ] 8.2 `subsystem.py` opens scopes instead of prefixing strings; flattened names unchanged (`/` separator)
- [ ] 8.3 YAML `circuit.subsystems` (nested); emitter writes the scope tree
- [ ] 8.4 Oracle: `test_subsystems.py` unchanged; the 302-branch subsystem circuit has identical `components()` names and results; round trip preserves the tree

## 9. Domains
- [ ] 9.1 `ir::Domain`; `builder.node(name, domain)`; YAML node entries with `domain`
- [ ] 9.2 Pin domain tables per kind; connection check; per-domain reference check in the preflight; units on probes
- [ ] 9.3 Tag the MNA PMSM's mechanical nodes (own PR)
- [ ] 9.4 Tag the thermal helpers' nodes (own PR)
- [ ] 9.5 Oracle: untagged circuits bit-identical; PMSM and electro-thermal tests unchanged; a thermal pin on an electrical node is refused by name
- [ ] 9.6 Docs: domains page; `docs/known-limitations.md`

## 10. Structure sharing
- [x] 10.1 `SymbolicPlan` held by structure hash and reused by `with_params` (requires `update-linear-time-assembly`; skipped with a note otherwise)
- [x] 10.2 Measure the 100-point sweep of 0.2 again; counter gate: one symbolic analysis for the whole sweep
- [x] 10.3 Oracle: sweep results equal

## 11. Remove the deprecated surface (one release later)
- [ ] 11.1 Delete the four `update_*` bindings and the string-prefix code of the subsystem proxy
- [ ] 11.2 Oracle: suite; stub check; `openspec/CORE_ROADMAP.md` status
