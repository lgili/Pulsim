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
  immutable objects and enter `value_hash` by identity; the PWL
  current-source knots enter by content.
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
