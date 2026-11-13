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
