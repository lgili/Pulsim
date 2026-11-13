## Context

Paths: PY = `python/pulsim/__init__.py`, BN = `python/bindings.cpp`,
`core/…` = `core/include/pulsim/…`.

Today's layers:

```
YAML v2 (circuit: {nodes, devices}, simulation:)  ─┐
SPICE importer (Python)                            ├─►  CircuitBuilder  ──►  Graph + DevicePool  ──►  cache / drivers
Python add_* calls, Subsystem proxy                ─┘     (mutable)            (mutable, flat)
                                                          + Python attributes: _c_blocks, _pulsim_ic,
                                                            _pulsim_aliases, _shockley_params
```

- `CircuitBuilder` (`core/builder/circuit_builder.hpp`, 1900 lines)
  maps string names to indices, validates parameters by name and
  forwards to `Graph::add_branch` and `DevicePool::add_*`. It exposes
  `components()` (`DeviceInfo`, `:1659`), aliases (`AliasTarget`,
  `:1717`), `initial_state()` and `run_preflight` (`:850`), which
  appends ties when `auto_regularize` is set.
- `Graph` + `DevicePool` are what every engine consumes. They are flat:
  one namespace, literal parameter values, no scope, no domain.
- The YAML loader builds a `CircuitBuilder`; there is no emitter.
- `subsystem.py` is a proxy that rewrites the first positional string
  (the device name) and the node names with a path prefix
  (`SEP = "/"`), so "define once, instantiate many" works, and the
  instance structure is then lost.

What is well designed and stays: named refusals at construction
("error names the diode"), scoped subsystem names, `components()`,
`explain_singular` row names, the preflight findings, the builder's
`add_*` vocabulary (every tutorial and test uses it).

Owner notes that are requirements:

- *gui-swallows-backend-refusals*: construction and lowering errors are
  typed (`NetlistError`) and name the element.
- *shipped-parts-are-synthetic*: provenance attached to a part must
  survive a round trip; the IR carries device metadata it does not
  interpret.
- *simulate-engine-dispatch-order*: the preflight changes the state
  vector's length, so it must run before initial conditions are sized;
  with an immutable circuit the order is fixed by construction
  (regularise, then lower, then size).
- *switchstatemask-takes-a-size*: the number and order of switches is a
  property of the lowered circuit and is exposed as such.

## Goals / Non-Goals

Goals:

- A circuit value that is immutable, cheap to copy, comparable,
  serialisable and shareable across threads.
- Named parameters and expressions; re-parameterisation without
  rebuilding or re-analysing.
- Hierarchy and domains as data, not as naming conventions.
- A complete YAML round trip.
- No change to how circuits are written today and no change to
  results.

Non-Goals:

- No new solver for non-electrical domains: one MNA system, as today.
- No SPICE export (see Open Questions).
- No graphical / schematic data in the IR (the schematic renderer keeps
  its own layout inputs).
- No control-graph representation: the IR stores control and gate
  *declarations* opaquely for `add-compiled-control-graph`.
- No parameter derivatives (`add-parameter-sensitivities`).
- No change to `DevicePool` / `Graph` as the lowered form in this
  change.

## Decisions

### D1. Two-level representation: IR and lowered form

```cpp
namespace pulsim::ir {

enum class Domain : std::uint8_t { Electrical, Thermal, Mechanical, Magnetic };

struct ScopeId  { std::uint32_t v; };   struct NodeId   { std::uint32_t v; };
struct DeviceId { std::uint32_t v; };   struct ParamId  { std::uint32_t v; };
struct ExprId   { std::uint32_t v; };

using Value = std::variant<Real, ExprId>;                    // literal or expression

struct Scope     { std::string name; ScopeId parent; std::string kind; };   // kind: the subsystem definition's name
struct Node      { std::string name; ScopeId scope; Domain domain; };
struct Parameter { std::string name; ScopeId scope; Value value; std::string unit;
                   std::optional<Real> lo, hi; };
struct Device    { std::string name; ScopeId scope; std::uint16_t kind;      // index in the device registry
                   small_vector<NodeId, 4> pins;
                   small_vector<std::pair<std::uint16_t, Value>, 8> fields;  // (field index, value)
                   Metadata meta; };                                          // provenance, uninterpreted
struct InitialCondition { DeviceId device; std::uint16_t what; Value value; };
struct Alias     { std::string name; AliasTarget target; };
struct ProbeDecl; struct GateDecl; struct ControlDecl;        // owned by their changes; stored here

class Circuit {                                               // handle: shared_ptr<const Data>
public:
    // queries
    std::span<const Node> nodes() const;  std::span<const Device> devices() const;
    std::span<const Scope> scopes() const; std::span<const Parameter> parameters() const;
    std::uint64_t structure_hash() const;   // topology, kinds, scopes, which fields are expressions
    std::uint64_t value_hash() const;       // structure + every evaluated value

    // derivation: each returns a new Circuit; the receiver is unchanged
    Circuit with_params(std::span<const std::pair<std::string_view, Real>>) const;
    Circuit with_value(std::string_view device_dot_field, Value) const;
    Circuit with_ties(std::span<const Tie>) const;            // preflight regularisation
    Circuit with_probe(ProbeDecl) const;

    // serialisation
    std::string to_yaml() const;  static Circuit from_yaml(std::string_view);

    // lowering (cached inside the shared data, thread-safe, computed once)
    const Lowered& lowered() const;
};

struct Lowered {                           // what the engines consume
    topology::Graph graph; pwl::DevicePool pool;   // or the DeviceSet of the unified interface
    NameTable names; std::vector<Real> initial_state;
    std::shared_ptr<const SymbolicPlan> symbolic;  // shared by every Circuit with the same structure_hash
};
}
```

- `Circuit` is a value: copying it copies a pointer. All derivation
  functions are structural-sharing copies (the node, scope and device
  arrays are shared; only the changed table is new).
- Lowering evaluates expressions, flattens scopes into names
  (`a/b/R1`), validates, and produces exactly today's `Graph` +
  `DevicePool`, built in the same order as today so that unknown
  indices, switch order and results are bit-identical.
- `SymbolicPlan` (fill-reducing ordering, elimination tree, stamp-slot
  maps) depends on structure only. Circuits that differ in values share
  it. It is populated by `update-linear-time-assembly` when that change
  is present; otherwise the slot is empty and nothing is lost.

- Alternatives considered.
  - *Make `CircuitBuilder` itself copyable and call it the IR.*
    Rejected: it stays mutable (so the preflight keeps mutating
    something), and a copy is a deep copy of graph and pool, which
    defeats structure sharing.
  - *Persistent (HAMT) containers throughout.* Rejected as
    over-engineering: circuits have 10–10⁴ devices, a parameter change
    copies one small table, and the device array is shared whole.
  - *Make the IR the form the engines consume and delete
    `Graph` / `DevicePool`.* Deferred: it would couple this change to
    the device-interface migration. The lowered form is an
    implementation detail behind `lowered()`, replaceable later.

### D2. The builder stays; it builds an IR

```cpp
class CircuitBuilder {
public:
    /* every add_* that exists today, unchanged */
    ParamRef  param(std::string_view name, Value v, std::string_view unit = {});
    ScopeGuard scope(std::string_view instance_name, std::string_view kind = {});   // RAII
    NodeRef   node(std::string_view name, ir::Domain d = ir::Domain::Electrical);
    ir::Circuit build() const;            // snapshot; the builder can keep being edited
    // kept for compatibility: graph(), pool(), components(), initial_state(), run_preflight()
};
```

- Every `add_*` gains overloads accepting `Value` where it accepts a
  number, so `add_inductor("L1", "a", "b", L=p.L)` and
  `L="2*Lbase"` work. Literal calls are unchanged.
- `simulate(builder, …)` calls `builder.build()` and never mutates the
  builder. Code that reads `builder.graph` / `builder.pool` after the
  call sees the circuit as the user wrote it.
- The Python side tables move: `set_initial` / `c0=` / `i0=` write
  `InitialCondition`s; `alias` writes `Alias`; `add_c_block` writes a
  `ControlDecl`; Shockley parameters are ordinary device fields.

### D3. Parameters and expressions

- Grammar: decimal numbers, identifiers (scoped lookup: innermost scope outward),
  `+ − * / ^`, unary minus, parentheses, and the functions `sqrt`,
  `exp`, `ln`, `log10`, `abs`, `min`, `max`, `sin`, `cos`, `tan`,
  `atan`, with the constant `pi`.
- Evaluation: once per lowering, in dependency order. A cycle, an
  unknown identifier or a non-finite result is a `NetlistError` naming
  the device, the field and the expression text.
- Implementation: a recursive-descent parser to a small postfix
  program stored in the IR (`ExprId`). No Python `eval`, no external
  dependency. It is about 200 lines and is fuzzed against a Python
  reference in the tests.
- `with_params({{"Lbase", 2e-3}})` replaces parameter values; fields
  that are expressions over them are re-evaluated at the next lowering.
  Bounds (`lo`, `hi`) on a parameter and the field bounds of the device
  table are both checked.
- The SPICE importer keeps `.param` definitions as IR parameters
  instead of substituting numbers, so `{2*L}` works.

Extension point: `add-parameter-sensitivities` differentiates with
respect to `Parameter`s; the postfix program is evaluated over the AD
scalar.

- Alternative considered: allow arbitrary Python callables as
  parameter values. Rejected: not serialisable, not usable from C++,
  and not differentiable.

### D4. Hierarchy

- A `Scope` is an instance of a subsystem; `kind` records which
  definition it came from, when there is one. Devices, nodes and
  parameters name their scope.
- Port nodes of a subsystem are nodes of the parent scope; internal
  nodes belong to the instance scope. This is what the `Subsystem`
  proxy already does with name prefixes; the proxy now opens a scope
  instead of rewriting strings, and produces the same flattened names.
- The flattened name of an element is `scope_path + "/" + name`. Name
  lookups (`res.v("a/b/out")`, aliases, probes) are unchanged.
- The IR does not store shared definitions ("define once"): each
  instance carries its own devices. `to_yaml()` therefore emits nested
  instances, not templates; `kind` is kept so that a later change can
  factor them.

### D5. Domains

- Each node has a `Domain`. Each device kind's pin table (the `pins[]`
  of the unified device interface, or a table in `ir/` until it exists)
  gives the domain of every pin.
- Connection rule: a pin may connect only to a node of its domain;
  otherwise `NetlistError` naming the device, the pin and the node.
  Kinds that couple domains do so through pins of different domains
  (the MNA PMSM: three electrical pins, mechanical speed and angle
  pins; a lossy device with a thermal pin, later).
- Reference rule: the preflight checks, per domain, that every
  connected component reaches that domain's reference (`gnd`, the
  ambient node, the mechanical frame). Regularising ties are created
  per domain.
- Units: a probe on a node reports the domain's across unit (V, K,
  rad/s or rad for the PMSM's two mechanical nodes; the pin table
  states which); branch quantities report the through unit (A, W, N·m).
- The solver does not use the tag for anything except diagnostics and
  default tolerances per domain (an extension; this change leaves the
  tolerances as they are).
- Default: `Electrical`. A circuit that never mentions a domain is
  today's circuit.
- `Magnetic` is reserved for `add-permeance-magnetic-domain`.

- Alternative considered: separate systems per domain coupled at each
  step. Rejected: it reintroduces the one-step lag that the audit
  measured on the observer machines (13.6 % against 0.48 % at
  dt = 100 µs) and is what "solved in the same MNA" avoids.

### D6. YAML round trip

```yaml
circuit:
  parameters:
    - {name: Lbase, value: 1e-3, unit: H}
    - {name: fsw,   value: 100e3}
  nodes:
    - out                          # electrical (default)
    - {name: tj, domain: thermal}
  devices:
    - {type: inductor, name: L1, from: sw, to: out, L: "2*Lbase"}
  subsystems:
    - name: leg_a
      kind: half_bridge
      parameters: [{name: Ron, value: 10e-3}]
      devices: [ … ]
      subsystems: [ … ]
  probes: ["i(L1)", "v(out)"]
simulation: { … }
```

- New keys are optional; a v2 file without them loads exactly as
  today. The loader's unknown-key check learns the four keys.
- A field whose YAML value is a string is an expression; a number is a
  literal. This is unambiguous because no existing field takes a
  string that is not an enumeration (enumerated fields are declared as
  such in the field table and are not parsed as expressions).
- The emitter writes: parameters and expressions as written (not their
  evaluated values), scopes as nested `subsystems`, node domains only
  when not electrical, every field that differs from its default, and
  device metadata.
- Guarantee: `from_yaml(to_yaml(c))` has the same `value_hash` as `c`,
  and `to_yaml` is a fixed point after one round. Real numbers are
  written with 17 significant digits.
- Pickle and `clone()` are `to_yaml()` / `from_yaml()` plus the shared
  data pointer when in-process (`clone()` is a pointer copy).
- Not serialised: Python callables (gate functions, observers). They
  are attachments of a `Simulation`, not part of a circuit; a circuit
  with only declarative gates and controls round-trips completely.

### D7. Generic value updates

`circuit.with_value("R1.R", 2.0)` resolves the device by scoped name
and the field through the kind's field table. It replaces the four
`update_*` bindings and covers every kind. On an existing run,
`Simulation` exposes `set_value` only for fields the engine can change
without re-analysis (the parametric refactor the electro-thermal path
already uses); otherwise the caller derives a new `Circuit`.

## Performance constraints

- `build()`: one pass over the builder's tables; no numerical work.
- Lowering: per circuit value, cached in the shared data, computed at
  most once even under concurrent access (`std::call_once`).
- `with_params` on the same structure reuses `SymbolicPlan`; the
  per-point cost of a sweep drops from "build + analyse + assemble" to
  "evaluate expressions + assemble values". Measured in the task list
  on a 100-point sweep of the audit's buck and of the n = 1502 ladder
  (the audit's segment-build figures, row 14, are the baseline;
  the expected gain is an estimate until `update-linear-time-assembly`
  provides the shared symbolic analysis).
- Nothing in this change runs in a step loop; the engines consume the
  same lowered form as today.
- `Circuit` is immutable after construction and safe to read from any
  thread; the lowered cache is the only lazily written member and is
  guarded.
- Memory: a derived circuit shares every unchanged table.

## Risks / Trade-offs

- **Lowering must reproduce today's index order exactly**, or every
  golden moves and switch indices change under users' feet. → Lowering
  replays the builder's own insertion order; the oracle of step 1 is
  bit-identity on the whole suite and equality of `components()`.
- **"String means expression" in YAML** could misread a typo as a
  parameter name. → An unknown identifier is an error naming the field,
  not a silent zero.
- **No longer mutating the builder** changes what users who inspect
  `builder` after `simulate()` see (no ties). → The ties are in
  `result.plan.ties` and `result.circuit`; the warning text is updated;
  migration note.
- **Round-trip completeness depends on every kind having YAML keys.**
  → The emitter and loader are generated from one schema table; a test
  enumerates the registry and fails for a kind without a schema entry.
- **Domain checks can refuse circuits that work today** (a thermal
  Cauer chain built from electrical R and C on electrical nodes). →
  Domains are opt-in: untagged nodes are electrical and electrical R /
  C on them are legal. The thermal helpers tag their nodes in a
  separate PR with its own tests.
- **Two representations to keep in step** (IR and lowered). → The
  lowered form is derived only by `lowered()`; nothing else constructs
  a `Graph` / `DevicePool` after the migration.

## Migration Plan

Precondition: `add-kernel-simulation-object` Stage B4
(`Simulation{Circuit, RunOptions}` with `Circuit` as a view). Each row
is shippable alone.

| # | Step | Invariant |
|---|---|---|
| 1 | `ir::Circuit` as an immutable snapshot of the builder (nodes, devices with literal fields, one root scope, initial conditions, aliases); `builder.build()`; `lowered()` reproduces `Graph` + `DevicePool`; `Simulation` takes `ir::Circuit`; `simulate()` calls `build()`. Preflight ties still appended to the builder. | Whole suite and fixed goldens bit-identical; `components()` equal before and after a `build()` + lowering round. |
| 2 | Preflight on the run's own circuit (`with_ties`); the builder is not mutated (**BREAKING**). | A builder is byte-equal (`value_hash`) before and after `simulate()`; re-running with `auto_regularize=False` raises the original singular error; results unchanged. |
| 3 | Python side state into the IR: initial conditions, aliases, C-block declarations, Shockley fields; delete `_pulsim_ic`, `_pulsim_aliases`, `_c_blocks`, `_shockley_params`. | `test_initial_conditions_across_engines.py`, alias and C-block tests unchanged. |
| 4 | `to_yaml()` / `from_yaml()` for every kind (schema table shared by loader and emitter); `clone()`, pickle, `structure_hash`, `value_hash`. | For every shipped YAML example, every harness circuit and one circuit per kind: `from_yaml(to_yaml(c))` simulates bit-identically; `to_yaml` is a fixed point; a registry kind without a schema entry fails a test. |
| 5 | Sweeps accept a `Circuit` and a parameter table; `builder_factory` keeps working. | Sweep results equal the factory form; an 8-job thread sweep scales as audit row 23. |
| 6 | Named parameters with literal values; `with_params`, `with_value`; deprecate the four `update_*` bindings. | `with_value` equals rebuilding with the new literal, bit for bit, for one field of every kind. |
| 7 | Expressions: parser, evaluator, YAML string fields, `circuit.parameters`; the SPICE importer keeps `.param` as parameters. | Expression fuzz test against a Python reference; a circuit written with expressions equals the same circuit written with the evaluated literals, bit for bit; cycle and unknown-name errors name the field. |
| 8 | Scopes: the `Subsystem` proxy opens scopes; YAML `subsystems`; flattened names unchanged. | `test_subsystems.py` unchanged; a 302-branch subsystem circuit has identical `components()` names and results; round trip preserves the scope tree. |
| 9 | Domains: node tag, pin tables, connection and reference checks, units on probes; tag the PMSM's mechanical nodes and the thermal helpers' nodes (one PR each). | Untagged circuits bit-identical; a thermal pin on an electrical node is refused by name; PMSM and electro-thermal tests unchanged. |
| 10 | Structure sharing in lowering: `SymbolicPlan` reused across `with_params` (when `update-linear-time-assembly` is present). | Counter gate: a 100-point sweep performs one symbolic analysis; results equal. |
| 11 | One release later: remove the `update_*` bindings and the string-prefix code of the proxy. | Suite; stub check. |

Rollback: steps 1–3 revert individually; from step 4 on every step is
additive.

## Open Questions

1. YAML expression syntax: a bare string (`L: "2*Lbase"`, proposed), or
   an explicit marker (`L: "=2*Lbase"` or `{expr: …}`) to rule out any
   ambiguity with future string-valued fields?
2. Schema name: keep `pulsim-v2` with optional keys (proposed), or
   introduce `pulsim-v2.1` so that older loaders refuse new files by
   version rather than by unknown key?
3. Should `to_yaml()` emit subsystem *definitions* plus instances
   (smaller files, needs definition identity in the IR) or nested
   instances only (proposed)?
4. SPICE export: the audit lists its absence; the roadmap row does not
   include it. Out of scope here unless the owner wants it.
5. Mechanical domain across variable: the PMSM has both speed and
   angle as nodes. One `Mechanical` domain with per-pin units
   (proposed) or two domains?
6. Should the lowered form move from `Graph` + `DevicePool` to the
   device set of the unified interface inside this change, or stay as
   is until that migration finishes (proposed)?
7. Is the roadmap's dependency list complete? `to_yaml()` for every
   kind is far cheaper with the field tables of
   `add-unified-device-interface` (its step 12); without them this
   change keeps a second schema table.
