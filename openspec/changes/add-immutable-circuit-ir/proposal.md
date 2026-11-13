## Why

A circuit exists only as a mutable `CircuitBuilder` (a `Graph`, a
`DevicePool` and name tables) plus private attributes that Python hangs
on it. It can be built and simulated; it cannot be copied, saved,
compared, parameterised or handed to another thread. Counted on this
branch (PY = `python/pulsim/__init__.py`, BN = `python/bindings.cpp`,
`core/…` = `core/include/pulsim/…`):

- **No export and no copy.** There is no `to_yaml`, `to_spice`, `clone`
  or `copy` on the builder
  (`grep -rnE 'def (to_yaml|dump_yaml|to_spice|clone|copy)\b|"to_yaml"|"clone"' python/pulsim python/bindings.cpp core/include/pulsim/{builder,yaml}`
  finds nothing), and the builder is not picklable (the only
  `py::pickle` in BN is `SolverSnapshot`, BN:2545).
- **Sweeps cannot take a circuit.** `sweep` and `monte_carlo` take a
  `builder_factory(**params)` and rebuild the circuit for every point
  (`python/pulsim/sweep.py:112-124, 240`).
- **Parameters can be changed after construction for four kinds only:**
  `update_resistor_R`, `update_inductor_L`, `update_capacitor_C`,
  `update_voltage_source_V` (BN:1000-1040).
- **`simulate()` mutates the caller's circuit.** The preflight appends
  reference ties to the builder (PY:1904-1932); the warning text itself
  says the ties persist and that re-running the same builder will not
  restore the original error.
- **Circuit state that is not in the circuit.** Python sets
  `builder._c_blocks`, `builder._pulsim_ic`, `builder._pulsim_aliases`
  and `self._shockley_params` (one assignment each in
  `python/pulsim/`), read back with `getattr` at seven sites. None of
  it survives a C++ round trip.
- **Hierarchy is a naming convention.** `subsystem.py` rewrites device
  and node names with a `/`-separated prefix as they are added
  (`python/pulsim/subsystem.py:118-126`); the kernel sees a flat
  netlist and no scope.
- **No named parameters, no expressions.** The YAML v2 loader accepts
  only `circuit: {nodes, devices}` and `simulation:`
  (`core/yaml/loader.hpp:687, 707`) with literal values. The SPICE
  importer resolves `{name}` references at import time and treats any
  other brace content as an undefined parameter
  (`python/pulsim/spice_import.py:120-128`; read, not run).
- **Nodes have no domain.** Thermal networks (Cauer chains) and
  mechanical variables (the PMSM's speed and angle) are ordinary
  electrical nodes; units, validation and reference checks cannot tell
  them apart.
- **Five device kinds have no YAML form** (`shockley_diode`,
  `lauritzen_diode`, `nonlinear_capacitor`, `pmsm_mna`, the flux-table
  inductor; audit `raw/models.md` §a), so even a hand-written export
  could not round-trip every circuit.

Consequences seen or measured:

| Symptom | Cause |
|---|---|
| Re-running a builder after an auto-regularised run hides the singular-matrix error for good | the circuit is mutated by the run |
| An 8-job sweep through `multiprocessing` takes 18.9 s against 695 ms sequential; threads scale (3.0–3.1× on four) only when nothing Python is shared (audit `raw/perf.md` rows 23–25) | no shareable, immutable circuit |
| A snapshot cannot be tied to the circuit it belongs to except by vector sizes | no structural identity |
| C blocks on `dsed` run open loop with a warning (PY:1996-2003; refused once `add-engine-conformance-matrix` lands) | control declarations live in a Python attribute the kernel cannot see |
| PulsimGUI builds circuits through its own converter (`CircuitConverter.build`, named in the owner note *gui-swallows-backend-refusals*; not inspected here) | no IR to hand over |

## What Changes

- **An immutable `Circuit`.** `CircuitBuilder` stays the authoring
  API; `builder.build()` returns a `Circuit`: a cheap-to-copy handle to
  shared, immutable data holding nodes (with a domain), devices (kind,
  pins, fields), scopes, named parameters, initial conditions, aliases,
  probe declarations and declarative control / gate declarations.
- **Named parameters and expressions.** A device field is a literal or
  an expression over named parameters (`+ − * / ^`, parentheses, a
  fixed set of functions). Expressions are evaluated when the circuit
  is lowered; a cycle or an unknown name is a `NetlistError`.
- **`with_params(...)`** returns a new `Circuit` that shares structure
  with the original; the lowered sparsity pattern, and therefore the
  symbolic analysis and the stamp-slot maps, are reused.
  `with_value("R1.R", 2.0)` works for every field of every kind.
- **Hierarchy kept.** Scopes form a tree; each device and node belongs
  to a scope and has a scoped name. Flattened names are today's
  (`a/b/R1`), so accessors and results do not change.
- **`to_yaml()` / `from_yaml()`, `clone()`, pickle, `structure_hash()`
  and `value_hash()`.** Every kind round-trips.
- **Domain-tagged nodes** (`electrical`, `thermal`, `mechanical`,
  `magnetic` reserved): pins declare their domain, a cross-domain
  connection is refused by name, probes carry the domain's unit, and
  the preflight checks one reference per domain. All domains are solved
  in the same MNA system.
- **Python side state moves into the IR:** initial conditions, aliases,
  C-block declarations, the Shockley parameter table.
- **BREAKING:** `simulate(builder, …)` no longer mutates the builder:
  regularising ties are applied to the run's own `Circuit` and reported
  in the result. A builder re-run with `auto_regularize=False` raises
  the original error again.
- **BREAKING:** `update_resistor_R`, `update_inductor_L`,
  `update_capacitor_C`, `update_voltage_source_V` are deprecated for
  one release in favour of `with_value` / `with_params`.
- **BREAKING (YAML, additive keys):** `circuit.parameters`,
  `circuit.subsystems`, node entries with a `domain`, string-valued
  fields as expressions. Existing v2 files load unchanged.
- **Deleted at the end:** `builder._c_blocks`, `_pulsim_ic`,
  `_pulsim_aliases`, `_shockley_params`; the name-rewriting proxy's
  string prefixing (replaced by scopes); the `update_*` bindings.

## Impact

- Affected specs: `circuit-ir` (new), `netlist-yaml`.
- Affected code: new `core/include/pulsim/ir/`;
  `builder/circuit_builder.hpp`; `topology/graph.hpp` (node domain);
  `pwl/{device_pool,preflight}.hpp`; `yaml/loader.hpp` and a new
  `yaml/emitter.hpp`; `sim/simulation.hpp` (takes `Circuit`);
  `python/bindings.cpp`;
  `python/pulsim/{__init__,_builder_ergonomics,subsystem,sweep,c_block,spice_import,thermal,motor_helpers,steady_state}.py`;
  stubs; `AI_GUIDE.md`; docs.
- Depends on: `add-kernel-simulation-object` (`Simulation{Circuit,
  RunOptions}`; the plan owns the regularised copy).
- Uses, when present: the device field tables of
  `add-unified-device-interface` (they generate the YAML keys of every
  kind; without them the emitter and the loader share a schema table
  kept in `yaml/`), the probe declarations of
  `add-probes-and-recorders`, the shared symbolic analysis of
  `update-linear-time-assembly`, the thread-pool sweeps of
  `update-sweep-threading-and-import-time`.
- Unblocks: `add-parameter-sensitivities` (named parameters are the
  differentiation variables), `add-permeance-magnetic-domain` (the
  `magnetic` domain), `add-compiled-control-graph` (control
  declarations in the IR), `add-in-kernel-event-losses` (the thermal
  domain).
- Performance constraints: `build()` and lowering are per circuit, not
  per step; `with_params` on an unchanged structure must not repeat
  symbolic analysis or slot-map construction (measured on a 100-point
  sweep, before and after); the lowered form consumed by the engines is
  today's `Graph` + pool (or the device set of the unified interface),
  so the step loop is untouched. `Circuit` is safe to share across
  threads.
- Results: every existing circuit simulates bit-identically; the one
  behavioural change is the first **BREAKING** item.
