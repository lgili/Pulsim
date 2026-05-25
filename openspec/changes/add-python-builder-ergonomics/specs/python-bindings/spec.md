## ADDED Requirements

### Requirement: Initial Conditions via Builder

The `CircuitBuilder` SHALL accept optional initial-condition kwargs on its passive helpers — `c0=None` on `add_capacitor(name, from, to, C, c0=…)` (capacitor voltage in volts) and `i0=None` on `add_inductor(name, from, to, L, i0=…)` (inductor current in amps). When set to a non-`None` value, the builder SHALL store the IC against the device's branch metadata.

Additionally, `CircuitBuilder.set_initial(device_name, value)`
SHALL be supported to attach or override an IC after the device
has been added. The method SHALL raise `KeyError` if
`device_name` doesn't match a registered capacitor or inductor.

When `pulsim.simulate(builder, ..., initial_state=None)` is
invoked with `initial_state=None` AND the builder has recorded
one or more ICs, the simulator SHALL auto-synthesise the
`initial_state` ndarray by populating each IC's slot from the
builder record and zero elsewhere. When the caller passes an
explicit `initial_state=array`, that array SHALL take precedence
and the recorded ICs SHALL be ignored (no merging, no warning).

#### Scenario: Capacitor IC via kwarg
- **GIVEN** a builder with
  `add_capacitor("C1", "v", "gnd", 1e-6, c0=5.0)`
- **WHEN** Python calls
  `pulsim.simulate(builder, t_end=1e-3, dt=1e-6)` (no
  `initial_state`)
- **THEN** the first sample voltage at node `"v"` equals `5.0` V
  within `1e-9` floating-point tolerance.

#### Scenario: Inductor IC via kwarg
- **GIVEN** a builder with
  `add_inductor("L1", "a", "b", 1e-3, i0=0.5)`
- **WHEN** Python calls
  `pulsim.simulate(builder, t_end=1e-3, dt=1e-6)`
- **THEN** the first sample current through `"L1"` equals
  `0.5` A within `1e-9` tolerance and the sign convention
  matches `add_inductor(name, from, to, …)` (current positive
  from `"a"` to `"b"`).

#### Scenario: Override IC post-hoc
- **GIVEN** a builder with `add_inductor("L1", "a", "b", 1e-3)`
  (no IC) followed by `builder.set_initial("L1", 0.5)`
- **WHEN** Python calls `pulsim.simulate(builder, t_end=1e-3,
  dt=1e-6)`
- **THEN** the first sample current through `"L1"` equals
  `0.5` A within `1e-9` tolerance.

#### Scenario: set_initial on unknown device
- **GIVEN** a builder with no device named `"X1"`
- **WHEN** Python calls `builder.set_initial("X1", 0.0)`
- **THEN** the call raises `KeyError` with a message that
  contains `"X1"`.

#### Scenario: Explicit initial_state wins over builder ICs
- **GIVEN** a builder with `c0=5.0` on `"C1"`
- **WHEN** Python calls
  `pulsim.simulate(builder, ..., initial_state=np.zeros(builder.state_size))`
- **THEN** the first sample at node `"v"` equals `0.0` V, NOT
  `5.0`.

### Requirement: Builder Aliases for GUI Round-Tripping

The `CircuitBuilder` SHALL expose `set_alias(human_name, *, node=None, branch=None)` to register a secondary name that resolves to the same canonical electrical entity as the underlying node or branch. Exactly one of `node=` or `branch=` SHALL be non-`None`; passing both, neither, or an empty string SHALL raise `ValueError`.

The `human_name` SHALL NOT collide with an existing canonical
node, branch, or device name; collision SHALL raise `ValueError`.

`CircuitBuilder.aliases()` SHALL return a `dict[str, AliasTarget]`
mapping each registered human name to its canonical target.
`AliasTarget` is a named tuple
`(kind: Literal["node", "branch"], name: str)`.

`node_id_of` / `branch_index_of` / `switch_index_of` (when the
named-lookup proposal lands) SHALL consult `aliases_` before
raising `KeyError`. Alias lookups never override canonical
lookups; canonical wins on collision (but the collision is
rejected at `set_alias` time so this is dead code in practice).

#### Scenario: Alias resolves through node_id_of
- **GIVEN** a builder with
  `add_voltage_source("V1", "node_42", "gnd", 12.0)` and a
  subsequent `builder.set_alias("vin", node="node_42")`
- **WHEN** Python calls `builder.node_id_of("vin")`
- **THEN** the returned value equals
  `builder.node_id_of("node_42")`.

#### Scenario: Alias collision with canonical name is rejected
- **GIVEN** a builder with a node named `"vout"`
- **WHEN** Python calls `builder.set_alias("vout",
  node="node_42")`
- **THEN** `set_alias` raises `ValueError` because `"vout"` is
  already a canonical name.

#### Scenario: Setting both `node=` and `branch=` is rejected
- **WHEN** Python calls `builder.set_alias("x", node="a",
  branch="L1")`
- **THEN** `set_alias` raises `ValueError`.

#### Scenario: Setting neither `node=` nor `branch=` is rejected
- **WHEN** Python calls `builder.set_alias("x")` (no kwargs)
- **THEN** `set_alias` raises `ValueError`.

#### Scenario: aliases() round-trips through GUI metadata
- **GIVEN** a builder with
  ```
  builder.set_alias("vin",  node="node_42")
  builder.set_alias("vout", node="node_43")
  ```
- **WHEN** Python calls `builder.aliases()`
- **THEN** the returned dict equals
  ```
  {"vin":  AliasTarget(kind="node", name="node_42"),
   "vout": AliasTarget(kind="node", name="node_43")}
  ```.

### Requirement: Cancellation on All Top-Level Analyses

Every long-running top-level analysis SHALL accept a `should_continue: Callable[[], bool] | None = None` kwarg — concretely `pulsim.simulate`, `pulsim.compute_dc_op`, `pulsim.run_ac_sweep`, `pulsim.run_mna_sweep`, and `pulsim.compute_temperature`. When the callback is non-`None`, each function SHALL invoke it at well-defined checkpoints:

- `simulate`: between accepted steps (existing behavior;
  unchanged for backward compatibility).
- `compute_dc_op`: between Newton iterations of the main solve.
- `run_ac_sweep` / `run_mna_sweep`: after each frequency point /
  sweep parameter completes.
- `compute_temperature`: between Foster convolution chunks,
  every 1000 samples or every 1 % of the trace (whichever is
  more frequent).

When the callback returns `False`, the function SHALL terminate
promptly (within one checkpoint interval) with a
`pulsim.Cancelled` exception (subclass of `RuntimeError`). The
exception SHALL carry context attributes identifying where the
cancellation occurred:
- `iteration: int` for `compute_dc_op`
- `point_index: int` for `run_ac_sweep` / `run_mna_sweep`
- `chunk_index: int` for `compute_temperature`
- `t: float` for `simulate`

When the callback is `None` (default), behavior SHALL be
bit-identical to the pre-change implementation (no perf
overhead from the cancellation infrastructure).

#### Scenario: Cancel DC OP mid-Newton
- **GIVEN** a builder for which `compute_dc_op` takes 6 Newton
  iterations to converge under default settings
- **WHEN** Python passes a `should_continue` that returns
  `True` for the first 3 calls and `False` thereafter
- **THEN** the call raises `pulsim.Cancelled`
- **AND** the exception's `iteration` attribute equals `3`.

#### Scenario: Cancel AC sweep between points
- **GIVEN** an `run_ac_sweep` configured for 50 frequency points
- **WHEN** Python passes a `should_continue` that returns
  `False` after the 10th point completes
- **THEN** the call raises `pulsim.Cancelled` with
  `point_index == 10`
- **AND** any partial results are NOT silently returned (the
  exception is the only failure mode; no half-populated
  ndarray escapes).

#### Scenario: Cancel thermal convolution
- **GIVEN** a `compute_temperature` over a 1 s, 1 µs-step power
  trace (10⁶ samples)
- **WHEN** Python passes a `should_continue` that returns
  `False` after the 5000-th sample
- **THEN** the call raises `pulsim.Cancelled` with
  `chunk_index ∈ [5, 6]` (the 5th or 6th 1000-sample chunk).

#### Scenario: should_continue=None preserves prior behavior
- **GIVEN** an existing script that calls
  `pulsim.compute_dc_op(builder, ...)` without `should_continue`
- **WHEN** the script runs on the post-change build
- **THEN** behavior is identical to the pre-change
  implementation (same iterations, same numeric result, no
  Cancelled exception)
- **AND** the wall-clock time is within ±2 % of the pre-change
  baseline on the same fixture (the cancellation check
  introduces no measurable overhead when disabled).
