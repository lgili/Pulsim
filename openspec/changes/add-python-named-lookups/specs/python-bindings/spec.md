## ADDED Requirements

### Requirement: Named Result Accessors

`SimulationResult` SHALL expose `v(name, t=None)`,
`i(name, t=None)`, and `power(device_name)` methods on the Python
wrapper that resolve a human node, branch, or device name into the
corresponding state-vector trace, without requiring the caller to
compute `node_id_of` / `branch_index_of` offsets or apply the
`num_nodes` shift for branch currents.

The `name` argument SHALL accept either a node name passed to
`CircuitBuilder.add_*` (resolved via `node_id_of` for `v`, or
`branch_index_of` for `i`), or a registered alias when alias
support lands separately. When `t=None` the method SHALL return
the full per-sample trace as a contiguous `numpy.ndarray`. When
`t` is an `int`, the method SHALL return a scalar `float64`. When
`t` is a slice or boolean mask, the method SHALL return the
corresponding subset with shape matching the slice semantics.

#### Scenario: Read node voltage by name
- **GIVEN** a transient run of a buck converter with a node named
  `"vout"`
- **WHEN** Python calls `result.v("vout")`
- **THEN** the returned ndarray equals
  `np.asarray(result.states)[:, builder.node_id_of("vout")]`
  element-wise
- **AND** has shape `(num_steps,)` and dtype `float64`.

#### Scenario: Read branch current by name
- **GIVEN** the same buck transient with an inductor named `"L1"`
- **WHEN** Python calls `result.i("L1")`
- **THEN** the returned ndarray equals
  `np.asarray(result.states)[:, builder.graph.num_nodes +
  builder.branch_index_of("L1")]` element-wise
- **AND** the sign convention follows the
  `add_inductor(name, from, to, …)` ordering: current is positive
  when flowing from the `from` terminal toward the `to` terminal.

#### Scenario: Single-step snapshot
- **GIVEN** a transient with 10 000 samples
- **WHEN** Python calls `result.v("vout", t=-1)`
- **THEN** the returned value is a scalar `float64` (not a 0-d
  ndarray)
- **AND** equals `result.states[-1, builder.node_id_of("vout")]`.

#### Scenario: Device power lookup
- **GIVEN** a transient that exercised a MOSFET `"Q1"`
- **WHEN** Python calls `result.power("Q1")`
- **THEN** the returned float equals
  `device_loss_summary(result, builder)["Q1"]["P_total"]`
  within floating-point precision.

#### Scenario: Unknown name produces typed error with suggestions
- **GIVEN** a builder with nodes `["vin", "sw", "vout", "gnd"]`
- **WHEN** Python calls `result.v("voutt")` (typo)
- **THEN** the call raises `pulsim.NameNotFoundError` (a subclass
  of `KeyError`)
- **AND** the exception's `suggestions` attribute equals
  `["vout"]`
- **AND** the exception's `kind` attribute equals `"node"`
- **AND** the exception's `name` attribute equals `"voutt"`.

### Requirement: Builder Device-Name Index Lookups

`CircuitBuilder` SHALL expose `branch_index_of(name)`,
`switch_index_of(name)`, and `devices()` methods that let callers
look up state-vector indices and enumerate registered devices
without re-counting `add_*` call order.

`branch_index_of(name)` SHALL return the zero-based offset of the
named branch's current within the post-node section of the state
vector (i.e., the value `i` such that `state_idx == num_nodes + i`
holds for the branch's current variable).

`switch_index_of(name)` SHALL return the zero-based bit position
of the named switching device in the `SwitchStateMask` produced
by `switch_fn(t)`. It SHALL raise `KeyError` if `name` refers to
a non-switching device (resistor, capacitor, source, …), with a
message that names the device's actual kind.

`devices()` SHALL return an ordered list of
`DeviceInfo(name, kind, terminals)` tuples covering every device
the builder has accepted, in `add_*` call order. `kind` is a
lowercase string identifying the device family
(`"resistor" / "capacitor" / "inductor" / "diode" / "mosfet" /
"igbt" / "switch" / "voltage_source" / "current_source" /
"transformer" / …`). `terminals` is the ordered list of node
names the device is wired to.

#### Scenario: PWM lookup by device name
- **GIVEN** a builder with `add_mosfet_with_body_diode("Q1", "vin", "sw", …)`
  as the only switching branch
- **WHEN** Python calls `builder.switch_index_of("Q1")`
- **THEN** the returned value equals `0`.

#### Scenario: Multiple switches enumerate in call order
- **GIVEN** a builder with calls
  ```
  add_mosfet_with_body_diode("Q1", "vin", "sw", …)
  add_mosfet_with_body_diode("Q2", "sw", "gnd", …)
  ```
- **WHEN** Python calls `builder.switch_index_of("Q1")` and
  `builder.switch_index_of("Q2")`
- **THEN** the values equal `0` and `1` respectively
- **AND** `builder.graph.num_switches == 2`.

#### Scenario: Non-switching device raises with kind info
- **GIVEN** a builder with a resistor `"R1"` and a MOSFET `"Q1"`
- **WHEN** Python calls `builder.switch_index_of("R1")`
- **THEN** the call raises `KeyError`
- **AND** the message contains both `"R1"` and the device's kind
  string `"resistor"`.

#### Scenario: Enumerate registered devices
- **GIVEN** a builder with calls (in this order)
  ```
  add_voltage_source("V1", "vin", "gnd", 12.0)
  add_mosfet_with_body_diode("Q1", "vin", "sw", …)
  add_inductor("L1", "sw", "vout", 220e-6)
  ```
- **WHEN** Python calls `builder.devices()`
- **THEN** the returned list equals
  ```
  [DeviceInfo("V1", "voltage_source", ["vin", "gnd"]),
   DeviceInfo("Q1", "mosfet", ["vin", "sw"]),
   DeviceInfo("L1", "inductor", ["sw", "vout"])]
  ```
- **AND** the list order matches the `add_*` call sequence.

#### Scenario: Branch-index round-trip with state vector
- **GIVEN** an inductor `"L1"` with
  `builder.branch_index_of("L1") == 3` and
  `builder.graph.num_nodes == 5`
- **WHEN** Python computes
  `np.asarray(result.states)[:, 5 + 3]`
- **THEN** the resulting ndarray is element-wise equal to
  `result.i("L1")`.
