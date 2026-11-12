## ADDED Requirements

### Requirement: Kernel Loss Tables

The kernel SHALL provide a switching-energy table `E(v, i, T_j)` and an
on-state table `V_on(i, T_j)` on rectilinear, strictly ascending axes.
Evaluation SHALL be multilinear inside the grid, SHALL extrapolate
linearly beyond it, SHALL treat a single-point axis as constant, and
SHALL clamp the result at zero. Results SHALL equal
`pulsim.loss_tables.LossTable` for the same data. Construction SHALL
refuse an empty or non-ascending axis, a shape mismatch, and a negative
or non-finite entry, naming the table.

#### Scenario: Same numbers as the Python table
- **GIVEN** one table held by the Python class and by the kernel class
- **WHEN** both are evaluated at 10 000 random points inside and outside
  the grid
- **THEN** every pair of results agrees within 1e-12 relative

#### Scenario: Temperature axis interpolates and extrapolates
- **GIVEN** `E = 1 mJ` at 25 °C and `2 mJ` at 125 °C for every `(v, i)`
- **WHEN** the table is read at 75 °C and at 175 °C
- **THEN** it returns 1.5 mJ and 2.5 mJ

### Requirement: Device Loss Port

A PWL switching device SHALL accept a loss specification (switch,
MOSFET switch, IGBT, PWL diode, and body or freewheel diode) with
optional `E_on`, `E_off` and `V_on` tables and exactly one temperature
source: `junction`, a thermal node of the circuit, or `Tj`, a fixed
temperature in °C. The specification SHALL be available through the
builder, Python and a YAML `losses:` block. A specification that names
an unknown device or an unsupported device kind, gives both or neither
temperature source, or selects table conduction without a `V_on` table
SHALL be refused with an error naming the device. A circuit with no loss
specification SHALL be bit-identical to the same circuit before this
capability existed.

#### Scenario: Both temperature sources refused
- **WHEN** `set_device_losses("Q1", junction="Tj", Tj=125.0, ...)` is
  called
- **THEN** it raises an error naming `Q1`, `junction` and `Tj`

#### Scenario: YAML block
- **WHEN** a YAML `igbt` carries `losses: {Tj: 125, e_on: {...}}`
- **THEN** the loaded circuit holds that table for the device
- **AND** an unknown key inside `losses:` is refused naming the key

### Requirement: Event Energy Reading Rule

At each commutation of a device with a loss port the kernel SHALL
evaluate the matching table once. For a turn-on it SHALL use the
magnitude of the device voltage in the last accepted state before the
commutation and the magnitude of the device current in the first
accepted state after it. For a turn-off it SHALL use the voltage after
and the current before. The temperature SHALL be the junction node's
value in the last accepted state before the commutation, or the fixed
`Tj`. The rule SHALL apply equally to gate-commanded commutations and to
commutations the kernel decides. Each event SHALL be recorded as
`(t, kind, v, i, T_j, E)`.

#### Scenario: Agreement with the post-hoc summary at fixed temperature
- **GIVEN** a three-phase inverter whose six switches carry `E_on` and
  `E_off` tables with `Tj = 125`
- **WHEN** it runs on `pwl` and `device_loss_summary` is computed from
  the same result with the same tables and `Tj`
- **THEN** each device's kernel `E_on` and `E_off` totals equal the
  summary's within 1e-9 relative

#### Scenario: Diode recovery at natural turn-off
- **GIVEN** a PWL diode with a recovery table on its `E_off`
- **WHEN** the diode turns off N times in a run
- **THEN** exactly N events are recorded for it

### Requirement: Energy-Conserving Thermal Injection

The energy of each event SHALL be deposited as heat into the junction
node exactly once, no later than the accepted step that follows the
commutation, and the total heat injected over a run SHALL equal the sum
of the recorded event energies within 1e-12 relative. A junction node
whose thermal network offers a purely resistive path to a temperature
source SHALL be refused with an error naming the device and the node.

#### Scenario: Adiabatic junction
- **GIVEN** a junction node with `C_th = 0.01 J/K` and `R_th = 1e9 K/W`
  to ambient, and constant tables `E_on = E_off = 1 mJ`
- **WHEN** the device commutates ten times
- **THEN** the junction temperature has risen by 1.000 K within 1e-6
  relative

#### Scenario: Resistive-only junction refused
- **WHEN** a loss port names a junction node connected to the ambient
  source through a resistor only
- **THEN** building fails with an error naming the device and the node

### Requirement: Closed-Loop Junction Temperature

When the temperature source is a junction node, every table read SHALL
use the temperature that the run has produced so far, on `pwl`, `trbdf2`
and `dsed`. An engine that cannot realise the injection SHALL refuse the
run with the conformance error naming the device and the engine; it
SHALL NOT run with the temperature held constant.

#### Scenario: Open loop reference
- **GIVEN** one Cauer stage `R_th = 1 K/W`, `C_th = 0.01 J/K`, ambient
  25 °C, a switch commutating at 10 kHz with constant tables
  `E_on = E_off = 1 mJ`
- **WHEN** the run reaches steady state
- **THEN** the time-weighted mean junction temperature over one period
  is 45.0 °C within 0.05 K

#### Scenario: Loop closed
- **GIVEN** the same circuit with `E(T) = 1 mJ·(1 + 0.01·(T − 25))` for
  both tables and `junction=` the Cauer node
- **WHEN** the run reaches steady state
- **THEN** the mean junction temperature is 50.0 °C within 0.1 K
- **AND** the same circuit with `Tj = 25` gives 45.0 °C

#### Scenario: Engines agree
- **WHEN** the closed-loop circuit runs on `pwl`, `trbdf2` and `dsed`
- **THEN** the mean junction temperatures differ by less than 0.2 K, or
  an engine refuses by name

### Requirement: Conduction Heat Source

The conduction heat of a device with a loss port SHALL be selectable.
With `conduction="electrical"` it SHALL equal the device's electrical
dissipation over each step. With `conduction="table"` it SHALL equal
`V_on(|i|, T_j)·|i|` read from the on-state table without collapsing it
to an offset and a slope, and the result SHALL report the electrical and
the table conduction energies separately.

#### Scenario: Electrical dissipation heats the junction
- **GIVEN** 10 A through a PWL diode with `V_f = 0.7 V`, `g_on = 20 S`,
  and `R_th = 1 K/W` to a 25 °C ambient
- **WHEN** the run settles
- **THEN** the injected power is 12.0 W and `T_j = 37.0 °C` within
  0.01 K

#### Scenario: Temperature-dependent on-state table
- **GIVEN** `V_on = 1.0 + 0.002·(T_j − 25)` V at 10 A, `R_th = 1 K/W`,
  `conduction="table"`, closed loop
- **WHEN** the run settles
- **THEN** `T_j = 35.20 °C` within 0.01 K

### Requirement: Part Loss Attachment

A part that carries loss tables and a thermal chain SHALL attach to a
device in one call that builds the Cauer network and sets the tables.
A part whose provenance is `synthetic` SHALL be refused unless the
caller passes `allow_synthetic=True`.

#### Scenario: Imported thermal description
- **GIVEN** a part returned by `import_plecs_xml`
- **WHEN** `part.attach_losses(builder, "Q1", T_amb=40.0)` is called
- **THEN** `Q1` has `E_on`, `E_off` and `V_on` tables and a junction
  node whose steady-state thermal resistance to ambient equals the sum
  of the part's chain resistances

#### Scenario: Synthetic part refused
- **WHEN** `attach_losses` is called on a shipped synthetic part without
  `allow_synthetic=True`
- **THEN** it raises `ProvenanceError`
