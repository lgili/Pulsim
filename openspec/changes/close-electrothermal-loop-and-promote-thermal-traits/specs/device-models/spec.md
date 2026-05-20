## MODIFIED Requirements

### Requirement: Device Trait `has_thermal_model` Reflects Actual Wiring

`device_traits<T>::has_thermal_model` SHALL be `true` for any device
class that exposes ALL of the following:

1. A `Params::R_th_ja` field (or equivalent thermal-resistance
   parameter, e.g. `R_th_winding_to_ambient`).
2. A `Params::T_amb` field.
3. A `Real T_j_` (or equivalent) private member.
4. A `void set_T_j_init(Real)` public method.
5. A `Real junction_temperature() const noexcept` public method.

Today (pre-change) the trait is `true` only for `MOSFET` and `IGBT`,
even though `IdealDiode`, `Resistor`, `Inductor`, and `Capacitor` all
satisfy criteria (1)-(5). The trait MUST be promoted to `true` on
those four device classes so that
`DefaultThermalService::reset` (`transient_services.cpp:1158-1162`)
enrolls them in the closed electrothermal loop.

#### Scenario: IdealDiode now appears in the thermal walker

- **GIVEN** a circuit containing a single `IdealDiode` with
  `R_th_ja > 0`
- **WHEN** the simulator constructs `DefaultThermalService` and calls
  `reset()`
- **THEN** the thermal-state vector SHALL contain one entry for the
  diode
- **AND** `result.thermal_summary.device_temperatures` SHALL contain
  a `"D1"` (or whatever the diode name is) entry after the simulation
  completes
- **AND** the diode's `diode_junction_temperature("D1")` SHALL match
  that entry within 1e-9 °C.

#### Scenario: Resistor, Inductor, Capacitor with R_th_ja > 0 walk temperature

- **GIVEN** a buck converter with PWM drive on a MOSFET that has
  `R_th_ja = 0` (no FET thermal), and where the output capacitor has
  `R_th_ja = 5 K/W` and `T_amb = 25 °C`
- **AND** the capacitor's ripple current is ≥ 1 A_rms
- **WHEN** the simulator runs for ≥ 5·R_th_ja·C_th
- **THEN** `circuit.capacitor_junction_temperature("Cout")` SHALL be
  strictly greater than 25 °C
- **AND** the capacitor's ESR(T_j) used in `accumulate_loss` SHALL
  reflect the integrated T_j (i.e. `ESR_at_Tj()` returns
  `ESR · (1 + ESR_tc · (T_j − T_ref))` with the walking T_j, not the
  static T_amb).

## ADDED Requirements

### Requirement: Per-Device Thermal Accessors on Circuit

The `Circuit` class SHALL expose, for every device class with
`device_traits<T>::has_thermal_model == true`, three accessor methods:

1. `Real <dev>_junction_temperature(std::string_view name) const`
2. `Real <dev>_steady_state_junction_temperature(std::string_view name) const`
3. `void <dev>_T_j_init(std::string_view name, Real T_j)`

Where `<dev>` is the device-name token used elsewhere on Circuit
(e.g. `mosfet_*`, `igbt_*`, `diode_*`, `resistor_*`, `inductor_*`,
`capacitor_*`). The three accessor methods MUST return / accept
`Real` and SHALL return `std::numeric_limits<Real>::quiet_NaN()`
when the device with the requested name does not exist.

#### Scenario: Diode T_j_init from YAML round-trips through C++

- **GIVEN** a YAML circuit with a `diode` device that has a
  `thermal_devices` block specifying `T_j_init: 40.0`
- **WHEN** the YAML parser builds the Circuit
- **THEN** `circuit.diode_junction_temperature("D1")` SHALL return
  `40.0` (within 1e-9 °C) BEFORE the simulation starts
- **AND** the simulator SHALL begin integrating T_j(t) from that
  initial value, not from T_amb.

#### Scenario: Missing device name returns NaN, does not throw

- **GIVEN** a Circuit with no resistor named "R_does_not_exist"
- **WHEN** the user calls
  `circuit.resistor_junction_temperature("R_does_not_exist")`
- **THEN** the call SHALL return `std::numeric_limits<Real>::quiet_NaN()`
- **AND** SHALL NOT throw an exception
- **AND** SHALL NOT log a diagnostic (this is a silent NaN return,
  mirroring the existing `mosfet_*` accessor convention).

## ADDED Requirements

### Requirement: YAML Thermal Block Accepts Passive + Diode Devices

The YAML parser SHALL accept a `thermal_devices` block entry for any
device type whose `device_traits::has_thermal_model == true`. Today
(pre-change) the parser allow-list is restricted to `mosfet`, `igbt`,
and the `bjt_*` family (`yaml_parser.cpp:88-93`). The allow-list MUST
be extended to include `diode`, `resistor`, `inductor`, `capacitor`.

The accepted keys per entry are:
- `R_th_ja` (K/W) — required when the entry is present
- `T_amb` (°C) — defaults to the device's construction T_amb
- `T_j_init` (°C) — defaults to T_amb
- `cth` (J/K) — thermal capacitance, defaults to a sensible value
  per device class (matches the C++ default in
  `ThermalDeviceConfig::cth`)
- `temp_ref` (°C) — reference temperature for `alpha`
- `alpha` (1/K) — temperature coefficient of the temperature-scaled
  parameter

Unknown keys inside the entry MUST trigger a `ParseError` with a
message naming the offending key.

#### Scenario: YAML resistor thermal block parses cleanly

- **GIVEN** a YAML file with:
  ```yaml
  components:
    - type: resistor
      name: R_load
      nodes: [out, 0]
      value: 10
  simulation:
    thermal:
      enable: true
      ambient: 25
    thermal_devices:
      R_load:
        R_th_ja: 50
        T_amb: 25
        T_j_init: 30
  ```
- **WHEN** the YAML parser loads the file
- **THEN** the resulting Circuit SHALL have one `Resistor` with
  `params.R_th_ja == 50`, `params.T_amb == 25`, and
  `T_j_init == 30` (verifiable via
  `circuit.resistor_junction_temperature("R_load")`)
- **AND** the resulting `SimulationOptions::thermal_devices` SHALL
  contain a `"R_load"` entry with the same values
- **AND** parsing SHALL succeed without errors or warnings.

#### Scenario: YAML thermal block on a non-thermal device type errors out

- **GIVEN** a YAML file with `thermal_devices` entry for a `vsource`
  (which has `has_thermal_model = false`)
- **WHEN** the parser encounters the entry
- **THEN** the parser SHALL emit a `ParseError` naming the device
  type and explaining that thermal config is not supported for it
- **AND** the parser SHALL NOT silently ignore the block (silent
  ignore would be confusing for the user).
