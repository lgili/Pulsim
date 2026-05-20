# device-models Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: Diode Stamp with Limiting

The diode stamp function SHALL apply voltage limiting before computing current and conductance.

The stamp SHALL:
- Retrieve previous diode voltage from device state
- Apply voltage limiting to new voltage
- Compute I and G using limited voltage
- Store new voltage in device state

#### Scenario: Diode stamp with limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_diode() is called with V_new from Newton
- **THEN** V_limited = limit_diode_voltage(V_new, V_old)
- **AND** I and G are computed using V_limited
- **AND** V_old is updated to V_limited

### Requirement: MOSFET Stamp with Limiting

The MOSFET stamp function SHALL apply voltage limiting before computing currents.

#### Scenario: MOSFET stamp with Vgs and Vds limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_mosfet() is called
- **THEN** Vgs_limited = limit_mosfet_vgs(Vgs_new, Vgs_old)
- **AND** Vds_limited = limit_mosfet_vds(Vds_new, Vds_old)
- **AND** drain current is computed using limited voltages

### Requirement: Modular Component Model Library
The system SHALL define each built-in electrical component model in a dedicated component file under a stable component library path, while preserving a compatibility aggregator include for legacy callers.

#### Scenario: Legacy include compatibility after modularization
- **GIVEN** existing code that includes `pulsim/v1/device_base.hpp`
- **WHEN** the project is built after model modularization
- **THEN** all existing built-in component types remain available
- **AND** no caller migration is required for include-path compatibility

#### Scenario: Isolated model evolution per component
- **GIVEN** a change to one component model file
- **WHEN** tests and benchmarks are executed
- **THEN** only that component module and dependent integration paths are impacted
- **AND** unrelated models do not require structural edits in the same file

### Requirement: Controlled Numerical Regularization for Switching Models
The system SHALL support controlled, bounded numerical regularization for switching/nonlinear component models to improve convergence in pathological switching regimes without unbounded physical distortion.

#### Scenario: Automatic regularization in repeated switching-step failure
- **GIVEN** repeated transient failures near switching discontinuities
- **WHEN** recovery policy escalates through configured stages
- **THEN** bounded regularization parameters are applied to eligible component models
- **AND** each escalation is recorded in structured telemetry

#### Scenario: Regularization bounded by policy limits
- **GIVEN** auto-regularization is active
- **WHEN** the solver escalates regularization intensity
- **THEN** configured maximum bounds are never exceeded
- **AND** simulation aborts with typed diagnostics if convergence still fails

### Requirement: Catalog Device Tier
The device library SHALL provide a `Catalog` tier with datasheet-calibrated parameters covering nonlinear capacitances, temperature dependence, and reverse-recovery behavior, distinct from the simple Level-1 tier.

#### Scenario: MosfetCatalog stamps with Vds-dependent Coss
- **GIVEN** a `MosfetCatalog` device with `Coss(Vds)` lookup table
- **WHEN** the device stamps at Vds = 200 V and the table interpolates to 60 pF
- **THEN** the stamped capacitance equals 60 pF within interpolation tolerance
- **AND** the derivative `dCoss/dVds` is included in the AD Jacobian if `Behavioral` mode

#### Scenario: Temperature-dependent Rds_on
- **GIVEN** a `MosfetCatalog` device with `Rds_on_25C = 19 mΩ` and `Rds_on_temp_coef` curve
- **WHEN** Tj reaches 100 °C and the table interpolates to `Rds_on = 28 mΩ`
- **THEN** subsequent stamps use the updated Rds_on
- **AND** loss accumulation uses the temperature-corrected value

#### Scenario: Body-diode reverse recovery
- **GIVEN** a `MosfetCatalog` with embedded body-diode `Qrr = 100 nC` at `di/dt = 500 A/µs`
- **WHEN** the diode commutates from on to off with the matching di/dt
- **THEN** the simulated reverse-recovery charge integrates within 15% of `Qrr`
- **AND** the recovery loss is added to the device's switching-loss accumulator

### Requirement: IGBT Tail-Current Modeling
The `IgbtCatalog` device SHALL model post-turnoff tail current as `I_tail(t) = I0 · exp(-t/τ_tail)` with parameters from datasheet.

#### Scenario: Tail current after turn-off
- **GIVEN** an `IgbtCatalog` with `I_tail = 5 A`, `τ_tail = 200 ns`
- **WHEN** the device turns off at time t0 with collector current 50 A
- **THEN** the device current at `t0 + 200 ns` is approximately `5 A · e^{-1} = 1.84 A`
- **AND** the tail-current contribution is accumulated in `E_off`

### Requirement: Diode Reverse-Recovery Shape
The `DiodeCatalog` device SHALL produce a reverse-recovery transient whose total charge equals `Qrr(If, di/dt)` from datasheet within 15% on hard commutation.

#### Scenario: Reverse recovery on hard switching
- **GIVEN** a `DiodeCatalog` with `Qrr = 80 nC`
- **WHEN** the diode commutes from `If = 20 A` with external `di/dt = 500 A/µs`
- **THEN** the integrated reverse current matches `Qrr` within 15%
- **AND** the simulator records `E_rr` in switching-loss telemetry

### Requirement: Catalog Device Loss Integration
Catalog devices SHALL feed switching-energy lookup tables `E_on(Ic, Vds)`, `E_off(Ic, Vds)`, `E_rr(If, di/dt)` directly to the loss accumulator without manual user wiring.

#### Scenario: Hard-switching loss telemetry
- **GIVEN** a hard-switching half-bridge with `MosfetCatalog` devices
- **WHEN** the simulation completes one switching cycle
- **THEN** `BackendTelemetry` includes total `E_on`, `E_off`, `E_rr` per device
- **AND** the values match vendor datasheet within 10% under matching conditions

### Requirement: Catalog Device Library Structure
The repository SHALL maintain a curated catalog under `devices/catalog/<vendor>/<part>.yaml` with at least 6 reference devices: Si MOSFET, SiC MOSFET, GaN HEMT, Si IGBT, SiC Schottky, fast-recovery Si diode.

#### Scenario: Reference catalog present
- **WHEN** the repository builds
- **THEN** `devices/catalog/` contains the 6 reference devices
- **AND** each device has a parity test under `benchmarks/circuits/catalog_vendor_parity/`

#### Scenario: Catalog model lookup by name
- **GIVEN** YAML netlist with `model: wolfspeed/C3M0065090J`
- **WHEN** the parser loads the netlist
- **THEN** the catalog device is instantiated from `devices/catalog/wolfspeed/C3M0065090J.yaml`
- **AND** missing model name produces a deterministic diagnostic

### Requirement: Datasheet Importer Pipeline
The library SHALL provide importers for SPICE `.lib` files, PLECS `.xml` files (where license permits), and PDF datasheets via the `datasheet-intelligence` skill.

#### Scenario: SPICE library import
- **GIVEN** a vendor `.lib` file containing a Level-3 MOSFET subcircuit
- **WHEN** `pulsim.import_spice_lib("vendor.lib", "M_PART")` is called
- **THEN** a `MosfetCatalogParams` dataclass is returned with parameter mapping
- **AND** unsupported SPICE elements are reported with suggested manual fixes

#### Scenario: PDF datasheet import
- **GIVEN** a datasheet PDF with conventional layout
- **WHEN** `pulsim.import_datasheet("part.pdf", device_class="mosfet")` is called
- **THEN** the importer extracts Rds_on, Vth, Coss(Vds), body-diode params
- **AND** the user can review/edit before saving as catalog YAML

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

