## ADDED Requirements

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
