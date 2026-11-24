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
The repository SHALL maintain a curated catalog under `python/pulsim/lib/data/catalog/<vendor>/<part>.yaml` (package data, behind `pulsim.lib`). Every part file SHALL carry a `provenance` block. A shipped part whose data is not the manufacturer's SHALL declare `provenance.method: synthetic`, and `pulsim.lib.part()` SHALL refuse to resolve a synthetic part by manufacturer part number unless `allow_synthetic=True`.

A shipped part file SHALL NOT carry a manufacturer part number whose package, topology or existence its own content contradicts. The v2.0 release DELETED four files that did (a SiC MOSFET filed under the wrong package variant, a dual common-cathode diode described as a single device, a GaN HEMT in the wrong package, and a part number absent from the vendor's line) rather than correcting the headers, because the electrical data behind them was invented either way — the count of reference devices is therefore no longer a requirement. Real device data enters through `pulsim.lib.import_plecs_xml()` or a user directory on `PULSIM_PARTS_PATH`.

#### Scenario: Reference catalog present
- **WHEN** the repository builds
- **THEN** `python/pulsim/lib/data/catalog/` contains at least one device per shipped class
- **AND** every file in it loads through `pulsim.lib.list_parts()` with a valid provenance block

#### Scenario: Catalog model lookup by name
- **GIVEN** YAML netlist with `model: infineon/IKW40N120T2`
- **WHEN** the parser loads the netlist
- **THEN** the catalog device is instantiated from `python/pulsim/lib/data/catalog/Infineon/IKW40N120T2.yaml` via `pulsim.lib.igbt('IKW40N120T2', allow_synthetic=True)`
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

### Requirement: Linear Controlled Sources

The kernel SHALL provide a voltage-controlled current source
(`i = g_m·v_c`), a current-controlled voltage source (`v = r_m·i_c`), a
current-controlled current source (`i = β·i_c`) and a gyrator
(`i_1 = g·v_2`, `i_2 = −g·v_1`). Each SHALL be a constant linear stamp
that does not depend on the switch mask. A current output SHALL follow
the terminal convention of the DC current source: a positive value
leaves the source at `from`. The controlling branch of a
current-controlled source SHALL carry a current unknown; any other
branch SHALL be refused with an error naming both elements and the
valid kinds. A non-finite parameter SHALL be refused by name.

#### Scenario: Transconductance
- **GIVEN** a VCCS with `g_m = 2 mS` controlled by 5 V, its `from` node
  loaded with 1 kΩ to ground and `to` grounded
- **WHEN** the DC operating point is solved
- **THEN** the `from` node is at +10.000 V within 1e-12 relative

#### Scenario: Transresistance
- **GIVEN** 10 mA flowing from `from` to `to` through a 0 V sense
  source, and a CCVS with `r_m = 100 Ω` controlled by it
- **WHEN** the circuit is solved
- **THEN** the CCVS output is 1.000 V within 1e-12 relative

#### Scenario: Current gain
- **GIVEN** 1 mA sensed, a CCCS with `β = 50` into a 100 Ω load
- **WHEN** the circuit is solved
- **THEN** the load voltage magnitude is 5.000 V within 1e-12 relative

#### Scenario: Controlling branch without a current unknown
- **WHEN** `add_cccs("F1", control="R1", ...)` names a resistor
- **THEN** it raises an error naming `F1`, `R1` and the branch kinds
  that can be sensed

### Requirement: Gyrator Is Lossless and Inverts Impedance

The gyrator SHALL absorb zero net power at every instant and SHALL
present at one port the impedance `1/(g²·Z)` of the element `Z`
connected to the other.

#### Scenario: Capacitor seen as inductor
- **GIVEN** a gyrator with `g = 0.01 S` and 1 µF on port 2, driven on
  port 1 by a 1 V step through 10 Ω
- **WHEN** it runs on `pwl` beside a 10 mH inductor driven the same way
  on the same grid
- **THEN** the two port-1 currents agree within 1e-9 relative, and the
  current at 1 ms is 63.21 mA

#### Scenario: Power balance
- **WHEN** any transient with a gyrator is run
- **THEN** `|v_1·i_1 + v_2·i_2| ≤ 1e-12·|v_1·i_1|` at every step

#### Scenario: Small-signal impedance
- **WHEN** the AC sweep is run on the gyrator with 1 µF on port 2
- **THEN** the port-1 impedance at 1 kHz is 62.83 Ω at +90° within
  1e-9 relative

### Requirement: Time-Varying Current Sources

The kernel SHALL provide sine, pulse and piecewise-linear table current
sources evaluated without a Python callback. The sine source SHALL
produce `i_dc + i_amplitude·sin(2π·f·t + φ)`. The pulse source SHALL
take the parameters and edge rules of the pulse voltage source. The
table source SHALL interpolate linearly between knots `(t_k, i_k)`,
hold the first value before the first knot, and after the last knot
either hold the last value or repeat the table when `periodic` is set.
A table with fewer than two knots, non-increasing times, a non-finite
value, or `periodic` with unequal end values SHALL be refused naming the
source. The DC operating point SHALL use the value at `t = 0`.

#### Scenario: Sine current into a resistor
- **GIVEN** a sine current source of 1 A amplitude at 1 kHz into 10 Ω
- **WHEN** it runs on `pwl`
- **THEN** the resistor voltage equals `10·i(t)` at every grid point
  within 1e-12 relative

#### Scenario: Table source delivers its charge
- **GIVEN** a table source ramping 0 → 1 A → 0 over 2 ms into 1 mF
- **WHEN** it runs on `pwl` with the knots on the grid
- **THEN** the capacitor voltage rises by 1.000 V within 1e-9 relative

#### Scenario: Breakpoints are not stepped over
- **WHEN** the same source runs on `trbdf2` with knots that no natural
  step would hit
- **THEN** the capacitor voltage rises by 1.000 V within 1e-6 relative

#### Scenario: Invalid table refused
- **WHEN** a table source is given decreasing times
- **THEN** building fails with an error naming the source

### Requirement: Controlled Sources in Every Analysis

The controlled sources and the time-varying current sources SHALL give
the same result on `pwl` and `trbdf2`, SHALL be part of the DC operating
point and of the AC sweep, and on `dsed` SHALL either run and agree or
be refused with the conformance error naming the element and the
engine. Circuits that use none of them SHALL be bit-identical to before.

#### Scenario: Engines agree
- **WHEN** a circuit with each of the four controlled elements runs on
  `pwl` and `trbdf2`
- **THEN** node voltages agree within 1e-6 relative

#### Scenario: No silent engine
- **WHEN** the same circuit is run on `dsed`
- **THEN** it either agrees within 1e-6 or raises the conformance error
  naming the element

### Requirement: PWL Switched Diode Forward Voltage Drop

The PWL `SwitchedDiode` SHALL accept a forward drop `V_f` (default 0).
While ON it SHALL conduct `i = g_on·(v − V_f)` (`v = v_anode −
v_cathode`); while OFF it SHALL conduct `i = g_off·v` with no offset. It
SHALL turn ON when `v ≥ V_on`, where `V_on = V_th` if `V_f = 0` and
`max(V_th, V_f)` otherwise, and SHALL stay ON while `g_on·(v − V_f) > 0`.
Its terminal power SHALL be `V_f·i + i²/g_on` while ON. A `V_f` that is
negative or not finite SHALL be refused with an error naming the diode.
With `V_f = 0` the stamp and every decision SHALL be bit-identical to
the diode without the parameter.

#### Scenario: Forced current gives the drop plus the resistive part
- **GIVEN** 10 A forced through a diode with `g_on = 20 S`,
  `V_f = 0.7 V`
- **WHEN** the circuit is solved (DC operating point, pwl or trbdf2)
- **THEN** `v_D = 1.2 V` and the conduction power is 12.0 W

#### Scenario: Turn-on at the larger of V_th and V_f
- **GIVEN** an OFF diode with `V_f = 0.7`
- **WHEN** `V_th = 0` and `v` reaches 0.7 V, or `V_th = 0.75` and `v`
  reaches 0.75 V
- **THEN** the diode turns ON there and not below

#### Scenario: Turn-off at zero current
- **GIVEN** an ON diode with `V_f = 0.7`
- **WHEN** `v` falls to `V_f − ε`
- **THEN** the diode turns OFF; at `V_f + ε` it stays ON

#### Scenario: Zero drop is unchanged
- **WHEN** a circuit is built with `V_f = 0.0` and run on every engine
- **THEN** its states, switch-state record and loss summary are
  bit-identical to the engine before `V_f` existed (checked once against
  that build; the suite pins its per-device `P_avg` to 1e-9 on pwl and
  1e-6 on the variable-step engines)

#### Scenario: Invalid drop refused by name
- **WHEN** `add_diode("D1", ..., V_f=-0.1)` or `V_f=nan` is called
- **THEN** it raises an error that names `D1` and `V_f`

### Requirement: Retired Smooth-Blend Diode Requires an Explicit Opt-In

The smooth-blend diode (`models::IdealDiode`) SHALL NOT be built from
Python or YAML without an explicit opt-in, because its law
`logistic(κ·(v − V_F0))·(v − V_F0)/R_d` is negative for
`0 < v < V_F0`. `CircuitBuilder.add_nonlinear_diode` SHALL raise
`pulsim.RetiredDeviceError`, a subclass of `ValueError`, unless called
with `allow_non_passive=True`; nothing SHALL be stamped on refusal. The
message SHALL name the device, the measured −1.39 A at +0.636 V and the
replacements `add_diode(..., V_f=)` and `add_shockley_diode`. A YAML
device of `type: nonlinear_diode` SHALL be refused unless it sets
`allow_non_passive: true`. With the opt-in, the law SHALL be unchanged.

#### Scenario: The builder refuses the device
- **WHEN** `add_nonlinear_diode("Dx", "a", "gnd", IdealDiodeParams())`
  is called without the opt-in
- **THEN** `RetiredDeviceError` is raised, its message contains `Dx`,
  `-1.39 A`, `+0.636 V`, `add_diode`, `add_shockley_diode` and
  `allow_non_passive=True`, and the builder has no branch

#### Scenario: The opt-in keeps the legacy law
- **GIVEN** the device built with `allow_non_passive=True` and default
  parameters
- **WHEN** 0.636 V is forced across it on `engine='pwl'`
- **THEN** the diode current is −1.3923 A within 1e-3 A

#### Scenario: YAML refuses without the key
- **WHEN** a YAML circuit declares `type: nonlinear_diode` without
  `allow_non_passive: true`
- **THEN** loading fails with a message naming the device and the key

### Requirement: Composite Body and Freewheel Diode Forward Drop

The diode inside a composite switching device SHALL take the device's
`V_F` as its conduction drop. This applies to the `<name>_body` diode of
`add_mosfet`, `add_mosfet_with_body_diode` and `add_mosfet_level1`, and
to the `<name>_fwd` diode of `add_igbt_level1`. The diode SHALL be a PWL
switched diode with `V_f = V_F` and `V_th = 0`: it SHALL turn on at
`v ≥ V_F`, conduct `i = g_on·(v − V_F)` while ON, conduct `g_off·v`
while OFF, and turn off when its ON current reaches zero. `V_F` SHALL
default to 0.7 V on all four builders. A `V_F` that is negative or not
finite SHALL be refused with an error naming the device. `add_igbt`
SHALL remain a controlled switch with no diode.

#### Scenario: Body diode carries the load with the gate off
- **GIVEN** `add_mosfet` with `R_on = 1 mΩ`, `V_F = 0.7 V` and the
  default diode conductance 1e3 S, gate off
- **WHEN** 10 A are forced from source to drain
- **THEN** `v_SD = 0.710 V` and the diode dissipates 7.10 W (it read
  0.010 V and 0.10 W before)

#### Scenario: Channel conducts with the gate on
- **WHEN** the same 10 A are forced with the gate on
- **THEN** `v_SD = 0.010 V`, the body diode carries less than 1e-6 A
  and the result is the same as before this requirement

#### Scenario: Turn-on voltage is V_F
- **WHEN** the voltage across an OFF body diode with `V_F = 0.7 V`
  rises
- **THEN** the diode turns ON at 0.7 V and not below

#### Scenario: Level-1 devices
- **WHEN** 10 A are forced through the `_body` diode of
  `add_mosfet_level1` with the gate at 0 V, and through the `_fwd`
  diode of `add_igbt_level1(with_fwd=True)` with the gate off, both
  with the default `V_F`
- **THEN** each diode drops 0.710 V (0.010 V before), and with
  `V_F = 1.0` each drops 1.010 V

#### Scenario: Freewheeling buck
- **GIVEN** a 24 V buck at `D = 0.5`, 100 kHz, 100 µH, 100 µF, 1.2 Ω,
  whose low-side device is an `add_mosfet` held off
- **WHEN** 400 periods run on `pwl`, `trbdf2` and `dsed`
- **THEN** the time-weighted output over the last 100 periods is
  11.640 V within 0.005 V and the body-diode loss over the same window is
  3.442 W within 0.01 W on each engine (11.990 V before)

#### Scenario: Invalid drop refused by name
- **WHEN** `add_mosfet("Q1", ..., V_F=-0.1)` is called
- **THEN** it raises an error naming `Q1` and `V_F`

### Requirement: Thermal-Port Capability Declaration and Enforcement
Device models SHALL declare thermal-port capability, and this capability SHALL be enforced by parser/runtime validation.

#### Scenario: Thermal-capable model with thermal port enabled
- **GIVEN** a component type that declares thermal capability
- **WHEN** thermal port configuration is enabled for an instance
- **THEN** the runtime accepts thermal configuration and participates in electrothermal updates

#### Scenario: Non-thermal-capable model with thermal port enabled
- **GIVEN** a component type that declares no thermal capability
- **WHEN** thermal port configuration is enabled for an instance
- **THEN** the configuration is rejected with deterministic diagnostics

### Requirement: Consistent Thermal Parameter Semantics
Thermal-capable device models SHALL apply `rth`, `cth`, `temp_init`, `temp_ref`, and `alpha` with consistent electrothermal semantics.

#### Scenario: Loss-only thermal policy
- **WHEN** global policy is `LossOnly`
- **THEN** losses feed temperature evolution
- **AND** temperature-dependent electrical scaling is not applied

#### Scenario: Loss-with-temperature-scaling policy
- **WHEN** global policy is `LossWithTemperatureScaling`
- **THEN** losses feed temperature evolution
- **AND** bounded temperature-dependent scaling is applied using component thermal parameters

