## ADDED Requirements

### Requirement: Power MOSFET Builder Binding

Python SHALL expose `CircuitBuilder.add_power_mosfet(name, drain,
source, gate, *, K, V_T, lambda_=0.0, kappa=15.0, R_d=0.0, C_gs, C_gd,
C_ds, R_g=0.0, L_s=0.0, body_diode=None, T_j=25.0, rds_tc1=0.0,
rds_tc2=0.0, vt_tc=0.0)`. `C_gd` and `C_ds` SHALL accept a float (linear), a
`ChargeTable`, or a law object. `components()` SHALL report one
`power_mosfet` entry carrying every parameter and the names of its
sub-elements. The type stubs SHALL describe the method and `ChargeTable`.

#### Scenario: One call replaces the hand-built circuit
- **GIVEN** the hand-built double pulse of the review probe `p4_dpt.py` (Level-1
  device, external 1 nF and 100 pF capacitors, 10 Ω gate resistor)
- **WHEN** the same circuit is built with one `add_power_mosfet` call
- **THEN** E_on = 136.14 µJ and E_off = 158.96 µJ within 1e-9 relative

#### Scenario: Parameters are introspectable
- **WHEN** `components()` is called on a builder holding a power MOSFET
- **THEN** the entry lists `R_g`, `C_gs`, the law kind of `C_gd` and
  `C_ds`, and `T_j`

### Requirement: Double-Pulse Test Helper

Python SHALL provide `pulsim.double_pulse_test(device, *, V_dd, I_load,
R_g_ext, V_drive, L_load, L_stray=0.0, freewheel=None, T_j=25.0,
engine="trbdf2")`. It SHALL build the clamped inductive half-bridge
test, run it, and return E_on,
E_off, the Miller plateau duration, dv/dt, di/dt, the peak reverse
recovery current and the waveforms. Energies SHALL be time-weighted
integrals of `v_DS·i_D` on the result's own grid.

#### Scenario: Energies do not depend on the grid
- **WHEN** the helper runs the same device on `pwl` (uniform grid) and
  on `trbdf2` (adaptive grid)
- **THEN** the returned energies differ by less than 0.5 %

### Requirement: Datasheet Fitters

`pulsim.datasheet` SHALL provide `fit_charge_table(v, C, *, tol=0.01)`,
`fit_mosfet_channel(v_gs, i_d, *, v_ds)`,
`fit_rdson_temperature(T_j, r_norm)`,
`fit_body_diode_recovery(*, Q_rr, I_rrm, I_F, di_dt, V_R)` and
`power_mosfet_from_datasheet(...)`. Each SHALL return the fitted
parameters together with the largest residual, and SHALL raise when the
residual exceeds its documented limit instead of returning a poor fit.

#### Scenario: Round trip recovers the parameters
- **GIVEN** a transfer curve generated from `K = 1.43 A/V²`,
  `V_T = 3.0 V`
- **WHEN** `fit_mosfet_channel` is applied
- **THEN** it returns `K` and `V_T` within 1e-6 relative

#### Scenario: Capacitance curve to charge table
- **GIVEN** 30 points of a C_rss(V_DS) curve spanning two decades
- **WHEN** `fit_charge_table` is applied with `tol=0.01`
- **THEN** the table's `C(v)` reproduces every input point within 1 %
  and the returned residual says so

### Requirement: VDMOS Import Keeps Charge Parameters

The SPICE importer SHALL map a VDMOS card's CGS to `C_gs`,
CGDMAX/CGDMIN/A to the `vdmos_cgd` law, CJO/VJ/M to the `C_ds` junction
law, TT to the body diode's `tau`, and RG to `R_g`, and SHALL build a
power MOSFET when any of them is present. It SHALL emit one summary
warning listing what was kept and what is still dropped.
`charge="drop"` SHALL restore the previous ideal-gate Level-1 device. A
card carrying none of these parameters SHALL import bit-identically.

#### Scenario: Capacitances survive the import
- **GIVEN** a card with `CGS=1n CGDMAX=1n CGDMIN=50p A=0.5 RG=3`
- **WHEN** the deck is imported
- **THEN** `components()` shows a `power_mosfet` with `C_gs = 1 nF`,
  `R_g = 3 Ω` and a `vdmos_cgd` law with those three values

#### Scenario: Opt-out
- **WHEN** the same deck is imported with `charge="drop"`
- **THEN** the result equals the importer's output before this change

### Requirement: PLECS Formula Evaluation at Import

`import_plecs_xml(path, variables=None)` SHALL evaluate a
"table and formula" computation method at import time when `variables`
supplies a value for every variable the formula uses, producing a plain
table and a provenance note recording the formula and the values. The
evaluator SHALL accept arithmetic expressions only and SHALL NOT execute
arbitrary code. A formula that uses a custom lookup table, or a variable
without a supplied value, SHALL be refused with the existing error.

#### Scenario: Gate-resistance scaling applied
- **GIVEN** a file whose E_on is "table and formula", whose formula
  multiplies the table value by `(1+0.02*(Rg-10))`, with a declared
  variable `Rg`
- **WHEN** it is imported with `variables={"Rg": 20}`
- **THEN** every E_on entry is 1.2 times the raw table entry and the
  part's provenance notes name the formula and `Rg = 20`

#### Scenario: Missing variable refused
- **WHEN** the same file is imported without `variables`
- **THEN** the import raises `PlecsImportError` naming `Rg`
