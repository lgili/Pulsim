## ADDED Requirements

### Requirement: SPICE Piecewise-Linear Diode Import

The SPICE importer SHALL map a diode model that sets any of `Ron`,
`Roff` or `Vfwd` to the PWL switched diode:
`add_diode(g_on = area/Ron, g_off = area/Roff, V_th = 0, V_f = Vfwd)`.
It SHALL NOT create a smooth Newton diode for such a model. The
refusals for reverse breakdown (`Vrev`, `Rrev`, `Revilimit`) and for
parameter sets that mix the piecewise-linear and the junction
parameters SHALL remain.

#### Scenario: Imported diode drops Vfwd plus Ron·i
- **GIVEN** a netlist with `.model DX D(Ron=0.05 Roff=1Meg Vfwd=0.7)`
  and a diode using it
- **WHEN** it is imported and 10 A are forced through the diode
- **THEN** the branch kind is `StoredKind.Diode`, its parameters are
  `g_on = 20 S`, `g_off = 1e-6 S`, `V_f = 0.7 V`, the voltage is 1.2 V
  and `device_loss_summary` reports 12.0 W

#### Scenario: Area scales the conductances
- **WHEN** the same diode is instantiated with `area = 2`
- **THEN** `g_on = 40 S` and `g_off = 2e-6 S`

#### Scenario: Breakdown is still refused
- **WHEN** the model also sets `Vrev=50`
- **THEN** the import raises an error naming the diode and `Vrev`
