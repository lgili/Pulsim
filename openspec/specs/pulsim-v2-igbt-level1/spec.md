# pulsim-v2-igbt-level1 Specification

## Purpose

`IgbtLevel1` (Layer 2 V14) is a behavioural Insulated-Gate Bipolar Transistor model with linear conduction during the ON state. It exposes a simpler dynamic envelope than the SH1 MOSFET — IGBTs are typically used in higher-power applications where the cubic drain-current law is less relevant; the loss curve is dominated by the on-state `V_CE_sat` drop plus a linear `R_CE_sat · I_C` slope.

The conduction law is:

```
I_C(V_CE, V_GE) =
    0                                          if V_GE ≤ V_T  (off)
    (V_CE − V_CE_sat) / R_CE_sat               if V_GE > V_T  (on, linear)
```

with smooth `sigmoid(V_GE − V_T)` gating to keep the Jacobian continuous for Newton iteration.

Architecturally identical to `MosfetLevel1`: a 3-terminal device with `collector → emitter` as a `BranchKind::Nonlinear` branch and `gate` as a sensed node (no gate current). The `refresh_igbts_level1` Newton refresh stamps the per-iteration contributions.

## Requirements

### Requirement: Linear-conduction IGBT current law

`IgbtLevel1::current(v, p)` SHALL evaluate the linear-conduction collector current at `(V_CE, V_GE)` with smooth on/off blending.

For nominal IGBT parameters (`V_CE_sat = 2.0 V`, `R_CE_sat = 0.05 Ω`, `V_T = 5.0 V`, `kappa = 10`):

- Off state (V_GE ≤ V_T − 0.3): `I_C` SHALL be within 1 mA of zero (even for V_CE up to 600 V).
- On state (V_GE ≥ V_T + 0.5): `I_C` SHALL match `(V_CE − V_CE_sat) / R_CE_sat` within 5 %.

#### Scenario: Off-state blocks high V_CE

- **GIVEN** `IgbtLevel1` with `V_T = 5 V`
- **WHEN** evaluated at `V_GE = 0`, `V_CE = 600`
- **THEN** `I_C` SHALL be less than 10 mA (effectively blocking)

#### Scenario: On-state matches linear conduction

- **GIVEN** `IgbtLevel1{V_CE_sat = 2.0, R_CE_sat = 0.05, V_T = 5.0}`
- **WHEN** evaluated at `V_GE = 15 V`, `V_CE = 4 V`
- **THEN** `I_C` SHALL be approximately `(4 − 2) / 0.05 = 40 A` within 5 %

### Requirement: CircuitBuilder helper

`CircuitBuilder::add_igbt_level1(name, collector, emitter, gate, V_CE_sat, R_CE_sat, V_T, kappa)` SHALL register a 3-terminal IGBT as a `BranchKind::Nonlinear` branch.

Default parameters: `V_CE_sat = 2.0`, `R_CE_sat = 0.05`, `V_T = 5.0`, `kappa = 10.0`.

#### Scenario: Builder creates one Nonlinear branch with gate sense

- **GIVEN** `add_igbt_level1("T1", "C", "E", "G")`
- **WHEN** the builder is queried
- **THEN** `num_branches` SHALL equal 1
- **AND** `pool.kind_of(0)` SHALL equal `StoredKind::IgbtLevel1`
- **AND** `pool.igbt_level1_gate_node(0)` SHALL equal `node_id_of("G")`

### Requirement: Boost converter integration

When used in a boost-converter test fixture with a `PulseVoltageSource` rise-time gate drive, the IGBT SHALL switch correctly between off (blocking V_in_max) and on (conducting `I_L`), and the output capacitor SHALL charge to the boost-voltage steady-state within ~5 % tolerance.

#### Scenario: Boost converter with realistic IGBT settles

- **GIVEN** a boost converter with `V_in = 12 V`, IGBT gate at 50 kHz / 40 % duty
- **WHEN** the transient simulator runs for 5 ms with Newton refresh enabled
- **THEN** the mean output voltage SHALL be within 5 % of the duty-corrected boost prediction `V_in / (1 − D)` (after subtracting `V_CE_sat` and the diode V_F)
