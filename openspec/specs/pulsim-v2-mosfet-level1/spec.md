# pulsim-v2-mosfet-level1 Specification

## Purpose

`MosfetLevel1` (Layer 2 V13 / V13.1) is the v2 implementation of the Shichman-Hodges Level 1 MOSFET drain-current law:

```
I_D(V_GS, V_DS) =
    0                              if V_GS ≤ V_T              (cutoff)
    K · (2·V_OV·V_DS − V_DS²)·(1 + λ·V_DS)   if V_DS < V_OV   (triode)
    K · V_OV² · (1 + λ·V_DS)       if V_DS ≥ V_OV             (saturation)
```

where `V_OV = V_GS − V_T`. Pulsim's implementation differs from textbook SH1 by:

1. **C¹-smooth blending** between cutoff/triode/saturation via two sigmoids of sharpness `kappa`. This keeps the Jacobian continuous → Newton converges quadratically without limiter heuristics.
2. **Smooth V_DS clamp** (V13.2) to keep the AD pass out of the spurious `V_DS < 0` polynomial root that real Si transistors never visit.
3. **Forward-mode AD** via `ADRealN<3>` over `(V_DS, V_GS, V_DS_blend)` to stamp the Newton Jacobian analytically.

The branch SHALL be added with `BranchKind::Nonlinear`; `refresh_mosfets_level1` stamps the per-iteration contributions.

## Requirements

### Requirement: SH1 drain-current law with smooth region blending

The `MosfetLevel1::current(v, p)` function SHALL evaluate the SH1 drain current at the operating point `(V_DS, V_GS)` blending cutoff/triode/saturation through C¹-smooth sigmoids parameterised by `kappa`.

For nominal Si NMOS parameters (`K = 1 mA/V²`, `V_T = 2 V`, `λ = 0.02 1/V`, `kappa = 15 1/V`):

- Cutoff (V_GS ≤ V_T − 0.1): `I_D` SHALL be within 100 nA of zero.
- Deep saturation (V_GS ≫ V_T + 0.5, V_DS ≫ V_OV): `I_D` SHALL match the saturation formula within 1 %.
- Deep triode (V_DS ≪ V_OV): `I_D` SHALL match the triode formula within 1 %.

#### Scenario: Cutoff region produces ~0 current

- **GIVEN** `MosfetLevel1` with `V_T = 2 V`
- **WHEN** `current` is evaluated at `V_GS = 0`, `V_DS = 5`
- **THEN** the result SHALL be < 100 nA in magnitude

#### Scenario: Saturation region matches K·V_OV²

- **GIVEN** `MosfetLevel1{K = 1 mA/V², V_T = 2, λ = 0}`
- **WHEN** evaluated at `V_GS = 3`, `V_DS = 5` (V_OV = 1, deep saturation)
- **THEN** `I_D` SHALL be 1 mA within 1 %

#### Scenario: Common-source DC operating point

- **GIVEN** a common-source amplifier with `V_DD = 10 V`, `R_D = 5 kΩ`, `V_GS = 3 V`, `V_T = 2 V`, `K = 1 mA/V²`
- **WHEN** the transient simulator runs to steady state with Newton refresh enabled
- **THEN** the drain voltage SHALL converge to `~4.55 V` (the analytical self-consistent solution)

### Requirement: CircuitBuilder helper

`CircuitBuilder::add_mosfet_level1(name, drain, source, gate, K, V_T, lambda, kappa, with_body_diode)` SHALL register a 3-terminal SH1 MOSFET as a `BranchKind::Nonlinear` branch with `gate` as a sensed node (no gate current).

Default values: `lambda = 0.02`, `kappa = 15.0`, `with_body_diode = false`.

When `with_body_diode = true` (proposal #3.1), the helper SHALL also create an anti-parallel `SwitchedDiode` (source → drain) with `V_th = 0.5 V` to handle inductive-load freewheeling.

#### Scenario: Builder creates one Nonlinear branch

- **GIVEN** `add_mosfet_level1("M1", "drain", "source", "gate", K=1e-3, V_T=2.0)`
- **WHEN** the builder is queried
- **THEN** `num_branches` SHALL equal 1 (no body diode by default)
- **AND** `pool.kind_of(0)` SHALL equal `StoredKind::MosfetLevel1`
- **AND** `pool.mosfet_level1_gate_node(0)` SHALL equal `node_id_of("gate")`

#### Scenario: with_body_diode adds an anti-parallel diode

- **GIVEN** `add_mosfet_level1(..., with_body_diode=true)`
- **WHEN** the builder is queried
- **THEN** `num_branches` SHALL equal 2
- **AND** branch 1 SHALL be `StoredKind::Diode` with `from = source`, `to = drain` (anti-parallel direction)

### Requirement: Python binding

The `pulsim.v2.CircuitBuilder.add_mosfet_level1` Python method SHALL expose `K`, `V_T`, `lambda_`, `kappa`, and `with_body_diode` keyword arguments matching the C++ signature.

Note: the C++ `lambda` keyword is bound as `lambda_` in Python (avoiding the Python keyword conflict).

#### Scenario: Python builder honours with_body_diode

- **GIVEN** a Python `CircuitBuilder` with `add_mosfet_level1("M1", "drain", "source", "gate", K=1e-3, V_T=2.0, with_body_diode=True)`
- **WHEN** `builder.num_branches` is queried
- **THEN** it SHALL return 2 (MOSFET + body diode)
