## ADDED Requirements

### Requirement: Permeance Magnetic Network

The library SHALL provide a magnetic-circuit domain in which a magnetic
node's potential is the magnetomotive force in ampere, the through
variable is the flux rate in volt, a linear permeance `P` [H] obeys
`Φ = P·F`, and a winding of `N` turns couples an electrical port to a
magnetic port by `v = N·dΦ/dt` and `F = N·i`. The elements SHALL be
available through the builder, Python and the YAML types `winding`,
`permeance` and `air_gap` (`P = μ₀·A/l`). A magnetic island without a
reference node SHALL be refused naming one of its nodes. The network
SHALL conserve flux at every node.

#### Scenario: Gapped core inductance
- **GIVEN** a 25-turn winding on a core of 76 mm² and 72 mm with
  `μ_r = 2000` (2.653 µH) in series with a 0.5 mm gap (191.0 nH)
- **WHEN** a voltage step is applied to the winding
- **THEN** the terminal inductance is 111.36 µH within 1e-9 relative

#### Scenario: Three-limb core
- **GIVEN** three limbs of 5 µH, each with a 100-turn winding, joined by
  ideal yokes, and an air return path of 50 nH between the yokes
- **WHEN** the windings are excited in positive sequence and in zero
  sequence
- **THEN** the per-phase inductances are 50 mH and 166.1 µH, each within
  1e-6 relative

#### Scenario: Integrated coupled inductor
- **GIVEN** an EE core with 10-turn windings on outer legs of 2 µH and a
  centre leg of 0.5 µH
- **WHEN** the self and mutual inductances are measured
- **THEN** `L = 111.1 µH` and `k = 0.8` within 1e-6 relative

#### Scenario: Flux is conserved
- **WHEN** any transient with a magnetic network runs
- **THEN** the sum of the fluxes entering each internal magnetic node
  is below 1e-12 of the largest flux, at every step

#### Scenario: Island without a reference
- **WHEN** a magnetic network has no node tied to the reference
- **THEN** building fails with an error naming a node of the island

### Requirement: Equivalence with the T-Model Transformer

A two-winding transformer SHALL give the same terminal behaviour in
permeance form (each winding shunted by its leakage permeance, joined
through the core permeance) as the T-model saturable transformer with
`L_m = N_p²·P_core`, `L_leak,p = N_p²·P_σp`, `L_leak,s = N_s²·P_σs`.

#### Scenario: Below saturation
- **WHEN** both forms run on the same engine and grid with the core
  below its knee
- **THEN** primary and secondary currents agree within 1e-6 relative

#### Scenario: Into saturation
- **GIVEN** the same material law in both forms and a drive reaching
  three times the knee current
- **WHEN** both run
- **THEN** the currents agree within 1e-3 relative
- **AND** the saturated inductance of the 25-turn, 76 mm², 72 mm,
  0.5 mm-gap core is 0.823 µH within 1 %

### Requirement: Nonlinear Permeance

The library SHALL provide a saturable permeance `Φ(F) = A·B(F/l)` built
from the gapped-core material law in closed form or from a `FluxTable`,
and a hysteretic permeance built from the Jiles–Atherton law with its
exact tangent. Both SHALL be stamped in flux form, with `dΦ/dF` the
exact derivative of the evaluated `Φ(F)` and strictly positive. Invalid
geometry or material parameters SHALL be refused naming the element.

#### Scenario: Law is monotone at fine spacing
- **WHEN** `Φ(F)` of a saturable permeance is evaluated at 1 pA-turn
  spacing around zero, the knee and ten times the knee
- **THEN** it is strictly increasing

#### Scenario: Hysteretic permeance equals the hysteretic inductor
- **GIVEN** one core described as a hysteretic permeance with a winding
  and as `add_hysteretic_core_inductor`
- **WHEN** both run with the same excitation
- **THEN** the winding currents agree within 1e-6 relative

### Requirement: In-Loop Core Loss

The library SHALL provide a magnetic resistance `F = R_m·dΦ/dt` for
classical eddy loss and an excess-loss element whose magnetomotive
force follows `c·sign(u)·|u|^0.5` for flux rates `|u| ≫ ε` and is
continuously differentiable at zero. Their dissipation SHALL load the
circuit during the run and SHALL be reported per element.

#### Scenario: Eddy loss under sinusoidal flux
- **GIVEN** a 10-turn winding driven by a 100 V peak sine, with
  `R_m = 0.01 S` in the core branch
- **WHEN** the run reaches steady state
- **THEN** the element dissipates 0.500 W within 1e-4 relative, and the
  source supplies that power

#### Scenario: Excess loss under sinusoidal flux
- **GIVEN** the same drive with an excess-loss element `c = 0.01`,
  `ε = 0.01 V`
- **WHEN** the run reaches steady state
- **THEN** the element dissipates 0.1759 W within 0.5 %

### Requirement: Magnetostatic DC Operating Point

The DC operating point SHALL solve a magnetic network by flux
conservation: at a node whose branches are all permeances, the net flux
equals its initial value.

#### Scenario: Gapped core under DC bias
- **GIVEN** a winding carrying 2 A DC on a core with a gap
- **WHEN** the DC operating point is computed
- **THEN** the magnetomotive force divides between core and gap in
  proportion to their reluctances within 1e-9 relative
- **AND** the transient that follows starts without a flux jump
