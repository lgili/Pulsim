# magnetic-models Specification

## Purpose
TBD - created by archiving change add-magnetic-core-models. Update Purpose after archive.
## Requirements
### Requirement: Saturable Inductor Device
The library SHALL provide a `SaturableInductor` device with flux-linkage state and a configurable B-H curve (table, arctan, or Langevin).

#### Scenario: Linear regime equivalence
- **GIVEN** a `SaturableInductor` with B-H slope = `L_lin / N²` and below-saturation operation
- **WHEN** simulation runs vs equivalent linear `Inductor`
- **THEN** the two simulations differ by ≤0.1% on terminal current waveform

#### Scenario: Saturation transition
- **GIVEN** a `SaturableInductor` with table-defined saturation knee
- **WHEN** the operating flux exceeds the knee
- **THEN** the simulated current accelerates as predicted by the B-H curve within 5%
- **AND** the inverse curve `i(λ)` is monotonic to ensure unique solution

### Requirement: Saturable Transformer Device
The library SHALL provide a `SaturableTransformer` device with per-winding leakage inductance, a saturable magnetizing branch, and optional eddy-current branch per winding.

#### Scenario: Two-winding saturable transformer
- **GIVEN** primary leakage `Lp = 5 µH`, secondary leakage `Ls = 10 µH`, magnetizing `Lm = 2 mH` saturable at λ_sat
- **WHEN** simulation runs with applied primary voltage above saturation knee
- **THEN** the magnetizing current demonstrates saturation behavior matching expected `i_m(λ)`
- **AND** secondary current reflects the saturated magnetizing component

#### Scenario: Cold-start inrush
- **GIVEN** a 1500 VA mains transformer applied at voltage zero crossing
- **WHEN** simulation runs for 5 mains cycles
- **THEN** peak primary current matches analytical Faraday integral within 20%
- **AND** the inrush decays consistent with parasitic resistance

### Requirement: Steinmetz Core Loss
Saturable magnetic devices SHALL accept Steinmetz parameters `(k, α, β)` and report cycle-averaged core loss `P = k · f^α · B^β` in `BackendTelemetry`.

#### Scenario: Sinusoidal flux core loss
- **GIVEN** a saturable inductor with Steinmetz `(k=2.5, α=1.6, β=2.6)` driven by 100 kHz sine flux of B = 0.1 T
- **WHEN** the simulation reaches steady state
- **THEN** the reported `core_loss_avg_W` matches `2.5 · 100000^1.6 · 0.1^2.6` within 10%

#### Scenario: Non-sinusoidal flux uses iGSE
- **GIVEN** a triangular flux waveform from a buck-derived inductor
- **WHEN** core loss is computed
- **THEN** the iGSE (improved Generalized Steinmetz) integral is used over the cycle
- **AND** the result is reported separately from the sinusoidal Steinmetz formula

### Requirement: Jiles-Atherton Hysteresis Model
The library SHALL support an opt-in Jiles-Atherton hysteresis model for saturable devices, parameterized by `(Ms, a, α_jt, k, c)`.

#### Scenario: Hysteresis disabled by default
- **GIVEN** a saturable device with no `jiles_atherton` block
- **WHEN** the simulation runs
- **THEN** only the anhysteretic B-H curve is used
- **AND** no hysteresis loop area appears in the H-B plot

#### Scenario: Hysteresis enabled per device
- **GIVEN** a saturable device with `jiles_atherton: { Ms: ..., a: ..., ... }`
- **WHEN** the simulation runs
- **THEN** the H-B trajectory exhibits a closed hysteresis loop
- **AND** loop area corresponds to per-cycle hysteresis loss

### Requirement: Eddy-Current Lumped Model
Saturable transformer windings SHALL accept an optional `eddy_current: { r_eff, l_eff }` block for skin/proximity loss modeling.

#### Scenario: Litz wire skin effect
- **GIVEN** a winding declared as `litz_strands: 100, strand_diameter: 0.1mm`
- **WHEN** the parser builds the device
- **THEN** the effective `R_eddy(f)` matches analytical litz-wire formula within 10%
- **AND** the eddy-loss contribution is reflected in winding loss telemetry

### Requirement: Magnetic Core Catalog
The repository SHALL host a catalog of magnetic core materials under `devices/cores/<vendor>/<material>.yaml` with at least 4 reference materials.

#### Scenario: Core library lookup
- **GIVEN** YAML netlist with `core_model: ferroxcube/N87`
- **WHEN** the parser loads
- **THEN** the device uses the catalog parameters
- **AND** missing material name produces a deterministic diagnostic with suggestion list

### Requirement: Core Datasheet Importer
The library SHALL provide `pulsim.import_core_datasheet(pdf, manufacturer)` that extracts Steinmetz parameters and B-H curve from typical core datasheets.

#### Scenario: Datasheet import end-to-end
- **GIVEN** a Ferroxcube N87 datasheet PDF
- **WHEN** `import_core_datasheet("n87.pdf", "ferroxcube")` is called
- **THEN** a `CoreParams` dataclass is returned with Steinmetz `(k, α, β)` and B-H curve
- **AND** the dataclass can be serialized to a runnable catalog YAML

