## ADDED Requirements

### Requirement: Hysteretic Inductor with Jiles-Atherton Model

The library SHALL provide a `HystereticInductor` device that implements the simplified Jiles-Atherton (JA) magnetization model. The device SHALL maintain magnetization `M` as an internal state in addition to the inductor branch flux already tracked by MNA, and the parameter set SHALL be the canonical `(Ms, a, alpha, k, c)` quintuple.

#### Scenario: Major B-H loop matches analytical reference

- **GIVEN** a `HystereticInductor` configured with the standard 3F3 ferrite JA parameter set
- **WHEN** the device is excited by a 50 Hz sinusoidal field of amplitude sufficient to drive `M` to 90 % of `Ms`
- **THEN** the simulated B-H loop area matches the analytically integrated JA loop area within 5 %
- **AND** the simulated coercive field `H_c` matches the analytical `H_c(k, alpha)` within 5 %
- **AND** the simulated remanent flux density `B_r` matches the analytical value within 5 %

#### Scenario: Minor loops nested inside major loop

- **GIVEN** the same JA-parametrised inductor previously driven through a complete major loop
- **WHEN** the field excitation is reduced to a sub-major amplitude and held for at least three cycles
- **THEN** the resulting minor loop lies fully inside the major loop on the B-H plane
- **AND** consecutive minor loops differ by less than 1 % in enclosed area (numerical convergence)

#### Scenario: Inrush from non-zero residual flux

- **GIVEN** a `HystereticInductor` whose initial magnetization state is set to 80 % of the previous-shutdown remanent value
- **WHEN** a step voltage is applied at the worst-case zero-crossing phase
- **THEN** the simulated peak current exceeds the linear-inductor prediction by at least 3×
- **AND** the peak current is bounded by the analytical saturation-current limit `H_sat * l_m / N` within 10 %

#### Scenario: Linear-regime equivalence to plain Inductor

- **GIVEN** a `HystereticInductor` operated below 10 % of `Ms` so the JA non-linearity is negligible
- **WHEN** the simulation runs against an equivalent linear `Inductor` whose `L = (mu_0 * (1 + chi_init)) * (N² A / l_m)` matches the JA initial slope
- **THEN** terminal current waveforms match within 1 % on a 50 Hz sinusoidal excitation

### Requirement: JA Parameter Catalog and Fitter

The library SHALL ship pre-fitted Jiles-Atherton parameter sets for the same core materials already covered by the Steinmetz catalog, and SHALL provide a `fit_ja_from_bh_curve(B_array, H_array)` helper that returns a fitted `JilesAthertonParams` from a measured B-H loop.

#### Scenario: Material catalog lookup

- **GIVEN** a user calls `pulsim.magnetic.ja_params("3F3")`
- **WHEN** the lookup succeeds
- **THEN** the returned `JilesAthertonParams` contains finite, positive `Ms`, `a`, `alpha`, `k`, `c` values
- **AND** simulating the parameters in a single-period sinusoidal-excitation circuit produces a loop whose coercive field is within 10 % of the material's datasheet value

#### Scenario: Curve-fit reproduces source loop

- **GIVEN** a synthetic B-H loop generated from known JA parameters
- **WHEN** `fit_ja_from_bh_curve` is called on the synthetic data
- **THEN** the recovered parameters match the source within 5 % on each of `Ms`, `a`, `alpha`, `k`, `c`
