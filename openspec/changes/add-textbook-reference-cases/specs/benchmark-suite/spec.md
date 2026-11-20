## ADDED Requirements

### Requirement: Published-Values Validation Type
The benchmark suite SHALL support a `published` validation type that compares measured KPIs against numerical values quoted from a peer-reviewed textbook or published paper, with a per-key tolerance.

#### Scenario: Erickson buck steady-state matches printed values
- **GIVEN** the Erickson CCM buck (24 V, D = 0.5, 50 kHz, 47 uH, 100 uF, 5 ohm) declares `validation: { type: published, values: { v_out: 12.0, delta_il_pp: 2.553, delta_vc_pp: 0.0638 }, tolerance_pct: 2 }` (peak-to-peak ripples from dI = (Vg - V) D Ts / L and dV = dI / (8 f C))
- **WHEN** the runner executes the benchmark and extracts the corresponding KPIs
- **THEN** each measured value is within ±2 % of the declared value
- **AND** the per-key error is surfaced in results JSON as `published__v_out_err_pct`, etc.

### Requirement: Textbook Citation Inline Documentation
Every benchmark using the `published` validation type SHALL include an inline comment block citing the exact source (textbook title, edition, section, equation/figure number) so readers can verify the reference independently.

#### Scenario: Textbook benchmark includes citation block
- **GIVEN** a benchmark YAML with `validation: type: published`
- **WHEN** a reader opens the YAML
- **THEN** the YAML's header comment names the textbook, edition year, section number, and the specific equation or figure being reproduced

### Requirement: Textbook Cases Against an Independent Truth
Each textbook reference case SHALL be checked twice: its independent truth (closed form or mpmath orbit, never a simulator number) against the published formula within the published tolerance, and every engine that admits the circuit, run through explicit options, against that truth within a tolerance its method explains.

#### Scenario: DCM point mislabelled as CCM is caught by the truth
- **GIVEN** the Erickson buck at 10 ohm (K = 0.47 < 1 - D)
- **WHEN** its exact periodic orbit is computed
- **THEN** it is in DCM with V_out = 12.25 V, matching the DCM formula within 2 % and the CCM value D * Vg = 12 V not within 2 %

#### Scenario: Every engine meets the truth
- **GIVEN** a textbook case started on its exact orbit
- **WHEN** it runs on pwl (with and without discontinuity_restart), trbdf2, trbdf2 + esdirk43 and dsed
- **THEN** each engine's states and windowed KPIs are within its method's tolerance of the truth, or the case is a strict xfail naming a finding
