## ADDED Requirements

### Requirement: Published-Values Validation Type
The benchmark suite SHALL support a `published` validation type that compares measured KPIs against numerical values quoted from a peer-reviewed textbook or published paper, with a per-key tolerance.

#### Scenario: Erickson buck steady-state matches printed values
- **GIVEN** `textbook_erickson_buck_3_1.yaml` declares `validation: { type: published, values: { v_out: 12.0, delta_il: 1.6, delta_vc: 0.024 }, tolerance_pct: 2 }`
- **WHEN** the runner executes the benchmark and extracts the corresponding KPIs
- **THEN** each measured value is within ±2 % of the declared value
- **AND** the per-key error is surfaced in results JSON as `published__v_out_err_pct`, etc.

### Requirement: Textbook Citation Inline Documentation
Every benchmark using the `published` validation type SHALL include an inline comment block citing the exact source (textbook title, edition, section, equation/figure number) so readers can verify the reference independently.

#### Scenario: Textbook benchmark includes citation block
- **GIVEN** a benchmark YAML with `validation: type: published`
- **WHEN** a reader opens the YAML
- **THEN** the YAML's header comment names the textbook, edition year, section number, and the specific equation or figure being reproduced
