## ADDED Requirements

### Requirement: Multilevel NPC RMS-error parity gate

The system SHALL provide a regression test
`core/tests/test_multilevel_npc.cpp` that loads
`benchmarks/multilevel/3level_npc.yaml`, runs the transient under
`Preset::Robust`, and computes the per-observable RMS error against
the PLECS-exported golden CSV over one fundamental period.

The test SHALL fail if the RMS error on any of `V_phase_A`, `I_load_A`,
or `V_cap_neutral` exceeds 0.5 %.

#### Scenario: NPC RMS within tolerance

- **GIVEN** the shipped 3-level NPC YAML and PLECS golden CSV
- **WHEN** `test_multilevel_npc` runs under CI
- **THEN** all three observables pass the ≤ 0.5 % RMS gate

### Requirement: Multilevel Flying-Cap RMS-error parity gate

The system SHALL provide
`core/tests/test_multilevel_flying_cap.cpp` gating ≤ 0.5 % RMS on
`V_phase_A`, `I_load_A`, and per-stage `V_cap_FC_{1..4}` against
the PLECS-exported 5-level flying-cap golden.

#### Scenario: Flying-cap RMS within tolerance

- **GIVEN** the shipped 5-level flying-cap YAML and golden
- **WHEN** the test runs
- **THEN** all observables pass

### Requirement: Multilevel T-type RMS-error parity gate

The system SHALL provide
`core/tests/test_multilevel_ttype.cpp` gating ≤ 0.5 % RMS on
`V_phase_A`, `I_load_A` against the PLECS-exported T-type 3-level
golden.

#### Scenario: T-type RMS within tolerance

- **GIVEN** the shipped T-type YAML and golden
- **WHEN** the test runs
- **THEN** both observables pass

### Requirement: MMC RMS-error parity gate (PSIM golden)

The system SHALL provide
`core/tests/test_multilevel_mmc.cpp` gating ≤ 1 % RMS on arm
currents, circulating current, and per-submodule `V_cap` against
the PSIM-exported 9-submodule MMC golden.

The tolerance is looser than the NPC / flying-cap / T-type gates
because MMC controller implementations vary across simulators.

#### Scenario: MMC RMS within tolerance

- **GIVEN** the shipped 9-submodule MMC YAML and PSIM golden
- **WHEN** the test runs
- **THEN** all observables pass the ≤ 1 % RMS gate

### Requirement: Multilevel wall-clock parity contract

The system SHALL provide `tools/multilevel_bench_runner.py` that
runs the Pulsim simulation on each of the 4 multilevel benchmark
circuits and compares the wall-clock time against the
corresponding PLECS / PSIM-recorded baseline.

The contract SHALL be: Pulsim's wall-clock time on each circuit
SHALL be within 2× of the slower of the two competitors on the
same hardware reference machine.

#### Scenario: Pulsim within 2× of slower competitor

- **GIVEN** the 4 multilevel benchmark circuits with PLECS / PSIM
  baseline times recorded for each
- **WHEN** `multilevel_bench_runner.py` runs
- **THEN** Pulsim's wall-clock factor on each circuit is ≤ 2.0
- **AND** the report logs the per-circuit factor for review
