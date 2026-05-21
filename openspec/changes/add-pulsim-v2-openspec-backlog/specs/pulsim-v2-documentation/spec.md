## ADDED Requirements

### Requirement: OpenSpec Coverage for All Shipped V5-V15 Features

For every Layer-2 v2 feature that has been shipped to `main` between V5 and V15, the `openspec/specs/` directory SHALL contain a capability spec file with at least one requirement and one scenario reflecting the existing test-validated behaviour.

#### Scenario: All V5-V15 capabilities have spec files
- **WHEN** the catch-up change is archived
- **THEN** `openspec/specs/` SHALL contain at least 6 new spec files covering: source-helpers (V5-V10), sine source (V11), pulse source (V12), MOSFET-Level1 (V13), IGBT-Level1 (V14), VCVS/OpAmp (V15)
- **AND** every requirement SHALL have at least one `#### Scenario:` derived from an existing C++ or Python test

#### Scenario: openspec validate passes
- **WHEN** `openspec validate --strict` is run after the catch-up change archives
- **THEN** all spec files SHALL pass without errors
