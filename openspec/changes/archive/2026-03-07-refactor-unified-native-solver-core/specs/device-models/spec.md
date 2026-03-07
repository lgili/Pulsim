## ADDED Requirements
### Requirement: Loss Hooks for Segment and Event Commits
Device models SHALL expose deterministic loss contribution hooks for event transitions and accepted continuous segments.

#### Scenario: Switching-event loss contribution
- **WHEN** a switching-capable device commits an on/off transition event
- **THEN** the model returns switching-loss contribution terms for that event
- **AND** the runtime records the contribution exactly once per committed event

#### Scenario: Continuous-segment conduction loss contribution
- **WHEN** an accepted segment advances device currents/voltages
- **THEN** the model returns conduction-loss contribution terms for that segment
- **AND** rejected segment attempts do not contribute persistent loss energy

### Requirement: Temperature-Dependent Parameter Evaluation
Device models SHALL support deterministic temperature-dependent parameter evaluation for electrothermal coupling.

#### Scenario: Thermal state updates electrical parameters
- **WHEN** thermal coupling updates a device temperature state
- **THEN** the model applies bounded temperature scaling to configured parameters
- **AND** exposes the updated parameter state for the next accepted electrical segment

#### Scenario: Disabled thermal coupling
- **WHEN** thermal coupling is disabled for a device
- **THEN** temperature-dependent scaling is not applied
- **AND** the model remains numerically consistent with base electrical parameters
