## ADDED Requirements

### Requirement: Collapsed Public DC Strategy Enum

The system SHALL expose exactly two values in its public-facing
`DCStrategy` enumeration: `Auto` (default) and `Override`.

The internal 5-value implementation enum (`Direct`, `GminStepping`,
`SourceStepping`, `PseudoTransient`, `Homotopy`) SHALL remain
accessible under `opts.advanced.dc.strategy_override` for power
users who need to force a specific strategy.

#### Scenario: Default Auto orchestrates the full ladder

- **GIVEN** a user accepts the default `DCStrategy::Auto`
- **WHEN** Newton on the first attempt (Direct) fails to converge
- **THEN** the orchestrator falls back through SourceStepping →
  GminStepping → PseudoTransient → Homotopy until convergence or
  the full ladder is exhausted

#### Scenario: Override forces a single strategy

- **GIVEN** a user sets
  `opts.advanced.dc.strategy_override = DCStrategyImpl::PseudoTransient`
  with no fallback
- **WHEN** the DC analysis runs
- **THEN** only PseudoTransient is attempted
- **AND** failure of PseudoTransient surfaces as a fatal DC error
  without trying any other strategy

### Requirement: Homotopy Continuation as Last Resort

The system SHALL implement homotopy continuation as the fifth and
final strategy in the `DCStrategy::Auto` fallback ladder.

The homotopy SHALL step a parameter `λ` from `0` to `1` in
configurable increments. At `λ = 0`, all nonlinear devices (diodes,
MOSFETs, IGBTs, BJTs, behavioural switches) SHALL be replaced by
their linear off-state conductance (`g_off`), producing a fully
linear MNA system solvable in a single direct solve. At `λ = 1`,
the full nonlinear device models SHALL be active.

Each step in the ladder SHALL solve Newton-Raphson with the previous
λ's solution as warm-start. The default ladder length SHALL be 5
increments; `Preset::HighFidelity` SHALL use 10 increments.

#### Scenario: Cold-start 3-level NPC succeeds via homotopy

- **GIVEN** a 3-level NPC converter at cold start where Direct,
  SourceStepping, GminStepping, and PseudoTransient all fail
- **WHEN** the `DCStrategy::Auto` orchestrator reaches homotopy
- **THEN** the λ ladder runs from 0 to 1 in 5 steps
- **AND** each step solves Newton in ≤ 10 iterations using the prior
  λ's solution as warm-start
- **AND** the final DC operating point is established
- **AND** `homotopy_ladder_completed == true` is reported

#### Scenario: Simple linear circuit never invokes homotopy

- **GIVEN** a passive RC circuit
- **WHEN** the DC analysis runs with `DCStrategy::Auto`
- **THEN** Direct converges on iteration 1
- **AND** homotopy is never attempted
- **AND** `homotopy_steps == 0` is reported

#### Scenario: HighFidelity preset uses longer ladder

- **GIVEN** a user calls `from_preset(Preset::HighFidelity, ...)`
- **WHEN** the DC analysis ladder reaches homotopy
- **THEN** the ladder runs in 10 increments instead of 5

#### Scenario: Homotopy disabled via advanced override

- **GIVEN** a user sets `opts.advanced.dc.homotopy.enable = false`
- **WHEN** `DCStrategy::Auto` exhausts the first four strategies
- **THEN** homotopy is skipped
- **AND** the DC analysis fails with `DcConvergenceFailure` rather
  than attempting homotopy

## MODIFIED Requirements

### Requirement: Strategy Telemetry

The system SHALL report which DC strategy successfully produced the
operating point in `result.dc_result.strategy_used`, plus per-strategy
timing and iteration counts.

The `strategy_used` field SHALL accept any of the five internal
strategy values: `Direct`, `GminStepping`, `SourceStepping`,
`PseudoTransient`, `Homotopy`.

Additionally, the system SHALL report:
- `homotopy_steps` — number of λ increments executed (0 if homotopy
  not invoked)
- `homotopy_ladder_completed` — true if homotopy reached `λ = 1`,
  false if it gave up partway
- `total_dc_newton_iterations` — sum across all strategies attempted

#### Scenario: Telemetry reports the winning strategy

- **GIVEN** a DC analysis that succeeded via SourceStepping after
  Direct failed
- **WHEN** the user inspects `result.dc_result`
- **THEN** `strategy_used == DCStrategyImpl::SourceStepping`
- **AND** `total_dc_newton_iterations` reflects the sum of Direct's
  failed-iteration count plus SourceStepping's successful iterations

#### Scenario: Telemetry reports homotopy progress

- **GIVEN** a DC analysis that succeeded via homotopy after 5 λ steps
- **WHEN** the user inspects `result.dc_result`
- **THEN** `strategy_used == DCStrategyImpl::Homotopy`
- **AND** `homotopy_steps == 5`
- **AND** `homotopy_ladder_completed == true`
