## ADDED Requirements

### Requirement: TR-BDF2 Local Error Estimate Scale

The variable-step engine SHALL estimate the local error of a TR-BDF2
step with the factor of Bank et al. (1985),
`2·C·h·(f_n/γ − f_γ/(γ(1−γ)) + f_1/(1−γ))`, so that the estimate equals
the local error to leading order and `rtol` / `atol` bound the error
the user named.

#### Scenario: Estimate against one exact step
- **GIVEN** a linear RC and a step with `h/τ` = 0.05
- **WHEN** the estimate is compared with the difference to the exact
  solution over that step
- **THEN** their ratio is between 0.9 and 1.1 (it is 1.97 without the
  factor)

#### Scenario: Accuracy at a given tolerance does not get worse
- **GIVEN** a buck against its exact piecewise-LTI solution at rtol
  1e-4, 1e-6 and 1e-8
- **WHEN** it runs with the new estimate
- **THEN** the output error is not above 8e-4, 9e-5 and 2e-6 V
- **AND** the number of accepted steps is at most 35 % above the
  previous one

### Requirement: TR-BDF2 Stiff Error Filter

The variable-step engine SHALL filter the local error estimate once
through the inverse of the stage iteration matrix,
`(I − (γ/2)·h·J)⁻¹` (Hosea & Shampine 1996), using the factor the two
stages of the step already used, so that the estimate of a stiff
component stays bounded by a fixed multiple of that component's
amplitude at any step size. On circuits solved by Newton the filter
SHALL be applied when a factor of the converged Jacobian is available
and the statistics SHALL say when it was not.

#### Scenario: A fast mode at rest does not limit the step
- **GIVEN** an RC with τ = 1e-12 s whose state is at equilibrium to
  1e-9, beside an RC with τ = 1 ms, at rtol = 1e-5
- **WHEN** a step of 1 µs is attempted
- **THEN** the filtered estimate accepts it (the unfiltered estimate
  of the fast state is more than 1e5 times its distance from
  equilibrium and rejects it)

#### Scenario: Non-stiff estimate is unchanged
- **GIVEN** a step with `h·λ` = −0.05
- **WHEN** the estimate is filtered
- **THEN** it changes by less than 2 %

### Requirement: TR-BDF2 Step Bounds Independent Of The Run Length

The variable-step engine SHALL derive its step floor, probe floor,
probe retry cap, chatter floor and gate-pulse floor from the resolved
step
ceiling and from the time resolution
`16·ε·max(|t_start|, |t_end|)`, not from `t_end − t_start`. Its default step ceiling SHALL be the
smallest cap the circuit declares (periodic sources, pulse widths, the
edges of a schedule, a learned gate pulse) and SHALL fall back to
`(t_end − t_start)/1000` only when the circuit declares none.
`TrBdf2Stats` SHALL report the basis of the ceiling.

#### Scenario: Same answer at 40 periods whatever the run length
- **GIVEN** a buck driven by a schedule with an edge oracle
- **WHEN** it runs with `t_end` = 40 T, 400 T and 4000 T
- **THEN** the accepted steps up to 40 T are the same in the three
  runs and the state at 40 T agrees to 1e-12 relative (the output
  error is 8.2e-4 V and 1.04e-3 V in the first two without this
  change)

#### Scenario: Span-bound runs keep their constants
- **GIVEN** a circuit with no periodic source and no gates
- **WHEN** it runs
- **THEN** the ceiling basis is reported as the span and every floor
  equals its previous value

### Requirement: TR-BDF2 Source Breakpoint Landing

The variable-step engine SHALL treat the breakpoints of its built-in
sources (the start, end of rise, start of fall and end of fall of a
pulse source in every period, and the edges of a PWM voltage source)
as time events: a step SHALL be clamped to end on the next breakpoint,
and a step that ends on a jump SHALL evaluate the jumping source at
its left limit.

#### Scenario: Pulse into an RC
- **GIVEN** a 0 → 10 V pulse source (edge at 3.3 µs, zero rise time)
  into 1 kΩ and 1 nF at the default tolerances
- **WHEN** it runs to 100 µs
- **THEN** a sample lies on each jump and was reached by a clamped
  step, and no accepted step in the 1 µs before a jump is shorter than
  1e-9 s (steps of 1.2e-10 and 6.3e-11 s occur without this change)
- **AND** the largest error at rtol = 1e-6 is not above 1.9e-4 V

#### Scenario: Ramp corners
- **GIVEN** a pulse source with a 100 ns rise time
- **WHEN** it runs
- **THEN** the start and the end of the rise are both sample times
