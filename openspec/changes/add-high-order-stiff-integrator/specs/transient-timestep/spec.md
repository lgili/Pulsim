## ADDED Requirements

### Requirement: Fourth-Order ESDIRK Integrator Strategy

The transient stepper SHALL offer a fourth-order, L-stable, stiffly
accurate ESDIRK integrator with an embedded third-order error estimate
(Kennedy–Carpenter ESDIRK4(3)6L[2]SA, diagonal coefficient 1/4) as an
integrator strategy selected by `integrator='esdirk43'`. Its
coefficients SHALL satisfy the order conditions up to order four, the
embedded weights up to order three, and stage order two, exactly.

#### Scenario: Tableau properties
- **WHEN** the tableau is checked in rational arithmetic
- **THEN** the row sums equal `c = [0, 1/2, 83/250, 31/50, 17/20, 1]`,
  the eight order conditions of the main weights are exactly zero, the
  embedded weights satisfy the conditions of orders one to three, and
  `A·c = c²/2` holds for stages two to six

#### Scenario: Order four on an oscillator
- **GIVEN** an ideal LC tank (100 µH, 100 nF, 10 V) advanced by the
  integrator with step control disabled (kernel-level test)
- **WHEN** it is run for 10 periods at `T₀/20`, `T₀/40` and `T₀/80`
- **THEN** the amplitude error is 5.16e-4, 3.24e-5 and 2.02e-6 within
  5 %, an observed order of at least 3.9

#### Scenario: L-stability
- **GIVEN** an RC circuit charged in steps of one hundred time
  constants, step control disabled (kernel-level test)
- **WHEN** the first step completes
- **THEN** the remaining error is 7.57 % of the step with no overshoot,
  and below 0.05 % after three steps

### Requirement: Single Factorization per ESDIRK Step

Every implicit stage of the ESDIRK strategy SHALL solve with the
trapezoidal companion matrix at `dt = h/2`, obtained from the existing
per-(mask, step) factor cache, so that a step needs at most one
factorization whatever the number of stages, and no device SHALL carry
a formula specific to this integrator.

#### Scenario: Factor count on a linear circuit
- **GIVEN** the LC tank on the ESDIRK strategy with a variable step
- **WHEN** the solver counters are read
- **THEN** factorizations per accepted step are at most 1 and linear
  solves per accepted step are 6 (five stages and the error filter)

#### Scenario: Stateful device without a method-specific formula
- **GIVEN** a circuit with a nonlinear capacitor
- **WHEN** it is run on the ESDIRK strategy and on TR-BDF2 at
  `rtol = 1e-9`
- **THEN** the two agree within 1e-5 and the device model was not
  changed to support the new integrator

### Requirement: ESDIRK Error Control

The ESDIRK strategy SHALL control its step with the embedded error
estimate, filtered by one solve with the stage matrix so that stiff
components do not appear as error, measured on capacitor voltages and
inductor currents with the engine's `rtol` and `atol`.

#### Scenario: Resonant tank over 1000 periods
- **GIVEN** the LC tank at `rtol = 1e-6`
- **WHEN** it is run for 1000 periods
- **THEN** it takes at most 32 steps per period, loses at most 0.035 %
  of its energy and stays within 0.25 V of the exact waveform
  (TR-BDF2 at the default tolerance takes 121 steps per period and
  loses 0.99 %)

#### Scenario: Fewer solves than TR-BDF2 at equal tolerance
- **GIVEN** the LC tank on both integrators at `rtol` 1e-5, 1e-6 and
  1e-8
- **WHEN** linear solves per period are counted
- **THEN** the ESDIRK strategy uses at most one quarter of TR-BDF2's

#### Scenario: Stiff mode is not read as error
- **GIVEN** a circuit with an open switch in series with an inductor
  (a mode near 1e-13 s)
- **WHEN** the state is at rest on that mode
- **THEN** the step grows to the maximum step instead of staying at the
  minimum

### Requirement: ESDIRK Events

The ESDIRK strategy SHALL recompute its explicit first stage from the
consistent post-event solution after every event, and SHALL land state
events by re-taking the step to the located instant, so that event
instants and post-event states have the accuracy of the TR-BDF2
strategy.

#### Scenario: Buck converters against the references
- **GIVEN** the CCM and DCM bucks of `numerics/p2_buck.py`
- **WHEN** they are run on the ESDIRK strategy at `rtol = 1e-8`
- **THEN** the states at 40 periods are within 1e-6 V and 1e-8 A of
  `(11.8015131102418 A, 12.2288664641139 V)` for CCM, and the DCM orbit
  is within 1e-6 V of 8.99624708728943 V after 4000 periods
- **AND** gate edges are landed within 1e-12 s

### Requirement: ESDIRK Selection Rule

The ESDIRK strategy SHALL be used only when requested by name in this
release, MUST NOT replace the exact propagator where that strategy
applies under `engine='auto'`, and SHALL be refused with a fixed-step
controller with an error naming the option.

#### Scenario: Auto is unchanged
- **GIVEN** any circuit run with `engine='auto'`
- **WHEN** this change is installed
- **THEN** the integrator used and the results are the same as before

#### Scenario: Fixed step refused
- **WHEN** `integrator='esdirk43'` is combined with `dt=`
- **THEN** an `UnsupportedError` names the integrator and the
  fixed-step controller
