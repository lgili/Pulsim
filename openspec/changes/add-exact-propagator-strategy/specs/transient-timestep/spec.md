## ADDED Requirements

### Requirement: Exact Propagator Integrator Strategy

The transient stepper SHALL offer the exact propagator as an integrator
strategy beside the implicit ones, with an event-driven controller and
a fixed-step controller. `engine='auto'` SHALL select it when the
applicability rule of the `exact-propagator` capability holds, and the
result SHALL report which integrator advanced the run and why. The
exact strategy's fallback SHALL be an implicit L-stable integrator; an
explicit integrator MUST NOT be used as a silent fallback.

#### Scenario: Automatic selection on a PWL converter
- **GIVEN** a synchronous buck with DC sources and a C++ PWM schedule
- **WHEN** it is run with `engine='auto'` and no `dt`
- **THEN** `result.integrator_used` is the exact propagator and the run
  takes two steps per PWM period

#### Scenario: A requested fixed step stays trapezoidal
- **GIVEN** the same circuit run with `engine='pwl'` and `dt=1e-7`
- **WHEN** the run completes
- **THEN** the integrator is the trapezoidal one and the fixed engine's
  golden hashes are unchanged

#### Scenario: Fallback is implicit and reported
- **GIVEN** a circuit with a user `b_extra_fn`
- **WHEN** the exact strategy is requested
- **THEN** the steps are taken by the implicit integrator and the
  result's route reason says so
