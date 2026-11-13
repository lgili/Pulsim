## ADDED Requirements

### Requirement: TR-BDF2 Local Error Estimate Scale

The variable-step engine SHALL estimate the local error of a TR-BDF2
step with the factor of Bank et al. (1985),
`2·C·h·(f_n/γ − f_γ/(γ(1−γ)) + f_1/(1−γ))`, so that the estimate equals
the local error to leading order and `rtol` / `atol` bound the error
the user named.

#### Scenario: Smooth RC step response
- **GIVEN** a 1 kΩ, 1 nF RC driven by a 10 V source, run to 5 µs with
  a step ceiling of 1 µs at rtol 1e-5
- **WHEN** it runs with the new estimate
- **THEN** it takes 66 accepted steps (58 without the factor) and its
  largest error against the exact response is below 6e-4 V (8.17e-4 V
  without it)

#### Scenario: Accuracy at a given tolerance does not get worse
- **GIVEN** a buck against its exact piecewise-LTI solution at rtol
  1e-4, 1e-6 and 1e-8
- **WHEN** it runs with the new estimate
- **THEN** the output error is not above 8e-4, 9e-5 and 2e-6 V
- **AND** the number of accepted steps is at most 35 % above the
  previous one
- **AND** running at doubled `rtol` and `atol` reproduces the previous
  step sequence of a circuit with a span-bound ceiling

### Requirement: TR-BDF2 Step Bounds Independent Of The Run Length

The variable-step engine SHALL derive its step floor, probe floor,
probe retry cap, chatter floor and gate-pulse floor from the resolved
step ceiling `h_max` (as the previous fractions of `1000·h_max`) and
from the time resolution `16·ε·max(|t_start|, |t_end|)`, not from
`t_end − t_start`; while the ceiling equals `(t_end − t_start)/1000`
they SHALL be bit-identical to the previous values. The default step
ceiling remains `min((t_end − t_start)/1000, period/20, pulse
width/3)`. `TrBdf2Stats::h_max_basis` SHALL report the basis of the
ceiling (0 span, 1 source, 2 user).

#### Scenario: Same answer at 40 periods whatever the run length
- **GIVEN** a buck with a step ceiling of T/20
- **WHEN** it runs with `t_end` = 40 T and 400 T
- **THEN** the sample times and states up to 39 T are identical in
  the two runs (they diverge at the 24th sample without this change)

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
