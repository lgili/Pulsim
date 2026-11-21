## ADDED Requirements

### Requirement: Periodic Steady State by Newton Shooting on the True State

The library SHALL find the `T`-periodic steady state of a circuit by
Newton iteration on `G(x₀) = P(x₀) − x₀`, where `P` is the one-period
map on the true state (capacitor voltages, inductor currents, device
states and controller states) with every event located in time. The
solver MUST NOT assume that the commutation pattern at the starting
point is the pattern of the orbit.

#### Scenario: DCM buck from the zero state
- **GIVEN** the buck of `numerics/p2_buck.py` with R = 50 Ω, D = 0.3
- **WHEN** the periodic steady state is requested from the zero state
- **THEN** `v_C` at the rising gate edge is `8.99624708728943 V` and
  `i_L` is `6.00750582388986e-9 A`, within 1e-8
- **AND** the diode turn-off is at `0.799830698444468·T` within 1e-9·T
- **AND** at most 8 one-period runs were used

#### Scenario: CCM buck
- **GIVEN** the same buck with R = 1 Ω, D = 0.5
- **WHEN** the periodic steady state is requested
- **THEN** the state at the rising edge is
  `(11.6878790893 A, 11.9877289409 V)` within 1e-9

#### Scenario: Agreement with a long transient
- **GIVEN** either circuit above
- **WHEN** it is also simulated for 4000 periods from zero
- **THEN** the state at the last rising edge differs from the solver's
  orbit by less than 1e-9

### Requirement: Monodromy Matrix With Saltation

The solver SHALL compute the monodromy matrix as the ordered product of
the flow Jacobian of each segment and the saltation matrix of each
event, `S = R_x + (f⁺ − R_x·f⁻ − R_t)·∇gᵀ/(∇gᵀ·f⁻ + g_t)`, and the
matrix SHALL be the derivative of the period map as computed. It SHALL
report a grazing event (`∇gᵀ·f⁻ + g_t = 0`) instead of dividing by it.

#### Scenario: Peak-current-mode buck
- **GIVEN** a buck (24 V, 100 µH, 4700 µF, 2 Ω) whose switch is turned
  on by a 100 kHz clock and off when `i_L` reaches 7 A
- **WHEN** the periodic steady state is solved
- **THEN** the duty is 0.5590 and the Floquet multipliers are −1.267293
  and 0.998949 within 1e-5 (the analytic value is `−D/(1 − D)`)
- **AND** the plain product of flow Jacobians, 0.9993 ± 0.0146j, is not
  what is reported

#### Scenario: Continuous law
- **GIVEN** the DCM buck with the resistor-pair diode and `V_th = 0`
- **WHEN** the monodromy matrix is formed
- **THEN** its eigenvalues are 0 (within 1e-9) and 0.988977 (within
  1e-6), and it equals a central finite difference of the period map
  within 1e-7

#### Scenario: Quadratic convergence
- **GIVEN** the DCM buck solved from the zero state
- **WHEN** the residual history is inspected after the pattern has
  settled
- **THEN** each residual is at most 1e3 times the square of the
  previous one (prototype: 2.2e-4, 3.0e-7, 5.7e-13)

### Requirement: Steady-State Result

The solver SHALL return the orbit as a resumable snapshot with one
period of waveforms, the monodromy matrix with state labels, the
Floquet multipliers with a stability verdict, the commutation pattern
with its instants, the residual history and the number of one-period
runs. An orbit with a multiplier outside the unit circle SHALL be
returned with `stable = False` and the multiplier named; it MUST NOT be
reported as stable or silently replaced.

#### Scenario: Stability verdict
- **GIVEN** the peak-current-mode buck at D = 0.5590
- **WHEN** the result is read
- **THEN** `stable` is False and the multiplier −1.267 is reported as a
  flip (period-doubling) multiplier

#### Scenario: Resume from the orbit
- **GIVEN** the DCM result's snapshot
- **WHEN** one more period is simulated from it
- **THEN** the state returns to the snapshot within 1e-9

### Requirement: Sampled-Data Small-Signal Model

The result SHALL provide the sampled-data model `x[k+1] = M·x[k] +
B·u[k]`, `y[k] = C·x[k] + D·u[k]` for chosen inputs and outputs, built
from the same Jacobians as the monodromy matrix, and the frequency
response `H(z) = C·(zI − M)⁻¹·B + D` SHALL be available in CCM and in
DCM.

#### Scenario: DC gain in DCM
- **GIVEN** the DCM buck with the duty as input and `v_C` as output
- **WHEN** `H` is evaluated at `z = 1`
- **THEN** it equals `∂v_C*/∂D = 23.0895697 V` within 1e-6 relative

#### Scenario: CCM control-to-output
- **GIVEN** a 100 kHz CCM buck
- **WHEN** the duty-to-output response is computed below 5 kHz
- **THEN** it is within 1 % of `V_in/(1 + sL/R + s²LC)` (0.77 % from
  `frequency_response` before this change)

### Requirement: Steady-State Refusals

The solver SHALL refuse, with an error that names the cause: a source
or gate schedule whose period does not divide the requested period; a
device or controller whose state it cannot read and set; a monodromy
matrix with a multiplier equal to 1 (naming the states of the marginal
direction). When the iteration budget is exhausted it SHALL raise with
the residual history and the last commutation pattern.

#### Scenario: Incommensurate source
- **GIVEN** a 100 kHz converter fed by a 50.3 Hz sine source
- **WHEN** the steady state is requested with `period = 1/100e3`
- **THEN** the error names the source and its frequency

#### Scenario: Marginal direction
- **GIVEN** an ideal LC loop with no loss
- **WHEN** the steady state is requested
- **THEN** the error states that the orbit is not isolated and names the
  capacitor and the inductor

### Requirement: Single Steady-State Entry Point

Every public steady-state routine SHALL obtain the orbit from the one
solver (`steady_state`, `run_periodic_shooting`, `run_harmonic_balance`
and `frequency_response`), on the engine the caller selected, and on
each engine the orbit SHALL equal that engine's own long transient.

#### Scenario: Legacy shooting on the DCM buck
- **GIVEN** `run_periodic_shooting` on the DCM buck with `dt = T/100`
- **WHEN** it converges
- **THEN** `v_C` is within 1e-4 of 8.996247 V (8.8998 V before this
  change)

#### Scenario: Engine self-consistency
- **GIVEN** the DCM buck solved on the fixed engine at `dt = T/100`
- **WHEN** the same engine simulates 4000 periods from zero
- **THEN** the two states at the rising edge differ by less than 1e-8
