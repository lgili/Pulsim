## ADDED Requirements

### Requirement: Forward Parameter Sensitivities

The transient kernel SHALL, on request, propagate the sensitivities
`∂x/∂p` of the state to named parameters along the run by
differentiating its own discrete step, reusing the factorization of
each implicit stage (one solve per parameter per stage and no
additional factorization). A run without the request SHALL be
bit-identical to the run before this capability existed.

#### Scenario: RC step against the closed form
- **GIVEN** a 10 V step into 1 kΩ and 1 µF with
  `sensitivities=["R1.R", "C1.C"]`
- **WHEN** it is run to 1 ms with the exact strategy
- **THEN** `∂v/∂R = −3.678794e-3 V/Ω` and `∂v/∂C = −3.678794e6 V/F`
  within 1e-9 relative

#### Scenario: Newton device parameters
- **GIVEN** the Shockley rectifier of
  `probes/forward_sensitivity_mna.py` on a fixed trapezoidal step
- **WHEN** sensitivities to R, C, the load, `I_s` and `n·V_t` are
  requested
- **THEN** each sensitivity of the capacitor voltage equals a central
  finite difference of the same scheme within 1e-6 relative

#### Scenario: No extra factorization
- **GIVEN** a linear circuit run with and without eight sensitivities
- **WHEN** the solver counters are compared
- **THEN** the number of factorizations is equal, the states are
  bit-identical and the run reports `8 × stages × steps` sensitivity
  solves

### Requirement: Sensitivities Across Switching Events

The sensitivities SHALL be corrected at every event by the event-time
term, `S⁺ = R_x·S⁻ + R_p + (R_x·f⁻ + R_t − f⁺)·dτ/dp` with
`dτ/dp = −(∇gᵀ·S⁻ + g_p)/(∇gᵀ·f⁻ + g_t)` for a state event and the
modulator's analytic `dτ/dp` for a time event, so that a parameter
which only moves a switching instant has the correct sensitivity.

#### Scenario: Duty
- **GIVEN** the DCM buck of `numerics/p2_buck.py` (R = 50 Ω, D = 0.3)
  with `sensitivities=["pwm.duty"]`
- **WHEN** it is run for 40 periods from zero
- **THEN** `∂v_C/∂D = 43.093591 V` within 1e-6 relative (zero without
  the event term)

#### Scenario: Element parameters through a diode commutation
- **GIVEN** the same run with `sensitivities=["L.L", "RL.R"]`
- **WHEN** it completes
- **THEN** `∂v_C/∂L = 1758.1254 V/H` and `∂v_C/∂R = 2.5551883e-2 V/Ω`
  within 1e-6 relative on the exact strategy

#### Scenario: Field jump at the event
- **GIVEN** a rectifier whose diode has `V_th = V_f + 1e-3`
- **WHEN** sensitivities are requested on the exact strategy
- **THEN** they equal central finite differences within 1e-5 relative

### Requirement: Steady-State Gradients

The periodic-steady-state solver SHALL return, on request, the
gradient of the orbit with respect to named parameters as
`dx*/dp = (I − M)⁻¹·∂P/∂p`, using the monodromy matrix it already
computed.

#### Scenario: DCM buck orbit
- **GIVEN** the DCM buck solved with
  `sensitivities=["L.L", "RL.R", "pwm.duty"]`
- **WHEN** the gradients are read
- **THEN** `∂v_C*/∂L = −3.4575154e4 V/H`, `∂v_C*/∂R = 6.92255e-2 V/Ω`
  and `∂v_C*/∂D = 23.0895697 V` within 1e-6 relative
- **AND** each equals a central finite difference of the steady-state
  solve within 1e-6 relative

### Requirement: Gradients of Scalar Metrics

The metric layer SHALL return, with the value of a scalar metric
(final value, time-weighted average, RMS, energy, peak, ripple) over a
window or over the periodic orbit, its gradient with respect to the
requested parameters, and SHALL flag a peak or ripple gradient whose
extremum is not unique. A helper SHALL expose an objective as
`(value, gradient)` for a gradient-based optimiser.

#### Scenario: Average output voltage
- **GIVEN** the DCM buck with a window over its last ten periods
- **WHEN** the average of `v_out` is requested with its gradient
- **THEN** the gradient equals a central finite difference on the
  exact path within 1e-6 relative

#### Scenario: Optimiser interface
- **GIVEN** an LC output filter with the ripple as objective and L, C
  as parameters
- **WHEN** `scipy.optimize.minimize(…, jac=True)` is called on
  `pulsim.value_and_grad(objective, params)`
- **THEN** it converges, and the gradient at the starting point equals
  a central finite difference within 1e-6 relative

### Requirement: Sensitivity Refusals

A sensitivity request SHALL be refused, with an error naming the
parameter or the feature, when the result would not be a derivative:
events that are not located in time, a parameter that is an integer, a
topology choice or a table breakpoint, an opaque block or callback
between the parameter and the output, or a grazing event.

#### Scenario: Fixed engine without event location
- **GIVEN** the DCM buck on the fixed engine with no event splitting
- **WHEN** `sensitivities=["L.L"]` is requested
- **THEN** the run is refused naming the diode whose commutation is
  not located (a finite difference there is wrong by a factor of ten,
  with either sign)

#### Scenario: Unknown or non-differentiable parameter
- **WHEN** `sensitivities=["L.turns"]` names a field that does not
  exist or is an integer
- **THEN** the error names the component and the field
