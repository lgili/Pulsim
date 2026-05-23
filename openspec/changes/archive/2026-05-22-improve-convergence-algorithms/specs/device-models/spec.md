# Device Models Convergence Improvements

## ADDED Requirements

### Requirement: Diode Voltage Limiting

The system SHALL implement voltage limiting for diode models to prevent Newton-Raphson divergence.

Diode voltage limiting SHALL:
- Limit voltage changes per Newton iteration to ~4 thermal voltages
- Apply logarithmic limiting for forward-biased diodes
- Prevent exp() overflow for large forward voltages
- Preserve convergence accuracy within tolerance

#### Scenario: Large forward voltage swing limiting

- **GIVEN** a diode with previous voltage V_old = 0.5V
- **WHEN** Newton suggests V_new = 2.0V (delta = 1.5V)
- **THEN** the voltage is limited to approximately V_old + 4*Vt
- **AND** the limited voltage is around 0.6V
- **AND** Newton continues with limited voltage

#### Scenario: Small voltage change not limited

- **GIVEN** a diode with previous voltage V_old = 0.65V
- **WHEN** Newton suggests V_new = 0.67V (delta = 0.02V)
- **THEN** no limiting is applied
- **AND** the full voltage change is used

#### Scenario: Reverse bias limiting

- **GIVEN** a diode with previous voltage V_old = -5V
- **WHEN** Newton suggests V_new = -50V
- **THEN** voltage limiting is applied
- **AND** the rate of change is controlled to prevent large jumps

#### Scenario: Critical voltage calculation

- **GIVEN** a diode with Is = 1e-14A and Vt = 26mV
- **WHEN** the critical voltage is computed
- **THEN** V_critical = Vt * ln(Vt / (sqrt(2) * Is))
- **AND** V_critical is approximately 0.7V

### Requirement: MOSFET Voltage Limiting

The system SHALL implement voltage limiting for MOSFET models to prevent Newton-Raphson divergence.

MOSFET voltage limiting SHALL:
- Limit Vgs changes to maximum 0.5V per iteration
- Limit Vds changes to maximum 2.0V per iteration
- Apply limiting independently to each voltage
- Handle both enhancement and depletion modes

#### Scenario: Vgs limiting during turn-on

- **GIVEN** a MOSFET with Vgs_old = 2V
- **WHEN** Newton suggests Vgs_new = 10V
- **THEN** Vgs is limited to Vgs_old + 0.5V = 2.5V
- **AND** multiple iterations gradually reach final Vgs

#### Scenario: Vds limiting during switching

- **GIVEN** a MOSFET with Vds_old = 30V
- **WHEN** Newton suggests Vds_new = 0V (device turning on)
- **THEN** Vds is limited to Vds_old - 2V = 28V
- **AND** gradual reduction prevents overshoot

#### Scenario: No limiting for small changes

- **GIVEN** a MOSFET with Vgs_old = 5V and Vds_old = 10V
- **WHEN** Newton suggests Vgs_new = 5.1V and Vds_new = 9.5V
- **THEN** no limiting is applied to either voltage
- **AND** Newton uses exact suggested values

#### Scenario: Region transition smoothing

- **GIVEN** a MOSFET transitioning from saturation to linear
- **WHEN** voltage limiting is applied during transition
- **THEN** the device smoothly changes regions
- **AND** no discontinuities cause Newton failure

### Requirement: IGBT Voltage Limiting

The system SHALL implement voltage limiting for IGBT models similar to MOSFET limiting.

#### Scenario: IGBT Vge limiting

- **GIVEN** an IGBT with Vge_old = 10V
- **WHEN** Newton suggests Vge_new = 0V (turn-off)
- **THEN** Vge is limited to Vge_old - 0.5V = 9.5V
- **AND** gradual turn-off prevents convergence issues

#### Scenario: IGBT Vce limiting

- **GIVEN** an IGBT with Vce_old = 400V
- **WHEN** Newton suggests Vce_new = 2V (saturation)
- **THEN** Vce is limited to reduce by max 2V per iteration
- **AND** multiple iterations reach final Vce

### Requirement: BJT Voltage Limiting

The system SHALL implement voltage limiting for BJT models.

#### Scenario: BJT Vbe limiting

- **GIVEN** a BJT with Vbe_old = 0.6V
- **WHEN** Newton suggests Vbe_new = 1.5V
- **THEN** Vbe is limited similar to diode limiting
- **AND** exponential function does not overflow

#### Scenario: BJT Vbc limiting

- **GIVEN** a BJT with Vbc reverse biased
- **WHEN** Newton suggests large Vbc change
- **THEN** limiting prevents numerical instability

### Requirement: Voltage Limiter Configuration

The system SHALL provide a `VoltageLimiter` utility with configurable parameters.

VoltageLimiter SHALL provide:
- Static methods for each device type (diode, MOSFET, IGBT, BJT)
- Configurable maximum change per iteration
- Device-specific critical voltage calculations

#### Scenario: Custom MOSFET Vgs limit

- **GIVEN** user sets max_vgs_change = 1.0V
- **WHEN** MOSFET voltage limiting is applied
- **THEN** Vgs changes up to 1.0V per iteration are allowed
- **AND** default 0.5V limit is overridden

#### Scenario: Disabled voltage limiting

- **GIVEN** voltage limiting is disabled in solver options
- **WHEN** Newton iteration computes new voltages
- **THEN** no limiting is applied
- **AND** Newton uses full voltage changes (may diverge)

### Requirement: Previous Voltage State Tracking

The system SHALL track previous voltages for all nonlinear devices to enable voltage limiting.

#### Scenario: Initial voltage state

- **GIVEN** a circuit at the start of DC analysis
- **WHEN** voltage limiting is applied
- **THEN** initial voltage guesses (0V or user-specified) are used as V_old
- **AND** first iteration uses limiting relative to initial guess

#### Scenario: Voltage state update

- **GIVEN** Newton iteration n has converged
- **WHEN** iteration n+1 begins
- **THEN** V_old is updated to converged voltage from iteration n
- **AND** limiting is relative to the new V_old

#### Scenario: Transient voltage tracking

- **GIVEN** a transient simulation at timestep t_n
- **WHEN** Newton iterations occur for timestep t_{n+1}
- **THEN** V_old starts as the solution from timestep t_n
- **AND** V_old is updated within Newton iterations

## ADDED Requirements

### Requirement: Diode Stamp with Limiting

The diode stamp function SHALL apply voltage limiting before computing current and conductance.

The stamp SHALL:
- Retrieve previous diode voltage from device state
- Apply voltage limiting to new voltage
- Compute I and G using limited voltage
- Store new voltage in device state

#### Scenario: Diode stamp with limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_diode() is called with V_new from Newton
- **THEN** V_limited = limit_diode_voltage(V_new, V_old)
- **AND** I and G are computed using V_limited
- **AND** V_old is updated to V_limited

### Requirement: MOSFET Stamp with Limiting

The MOSFET stamp function SHALL apply voltage limiting before computing currents.

#### Scenario: MOSFET stamp with Vgs and Vds limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_mosfet() is called
- **THEN** Vgs_limited = limit_mosfet_vgs(Vgs_new, Vgs_old)
- **AND** Vds_limited = limit_mosfet_vds(Vds_new, Vds_old)
- **AND** drain current is computed using limited voltages
