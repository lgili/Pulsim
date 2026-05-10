## ADDED Requirements

### Requirement: Switching Mode Per Device
Switching device models (`IdealDiode`, `IdealSwitch`, `VoltageControlledSwitch`, `MOSFET`, `IGBT`) SHALL expose a `SwitchingMode` selector with values `Ideal`, `Behavioral`, and `Auto`.

#### Scenario: Default Auto mode
- **GIVEN** a device created without explicit mode
- **WHEN** the simulation resolves
- **THEN** the device defaults to `SwitchingMode::Auto`
- **AND** mode resolution follows the simulation-level `switching_mode`

#### Scenario: Per-device override
- **GIVEN** a YAML netlist with `simulation.switching_mode: ideal` and one device with `switching_mode: behavioral`
- **WHEN** the netlist is parsed
- **THEN** the kernel resolves to the DAE path because not all devices support `Ideal`
- **AND** the diagnostic message indicates which device forced the fallback

### Requirement: PWL Two-State Contract
Devices supporting `SwitchingMode::Ideal` SHALL implement a two-state PWL contract: `current_state()`, `commit_state(bool)`, and `should_commute(state, voltage, current) const`.

#### Scenario: Diode commutation rules
- **GIVEN** an `IdealDiode` in the `on` state with `i < 0`
- **WHEN** `should_commute()` is queried
- **THEN** it returns `true`

- **GIVEN** an `IdealDiode` in the `off` state with `v > 0`
- **WHEN** `should_commute()` is queried
- **THEN** it returns `true`

#### Scenario: Voltage-controlled switch commutation
- **GIVEN** a `VoltageControlledSwitch` with control voltage crossing `v_threshold`
- **WHEN** `should_commute()` is queried
- **THEN** it returns `true`
- **AND** the commute direction matches the threshold-crossing sign

### Requirement: PWL Diode Without Smoothing
When `IdealDiode` is in `SwitchingMode::Ideal`, the device SHALL stamp a sharp piecewise-linear conductance with no `tanh` smoothing.

#### Scenario: On-state stamp
- **GIVEN** an `IdealDiode` in `on` state with parameters `g_on, g_off`
- **WHEN** the device stamps into the system matrix
- **THEN** the stamped conductance equals `g_on`
- **AND** no derivative-of-conductance term is added to the Jacobian

#### Scenario: Off-state stamp
- **GIVEN** an `IdealDiode` in `off` state
- **WHEN** the device stamps into the system matrix
- **THEN** the stamped conductance equals `g_off`
- **AND** no derivative-of-conductance term is added to the Jacobian

### Requirement: Event Hysteresis Parameter
PWL switching devices SHALL expose an optional `event_hysteresis` parameter to suppress chatter near zero-crossing events.

#### Scenario: Hysteresis below default
- **GIVEN** a diode with default `event_hysteresis = 1e-9 V`
- **WHEN** the voltage oscillates within `±1e-9 V` of the commute threshold
- **THEN** the device does not commute
- **AND** no event is recorded

#### Scenario: Hysteresis disabled
- **GIVEN** `event_hysteresis = 0`
- **WHEN** the voltage crosses zero
- **THEN** the device commutes immediately on first crossing detection

## MODIFIED Requirements

### Requirement: Diode Stamp with Limiting

The diode stamp function SHALL apply voltage limiting before computing current and conductance, except when `SwitchingMode::Ideal` is active (where the device behaves as a pure piecewise-linear switch with no Newton-iteration semantics).

The stamp SHALL:
- Retrieve previous diode voltage from device state
- Apply voltage limiting to new voltage
- Compute I and G using limited voltage
- Store new voltage in device state

#### Scenario: Diode stamp with limiting (Behavioral)
- **GIVEN** MNA assembly with voltage limiting enabled and `SwitchingMode::Behavioral`
- **WHEN** stamp_diode() is called with V_new from Newton
- **THEN** V_limited = limit_diode_voltage(V_new, V_old)
- **AND** I and G are computed using V_limited
- **AND** V_old is updated to V_limited

#### Scenario: Diode stamp in PWL mode
- **GIVEN** `SwitchingMode::Ideal` is active
- **WHEN** stamp_diode() is called
- **THEN** voltage limiting is bypassed
- **AND** the stamped conductance is exactly `g_on` or `g_off` per current state
