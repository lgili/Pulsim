## ADDED Requirements

### Requirement: Device Interface
All devices SHALL implement a common interface for simulation.

#### Scenario: Device stamp contribution
- **WHEN** MNA matrix is being assembled
- **THEN** each device provides its stamp (conductance, current contributions)
- **AND** the interface is: `stamp(matrix, rhs, state)`

#### Scenario: Device evaluation
- **WHEN** device currents/voltages are needed
- **THEN** the device evaluates its equations given node voltages
- **AND** returns currents and partial derivatives (Jacobian entries)

#### Scenario: Device state
- **WHEN** a device has internal state (e.g., capacitor charge)
- **THEN** state is stored in the solution vector
- **AND** is available for output

### Requirement: Basic Passive Components
The kernel SHALL support fundamental passive components.

#### Scenario: Resistor
- **WHEN** a resistor R is defined
- **THEN** it enforces I = V/R (linear)
- **AND** supports temperature coefficient: R(T) = R0*(1 + tc1*(T-Tnom) + tc2*(T-Tnom)^2)

#### Scenario: Capacitor
- **WHEN** a capacitor C is defined
- **THEN** it enforces I = C*dV/dt
- **AND** supports voltage-dependent capacitance: C(V) = C0*(1 + vc1*V + vc2*V^2)

#### Scenario: Inductor
- **WHEN** an inductor L is defined
- **THEN** it enforces V = L*dI/dt
- **AND** supports current-dependent inductance: L(I) = L0*(1 + lc1*I + lc2*I^2)

#### Scenario: Mutual inductance
- **WHEN** two inductors are coupled with coefficient k
- **THEN** M = k*sqrt(L1*L2) links their fluxes
- **AND** both inductors must be defined in the netlist

### Requirement: Voltage and Current Sources
The kernel SHALL support independent and dependent sources.

#### Scenario: Independent voltage source
- **WHEN** a voltage source V is defined
- **THEN** it enforces v+ - v- = V(t)
- **AND** waveform is defined by type (dc, pulse, sin, pwl)

#### Scenario: Independent current source
- **WHEN** a current source I is defined
- **THEN** it injects I(t) from - terminal to + terminal
- **AND** waveform follows same types as voltage source

#### Scenario: VCVS (E source)
- **WHEN** a voltage-controlled voltage source is defined
- **THEN** Vout = gain * Vcontrol
- **AND** control nodes are specified separately

#### Scenario: VCCS (G source)
- **WHEN** a voltage-controlled current source is defined
- **THEN** Iout = transconductance * Vcontrol
- **AND** control nodes are specified separately

#### Scenario: CCVS (H source)
- **WHEN** a current-controlled voltage source is defined
- **THEN** Vout = transresistance * Icontrol
- **AND** control current is through a specified voltage source

#### Scenario: CCCS (F source)
- **WHEN** a current-controlled current source is defined
- **THEN** Iout = gain * Icontrol
- **AND** control current is through a specified voltage source

### Requirement: Ideal Switch
The kernel SHALL support ideal switches for power electronics.

#### Scenario: Voltage-controlled switch
- **WHEN** a switch is defined with control voltage nodes
- **THEN** switch is closed when Vcontrol > Vthreshold
- **AND** closed = low resistance (Ron), open = high resistance (Roff)

#### Scenario: Current-controlled switch
- **WHEN** a switch is controlled by a current
- **THEN** switch is closed when |Icontrol| > Ithreshold
- **AND** control current is through a specified element

#### Scenario: Timed switch
- **WHEN** a switch has time-based control
- **THEN** state changes at specified times
- **AND** switching times can be periodic (PWM pattern)

### Requirement: Diode Model
The kernel SHALL support diode models from ideal to detailed.

#### Scenario: Ideal diode
- **WHEN** an ideal diode is defined
- **THEN** it conducts when forward biased (V > 0) with Ron
- **AND** blocks when reverse biased with Roff

#### Scenario: Shockley diode
- **WHEN** a Shockley diode model is used
- **THEN** I = Is*(exp(V/(n*Vt)) - 1)
- **AND** parameters: Is (saturation current), n (ideality factor), Vt = kT/q

#### Scenario: Diode with junction capacitance
- **WHEN** junction capacitance is modeled
- **THEN** Cj = Cj0 / (1 - V/Vj)^M for V < Fc*Vj
- **AND** linear extrapolation for higher voltages

#### Scenario: Diode with recovery time
- **WHEN** reverse recovery is modeled
- **THEN** stored charge Qrr affects turn-off behavior
- **AND** recovery current is modeled during reverse bias transition

### Requirement: MOSFET Model
The kernel SHALL support MOSFET models for power electronics.

#### Scenario: Ideal MOSFET switch
- **WHEN** ideal MOSFET model is used
- **THEN** it acts as a switch controlled by Vgs > Vth
- **AND** on-state is Ron, off-state is Roff

#### Scenario: Level 1 MOSFET
- **WHEN** Level 1 (Shichman-Hodges) model is used
- **THEN** square-law I-V characteristics apply
- **AND** parameters: Vth, Kp, lambda, W, L

#### Scenario: MOSFET with body diode
- **WHEN** body diode is included
- **THEN** an anti-parallel diode is added
- **AND** diode parameters can be specified

#### Scenario: MOSFET with capacitances
- **WHEN** MOSFET capacitances are modeled
- **THEN** Cgs, Cgd, Cds are included
- **AND** Miller effect is captured

### Requirement: IGBT Model
The kernel SHALL support IGBT models for high-power applications.

#### Scenario: Simplified IGBT
- **WHEN** simplified IGBT model is used
- **THEN** it combines MOSFET input with BJT output
- **AND** on-state is Vce,sat plus Ron*Ic

#### Scenario: IGBT with tail current
- **WHEN** detailed IGBT model is used
- **THEN** turn-off tail current is modeled
- **AND** affects switching loss calculation

### Requirement: Transformer Model
The kernel SHALL support transformer models.

#### Scenario: Ideal transformer
- **WHEN** an ideal transformer with ratio n is defined
- **THEN** V2 = V1/n and I1 = I2/n
- **AND** power is conserved (lossless)

#### Scenario: Transformer with magnetizing inductance
- **WHEN** a real transformer is modeled
- **THEN** a magnetizing inductance Lm is in parallel with primary
- **AND** leakage inductances Ll1, Ll2 are in series

#### Scenario: Multi-winding transformer
- **WHEN** multiple secondaries are defined
- **THEN** each has its own turns ratio
- **AND** coupling is through shared magnetic core

### Requirement: Device Parameter Library
The kernel SHALL support a library of device parameters.

#### Scenario: Built-in models
- **WHEN** a standard model name is referenced
- **THEN** parameters are loaded from built-in library
- **AND** library includes common power devices

#### Scenario: User-defined models
- **WHEN** a model is defined in the netlist
- **THEN** user parameters override defaults
- **AND** incomplete parameters use defaults

#### Scenario: Datasheet import
- **WHEN** datasheet parameters are provided
- **THEN** model parameters are extracted
- **AND** curves are fitted to the model equations
