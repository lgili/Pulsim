## ADDED Requirements
### Requirement: Closed-Loop Regulation Scenarios
The benchmark suite SHALL include complex closed-loop topologies that stress the mixed-domain solving capabilities, specifically the interactions between non-linear power stages and discrete/continuous control blocks (PI controllers, PWM generators).

#### Scenario: Buck Converter Closed-Loop
- **WHEN** simulating `ll14_buck_closed_loop.yaml`
- **THEN** the solver reliably resolves the timestep constraints imposed by the PWM generator and limits the PI controller windup, achieving steady-state regulation

#### Scenario: Boost Converter Closed-Loop
- **WHEN** simulating `ll15_boost_closed_loop.yaml`
- **THEN** the solver maintains numerical stability during the Right-Half-Plane (RHP) zero transient effects inherent to closed-loop boost topologies

#### Scenario: Flyback Converter Closed-Loop
- **WHEN** simulating `ll16_flyback_closed_loop.yaml`
- **THEN** the solver accurately resolves the discontinuous/continuous conduction modes across the isolation barrier while the control loop maintains the setpoint
