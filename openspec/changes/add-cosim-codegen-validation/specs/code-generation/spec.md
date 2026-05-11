## ADDED Requirements

### Requirement: Generated C Controller Round-Trip Parity
The code-generation pipeline SHALL produce a compilable C implementation of a virtual control block (initial scope: `pi_controller`) that, when called from a Python harness in lockstep with a Pulsim plant simulation, reproduces the all-in-simulator closed-loop output to within 1 %.

#### Scenario: PI buck controller code-gen matches in-sim trace
- **GIVEN** the YAML `cl_buck_pi` defines a PI controller with kp/ki/limits
- **WHEN** the codegen pipeline emits a C source, the host CC compiles it, and a Python driver invokes it in a step-by-step loop alongside `Simulator.step()`
- **THEN** the V(out) trace produced by the co-simulation differs from the all-in-simulator baseline by at most 1 % at every sample
- **AND** the test fails with a clear diff if the divergence exceeds the threshold

### Requirement: FMU Export Round-Trip Parity
The FMI-export pipeline SHALL produce a Functional Mock-up Unit for a virtual control block that, when consumed via FMPy (or equivalent) in a co-simulation loop, reproduces the all-in-simulator output to within 1 %.

#### Scenario: PI controller FMU matches in-sim trace
- **GIVEN** the PI controller from `cl_buck_pi` is exported as an FMU
- **WHEN** a Python driver uses FMPy to call the FMU in lockstep with `Simulator.step()`
- **THEN** the V(out) trace matches the all-in-simulator baseline within 1 %
