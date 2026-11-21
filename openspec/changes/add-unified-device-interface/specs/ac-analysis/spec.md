## ADDED Requirements

### Requirement: Small-Signal System Derived From Device Residuals

The kernel SHALL build the small-signal system at an operating point
`x0` and switch mask `m` as
`(G_lin(m) + Σ ∂f/∂x|x0) + jω·(C_lin + Σ ∂q/∂x|x0)`, where the sums run
over every Newton device and `G_lin`, `C_lin` are the matrices the
transient solver uses (mutual inductance included). `run_mna_sweep`
SHALL use this system, SHALL accept a caller-supplied mask, and SHALL
NOT refuse or silently drop any device kind that the transient solver
supports. No kind SHALL need AC-specific code.

#### Scenario: Coupled transformer
- **GIVEN** a two-winding transformer with coupling `k = 0.99` driving
  a resistive load (the circuit of the audit probe `p3_ac.py`)
- **WHEN** the kernel sweep runs at 100 Hz, 1 kHz and 10 kHz
- **THEN** `|H|` is 0.062, 0.389 and 0.495 within 1e-3 relative of the
  analytic value

#### Scenario: Switch state is the caller's
- **GIVEN** an RC behind a switch
- **WHEN** the sweep runs with the switch bit set
- **THEN** `|H|` is 0.998, 0.847 and 0.157 at 100 Hz, 1 kHz and 10 kHz
- **AND** with the bit clear the response is that of the open circuit

#### Scenario: Newton devices are linearised at the operating point
- **WHEN** the sweep runs on a Shockley diode biased at 1 A, on a
  saturable inductor and on a nonlinear capacitor (audit cases)
- **THEN** `|H|` is 0.0252 for the diode, 0.500 / 0.477 / 0.152 for the
  inductor and 0.998 / 0.847 / 0.157 for the capacitor, each within
  1e-3 relative of the analytic value

#### Scenario: A new kind appears in AC without AC code
- **GIVEN** the test-only Newton kind of the `device-interface`
  capability (a nonlinear RC)
- **WHEN** it is swept at its operating point
- **THEN** the response matches its analytic small-signal impedance
  within 1e-6 relative

### Requirement: DC Operating Point From The Static Residual

The DC operating point SHALL stamp every Newton device with its static
residual `f` alone (the time-derivative coefficient set to zero), so
that a device with only `q` on a row is open, a device whose branch row
is a flux derivative is a short, and every kind that runs in transient
has a DC stamp. The DC continuation cascade and its absolute `dx` rule
SHALL be unchanged.

#### Scenario: Operating points unchanged
- **WHEN** `start_from_dc_op` runs on the harness circuits
- **THEN** the operating points equal the previous build within 1e-9
  relative

#### Scenario: MNA PMSM has an operating point
- **GIVEN** an MNA PMSM at standstill fed by DC phase voltages
- **WHEN** the DC operating point is requested
- **THEN** it converges to `i = v/R_s` per phase instead of failing for
  a missing DC stamp
