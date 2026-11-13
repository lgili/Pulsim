## ADDED Requirements

### Requirement: Series Parasitics on Capacitors and Inductors

The capacitor SHALL accept an equivalent series resistance `esr` [Ω] and
inductance `esl` [H], and the inductor a series resistance `esr` [Ω] and
a parallel capacitance `c_par` [F], through the builder, Python and
YAML. The terminal impedance SHALL be that of the ideal series (and
parallel) network. With every such parameter zero the component SHALL
be bit-identical to the component without them. The loss `i²·esr` SHALL
be reported under the component. A negative or non-finite value SHALL be
refused naming the component and the parameter.

#### Scenario: Capacitor series resonance
- **GIVEN** a 100 µF capacitor with `esr = 10 mΩ`, `esl = 10 nH`
- **WHEN** its impedance is swept
- **THEN** the minimum is 10 mΩ at 159.15 kHz, each within 1e-9
  relative

#### Scenario: ESR step in the ripple
- **GIVEN** a ±1 A square current into a 100 µF capacitor with
  `esr = 50 mΩ`
- **WHEN** the current reverses
- **THEN** the terminal voltage steps by 100 mV within 1e-6 relative

#### Scenario: ESR loss is reported
- **WHEN** 1 A rms flows through a capacitor with `esr = 50 mΩ`
- **THEN** `device_loss_summary` lists the capacitor with 50.0 mW

#### Scenario: Zero parasitics are unchanged
- **WHEN** a circuit that sets no parasitic parameter is run
- **THEN** its node count and states are bit-identical to the same
  circuit before this requirement

### Requirement: Bias-Dependent Capacitance Table

The nonlinear capacitor SHALL accept an odd charge table `Q(v)` whose
derivative `C(v)` is even and strictly positive, for capacitors whose
value falls with the magnitude of the DC bias. It SHALL be stamped in
charge form with `C(v)` the exact derivative of the evaluated `Q(v)`. A
fitter SHALL build the table from a capacitance-against-bias curve and
return its residual.

#### Scenario: Known bias law
- **GIVEN** a table built at 64 knots from `C0/(1 + (v/V1)²)` with
  `C0 = 10 µF`, `V1 = 10 V`
- **WHEN** it is evaluated at 20 V
- **THEN** `Q = 110.71 µC` within 1e-4 relative and `C = 2.000 µF`
  within 1e-3 relative

#### Scenario: Ripple sees the biased capacitance
- **GIVEN** that capacitor biased at 20 V with a small ripple current
- **WHEN** the ripple voltage is measured
- **THEN** it corresponds to 2.000 µF within 1e-3, not to 10 µF

### Requirement: Winding AC-Resistance Ladder

The library SHALL fit a passive ladder of parallel resistor–inductor
cells in series with a DC resistance to a winding's `R_ac(f)` data, and
SHALL build an inductor whose resistance follows
`R_dc + Σ R_k·(ωτ_k)²/(1 + (ωτ_k)²)`. Every fitted element SHALL be
non-negative. The fit SHALL report its largest residual and SHALL be
refused, naming the component, when the residual exceeds the tolerance
or the remaining series inductance is not positive.

#### Scenario: Ladder resistance against frequency
- **GIVEN** `R_dc = 10 mΩ` and one cell of 90 mΩ in parallel with
  143.24 nH
- **WHEN** the resistance is evaluated at 1 kHz, 100 kHz and 1 MHz
- **THEN** it is 10.009 mΩ, 55.00 mΩ and 99.11 mΩ within 1e-9 relative

#### Scenario: Harmonic loss in the time domain
- **WHEN** 1 A peak at 1 kHz plus 1 A peak at 100 kHz flows through
  that ladder
- **THEN** the dissipated power is 32.50 mW within 1e-3 relative

#### Scenario: Fit that cannot be realised
- **WHEN** the requested resistance rise needs more cell inductance than
  the inductor has
- **THEN** the fit fails naming the component and the inductance
  deficit

### Requirement: Transmission Line

The library SHALL provide a two-conductor transmission line defined by
its characteristic impedance `Z0`, one-way delay and series resistance,
in two forms: `pi`, a ladder of lumped sections usable in every
analysis, and `bergeron`, a travelling-wave model whose delay is exact.
The Bergeron form SHALL reproduce the ideal-line reflection lattice
without overshoot, SHALL refuse a fixed step larger than the delay, and
SHALL be refused by the conformance error in analyses that cannot
represent a delay.

#### Scenario: Matched source, open end
- **GIVEN** a 50 Ω, 100 ns Bergeron line driven by a 1 V step through
  50 Ω, far end open, with the delay a multiple of the step
- **WHEN** it runs on `pwl`
- **THEN** the far end is 0 V before 100 ns and 1.000 V after, within
  1e-9, with no overshoot

#### Scenario: Reflection lattice
- **GIVEN** the same line driven through 10 Ω
- **WHEN** it runs
- **THEN** the far-end voltage is 1.667 V, 0.556 V and 1.296 V after
  one, three and five delays, within 1e-9 relative

#### Scenario: Lumped form in the frequency domain
- **GIVEN** the `pi` form with 50 sections, far end open
- **WHEN** the input impedance is computed at `ωτ = 0.5`
- **THEN** it is −j91.52 Ω within 1e-3 relative

#### Scenario: Step larger than the delay
- **WHEN** a Bergeron line with a 100 ns delay is run on `pwl` with
  `dt = 1 µs`
- **THEN** the run is refused naming the line, the delay and the step

#### Scenario: Analysis without delays
- **WHEN** a circuit with a Bergeron line is run on `dsed` or swept in
  AC
- **THEN** it raises the conformance error naming the line and
  suggesting the `pi` form
