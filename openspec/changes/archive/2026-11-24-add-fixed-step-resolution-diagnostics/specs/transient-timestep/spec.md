## ADDED Requirements

### Requirement: Fixed-Step Truncation Estimate

The fixed-step engine SHALL estimate, at every step, the local
truncation error of each linear capacitor voltage and inductor current
as `(dt/12)·|f_{k+1} − 2·f_k + f_{k−1}|`, with `f` the derivative the
trapezoidal companion stores (`i_C/C`, `v_L/L`), and SHALL record per
stored sample the largest estimate since the previous sample, relative
to the running peak magnitude of its variable, together with the
worst value of the run, its branch and its time. An estimate SHALL be
formed only when its three derivative values were produced under one
stamped switch mask with no restart step and no source jump among
them. The estimate SHALL NOT change any state: a run with
`resolution_diagnostics` on and off yields bit-identical states.

#### Scenario: RC stepped far above its time constant
- **GIVEN** a 10 V step into an RC run at `dt = 10·τ`
- **WHEN** the result is read
- **THEN** the worst relative truncation is above 0.1 (the node is
  1.87 V off and overshoots to 11.11 V, and nothing is reported
  without this change)

#### Scenario: RC resolved
- **GIVEN** the same RC at `dt = τ/100`
- **WHEN** the result is read
- **THEN** the worst relative truncation is below 1e-5

#### Scenario: A well-resolved converter
- **GIVEN** a 24 V buck (100 µH, 47 µF, 1 Ω, 100 kHz) in continuous
  conduction at T/100
- **WHEN** it runs for 40 periods
- **THEN** the worst relative truncation is below 1e-3 and no estimate
  was formed across a gate edge

#### Scenario: Diagnostics do not change results
- **GIVEN** any fixed-step circuit, with or without Newton devices
- **WHEN** it runs with `resolution_diagnostics` on and off
- **THEN** the two state trajectories are bit-identical

### Requirement: Fixed-Step Nyquist Ringing Detector

The fixed-step engine SHALL record every node voltage or branch
current whose step-to-step difference changes sign on four or more
consecutive steps while each difference exceeds 1 % of that entry's
magnitude at the two samples it joins. A record SHALL carry the
entry, the time the run of alternating steps started, its length and
its largest difference.

#### Scenario: Under-resolved RC rings
- **GIVEN** the RC at `dt = 10·τ`
- **WHEN** it runs
- **THEN** a ringing record names the capacitor node with a run of at
  least 4 alternating steps

#### Scenario: Idle interval of a DCM buck
- **GIVEN** a buck in discontinuous conduction at T/100 on the engine
  without the discontinuity restart rule
- **WHEN** it runs
- **THEN** a ringing record names node `sw` (its samples alternate
  0.50 V / 26.15 V around the true 13.31 V)

#### Scenario: Smooth run is silent
- **GIVEN** an ideal LC oscillation resolved with 100 steps per period
- **WHEN** it runs for 1000 periods
- **THEN** no ringing record is produced
