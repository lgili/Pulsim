## ADDED Requirements

### Requirement: Fixed-Step Resolution Diagnostics in Python

`pulsim.simulate` SHALL accept `resolution_diagnostics` (default
`True`), and a fixed-step result SHALL expose `res.resolution` with the
per-sample truncation trace aligned with `res.times`, the worst value
with the name of its branch and its time, the number of branches the
estimate covered, and the ringing records with node or branch names.
`res.resolution` SHALL be `None` for the variable-step and exact
engines and when the option is off. `simulate` SHALL emit exactly one
`pulsim.ResolutionWarning` (a `UserWarning` subclass) per call when
the worst relative truncation exceeds 1e-2 or a ringing record exists;
the message SHALL name the variable and the time and SHALL offer a
smaller `dt` or omitting `dt`.

#### Scenario: Under-resolved run warns once
- **GIVEN** a 10 V step into 1 kΩ and 1 nF (τ = 1 µs) with the
  capacitor on node `a`
- **WHEN** `simulate(b, t_end=100e-6, dt=1e-5)` is called
- **THEN** exactly one `ResolutionWarning` is emitted and it names
  node `a`
- **AND** `res.resolution.lte_max > 0.1` and
  `len(res.resolution.ringing) >= 1`

#### Scenario: Resolved run is silent
- **GIVEN** the same RC
- **WHEN** `simulate(b, t_end=100e-6, dt=1e-8)` is called
- **THEN** no `ResolutionWarning` is emitted and
  `res.resolution.lte_max < 1e-5`

#### Scenario: Trace is aligned with the samples
- **GIVEN** a fixed-step run with `store_every=10`
- **WHEN** the result is read
- **THEN** `len(res.resolution.lte) == len(res.times)` and each entry
  is the largest estimate of the ten steps before its sample

#### Scenario: Option off and other engines
- **GIVEN** `resolution_diagnostics=False`, or a run without `dt`
- **WHEN** the result is read
- **THEN** `res.resolution` is `None` and no `ResolutionWarning` is
  emitted
