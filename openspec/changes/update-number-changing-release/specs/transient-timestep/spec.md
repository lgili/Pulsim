## ADDED Requirements

### Requirement: One Number-Changing Release

The library SHALL change every number-changing default in one release, from
one profile defined once as data (`benchmarks/tools/next_defaults.py`), and
SHALL NOT change any of them before it. Each profile entry SHALL name the
OpenSpec change and decision that own its value. Every flipped default SHALL
stay reachable by its keyword with the old value for at least one release,
and the CHANGELOG SHALL list each old value.

#### Scenario: Defaults before the release

- **WHEN** a run is made on a tree before the release with no keyword set
- **THEN** every option of the profile has today's value and
  `benchmarks/REFERENCE_STATUS.md` is unchanged

#### Scenario: The release flips the whole profile

- **WHEN** the release is cut
- **THEN** every profile option has its profile value as the default, the
  reference status, goldens and scoreboard are regenerated once, and the
  rehearsal's configuration "today" reproduces the new reference status

#### Scenario: A blocker stops the release

- **WHEN** the rehearsal shows a reference circuit that agrees under today's
  defaults and is refused, errors or disagrees under the profile, and the
  truth does not show the new number to be the better one
- **THEN** the release is not cut until that cell is fixed or recorded as an
  owner-accepted change with its evidence

### Requirement: Exact Engine Output Grid Without dt

The exact engine SHALL, under `simulate(engine='exact', output_rtol=r)`
without `dt`, `store_every` or recorders, return every event instant (both limits where the state
jumps) and, between events, samples spaced so that linear interpolation
between consecutive samples is within `r` times each state's scale (the
largest magnitude seen so far, floored at `atol`), from a closed-form bound
of the chord error over the mode's eigen-decomposition. A step cap per
interval SHALL bound the samples; when it bites the run SHALL say so by name
(`diagnostics['output_grid']['n_output_capped']` and a `RuntimeWarning`).
With `output_rtol` set the event projection SHALL remove only the parasitic
cluster, not every mode faster than the default step cap. `output_rtol=None`
(the default until the number-changing release) SHALL leave every result
byte-identical; an explicit `dt`, `store_every` or recorders SHALL keep
today's samples; the other engines SHALL refuse the keyword by name.

#### Scenario: The release route without dt

- **WHEN** a piecewise-linear buck runs on the exact engine without `dt`
  and with `output_rtol=1e-4`
- **THEN** its samples score below 1e-7 of span against the mpmath truth
  (the D1 turn-off right limit included) and the strict gate, interpolating
  linearly, agrees where the 10 µs default grid disagreed

#### Scenario: Default unchanged

- **WHEN** `output_rtol` is not given
- **THEN** every recorded time and state is bit-identical to the tree before
