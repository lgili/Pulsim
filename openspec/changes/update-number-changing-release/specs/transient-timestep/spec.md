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
