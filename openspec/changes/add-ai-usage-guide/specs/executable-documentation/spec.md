## ADDED Requirements

### Requirement: Executed Documentation Snippets

The Python test suite SHALL execute every fenced code block whose info
string is `python` in the following files:

- `python/pulsim/AI_GUIDE.md`;
- `README.md`;
- `docs/getting-started.md`;
- `docs/tutorials/*.md`.

Execution rules:

- The blocks of one file SHALL run in document order in a single shared
  namespace, inside a temporary working directory, with
  `MPLBACKEND=Agg`.
- Each file SHALL be one test item.
- The tests SHALL carry the pytest marker `doc_snippets`, registered in
  `pyproject.toml`, and SHALL be collected by the default CI invocation
  `pytest python/tests`.
- A failure SHALL report the source file and the line number of the
  failing fence.

#### Scenario: Broken call in a tutorial
- **GIVEN** a tutorial block that calls a builder method with a
  misspelled keyword argument
- **WHEN** the test suite runs
- **THEN** the test for that tutorial fails
- **AND** the report names the tutorial file and the line of the
  failing fence

#### Scenario: Later blocks see earlier definitions
- **GIVEN** a tutorial whose second block uses the builder `b` and
  result `res` created in its first block
- **WHEN** the file is executed
- **THEN** the second block runs without `NameError`

#### Scenario: Non-python fences are not executed
- **WHEN** a file contains `bash`, `text`, `yaml` or unlabeled fences
- **THEN** those fences are not executed

### Requirement: Expected Output Comparison

The test suite SHALL compare a `python` block's captured stdout with
the fence that follows it, when that fence is unlabeled or `text`.
Between the two fences only blank lines, or a single prose line ending
in `:`, are allowed. Comparison rules:

- whitespace is normalised;
- non-numeric tokens SHALL match exactly;
- each numeric token SHALL match within one unit of its last printed
  digit, or within a relative 1e-6 if that is larger.

#### Scenario: Stale expected value
- **GIVEN** a tutorial whose expected-output fence says `3.161 V` while
  the code now prints `3.151 V`
- **WHEN** the test suite runs
- **THEN** the test fails and shows both values

#### Scenario: Last-digit platform difference
- **GIVEN** an expected-output fence printing `6.319` and a platform
  that prints `6.320`
- **WHEN** the test suite runs
- **THEN** the comparison passes

### Requirement: Snippet Annotations

The snippet runner SHALL recognise annotations given as an HTML
comment on the line directly above a fence. Four forms exist:

- `<!-- doctest: requires=<module>[,<module>] -->` SHALL skip the block,
  and the rest of its file, with the module named in the skip reason,
  when any listed module cannot be imported;
- `<!-- doctest: skip reason="<text>" -->` SHALL skip the block. The
  reason SHALL be non-empty;
- `<!-- doctest: no-output-check -->` SHALL execute the block without
  output comparison;
- `<!-- doctest: xfail reason="<text>" -->` SHALL require the block to
  raise `AssertionError`, SHALL fail the test when the block runs
  without error, and SHALL continue with the following blocks. The
  reason SHALL be non-empty.

An unrecognised `doctest:` annotation SHALL fail the test. Fences inside
blockquotes or indented under list items SHALL NOT be collected.

#### Scenario: Known defect fixed
- **GIVEN** a block annotated `doctest: xfail` whose assertion documents
  an open defect
- **WHEN** the defect is fixed and the block runs without error
- **THEN** the test fails, asking for the annotation to be removed

#### Scenario: Optional dependency absent
- **GIVEN** a block annotated `requires=matplotlib` in an environment
  without matplotlib
- **WHEN** the test suite runs
- **THEN** the file's test is reported as skipped, naming matplotlib
- **AND** it is not reported as passed

#### Scenario: Skip without reason
- **GIVEN** a block annotated `<!-- doctest: skip -->` with no reason
- **WHEN** the test suite runs
- **THEN** the test fails, asking for a reason

### Requirement: Snippet Runtime and Coverage in CI

All executed documentation snippets together SHALL complete within 60
seconds on a CI runner. The suite SHALL emit a warning above 30 seconds.
At least one CI job SHALL install every module named in a `requires=`
annotation, so that no annotated block is skipped in every job.

#### Scenario: Runtime budget
- **WHEN** the `doc_snippets` tests run in a CI job
- **THEN** their total wall time is at most 60 s
- **AND** a warning is emitted if it exceeds 30 s

#### Scenario: Plotting blocks run somewhere
- **GIVEN** README and getting-started blocks annotated
  `requires=matplotlib`
- **WHEN** the designated CI job that installs matplotlib runs
- **THEN** those blocks execute and are not skipped
