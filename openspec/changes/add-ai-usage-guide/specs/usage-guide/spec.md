## ADDED Requirements

### Requirement: In-Package Usage Guide

The `pulsim` package SHALL ship a usage guide for AI coding assistants
and human users at `pulsim/AI_GUIDE.md`, in the wheel and in the
build-tree package copy. The guide SHALL describe the API of the version
it ships with. It SHALL be at most 32 KiB.

#### Scenario: Guide ships with the package
- **GIVEN** pulsim installed from a wheel, in an environment with no
  network access
- **WHEN** Python reads
  `importlib.resources.files("pulsim").joinpath("AI_GUIDE.md")`
- **THEN** the file exists and is non-empty UTF-8 text

#### Scenario: Size budget
- **WHEN** the test suite measures `AI_GUIDE.md`
- **THEN** its size is at most 32768 bytes

### Requirement: Guide Accessor and Command-Line Entry

The package SHALL expose `pulsim.ai_guide() -> str`, which returns the
guide text prefixed by one header line containing `pulsim.__version__`
and the guide's file path. `python -m pulsim.guide` SHALL print the same
text to stdout and exit 0. `python -m pulsim.guide --path` SHALL print
only the file path. Importing `pulsim` SHALL NOT import the
`pulsim.guide` module, so the command-line entry runs without a runpy
`RuntimeWarning`. The name `ai_guide` SHALL NOT shadow any existing
attribute of `pulsim`.

#### Scenario: Function returns version-matched text
- **WHEN** Python calls `pulsim.ai_guide()`
- **THEN** the first line contains `pulsim.__version__`
- **AND** the remainder equals the content of `AI_GUIDE.md`

#### Scenario: Command-line entry
- **WHEN** a user runs `python -W error -m pulsim.guide`
- **THEN** the process exits 0, with no warning
- **AND** stdout equals `pulsim.ai_guide()` followed by a newline

#### Scenario: Path mode
- **WHEN** a user runs `python -m pulsim.guide --path`
- **THEN** stdout is a single line naming an existing file whose content
  is the guide

### Requirement: Guide Content Rules

The guide SHALL contain, each with an executed example:

- how to query the installed package for the API (the guide CLI, the
  first line of `__doc__` holding the signature, the stubs,
  `pulsim.catalog()` / `pulsim.example()`);
- a minimal correct script that uses `CircuitBuilder`,
  `pulsim.simulate` and named result access;
- the argument order of the commonly mis-guessed builder calls;
- the semantic traps listed below;
- how to verify a result physically: a charge or energy balance from
  independent branch currents, and a comparison of two engines with
  time-weighted statistics.

The semantic traps are:

- PWL diode `V_th` and the topology-helper parameters that feed it are
  commutation thresholds, not forward drops;
- the default engine is variable-step, so sample means are biased and
  `pulsim.stats` or `np.trapezoid` must be used;
- `SwitchStateMask(n)` takes a size, and bit indices come from
  `switch_index_of`;
- `tol_newton_dx` / `tol_newton_res` are absolute and are not to be
  loosened;
- `pulsim.scope` is the plotting function, and `LiveScope` is imported
  from the `pulsim.scope` module;
- isolated subnets are referenced through a large resistor;
- a floating DC side may need one rail referenced;
- the fixed engine starts first order from an inconsistent state.

The guide SHALL NOT contain:

- code that is not executed by the test suite;
- Pulsim 1.x API in code;
- performance figures, roadmap or release history;
- workarounds that compensate for an open defect in reported results.

Where a known defect affects results, the guide SHALL teach the check
that exposes it. It SHALL phrase the consequence so that it stays true
after the fix ships (for example, "if the charge balance does not close,
do not trust the result").

#### Scenario: Diode threshold is taught correctly
- **WHEN** the guide's diode section is executed
- **THEN** it shows that the on-state voltage at a forced current is
  `I/g_on` for `V_th = 0` and for `V_th = 0.7` alike

#### Scenario: Charge balance closes for a thresholded diode
- **GIVEN** the guide's charge-balance example on a rectifier whose
  diodes have `V_th > 0`
- **WHEN** the example is executed
- **THEN** the source charge and the sum of the diode branch charges
  agree within the tolerance stated in the example

#### Scenario: No 1.x code
- **WHEN** the test suite scans the guide's code fences
- **THEN** no fence references `p.<name>` or `pulsim.<name>` where
  `<name>` is a key of `pulsim._V1_SYMBOL_HINTS` that is not an
  attribute of the installed package

### Requirement: Guide Names Resolve

A test SHALL check that every inline reference in the guide written as
`p.<name>`, `pulsim.<name>` or `CircuitBuilder.<method>` resolves to an
attribute of the installed package.

#### Scenario: Invented name in the guide
- **GIVEN** a guide edit that mentions `p.add_buck_converter`, which does
  not exist
- **WHEN** the test suite runs
- **THEN** the name-resolution test fails and names `add_buck_converter`

### Requirement: Unknown-Attribute Hint Points to the Guide

The package's module-level `__getattr__` SHALL name
`python -m pulsim.guide` in the `AttributeError` it raises when code
accesses a `pulsim` attribute that does not exist. This applies both to v1 names, which keep
their migration hint, and to unknown names.

#### Scenario: Invented top-level name
- **WHEN** Python evaluates `pulsim.Simulator`
- **THEN** an `AttributeError` is raised whose message contains the
  existing migration hint and `python -m pulsim.guide`

### Requirement: In-Package Text Consistency

Docstrings and catalogue text shipped in the package SHALL NOT
contradict the guide. In particular:

- no docstring SHALL describe a PWL diode threshold parameter (`V_th`,
  `V_f`, or `V_F` where it feeds a PWL diode) as a forward drop;
- `pulsim.catalog()` SHALL NOT recommend micro-ohm resistors to
  reference floating subnets.

#### Scenario: VSI diode parameter wording
- **WHEN** a user reads `help(pulsim.add_three_phase_vsi)`
- **THEN** `V_f` is described as a commutation threshold, with the
  on-state set by `R_d` alone
- **AND** the text does not call it a forward drop

#### Scenario: Catalogue reference advice
- **WHEN** a user calls `pulsim.catalog()`
- **THEN** the `add_resistor` entry advises a large resistor to
  reference an isolated subnet
- **AND** it contains no micro-ohm recommendation

### Requirement: Machine-Readable Docs Index

The documentation SHALL include `docs/llms.txt` following the llms.txt
convention:

- an H1 title;
- a blockquote summary;
- `##` sections of markdown link lists.

Its first link SHALL be the usage guide. Its summary SHALL direct
assistants to `python -m pulsim.guide` for the installed version. The
MkDocs build SHALL publish the file, and SHALL render the guide as a
site page included from `python/pulsim/AI_GUIDE.md`, so that there is a
single source.

#### Scenario: Index structure
- **WHEN** the test suite parses `docs/llms.txt`
- **THEN** line 1 is an H1 and the first non-blank line after it is a
  blockquote
- **AND** every link target is an existing repository file or an
  absolute URL

#### Scenario: Published with the site
- **WHEN** `mkdocs build --strict` runs
- **THEN** the build succeeds
- **AND** `site/llms.txt` exists
- **AND** the rendered guide page contains the guide's first heading
