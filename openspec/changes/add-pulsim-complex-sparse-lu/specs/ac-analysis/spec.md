## ADDED Requirements

### Requirement: AC Sweep Uses In-House Complex Sparse Solver

The kernel SHALL use `PulsimComplexSparseLuSolver`
(= `PulsimSparseLuSolver<std::complex<Real>>`) for the per-frequency
factorisation `(j·ω·E + J) · X = B` inside `mna_sweep.hpp` and any
descendant AC-sweep code paths. References to
`Eigen::SparseLU<std::complex<Real>>` SHALL NOT appear in the AC
sweep call site after this change lands.

Numerical accuracy at the user-observable level (Bode magnitude /
phase) SHALL match the previous Eigen-backed implementation within
the existing spec tolerances (0.1 dB magnitude / 1° phase on the
RC tank scenario; the existing `Multi-frequency factorization
reuse` and `Multi-input AC sweep` scenarios SHALL continue to hold
under the new solver).

#### Scenario: AC sweep call site declares in-house solver
- **GIVEN** the file `core/include/pulsim/analysis/mna_sweep.hpp`
  after this change lands
- **WHEN** the source is grepped for the solver declaration in
  the per-frequency loop
- **THEN** the declared type is `PulsimComplexSparseLuSolver`
  (or `PulsimSparseLuSolver<std::complex<Real>>`)
- **AND** the file does NOT include `<Eigen/SparseLU>`

#### Scenario: RC low-pass Bode parity vs reference
- **GIVEN** an RC low-pass with `R = 1 kΩ`, `C = 1 µF`
- **WHEN** `run_ac_sweep` covers 1 Hz–1 MHz with 20 points/decade
  on the new in-house complex solver
- **THEN** the Bode magnitude matches the analytic `1/(1 + jωRC)`
  form within `0.1 dB` across the full sweep
- **AND** the phase matches within `1°` across the full sweep

#### Scenario: 10 reference converter AC sweeps bit-identical
- **GIVEN** any of the 10 reference converter projects under
  `projects/` (buck, boost, buck-boost, flyback, forward, half-
  bridge, boost-pfc, vsi-3phase, npc-3phase, mmc) and an AC
  sweep configuration matching that project's existing notebook
- **WHEN** the sweep runs under v1.3.0 (Eigen-backed) and under
  the post-change build (Pulsim-backed)
- **THEN** the produced `H(jω)` arrays match element-wise within
  `1e-10` complex magnitude tolerance
- **AND** any small-signal-mode KPI extracted from the sweep
  (resonant frequency, peak Q, crossover frequency, phase
  margin) matches within `0.5 %` (relative) or `0.5°` (absolute
  for phase quantities)

#### Scenario: Singular-matrix failure path is preserved
- **GIVEN** an AC sweep that hits a frequency where the MNA
  matrix becomes numerically singular (e.g. a structurally
  singular topology configuration)
- **WHEN** the per-frequency factorize call runs
- **THEN** the kernel raises `std::runtime_error` with a
  diagnostic message identifying the failing frequency and
  identifying `PulsimComplexSparseLuSolver` as the active backend
- **AND** the failure mode (early termination of the sweep) is
  consistent with the v1.3.0 Eigen-backed behaviour
