## ADDED Requirements

### Requirement: Design Evaluation Contract

The library SHALL provide `pulsim.design.Evaluator` whose `evaluate`
method takes a design (a mapping of named values) and returns an
immutable `EvaluationRecord` holding the design and its hash, a status
(`ok`, `refused`, `failed`, `infeasible`), the KPIs (efficiency, losses
per device and category, junction temperatures, ripple, voltage and
current stresses, user KPIs) computed on one steady period with
time-weighted integrals, the constraint violations, the solver and
route counters, and the provenance (version, build hash, options).

#### Scenario: Buck KPIs against a reference
- **GIVEN** a CCM buck with resistor-pair switches (`g_on = 1e3`,
  `g_off = 1e-9`) and a 10 mΩ inductor resistance, evaluated with
  `engine='dsed'` and `method='pss'`
- **WHEN** it is evaluated
- **THEN** the status is `ok` and efficiency, loss and output ripple
  agree with a matrix-exponential reference of the same periodic orbit
  within 1e-8 relative

### Requirement: Deterministic And Thread-Safe Evaluation

An evaluation SHALL depend only on the design, the evaluator's options
and the Pulsim build: two evaluations of the same design SHALL return
records equal in every field except the wall time, whether they run
sequentially or concurrently on any number of threads, and evaluations
SHALL share no mutable state.

#### Scenario: Batch on one and eight workers
- **GIVEN** 64 designs of the worked buck example
- **WHEN** they are evaluated with `evaluate_many` on 1 worker and on
  8 workers
- **THEN** the two record lists are equal element by element, except
  the wall time

### Requirement: Typed Failure Records

An evaluation SHALL return a `refused` record carrying the engine,
feature, device and kind of a `pulsim.UnsupportedError`, a `failed`
record with the reason and counters on non-convergence, and an
`infeasible` record with the violation amounts when a constraint bound
is exceeded or the thermal loop runs away. It SHALL NOT substitute a
default KPI value for a failure, and SHALL re-raise any other
exception. `evaluate_many` SHALL continue past design-specific
failures and SHALL stop with the error when its first three
evaluations are refused with identical fields.

#### Scenario: Refusal recorded
- **GIVEN** a design whose circuit contains Shockley diodes, evaluated
  with `engine='dsed'` (refused by today's capability table)
- **WHEN** it is evaluated
- **THEN** the record has status `refused`, the four refusal fields,
  and no KPIs

#### Scenario: Ripple constraint
- **GIVEN** a constraint `ripple_v ≤ 0.12 V` and a design whose ripple
  is 0.15 V
- **WHEN** it is evaluated
- **THEN** the status is `infeasible`, the KPIs are present and
  `violations['ripple_v']` is 0.03 V within the KPI's accuracy

### Requirement: Junction Temperature Fixed Point

The evaluator SHALL, until junction temperatures are closed inside
the kernel, compute them by a damped fixed-point iteration between
the loss evaluation at the current temperatures and the steady thermal
network, to 0.1 K within 20 iterations, SHALL re-simulate only when
the circuit's electrical behaviour depends on temperature, and SHALL
return `infeasible` with `thermal_runaway` when the iteration diverges
or exceeds the temperature limit.

#### Scenario: Linear loss model
- **GIVEN** a device loss `P = P₀·(1 + α·(T − 25))` with `P₀ = 10 W`,
  `α = 0.004 K⁻¹`, `R_th = 2 K/W`, `T_amb = 40 °C`
- **WHEN** the design is evaluated
- **THEN** `T_j` equals `(T_amb + R_th·P₀·(1 − 25α)) / (1 − R_th·P₀·α)`
  = 63.043 °C within 0.01 K

#### Scenario: Runaway
- **GIVEN** the same model with `R_th·P₀·α = 1.2`
- **WHEN** the design is evaluated
- **THEN** the status is `infeasible` with a `thermal_runaway`
  violation

### Requirement: Derivative-Free Default For Optimisation

The evaluator SHALL NOT offer finite-difference gradients; gradients
SHALL be available only through the parameter-sensitivity change, and
the worked optimisation example SHALL use a derivative-free method by
default, with its optimiser dependency optional and a numpy fallback
when it is not installed.

#### Scenario: Example without the optional dependency
- **GIVEN** an environment without `pymoo`
- **WHEN** the worked buck example runs with 32 designs
- **THEN** it completes with a Latin-hypercube sample and a
  non-dominated filter, prints the counts of each status, and every
  member of the reported front satisfies the constraints
