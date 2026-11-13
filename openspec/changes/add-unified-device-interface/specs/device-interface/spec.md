## ADDED Requirements

### Requirement: Newton Device Residual Contract

A Newton device kind SHALL describe its contribution to the circuit
equations as `f(x, t) + d/dt q(x) = 0` through one templated function
`eval(params, x, ctx, out)` that fills `out.f[r]` and `out.q[r]` for its
`n_pins` terminal rows and its `n_internal` own rows. The kernel SHALL
obtain `∂f/∂x` and `∂q/∂x` by evaluating the same function over the AD
scalar; a kind SHALL NOT supply a separately written Jacobian. A law
with an exact tangent that AD cannot derive (a table) SHALL return it
through the AD scalar from inside `eval`.

#### Scenario: One function yields residual and Jacobian
- **GIVEN** a device kind that defines only `Params`, `name`, `n_pins`,
  `n_internal` and `eval`
- **WHEN** it is stamped at a state `x`
- **THEN** the stamped Jacobian equals the central finite difference of
  the stamped residual within 1e-6 relative at 100 random states in the
  kind's operating range

#### Scenario: Re-expressed Shockley diode is unchanged
- **GIVEN** the Shockley diode written against the contract (`f` only)
- **WHEN** the half-wave clamp and the Shockley buck of the reference
  harness run on the fixed engine
- **THEN** every state agrees with the previous build within 1e-10
  relative and the Newton iteration count per step is equal

#### Scenario: Table law keeps its exact tangent
- **GIVEN** a flux-table inductor whose `eval` returns `λ(i)` with the
  Hermite tangent
- **WHEN** it is stamped
- **THEN** the Jacobian entry of the flux row equals the table's exact
  `dλ/di` to machine precision

### Requirement: Integrator Owns The Time Formula

The kernel SHALL turn `(f, q)` into a Newton row with one generic stamp
for the formulas `Dc`, `BackwardEuler`, `Trapezoidal` and `Bdf2Stage2`,
as `f + a·q + r` with `a` and `r` taken from the step context of that
solve. The step context SHALL be an argument of every solve; a device
stamp MUST NOT read a step size or stage from any other place. The row
SHALL NOT be rescaled by the step size. The time a solve spans SHALL be
the time its commit integrates over.

#### Scenario: A probe solve stamps devices with its own step
- **GIVEN** a circuit with a PWL diode beside a saturable inductor and
  beside an MNA PMSM
- **WHEN** the variable-step engine runs its zero-time diode settle and
  its event probe solves
- **THEN** every device is stamped with the probe's step, the run
  completes, and no state is NaN

#### Scenario: Flux row stays in conductance form
- **WHEN** a saturable inductor is stamped for the trapezoidal formula
  at step `h` and for the BDF2 stage
- **THEN** the diagonal of its flux row equals `−(2/h)·L(i)` and
  `−(c1/h)·L(i)` respectively
- **AND** the flux-closure test on the fixed engine shows no flux
  ratchet (below its existing bound)

#### Scenario: Backward Euler needs no device code
- **GIVEN** a kind written before the `BackwardEuler` formula was used
  by any engine
- **WHEN** a solve requests `BackwardEuler` over `h`
- **THEN** the kind's rows are stamped as `f + (q − q_n)/h` with no
  change to the kind's source

#### Scenario: No formula is built at another step size
- **WHEN** any engine takes a step that spans `h`
- **THEN** every charge-integrating device commits over exactly `h`
  (a charge-based capacitor's committed charge after one step equals
  `q(v)` at the accepted state)

### Requirement: Single Device Registry

The kernel SHALL have one compile-time list of device kinds, from which
the kind enumeration, the storage, the stamp loops, the state layout and
the capability table are derived. Adding a kind SHALL require the kind's
own header and one registry entry; it MUST NOT require an edit to any
transient driver, analysis routine, snapshot structure or Python source
for the kind to be stamped, integrated, snapshotted, linearised and
reported.

#### Scenario: Adding a device kind touches no engine source file
- **GIVEN** a test-only Newton kind registered in the test build
- **WHEN** a circuit containing it runs on every engine the conformance
  matrix lists
- **THEN** it either runs and agrees with the reference engine or is
  refused by the single `UnsupportedError` naming the kind and the
  engine
- **AND** the test build differs from the release build in no file
  under `solver/`, `analysis/`, `pwl/` or `python/`

#### Scenario: Capability table cannot omit a kind
- **WHEN** a kind with a charge row is added to the registry
- **THEN** every engine guard sees it as a Newton kind without any list
  being edited (the build fails if a kind has no category)

### Requirement: Stateful Devices Compose

Device stamps SHALL be additive and independent: the stamp of one
instance MUST NOT depend on which other kinds are present in the
circuit, and no kind's stamp may replace another's.

#### Scenario: Every pair of stateful kinds
- **GIVEN** each unordered pair of the stateful kinds (charge-based
  capacitor, Lauritzen diode, IGBT tail, MNA PMSM, saturable inductor)
  in one circuit, electrically independent
- **WHEN** the circuit runs on the fixed and the variable-step engine
- **THEN** each device's result equals its single-kind result within
  1e-9 relative

#### Scenario: A disconnected saturable inductor changes nothing
- **GIVEN** an IGBT with a turn-off tail and a saturable inductor on an
  isolated node
- **WHEN** the turn-off energy is measured
- **THEN** it equals the value without the inductor (16.87 mJ in the
  regression circuit) within 1e-6 relative

### Requirement: Flat Device State And Engine-Independent Snapshot

All device state SHALL live in one flat vector owned by the run, laid
out per kind in registry order, holding `(q_n, q̇_n)` per charge row and
the reals of the kind's `State`. Rolling back a rejected step SHALL be
one copy of that vector. A snapshot SHALL be
`{t, x, dev_state, mask}` plus the stamped mask and a layout signature,
SHALL hold physical quantities only, and SHALL be refused by name when
its layout does not match the circuit.

#### Scenario: Exact resume
- **GIVEN** a circuit containing every stateful kind
- **WHEN** a fixed-step run is split at an arbitrary step and resumed
  from the snapshot
- **THEN** the resumed trace is bit-identical to the continuous run

#### Scenario: A snapshot taken on engine A resumes on engine B
- **GIVEN** a snapshot taken at the end of a fixed-step run
- **WHEN** the variable-step engine resumes from it, and the reverse
- **THEN** the resumed run starts from the same `x`, device state and
  mode bits (first resumed sample equal to the snapshot's `x`) and
  agrees with the continuous run within the variable-step tolerance

#### Scenario: Snapshot of another circuit is refused
- **WHEN** a run is resumed from a snapshot whose layout signature
  differs
- **THEN** it raises an error naming the first kind block that differs
  and does not start from zero

#### Scenario: Version 2.0 pickles load
- **WHEN** a snapshot pickled by Pulsim 2.0 (10 or 11 fields) is
  unpickled and passed to `resume_from`
- **THEN** the run resumes with the same result as on 2.0, during the
  compatibility release

### Requirement: PWL Mode Devices Share One Decision Law

A two-state PWL device SHALL supply a constant stamp per mode and, when
it commutates by itself, exactly one `decide` and one `crossing_signal`
function. Every place that decides a mode (fixed-step event iteration,
DC operating point, seeding, variable-step settle and crossing
detection, event-driven settle) SHALL call those functions; none SHALL
restate the rule. The mode stamp MUST NOT change the sparsity pattern.

#### Scenario: One law at every decision site
- **GIVEN** a diode with `V_f = 0.7`, `V_th = 0.75`
- **WHEN** it is held at `v = V_f ± ε` and at `v = V_th ± ε` on every
  engine and in the DC operating point
- **THEN** the conduction state is the one `decide` returns and the
  diode current is never below `−(1e-7 A + 1e-4·max|i|)`

#### Scenario: Cached factors stay valid
- **WHEN** a mode device's offset parameter changes (`V_f`)
- **THEN** only `b` changes for the ON mode and the matrix of every
  mask is entry-by-entry identical

### Requirement: Optional State, Events And Outputs

The kernel SHALL honour three optional members of a device kind on
every engine: a trivially copyable `State` with `init` and `accept`,
zero-crossing functions `events`, and extra output channels. `accept`
SHALL run
exactly once per accepted step and never on a rejected or probe solve;
`events` SHALL be offered to the event locator; outputs SHALL be
available by name. Pin currents `i_k = f_k + d/dt q_k` and power
`Σ v_k·i_k` SHALL exist for every device without any per-kind code.

#### Scenario: Hysteresis state advances only on accepted steps
- **GIVEN** a Jiles–Atherton core on the variable-step engine with
  rejected steps
- **WHEN** the run completes
- **THEN** its `(H, M)` trajectory equals the one obtained with the
  rejected steps removed (accept count equals `n_accept`)

#### Scenario: Generic current of a charge device
- **GIVEN** a nonlinear capacitor in series with a 1 mΩ resistor
- **WHEN** the device current is read
- **THEN** it equals the resistor current within 1e-6 relative at every
  stored sample on both engines

### Requirement: Device Field Table

Each kind SHALL declare its parameters once, as a table of name, unit,
default and bounds. The builder's keyword validation, `components()`,
the YAML keys, the type stubs and the reference documentation SHALL be
derived from that table, and a value outside its bounds SHALL be refused
with an error naming the device and the field.

#### Scenario: Every field is reachable everywhere
- **WHEN** the field tables are enumerated
- **THEN** for every kind and field, the builder accepts the keyword,
  `components()` reports it and the YAML loader accepts the key

#### Scenario: Out-of-range value refused by name
- **WHEN** a Shockley diode is added with `I_S = -1`
- **THEN** an error naming the device and `I_S` is raised

### Requirement: No Per-Device Indirection In The Newton Loop

Stamping the Newton devices SHALL use static dispatch over the registry
and precomputed matrix slots. The per-iteration path MUST NOT perform a
virtual call, a `std::function` call or a heap allocation per device,
and the per-step path MUST NOT allocate.

#### Scenario: Zero allocations with every Newton kind
- **GIVEN** a circuit with one instance of every Newton kind
- **WHEN** 1000 steps run after warm-up under the allocation counter
- **THEN** the number of heap allocations is zero

#### Scenario: No slowdown from the port
- **WHEN** the Shockley buck and the MOSFET level-1 circuit of the audit
  are timed before and after a kind is ported
- **THEN** Newton iterations and factorisations per step are equal and
  ns/step is not above the pre-port value (best of five, same machine)

### Requirement: PWL Category Carries The Complementarity Form

Every self-commutated PWL device SHALL expose, for each of its ports,
the ideal element's orientation, `V_f`, `R_on`, `g_off` and `V_th` to
the mode-selection service through the device registry, with
`R_on = 1/g_on` in the conductance form, and SHALL NOT change the
sparsity pattern by doing so. When the complementarity selection is in
effect, decision sites SHALL take modes from it instead of calling
`decide`.

#### Scenario: Conductance-form diode ports
- **GIVEN** a diode with `g_on = 1e3`, `g_off = 1e-9`, `V_f = 0.7`,
  `V_th = 0`
- **WHEN** its ports are read from the registry
- **THEN** one port is returned with `R_on = 1e-3`, `g_off = 1e-9`,
  `V_f = 0.7` and no hysteresis
