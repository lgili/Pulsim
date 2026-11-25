## Context

Paths: PY = `python/pulsim/__init__.py`, DD = `python/pulsim/_dsed_dispatch.py`,
BN = `python/bindings.cpp`, `core/…` = `core/include/pulsim/…`.

The shape of `simulate()` today:

```
PY:1815  LoadedCircuit + run window
PY:1837  merge the `solver=` bundle into the flat keywords
PY:1885  validate engine keywords
PY:1901  preflight (mutates the builder)
PY:1956  resume_from validation, builder ICs
PY:1990  if engine == "dsed": … return            (PY:2116)
PY:2117  routing: _trbdf2_blockers, 'auto'
PY:2196  if engine == "trbdf2": compose, default mask, run … return   (PY:2363)
PY:2365  engine == "pwl": compose again, default mask again, MMC, C blocks,
         progress, live stream, cache, run, voltage sanity … return   (PY:3006)
```

Everything above PY:1990 applies to every engine. Everything below it
applies to one. The owner's note *simulate-engine-dispatch-order*
records the rule ("cross-cutting logic must sit above the dispatch") and
the bug that taught it. The rule is kept by discipline, not by
structure.

What is well designed and must survive: the refusal culture (named
refusals instead of fallbacks), `SimulationAborted` carrying the partial
result, `engine_used` and `engine_route_reason`, the preflight findings,
`explain_singular` row names, the rule "an explicit `dt` is a request
for a fixed step" (PY:2150-2165), the exact snapshot resume.

Constraints from the owner's notes:

- *simulate-engine-dispatch-order*: anything cross-cutting is resolved
  before any engine code.
- *gui-swallows-backend-refusals*: a refusal must be a type a caller can
  let through, and a refusal fix is complete only when every caller up
  to the UI lets it through.
- *switchstatemask-takes-a-size*: the default mask is a policy with a
  name, never an implicit pattern.
- *fixed-engine-first-order-start*: the initial condition is a declared
  thing (zero, state, DC operating point, snapshot), resolved once.

## Goals / Non-Goals

Goals:

- Every `simulate()` keyword is resolved exactly once, before any engine
  code runs, and the engines read only the resolved plan.
- A kernel object that *is* a simulation, usable from C++ with no
  Python and no callback the user must know to pass.
- One options structure with one defaults table and no sentinel values.
- A typed error hierarchy and a structured diagnostics list.
- A migration in which Stage A never changes a result or a warning.

Non-Goals:

- No change to any engine's step loop (that is
  `refactor-unified-transient-stepper`; `Simulation::step()` is
  delivered there).
- No new circuit representation: until `add-immutable-circuit-ir`,
  `Circuit` is a const view of the builder's graph and pool.
- No change to results storage (`add-probes-and-recorders`).
- Plotting, sweeps, datasheet and loss tables, schematic rendering,
  YAML sugar and the callback adapters stay in Python.
- The MMC Thevenin driver and the loss / thermal post-processing are
  not moved in this change (see Open Questions).

## Decisions

### D1. Stage A: a plan, then a dispatch table

```python
@dataclass(frozen=True)
class RunPlan:
    circuit: "CircuitBuilder"            # until add-immutable-circuit-ir
    engine: str                          # resolved: 'pwl' | 'trbdf2' | 'dsed'
    engine_asked: str
    route_reasons: tuple[str, ...]
    t_start: float; t_end: float
    step: "FixedStep | VariableStep | ExactStep"     # resolved numbers, no None
    newton: NewtonOptions; events: EventOptions
    initial: "Zero | State | BuilderICs | DcOperatingPoint | Resume"
    gates: tuple["GateSource", ...]      # composed by switch index
    undriven: tuple[int, ...]            # controlled switches no source owns
    ticks: tuple["Tick", ...]            # (rate, handler)
    source_terms: tuple[Callable, ...]   # b_extra providers
    features: frozenset["Feature"]       # StoreEvery, LiveStream, MmcArms, CBlocks, OnCache, …
    record: RecordOptions
    preflight: "PreflightReport | None"; ties: tuple["Tie", ...]
    diagnostics: tuple["Diagnostic", ...]
    resolved_from: Mapping[str, str]     # keyword -> 'explicit'|'solver'|'yaml'|'default'

def simulate(builder, t_end=None, dt=None, **kw):
    plan = _plan_run(builder, t_end, dt, **kw)       # pure; raises typed errors
    result = _ENGINES[plan.engine](plan)             # _run_pwl / _run_trbdf2 / _run_dsed
    return _finalize(result, plan)                   # same code for every engine
```

- `_plan_run` is pure: it does not run a solver, does not touch a
  cache and does not mutate the circuit. The regularising ties are
  computed and returned in `plan.ties`; applying them is one named step
  (`_apply_ties`) that, until `add-immutable-circuit-ir`, still appends
  to the builder as today.
- `resolved_from` has one entry per `simulate()` parameter. A test
  asserts that its key set equals the signature's parameter set, so a
  new keyword cannot be added without being planned.
- A second test parses `_run_pwl`, `_run_trbdf2`, `_run_dsed` with
  `ast` and asserts that their only free input is `plan`.
- `_finalize` attaches the handles, `engine_used`, the route reason and
  the diagnostics, runs the voltage-sanity check and emits warnings,
  for every engine. The check that is skipped on `dsed` today starts
  running there (**BREAKING**, its own PR).

- Alternatives considered.
  - *Keep the function and add comments and tests.* Rejected: that is
    the present state; the IC bug happened with the comment in place.
  - *Go straight to the C++ `Simulation`.* Rejected: it would move
    1 190 lines of behaviour and change language in one step. Stage A
    is a no-behaviour-change refactor that the existing 127 test files
    verify, and it produces the specification of what Stage B must do.
  - *One class per engine with a common base.* Rejected: it keeps
    "per engine" as the primary axis, which is the axis the unified
    stepper removes.

### D2. One options structure

```cpp
namespace pulsim::sim {

struct NewtonOptions { Size max_iterations = 50; Real tol_dx = 1e-9; Real tol_res = 1e-9;
                       bool line_search = false; bool lm = false; };
struct EventOptions  { Size max_iterations = 16; bool strict = false; };
struct InductorGuards { std::optional<Real> freeze_di_max; std::optional<Real> abs_clamp; };

struct FixedStep    { Real dt; Size max_dt_halvings = 6; bool substep_state_correction = false;
                      InductorGuards guards; };
struct VariableStep { Real rtol = 1e-5; Real atol = 1e-8;
                      std::optional<Real> h_init, h_max, h_min; Size max_steps = 10'000'000; };
struct ExactStep    { Real rtol = 1e-6; Real atol = 1e-9; std::optional<Real> dt_max;
                      std::optional<std::string> integrator; };
struct AutoStep     { std::optional<Real> dt; VariableStep variable; };   // dt given → fixed

struct Zero {}; struct FromCircuit {}; struct DcOperatingPoint {};
struct State  { Vector x; }; struct Resume { SolverSnapshot snapshot; };
using InitialCondition = std::variant<FromCircuit, Zero, State, DcOperatingPoint, Resume>;

enum class UndrivenSwitchPolicy { Refuse, Open, Closed };

struct RunOptions {
    Real t_start = 0; Real t_end = 0;
    std::variant<AutoStep, FixedStep, VariableStep, ExactStep> step = AutoStep{};
    NewtonOptions newton; EventOptions events;
    InitialCondition initial = FromCircuit{};
    UndrivenSwitchPolicy undriven = UndrivenSwitchPolicy::Refuse;
    bool auto_regularize = true;
    std::optional<Real> voltage_sanity_factor;       // nullopt → default 100
    RecordOptions record;                            // store_every today; add-probes-and-recorders extends
};
}
```

- Every default is the literal in this header. `SimulationOptions` and
  `TrBdf2Options` are built from `RunOptions` by two private functions
  for as long as the old drivers exist.
- No sentinel: "not set" is `std::nullopt` / `None`. The four
  recorded-only fields of `SimulationOptions` are not carried over.
- The tolerance defaults are the ones `simulate()` uses today (1e-5 /
  1e-8 on the MNA variable-step path, 1e-6 / 1e-9 on the event-driven
  path); the raw TR-BDF2 binding's 1e-4 / 1e-6 goes away.
- `InitialCondition::FromCircuit` means: the builder's `c0=` / `i0=` /
  `set_initial` values if any, else zero. `Resume` excludes ICs (PY:1976-1988).
- A feature that an engine cannot honour is not an option that is
  ignored: it is a refusal from the capability check (D4).

- Alternative considered: keep flat keywords in C++ mirroring
  `simulate()`. Rejected: 37 flat fields with engine-specific meaning is
  what produced the three tolerance defaults.

### D3. The kernel object

```cpp
namespace pulsim::sim {

class Simulation {
public:
    Simulation(Circuit circuit, RunOptions options);   // plans; throws PulsimError subclasses

    // Run-time attachments: things that are not part of the circuit.
    void add_gate(events::AnyGateSource);              // index-composed; overlap → NetlistError
    void add_tick(events::Rate, events::TickHandler);
    void add_source_term(SourceTermFn);                // b_extra provider
    void set_cancel(std::function<bool()>);

    const RunPlan& plan() const noexcept;              // engine, route reasons, IC, ties, features
    Result   run();                                    // to t_end
    Result   run_until(Real t);                        // resumable
    Snapshot snapshot() const;                         // state after the last completed run call
    void     restore(const Snapshot&);
    std::span<const Diagnostic> diagnostics() const noexcept;
    // StepStatus step();   — delivered by refactor-unified-transient-stepper
};
}
```

What the constructor and `run()` own (each is today a block of
`simulate()`):

| Responsibility | Today | In `Simulation` |
|---|---|---|
| Preflight | PY:1901-1932 | constructor; ties returned in the plan, applied to a private copy of the lowered circuit |
| Capability check | `_trbdf2_blockers` (PY:3177-3289), `_dsed_unsupported` (PY:2010+), `has_nonlinear_devices` guards | one lookup in the table of `add-engine-conformance-matrix` |
| Routing | PY:2134-2195 | `AutoStep` resolution; reasons recorded |
| Initial condition | PY:1976-1988 and per driver | `InitialCondition`, resolved once |
| Device stamps | binding (`make_combined_diode_mosfet_refresh`) | built from the pool (or the registry of `add-unified-device-interface`) |
| Gate and tick composition | PY:2215-2283, PY:2420-2455 | `add_gate` / `add_tick` (on the queue of `add-unified-event-queue` when present, else adapted to `switch_fn` / observer) |
| Undriven switches | PY:2304, PY:2664-2673, DD:617-630 | `UndrivenSwitchPolicy` |
| Cache construction | per branch | `run()` |
| Voltage sanity, guard reports | PY:2360, PY:2922-2930 | `run()` → diagnostics |
| Result decoration | attributes set per branch | fields of the result |

`run_until` + `snapshot` / `restore` give stepping between runs with
the exact resume the fixed engine already has; a per-step `step()`
needs a steppable loop and comes with the unified stepper.

`Circuit` in this change:

```cpp
struct Circuit { const topology::Graph* graph; const pwl::DevicePool* pool;
                 const builder::CircuitBuilder* builder; };   // non-owning view
```

`add-immutable-circuit-ir` replaces it with the owning immutable type;
the `Simulation` constructor signature does not change.

- Alternatives considered.
  - *Free function `simulate(circuit, options)` in C++.* Rejected: no
    place for attachments, snapshots or stepping.
  - *Put the planning in the binding layer.* Rejected: it leaves C++
    users where they are and keeps three bindings to keep in step.

### D4. Capability check and routing

- The check is one function:
  `check(circuit, features, engine) -> std::vector<Refusal>`, reading
  the device × engine and feature × engine table that
  `add-engine-conformance-matrix` introduces. This change does not
  define that table.
- `engine='auto'`: an explicit `dt` requests the fixed step; without
  `dt` the variable-step engine is used when the check is empty;
  otherwise the run is refused with the reasons and the advice to pass
  `dt` (today's behaviour at PY:2166-2176, kept).
- An explicit engine that fails the check raises `UnsupportedError`
  with every refusal attached, not only the first.
- The plan records `engine`, `engine_asked` and `route_reasons`; the
  result exposes them as `engine_used` and `engine_route_reason`, as
  today.

### D5. Errors

```cpp
namespace pulsim {
// A mix-in, not a std::exception: each class keeps the std base its call
// sites throw today, so `catch (const std::invalid_argument&)` still works.
class PulsimError {
public: virtual ~PulsimError() = default;
        virtual const char* message() const noexcept = 0;
        ErrorCode code() const; std::span<const Diagnostic> context() const; };
class NetlistError     : public std::invalid_argument, public PulsimError {};  // names device / node / field
class OptionsError     : public std::invalid_argument, public PulsimError {};  // names the option and the value
class UnsupportedError : public std::invalid_argument, public PulsimError {};  // {engine, feature, device, kind};
                                                                               // defined by add-engine-conformance-matrix
class SimulationAborted: public std::runtime_error,    public PulsimError {};  // carries the partial result (exists today)
class ConvergenceError : public SimulationAborted {};  // t, iterations, worst row name
class SingularError    : public SimulationAborted {};  // explain_singular row names
class Cancelled        : public std::runtime_error,    public PulsimError {};  // exists today
}
```

`UnsupportedError` is not redefined here: Phase 0 creates it (deriving
from `std::invalid_argument`, Python `ValueError`, with `engine`,
`feature`, `device`, `kind`). This change adds the `PulsimError` mix-in
to it and, in Python, the `NotImplementedError` base for the two sites
that raise that today.

Python classes, with bases that keep today's `except` clauses working:

| Class | Bases |
|---|---|
| `PulsimError` | `Exception` |
| `NetlistError` | `PulsimError`, `ValueError` |
| `OptionsError` | `PulsimError`, `ValueError` |
| `UnsupportedError` | `PulsimError`, `ValueError`, `NotImplementedError` |
| `SimulationAborted` | `PulsimError`, `RuntimeError` (as today) |
| `ConvergenceError`, `SingularError` | `SimulationAborted` |
| `Cancelled` | `PulsimError`, `RuntimeError` (as today) |
| `NameNotFoundError` | `NetlistError`, `KeyError` |

- `OptionsError` is an addition to the four classes the audit lists; a
  bad `rtol` is not a netlist problem.
- The conversion is by call path, not by search-and-replace: first
  every raise reachable from `simulate()` (routing, validation,
  preflight, resume), then the builder, then the analysis modules. The
  406 `ValueError` raises are not all converted in this change; a
  ratchet test records the count and forbids it from rising.
- Broad handlers: the 13 in PY are each replaced by a typed catch or
  removed. For the 68 in the package, the same ratchet applies, plus an
  `ast` test that no handler without a re-raise wraps a kernel call in
  PY. The owner's note warns that a handler one call deeper passes a
  shallow scan, so the test walks the call graph of `simulate()` one
  level into the package.

- Alternative considered: error codes on a result object instead of
  exceptions. Rejected: the existing culture and every test use
  exceptions, and `SimulationAborted` already carries the partial
  result.

### D6. Diagnostics

```cpp
enum class Severity : std::uint8_t { Info, Warning, Error };
struct Diagnostic { Severity severity; DiagCode code; std::string message;
                    std::string device; std::optional<Real> t; };
```

- Planning diagnostics (preflight ties, deprecated keyword, routed
  engine) and run diagnostics (forced accepts, chatter breaks, gate
  under-sampling, implausible voltage, dt retries, inductor guard
  actions) go into one list on the result, in order of occurrence.
- Python: `_emit_diagnostics(result)` calls `warnings.warn` once per
  warning-severity diagnostic with the category and text used today, so
  every `pytest.warns` test passes unchanged. `result.diagnostics` is
  the API for tools; PulsimGUI reads it instead of catching.
- The existing records (`dt_retries`, `event_iteration_breaches`,
  `inductor_guard_actions`, `_preflight`) stay as typed fields; a
  diagnostic references them by code.

## Performance constraints

- Planning cost is per run, not per step. The fixed overhead of
  `simulate()` (audit row 21: 0.3–0.5 ms at n = 5, 7.9 ms at n = 1502,
  dominated by the cache build) is measured before and after each stage
  and must not rise.
- Nothing in this change runs inside a step loop.
- With native gate sources and no Python handlers, `Simulation::run()`
  holds no Python object and the binding releases the GIL for the whole
  run, so thread-pool sweeps (`update-sweep-threading-and-import-time`)
  scale.
- `Simulation` owns its workspaces; two `Simulation` objects on two
  threads share nothing mutable. The thread-local scratch pools remain.

## Risks / Trade-offs

- **Stage A moves 1 190 lines.** → It is done block by block (one PR
  per row of the table in the Migration Plan), each bit-identical on the
  goldens and warning-identical on the suite.
- **Typed errors with builtin bases can hide a missing conversion**
  (`except ValueError` still catches). → Intended for one major
  version; the ratchet and the documentation point to the new classes.
- **One undriven-switch policy changes scripts that rely on the
  all-CLOSED default.** → The current warning already announces it
  ("In Pulsim v2.0 this becomes an error"); a migration note gives the
  one-line fix (`UndrivenSwitchPolicy.Closed` or an explicit schedule).
- **A facade over three drivers can drift from them.** → The facade
  owns no numerics; the conformance matrix runs through it once the
  wrapper lands, so a cell that the old path refused and the facade
  accepts (or the reverse) fails CI.
- **Callers inside the package** (`steady_state.py`, `sweep.py`,
  `adaptive.py`, `mmc.py`, `fra.py`) call `simulate()` with engine
  keywords. → They are unaffected by Stage A; they move to `RunOptions`
  one module at a time after Stage B.

## Migration Plan

Precondition: `add-engine-conformance-matrix` merged.

**Stage A — Python, no behaviour change.** One PR per row; the
invariant for all of them: the full suite passes with identical
warnings, and the fixed goldens are bit-identical.

| # | Block moved into `_plan_run` | Lines today |
|---|---|---|
| A1 | Tests first: `test_plan_resolves_every_kwarg`, the engine-branch `ast` test (expected to fail until A7), the broad-except and `ValueError` ratchets | — |
| A2 | `RunPlan` dataclass; `LoadedCircuit` resolution; `solver=` merge; keyword validation; `resolved_from` | PY:1815-1900 |
| A3 | Preflight (ties returned, applied by `_apply_ties`); `resume_from` validation; initial condition | PY:1901-1988 |
| A4 | Routing and the capability check (`_trbdf2_blockers`, `_dsed_unsupported`) | PY:2010-2030, PY:2134-2195, PY:3177-3289 |
| A5 | The composition of loops, observers and gate drives (one function after `fix-control-tick-semantics`; PY:2215-2283 and PY:2420-2455 on this branch) becomes part of the plan; one undriven-switch function that keeps today's three behaviours keyed by engine | PY:2215-2283, PY:2304, PY:2420-2455, PY:2664-2673, DD:617-630 |
| A6 | `_finalize`: handles, `engine_used`, route reason, voltage sanity, stats warnings; voltage sanity on `dsed` in its own PR (**BREAKING**) | PY:2340-2362, PY:2900-3006 |
| A7 | The three branches become `_run_dsed(plan)`, `_run_trbdf2(plan)`, `_run_pwl(plan)`; `simulate()` is plan → table → finalize | whole function |

**Stage B — kernel.**

| # | Step | Invariant |
|---|---|---|
| B1 | Python error classes with builtin bases; convert every raise on the `simulate()` path; replace the 13 broad handlers in PY | Suite green (existing `pytest.raises(ValueError)` still pass); ratchets hold |
| B2 | `Diagnostic` in Python; `result.diagnostics`; the 14 `warnings.warn` sites in PY go through `_emit_diagnostics` | Identical warnings |
| B3 | C++ `sim::RunOptions`, defaults table, C++ error hierarchy with pybind translators, `sim::Diagnostic`; `SimulationOptions` / `TrBdf2Options` built from it; remove sentinels and the four dead fields (**BREAKING**: `0` sentinel, raw binding tolerances) | Goldens bit-identical; defaults test reads one table |
| B4 | `sim::Simulation` facade over the existing drivers: preflight, capability check, routing, initial condition, device stamps, cache, voltage sanity; `pulsim.Simulation` binding | A C++ test runs a buck with a Shockley diode with no Python and no refresh argument and matches `simulate()` bit for bit; conformance matrix through the facade equals the matrix through `simulate()` |
| B5 | `simulate()` calls `Simulation` for runs with no Python callbacks, then for all runs through the attachment adapters; `_plan_run` shrinks to keyword → `RunOptions` | Goldens bit-identical; fixed overhead not above baseline |
| B6 | One undriven-switch policy (**BREAKING**) | Migration note; tests for the three policies on every engine |
| B7 | C-block hold as a tick handler in the kernel; package callers (`steady_state`, `sweep`, `adaptive`, `fra`) use `RunOptions` | Their tests unchanged |
| B8 | Deprecate the raw `run_transient*` bindings and free functions; `simulate()` body ≤ 150 lines (test) | Deprecation warnings only |
| B9 | One release later: remove the deprecated entry points and the Stage A helpers that B5 emptied | Conformance matrix; suite |

Rollback: Stage A PRs revert individually. In Stage B, `simulate()`
keeps a private switch to the Stage A path until B5 has been on `main`
for one nightly harness cycle.

## Open Questions

1. Undriven controlled switch: refuse on every engine (proposed, and
   what the current warning announces), or keep a default (`Open`)?
2. Default tolerances: adopt the `simulate()` values (1e-5 / 1e-8 for
   the MNA variable-step path, 1e-6 / 1e-9 for the event-driven path) as
   the single table, or one pair for all strategies?
3. Is `OptionsError` wanted as a fifth class, or should bad options be
   `NetlistError`?
4. Should the MMC Thevenin driver move into the kernel in this change
   (the audit lists it; `mmc.py` was not audited), or stay in Python
   behind the attachment interface?
5. How long do the builtin bases (`ValueError`, `NotImplementedError`)
   stay on the typed errors: until 3.0?
6. Should `simulate()` keep accepting the flat engine-specific keywords
   indefinitely, or deprecate them in favour of
   `simulate(circuit, options=RunOptions(...))`?
