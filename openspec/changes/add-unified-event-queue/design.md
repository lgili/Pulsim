## Context

Paths: `core/…` = `core/include/pulsim/…`; RT = `core/solver/run_transient.hpp`,
TB = `core/solver/trbdf2_transient.hpp`, PY = `python/pulsim/__init__.py`,
BN = `python/bindings.cpp`, DB = `python/dsed_bindings.cpp`.

Today a run has four independent notions of an event:

| Event | Fixed (`pwl`) | Variable (`trbdf2`) | Event-driven (`dsed`) |
|---|---|---|---|
| Gate edge | mask sampled at the step end (RT:892, RT:956); quantised to `dt` | analytic oracle if the binding recognised the object, else 5 samples per step plus bisection (TB:960-1190) | analytic for three native types, else poll at `dt_max/10` plus bisection (DB:174-258) |
| Control tick | observer called every step; the observer throttles itself | scheduled at `k·T` (TB:1063-1078), one period for all observers | refused |
| Source breakpoint | sampled on the grid | step capped at `period/20`, `width/3` (TB:361-392) | not checked |
| Diode commutation | re-solve the step until the mask repeats (RT:930-1040) | sign change of `crossing_signal`, Illinois on trial solves (TB:1375-1710) | analytic predicate plus Illinois |
| Output sample | every step, or every `store_every`-th | every accepted step | every accepted step |

What is already right and must survive: `SwitchSchedule::next_edge_after`
(a returned edge may be a no-op, never a missed one), the grid-exact
`pwm_phase` arithmetic, the integer step counter `t = t0 + k·dt`, the
single diode law `SwitchedDiode::decide` / `crossing_signal`, the
contract that an observer fires before the mask of the coming step is
read, and the honest records (`n_gate_undersampled`, forced accepts,
event breaches).

Owner notes that are requirements here: *simulate-engine-dispatch-order*
(composition must not live in an engine branch),
*switchstatemask-takes-a-size* (compose by index, never by treating a
size as a pattern), *time-weight-averages-on-adaptive-grids* (events
create clustered samples; the left and right limits must both be
representable).

## Goals / Non-Goals

Goals:

- One place that knows every instant at which the circuit, its drive or
  its observers change discontinuously.
- Gate drives that state their edges; an opaque callable is an adapter
  with a declared resolution and a diagnostic, never a silent sampler.
- Exact, per-rate control ticks on every engine.
- One state-event locator and one mode-decision loop.
- A migration in which the fixed goldens stay bit-identical and the
  variable-step step sequence stays identical for analytic schedules.

Non-Goals:

- No change to what a step does at an event. The restart rule and the
  fixed-step event splitting belong to `add-discontinuity-restart-rule`
  and `add-fixed-step-event-splitting`; the TR-BDF2 post-event restart
  to `update-trbdf2-post-event-restart`. This change supplies the event
  times and the ordering they use.
- No control graph (`add-compiled-control-graph`). A tick handler here
  is a callback slot; the graph becomes one handler.
- No event-triggered loss injection (`add-in-kernel-event-losses`); the
  hook is named below.
- The step loop itself is unified by `refactor-unified-transient-stepper`.

## Decisions

### D1. Time events are a k-way merge over sources

```cpp
namespace pulsim::events {

enum class Kind : std::uint8_t {         // also the order at one instant
    OutputSample = 0, ControlTick = 1, GateEdge = 2, SourceBreakpoint = 3, EndOfRun = 4 };

struct TimeEvent { Real t; Kind kind; std::uint32_t source; std::uint64_t ordinal; };

// A source of time events. `next_after` is pure and strictly increasing in t.
template <class S>
concept TimeSource = requires(const S& s, Real t) {
    { s.kind() }        -> std::same_as<Kind>;
    { s.next_after(t) } -> std::convertible_to<Real>;   // +inf when none
    { s.lands() }       -> std::convertible_to<bool>;   // must a step end here?
};

class EventQueue {
public:
    template <TimeSource S> std::uint32_t add(S source);       // setup only; allocates
    void  start(Real t0, Real t_end);
    Real  next_landing(Real t) const noexcept;                 // earliest event that must end a step
    std::span<const TimeEvent> take_at(Real t) noexcept;       // all events at t, in Kind order
    void  invalidate(std::uint32_t source) noexcept;           // its next_after changed (duty written)
    const EventCounters& counters() const noexcept;            // n_gate_events, n_ticks, ...
};
}
```

- Storage: a fixed vector of type-erased sources (erased once, at
  `add`; the erased call is one function pointer per *event*, not per
  step) and a binary heap holding one pending event per source. Taking
  an event refills that source's slot. Nothing allocates after `start`.
- Equality of times: two events are simultaneous when their times
  differ by less than `time_eps(t)` (the existing `dsed/time_eps.hpp`
  rule, one relative epsilon); the earlier `Kind` goes first.
- Periodic sources compute `t0 + k·T` from an integer `k`. No source
  accumulates `t += T`.
- `lands()`: gate edges, ticks, breakpoints and the end of the run cut
  the step on a variable-step controller. Output samples cut it only
  when the recorder asks for exact samples (`add-probes-and-recorders`).

- Alternatives considered.
  - *A sorted list of all events built before the run.* Rejected: a
    closed loop changes future edges at every tick, and a 1 s run at
    100 kHz with dead time is 4·10⁵ edges per leg.
  - *Each engine keeps its own `min()` over its own candidates* (what TB
    does at TB:1100-1135). Rejected: it is the duplication being
    removed, and it is why the fixed engine has no ticks.
  - *A general discrete-event simulator with a priority queue of
    closures.* Rejected: allocation per event and no static knowledge of
    kinds, which the ordering rule and the counters need.

### D2. Order at one instant

At an instant `t` with pending events, in this order:

1. `OutputSample`: record `x(t⁻)` under the mask that produced it (the
   left limit).
2. `ControlTick`: every handler whose rate fires at `t`, in
   registration order, reads `x(t)` and may write duty registers and
   source terms. Handlers of different rates never see each other's
   skipped ticks.
3. `GateEdge`: gate sources apply their bits. Sources whose register
   changed in step 2 were invalidated and have recomputed their edge.
4. `SourceBreakpoint`: sources take their post-break value.
5. State settle: `ModeSet::redecide` under the new mask until stable
   (zero-time cascade), as the engines do today after an edge.
6. If anything in 3–5 changed the mask or a source: record the right
   limit `x(t⁺)` when the recorder keeps event rows, and raise the
   `restart` flag that the Phase 2 restart rules consume.

This is today's contract ("the observer fires before `switch_fn(t)` is
sampled", RT:866-871, TB:1063) made explicit and shared.

- Alternative considered: ticks after gate edges (the controller sees
  the post-edge state). Rejected: a sampled controller reads the ADC at
  the tick and its duty applies to the coming carrier period; the
  existing tests pin this order.

### D3. Gate sources

```cpp
namespace pulsim::events {

struct DutyRegister { Real value; Real shadow; LatchPolicy latch; bool dirty; };
enum class LatchPolicy : std::uint8_t { Immediate, AtCarrierStart };

template <class G>
concept GateSource = requires(const G& g, Real t, topology::SwitchStateMask& m) {
    { g.owned_bits() } -> std::convertible_to<std::span<const Index>>;
    { g.apply(t, m) }  -> std::same_as<void>;           // writes its own bits only
    { g.next_edge(t) } -> std::convertible_to<Real>;    // strictly after t; +inf if none
    { g.resolution() } -> std::convertible_to<Real>;    // 0 = exact
};

struct CarrierPwm {       // one leg or one switch
    Real T, phase;  Carrier carrier;                    // sawtooth / triangle
    const DutyRegister* duty;  Real dead_time;
    Index hs, ls;                                       // ls = kInvalidIndex for a single switch
};
struct EdgeList     { std::vector<Edge> edges; };       // (t, bit, value), sorted; setup-time allocation
struct ScheduleGate { sources::SwitchSchedule sched; std::vector<Index> bits; };   // today's helpers
struct SampledGate  { std::function<topology::SwitchStateMask(Real)> fn;
                      std::vector<Index> bits; Real resolution; };
}
```

- **Composition is by index.** Each source declares the switch bits it
  owns. Two sources owning the same bit is a `NetlistError` naming the
  switch and both sources. The run's mask is built by letting every
  source write its bits into one mask of `num_switches` bits; there is
  no `mask | mask` and no "default mask" lambda. A controlled switch
  that no source owns is handled by the single undriven-switch policy of
  `add-kernel-simulation-object`.
- **`ScheduleGate`** wraps the existing helpers unchanged
  (`make_pwm_switch_fn`, `make_dead_time_pwm_pair_fn`,
  `make_spwm_pair_fn`, `make_phase_shift_full_bridge_fn`,
  `make_three_phase_spwm_fn`, `make_combined_switch_fn`,
  `NativePwm2Switch`, `NativeMultiMaskPwm`). Their `pwm_phase`
  arithmetic is not touched.
- **`CarrierPwm` with a `DutyRegister`** is the first member of the
  modulator family (`add-compiled-control-graph` calls it `Modulator`
  and writes the same register) and is what `bind_pi_to_switch`
  builds. A tick writes `duty->value` and sets `dirty`; the queue calls
  `invalidate` for the modulators that read it. With
  `LatchPolicy::Immediate` the mask at any `t` equals what today's
  closure returns (the closure reads the same variable), which is the
  invariant of the migration step.
- **`SampledGate`** is the only way an opaque Python callable enters.
  It finds edges by the algorithm that TB:960-1190 uses today (sample,
  learn the narrowest pulse, bisect), to a declared `resolution`.
  Defaults reproduce today: `h_max/4` on the variable-step engine,
  `dt` on the fixed one. It reports `n_gate_undersampled` and
  `gate_pulse_min`, and a pulse found to be narrower than the resolution
  is a `Diagnostic` of severity warning (error under strict mode).
- `next_edge` may return a no-op edge; it must never skip a real one
  (the `SwitchSchedule` contract, kept).

Extension points: `add-compiled-control-graph` writes `DutyRegister`s
and provides modulator blocks; `add-in-kernel-event-losses` subscribes
to `GateEdge` events with the pre- and post-edge states.

- Alternatives considered.
  - *Keep `t -> mask` and require a companion `next_edge` from the
    user.* Rejected: it is today's `SwitchSchedule`, and it does not
    compose (two schedules cannot be merged without sampling) nor let a
    tick change one duty without invalidating everything.
  - *Forbid opaque callables.* Rejected: it breaks every user script.
    The adapter keeps them working and makes their cost and resolution
    visible.

### D4. Control ticks

```cpp
struct Rate { Real period; Real offset = 0; };      // fires at t0 + offset + k·period, k ≥ 0
using TickHandler = std::function<void(Real t, const Vector& x)>;   // called at ticks only
std::uint32_t EventQueue::add_tick(Rate, TickHandler);
```

- Every `step_observer` with a period, every `ClosedLoop`, every C
  block and every `BlockChain` registers its own `Rate`. A handler
  without a period is legal only on a fixed-step controller, where it is
  registered at `Rate{dt}` (today's "every step").
- A tick fires at every `t0 + offset + k·T` (`k ≥ 0`) earlier than
  `t_end`: `ceil((t_end − t0 − offset)/T)` ticks on every engine (40 for
  a 2 kHz loop over 20 ms, the count the audit uses; `TrBdf2Stats`
  documents `floor(span/T) + 1`, see Open Questions). On the fixed engine a tick that is not on the grid
  fires at the first grid point at or after it, and the handler receives
  the scheduled tick time through `TickInfo` so that it does not measure
  its own period from step times.
- The Python throttles (`control.py:1350-1355`, `c_block.py:387-390`)
  and `BlockChain::make_step_observer(dt)` are deleted when their owners
  register rates. The unthrottled `.tick` attribute (PY:2231) becomes
  the handler.

### D5. Source breakpoints

Each time-varying source kind supplies `next_breakpoint(t)` from its
parameters: a pulse source its delay, rise, width, fall and period
corners; a PWM voltage source its carrier edges; a sine source none.
The step caps of TB:361-392 remain as accuracy caps for smooth sources;
for sources with breakpoints they are replaced by landing. When
`update-trbdf2-error-control` has already introduced a breakpoint table
in TB, that table is moved into a `TimeSource`, not rewritten.

### D6. State events

```cpp
// g(t) sampled at the two ends of an accepted step with opposite signs.
struct Bracket { Real t0, g0, t1, g1; };

template <class Trial>   // Real trial(Real t): g at t, by whatever means the strategy has
Real locate(Bracket b, Trial&& trial, Real t_tol, Real g_tol, int max_iter = 60);   // Illinois

class ModeSet {          // every self-commutated mode device of the circuit
public:
    bool redecide(const Vector& x) noexcept;       // applies decide(); true if any bit changed
    Real crossing_signal(Size i, const Vector& x) const noexcept;
    bool settle(SolveAt solve, Size max_iter, CycleReport* out);   // redecide + re-solve until stable
};
```

- One Illinois. The three trial evaluators are supplied by the
  strategies: a trial trapezoidal solve from the step start (TB today),
  the analytic trajectory (DSED today), linear interpolation between
  the step ends (`add-fixed-step-event-splitting`).
- One `settle`: the fixed-step event iteration, the DC operating point,
  the TR-BDF2 zero-time settle and boundary settle, and the DSED settle
  call it. Its cycle detection replaces the 64-entry `masks_seen` array
  (kept as the implementation, hoisted once, as
  `update-fixed-step-loop-bookkeeping` requires).
- Device `events()` of the unified interface are offered to `locate`
  exactly like `crossing_signal`.
- The chatter guard, the held-diode re-decision and the forbidden-side
  tolerance of TB stay where they are in this change; they move with
  the step loop in `refactor-unified-transient-stepper`.

### D7. One detection in the binding

```cpp
events::AnyGateSource to_gate_source(py::object obj, Size num_switches, Real default_resolution);
```

Order: `SwitchSchedule` (→ `ScheduleGate`, analytic when
`has_edges()`), `NativePwm2Switch`, `NativeMultiMaskPwm`, a Python
`Modulator` object, else any callable (→ `SampledGate`). Used by the
fixed, variable-step and DSED bindings. `add-engine-conformance-matrix`
already replaces the three detections (BN:2598-2636, BN:4016-4047,
DB:749-815 on this branch) by one shared adapter; this function is that
adapter with a gate source as its result. If Phase 0 has not landed,
step 1 does the replacement itself.

### D8. How each engine uses the queue during the migration

| Engine | Use | Invariant |
|---|---|---|
| `trbdf2` | `h = min(h, queue.next_landing(t) − t)` replaces the three separate landings; `take_at` replaces the tick block | accepted-step sequence identical for analytic schedules |
| `pwl` | the mask of step `k` is obtained from the gate sources at `t_k`; between two edges the mask is reused without evaluating anything; ticks come from the queue | fixed goldens bit-identical; `n_gate_events` equals the analytic count |
| `dsed` | the scheduler asks the queue instead of its own poll | native schedules unchanged; opaque callables land to the declared resolution |

On the fixed engine an edge inside a step is applied at the first grid
point at or after it (today's sampling). Two edges of the same bit
inside one step cancel on the grid: the queue still counts both and
raises `n_gate_events_unresolved`, with a warning diagnostic. Splitting
the step at the edge is `add-fixed-step-event-splitting`, which reads
the edge time from `next_landing`.

## Performance constraints

- No heap allocation after `EventQueue::start`: fixed-size heap, the
  event span returned by `take_at` points into a preallocated buffer.
- Cost per step with no pending event: one comparison
  (`t_next > t + h`). No mask is built and no gate source is called
  between edges.
- A Python callable is invoked only by a `TickHandler` or inside a
  `SampledGate`; with native gate sources and no observers the run
  never takes the GIL.
- The fixed engine's per-step work must not rise: the counter gate of
  the harness (`n_solves`, cache hits) is unchanged and ns/step on audit
  rows 1–2 is not above the value before the change.
- Thread safety: the queue is owned by the run; sources are immutable
  except `DutyRegister`, which is written only inside tick handlers.

## Risks / Trade-offs

- **Reordering simultaneous events changes closed-loop results.** → The
  order in D2 is today's; the invariant of each migration step is an
  identical step sequence or identical goldens, checked by the harness.
- **`Immediate` duty latching hides a modelling choice** (real PWM
  peripherals latch at the carrier start). → `LatchPolicy` is explicit;
  the default reproduces today; the other policy is opt-in and
  documented.
- **A `SampledGate` default resolution that is too coarse** keeps
  today's silent miss. → The default equals today's sampling, and the
  miss is no longer silent for pulses that are seen; a one-shot pulse
  that is never sampled cannot be detected by construction, which the
  documentation and the diagnostic text state.
- **Tick time on the fixed grid.** A tick between grid points fires up
  to `dt` late. → The handler gets the scheduled time; the count is
  exact; landing a fixed-step sub-step on the tick is left to
  `add-fixed-step-event-splitting`.
- **Type erasure cost.** → One indirect call per event, measured
  against the 380–640 ns step.

## Migration Plan

Precondition: `add-engine-conformance-matrix` merged. Each step is
shippable alone.

| # | Step | Invariant |
|---|---|---|
| 1 | `to_gate_source` in the binding (the Phase 0 adapter, returning a gate source), used by the three engine bindings; no kernel change (the result is converted back to today's `switch_fn` + `next_edge_fn`). | Goldens bit-identical; TR-BDF2 counters identical; the schedule × engine cells of the conformance matrix unchanged or improved. |
| 2 | `events/` with `EventQueue`, `ScheduleGate`, tick and end-of-run sources. TB uses it for gate, tick and end landing (TB:1063-1135). | For every harness circuit with an analytic schedule: `n_accept`, `n_reject`, `n_gate_events`, `n_ctrl_ticks` and the states identical. |
| 3 | `SampledGate`: move TB:960-1190 into it unchanged; TB calls it. Then DSED calls it instead of DB:174-258. | TB identical. DSED with an opaque callable: edges within the declared resolution of the analytic edge (**BREAKING** for that case; the sync-buck regression reads 11.49 V). |
| 4 | Per-rate ticks: handlers register `Rate` on the queue; the per-source period list that `fix-control-tick-semantics` passes to the TR-BDF2 driver and its fixed-engine tick counters are replaced by queue sources. | The tests of `fix-control-tick-semantics` (40 and 400 ticks; two loops without `dt`; `control_tick_counts`) pass unchanged on the queue. |
| 5 | Fixed engine on the queue: mask from gate sources with reuse between edges; ticks from the queue; delete PY:2420-2455 in favour of index composition. | Fixed goldens bit-identical for circuits without ticks; with ticks, identical to the `fix-control-tick-semantics` behaviour; `n_gate_events` equals the analytic count on pwl, trbdf2 and dsed. |
| 6 | `CarrierPwm` + `DutyRegister`; `bind_pi_to_switch` builds one with `Immediate` latching; Python runs at ticks only. | Closed-loop tests identical on pwl and within tolerance on trbdf2; measured ns/step before and after. |
| 7 | Source breakpoints as time sources (moving the table of `update-trbdf2-error-control` when present). | Pulse-driven RC error on TR-BDF2 decreases with `rtol` below the 1.8e-4 V floor; fixed goldens bit-identical. |
| 8 | `locate` and `ModeSet`; TB's Illinois, the DSED predictor, the fixed-step event iteration and the DC operating point call them. | `commutation_events`, `event_iteration_count` and `n_diode_events` identical on the harness; half-wave conduction instants unchanged. |
| 9 | Delete the adapters' legacy inputs after one release: `next_edge_fn`, `observer_period`, DB:174-258, the Python throttles, `BlockChain::make_step_observer`. | Conformance matrix; full suite. |

Rollback: each step reverts alone; steps 2, 5 and 8 keep the previous
code path behind a compile-time switch until the harness has run on
both for one nightly cycle.

## Open Questions

1. Default `SampledGate` resolution: keep today's (`h_max/4`, `dt`), or
   require the user to state it once a callable is opaque?
2. Should a pulse narrower than the declared resolution be an error by
   default (refusal culture) or a warning (today's `n_gate_undersampled`
   behaviour)?
3. Fixed engine, tick between grid points: this change follows
   `fix-control-tick-semantics` (fire at the first grid point at or
   after the tick, warn when the period is not a multiple of `dt`,
   refuse a period shorter than `dt`). Should the queue instead cut a
   fixed step at the tick once `add-fixed-step-event-splitting` exists?
4. `LatchPolicy` default for `bind_pi_to_switch`: `Immediate` (today's
   behaviour) or `AtCarrierStart` (what a real PWM peripheral does),
   with the change of result that implies?
5. Is the DSED change of result for opaque callables (step 3)
   acceptable in a minor release?
6. A tick that falls exactly on `t_end` does not fire (half-open
   `[t0, t_end)`, the rule of `fix-control-tick-semantics`: 40 ticks for
   2 kHz over 20 ms). The comment at `solver/trbdf2_transient.hpp:180-183`
   says `floor(span/T) + 1` and should be corrected there; confirm.
