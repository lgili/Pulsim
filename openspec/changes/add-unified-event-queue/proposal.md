## Why

"When does something happen" is answered separately by every engine and
by the Python layer, from an interface that hides the answer: the gate
drive is an opaque `t -> mask` callable. Counted on this branch
(`core/…` = `core/include/pulsim/…`, TB = `core/solver/trbdf2_transient.hpp`,
RT = `core/solver/run_transient.hpp`, PY = `python/pulsim/__init__.py`,
BN = `python/bindings.cpp`):

- **Three gate-edge locators.** TB:960-1190 samples the callable five
  times per step, learns the pulse width and bisects each edge;
  `python/dsed_bindings.cpp:174-258` polls every `dt_max/10` and
  bisects; `sources/switch_schedule.hpp:59-64` has the analytic
  `next_edge_after`. The fixed engine has none: it samples the mask at
  the step end (RT:892, RT:956).
- **Three detections of a native schedule in the binding, not equal.**
  BN:2598-2636 (`SwitchSchedule`, `NativePwm2Switch`,
  `NativeMultiMaskPwm`), BN:4016-4047 (no `NativeMultiMaskPwm`),
  `python/dsed_bindings.cpp:749-815` (all three).
- **Four diode decision loops**: the fixed-step event iteration
  (RT:930-1040, with its own 64-entry mask-cycle detector), the TR-BDF2
  settle / crossing / Illinois / chatter code (TB:839-900, TB:1375-1710),
  the DSED scheduler, and the DC operating point.
- **Three throttle idioms for control ticks**:
  `python/pulsim/control.py:1350-1355`, `python/pulsim/c_block.py:387-390`,
  `blockchain/chain.hpp:293-298` (every step at `chain_dt`). Only TB
  schedules ticks (`observer_period`, TB:1063-1078), and it invalidates
  its edge cache at every tick because the callable may have changed.
- **Two compositions of loops and observers in `simulate()`**, diverged:
  PY:2215-2283 (variable step; ticks every observer at
  `min(periods)`, PY:2253; composes masks with `|`, PY:2281) and
  PY:2420-2455 (fixed step; `make_combined_switch_fn`).
- **Source discontinuities are not events.** TB caps `h` at `period/20`
  and `pulse_width/3` (TB:361-392) instead of landing on the edge; the
  fixed engine samples sources on the grid.

Measured consequences (audit `raw/arch.md` §0, `raw/numerics.md`,
`raw/perf.md`; probes `arch/probe6.py`, `arch/probe7.py`,
`numerics/p1_edge.py`):

| Symptom | Cause above | Stop-gap change |
|---|---|---|
| D5: two closed loops without `dt` raise `TypeError` at PY:2281 | second composition | `fix-control-tick-semantics` |
| D6: a 2 kHz PI beside a 20 kHz monitor ticks 400 times instead of 40; V_out 13.03 V → 12.09 V | ticks at `min(periods)` | `fix-control-tick-semantics` |
| D7: `controller_period` ignored on pwl (2001 calls against 20) | throttle idiom, no schedule | `fix-control-tick-semantics` |
| D10: `NativeMultiMaskPwm` loses exact edges on trbdf2 | second detection | `add-engine-conformance-matrix` |
| Fixed engine fires 198 of 200 ticks at 10 kHz, dt = 2 µs | throttle drift | — |
| A one-shot gate pulse narrower than `h_max/4` can be stepped over with no signal (TB:121-128) | sampling an opaque callable | — |
| DSED moved every edge onto a 1 µs poll grid: 14.28 V against 11.49 V | poll + bisect | fixed for native schedules only |
| TR-BDF2 on a pulse-driven RC floors at 1.8e-4 V on 10 V at any `rtol` | no source breakpoints | `update-trbdf2-error-control` |
| A Python `switch_fn` costs 2522 ns/step against 393 ns (6.4×) and is called twice per step | the callable is the interface | `update-fixed-step-loop-bookkeeping` (one call) |

## What Changes

- **One `EventQueue` per run** for time events of five kinds: gate
  edge, control tick, source breakpoint, output sample, end of run. It
  is a k-way merge over registered sources, each of which answers
  `next_after(t)`; times are computed on integer counters
  (`t0 + k·T`), never accumulated.
- **A fixed order for events at the same instant**: record the left
  limit, fire control ticks, apply gate edges, apply source
  breakpoints, settle the self-commutated devices, record the right
  limit.
- **Declarative gate sources** composed by switch index:
  `CarrierPwm` (carrier, duty register, dead time), `EdgeList`,
  `ScheduleGate` (today's `SwitchSchedule`, `NativePwm2Switch`,
  `NativeMultiMaskPwm`), each with an analytic `next_edge(t)`; and
  `SampledGate`, the adapter for an opaque Python callable, with a
  declared time resolution.
- **Duty registers.** A control tick writes a register; only the
  modulators reading a changed register recompute their next edge. The
  Python callable runs at ticks only.
- **Per-rate control ticks.** Each observer, closed loop, C block and
  `BlockChain` registers `Rate{period, offset}`; a tick fires at every
  `t0 + offset + k·T` earlier than `t_end`, on every engine.
- **Source breakpoints** from the source parameters (pulse, PWM) as
  time events.
- **State events**: one `EventLocator` (Illinois) parameterised over a
  trial evaluator, and one `ModeSet::redecide` used by every decision
  site.
- **One binding function** `to_gate_source(obj)`: the shared
  switch-function adapter that `add-engine-conformance-matrix`
  introduces for the three binding sites, returning a gate source
  instead of a callable plus an optional edge oracle.
- **BREAKING (results):** on `engine='dsed'`, an opaque `switch_fn` is
  located to the declared resolution instead of the `dt_max/10` poll
  grid. On every engine an opaque callable whose pulses are narrower
  than its resolution raises a diagnostic instead of passing silently.
- **BREAKING (kernel API):** the `next_edge_fn` and `observer_period`
  parameters of `run_transient_trbdf2`, and the callable-only
  `SwitchScheduleFn` parameter of the drivers, are replaced by the
  queue; the old parameters stay for one release as adapters.
- **Deleted at the end:** TB:960-1190 (sampling and bisection),
  `python/dsed_bindings.cpp:174-258`, any binding detection that Phase 0
  left, the three throttle idioms, the mask-and-observer composition in
  Python (PY:2215-2283 and PY:2420-2455 on this branch; one function
  after `fix-control-tick-semantics`), the 64-entry `masks_seen`
  detector as a separate implementation.

## Impact

- Affected specs: `event-scheduling` (new).
- Affected code: new `core/include/pulsim/events/`;
  `sources/{switch_schedule,native_pwm*,pulse_b_extra,pwm_b_extra}.hpp`;
  `solver/{run_transient,trbdf2_transient}.hpp`; `dsed/scheduler*.hpp`,
  `dsed/event_predictor.hpp`; `pwl/diode_event_state.hpp`,
  `pwl/dc_operating_point.hpp`; `blockchain/chain.hpp`;
  `python/bindings.cpp`, `python/dsed_bindings.cpp`;
  `python/pulsim/{__init__,control,c_block,blockchain,_dsed_dispatch}.py`;
  stubs; docs.
- Depends on: `add-engine-conformance-matrix` (the schedule × engine and
  feature × engine cells are the safety net).
- Carries over, when present: the tick semantics of
  `fix-control-tick-semantics`, the breakpoint table of
  `update-trbdf2-error-control`, the single `switch_fn` call of
  `update-fixed-step-loop-bookkeeping`, and the event splitting of
  `add-fixed-step-event-splitting` (which becomes a consumer of the
  queue's edge times).
- Unblocks: `refactor-unified-transient-stepper`,
  `add-in-kernel-event-losses`, `add-compiled-control-graph` (its tick
  handlers and modulator outputs are the extension points defined
  here).
- Performance constraints: no allocation after setup (fixed-size heap
  of sources); no mask evaluation between edges for an analytic source;
  no Python call except at ticks and inside `SampledGate`. Expected
  gain, an estimate from the audit's time split (rows 16–18): a
  closed-loop buck goes from 1.3–2.5 µs/step to about 0.4 µs/step;
  `tasks.md` carries the before/after measurement.
- Results for analytic schedules on the fixed and the variable-step
  engine are unchanged at every migration step (bit-identical fixed
  goldens, identical TR-BDF2 step counters).

## Re-centred scope, 2026-11-13

> Appended; the sections above stand except where named.

The queue becomes the driver of the one kernel
(`refactor-unified-transient-stepper`, re-centred scope D8): the kernel
advances exactly to the queue's next event, and every strategy
(exact propagation, the trapezoid with the SDIRK restart, TR-BDF2)
advances a segment between two events. Two changes to this change's
plan follow:

- **Recorded output samples and controller ticks are queue events.** A
  sample is a time event that records the state and changes nothing
  else; on the exact strategy it is an evaluation of the flow, not a
  step (`add-exact-propagator-strategy` D15). Ticks already are events
  (D4).
- **The mode decision at an event is the complementarity service**
  (`add-complementarity-mode-selection`), called once per event
  instant after the instant's gates and sources are applied. This
  supersedes `ModeSet::settle` (D6) as the decision rule at the default
  flip; `settle` stays as the legacy policy until the settle loops are
  deleted. The state-event locator (one Illinois, three trial
  evaluators) is unchanged.

Before the flip, the requirement "One State-Event Locator And One Mode
Decision" is met by `settle`; after it, by the selection service. Its
scenario "Mask cycle is reported once, the same way" applies to the
legacy policy only: the selection service has no cycles to report.
