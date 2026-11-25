## Context

Paths: PY = `python/pulsim/__init__.py`, DD = `python/pulsim/_dsed_dispatch.py`,
`core/…` = `core/include/pulsim/…`.

Today:

- `SimulationResult` (`core/solver/result.hpp:208-…`) holds `times`
  (a `std::vector<Real>`), `states` (a contiguous `StateTrajectory`,
  one row of all unknowns per stored step), the packed mode record
  (`switch_state_bits`, with an "unsolved" list), the fixed engine's
  records (`dt_retries`, `inductor_guard_actions`,
  `event_iteration_breaches`, `event_iteration_count`,
  `commutation_events`) and `final_snapshot`.
- The variable-step engine returns the same type on the accepted grid,
  with its statistics in a separate struct that the binding turns into
  a dict stored as `res._trbdf2_stats`.
- The DSED path cannot construct that type from Python, so it returns
  `_PEDSimulationResult`, a mimic with the same accessor names plus
  right-limit rows at events.
- Everything else is derived in Python after the run:
  `result.v(name)` (a column), `result.i(name)` (the kind ladder),
  `device_loss_summary`, `device_thermal_summary`, `stats.py`.
  `_result_views.py` already holds the right pieces: `pl_integral`
  (exact integral of a piecewise-linear signal),
  `pl_integral_product`, `step_start_states` (start each interval from
  the right limit where one exists), `recorded_switch_states`.
- Streaming: `live_stream` feeds a lock-free `LiveRing`
  (`core/streaming/live_ring.hpp`) from the fixed engine only.

What is well designed and stays: the contiguous `StateTrajectory`, the
pre-reserved buffers (about 35 ns/step), the mode record with its
solved / unsolved distinction, the exact per-step integrals in the loss
post-processing, `stats.py`'s time-weighted measures, the `LiveRing`.

Owner notes that are requirements:

- *time-weight-averages-on-adaptive-grids*: an average, RMS or integral
  over a variable-step result is time-weighted; when two engines
  disagree, check the metric first.
- *pwl-diode-vth-is-not-a-drop* / the `V_f` change: a device current is
  the device law under the stamped mode, one law for every consumer.
- *switchstatemask-takes-a-size*: a probe of a switched branch is
  meaningless without the mode it was solved under; the record must
  carry it.
- *gui-swallows-backend-refusals*: a quantity that cannot be produced is
  a typed refusal, not a silent zero column.

## Goals / Non-Goals

Goals:

- The quantities users read (currents, powers, internal variables) are
  computed in the kernel, once, from the committed step, for every
  device kind and engine.
- The user chooses what is stored and when.
- One result type that knows its grid and its events, with
  time-weighted measures as methods.
- Default behaviour and stored bytes unchanged.

Non-Goals:

- No change to loss or thermal models. `losses.py` and `thermal.py`
  keep their algorithms and read currents from probes when present;
  in-kernel event losses are `add-in-kernel-event-losses`.
- No expression probes (`p = v(a)·i(R1) + …`) in this change; the plan
  format leaves room for them.
- No higher-order dense output. Requested times are met by landing a
  step on them or by linear interpolation, stated in the result.
- No file formats (HDF5, CSV streaming); the sink interface is the
  extension point.
- No change to the scope GUI beyond reading the new result.

## Decisions

### D1. Probes

```cpp
namespace pulsim::record {

enum class ProbeKind : std::uint8_t {
    Unknown,          // x[index]
    NodeVoltage,      // v(node)
    DiffVoltage,      // v(a) − v(b)
    PinCurrent,       // i(device, pin)   — current into the device at that pin
    DevicePower,      // Σ v_k·i_k
    DeviceOutput,     // outputs()[channel] of the unified device interface
    ModeBit,          // 0 / 1
};

struct ProbeDecl { std::string name; ProbeKind kind; std::string target;   // "M1", "out", "a,b"
                   std::string channel; std::string unit; };               // "drain", "q_M", …

struct ProbePlanEntry { ProbeKind kind; std::uint32_t a, b; std::uint32_t column; };

class ProbeSet {
public:
    static ProbeSet compile(const Circuit&, std::span<const ProbeDecl>);   // resolves names; NetlistError if unknown
    void evaluate(const StepView&, std::span<Real> row) const noexcept;    // flat loop over the plan
    std::span<const ProbeInfo> columns() const noexcept;                   // name, unit, kind
};

struct StepView {            // the committed step, handed to recorders
    Real t; std::span<const Real> x; std::span<const Real> dev_state;
    const topology::SwitchStateMask* mask; bool mask_solved;
    const device::StepCtx* ctx;            // formula and step of the solve that produced x
    std::span<const Real> pin_currents;    // i_k for devices with a q row, filled at commit
};
}
```

- Names follow the accessors users already type: `v(out)`, `v(a,b)`,
  `i(R1)`, `i(M1.drain)`, `p(D1)`, `M1.q_gate`, `S1.on`. A two-pin
  device's `i(name)` is the current at its first pin, with today's sign
  convention (the existing tests pin it).
- `PinCurrent` and `DevicePower` use the generic outputs of
  `add-unified-device-interface` (design D7): `i_k = f_k + (a·q_k + r_k)`
  for a Newton device, `g·v − i_offset` under the stamped mode for a
  mode device, the stamped companion law for a static device. They are
  evaluated at commit, when `q̇` is known.
- Where probes are declared:
  - on the circuit: `builder.add_probe("i(M1)")`, YAML
    `probes: ["i(M1)", {name: vds, expr: "v(d,s)"}]`; they travel with
    the circuit;
  - on the run: `RecordOptions.what = Probes{names…}`.
- Post-hoc evaluation: a probe that is a function of the stored row
  and the mode record alone (`Unknown`, `NodeVoltage`, `DiffVoltage`,
  `ModeBit`, and the current and power of an `f`-only or mode device)
  can be evaluated after the run from a full-state record by the same
  `ProbeSet::evaluate`. A probe that needs `q̇` cannot; asking for it
  after a run that did not record it raises `UnsupportedError` naming
  the probe to add. With `what = all` (the default) the kernel records
  the pin currents of devices with a `q` row alongside the state, so
  every `i()` works after a default run.

- Alternatives considered.
  - *Keep reconstruction in Python and extend the ladder.* Rejected: it
    is per kind, per engine and inexact for `q` devices; it is the
    reason five kinds raise today.
  - *Record every device's current always.* Rejected: memory grows by
    one column per branch; most are recomputable exactly from the row.
  - *Sense devices (zero-volt sources) as the only way to read a
    current.* Rejected: it changes the matrix to observe it, and is the
    "add a 1 mΩ resistor" advice in another form.

### D2. Recorders

```cpp
struct All {};                                   // every unknown (+ pin currents of q devices)
struct Probes { std::vector<std::string> names; };
using Selection = std::variant<All, Probes>;

struct EveryStep {};
struct Decimate { Size n; };                     // every n-th step, plus event rows
struct Times    { std::vector<Real> t; bool land = true; };   // t_eval
struct Period   { Real T; Real offset = 0; bool land = true; };
using Sampling = std::variant<EveryStep, Decimate, Times, Period>;

struct RecordOptions {
    Selection what = All{};
    Sampling  when = EveryStep{};
    bool      event_rows = true;                 // left and right limits at discontinuities
};

template <class S>
concept Sink = requires(S& s, const RowView& r) {
    { s.begin(std::span<const ProbeInfo>{}, GridKind{}) };
    { s.row(r) };                                // t, values, mask, flags
    { s.end() };
};

class Recorder {                                 // owned by the run
public:
    void on_step(const StepView& left);          // called once per accepted step
    void on_event(const StepView& right);        // after the settle at an event instant
};
```

- `Decimate{n}` on the fixed grid is today's `store_every` (uniform
  stride, same rows). On the accepted grid it keeps every n-th accepted
  step and never drops an event's two rows; the result's grid kind says
  `Accepted`, so no consumer can mistake it for uniform. This is what
  lets `store_every` stop being refused on the variable-step engine.
- `Times` / `Period` with `land = true` register `OutputSample` time
  events on the queue of `add-unified-event-queue`; the variable-step
  controller ends a step on each, so the sample is a solved state, not
  an interpolation. With `land = false`, or on the fixed grid when a
  time is not a grid point, the row is linearly interpolated between
  the two neighbouring accepted states (starting from the right limit
  when the left neighbour is an event) and flagged `interpolated`; the
  error is second order, the order of the methods.
- Sinks: `MemorySink` (the result's table, pre-reserved from the
  sampling policy; chunked geometric growth on the accepted grid),
  `RingSink` (wraps the existing `LiveRing`, on any engine). A user
  sink can be attached through `Simulation`.
- Until `refactor-unified-transient-stepper` lands, `on_step` is called
  at the existing push sites (four in `run_transient.hpp`, three in
  `trbdf2_transient.hpp`, two in `bdf1.hpp`); afterwards from the one
  `hooks.on_step`.

- Alternatives considered.
  - *Dense output by Hermite interpolation of the integrator.* Deferred:
    TR-BDF2 has a continuous extension, but derivatives are available
    only for the differential variables; landing is exact and simple.
  - *Decimation that drops event rows.* Rejected: an integral over a
    decimated record would then be wrong by the jump at each dropped
    event.

### D3. One result type

```cpp
enum class GridKind : std::uint8_t { Uniform, Accepted, Requested };

struct RunStats {                 // engine-independent; unused counters are zero
    Size n_steps, n_accept, n_reject, n_solves, n_factorizations;
    Size n_gate_events, n_state_events, n_ctrl_ticks;
    Size n_newton_iterations, n_newton_retries, n_forced_accepts, n_chatter_breaks;
    Size n_gate_undersampled; Real gate_pulse_min, gate_sample_spacing_max;
    Real cpu_time_seconds;
};

struct EventRows {                // right limits
    std::vector<Size> sample_index;      // row k of the table is the left limit
    StateTrajectory   values;            // same columns as the table
    ModeRecord        masks;             // mask the next step starts from
};

class SimulationResult {
public:
    SimulationResult();                                   // constructible; adapters fill it
    GridKind grid() const;
    std::span<const Real> times() const;                  // zero-copy to NumPy
    const Table& table() const; std::span<const ProbeInfo> columns() const;
    const EventRows& events() const;  const ModeRecord& modes() const;
    const RunStats& stats() const;    std::span<const Diagnostic> diagnostics() const;
    const SolverSnapshot& final_snapshot() const;
    std::string engine_used, engine_route_reason;
    // records of the fixed controller, kept as typed fields
    std::vector<DtRetry> dt_retries; std::vector<InductorGuardAction> inductor_guard_actions; …

    // measures: time-weighted, right-limit aware
    Real avg(std::string_view probe, Real t0 = -inf, Real t1 = +inf) const;
    Real rms(std::string_view probe, Real t0 = -inf, Real t1 = +inf) const;
    Real integral(std::string_view probe, Real t0 = -inf, Real t1 = +inf) const;
    Real at(std::string_view probe, Real t) const;        // linear, right limit at an event
    std::pair<Real, Real> minmax(std::string_view probe, Real t0, Real t1) const;
};
```

- Python keeps every accessor that exists: `res.times`, `res.states`
  (present when `what = All`; otherwise `UnsupportedError` naming the
  option), `res.v(name)`, `res.i(name)`, `res.switch_states`,
  `num_steps()`, `empty()`, the fixed controller's records,
  `final_snapshot`. New: `res.grid`, `res["i(M1)"]`, `res.probes`,
  `res.events`, `res.stats`, `res.diagnostics`, `res.avg(...)`,
  `res.rms(...)`, `res.integral(...)`, `res.at(...)`,
  `res.circuit`, `res.plan`.
- The measures are implemented once, in C++, by the rule that
  `_result_views.pl_integral` and `step_start_states` already encode:
  the signal between two rows is linear from the right limit of the
  first (when it is an event row) to the left limit of the second.
  `avg = integral/(t1 − t0)`; `rms² = ∫y²/(t1 − t0)` with the exact
  `h·(a² + ab + b²)/3` per interval. `stats.time_average` /
  `time_rms` call them.
- `_PEDSimulationResult` is deleted: the DSED adapter fills a
  `SimulationResult` with `grid = Accepted`, its event rows and
  `RunStats`. Its attributes `n_accept`, `n_reject`, `n_events`,
  `n_rk45_steps`, `n_bdf2_steps`, `cpu_time_seconds` are read-only
  properties over `stats`.
- The nineteen attribute assignments in PY become fields set once in
  the finalisation step of `add-kernel-simulation-object`
  (`circuit`, `plan`, `diagnostics`, `stats`).
- Records that other changes add to the result are typed fields of the
  same type, not attributes: `control_tick_counts`
  (`fix-control-tick-semantics`) and the per-step normalised LTE of
  `add-fixed-step-resolution-diagnostics`, which is stored as an
  optional column of the table so that it follows the sampling policy.

- Alternative considered: returning a pandas / xarray object. Rejected:
  a new dependency for the core package; an adapter
  (`res.to_dataframe()`) can live in Python without it being the type.

### D4. Guarding the unweighted mean

`res["v(out)"]` and `res.v("out")` return NumPy arrays, so
`.mean()` cannot be removed. Two mitigations:

- the result's methods are the documented way, and every example and
  the AI guide use them;
- arrays returned from a result whose grid is not `Uniform` are a thin
  `ndarray` subclass (`Signal`) carrying `t` and offering `.avg()`,
  `.rms()`; its `.mean()` and `.std()` emit a `GridWarning` that names
  `avg()` (see Open Questions for whether this should be an error).

### D5. Extension points

- `add-in-kernel-event-losses`: a probe kind `EventEnergy` evaluated in
  `on_event` from the pre- and post-event views, and a sink that
  injects into a thermal node.
- `add-compiled-control-graph`: `Signal` probes for block outputs.
- `add-periodic-steady-state-shooting`: `Period` sampling and cycle
  measures.
- Expression probes: `ProbePlanEntry` gains an op-code stream; not in
  this change.

## Performance constraints

- Per accepted step with `what = All`, `when = EveryStep`: one row copy
  into the pre-reserved buffer, as today (about 35 ns/step); stored
  bytes equal to today's plus `n_pins` reals per device with a `q` row.
- With a probe selection: one pass over the flat plan; no virtual call,
  no map lookup, no string in the loop; stored bytes
  `n_probes · 8` per row.
- No heap allocation per step on the uniform grid. On the accepted
  grid the table grows geometrically by chunks; the growth is outside
  the per-step zero-allocation assertion and is bounded by
  `log2(n_rows)` allocations per run.
- Event rows are stored only at instants where the mask or a source
  changed.
- `res.times` and every column are zero-copy views
  (`update-fixed-step-loop-bookkeeping` already makes `times` one).
- Measures are O(rows) with no temporary array.
- Thread safety: a recorder belongs to its run; `RingSink` keeps the
  `LiveRing`'s single-producer / single-consumer contract.

## Risks / Trade-offs

- **Capacitor current changes meaning** from a smoothed derivative to
  the companion current, which on the fixed engine carries the
  trapezoidal rule's Nyquist ripple after a discontinuity. → It is the
  current KCL was solved with and it integrates to the right charge;
  the smoothed form stays available as `res.derived.i_smooth(name)`;
  the restart rule of `add-discontinuity-restart-rule` removes the
  ripple at its source. Flagged as an owner decision.
- **Memory of the default record grows** by the pin currents of `q`
  devices. → A few columns for the circuits that have such devices;
  measured in the task list; a probe selection removes it.
- **`Signal` as an `ndarray` subclass** can surprise code that checks
  types exactly. → It is returned only for non-uniform grids, and
  `np.asarray(signal)` is the plain array.
- **Two paths for a current** (recorded, or recomputed post hoc) could
  disagree. → Both call the same `ProbeSet::evaluate`; a test records
  and recomputes every recomputable probe and asserts bit equality.
- **Event rows change the row count** seen by code that zips `times`
  with something external. → Event rows live in `res.events`, not in
  the main table; the main table's rows are the left limits, as today.

## Migration Plan

Precondition: `add-unified-device-interface` step 9 (generic pin
currents and power). Each row is shippable alone.

| # | Step | Invariant |
|---|---|---|
| 1 | `SimulationResult` constructible and fillable (C++ and binding); the DSED adapter fills one; delete `_PEDSimulationResult`; its attributes become properties. | Every dsed test passes unmodified; `isinstance(res, SimulationResult)` on every engine. |
| 2 | `GridKind`, `RunStats`; `res.grid`, `res.stats`; `_trbdf2_stats` kept as an alias dict. | Suite green; `_trbdf2_warn_stats` reads `stats`. |
| 3 | Event rows on the fixed and variable-step engines (the DSED engine has them): right-limit state and mask at every instant where the mask or a source changed. Additive. | `times`, `states`, mode record bit-identical; `_result_views.step_start_states` uses `res.events` on every engine and the DCM boost diode average reads 0.0535 A on all three. |
| 4 | Measures in C++ (`avg`, `rms`, `integral`, `at`, `minmax`); Python methods; `stats.py` delegates. | `stats.py` tests unchanged; the regression of PR #140 (six runs agree to 0.3 %) expressed with `res.avg`. |
| 5 | `ProbeSet` with post-hoc evaluation; port the `_result_i` ladder kind by kind (resistor, switch, PWL diode with `V_f`, sources, inductor, transformer windings), then delete each rung. | `res.i` unchanged to 1e-12 for each ported kind on the harness. |
| 6 | Capacitor current from the companion law (**BREAKING**, own PR), per the owner's answer. | Charge balance `∫i dt = C·Δv` to 1e-12 per step. |
| 7 | In-kernel recording at commit: `Recorder`, `MemorySink`, `RecordOptions.what`; pin currents of `q` devices recorded with `All`; `builder.add_probe`, YAML `probes:`; `res["…"]`, `res.probes`. | Default runs: stored `times` / `states` bit-identical; recorded and recomputed probes bit-equal; zero-alloc tests green; ns/step of audit rows 1, 20 not above baseline. |
| 8 | Sampling policies: `Decimate` on any grid (lifting the `store_every` refusal on the variable-step engine), `Times` / `Period` (landing through the event queue when present, else interpolation with the flag), `RingSink` on every engine. | `store_every` on the fixed engine bit-identical to today; a decimated accepted-grid record integrates to the same `avg` as the full record within 1e-12 when events are kept; `t_eval` samples equal a run that landed there. |
| 9 | `Signal` arrays and `GridWarning` for non-uniform grids. | Suite green with warnings promoted to errors in the pulsim test configuration except where a test asserts the warning. |
| 10 | Delete the remaining ladder (PY:861-1188), the interim per-device current table of the device interface, and the per-branch attribute assignments; `losses.py` / `thermal.py` read probes. | Loss and thermal summaries unchanged to their test tolerances; stub check. |

Rollback: every step reverts alone; steps 5–7 keep the Python ladder as
a fallback behind a private switch until the harness has compared both
for one nightly cycle.

## Open Questions

1. Capacitor `res.i`: switch to the companion current (proposed; exact
   with respect to the discrete solution, shows trapezoidal ripple on
   the fixed engine) or keep `C·dv/dt` as the default and offer the
   companion current as `i_kcl`?
2. `.mean()` on a non-uniform `Signal`: warning (proposed), error, or
   nothing?
3. Should the default record (`what = All`) include the pin currents of
   `q` devices (proposed, so that every `res.i()` works after a default
   run) or should those require an explicit probe?
4. YAML `probes:` is a netlist key. The roadmap row lists only
   `simulation-results`; is it acceptable to specify the key there, or
   should a `netlist-yaml` delta be added to this change?
5. `Times` on the fixed grid when a requested time is not a grid point:
   interpolate and flag (proposed) or refuse?
6. Is a `to_dataframe()` convenience wanted (optional pandas import)?
