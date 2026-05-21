## ADDED Requirements

### Requirement: HistoryState — Per-Step History Tracker

`pulsim::v2::pwl::HistoryState` SHALL track the previous-step
state of every dynamic device (Capacitor, Inductor) in the
circuit. It is owned by the `run_transient` loop and updated
once per accepted step.

The class MUST expose:

```cpp
class HistoryState {
public:
    HistoryState(const topology::Graph& graph,
                 const DevicePool& pool);

    /// Returns a state-size Vector populated with the trap
    /// companion's history-term contributions on the right
    /// rows. Capacitor history is a current source into the
    /// device's nodes; inductor history is a voltage source
    /// on the inductor's constraint row.
    [[nodiscard]] Vector compute_b_extra(Real dt) const;

    /// Reads the current step's (v, i) per dynamic device
    /// from `x` and stores them for the next step's history.
    void update_from_state(const Vector& x, Real dt);

    /// Zeros all entries — used at the start of run_transient
    /// (all-zero initial conditions for V0).
    void reset() noexcept;

    /// Diagnostic helper.
    [[nodiscard]] Size num_dynamic_branches() const noexcept;
};
```

The class SHALL initialise all entries to zero on construction
(V0 IC convention).

#### Scenario: Empty graph yields empty HistoryState

- **GIVEN** a graph with no Capacitors or Inductors
- **WHEN** the user constructs `HistoryState`
- **THEN** `num_dynamic_branches()` SHALL be `0`
- **AND** `compute_b_extra(dt)` SHALL return a zero vector.

#### Scenario: Capacitor history contributes to its node row

- **GIVEN** a 1-node graph with a 1 µF capacitor to ground,
  `dt = 1µs`
- **AND** a previous state vector with v_node = 10 V
- **WHEN** the user calls `history.update_from_state(x, dt)`,
  then `history.compute_b_extra(dt)`
- **THEN** the returned vector's row 0 SHALL hold the history
  current contribution sign-consistent with the companion
  stamp.

#### Scenario: reset zeros all entries

- **GIVEN** a HistoryState that has been populated via
  `update_from_state`
- **WHEN** the user calls `history.reset()`
- **THEN** the next `compute_b_extra(dt)` call SHALL return
  a zero vector.

### Requirement: run_transient — History-Aware Loop

`pulsim::v2::solver::run_transient` SHALL be extended to manage
HistoryState internally for circuits with dynamic devices. The
new V1 signature MUST be:

```cpp
SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});
```

The function MUST:

1. Throw `std::invalid_argument` if `!opts.valid()`,
   `!switch_fn`, or `cache.dt() != opts.dt` (a mismatched cache
   would silently produce wrong numerical results).
2. Compute `state_size = pool.state_size(graph)` internally.
3. Construct a `HistoryState` from `(graph, pool)`.
4. Loop `k = 0 .. expected_step_count - 1`:
   a. `t = t_start + k * dt`.
   b. `b_extra = history.compute_b_extra(dt)`.
   c. If `b_extra_fn` is non-empty: `b_extra += b_extra_fn(t)`.
   d. `mask = switch_fn(t)`.
   e. `cache.solve(mask, b_extra, x)`.
   f. `history.update_from_state(x, dt)`.
   g. Record `(t, x)` into the result.
5. Return the populated SimulationResult.

Backwards-compat: the V0 5-arg `run_transient(cache, state_size,
opts, switch_fn, b_extra_fn)` is DEPRECATED. Callers MUST migrate
to the new 6-arg signature (the V0 chopper-PWM test is updated
as part of this change).

#### Scenario: Mismatched cache.dt() throws

- **GIVEN** a cache built with `cache.build(dt = 1e-6)` and
  options with `opts.dt = 2e-6`
- **WHEN** the user calls `run_transient`
- **THEN** the call SHALL throw `std::invalid_argument`.

#### Scenario: Static-only circuit produces V0-identical result

- **GIVEN** a chopper graph with no Capacitors or Inductors,
  built with `cache.build(dt)` for a PWM dt
- **WHEN** the user runs `run_transient` (V1 6-arg)
- **THEN** the resulting `SimulationResult` SHALL be
  numerically identical to the Layer 5 V0 chopper PWM test
  (same mean, same per-sample values).

#### Scenario: RC charging matches analytical solution

- **GIVEN** an RC series circuit: V_dc(5 V) → R(1 Ω) →
  C(1 µF) → GND
- **AND** `cache.build(dt = 1e-8)`, `opts{t_end=5e-6}`
- **WHEN** the user runs `run_transient`
- **THEN** for every recorded sample k, `v_C(t_k)` SHALL be
  within `0.5 %` of the analytical
  `V_dc · (1 − e^{−t_k / τ})` (τ = RC = 1 µs).

#### Scenario: RLC underdamped ringdown matches analytical period

- **GIVEN** an RLC series circuit with `R=0.5 Ω`, `L=1 µH`,
  `C=1 µF` (ζ ≈ 0.25, underdamped)
- **WHEN** the user runs a long enough transient at
  `dt = T_d/100`
- **THEN** the first zero-crossing of v_C(t) SHALL occur at
  the analytical `t_1 = (π/2 − φ) / ω_d` within `2 %`.
