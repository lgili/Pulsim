## Context

Pulsim's stated architecture in `openspec/specs/kernel-v1-core/spec.md` includes a "hybrid event-driven engine with state-space primary path and DAE fallback". In practice (`core/src/v1/transient_services.cpp:158-200`), the primary path bails out as soon as a `IdealDiode | MOSFET | IGBT` is in the netlist — i.e., for every realistic power-electronics circuit. This proposal makes the state-space path actually work for switching converters by treating ideal switching devices as piecewise-linear (PWL) components.

This is the architectural pattern proven by PLECS (Plexim, Allmeling & Hammer 1999) and by `Modelica.Electrical.PowerConverters` in OpenModelica. The technique is well-established in the literature; the work here is one of careful implementation, not research.

## Goals / Non-Goals

**Goals:**
- ≥10× speedup on switching converter benchmarks at equal accuracy.
- Eliminate Newton iteration from stable topology windows.
- Make `state_space_primary_steps` telemetry truthful.
- Preserve smooth-model path (`Behavioral`) as opt-in for users who need its physics (e.g. quasi-saturation, soft-switching transitions).

**Non-Goals:**
- Importing PLECS device libraries directly (handled by separate `add-catalog-device-models` change).
- Symbolic state-space *reduction* (Kron, modal). We keep MNA-derived form, which is enough for PLECS-equivalent speedup.
- Hybrid-mode-per-step switching (e.g., `Ideal` then `Behavioral` mid-simulation). Mode is set per device once per simulation.
- Multi-rate / multi-domain (mechanical, magnetic). Out of scope for this change.

## Decisions

### Decision 1: PWL is opt-in via `SwitchingMode::Auto` defaulting to `Ideal` when supported
- **What**: Each switching device exposes `SwitchingMode { Ideal, Behavioral, Auto }`. `Auto` resolves to `Ideal` if all switching devices in the circuit support it. User can force `Behavioral` per device or globally.
- **Why**: Backward compat. Existing tests pass without modification. Power-electronics users get the speedup automatically.
- **Alternatives considered**:
  - *Always `Ideal`* — breaks users relying on tanh smoothing for diodes near zero crossing. Rejected.
  - *Always `Behavioral` until explicit opt-in* — defeats the speedup story; users won't change defaults. Rejected.

### Decision 2: Topology signature = bitmask over switch states only
- **What**: For k switching devices, signature is a `std::uint64_t` bitmask. State 0 = open/off, 1 = closed/on. (For k > 64, fall back to `boost::dynamic_bitset` or hash of `vector<uint8_t>`.)
- **Why**: Bitmask is O(1) to compute, O(1) to compare, deterministic, and exactly captures the topology equivalence class. Drops the broken O(nnz) numeric hash in `transient_services.cpp:35`.
- **Alternatives considered**:
  - *Hash sparsity pattern* — works but slower than bitmask, and pattern only changes when bitmask changes for piecewise linear systems. Equivalent but more expensive.
  - *Numeric value hash* — current approach, breaks reuse across steps. Rejected.

### Decision 3: State-space form `M ẋ + N x = b(t)`, integrated via Tustin
- **What**: For each topology, `assemble_state_space()` walks devices and builds:
  - `M` — diagonal/sparse matrix from `C_eq * dx/dt` and `L_eq * di/dt` contributions.
  - `N` — sparse matrix from `G * x` contributions (resistive, switch on/off conductance).
  - `b(t)` — time-varying source vector from voltage/current sources, PWM, sine, pulse waveforms.
- Time-step via Tustin (trapezoidal): `(M + dt/2 · N) x_{n+1} = (M - dt/2 · N) x_n + dt/2 · (b_{n+1} + b_n)`.
- One linear solve per step. Solver = KLU with cached symbolic + numeric factorization for the topology.
- **Why**: Tustin is A-stable (matches existing `Trapezoidal` semantics), preserves passivity, second-order accurate. Companion-model already in [integration.hpp:136](core/include/pulsim/v1/integration.hpp:136) provides the same coefficients per-element.
- **Alternatives considered**:
  - *Matrix exponential `expm(A·dt)`* — exact for LTI segments but expensive for varying dt. Future enhancement, behind `simulation.pwl_integrator: tustin | matrix_exp`.
  - *Backward Euler* — first-order, more numerical damping. Available as `pwl_integrator: backward_euler` for stiffness recovery.

### Decision 4: Event detection by sign-of-state, bisected to `event_tolerance`
- **What**: PWL diode commutes when (in `on` state) `i_diode < 0` or (in `off` state) `v_diode > 0`. Bisect to event time within `event_tolerance` (default `1e-12`). Snap step boundary, commit new state, rebuild matrices.
- **Why**: PLECS-equivalent semantics. With sharp PWL there is no smooth transition; only the event matters.
- **Alternatives**:
  - *Newton-style continuous tracking* — defeats PWL speedup. Rejected.
  - *Wider event tolerance* — accuracy degrades on fast PWM (>500 kHz). Default `1e-12` works for ≤10 MHz.

### Decision 5: LRU cache bound + diagnostic on explosion
- **What**: Default 4096 distinct topologies cached. LRU eviction beyond. Diagnostic `PwlTopologyExplosion` if eviction occurs >100× in one run (suggests pathological behavior).
- **Why**: For k switches, theoretical max 2^k topologies. For k=12 (typical interleaved 3φ converter with control), 4096 is comfortable. For k=20+ (large grid simulations), eviction is expected and harmless. Diagnostic catches the "all 2^k topologies are reachable" pathology.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| User relies on smooth diode behavior (e.g. control-loop including diode soft transition) | Keep `Behavioral` mode and document when to use it |
| Sharp PWL diode oscillates at zero-crossing in noisy circuits | Add `event_hysteresis` parameter (default `1e-9 V`) to require minimum overshoot before commute |
| Topology cache memory grows on grid simulations | LRU eviction + telemetry for monitoring |
| MOSFET body-diode subtlety (reverse conduction) | Body-diode modeled as embedded PWL diode with separate state; tested in `pwl_mosfet_body_diode` test |
| Existing tests assume Newton iterations occur per step | Add explicit `expect_newton: false` assertion only in PWL test files; legacy tests untouched |
| Algebraic loop on simultaneous switch transitions | Detect simultaneous events via topology-bitmask delta; iterate event resolution to fixed point (Gauss-Seidel) |

## Migration Plan

1. **Phase 0 (this PR)**: Land PWL infrastructure behind `SwitchingMode::Auto` defaulting to `Behavioral`. No user-visible behavior change.
2. **Phase 1**: Flip default `Auto` → `Ideal` for circuits where all switching devices support it. Add migration guide.
3. **Phase 2**: Deprecate Python wrapper retry/auto-bleeder layer (`python/pulsim/__init__.py:_apply_auto_bleeders`) when `Ideal` is the resolved mode. Remove after one release window.

Rollback: `simulation.switching_mode: behavioral` in YAML or `SimulationOptions.switching_mode = SwitchingMode::Behavioral` in code restores pre-change behavior bit-for-bit.

## Open Questions

- Should we expose state-space matrices `M`, `N`, `b` to Python for external analysis (e.g., user computing eigenvalues)? *Lean: yes, post-MVP, behind `SimulationResult.export_pwl_segments`.*
- How to handle algebraic loops with two ideal switches sharing a node? *Lean: add structural diagnostic at parse time, suggest user add small parasitic.*
- Multi-rate event detection for circuits with vastly different switching frequencies (e.g., 100 kHz buck + 50 Hz grid)? *Defer to dedicated change post-MVP.*
