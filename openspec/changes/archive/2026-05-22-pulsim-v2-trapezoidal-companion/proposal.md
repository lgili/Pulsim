## Why

Layers 0-5 V0 together can simulate any **piecewise-linear,
static-only** circuit:
- Resistors, voltage sources, switches → DC operating point per
  switch state, pre-factorized.
- Fixed-dt time stepping with user-supplied switch schedule.

That's enough for chopper-PWM demonstrations but it's NOT a real
PE simulator. **Real power electronics needs Capacitors and
Inductors.** A buck converter without an output capacitor doesn't
filter the PWM; an LC tank without storage elements doesn't
oscillate.

This OpenSpec adds Capacitor + Inductor support via the
**trapezoidal companion model** — the textbook SPICE / PLECS
technique that turns a dynamic device into a static conductance +
a history term per timestep:

- **Capacitor**: `C dv/dt = i`  →  trap rule  →  `i_{n+1} = G_eq · v_{n+1} − I_hist`
  - `G_eq = 2C/dt`        (stamped into the MNA matrix)
  - `I_hist = G_eq · v_n + i_n`  (history term, contributes to b)
- **Inductor**: dual — `G_eq = dt/(2L)`, `V_hist` term

The companion model is **the entire conceptual content** of how
PSIM / PLECS / SPICE handle dynamic elements with implicit
integration. With it, the v2 kernel can simulate ANY linear PE
circuit at full fidelity at competitive performance.

## What Changes

**Scope decision — Layer 4 V1 + Layer 5 V1**:

- **Two new device classes** in Layer 2 — `Capacitor` and
  `Inductor`. Each has the trap-companion API:
  - `companion_stamp(dt, p) → (g_eq, history_factor)` — what
    Layer 3 stamps into J.
  - `history_term(v_prev, i_prev, dt, p) → Real` — what Layer 5
    pushes into b every step from previous-step state.

- **Layer 4 cache becomes dt-aware**:
  - `PwlStateSpaceCache::build(Real dt)` — must be called with a
    dt; the cache stores it and exposes via `cache.dt()`.
  - Calling `build(dt2)` with a different dt rebuilds all
    segments (factors become stale).
  - V0's static-only `build()` (no dt) is kept as a thin shim
    that picks `dt = 0` and SKIPS dynamic-device stamping — used
    by static-only callers (chopper PWM tests).

- **Layer 5 manages history state**:
  - New `HistoryState` type holds per-dynamic-branch (v_prev,
    i_prev) entries.
  - `run_transient` initialises history to zeros (all-zero IC for
    V0), updates history from x_{n-1} BEFORE each cache.solve,
    feeds the history contribution to cache.solve via b_extra.
  - Backwards-compatible: circuits with NO caps/L behave
    identically to Layer 5 V0.

- **Initial conditions still all-zero**. V_C(0) = 0, I_L(0) = 0
  for every cap/inductor. This is correct for power-on transients
  of most PE converters. DC operating-point pre-charge is a
  follow-up OpenSpec (`pulsim-v2-dc-operating-point`).

- **Nonlinear devices still skipped**. Newton iteration on top
  of the cached factor lands in `pulsim-v2-nonlinear-segment-newton`.

**New files** (estimated 1200-1500 LOC added):

```
core/include/pulsim/v2/models/
├── capacitor.hpp                     # NEW — trap-companion 2-pin C
└── inductor.hpp                      # NEW — trap-companion 2-pin L

core/include/pulsim/v2/stamping/
└── stamp_companion.hpp               # NEW — companion-stamp helper

core/include/pulsim/v2/pwl/
├── device_pool.hpp                   # MODIFIED — add_capacitor/inductor
├── assemble.hpp                      # MODIFIED — dispatch C/L stamping
├── cache.hpp                         # MODIFIED — build(dt) overload
└── history_state.hpp                 # NEW — per-step history container

core/include/pulsim/v2/solver/
├── run_transient.hpp                 # MODIFIED — history-state loop
└── history_collector.hpp             # NEW — builds HistoryState from
                                      # graph + pool

core/tests/v2/layer4_v1/              # NEW test directory
├── test_capacitor.cpp                # Companion stamp math
├── test_inductor.cpp                 # Companion stamp math
├── test_dt_aware_cache.cpp           # Cache rebuild on dt change
└── test_integration_rc.cpp           # RC charging transient

core/tests/v2/layer5_v1/              # NEW test directory
├── test_history_state.cpp
├── test_run_transient_history.cpp
└── test_integration_rlc.cpp          # RLC tank oscillation
```

**Validation:**

- **RC charging**: V_C(t) = V_dc · (1 − e^{−t/RC}). Simulate
  with dt = τ/100 over t ∈ [0, 5τ], verify match to analytical
  within < 1 %.
- **RL ramp**: I_L(t) = V/R · (1 − e^{−Rt/L}). Same validation
  pattern.
- **RLC underdamped**: ζ = R/2·√(C/L) < 1, verify period
  2π/ω_d (damped frequency) within < 2 %.
- **Static circuits unchanged**: All Layer 4 V0 + Layer 5 V0
  tests stay green (backwards compat).

## Impact

- **Affected specs**:
  - MODIFIED `kernel-v2-pwl-cache` (dt-aware build + Capacitor /
    Inductor device support).
  - MODIFIED `kernel-v2-solver` (history-state plumbing).
  - **NEW** capability `kernel-v2-dynamic-devices` (Capacitor +
    Inductor models with companion-stamp contract).

- **Affected code** (estimate):
  - NEW headers: ~600 LOC.
  - MODIFIED headers: ~200 LOC (cache.hpp, device_pool.hpp,
    assemble.hpp, run_transient.hpp).
  - NEW tests: ~700 LOC across 7 files.
  - NEW CMake targets: `pulsim_v2_layer4_v1_tests`,
    `pulsim_v2_layer5_v1_tests`.

- **Migration**: zero for static-only callers. The Layer 4 V0
  `cache.build()` (no-arg) signature is preserved as a shim that
  picks the static-only path. New callers that use Capacitor or
  Inductor MUST call `cache.build(dt)`.

- **Risk**: medium. The math is well-understood (companion model
  is 50 years old) but the API touches three layers. Mitigation:
  - Pure-math unit tests for stamp_companion before integration.
  - RC analytical test catches sign / scale bugs immediately.
  - Backwards-compat regression on every Layer 4 / Layer 5 V0
    test (no behaviour change for static circuits).

- **What this proposal explicitly does NOT do**:
  - **No DC operating-point pre-charge**. V_C(0) = I_L(0) = 0.
    Real-world circuits often need the steady-state IC; that
    lands in `pulsim-v2-dc-operating-point`.
  - **No adaptive dt + LTE estimation**. The cache becomes dt-
    dependent; adaptive dt requires invalidation, which lands
    in a Layer 5 V1.5 follow-up.
  - **No backwards Euler / Gear-2 / BDF**. Trapezoidal only.
    Other implicit methods would each need their own companion
    derivation; trap is the default for power electronics.
  - **No event detection**. Switch transitions still come from
    the user-supplied `switch_fn`. Auto zero-crossing is a
    separate Layer 5 V1 follow-up.
  - **No nonlinear devices**. Newton iteration on top of the
    cached factor is `pulsim-v2-nonlinear-segment-newton`.
