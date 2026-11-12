## Context

Paths: `core/…` = `core/include/pulsim/…`; RT = `core/solver/run_transient.hpp`,
TB = `core/solver/trbdf2_transient.hpp`, PY = `python/pulsim/__init__.py`,
BN = `python/bindings.cpp`.

What exists today:

- `models::DeviceModel` (`core/models/device_model.hpp:44-52`): a
  concept with `Params`, `kind`, `num_terminals`, `is_linear` and one
  templated `current<S>(v, p)`. `evaluate_current_and_jacobian` seeds
  `ADRealN<N>` and returns value plus partials. It is used by
  `pwl/nonlinear_refresh_device.hpp` for the three algebraic Newton
  devices (smooth diode, MOSFET level 1, IGBT level 1).
- Everything with memory is outside it. `NonlinearCapacitor`,
  `LauritzenDiode`, the IGBT tail, `PmsmMna` and the saturable-inductor
  family each have a history class that owns (a) its state, (b) the
  trapezoidal and BDF2 stage formulas, (c) snapshot / restore / flat
  forms, and a `refresh_*` function that stamps by hand with
  `J_nl.coeffRef`.
- The drivers compose them by wrapping a `std::function` once per kind
  (RT:471-600, TB:588-697) and by naming each history at every init,
  seed, commit, snapshot and restore (99 call sites).
- `DevicePool` is a 20-alternative `std::variant` keyed by branch id,
  locked to `StoredKind` by `static_assert` (`pwl/device_pool.hpp:906-957`),
  with side containers per kind (eight found by grep at
  `pwl/device_pool.hpp:1028-1060`).
- The PWL diode already has one law, `SwitchedDiode::decide` /
  `crossing_signal`, used by three engines.
- The linear part is a `(G, C, b)` split with `solve_at(mask, dt)` and a
  lazy LRU factor cache; the linear companion history is stored as
  physical `(v, i)`.

Owner notes that are requirements on this design:
*nonlinear-refresh-wrapper-chain*, *trbdf2-probe-solves-set-the-device-step*,
*flux-row-keep-conductance-form*, *fixed-engine-first-order-start*,
*newton-devices-need-smooth-laws-to-1e-12*, *switchstatemask-takes-a-size*.

## Goals / Non-Goals

Goals:

- One contract for Newton devices: `f + dq/dt`, AD Jacobians, the
  integrator owns the time formula.
- Adding a device kind edits the kind's own header and one registry
  line; no file under `solver/`, `analysis/` or `python/` changes for it
  to stamp, integrate, snapshot, report its current, appear in AC / DC,
  in YAML and in `components()`.
- One flat device-state vector; snapshot `{t, x, dev_state, mask}`.
- Generic currents and power; AC and DC derived from `f` and `q`.
- A migration in which every merge leaves `main` green and, for linear
  and PWL-only circuits, bit-identical.

Non-Goals:

- No new device models (`add-power-mosfet-datasheet-device`,
  `add-mna-native-machines`, `add-pwl-igbt-and-thyristor`).
- No user-defined devices (`add-external-device-abi`) and no parameter
  derivatives (`add-parameter-sensitivities`); the extension points are
  named below.
- No change of the step loop (`refactor-unified-transient-stepper`), of
  event location (`add-unified-event-queue`) or of result storage
  (`add-probes-and-recorders`).
- No change of any device law. The `IdealDiode` law is fixed by
  `fix-nonlinear-diode-passivity`, the body-diode drop by
  `update-composite-diode-forward-drop`.
- The Newton acceptance criteria and the DC cascade's absolute `dx`
  rule stay as they are.

## Decisions

### D1. Three categories, one registry

```cpp
namespace pulsim::device {

template <class... Ds> struct TypeList {};

// The only list of kinds. StoredKind values are the indices of this list.
using Builtin = TypeList<
    Resistor, VoltageSource, Switch, Capacitor, Inductor, SwitchedDiode,
    IdealDiode, CurrentSource, PWMVoltageSource, SineVoltageSource,
    PulseVoltageSource, MosfetLevel1, IgbtLevel1, VCVS, SaturableInductor,
    NonlinearCapacitor, ShockleyDiode, LauritzenDiode, PmsmMna,
    IdealTransformer>;

enum class Category { Static, Mode, Newton };
template <class D> inline constexpr Category category_of = /* from the concept D models */;
}
```

| Category | Concept | What it supplies | Where it is stamped |
|---|---|---|---|
| Static | `StaticDevice` | constant `G`, `C` entries, source waveform | once per mask into the cached split |
| Mode | `ModeDevice` | per-mode constant stamp, `decide`, `crossing_signal` | once per mask into the cached split |
| Newton | `Device` | `eval → (f, q)` | every Newton iteration |

Every category shares `Params`, `name`, `n_pins`, and the optional
`fields`, `outputs`.

- Alternatives considered.
  - *A virtual `IDevice` base.* Rejected: one indirect call per device
    per Newton iteration, no templated AD, heap-allocated instances. The
    audit measured device evaluation at under 0.5 % of a Newton step;
    the cost that matters is the plumbing, and a virtual interface
    keeps it.
  - *Keep the variant and add visitors.* Rejected: it is today's
    design; the 123 sites are visitors written by hand.
  - *Make every device a Newton device.* Rejected: it would move the
    linear and PWL parts out of the `(G, C, b)` split and the factor
    cache, which is the property the audit lists first under "preserve".
  - *Code generation from a device description language.* Deferred to
    `add-external-device-abi` (sympy-to-C, OSDI); the ABI it needs is
    the `eval` signature defined here.

### D2. The Newton device contract

```cpp
namespace pulsim::device {

// Time formula of one solve. Chosen by the integrator, never by a device.
enum class Formula : std::uint8_t { Dc, BackwardEuler, Trapezoidal, Bdf2Stage2 };

// Passed by value into every solve. There is no other source of h.
struct StepCtx {
    Real    t;            // time at which f is evaluated
    Formula formula;
    Real    a;            // d/dt q ≈ a·q(x) + r ; 0 for Dc
    Real    h;            // elapsed time this solve spans (used by accept)
    Real    c2, c3;       // BDF2 stage weights of q_γ and q_n
    Real    source_scale; // DC continuation
};

template <class S, std::size_t R>
struct Out { std::array<S, R> f{}; std::array<S, R> q{}; };

template <class D>
concept Device = requires {
    typename D::Params;
    { D::name }       -> std::convertible_to<std::string_view>;
    { D::n_pins }     -> std::convertible_to<std::size_t>;
    { D::n_internal } -> std::convertible_to<std::size_t>;   // branch currents, charges
} && requires(const typename D::Params& p,
              std::span<const Real, D::n_pins + D::n_internal> x,
              const EvalCtx<D>& c,
              Out<Real, D::n_pins + D::n_internal>& o) {
    D::template eval<Real>(p, x, c, o);       // must also compile for ADRealN<R>
};

// Optional members, each detected with a `requires` expression:
//   struct State;                                   trivially copyable, made of Real
//   static void init  (const Params&, std::span<const Real> x, State&);
//   static void accept(const Params&, std::span<const Real> x, const StepCtx&, State&);
//   static constexpr std::size_t n_events;
//   static void events(const Params&, std::span<const Real> x, std::span<Real> g);
//   static constexpr OutputDesc output_table[];      {"q_M","C"}, {"T_e","N·m"}, ...
//   static void outputs(const Params&, std::span<const Real> x, const State*, std::span<Real>);
//   static constexpr Field fields[];                 {"I_S", &Params::I_S, "A", lo, hi}
//   static constexpr PinDesc pins[];                 {"anode"}, {"cathode"}  (domain: see add-immutable-circuit-ir)
}
```

Row convention: rows `0 … n_pins−1` are the KCL rows of the pins
(`f` = current leaving the node into the device, `q` = charge whose
derivative is such a current); rows `n_pins …` are the device's own
equations for its internal unknowns. `x` is `[v_pin…, internal…]`.
`EvalCtx` gives read access to `State` (when present) and to `t`.

Mapping of the existing Newton kinds (from audit `raw/models.md` §c):

| Kind | `n_internal` | `f` | `q` | `State` |
|---|---|---|---|---|
| `IdealDiode`, `ShockleyDiode` | 0 | diode current | — | — |
| `MosfetLevel1`, `IgbtLevel1` | 0 | channel current (3 pins) | — | — |
| `NonlinearCapacitor` | 0 | — | junction charge `q(v)` | — |
| `LauritzenDiode` | 1 (`q_M`) | `(q_E − q_M)/T_M`; row: `q_M/τ − (q_E − q_M)/T_M` | `q_M` on its row | — |
| IGBT tail | 1 (tail charge) | tail current; row: decay law | tail charge | — |
| `SaturableInductor` (atan, `FluxTable`, `GappedCore`) | 1 (`i_L`) | KCL `±i_L`; row: `v_a − v_b` | `−λ(i_L)` on its row | — |
| Jiles–Atherton core | 1 (`i_L`) | as above | `−λ(i_L; H, M)` | `(H, M, direction, substeps)` + `accept` |
| `PmsmMna` | 3 (`i_abc`) | KCL, torque on the mechanical node | `λ_abc(i, θ)` | — |

The table is a sketch: the exact rows of each kind are fixed in its own
port, against its existing tests. `MosfetLevel1`, `IgbtLevel1`,
`IdealDiode` and `ShockleyDiode` already expose the `current<S>` law; an adapter `FromCurrentModel<T>` (`f[0] = +i`,
`f[1] = −i`) makes each a `Device` without touching its law.

- Alternatives considered.
  - *Hand Jacobians behind the same signature.* Rejected as the
    default: "one stamp per device, AD-derived" is on the audit's
    preserve list. A law with an exact tangent that AD cannot produce (a
    Hermite `FluxTable`, the JA tangent) returns
    `ad::chain(x, value, dvalue_dx)`, which stays inside the one `eval`.
  - *Keep internal states eliminated inside the device* (today's
    Lauritzen and tail histories integrate their charge themselves).
    Rejected: that is exactly "each device hand-derives TR and BDF2".
    Cost: Lauritzen and the tail add one unknown each, so `state_size`
    grows by one per such device. Internal unknowns are appended after
    every existing unknown, so node and branch indices and all
    name-based accessors are unchanged; `res.states` gains columns.
    Listed under Open Questions.

### D3. The integrator owns the time formula

```cpp
// r for row k of an instance, from the committed state (q_n, qdot_n) and,
// in the BDF2 stage, the stage scratch q_γ.
//   Dc             : a = 0,      r = 0
//   BackwardEuler  : a = 1/h,    r = −q_n/h
//   Trapezoidal    : a = 2/h,    r = −(2/h)·q_n − qdot_n
//   Bdf2Stage2     : a = c1/h,   r = (c2·q_γ + c3·q_n)/h
template <Device D>
Real stamp_fq(const Instance<D>& in, std::span<const Real> x_global,
              const StepCtx& c, std::span<const Real> dev_state,
              std::span<const Real> stage_scratch, NewtonSystem& sys) {
    constexpr std::size_t R = D::n_pins + D::n_internal;
    auto xs = gather_seeded<R>(in.unknown, x_global);      // ADRealN<R> on the stack
    Out<ad::ADRealN<R>, R> o;
    D::template eval<ad::ADRealN<R>>(in.p, xs, eval_ctx(in, c, dev_state), o);
    Real worst = 0;
    for (std::size_t r = 0; r < R; ++r) {
        const Real F = o.f[r].value() + c.a * o.q[r].value()
                     + history_term<D>(in, r, c, dev_state, stage_scratch);
        sys.add_residual(in.unknown[r], F);
        for (std::size_t k = 0; k < R; ++k)
            sys.add_jacobian(in.jac_slot[r * R + k],
                             o.f[r].d(k) + c.a * o.q[r].d(k));
        worst = std::max(worst, std::abs(F));
    }
    return worst;
}
```

Consequences, each a requirement:

- **No ambient step size.** `StepCtx` is a parameter of the solve. A
  probe, settle or landing solve cannot stamp devices with another
  solve's `h`, which is the defect behind the TR-BDF2 SIGSEGV.
- **Conductance form is the only form.** The row is
  `f + a·q + r`; nothing multiplies it by `h`. For the flux row
  (`f = v`, `q = −λ(i)`) the diagonal is `−a·L(i)`, i.e. `−(2/h)L` and
  `−(c1/h)L`, as `stamp_saturable_inductor` has today.
- **Backward Euler is free.** The restart rule of
  `add-discontinuity-restart-rule` needs a `BackwardEuler` case in every
  stateful device; here it is one line of `history_term` and no device
  code.
- **Elapsed time is explicit.** `c.h` is the time the solve really
  spans; `accept` and the commit use it. No formula is obtained by
  assembling another at a different `h` (the rejected "trapezoid at
  2h").
- **Commit.** After an accepted step: one `eval<Real>`, then
  `q_n ← q(x)`, `qdot_n ← a·q(x) + r`, then `D::accept` when present.
  `(q_n, qdot_n)` are physical (a capacitor's charge and current, an
  inductor's flux and voltage), so a variable `h` stays legal.
- **Start.** `init` seeds `State`; `qdot_0` comes from the run's initial
  condition (zero, DC operating point, a snapshot). The consistent
  initial solve is owned by `add-discontinuity-restart-rule` and
  `add-kernel-simulation-object`.

- Alternatives considered: passing `(h, stage)` as today's
  `TrBdf2Stage` enum to each device (rejected: the device then owns the
  formula again); a per-device `integrate()` hook for "special"
  devices (rejected: no existing device needs one once the charge is
  an unknown).

### D4. PWL two-state devices: one law, shared

```cpp
struct ModeStamp { Real g; Real i_offset; };   // i = g·v − i_offset on the branch

template <class D>
concept ModeDevice = requires(const typename D::Params& p, bool on, Real v) {
    typename D::Params;
    { D::n_pins } -> std::convertible_to<std::size_t>;
    { D::stamp_mode(p, on) } -> std::same_as<ModeStamp>;
};

template <class D>
concept SelfCommutated = ModeDevice<D> &&
    requires(const typename D::Params& p, bool on, Real v) {
        { D::decide(on, v, p) }          -> std::same_as<bool>;
        { D::crossing_signal(on, v, p) } -> std::convertible_to<Real>;
    };
```

`Switch` is a `ModeDevice` (driven externally). `SwitchedDiode` is
`SelfCommutated`; its `decide` / `crossing_signal` are the functions
that exist today, unchanged. One `ModeSet::redecide(x)` calls `decide`
for every self-commutated instance; the pwl event iteration, the DC
operating point, the TR-BDF2 settle and the DSED settle become four
callers of it (the loops themselves are unified by
`add-unified-event-queue`). The mode stamp contributes to `b` as it does
now (`∓g_on·V_f` on the ON mask only), so `J` and its sparsity do not
depend on `V_f` and every cached factor stays valid.

Extension points: `add-pwl-igbt-and-thyristor` adds kinds here (a latch
is a `State` on a mode device); `add-branch-form-ideal-switch` replaces
`ModeStamp` with a branch-form stamp without touching `decide`.

### D5. Storage, stamp plan and state vector

```cpp
template <class D> struct Instance {
    typename D::Params p;
    std::array<Index, D::n_pins + D::n_internal> unknown;   // −1 = ground
    std::array<Slot,  (D::n_pins + D::n_internal) *
                      (D::n_pins + D::n_internal)> jac_slot; // into the Newton matrix values
    std::uint32_t state_offset;                               // into DeviceState::v
    std::uint32_t name;                                       // interned
};

template <class... Ds> struct DeviceSet {
    std::tuple<std::vector<Instance<Ds>>...> kinds;
    template <class F> void for_each_newton_kind(F&& f);      // fold, static dispatch
};

struct DeviceStateLayout {
    struct Block { std::uint16_t kind; std::uint32_t first, count, reals_per_instance; };
    std::vector<Block> blocks;        // registry order
    std::uint64_t signature;          // hash of (kind name, count, reals) per block
};

struct DeviceState { std::vector<Real> v; };   // one per run, plus one preallocated rollback twin
```

- The stamp loop is
  `set.for_each_newton_kind([&](auto& vec){ for (auto& in : vec) worst = max(worst, stamp_fq(in, …)); })`.
  The compiler emits one loop per kind; there is no indirect call and no
  `std::function` on the path.
- `jac_slot` is computed once per circuit. It is the stamp-slot map of
  `update-linear-time-assembly` and the in-place value write of
  `update-newton-persistent-solver`; before those land, `add_jacobian`
  falls back to `coeffRef` on the existing `J_nl`.
- Per instance the state vector holds `(q_n, qdot_n)` for each row with
  a non-zero `q`, then `sizeof(State)/sizeof(Real)` reals. The BDF2
  stage scratch `q_γ` is a workspace, not state, and is not in a
  snapshot.
- The linear companion history (`HistoryState`: `v_prev`, `i_prev` per
  L / C) keeps its arithmetic; only its storage becomes a block of the
  same vector.
- Workspaces (`NewtonSystem`, AD arrays, scratch) are per run and reuse
  the existing thread-local pools.

- Alternatives considered: one `std::vector<std::variant<…>>` with
  `std::visit` (rejected: a jump per instance and `sizeof` of the
  largest alternative per element); structure-of-arrays per kind
  (deferred: only useful for SIMD batches, see the audit's parallelism
  section).

### D6. Snapshot

```cpp
struct SolverSnapshot {
    Real t;
    Vector x;
    std::vector<Real> dev_state;        // DeviceState::v
    std::vector<bool> mask;             // mode bits the next step starts from
    std::vector<bool> stamped_mask;     // mask of the solve that produced x (record row 0)
    std::uint64_t layout_signature;
    bool valid;
};
```

- `mask` replaces `diode_on`: it carries every mode device's bit, the
  solver-owned ones included. `stamped_mask` keeps today's meaning
  (`solver/result.hpp:139-152`).
- Resume refuses a snapshot whose `layout_signature` or sizes differ
  from the circuit's, naming the first block that differs.
- A snapshot is engine-independent: it holds physical quantities only.
  Controller state (last accepted `h`) is an optional extension added by
  `refactor-unified-transient-stepper`.
- Python pickle: version 3 is the tuple above. Versions 1 and 2 (10 and
  11 fields, `python/bindings.cpp:2545-2562`) are read as a legacy
  payload and converted against the circuit's layout at resume time,
  because only the circuit knows which block each named field belongs
  to. Removed with the views, one release later.

### D7. Outputs

- Generic, for every device and every category: pin currents
  `i_k = f_k + (a·q_k + r_k)` evaluated at commit, and `p = Σ v_k·i_k`.
  For a mode device, `i = g·v − i_offset` under the stamped mode; for a
  static device, the stamped law. This is the current KCL was solved
  with, not a reconstruction.
- `outputs()` adds the kind's own channels (stored charge, flux, torque,
  junction temperature when Phase 4 adds one).
- Storage in this change: an `f`-only device's current is a function of
  the stored `x`, so it is evaluated on demand from the trajectory. A
  device with a `q` row needs `q̇` at commit, so its pin currents are
  recorded per stored sample in a contiguous table
  (`n_pins` reals per such device). `add-probes-and-recorders` replaces
  this fixed policy with declared probes.
- `device_loss_summary` integrates `p` over time (trapezoid in `t`, not
  a sample mean: owner note *time-weight-averages-on-adaptive-grids*).

Extension point: `add-in-kernel-event-losses` reads the same pin
currents and voltages at commutation events.

### D8. AC and DC from `f` and `q`

```cpp
struct Linearization { sparse::Matrix G; sparse::Matrix C; };   // real, same pattern for every mask
Linearization linearize(const Circuit&, const Vector& x0, const Mask& m);
// G = G_lin(m) + Σ ∂f/∂x|x0 ,  C = C_lin + Σ ∂q/∂x|x0
```

- DC operating point: every Newton device is stamped with
  `Formula::Dc`; a `q`-only device is open, a flux device is a short,
  `PmsmMna` gets the DC stamp it lacks today (inferred from the mapping;
  to be confirmed by the test in tasks 9.2).
- `run_mna_sweep` solves `(G + jωC)·X = B` with the existing complex
  sparse LU. `C_lin` is the cache's `C_split`, which contains the mutual
  terms that the transient already uses.
- The refusal that `fix-mna-sweep-correctness` adds for Newton kinds is
  removed here, kind by kind, as each kind is ported.

### D9. Field table

```cpp
struct Field { std::string_view name; /* member pointer, erased */ ; std::string_view unit;
               Real def, lo, hi; bool required; };
```

One table per kind drives: the builder's keyword names and validation
(error names the device and the field), `components()` params, the YAML
keys, the stub signature and the device reference page. It removes
Python side tables such as `_shockley_params`
(`python/pulsim/_builder_ergonomics.py:388-421`). The hand-written
`add_*` methods stay as thin forwards; nothing is renamed.

Extension points: `add-parameter-sensitivities` uses the table as the
list of differentiable parameters; `add-external-device-abi` exports it
through the C ABI; `add-immutable-circuit-ir` binds named parameters and
expressions to fields.

## Performance constraints

- No virtual call and no `std::function` per device per iteration. The
  only indirect call left on the Newton path is the linear solve.
- Zero heap allocations per step and per Newton iteration; the AD
  scalars live on the stack (`R ≤ 5` for the built-in kinds).
- One stamp loop per kind; `jac_slot` writes, no search.
- Rollback is `std::copy` of one vector.
- Gate (from `restore-reference-benchmark-harness`): `n_solves`,
  factorisation and Newton-iteration counters unchanged for every
  harness circuit at each port; ns/step on audit rows 10–13 (Shockley
  buck 20.4 µs, MOSFET level 1 14.5 µs) not above the pre-port value.
- Compile time of the extension measured before step 4 and after step
  8; budget +15 %.

## Risks / Trade-offs

- **Reassociation moves Newton-device results.** → Port one kind per
  PR; state the measured maximum deviation against the previous build
  on the harness (budget 1e-10 relative, iteration counts equal);
  re-baseline only the goldens containing that kind.
- **A smooth law that loses digits** (the Langevin `coth − 1/x` case)
  makes AD and the hand tangent disagree. → Each ported kind gets a
  monotonicity and finite-difference Jacobian test down to 1e-12 of its
  operating range before the old stamp is deleted.
- **Internal unknowns change `state_size`** for Lauritzen and the tail.
  → Appended last; `res.states` column count is documented as not
  stable across versions; name-based accessors are the API.
- **The flux ratchet returns** if a port silently changes the row form.
  → `test_saturable_inductor_flux_closure.py` is the oracle of step 7
  and a unit test asserts the stamped diagonal equals `−a·L`.
- **Snapshot format change breaks stored pickles.** → Legacy reader for
  one release; refusal by name afterwards.
- **Template bloat.** → Kinds × formulas is at most 8 × 4 small loops;
  measured, with a budget.
- **Two systems coexist during the migration.** → The registry is
  introduced empty; a kind is in exactly one system at any commit, and a
  `static_assert` forbids a kind being both in the legacy refresh and in
  the registry's Newton list.

## Migration Plan

Preconditions: `add-engine-conformance-matrix` and
`restore-reference-benchmark-harness` are merged. Each step is one or
more PRs, shippable alone; the invariant is what CI must show.

| # | Step | Invariant |
|---|---|---|
| 1 | Put the five history objects in a tuple and replace the 99 named calls in RT and TB by folds (`for_each_history`). No type changes. | Bit-identical on the whole suite and on the harness goldens (pwl hashes; trbdf2 step counters). |
| 2 | One kernel function builds the whole refresh from the pool: the static base that `add-engine-conformance-matrix` already builds in the kernel, plus the folded stateful passes. Both wrapper chains go; the binding stops passing a callback. `nl_refresh` stays as a deprecated override of the static base. | Bit-identical; the Shockley clamp through `run_transient` with no callback reads 0.594 V (the Phase 0 test, kept). |
| 3 | One `DeviceState` vector under the existing histories (`to_flat` layouts become blocks); snapshot and restore become one copy; `SolverSnapshot` gains `dev_state`, `mask`, `layout_signature`; the named fields become views; pickle v3 with the v1 / v2 reader. | Bit-identical; `test_snapshot_resume*.py` unchanged; cross-engine resume test added. |
| 4 | Add `device/` (concepts, `StepCtx`, `stamp_fq`, `DeviceSet`, registry with an empty Newton list), the AD functions and `ad::chain`. Nothing uses it. | Unit tests only; no golden moves. |
| 5 | Port the algebraic kinds through `FromCurrentModel` (Shockley, `IdealDiode`, MOSFET L1, IGBT L1), then `NonlinearCapacitor` (`q` only). Delete their `refresh_*`. | Per-device tests green; deviation ≤ 1e-10 and equal iteration counts on the harness; conformance matrix unchanged. |
| 6 | Port `LauritzenDiode` and the IGBT tail (one internal unknown each). Delete their histories. | `test_trbdf2_stateful_devices.py`, the double-pulse energies (E_on 234 µJ with the Lauritzen freewheel, audit finding 3) and the tail energy 16.87 mJ within their test tolerances. |
| 7 | Port the flux family (atan, `FluxTable`, `GappedCore`, JA with `State` + `accept`, the saturable transformer branch), then `PmsmMna`. | `test_saturable_inductor_flux_closure.py`, the JA loop tests and the PMSM tests green; the xfail `test_engines_agree_below_saturation` keeps its status. |
| 8 | The legacy chain is now empty: delete both wrapper chains, `NonlinearRefreshFn`, the five history headers, the eight refresh headers, `has_nonlinear_devices()`; the capability table of the conformance matrix is computed from the traits. Remove the snapshot views (**BREAKING**). | `test_stateful_device_stamps_compose.py` extended to every pair of stateful kinds; a registry-only test kind runs on every engine. |
| 9 | Generic outputs: pin currents and power; delete the kind ladder at PY:1004-1188; loss summary lists Newton kinds. | `∫v·i` against a 1 mΩ sense resistor within 1e-4 for every Newton kind; existing `res.i` values for PWL and linear kinds unchanged to 1e-12. |
| 10 | `linearize`; `run_mna_sweep` on it; DC through `Formula::Dc`. | The five analytic cases of `models/p3_ac.py`; DC operating points of the harness unchanged to 1e-9. |
| 11 | Static and mode kinds through the registry: `assemble.hpp`, `dc_assemble.hpp`, `history_state.hpp` and the source overlays iterate the type list instead of switching on `StoredKind`. | `J`, `C` and `b` identical entry by entry for every mask of every harness circuit; pwl goldens bit-identical. |
| 12 | Field tables for every kind; builder validation, `components()`, YAML keys and stubs generated from them. | `components()` output unchanged; every shipped YAML loads; the stub check passes. |

Rollback: every step is a revert of its own PRs. Steps 5–7 keep the
old stamp in the tree until the port's PR is merged and the harness has
run nightly once against ngspice.

Order with the other Phase 3 changes: steps 1–3 are enough for
`refactor-unified-transient-stepper` to start; step 9 is what
`add-probes-and-recorders` needs; step 12 is what
`add-immutable-circuit-ir` uses for `to_yaml()`.

## Open Questions

1. Lauritzen and the IGBT tail as internal unknowns (one extra row
   each, `res.states` gains columns), or statically condensed so that
   `state_size` is unchanged? The proposal assumes internal unknowns.
2. Budget for Newton-device goldens: is 1e-10 relative, with one
   re-baseline per ported kind, acceptable, or must each port be
   bit-identical (which forbids AD for the hand-stamped kinds)?
3. Step 11 (static and mode kinds through the registry) overlaps
   `update-linear-time-assembly`. Keep it here, after that change, or
   split it into its own change?
4. How long do the snapshot views and the pickle v1 / v2 reader live:
   one minor release, or until 3.0?
5. Should the default pin-current table for `q` devices (D7) be on by
   default before `add-probes-and-recorders` lands, given its memory
   cost of `n_pins` reals per such device per stored sample?
