# Design — `pulsim-v2-transformer` (Layer 2 V2)

## The coupled-inductors model

```
        i_p →     M     ← i_s
       ┌────┐ ─ ─ ─ ─ ─ ┌────┐
   v_p │ L_p│           │ L_s│ v_s
       └────┘ ─ ─ ─ ─ ─ └────┘
```

Constitutive equations:

```
v_p(t) = L_p · di_p/dt + M · di_s/dt
v_s(t) = M  · di_p/dt + L_s · di_s/dt
```

With `M = k · √(L_p · L_s)` and coupling coefficient
`0 ≤ k ≤ 1`:

- `k = 0`: independent inductors (no transformer action).
- `0 < k < 1`: real transformer with leakage.
- `k = 1`: idealised transformer (no leakage).

## Trap-companion discretization

Single inductor (existing v2 model):

```
v_n+1 + v_n  =  (2L/dt) (i_n+1 − i_n)
```

Coupled inductors — primary equation:

```
v_p,n+1 + v_p,n  =  (2L_p/dt) (i_p,n+1 − i_p,n)
                  + (2M/dt)  (i_s,n+1 − i_s,n)
```

Rearranging into MNA form `J·x + b_extra = 0`:

```
v_p,n+1 − (2L_p/dt)·i_p,n+1 − (2M/dt)·i_s,n+1
       + [v_p,n + (2L_p/dt)·i_p,n + (2M/dt)·i_s,n] = 0
                       └─────────── b_extra ────────┘
```

So compared to a single inductor, the primary's
constraint row gets:

1. An EXTRA `-(2M/dt)` entry on the J matrix at column
   `secondary_branch_var_id`.
2. An EXTRA `(2M/dt) · i_s,prev` term in
   `b_extra(p_row)`.

The secondary's constraint row gets the symmetric
contribution.

## Implementation map

### `models/transformer.hpp` (NEW)

```cpp
struct TwoWindingTransformer {
    struct Params {
        Real L_p;     // primary self-inductance [H]
        Real L_s;     // secondary self-inductance [H]
        Real k = 1.0; // coupling coefficient [unitless]
    };

    /// Mutual inductance M = k · √(L_p · L_s) [H].
    [[nodiscard]] static Real mutual_inductance(
        const Params& p) noexcept;

    /// Cross-coupling term used in J stamping:
    ///   J_cross_dt(p, dt) = 2·M / dt
    [[nodiscard]] static Real cross_dt(
        const Params& p, Real dt) noexcept;
};
```

### `device_pool.hpp` (MODIFIED)

Add a coupling registry (separate from the variant
`entries_` — couplings are PAIRS of branch_ids, not
single-branch entries):

```cpp
struct TransformerCoupling {
    Index primary_branch_id;
    Index secondary_branch_id;
    models::TwoWindingTransformer::Params params;
};

class DevicePool {
public:
    void add_transformer_coupling(
        Index p_branch_id, Index s_branch_id,
        const models::TwoWindingTransformer::Params& p);

    [[nodiscard]] const std::vector<TransformerCoupling>&
    transformer_couplings() const noexcept;

private:
    std::vector<TransformerCoupling> transformer_couplings_;
};
```

### `assemble.hpp` (MODIFIED)

After the per-branch stamping loop completes (so all
single-inductor diagonals are in place), iterate
`pool.transformer_couplings()` and stamp the cross-terms:

```cpp
if (dt > Real{0}) {
    for (const auto& tc : pool.transformer_couplings()) {
        const Index p_row = pool.branch_var_id_for_inductor(
            tc.primary_branch_id, graph);
        const Index s_row = pool.branch_var_id_for_inductor(
            tc.secondary_branch_id, graph);
        const Real cross_dt =
            TwoWindingTransformer::cross_dt(tc.params, dt);
        J.coeffRef(p_row, s_row) += -cross_dt;
        J.coeffRef(s_row, p_row) += -cross_dt;
    }
}
```

### `history_state.hpp` (MODIFIED)

`compute_b_extra(dt)` already iterates per-inductor
history entries. After that loop, iterate transformer
couplings and add cross-coupling contributions:

```cpp
for (const auto& tc : pool.transformer_couplings()) {
    const Index p_row = pool.branch_var_id_for_inductor(
        tc.primary_branch_id, graph);
    const Index s_row = pool.branch_var_id_for_inductor(
        tc.secondary_branch_id, graph);
    const Real cross_dt =
        TwoWindingTransformer::cross_dt(tc.params, dt);
    // i_prev for each inductor is cached in entries_.
    const Real i_p_prev = entry_for_branch(tc.primary_branch_id).i_prev;
    const Real i_s_prev = entry_for_branch(tc.secondary_branch_id).i_prev;
    b_extra[p_row] += cross_dt * i_s_prev;
    b_extra[s_row] += cross_dt * i_p_prev;
}
```

### `circuit_builder.hpp` (MODIFIED)

```cpp
CircuitBuilder& add_transformer(
    std::string name,
    std::string p_from, std::string p_to,
    std::string s_from, std::string s_to,
    Real L_p, Real L_s, Real k = Real{1}) {
    const Index p_from_idx = resolve_node_(p_from);
    const Index p_to_idx   = resolve_node_(p_to);
    const Index s_from_idx = resolve_node_(s_from);
    const Index s_to_idx   = resolve_node_(s_to);

    const Index p_branch = graph_.add_branch(
        p_from_idx, p_to_idx, BranchKind::PassiveLinear);
    pool_.add_inductor(p_branch, {L_p});

    const Index s_branch = graph_.add_branch(
        s_from_idx, s_to_idx, BranchKind::PassiveLinear);
    pool_.add_inductor(s_branch, {L_s});

    pool_.add_transformer_coupling(
        p_branch, s_branch,
        models::TwoWindingTransformer::Params{
            .L_p = L_p, .L_s = L_s, .k = k});

    (void)name;
    return *this;
}
```

## Test plan

### Unit tests

1. **`add_transformer` adds 2 branches** + 1 coupling.
2. **Coupling registry**: `pool.transformer_couplings()`
   returns the registered pair.

### Integration tests (V_dc step + sinusoid + flyback smoke)

3. **k=1 transformer 1:1 voltage step**: V_dc=10V on
   primary, R_load on secondary. Steady-state secondary
   voltage ≈ 10V (turns ratio 1, k=1).
4. **k=1 transformer 2:1 voltage ratio**: L_p = 4·L_s,
   k=1 (turns ratio 2). At sinusoidal steady state,
   |v_s| / |v_p| ≈ 1/2.
5. **k=0 isolation**: with k=0, the secondary side is
   electrically decoupled from the primary. A step on
   primary doesn't induce current on secondary.
6. **Flyback smoke**: build V_in + MOSFET + transformer
   + diode + output cap + load via CircuitBuilder; verify
   the system builds and factors without errors. NOT a
   full waveform validation (left for showcases).

### Python smoke

7. **Python `add_transformer` callable**: builds 2
   branches; cache.build() succeeds.

## What V0 deliberately does NOT do

- **Saturation modeling**: V0 treats L as constant. Real
  ferrite cores saturate at the knee; that's a future
  research OpenSpec (likely requires an AD-driven L(i)
  model).
- **Multi-winding transformer** (>2 windings): V0 is
  two-winding. Three-winding (e.g. push-pull centre-tap)
  is straightforward extension via more cross-couplings;
  V1.
- **Frequency-dependent core losses**: V0 is lossless
  except for the optional leakage from k < 1.
- **Hysteresis**: not modeled.
- **Magnetic-equivalent-circuit** approach (reluctance):
  V0 uses the inductor-pair approach instead.

## Files

- NEW `core/include/pulsim/v2/models/transformer.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/device_pool.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/assemble.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/history_state.hpp`
- MODIFIED `core/include/pulsim/v2/builder/circuit_builder.hpp`
- MODIFIED `python/bindings_v2_kernel.cpp`
- NEW `core/tests/v2/layer5_v1/test_transformer.cpp`
- MODIFIED `python/tests/v2/test_v2_python_bindings.py`
- NEW `docs/pulsim-v2/layer2-v2-transformer.md`
