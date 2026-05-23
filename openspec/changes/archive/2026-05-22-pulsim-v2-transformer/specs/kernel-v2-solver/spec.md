## ADDED Requirements

### Requirement: TwoWindingTransformer device model

A `TwoWindingTransformer` model SHALL be available in `pulsim::v2::models` with a `Params` struct exposing primary self-inductance `L_p` [H], secondary self-inductance `L_s` [H], and coupling coefficient `k` ∈ [0, 1].

The model MUST expose helpers:
- `mutual_inductance(params)` returning `M = k · √(L_p · L_s)`.
- `cross_dt(params, dt)` returning `2·M / dt`, the cross-coupling matrix entry magnitude used in trap-companion stamping.

The model is linear and dynamic; it does NOT carry a `current()` function (the constitutive relations are stamped via cross-coupling, not per-branch `i = f(v)`).

#### Scenario: Mutual inductance for perfect coupling

- **GIVEN** a `TwoWindingTransformer::Params{ L_p = 4e-3, L_s = 1e-3, k = 1.0 }`
- **WHEN** the user calls `mutual_inductance(params)`
- **THEN** the result SHALL equal `2e-3` (= 1·√(4e-3·1e-3)) within numerical precision.

#### Scenario: cross_dt is symmetric under swap

- **GIVEN** transformer params with `L_p = 1mH`, `L_s = 4mH`, `k = 0.95`, `dt = 10µs`
- **WHEN** the user computes `cross_dt(params, dt)`
- **THEN** the result SHALL equal `2 · 0.95 · √(L_p · L_s) / dt` to within FP tolerance.
- **AND** `cross_dt` MUST be invariant under swapping `L_p` and `L_s` (since `√(L_p · L_s) == √(L_s · L_p)`).

### Requirement: DevicePool transformer coupling registry

`DevicePool::add_transformer_coupling(p_branch_id, s_branch_id, params)` SHALL register a coupling pair between two already-added inductor branches.

The pool MUST expose `transformer_couplings()` returning a const ref to the list of registered `TransformerCoupling` entries (each holding the two branch IDs and the model params).

Registering a coupling MUST NOT modify the underlying inductor entries — the cross-coupling is purely an additive overlay applied during `assemble` and `compute_b_extra`.

#### Scenario: Registering a coupling preserves the inductors

- **GIVEN** a `DevicePool` with two inductor branches at IDs 0 and 1 (L_p=1mH, L_s=2mH)
- **WHEN** the user calls `add_transformer_coupling(0, 1, {L_p=1mH, L_s=2mH, k=0.9})`
- **THEN** `transformer_couplings()` SHALL contain exactly one entry with the registered branch IDs and params
- **AND** `inductor_params(0)` and `inductor_params(1)` SHALL be unchanged.

### Requirement: CircuitBuilder.add_transformer helper

`CircuitBuilder::add_transformer(name, p_from, p_to, s_from, s_to, L_p, L_s, k)` SHALL add a two-winding linear transformer to the circuit.

The helper MUST:
1. Create an inductor branch for the primary (`p_from → p_to`, inductance `L_p`).
2. Create an inductor branch for the secondary (`s_from → s_to`, inductance `L_s`).
3. Register the coupling via `DevicePool::add_transformer_coupling`.

The default `k = 1.0` (perfect coupling) MUST apply when omitted.

#### Scenario: add_transformer creates two branches plus a coupling

- **GIVEN** a fresh `CircuitBuilder`
- **WHEN** the user calls `add_transformer("T1", "p+", "p-", "s+", "s-", 1e-3, 4e-3, k=1.0)`
- **THEN** `num_branches` SHALL be 2 (one primary inductor + one secondary inductor)
- **AND** `pool().transformer_couplings()` SHALL have exactly 1 entry whose `params.L_p == 1e-3`, `params.L_s == 4e-3`, `params.k == 1.0`.

#### Scenario: k=0 isolates primary and secondary

- **GIVEN** a transformer registered with `k = 0`
- **AND** a sinusoidal voltage source on the primary side with a load on the secondary
- **WHEN** the simulation runs over several cycles
- **THEN** the secondary-side load current SHALL be within numerical noise of zero (no signal couples through).
