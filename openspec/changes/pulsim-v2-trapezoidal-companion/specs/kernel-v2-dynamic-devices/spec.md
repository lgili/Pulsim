## ADDED Requirements

### Requirement: Capacitor Device Model — Trapezoidal Companion

`pulsim::v2::models::Capacitor` SHALL be a 2-terminal dynamic
device that uses the trapezoidal companion model. The constitutive
relation `i_C = C · dv_C/dt` is approximated by the trap rule,
yielding at each timestep:

```
i_{n+1} = G_eq · v_{n+1} − I_hist
where
    G_eq    = 2C / dt
    I_hist  = G_eq · v_n + i_n
```

The struct SHALL expose:

```cpp
struct Capacitor {
    struct Params { Real C; };  // farads
    static constexpr topology::BranchKind kind =
        topology::BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_dynamic = true;

    static Real g_eq(Real dt, const Params& p) noexcept;
    static Real history_term(Real v_prev, Real i_prev,
                              Real dt, const Params& p) noexcept;
    template <numeric::FloatingPoint S>
    static constexpr S current(const S* /*v*/,
                                const Params& /*p*/) noexcept;
};
```

The `current` template MUST return 0 (the companion stamping
handles dynamic devices separately).

#### Scenario: g_eq for a 1 µF cap at 1 µs dt equals 2 S

- **GIVEN** `Capacitor::Params{.C = 1e-6}` and `dt = 1e-6`
- **WHEN** the user evaluates `Capacitor::g_eq(dt, params)`
- **THEN** the result SHALL equal `2.0` siemens.

#### Scenario: history_term reproduces companion form

- **GIVEN** `v_prev = 10`, `i_prev = 0.5`, `dt = 1e-6`,
  `C = 1e-6`
- **WHEN** the user evaluates `Capacitor::history_term(v, i,
  dt, p)`
- **THEN** the result SHALL equal `20.5`
  (= g_eq · v_prev + i_prev = 2 · 10 + 0.5).

#### Scenario: is_dynamic flag is true

- **WHEN** the consumer evaluates `Capacitor::is_dynamic`
- **THEN** it SHALL be `true` so Layer 4 dispatch can route
  it through companion stamping.

### Requirement: Inductor Device Model — Trapezoidal Companion

`pulsim::v2::models::Inductor` SHALL be a 2-terminal dynamic
device that uses the trapezoidal companion model in branch-current
formulation. The relation `v_L = L · di_L/dt` becomes (at each
timestep n → n+1):

```
v_{n+1, from} − v_{n+1, to} − (2L/dt) · (i_{n+1} − I_hist,L) = 0
where
    I_hist,L = i_n + (dt/2L) · v_n
```

The struct SHALL expose:

```cpp
struct Inductor {
    struct Params { Real L; };  // henries
    static constexpr topology::BranchKind kind =
        topology::BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_dynamic = true;

    /// Reciprocal of `g_eq` — used in the constraint row.
    /// g_eq_inv = dt / (2L).
    static Real g_eq_inv(Real dt, const Params& p) noexcept;

    static Real history_term(Real v_prev, Real i_prev,
                              Real dt, const Params& p) noexcept;
    template <numeric::FloatingPoint S>
    static constexpr S current(const S* /*v*/,
                                const Params& /*p*/) noexcept;
};
```

The `current` template MUST return 0.

Inductors require a branch-current unknown in the MNA state
vector (analogous to voltage sources). The DevicePool MUST track
the relative offset and add it to `state_size`.

#### Scenario: g_eq_inv for 1 mH at 1 µs dt equals 5e-4 Ω

- **GIVEN** `Inductor::Params{.L = 1e-3}` and `dt = 1e-6`
- **WHEN** the user evaluates `Inductor::g_eq_inv(dt, params)`
- **THEN** the result SHALL equal `5e-4`.

#### Scenario: history_term combines i_prev and v_prev correctly

- **GIVEN** `v_prev = 12`, `i_prev = 2`, `dt = 1e-6`,
  `L = 1e-3`
- **WHEN** the user evaluates `Inductor::history_term(v, i,
  dt, p)`
- **THEN** the result SHALL equal `2.006`
  (= i_prev + (dt/2L) · v_prev = 2 + 5e-4 · 12).
