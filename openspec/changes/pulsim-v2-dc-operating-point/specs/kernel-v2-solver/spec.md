## ADDED Requirements

### Requirement: dc_assemble — DC operating-point MNA matrix

`pulsim::v2::pwl::dc_assemble(graph, pool, mask, J, b)` SHALL
build the DC MNA matrix for the given switch state by:

1. Stamping Resistor / VoltageSource / Switch / Diode branches
   normally (their conductance stamps are dt-independent).
2. SKIPPING Capacitor branches (g_eq = 0 at DC, no contribution
   to J or b).
3. Stamping Inductor branches as a short-circuit constraint:
   `v_from − v_to = 0` with the inductor's branch-current
   unknown still present in `J` (so the DC i_L is recoverable
   from the solution).

`compute_dc_op(graph, pool, mask)` SHALL solve the DC system
using `make_default_solver()` and return the state vector. It
MUST throw `std::runtime_error` if the matrix is singular.

#### Scenario: V-R-GND DC OP

- **GIVEN** a graph with V_dc(5 V) → R(1 Ω) → GND and the
  empty switch mask
- **WHEN** the user calls `compute_dc_op`
- **THEN** the result SHALL have v_node ≈ 5 V at the V_dc node.

#### Scenario: V-R-C-GND DC OP

- **GIVEN** V_dc(5 V) → R(1 Ω) → n1 → C(1 µF) → GND
- **WHEN** the user calls `compute_dc_op`
- **THEN** v_n1 SHALL equal 5 V (cap fully charged at DC).

#### Scenario: V-R-L-GND DC OP

- **GIVEN** V_dc(12 V) → R(1 Ω) → L(10 µH) → GND
- **WHEN** the user calls `compute_dc_op`
- **THEN** i_L SHALL equal V_dc / R = 12 A
- **AND** v at the L-side of R SHALL be 0 (L is a short at DC).

### Requirement: run_transient — start_from_dc_op flag

`run_transient` SHALL accept an optional final parameter
`bool start_from_dc_op = false`. When `true`, the function MUST:

1. Get `initial_mask = switch_fn(t_start)`.
2. Run the diode-consistency iteration on the DC system to find
   the consistent initial diode state.
3. Compute `dc_x = compute_dc_op(graph, pool, mask)`.
4. Seed `HistoryState` from `dc_x` (caps → v_prev = v_C,
   i_prev = 0; inductors → v_prev = 0, i_prev = i_L).
5. Record sample 0 as `(t_start, dc_x)`.
6. Continue from sample 1 with the V2.1 trap-rule loop.

When `false`, the behaviour MUST be bit-identical to Layer 5
V2.1.

#### Scenario: V2.1 behaviour preserved when flag is false

- **GIVEN** an existing test that calls run_transient without
  the new flag
- **WHEN** the test runs
- **THEN** the result SHALL be bit-identical to Layer 5 V2.1
  (regression).

#### Scenario: RC circuit with DC OP starts at steady state

- **GIVEN** V_dc(5V) → R(1Ω) → C(1µF) → GND with all-zero
  switch state
- **AND** `start_from_dc_op = true`
- **WHEN** the user runs a 10·τ transient
- **THEN** v_C(0) SHALL equal 5 V (DC steady state)
- **AND** v_C(t) SHALL remain within 1 % of 5 V for all t
  (no charge transient — already at steady state).
