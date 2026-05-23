# pulsim-v2-vcvs-opamp Specification

## Purpose

The `VCVS` (Voltage-Controlled Voltage Source) model (Layer 2 V15) is a 4-terminal linear-controlled source enforcing the constraint:

```
V(out_pos) − V(out_neg) = Gain · (V(in_pos) − V(in_neg))
```

The two sense terminals `in_pos`, `in_neg` carry NO current (ideal high-impedance inputs). The two output terminals `out_pos`, `out_neg` form a Source-kind branch with a branch-current unknown (same MNA pattern as a `VoltageSource`).

`VCVS` is the building block for the ideal op-amp approximation: a `Gain` of `1e5 … 1e7` paired with a feedback network forces `V(in_pos) ≈ V(in_neg)` via the closed-loop virtual short. Without negative feedback the matrix is well-conditioned only at moderate gain (~1e3); higher gains require an external compensation pole.

Stamped LINEARLY in the PWL cache — no Newton refresh needed.

## Requirements

### Requirement: VCVS constraint stamping

The `stamp_vcvs(J, f, x, coord, in_pos_node, in_neg_node, branch_var_id, gain)` function SHALL stamp the following contributions:

KCL at output terminals:
- `f[out_pos] += i_branch`, `J(out_pos, branch_var_id) += 1`
- `f[out_neg] −= i_branch`, `J(out_neg, branch_var_id) −= 1`

Constraint row at `branch_var_id`:
- `f[branch_var_id] += (V_out_pos − V_out_neg) − gain·(V_in_pos − V_in_neg)`
- `J(branch_var_id, out_pos) += 1`, `J(branch_var_id, out_neg) −= 1`
- `J(branch_var_id, in_pos) −= gain`, `J(branch_var_id, in_neg) += gain`

Ground terminals SHALL be silently skipped (no stamp written to the ground row/col, but the constraint row still reads ground voltage as 0).

#### Scenario: Buffer (Gain = 1) tracks input

- **GIVEN** a VCVS with `gain = 1`, `in_pos` driven by a 2 V source, `in_neg` tied to ground, output loaded with 1 kΩ to ground
- **WHEN** the DC operating point is solved
- **THEN** `v_out_pos` SHALL equal 2 V within 1 nV

#### Scenario: Amplifier (Gain = 10) amplifies input

- **GIVEN** a VCVS with `gain = 10`, `in_pos` driven by 1 V, `in_neg` grounded
- **WHEN** the DC operating point is solved
- **THEN** `v_out_pos` SHALL equal 10 V within 1 nV

### Requirement: CircuitBuilder helper

`CircuitBuilder::add_vcvs(name, in_pos, in_neg, out_pos, out_neg, gain)` SHALL register a VCVS as a `Source`-kind branch from `out_pos → out_neg`, with the two sense nodes stored in the pool's VCVS metadata (separate from branch geometry).

#### Scenario: Builder records sense nodes

- **GIVEN** `add_vcvs("E1", "ip", "in", "op", "on", gain=100)`
- **WHEN** `pool.vcvs_input_nodes(branch_id)` is queried
- **THEN** it SHALL return `(node_id_of("ip"), node_id_of("in"))`
- **AND** `pool.vcvs_params(branch_id).gain` SHALL equal 100

### Requirement: LDO showcase integration

A linear-regulator (LDO) topology built from VCVS + R + C + Zener-style clamp SHALL produce a stable output close to its set-point under load transients within the validation tolerance.

#### Scenario: LDO with VCVS holds a set-point

- **GIVEN** the LDO showcase circuit (high-gain VCVS with negative feedback, internal R/C network, 12 V input, target 5 V output)
- **WHEN** the transient simulator runs to steady state
- **THEN** the output node SHALL settle within 5 % of the 5 V set-point
