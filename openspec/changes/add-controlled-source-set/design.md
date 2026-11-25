## Context

Review items: `raw/models.md` ranked item 9, synthesis M7. The existing
linear stamps are small free functions in `core/include/pulsim/stamping`
(`stamp_vcvs.hpp`, `stamp_ideal_transformer.hpp`,
`stamp_current_source.hpp`); the new ones are their siblings. The
kernel solves `J·x = −b`, with node rows as KCL and one extra row per
branch-current unknown.

## Goals / Non-Goals

- Goals: the four missing linear controlled elements; time-varying
  current sources evaluated in the kernel; YAML, builder, bindings;
  correct in every analysis.
- Non-Goals: nonlinear or expression-controlled (behavioural) sources;
  a PWL-table *voltage* source (it would reuse the same evaluator; left
  to a follow-up to keep the scope of this row); delay elements;
  controlled switches; an `add_ammeter` helper (a 0 V
  `add_voltage_source` is the sense element).

## Decisions

### Stamps

Notation: `v_c = v_cp − v_cn` control voltage; `i_c = x[col_c]` the
controlling branch's current unknown, positive from that branch's
`from` to its `to`; output nodes `from`, `to` (current outputs) or
`out_pos`, `out_neg` (voltage outputs). Current outputs follow
`stamp_current_source` (`stamping/stamp_current_source.hpp:50-60`): the
value leaves the source at `from`, so `f[from] −= i`, `f[to] += i`.

| Element | Law | Unknowns | Stamp |
|---|---|---|---|
| VCCS | `i = g_m·v_c` | 0 | `J[from,cp] −= g_m`, `J[from,cn] += g_m`, `J[to,cp] += g_m`, `J[to,cn] −= g_m` |
| CCCS | `i = β·i_c` | 0 | `J[from,col_c] −= β`, `J[to,col_c] += β` |
| CCVS | `v_op − v_on = r_m·i_c` | 1 (`i_o`) | KCL `±i_o` at the output nodes as for a voltage source; row: `J[row,op] += 1`, `J[row,on] −= 1`, `J[row,col_c] −= r_m` |
| Gyrator | `i_1 = g·v_2`, `i_2 = −g·v_1` (currents entering at `p1_pos`, `p2_pos`) | 0 | two VCCS of opposite sign: 8 entries |

- The gyrator is lossless: `v_1·i_1 + v_2·i_2 = g·v_1·v_2 − g·v_2·v_1
  = 0`. A capacitor `C` on port 2 appears at port 1 as an inductor
  `L = C/g²`; a resistor `R` as `1/(g²·R)`.
- All four are constant and independent of the switch mask: they enter
  every cached segment's `J` once, change its sparsity pattern at build
  time only, and make it non-symmetric (the LU does not assume
  symmetry).
- Parameters must be finite; `g = 0`, `g_m = 0`, `β = 0`, `r_m = 0`
  are accepted (a zero source). Errors name the element.

### Controlling branch

`control` must name a branch with a current unknown. The builder
resolves the column through the pool's existing lookups
(`branch_var_id_for_source`, `branch_var_id_for_inductor`,
`pwl/device_pool.hpp:646-680`). A resistor, capacitor or switch is
refused: "`R1` carries no current unknown; sense through a 0 V voltage
source, an inductor, a VCVS, a CCVS or a transformer secondary".

### Current-source waveforms

| Source | Value |
|---|---|
| sine | `i_dc + i_amplitude·sin(2π·f·t + φ)` |
| pulse | the trapezoidal pulse of `add_pulse_voltage_source` (same parameters and edge rules) |
| pwl | linear interpolation of `(t_k, i_k)`; before `t_0` the value `i_0`; after `t_last` the value `i_last`, or the table repeated with period `t_last − t_0` when `periodic` |

- The evaluators are the voltage sources' (`sources/sine_b_extra.hpp`,
  `sources/pulse_b_extra.hpp`) plus one new `PwlWaveform`; the value is
  written to the two node rows of `b` instead of a branch row.
- `PwlWaveform` refuses fewer than two knots, non-increasing `t`,
  non-finite values, and `periodic` with `i_0 ≠ i_last` (a jump at the
  wrap), naming the source.
- Each waveform exposes its breakpoints (pulse corners, table knots).
  The fixed engine samples at grid points (exact when the knots are on
  the grid, since the trapezoid integrates a piecewise-linear source
  exactly). `trbdf2` must not step across a breakpoint.
- DC operating point: the value at `t = 0`. AC sweep: a current source
  contributes no small-signal excitation unless it is the sweep's
  designated input.

### Engines

| Kind | `pwl` | `trbdf2` | `dsed` | DC | AC |
|---|---|---|---|---|---|
| VCCS, CCCS, CCVS, gyrator | in the cached `J` | in `J` | run or refuse | yes | in `J` |
| sine / pulse / pwl current source | sparse source pass | with breakpoints | run or refuse | value at 0 | input or zero |

`dsed` builds per-mask `(A, b)` in `dsed/builder_adapter.hpp` and today
rejects the VCVS and the DC current source. Whether its extractor takes
general constant stamps was not established in the review. The
requirement is therefore "runs and agrees, or refuses by name"; task
5.3 settles which.

### Outputs

`res.i(name)`: the source current for VCCS, CCCS and the current
sources (from the state and the waveform), the branch unknown for the
CCVS, and `res.i(name, port=1|2)` for the gyrator.

### Alternatives considered

- *Build VCCS as a VCVS driving a resistor.* Adds an unknown and a
  node, and cannot express the gyrator without a loop of two such
  constructions.
- *Gyrator in resistance form (`v_1 = r·i_2`, `v_2 = −r·i_1`).* Needs
  two branch unknowns; the conductance form needs none. `r = 1/g` is
  accepted as an alternative keyword.
- *Keep time variation in `b_extra_fn`.* A Python callback per step:
  slow, not thread-safe, and not available to every engine.
- *One generic "linear controlled source" with a 2×2 parameter
  matrix.* More general, harder to read in a netlist, and every tool
  the user knows names the four elements.

## Parameterisation & provenance

Ideal elements; parameters are the user's. A PWL table read from a
measured file keeps the file path in `components()` when loaded
through the YAML `file:` key. No data ships.

## Validation plan

All references are analytic.

| Check | Expected | Tolerance |
|---|---|---|
| VCCS: `g_m = 2 mS`, `v_c = 5 V`, 1 kΩ from the `from` node to ground | `+10.000 V` | 1e-12 |
| CCVS: 10 mA through a 0 V sense source, `r_m = 100 Ω` | `v_out = 1.000 V` | 1e-12 |
| CCCS: 1 mA sensed, `β = 50`, 100 Ω load | `5.000 V` | 1e-12 |
| Gyrator, `g = 0.01 S`, 50 Ω on port 2 | port-1 resistance 200 Ω | 1e-12 |
| Gyrator, `g = 0.01 S`, 1 µF on port 2, AC | `|Z_1| = 62.83 Ω` at 1 kHz, phase +90° | 1e-9 |
| Same, transient: 1 V step through 10 Ω | equals a 10 mH inductor (`i = 63.21 mA` at 1 ms) | 1e-9 against the inductor run on the same grid |
| Gyrator power | `v_1 i_1 + v_2 i_2 = 0` every step | 1e-12 of `|v_1 i_1|` |
| Sine current source into 10 Ω | `10·i(t)` at every grid point | 1e-12 |
| PWL current source, triangle 0 → 1 A → 0 over 2 ms into 1 mF | `Δv = 1.000 V` | 1e-9 (`pwl`, knots on the grid); 1e-6 (`trbdf2`, knots off the grid) |
| Pulse current source | same samples as the pulse voltage source across 1 Ω | 1e-12 |
| Engine agreement, each kind | `pwl` against `trbdf2` | 1e-6 |
| Existing circuits | bit-identical | exact |

## Risks / Trade-offs

- Controlled sources let a user write a singular or unstable circuit
  (a gain loop with no solution) → the LU's singular-matrix error names
  the row; no new handling.
- A CCVS or CCCS whose controlling branch is removed or renamed →
  resolved at build time, error names both elements.
- Non-symmetric off-diagonals in a mostly symmetric matrix → none of
  the solvers assumes symmetry; covered by the engine-agreement tests.
- Breakpoints on `trbdf2` before `update-trbdf2-error-control` → a
  local step clamp to the next breakpoint, removed when that change
  lands.
- `dsed` support is unknown → the conformance row cannot be silent.

## Migration Plan

1. Stamps, pool kinds, builder, DC and `pwl`; analytic tests.
2. `trbdf2`, AC sweep, `dsed` (support or refusal), conformance rows.
3. Waveform current sources; breakpoints.
4. Bindings, stubs, `res.i`, YAML, docs.

Nothing existing changes; rollback is removal.

## Open Questions

- Should the PWL-table voltage source be added in the same change, since
  it is one more builder method on the same evaluator?
- Gyrator parameter: `g` in siemens (the stamp's natural form) or `r`
  in ohms as the primary keyword? The magnetic domain will pass
  `g = 1/N`.
- Is a `dsed` refusal acceptable for these kinds in the first release?
