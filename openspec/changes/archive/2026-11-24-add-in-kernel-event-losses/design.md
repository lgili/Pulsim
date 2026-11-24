## Context

Review items: `raw/models.md` ranked item 5, synthesis M3. The post-hoc
pipeline is one of the things the review lists as already excellent
(stamped-mask records, exact per-step integrals, 3-D tables, PLECS
quadrant conventions, provenance refusal). This change keeps all of it
and moves the one step that cannot be done after the run — reading the
table at the temperature the run itself produces — into the kernel.

The thermal network already lives in the MNA: `add_cauer_thermal_network`
(`python/pulsim/thermal.py:308-354`) builds resistors between
consecutive nodes and a capacitor from each node to an ambient node
held by a voltage source, with temperature in °C as the node voltage and
heat in W as current.

## Goals / Non-Goals

- Goals: tables evaluated in C++ at each commutation; energy conserved
  into the thermal node; T_j closes the loop during the run on `pwl`,
  `trbdf2` and `dsed`; exact agreement with the Python path when T_j is
  held fixed.
- Non-Goals: drawing the switching energy from the electrical circuit
  (the PWL switch stays ideal, as in PLECS; efficiency adds the event
  energy separately); temperature-dependent Newton devices; formula
  evaluation for PLECS files (`add-power-mosfet-datasheet-device`);
  new thermal-network topologies; replacing the Python summaries in
  this change.

## Decisions

### Tables

`LossTable3D` and `ConductionTable` reproduce `LossTable.__call__`
(`python/pulsim/loss_tables.py:121-147`) operation for operation: index
and weight per axis with the weight allowed outside [0, 1] (linear
extrapolation), a single-point axis has weight 0, the multilinear sum,
and the result clamped at 0. Axes are magnitudes; PLECS quadrants are
folded by the importer as today (`lib/plecs_xml.py:217-234`).
Construction refuses a non-ascending or empty axis, a shape mismatch, a
negative or non-finite entry.

### Reading rule (one rule, every engine)

For a commutation of device `d` between the last accepted state with
the old mask, `x⁻` at `t⁻`, and the first accepted state with the new
mask, `x⁺` at `t⁺`:

| Event | Blocking voltage | Current | Temperature |
|---|---|---|---|
| turn-on | `|v_d(x⁻)|` | `|i_d(x⁺)|` | `T_j(x⁻)` |
| turn-off | `|v_d(x⁺)|` | `|i_d(x⁻)|` | `T_j(x⁻)` |

`E = table(v, i, T_j)`. This is the rule of `losses.py:1423-1433`; on
`pwl`, `x⁻` and `x⁺` are the samples the Python path reads, which is
what makes the parity requirement exact. A diode's turn-off reads its
recovery table by the turn-off row. Events that the kernel decides
(diodes, the PWL IGBT's natural turn-off) and events the gate commands
follow the same rule.

### Deposition

The energy enters the junction node as heat, once per event:

- `pwl`, `trbdf2`: as a rectangular pulse over the accepted step that
  follows `t⁺`: the junction node's charge balance for that step
  includes exactly `E`. For a node with thermal capacitance `C_1` and
  no other path this gives `ΔT = E / C_1` after the step.
- `dsed`: as a state jump at the event, `x⁺ ← x⁺ + Δ`, where `Δ` is
  the response of the thermal sub-network to a heat impulse `E` at the
  junction node (`E / C_1` on a Cauer junction capacitor).

Both conserve `E`; they differ by `O(h/τ_1)` in the first thermal time
constant. The injected energy is accumulated in compensated form and
compared with the sum of table values in the result (`1e-12`
relative).

The thermal network seen from the junction node must absorb an impulse:
there must be no purely resistive path from the junction node to a
temperature source. A network without thermal capacitance at the
junction is refused by name (its temperature rise would depend on the
step size).

### Conduction heat

- `electrical` (default): the heat is the device's own electrical
  dissipation over each step, computed with the exact per-step
  integrals the Python summary uses (`V_f·i + i²/g_on` for a current
  linear over the step), deposited over the next step. The heat equals
  the electrical energy removed from the circuit, so the energy balance
  closes. One step of lag; no energy error.
- `table`: the heat is `V_on(|i|, T_j)·|i|`, trapezoidal over the step.
  The electrical model is unchanged, so the electrical side and the
  heat can disagree; the result reports both.
- Electrical feedback (step 4, optional): when `|T_j − T_ref| >
  dT_refresh` (default 5 K) the device's `V_f` and `1/g_on` are
  re-fitted from the table at the present `T_j` and current range, and
  the cache is refactored parametrically
  (`electrothermal.py:36-47` describes the existing path). The number
  of refreshes is reported.

### State and events

No new unknowns. Per device: an accumulator for pending heat and the
event record. The thermal nodes are ordinary MNA nodes. Events are the
existing mask changes; with `add-unified-event-queue` the hook is the
queue's commutation callback.

### API

```python
b.set_device_losses("Q1", junction="Tj_Q1",
                    E_on=LossTable(...), E_off=LossTable(...),
                    V_on=ConductionTable(...), conduction="table")
b.set_device_losses("D1", Tj=125.0, E_off=erec_table)      # open loop
res.loss_events("Q1")     # t, kind, v, i, Tj, E  (structured array)
res.device_energy("Q1")   # {"E_on":, "E_off":, "E_cond":} in J
res.v("Tj_Q1")            # junction temperature, °C
```

YAML:

```yaml
- type: igbt
  name: Q1
  collector: c
  emitter: e
  losses:
    junction: Tj_Q1          # or  Tj: 125
    e_on:  {v: [...], i: [...], tj: [...], energy: [[[...]]]}
    e_off: {...}
    v_on:  {i: [...], tj: [...], v: [[...]]}
    conduction: table
```

### Alternatives considered

- *Keep everything post-hoc and iterate the run to a fixed point on
  T_j.* Rejected: it needs several full runs, cannot follow a load
  transient, and cannot feed temperature back during the run.
- *A Dirac state jump on every engine.* Rejected for the trapezoidal
  engines today: a jump restarts the trapezoid from an inconsistent
  derivative, which is the defect `add-discontinuity-restart-rule`
  addresses; the one-step pulse has no such restart. It can become the
  uniform rule once that change lands.
- *Smear the energy over the switching period (the current constant
  offset).* Rejected: it needs the period, and it is what makes the
  current path open-loop.
- *Clamp extrapolation to the table edge.* Rejected for parity and for
  the reason recorded in `loss_tables.py:30-35`.
- *Implicit conduction heat inside the same solve.* Rejected: `v·i` is
  nonlinear in the state and would make every thermal run a Newton
  problem for an `O(h/τ)` gain.

## Parameterisation & provenance

- Tables come from `LossTable.from_curves` (datasheet `E(I)` curves per
  `(V, T_j)`), from `import_plecs_xml` (vendor thermal description), or
  from a part file with a provenance block. Nothing in this change
  ships table data.
- The shipped catalog parts stay `synthetic`: their tables are
  `E = c·I` and proportional in `V`; `attach_losses` on one of them
  requires `allow_synthetic=True` and the result carries the label.
- A table with a single temperature knot cannot close the loop;
  `junction=` with such a table is accepted but warns that the loss
  does not depend on T_j.

## Validation plan

Analytic references (one Cauer stage, `C_th = 0.01 J/K`, `T_amb =
25 °C`, a switch commutating at 10 kHz with duty 0.5):

| Check | Expected | Tolerance |
|---|---|---|
| Deposition, adiabatic (`R_th = 1e9 K/W`), constant table `E_on = E_off = 1 mJ`, 10 events | `ΔT_j = 1.000 K` | 1e-6 relative |
| Injected energy against the sum of table values | equal | 1e-12 relative |
| Open loop (`Tj = 25`), `R_th = 1 K/W`: time-weighted mean of the junction node over one period after 100 ms | `25 + 1·20 W = 45.0 °C` | 0.05 K |
| Closed loop, `E(T) = 1 mJ·(1 + 0.01·(T − 25))`: same mean | `25 + 20/(1 − 0.2) = 50.0 °C` | 0.1 K |
| Conduction, `electrical`: 10 A through a diode with `V_f = 0.7 V`, `g_on = 20 S` | 12.0 W, `T_j = 37.0 °C` | 0.01 K |
| Conduction, `table`: `V_on = 1.0 + 0.002·(T − 25)` V at 10 A | `T_j = 25 + 10/(1 − 0.02) = 35.20 °C` | 0.01 K |
| Parity with `device_loss_summary` at fixed `Tj` | equal totals | 1e-9 (`pwl`), 1e-6 (`trbdf2`, `dsed`) |
| Engines against each other, closed loop | mean `T_j` | 0.2 K |

Means are time-weighted (`np.trapezoid`), never `.mean()` on an
adaptive grid.

External reference: a vendor PLECS thermal description run in PLECS
(loss and T_j of its demo model) is the comparison the review names.
No PLECS reference data is in the repository; obtaining it is an owner
decision (Open Questions). Until then the closed-loop result is
verified, not validated: the analytic fixed point and a manual
iteration of the Python path (run, read T_j, re-run with `Tj=`) are
checks of the implementation, not of the device data.

## Risks / Trade-offs

- Three engine loops implement the hook before the event queue exists →
  ship `pwl` first; conformance rows refuse the others until done.
- `dsed` rejects DC current sources today (`raw/models.md` inventory),
  so its heat path is new there → the jump form needs no source; the
  conduction heat between events needs the per-mode source vector. If
  it cannot be realised, `dsed` refuses with the conformance error
  rather than running open-loop.
- A very small step after an event makes the pulse a large power for
  one step → energy is still exact; the TR-BDF2 error controller must
  not count the thermal node's kink as an error to refine (the pulse is
  applied to the step already chosen).
- `conduction="table"` breaks the electrical/thermal energy identity →
  the result reports `E_cond_electrical` and `E_cond_table` side by
  side; electrical feedback (step 4) removes the gap.
- Parametric refactors cost time → bounded by `dT_refresh`; count in
  the result.

## Migration Plan

1. `LossTable3D`, `ConductionTable` in C++ with bindings; a test that
   evaluates 10 000 random points against the Python class (1e-12).
2. `set_device_losses`, reading rule, deposition and records on `pwl`;
   parity and the analytic checks.
3. `trbdf2` and `dsed` through `add-unified-event-queue`; YAML;
   `Part.attach_losses`.
4. Optional electrical feedback.
5. `device_loss_summary` / `device_thermal_summary` read kernel records
   when present (no behaviour change at fixed T_j).

Rollback: remove the `set_device_losses` call; nothing else changes.

## Open Questions

- Can the owner obtain a PLECS (or PSIM thermal-module) reference run
  for one vendor thermal description, so the closed loop has an
  external number?
- Should the event energy also be drawn from the electrical circuit
  (a charge taken from the DC link), or stay thermal-only as in PLECS?
- Default for `conduction`: `electrical` closes the energy balance;
  `table` is what a PLECS user expects. Which is the default when a
  `V_on` table is present?
- On `dsed`, is a refusal acceptable for the first release, or is
  closed-loop T_j on `dsed` required from the start?
