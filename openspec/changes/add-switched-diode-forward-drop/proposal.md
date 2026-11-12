## Why

The PWL `SwitchedDiode` was a pure conductance in both states: `V_th`
only gated the OFF → ON decision and a conducting diode dropped
`i/g_on`. PSIM, which the owner uses as the reference, models a diode as
an ideal switch plus a series "Diode Voltage Drop". Users who put a
datasheet 0.7 V into Pulsim got no conduction drop and no `V_f·i` loss,
and `add_three_phase_vsi(V_f=0.7)` named a parameter `V_f` that was in
fact passed as the threshold `V_th`.

## What Changes

- `SwitchedDiode::Params` gains `V_f` (default 0): while ON the diode is
  `g_on` in series with `V_f` (`i = g_on·(v − V_f)`); OFF stays `g_off`
  with no offset. It turns on at `V_on = max(V_th, V_f)` (`V_th` when
  `V_f = 0`) and off when the ON current reaches zero (`v = V_f`).
- One law (`turn_on_voltage`, `branch_current`, `decide`,
  `crossing_signal`) is used by every decision site: the pwl event
  iteration, DC operating point, seeding, the pwl sub-step watcher, the
  TR-BDF2 crossing / held / forbidden checks, and the DSED predicates and
  settle.
- The stamp adds `∓g_on·V_f` to `b` on the ON mask only; `J` and its
  sparsity are unchanged, so every cached factorisation stays valid.
- `CircuitBuilder.add_diode(..., V_f=0.0)` (validated: finite, ≥ 0,
  error names the diode), the YAML `diode` key `V_f`, and `V_f` in
  `components()` params.
- Python: `res.i`, `device_loss_summary` (new `V_f` key, `P_avg`
  includes `V_f·i`), `device_thermal_summary` and the loss window cuts
  use the same law.
- `add_bridge_rectifier` gains `V_f=0.0` (drop, passed through; `V_th`
  default unchanged). `add_three_phase_vsi` gains `V_th=0.0`.
- **BREAKING:** `add_three_phase_vsi(V_f=0.7)` now passes `V_f` as the
  drop (pure PSIM, `V_th = 0`). Freewheel-diode loss rises by
  `V_f·I_avg` and a diode no longer shares current with its gated-on
  switch while `R_on·|i| < V_f`.
- `add_mosfet(V_F=)`, the Level-1 body diode and the IGBT freewheel
  diode keep their behaviour; their docs now say `V_F` is a threshold.

## Impact

- Affected specs: `device-models`, `transient-timestep`,
  `python-bindings`, `netlist-yaml`.
- Affected code: `models/switched_diode.hpp`, `stamping/stamp_switch.hpp`,
  `pwl/{assemble,dc_assemble,device_pool,diode_event_state}.hpp`,
  `solver/{run_transient,trbdf2_transient,result}.hpp`,
  `dsed/{scheduler,event_predictor}.hpp`, `builder/circuit_builder.hpp`,
  `yaml/loader.hpp`, `python/{bindings,dsed_bindings}.cpp`,
  `python/pulsim/{_diode_state,losses,thermal,topology,__init__}.py`,
  `AI_GUIDE.md`, docs, stubs.
- `V_f = 0` (every existing `add_diode` call) is bit-identical.
- Zenodo DOI and a drop for body / freewheel diodes are out of scope.
