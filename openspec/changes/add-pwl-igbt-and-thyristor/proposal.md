## Why

The PWL device set is the fast path of Pulsim and the one PSIM and
PLECS users compare against, and two of its members are not what their
names say.

**`add_igbt` is a symmetric resistor pair.** It forwards to `add_switch`
with `1/R_on` and `1/R_off`
(`core/include/pulsim/builder/circuit_builder.hpp:1244-1254`). Measured
on this branch (`probes/igbt_onstate.py`, gate ON, control included):
`v_CE = 0.100 V` at +10 A and `−0.100 V` at −10 A. A real IGBT has a
collector-emitter knee (`V_ce0`) and does not conduct in reverse. Two
consequences:

- Conduction loss is `R_on·I_rms²` only. The datasheet form is
  `V_ce0·I_avg + r_ce·I_rms²`; the first term is absent. For a device
  with a 0.9 V knee carrying 10 A average that is 9 W not accounted
  for, and a loss model missing it can be made to "fit" by tuning
  another parameter.
- A circuit whose freewheel diode was forgotten runs anyway, with the
  load current flowing backwards through the transistor.

**The thyristor is a Python closure.** `make_thyristor_switch_fn`
(`python/pulsim/switchgear.py:84-156`) keeps the latch in a Python dict,
updates it once per step in a `step_observer`, and drives an ordinary
bidirectional `add_switch`; it runs on `pwl` only. Its turn-off test is
`|i| < i_holding + 1e-12` on a sampled current, with `i_holding = 0` by
default. Measured (`probes/thyristor_closure.py`, half-wave rectifier,
100 V peak, 10 Ω, dt = 10 µs, α = 30°):

| Line frequency | `i_holding` | Mean load voltage | Analytic `V_m(1+cos α)/2π` |
|---|---|---|---|
| 50 Hz (zero crossing on the grid) | 0 | 29.700 V | 29.699 V |
| 60 Hz (zero crossing off the grid) | 0 | **−2.140 V**, load swings to −99.99 V | 29.699 V |
| 60 Hz | 0.05 A | 29.687 V | 29.699 V |

With the default holding current the "thyristor" never turns off unless
a sample lands within 1 pA of zero; it then conducts the whole negative
half-cycle. A C++ chain block (`switchgear/switchgear_adapters.hpp`)
has the same explicit, per-step structure.

Other tools (general knowledge, not checked against current
documentation): PSIM and PLECS both ship IGBT, thyristor, GTO and TRIAC
as ideal-switch devices with a forward drop and an on-resistance, and
line-commutated converters are a standard use of both.

## What Changes

- **`GatedDiode`** kernel device (`models/gated_diode.hpp`): a PWL
  diode with forward drop whose conduction also depends on a gate bit.
  Three modes share one law, one stamp and one set of decision sites:
  - `gate_held` — conducts while the gate is on **and** the diode rule
    holds. This is the PWL IGBT (and a GTO whose mask bit is its gate
    drive state).
  - `latching` — fires when the gate is on and the device is forward
    biased; stays on after the gate is removed while `i > I_H`. This is
    the thyristor (SCR).
  - legacy `add_switch` behaviour is untouched.
- **`add_igbt(name, collector, emitter, R_on=10e-3, R_off=1e9, *,
  V_ce0=0.0, reverse_blocking=True, V_th=0.0)`**: on-state
  `v = V_ce0 + R_on·i` for `i > 0`, blocking otherwise.
  **BREAKING** (staged): `reverse_blocking` defaults to `False` with a
  `FutureWarning` for one release and to `True` after it, at which
  point an IGBT no longer conducts in reverse.
  `reverse_blocking=False` with `V_ce0 = 0` is bit-identical to today
  at every stage.
- **`add_thyristor(name, anode, cathode, R_on=1e-3, R_off=1e9, *,
  V_f=0.0, I_H=0.0, V_th=0.0)`**, **`add_gto(...)`** and
  **`add_triac(...)`** (two anti-parallel latching branches).
- The gate stays a bit of the `switch_fn` mask, at the device's existing
  switch index; the conduction state is decided by the kernel like a
  diode's and recorded in the result.
- Usable on `pwl`, `trbdf2` and `dsed`, in the DC operating point, and
  through YAML (`igbt` gains `V_ce0`, `reverse_blocking`; new types
  `thyristor`, `gto`, `triac`).
- `res.i`, `device_loss_summary` and `device_thermal_summary` use the
  law (`V_ce0·i + i²·R_on` while conducting).
- A C++ phase-control firing schedule for line-commutated bridges
  (`sources/phase_control_fn.hpp`), so the gate signals do not need a
  Python callback.
- `make_thyristor_switch_fn` and `add_thyristor_to_chain` are
  deprecated with a warning that names `add_thyristor`.

## Impact

- Affected specs: `device-models`.
- Affected code: new `models/gated_diode.hpp`,
  `sources/phase_control_fn.hpp`; `stamping/stamp_switch.hpp` (reused
  `v_drop_on`), `pwl/{device_pool,diode_event_state,assemble,dc_assemble}.hpp`,
  `solver/{run_transient,trbdf2_transient,result}.hpp`,
  `dsed/{scheduler,event_predictor}.hpp`, `builder/circuit_builder.hpp`,
  `yaml/loader.hpp`, `python/bindings.cpp`,
  `python/pulsim/{_diode_state,losses,thermal,switchgear,topology}.py`,
  stubs, docs.
- Dependencies:
  - `add-engine-conformance-matrix` (Phase 0, the roadmap's
    prerequisite): the new kinds enter the matrix; no silent cell.
  - `add-switched-diode-forward-drop` (on this branch): its one-law
    helpers (`turn_on_voltage`, `branch_current`, `decide`,
    `crossing_signal`) and the `v_drop_on` stamp are reused.
  - `update-composite-diode-forward-drop` (Phase 0) owns the drop of
    the co-packaged freewheel diode; this change adds no diode to
    `add_igbt`.
  - `add-fixed-step-event-splitting` (Phase 2) improves the timing of
    natural commutations on the fixed engine; not required.
  - `add-unified-event-queue` (Phase 3) will replace the firing
    schedule with a declarative modulator.
  - `add-in-kernel-event-losses` gives these devices a loss port.
  - Does not need `add-unified-device-interface`: PWL devices stay a
    separate category there.
- Can ship on today's structure in full.
