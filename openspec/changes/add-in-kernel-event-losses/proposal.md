## Why

Pulsim reads datasheet switching-energy tables correctly, but only
after the run, at a junction temperature the user must supply. The
temperature the tables need is the one the losses produce, so the loop
is open exactly where it matters.

Verified on this branch (review `raw/models.md`, "Where Pulsim stands on
semiconductors"; synthesis item M3):

- **T_j is an input, not a result.** A 3-D table `E(v, i, T_j)` is
  refused unless the caller passes `Tj` as a scalar, an array or a
  callable (`python/pulsim/losses.py:1315-1325`); the table is then read
  per edge from the measured blocking voltage and load current
  (`losses.py:1398-1412`, rule at `:1423-1433`).
- **Switching loss reaches the thermal network as a constant.**
  `device_thermal_summary` adds the run's average `P_sw_avg` to the
  conduction trace as a constant offset
  (`python/pulsim/thermal.py:622-626, 756-758, 811-813`).
- **Conduction tables are collapsed.** A PLECS on-state table
  `V_on(i, T_j)` becomes `V0 + r·i` at one temperature
  (`python/pulsim/lib/parts.py:442-450`).
- **Temperature reaches the matrix only for resistor branches**, on
  `pwl`, at a user cadence (`python/pulsim/electrothermal.py:1-47`). The
  lumped loop of `make_electrothermal_heatsink_observer`
  (`thermal.py:1389-1436, 1546`) scales reference powers by linear
  temperature coefficients; it does not read the table at a commutation.
- **What the temperature axis is worth.** `python/pulsim/loss_tables.py`
  records its own measurement on a 600 V / 100 A IGBT datasheet: the
  25 °C table read for a 125 °C junction under-reads the switching
  energy by 35.6 % (`loss_tables.py:9-23`).

Other tools (general knowledge, not checked against current
documentation): PLECS evaluates the thermal description's lookup tables
at every switching event from the pre- and post-switching voltage and
current and the present junction temperature, and injects the energy
into the thermal network as an impulse; PSIM's thermal module does the
equivalent with its device database. That is the feature a
power-electronics user compares against.

## What Changes

- **Loss tables in the kernel** (`core/include/pulsim/thermal/`):
  `LossTable3D` `E(v, i, T_j)` and `ConductionTable` `V_on(i, T_j)`,
  with the same semantics as `pulsim.loss_tables.LossTable` (rectilinear
  ascending axes, multilinear interpolation, linear extrapolation,
  result clamped at zero).
- **A thermal port on PWL switching devices** (switch, `add_mosfet`,
  `add_igbt`, PWL diode, and their body / freewheel diodes):
  `CircuitBuilder.set_device_losses(device, *, junction=None, Tj=None,
  E_on=None, E_off=None, V_on=None, conduction="electrical")`. Exactly
  one of `junction` (a thermal node of the circuit: closed loop) or `Tj`
  (°C, fixed: open loop) is given.
- **Event energy**: at each commutation of the device the kernel reads
  the blocking voltage, the commutated current and the junction
  temperature, evaluates the table, and deposits the energy into the
  junction node — exactly once, with the injected energy equal to the
  table value to 1e-12.
- **Conduction heat**: `conduction="electrical"` deposits the device's
  own electrical dissipation (closing the energy balance);
  `conduction="table"` deposits `V_on(|i|, T_j)·|i|` from the table.
- **Records in the result**: per device, the event list
  `(t, kind, v, i, T_j, E)` and the energy totals; `T_j(t)` is the
  junction node's waveform.
- **YAML**: a `losses:` block on switch-class components
  (`junction` | `Tj`, `e_on`, `e_off`, `v_on`, `conduction`, or
  `part:`), strict keys.
- **Parts**: `Part.attach_losses(builder, device, ...)` wires a part's
  tables and its Cauer chain (from `import_plecs_xml` or a part file) to
  a device.
- **Relation to the post-hoc path**: `device_loss_summary` and
  `device_thermal_summary` stay. With `Tj` fixed, the kernel totals
  equal theirs (1e-9 on `pwl`). In a later step the Python summaries
  read the kernel records when they exist, leaving one implementation
  of the reading rule.
- **Optional electrical feedback** (last step): `V_f` and `R_on` of the
  PWL device follow the on-state table as T_j moves, refreshed when
  T_j has changed by more than `dT_refresh`, through the existing
  parametric refactor.
- No **BREAKING** change: a circuit without `set_device_losses` is
  bit-identical.

## Impact

- Affected specs: `loss-thermal` (new capability).
- Affected code: new `thermal/{loss_table,device_losses}.hpp`;
  `solver/{run_transient,trbdf2_transient,result}.hpp`,
  `dsed/scheduler.hpp` (or the event queue, see below),
  `builder/circuit_builder.hpp`, `yaml/loader.hpp`,
  `python/bindings.cpp`,
  `python/pulsim/{losses,thermal,loss_tables,lib/parts}.py`, stubs,
  docs.
- Dependencies:
  - `add-unified-event-queue` (Phase 3) — the roadmap's prerequisite.
    The commutation is where the energy is read; today each engine
    detects it in its own loop, so shipping now means three
    implementations of the reading rule. Step 1 (tables in C++, parity
    tests) and step 2 on `pwl` alone can ship on today's structure;
    `trbdf2` and `dsed` should wait for the queue.
  - `add-engine-conformance-matrix` (Phase 0) — rows for
    `device_losses` × engine; an engine that cannot inject refuses.
  - `add-pwl-igbt-and-thyristor` — the PWL IGBT and thyristor take the
    same port.
  - `add-unified-device-interface` — Newton devices (which dissipate
    their switching loss electrically) join through its generic power
    output; out of scope here.
  - `add-power-mosfet-datasheet-device` — owns PLECS formula evaluation
    and the part schema this change reads.
