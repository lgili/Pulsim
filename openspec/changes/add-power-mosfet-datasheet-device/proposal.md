## Why

Pulsim can already compute a device-level hard-switching transient, but
only if the user assembles the device by hand, and no shipped part or
importer uses it.

Measured on this branch (review probe `models/p4_dpt.py` under
`core_review_2026-11/`, re-run 2026-11-12; double pulse at 400 V /
10 A, `add_mosfet_level1` with K = 1.43 A/V², V_T = 3 V):

| Configuration | E_on | E_off | Miller plateau |
|---|---|---|---|
| Bare device, R_g = 10 Ω or 100 Ω | 3.33 µJ | 1.53 µJ | none (0 ns voltage fall) |
| + external linear C_gs = 1 nF, C_gd = 100 pF, R_g = 10 Ω | 136.1 µJ | 159.0 µJ | 64.8 ns |
| Same, R_g = 47 Ω | 151.5 µJ | 748.6 µJ | 151.2 ns |
| + Lauritzen freewheel diode | 234.3 µJ | 158.9 µJ | 46.0 ns, I_peak 31.6 A |

- The hand estimate `Q_gd / I_g` = 40 nC / 0.636 A = 62.9 ns; the
  simulation gives 64.8 ns (+3 %).
- `trbdf2` gives 136.22 µJ in 1238 steps against 136.14 µJ in 7501 on
  `pwl`: the two engines agree to 0.06 %.
- The bare device ignores R_g because its gate is a node reference with
  no current (`core/include/pulsim/builder/circuit_builder.hpp:155-162`,
  `models/mosfet_level1.hpp:120-152`).

What is missing:

- **No packaged device.** There is no `add_power_mosfet`. The only
  voltage-dependent capacitance is the junction law
  `C0/(1 + v/V0)^m` (`models/nonlinear_capacitor.hpp:50-60`); a
  datasheet C_rss(V_DS) curve, which falls two to three decades, has no
  representation.
- **The SPICE importer discards the dynamics.** `_vdmos` appends RG,
  CJO, TT, CGDMIN, CGDMAX, A, CGS, M and VJ to the drop list
  (`python/pulsim/spice_import.py:829-836`) and builds an ideal-gate
  Level-1 device.
- **No real part.** `python/pulsim/lib/data` holds two catalog parts and
  four cores, all `provenance.method: synthetic`, refused by part number
  unless `allow_synthetic=True`. The PLECS importer refuses every
  formula (`python/pulsim/lib/plecs_xml.py:122-165`).
- **No device model has an external reference.** All model tests use
  analytic or self-consistency references (review `raw/models.md`,
  item 14); ngspice 45.2 is installed and
  `benchmarks/tools/benchmark_ngspice.py` exists.

Other tools (general knowledge, not checked against current
documentation): SPICE simulators run vendor subcircuits and the VDMOS
model with these capacitances; PSIM offers a datasheet-parameterised
"level 2" MOSFET; PLECS deliberately stays with ideal switches plus
loss tables. A datasheet device that runs on the PWL-cache engine and
on TR-BDF2, validated per part, is not offered by the ideal-switch
tools.

## What Changes

- **`ChargeTable`** (`models/charge_table.hpp`), sibling of `FluxTable`:
  knots `(v_k, Q_k, C_k)`, cubic Hermite with the stored `C_k` as
  tangents, so `C(v) = dQ/dv` is the exact derivative of the evaluated
  `Q(v)`; not odd (a junction is not symmetric); linear tails; refuses
  non-monotone data by name.
- **`NonlinearCapacitor` gains a charge-law variant**: `junction`
  (today, unchanged), `table` (`ChargeTable`), `vdmos_cgd` (the
  closed-form SPICE VDMOS law). One stateful device kind, no new
  wrapper in the refresh chain.
- **`CircuitBuilder.add_power_mosfet(name, drain, source, gate, ...)`**:
  channel (Level-1 law; `I_D(V_GS, V_DS)` table in a later step), drift
  resistance `R_d` [Ω], `C_gs` [F], `C_gd(v_dg)` and `C_ds(v_ds)` as
  charge laws, internal gate resistance `R_g` [Ω], optional common
  source inductance `L_s` [H], Lauritzen (or Shockley, or no) body
  diode, run-constant `T_j` [°C] acting through `R_DS(on)(T)` and
  `V_T(T)`.
- **YAML** type `power_mosfet` with the same keys, or `part:`.
- **VDMOS import keeps the charge**: CGS → `C_gs`; CGDMAX/CGDMIN/A →
  `vdmos_cgd`; CJO/VJ/M → `C_ds` junction law; TT → Lauritzen `tau`;
  RG → `R_g`. **BREAKING**: an imported VDMOS card that carries these
  now produces switching transients (and needs a time step that
  resolves them); `charge="drop"` restores the old device.
- **Datasheet fitters** in `python/pulsim/datasheet.py`, beside
  `fit_foster_from_zth` (`thermal.py:127`) and `fit_ja_from_bh_curve`
  (`hysteresis.py:545`): `fit_charge_table`, `fit_mosfet_channel`,
  `fit_rdson_temperature`, `fit_body_diode_recovery`,
  `power_mosfet_from_datasheet`. Each returns its residual.
- **`pulsim.double_pulse_test(...)`**: the standard half-bridge clamped
  inductive test; returns E_on, E_off, plateau time, dv/dt, di/dt,
  I_rr, with time-weighted integrals.
- **Part schema** gains a `power_mosfet` block with per-curve datasheet
  references, the datasheet double-pulse test condition, and disjoint
  `fitted_on` / `validated_on` lists. A part is `validated` only if its
  double-pulse acceptance check passes in CI at its declared tolerance.
- **PLECS formulas** of the "table and formula" kind are evaluated at
  import time for caller-supplied variable values
  (`import_plecs_xml(path, variables={"Rg": 10})`), producing a plain
  table and a provenance note.
- **ngspice cross-validation** of the Newton devices (Level-1 MOSFET
  static curves, Shockley diode, the three charge laws, VDMOS double
  pulse) in the nightly reference job.

## Impact

- Affected specs: `device-models`, `python-bindings`.
- Affected code: `models/{charge_table,nonlinear_capacitor}.hpp`,
  `pwl/{nonlinear_capacitor_history,nonlinear_refresh_nonlinear_capacitor,device_pool}.hpp`,
  `builder/circuit_builder.hpp`, `yaml/loader.hpp`,
  `python/bindings.cpp`, `python/pulsim/{spice_import,datasheet,lib/parts,lib/plecs_xml,__init__}.py`,
  stubs, `benchmarks/spice/ngspice`, docs.
- Dependencies:
  - `add-engine-conformance-matrix` (Phase 0) — needed first: today
    `dsed` and `bdf1` drop `NonlinearCapacitor` silently, so a device
    built on it would lose its capacitances there without an error.
  - `update-composite-diode-forward-drop` and
    `fix-nonlinear-diode-passivity` (Phase 0) own the PWL body-diode
    drop and the `IdealDiode` law; this change uses the Lauritzen and
    Shockley diodes and does not touch either.
  - `restore-reference-benchmark-harness` (Phase 0) owns the nightly
    ngspice job that the cross-validation netlists plug into.
  - `add-unified-device-interface` (Phase 3) is the roadmap's listed
    prerequisite. It is required for steps 4–5 only (table channel as
    one f/q device, generic `res.i` and loss rows, AC). Steps 1–3
    (charge law, composite helper, importer, fitters, harness, parts)
    need one new law variant inside an existing stateful device and no
    new history class, so they can ship on today's structure.
  - `add-in-kernel-event-losses` consumes the same part files; a live
    `T_j` for this device arrives there and with the device interface.
- Existing `add_mosfet`, `add_mosfet_level1` and `add_nonlinear_capacitor`
  calls are bit-identical.
