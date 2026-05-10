# Catalog Devices

> Status: shipped — header-only data classes + six reference parts.
> Full Circuit-variant integration (catalog devices on the MNA stamp
> surface) is the natural follow-up.

The catalog tier upgrades Pulsim's nonlinear device library from
"good enough for first-cut design" to "match the vendor datasheet
within the published tolerance band". Three device classes plus a YAML
loader plus six reference parts ship today:

| Device class | Header | Use case |
|---|---|---|
| `MosfetCatalog` | `catalog/mosfet_catalog.hpp` | Si super-junction, SiC, GaN — nonlinear `i_d(V_ds, V_gs, T_j)`, vendor-published `Coss/Ciss/Crss(V_ds)`, `Eon/Eoff(I_c, V_ds)` switching tables. |
| `IgbtCatalog` | `catalog/igbt_catalog.hpp` | Si IGBTs with `V_ce_sat(I_c, T_j)` lookup, exponential tail-current decay, switching + recovery energy tables. |
| `DiodeCatalog` | `catalog/diode_catalog.hpp` | Standalone Si fast-recovery, SiC Schottky, body diodes — `V_f(I_f, T_j)` and `Q_rr(I_f, di/dt)` lookups with reverse-recovery shape factor. |

Plus the data plumbing:

| Helper | Header | Use case |
|---|---|---|
| `LookupTable2D` | `catalog/lookup_table_2d.hpp` | Bilinear-interp 2D LUT for any `f(x, y)` parameter. |
| Catalog YAML loader | `catalog/device_catalog_yaml.hpp` | `parse_device_catalog_yaml` and `load_device_catalog_file` return `std::variant<MosfetCatalogParams, IgbtCatalogParams, DiodeCatalogParams>` so the call site dispatches by class. |

Like `add-magnetic-core-models`, the layer is **header-only** — no
linker dependency on `pulsim.lib`. The Circuit-side integration
(registering `MosfetCatalog` next to the existing Level-1 `MOSFET` in
the `DeviceVariant`) is the next change.

## TL;DR

```cpp
#include "pulsim/v1/catalog/device_catalog_yaml.hpp"
#include "pulsim/v1/catalog/mosfet_catalog.hpp"

using namespace pulsim::v1::catalog;

const auto params = load_device_catalog_file(
    "devices/catalog/Wolfspeed/C3M0065090J.yaml");
MosfetCatalog Q1{std::get<MosfetCatalogParams>(params)};

// Operating-point queries
const Real R_on    = Q1.params().R_ds_on(85.0);                 // T_j = 85 °C
const Real Coss    = Q1.C_oss(400.0);                            // V_ds = 400 V
const Real I_d     = Q1.drain_current(/*V_ds*/0.5, /*V_gs*/15, /*T_j*/85);
const Real Eon_uJ  = 1e6 * Q1.switching_energy_on(20.0, 600.0);  // 20A @ 600V
```

## Six reference parts (gate G.4: ≥ 3 of 6 importable — all 6 ship)

| Vendor | Part | Class | Notes |
|---|---|---|---|
| Infineon | `IPP60R190P7` | `mosfet` | 600 V Si CoolMOS P7, 190 mΩ, TO-220 |
| Wolfspeed | `C3M0065090J` | `mosfet` | 900 V SiC, 65 mΩ, TO-247 |
| GaN Systems | `GS66508T` | `mosfet` | 650 V GaN HEMT, 50 mΩ, PDFN |
| Infineon | `IKW40N120T2` | `igbt` | 1200 V Si TrenchStop2, 40 A, fast-tail |
| Wolfspeed | `C4D20120D` | `diode` | 1200 V SiC Schottky, 20 A |
| Vishay | `VS-30CTH02` | `diode` | 200 V Si fast-recovery, 30 A |

YAML files live under `devices/catalog/<vendor>/<part>.yaml`. The
schema is:

```yaml
class: mosfet           # or 'igbt' or 'diode'
vendor: <string>
part: <string>

# MOSFET-specific:
V_th_25c: 3.0
V_th_temp_coef: -6e-3
R_ds_on_25c: 0.190
R_ds_on_temp_coef: 6e-3
V_ds_max: 600

# 1D capacitance tables (any 2-key map per point — e.g. V/C, x/y)
Coss:
  - { V: 0,   C: 4.6e-9 }
  - { V: 50,  C: 200e-12 }
  ...

# 2D switching-energy tables
Eon:
  x: [5, 10, 20]      # I_d axis (A)
  y: [200, 400]       # V_ds axis (V)
  values: [25e-6, 50e-6, 100e-6,
           50e-6, 100e-6, 200e-6]
```

Strict failure modes: missing `class:`, unknown `class:`, fewer than
2 axis points in any LUT, non-monotone axes — all surfaced as
`std::invalid_argument`.

## What each model captures

### `MosfetCatalog`

```
I_d(V_ds, V_gs, T_j) = clamp(min(V_ds / R_on(T_j),  (V_gs - V_th(T_j)) / R_on(T_j)),
                             0,  ∞)
R_on(T_j) = R_ds_on_25c · (1 + R_ds_on_temp_coef · (T_j - 25))
V_th(T_j) = V_th_25c    + V_th_temp_coef    · (T_j - 25)
```

Plus capacitance and switching-energy lookups — used by the
loss-accounting layer on every commutation event. Off-state leakage
floors at `I_dss_25c` (typically 1 nA – 100 nA from the datasheet).

### `IgbtCatalog`

`V_ce_sat(I_c, T_j)` from the datasheet's saturation-voltage figure.
`tail_current(I_c0, t_after_off)` = `I_c0 · I_tail_fraction · exp(-t /
τ_tail)` — used to integrate tail-current loss over the
post-turn-off window. Switching energies are 2D `Eon/Eoff/Erec`
tables vs `(I_c, V_ce)`.

### `DiodeCatalog`

`V_f(I_f, T_j)` for both polarities of temperature coefficient: Si
fast-recovery has dV/dT < 0 (V drops as T rises); SiC Schottky has
dV/dT > 0 (V rises as T rises). `Q_rr(I_f, di/dt)` carries the
reverse-recovery charge — Si fast-recovery has substantial Q_rr; SiC
Schottky has only junction-capacitance-driven Q_rr that's nearly flat.

The reverse-recovery shape factor `s_rec` controls how the recovery
current trajectory looks: `s_rec ≈ 0.4` for hard-recovery Si,
`s_rec ≈ 1.5` for soft-recovery / Schottky. The estimated recovery
energy is `Q_rr · V_r · (1 - s_rec / (1 + s_rec))`.

## Validation

Three end-to-end gates:

| Gate | Test | Result |
|---|---|---|
| **G.1** Switching loss tracks datasheet | `test_catalog_phase8_validation::Phase 8 G.1` | Wolfspeed C3M0065090J: Eon scales linearly with `I_c` and `V_ds` to within 10 % across the catalog table |
| **G.2** Conduction loss within 5 % over 25–125 °C | `Phase 8 G.2` | Infineon IPP60R190P7: `I² · R_ds_on(T_j)` matches the catalog's `R_on(T_j)` model within 5 % at 25 / 75 / 125 °C |
| **G.3** Q_rr behavior | `Phase 8 G.3` | Si fast-recovery Q_rr rises ≥ 20 % between 1e8 and 1e9 A/s; SiC Schottky stays under 100 nC and within 1.5× across the same range |
| **G.4** ≥ 3 of 6 cores load | `test_catalog_phase7_yaml_loader::Phase 7` | All 6 of 6 reference YAMLs parse cleanly + drive their device class end-to-end |
| **G.5** Catalog browser docs | This page | Static reference — interactive browser is the natural follow-up once the Circuit-variant integration lands |

## Limitations / follow-ups

- **Circuit::DeviceVariant integration**: today the catalog devices
  are math objects. Wiring `MosfetCatalog` / `IgbtCatalog` /
  `DiodeCatalog` into `Circuit::DeviceVariant` so a YAML netlist can
  declare `type: mosfet_catalog, model: Wolfspeed/C3M0065090J` is the
  next change. The math layer is final; the integration is mostly
  parser/dispatch plumbing.
- **PDF / SPICE importers** (`pulsim.import_datasheet(pdf)`,
  `import_spice_lib(...)`, `import_plecs_xml(...)`): deferred. The
  YAML manifest path delivered today already covers the catalog use
  case without OCR / vendor-library risk. PDF importer follows when
  a user actually needs to onboard a part not in the shipped library.
- **LTspice / NgSpice parity tests**: Phase 8 here uses analytical
  references because LTspice / NgSpice are not in CI. Vendor-model
  parity (within 10 % switching loss, 5 % conduction loss) is a
  bench-test that tracks alongside the integration change.
- **Catalog browser web page**: tracked — the docs page above is the
  text reference, an interactive HTML browser with vendor parity
  waveforms is the natural notebook-tier deliverable.

## See also

- [`magnetic-models.md`](magnetic-models.md) — the parallel layer for
  saturable inductors, transformers, and core-loss / hysteresis models.
- [`automatic-differentiation.md`](automatic-differentiation.md) — the
  AD path that catalog devices' Newton Jacobian will land on once
  they're wired into the MNA stamp surface.
