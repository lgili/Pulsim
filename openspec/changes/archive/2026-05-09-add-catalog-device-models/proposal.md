## Why

Pulsim's MOSFET model in `core/include/pulsim/v1/components/mosfet.hpp` is Level 1 Shichman-Hodges only — vintage circa 1985. It has no parasitic capacitances, no body diode, no temperature dependence, no SiC/GaN-specific behaviors. For switching-loss analysis, EMI prediction, gate-driver dimensioning, or thermal stress evaluation, this model is inadequate.

PSIM ships a device-of-the-month from major vendors. PLECS lets you import `.xml` device files directly from Wolfspeed, Infineon, ROHM. Pulsim today asks the user to use an under-fidelity Level-1 SHM and approximate the rest. To compete, we need a catalog tier with:

- Datasheet-calibrated nonlinear capacitances (Coss/Ciss/Crss as f(Vds))
- Body-diode model with reverse recovery charge (Qrr)
- Temperature-dependent on-resistance (Rds_on(Tj))
- SiC- and GaN-specific behaviors (no Qrr for GaN, hard-switching tail-current for IGBT)
- Importer for vendor SPICE / PLECS .xml files where licensing permits

This change adds the catalog tier without breaking the Level-1 simple models, and adds an importer pipeline (manual + datasheet-mining via the existing `datasheet-intelligence` skill).

## What Changes

### New Catalog Device Tier
- `MosfetCatalog` device variant with parameters:
  - `Rds_on_25C`, `Rds_on_temp_coef` (linear or table)
  - `Coss(Vds)`, `Ciss(Vds)`, `Crss(Vds)` lookup tables
  - `body_diode` sub-model with `Vf`, `Qrr`, `trr`, optional reverse-recovery shape
  - `Vth(Tj)` table or linear coefficient
  - `g_fs(Vgs, Tj)` transconductance table
- `IgbtCatalog` device variant with parameters:
  - `Vce_sat(Ic, Tj)` table
  - Tail-current model: `Itail`, `tau_tail`
  - Co-pack diode (separate `body_diode`)
- `DiodeCatalog` for standalone power diodes:
  - `Vf(If, Tj)` table
  - `Qrr(If, di/dt)` table

### Datasheet Importer (Manual + Mining)
- `pulsim.import_datasheet(pdf_path) -> DeviceParams` using the `datasheet-intelligence` skill workflow.
- `pulsim.import_spice_lib(lib_path, device_name)` parses `.lib` / `.cir` MOSFET subcircuits, extracts Level-3/EKV/BSIM parameters.
- `pulsim.import_plecs_xml(xml_path)` for circuits where licensing allows.
- Output: a `DeviceParams` dataclass that can be serialized to YAML for reproducibility.

### Curated Device Library (Initial Set)
- 6 reference devices spanning Si MOSFET, SiC MOSFET, GaN HEMT, Si IGBT, SiC Schottky, fast-recovery diode. Datasheet-calibrated, validated against vendor LTspice models within 5% on switching loss.
- Stored under `devices/catalog/<vendor>/<part>.yaml`.

### Validation against LTspice / NgSpice Vendor Models
- Each catalog device has parity test that runs vendor LTspice subcircuit on same scenario and compares switching waveform, peak Vds overshoot, conduction loss, switching loss within tolerance.

### Loss Calculation Integration
- Catalog devices feed correct switching-energy curves (`E_on(Ic, Vds)`, `E_off(Ic, Vds)`, `E_rr(If, di/dt)`) directly into the existing `losses.hpp` accumulator.
- Tj feedback from `thermal.hpp` updates Rds_on / Vce_sat / Vf in real time.

### YAML Schema
- New `type: mosfet_catalog | igbt_catalog | diode_catalog` with `model: <vendor>/<part>` reference, or inline `parameters:` block.
- Parameter override: user can override any subset of parameters inline for sensitivity studies.

## Impact

- **Affected specs**: `device-models` (new catalog tier), `netlist-yaml` (new component types and model references).
- **Affected code**: new files `core/include/pulsim/v1/components/mosfet_catalog.hpp`, `igbt_catalog.hpp`, `diode_catalog.hpp`; importer scripts in `python/pulsim/import_/`; library files under `devices/catalog/`.
- **Performance**: catalog devices use AD-derived Jacobians (Phase 0); table interpolation cached per topology; expected ≤2× per-stamp cost vs Level-1 with much higher fidelity.
- **Backward compat**: existing `mosfet`, `igbt`, `diode` types unchanged; `_catalog` variants are additive.

## Success Criteria

1. **Fidelity**: switching-loss prediction within 10% of vendor LTspice model on hard-switching scenarios for all 6 reference devices.
2. **Conduction loss**: within 5% across 25–125 °C range vs vendor model.
3. **Body diode reverse recovery**: shape match within 15% on di/dt = 100–1000 A/µs.
4. **Importer**: `import_datasheet(<vendor_pdf>)` produces a runnable `DeviceParams` for ≥3 of the 6 reference devices.
5. **Documentation**: catalog browsable from docs site with side-by-side waveform plots vs vendor reference.
