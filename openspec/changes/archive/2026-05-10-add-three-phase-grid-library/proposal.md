## Why

Three-phase systems are pervasive in grid-tied power electronics: solar inverters, wind converters, EV chargers, motor drives, STATCOMs, HVDC. PSIM's grid library and PLECS's three-phase blocks exist precisely because every grid-tied design needs:

- Balanced and unbalanced three-phase sources (programmable voltage/frequency profile, sag/swell events)
- abc/αβ/dq frame conversions
- PLL (phase-locked loop) variants: SRF-PLL, DSOGI-PLL, MAF-PLL
- Symmetrical-component decomposition (positive/negative/zero sequence)
- Grid-following (current-controlled) and grid-forming (voltage-controlled) control templates
- Park/Clarke and inverse transforms
- Anti-islanding detection blocks (e.g. AFD, SFS for IEEE 1547)

Without these, even basic three-phase inverter examples are tedious to assemble. Pulsim's `signal_evaluator.py` already supports DAG-based control composition; this change adds the three-phase domain on top.

## What Changes

### Three-Phase Source Library
- `ThreePhaseSource { v_rms, frequency, phase_seq: abc|acb }`
- Programmable variants: `ThreePhaseSourceProgrammable` with time-varying `v_rms(t)`, `frequency(t)`, individual phase voltage profiles for unbalance studies, sag/swell events.
- `ThreePhaseHarmonicSource { fundamental, [{order, magnitude, phase}, ...] }` for power-quality testing.

### Frame Transformation Blocks (Domain-Aware)
- Park/Clarke transforms in **electrical domain** (3-phase electrical terminals → dq electrical terminals) for analytical model integration.
- Park/Clarke transforms in **signal domain** (3-phase signals → dq signals) for control loops.
- Both already partially covered by `add-motor-models`; this change formalizes and extends.

### PLL Variants
- `SrfPll { kp, ki, freq_init }` — synchronous reference frame PLL.
- `DsogiPll { ... }` — dual second-order generalized integrator (handles unbalanced grids).
- `MafPll { window_period, ... }` — moving-average filter PLL.
- Each exposes `theta_estimate`, `omega_estimate`, `vd_locked`.

### Sequence Decomposition
- `SymmetricalComponents { fundamental_freq }` decomposes 3-phase into positive/negative/zero sequences via Fortescue transform with delay buffer.
- Useful for grid-quality monitoring and unbalanced-fault analysis.

### Grid-Following Inverter Template
- `grid_following_inverter_template`: three-phase inverter + LCL filter + dq-decoupled current loop + SrfPll + power command.
- Default tuning: current loop bandwidth ~1 kHz, PLL bandwidth ~50 Hz.
- Output: full sub-netlist ready for transient + AC sweep.

### Grid-Forming Inverter Template
- `grid_forming_inverter_template`: virtual synchronous machine (VSM) or droop-control variant.
- Internal voltage / frequency command.
- Useful for microgrid and standalone studies.

### Anti-Islanding Detection (Stretch)
- `AfdBlock` (Active Frequency Drift), `SfsBlock` (Sandia Frequency Shift) per IEEE 1547.
- Documented as informative; full anti-islanding compliance certification is out of scope.

### Validation
- Symmetrical / unbalanced grid test fixture.
- PLL lock-time and steady-state error vs reference.
- Grid-following inverter active/reactive power command tracking.
- Grid-forming inverter under load step.
- AC sweep impedance: input-side admittance for stability assessment (e.g., resonance with grid impedance).

### YAML Schema
- New types: `three_phase_source`, `three_phase_source_programmable`, `three_phase_harmonic_source`, `srf_pll`, `dsogi_pll`, `maf_pll`, `symmetrical_components`, `afd_block`, `sfs_block`, plus templates `grid_following_inverter_template`, `grid_forming_inverter_template`.

## Impact

- **New capability**: `three-phase-grid`.
- **Affected specs**: `three-phase-grid` (new), `netlist-yaml`.
- **Affected code**: new `core/include/pulsim/v1/grid/`, additions to signal-evaluator, parser additions.
- **Performance**: three-phase blocks are linear and signal-domain; no AD overhead. PLLs are signal-domain ODE; lightweight.

## Success Criteria

1. **PLL lock**: SrfPll locks within 50 ms on nominal grid, steady-state phase error <0.5°.
2. **DsogiPll on unbalanced**: locks correctly on 50% Phase-A sag.
3. **Grid-following inverter**: active/reactive command tracking within 5% steady-state.
4. **Grid-forming inverter**: voltage regulation within 2% under 50% load step.
5. **Tutorial**: solar inverter end-to-end (PV source + DC link + 3φ inverter + LCL + grid-follow) producing IEEE 1547-compliant waveforms.
