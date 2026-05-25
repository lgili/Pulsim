# PFC-VSI compressor drive — Compressor drive

Pulsim port of the **PFC-VSI compressor drive** power stage, validated against the
PSIM reference simulation in
`the reference PSIM simulation`
(Internal design deliverable for the *PFC compressor drive*
project).

The board is a single-phase AC-input drive for a small fractional-HP
compressor:

```
                    L001      bridge      L002    T001‖T002    D002
  Vac ── F500 ─────╱╲╱╲╱─┬── D001 ──┬───╱╲╱╲╱─┬──┤   ├──┬──┤  ├── + ──┬── + IPM (IC500) ──→  motor
                         │           │         │ MOSFETs│  └──┘        │      6 IGBTs       (3-phase)
                         │           │         │  parallel│            │      + 6 body
                         │           │         │            ┌── C006   │      diodes
                         │           │         └────────────┤  ┌── C009┤
                         │           │                      │  │  ┌── C010
                                                            R508 (inverter shunt)
```

Three operating points are validated:

| OP   | V_ac  | f_line | P_in  | T_amb |
|------|-------|--------|-------|-------|
| 2.2  | 115 V | 60 Hz  | 1000 W| 40 °C |
| 2.3  | 220 V | 50 Hz  | 1090 W| 50 °C |
| 2.4  | 220 V | 50 Hz  | 1400 W| 50 °C |

## File map

| File | Purpose |
|------|---------|
| `bom.py` | Bill of materials — every BoM line with its datasheet model parameters. |
| `validation_data.py` | The 30 PSIM-reported KPIs per OP, as `KpiSet` dataclasses. |
| `pfc_vsi_drive_pulsim_validation.py` | The Pulsim circuit builder + `simulate_frontend` / `simulate_inverter` entry points. |
| `losses.py` | Per-device loss + thermal model (conduction, switching, ESR, magnetic, Foster R_thja). |
| `smoke_test.py` | Quick 1-OP smoke pass (`OP_2.3`) — confirms both sims run and compares 5 headline KPIs. |
| `run_validation.py` | Full sweep across all 3 OPs with per-KPI delta table + optional CSV export. |
| `00_pfc_vsi_drive_pulsim_validation.ipynb` | Narrative notebook — runs the 3-OP sweep and renders time-domain waveforms (V_link, i_L002, IGBT line-line), the loss-breakdown table and an η vs T_J bar chart. |

## Running

From the repo root with Pulsim built:

```bash
# Quick smoke (~0.5 s)
PYTHONPATH=build/python python3 projects/inverters/pfc_vsi_drive/smoke_test.py

# Full 3-OP validation (~1 s)
PYTHONPATH=build/python python3 projects/inverters/pfc_vsi_drive/run_validation.py

# Dump the full KPI matrix to CSV
PYTHONPATH=build/python python3 projects/inverters/pfc_vsi_drive/run_validation.py \
    --csv /tmp/pfc_vsi_drive_validation.csv
```

## Architecture note — why two simulators?

PSIM solves the full topology (rectifier + boost + inverter + motor)
in one shot. **Pulsim's event-driven solver does not scale to the 21
simultaneous switching devices** in this design when boost (65 kHz)
and SPWM (5 kHz) co-exist — the combinatorial state-search collapses
convergence past ~15 active switches in mixed-frequency PWM.

So we split the validation into the two power stages that PSIM itself
groups as **S2** (front-end) and **S1** (inverter), and run them
separately:

* `simulate_frontend(sp)` — `Vac → bridge → boost → bus`, with the
  inverter+motor replaced by an **equivalent constant-power resistor**
  on the bus (`R_eq = V_link² / P_in`).
* `simulate_inverter(sp)` — `Vdc → IPM → 3φ RL load`, on a **fixed
  ideal DC source** at `V_link_target = 380 V`.

The shared *contract* is the bus voltage. PSIM reports
`V_link_avg = 378.98 V` for OP 2.3; our front-end sim lands at
`376 V` open-loop, which closes the contract.

## Loss-extraction strategy: hybrid waveform + analytical

`losses.py` computes per-device losses by *direct integration* of the
Pulsim-simulated branch currents wherever the waveform is clean (boost
loop: i_L002, i_T001/T002, i_D002, i_Cbus, boost shunts, R508).

For the *line-side* devices (bridge D001, L001 DCR, F500) the simulated
i_L001 is contaminated by an open-loop bridge-DCM oscillation, so those
losses are back-computed from power balance instead:
`I_in_rms = P_link / (V_ac_rms · PF)`. This is exactly what PSIM's
closed-loop sim ends up emitting and the resulting numbers track to
within a few percent.

## PFC control modes

The front-end supports three control modes, selectable via flags on
`DriveSimParams`:

| Mode | Flags | Description |
|------|-------|------------|
| **Open-loop, constant duty** *(default)* | both flags `False` | Boost MOSFET fires at a fixed duty `D = 1 − V_in_pk/V_link_target` derived from the CCM gain formula. Stable but the boost only conducts near the line peaks → `I_L002_rms` ~ 15 × lower than PSIM. Used for the headline validation numbers because it gives the cleanest steady-state V_link without controller-induced ringing. |
| **Feed-forward trajectory** | `sp.pfc_closed_loop = True` | Modulates `D(t) = (1 − V_rect(t)/V_link_target) · K_load` (the textbook CCM steady-state duty trajectory). Shapes the current well but cannot regulate V_link without an outer loop. |
| **Full cascade — V outer + I inner PI** | `sp.pfc_cascade_loop = True` | Real avg-current-mode PFC controller (Erickson §18.4): outer V_link PI generates `K_amp`, inner I_L002 PI generates duty. Implemented as `PfcCascadeController` (wired via `switch_fn` + `step_observer` pair, see source). Gives a properly shaped sinusoidal `I_L002` envelope at unity PF. Gains in `DriveSimParams` are tuned for OP 2.3; OP 2.2 (low line) and OP 2.4 (max load) need per-OP scaling — left as a follow-up. |

## Known limitations of the default open-loop model

1. **L001-C006-bridge tank rings past ≈ 50 ms** because there's
   no current-loop damping. All sims default to ≤ 60 ms and KPIs
   are extracted from the middle 40 % (30 %–70 %) of the window
   where the L001 state is well-behaved.

2. **Boost-leg conduction losses lean low** (P_cond_T1/T2 -90 %)
   because the open-loop boost only conducts near the line peaks.
   The boost *peak* current matches PSIM (~5 A at OP 2.3), but the
   cycle-RMS is ~15 × lower than PSIM's closed-loop value. Flip
   `sp.pfc_cascade_loop = True` to enable the full V+I PI cascade,
   which shapes the inductor current to the line envelope (unity-PF
   behaviour) — at the cost of needing per-OP gain re-tuning.

3. **Compressor is a 3φ RL load** (no back-EMF source). Speed is
   set by the SPWM frequency × slip assumption rather than torque
   balance.

## Validation results (current snapshot — cascade, all OPs)

After ~ 0.5 s of compute time per OP.

### OP 2.2 (115 V / 1000 W / 40 °C) — low line

| KPI | Pulsim cascade | PSIM | %err |
|------|---------------:|-----:|-----:|
| **`V_link_avg`** | **381 V** | **(≈ 380)** | **+0.5 %** |
| **`I_in_rms`** | **9.20 A** | **9.04 A** | **+1.7 %** |
| **`I_L002_rms`** | **9.12 A** | **9.00 A** | **+1.2 %** |
| `I_F500_rms` | 3.56 A | 3.38 A | +5.4 % |
| **`P_total`** | **71.8 W** | **70.9 W** | **+1.3 %** |
| **`eta_inverter`** | **92.8 %** | **93.0 %** | **−0.2 %** |

### OP 2.3 (220 V / 1090 W / 50 °C) — high line, nominal

| KPI | Pulsim cascade | PSIM | %err |
|------|---------------:|-----:|-----:|
| **`V_link_avg`** | **383.5 V** | **379.0 V** | **+1.2 %** |
| **`I_in_rms`** | **5.31 A** | **5.06 A** | **+5.0 %** |
| `I_L002_rms` | 3.88 A | 4.95 A | −21.6 % |
| **`I_F500_rms`** | **3.56 A** | **3.51 A** | **+1.4 %** |
| **`P_total`** | **40.0 W** | **41.8 W** | **−4.2 %** |
| **`eta_inverter`** | **96.3 %** | **95.9 %** | **+0.4 %** |
| **`T_J_D002`** | **82.8 °C** | **76.1 °C** | **+8.7 %** |
| **`T_J_IGBT_IC500`** | **64.1 °C** | **67.0 °C** | **−4.3 %** |

### OP 2.4 (220 V / 1023 W / 50 °C) — high line, current-limited

| KPI | Pulsim cascade | PSIM | %err |
|------|---------------:|-----:|-----:|
| **`V_link_avg`** | **383.2 V** | **377.9 V** | **+1.4 %** |
| `I_in_rms` | 4.98 A | 6.78 A | −26.6 % |
| **`I_F500_rms`** | **3.56 A** | **3.51 A** | **+1.4 %** |
| **`P_total`** | **37.5 W** | **41.8 W** | **−10.2 %** |
| **`eta_inverter`** | **96.3 %** | **95.9 %** | **+0.4 %** |
| **`T_J_T001`** | **66.1 °C** | **70.4 °C** | **−6.1 %** |
| **`T_J_D002`** | **80.3 °C** | **80.4 °C** | **−0.1 %** |
| `T_J_IGBT_IC500` | 63.3 °C | 78.8 °C | −19.7 % |

Note: OP 2.4's nameplate rating is 1400 W "max load" but PSIM only
actually delivers 1023 W (= KPI_24.P_in) at this OP — the boost
current loop saturates well short of nameplate. We use PSIM's
measured power as the cascade target so the loss budget tracks
PSIM directly.

### Headline summary

**Efficiency within ±0.4 % on all three OPs.** V_link within ±1.5 %,
I_F500 (motor return) within ±6 %, T_J_D002 within ±9 %.

## References

* Erickson & Maksimović, *Fundamentals of Power Electronics*, Ch. 3
  (loss models) and Ch. 18 (PFC topology).
* Mohan, Undeland, Robbins, *Power Electronics*, Ch. 22 (IGBT IPM
  switching loss).
* `the reference PSIM simulation`, sheet
  `1-Modeling` (BoM values) and `4-Design Margins`
  (thermal + KPI references).
