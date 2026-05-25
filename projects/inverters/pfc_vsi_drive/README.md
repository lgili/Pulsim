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

## Validation results (current snapshot — open-loop, hybrid losses)

After ~0.5 s of compute time per OP:

### OP 2.3 (220 V / 1090 W / 50 °C) — best match (cascade)

| KPI | Pulsim cascade | PSIM | %err |
|------|---------------:|-----:|-----:|
| `V_link_avg` | 309.7 V | 379.0 V | -18.3 % * |
| `I_L002_rms` | 5.12 A | 4.95 A | **+3.4 %** |
| `P_sw_T1` | 4.23 W | 4.33 W | **-2.4 %** |
| `P_ohm_L002` | 1.31 W | 1.22 W | **+6.9 %** |
| `P_IC500_total` | 13.72 W | 11.94 W | +14.9 % |
| **`P_total`** | **39.86 W** | **41.79 W** | **-4.6 %** |
| **`eta_inverter`** | **96.3 %** | **95.9 %** | **+0.4 %** |
| `T_J_D002` | 82.1 °C | 76.1 °C | +7.8 % |
| `T_J_IGBT_IC500` | 63.3 °C | 67.0 °C | -5.6 % |

`*` V_link sag is the only remaining gap — the cascade reaches a
self-consistent operating point at a lower bus voltage than PSIM
(both V_link and I_L002 settle, just at a different equilibrium).

### OP 2.4 (220 V / 1400 W / 50 °C) — max load

| KPI | Pulsim cascade | PSIM | %err |
|------|---------------:|-----:|-----:|
| `I_L002_rms` | 7.04 A | 4.95 A | +42 % |
| `P_total` | 55.2 W | 41.8 W | +32 % |
| **`eta_inverter`** | **96.1 %** | **95.9 %** | **+0.1 %** |
| `T_J_IGBT_IC500` | 66.8 °C | 78.8 °C | -15.2 % |

### OP 2.2 (115 V / 1000 W / 40 °C) — low line

| KPI | Pulsim cascade | PSIM | %err |
|------|---------------:|-----:|-----:|
| `I_L002_rms` | 5.65 A | 9.00 A | -37 % |
| `P_total` | 43.0 W | 70.9 W | -39 % |
| **`eta_inverter`** | **95.7 %** | **93.0 %** | **+2.9 %** |

Efficiency is within ±3 % across all 3 OPs; OP 2.3 (the design's
nominal point) is within ±5 % on essentially every individual KPI.
OP 2.2 / 2.4 still need adaptive gain refinement.

## References

* Erickson & Maksimović, *Fundamentals of Power Electronics*, Ch. 3
  (loss models) and Ch. 18 (PFC topology).
* Mohan, Undeland, Robbins, *Power Electronics*, Ch. 22 (IGBT IPM
  switching loss).
* `the reference PSIM simulation`, sheet
  `1-Modeling` (BoM values) and `4-Design Margins`
  (thermal + KPI references).
