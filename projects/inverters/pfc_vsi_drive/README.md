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
| `measure_loop_gain.py` | Measures both cascade loop gains by injection on the switching model, per OP. |
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
`V_link_avg = 378.98 V` for OP 2.3; the front-end sim lands at
380.1 V with the default cascade loop (+0.3 %).

## Loss extraction: simulated waveforms + analytical switching

`losses.py` integrates the simulated branch currents over the last whole
line cycles: the line current i_L001 (bridge D001, L001 DCR), i_L002,
i_T001/T002, i_D002, i_Cbus, the boost shunts and R508. Switching losses
are analytical (energy per event × f_sw): Pulsim's switches are ideal.
The bridge loss is `2·(V_F·mean|i_in| + R_on·mean(i_in²))`, two diodes in
series with the line at every instant.

Until 2026-11 the line-side numbers (`I_in_rms`, `PF`, `P_cond_D001`,
`P_ohm_L001`) were not simulated: they were back-computed from the bus
power with an assumed PF = 0.95, and the bridge term counted two of the
four diodes. They are simulated now.

## PFC control modes

The front-end supports three control modes, selectable via flags on
`DriveSimParams`:

| Mode | Flags | Description |
|------|-------|------------|
| **Open-loop, constant duty** | both flags `False` | Boost MOSFET fires at a fixed duty `D = 1 − V_in_pk/V_link_target` derived from the CCM gain formula. Kept for ablation. |
| **Feed-forward trajectory** | `sp.pfc_closed_loop = True` | Modulates `D(t) = (1 − V_rect(t)/V_link_target) · K_load` (the textbook CCM steady-state duty trajectory). Shapes the current but cannot regulate V_link without an outer loop. |
| **Full cascade — V outer + I inner PI** *(default)* | `sp.pfc_cascade_loop = True` | Average-current-mode PFC (Erickson §18.4): outer V_link PI generates `K_amp`, inner I_L002 PI generates the duty around the CCM feed-forward `1 − V_rect/V_link`. `PfcCascadeController`, wired via `switch_fn` + `step_observer`. |

### Cascade design and measured margins

The gains are computed per operating point from the averaged plants
(`PfcCascadeController.__init__`):

* inner loop, plant `V_link/(L·s)`: `Kp_i = ω_i·L/V_link` for
  `pfc_i_bw_hz = 1 kHz`, PI zero at `ω_i/5`;
* outer loop, plant `(V_pk²/2V)/(C·s + 2/R_eq)` seen through the V_link
  low-pass (τ = 6.7 ms): `Kp_v` sets the loop magnitude to 1 at
  `pfc_v_bw_hz = 20 Hz`, PI zero a decade below.

`measure_loop_gain.py` measures both loops on the switching model by
injection (L = −X_ctrl/X_applied, one DFT bin over whole line cycles,
so the result is the line-cycle average):

| OP | inner crossover | inner PM | outer crossover | outer PM |
|----|----------------:|---------:|----------------:|---------:|
| 2.2 | 682 Hz | 57.9° | 19.7 Hz | 58.5° |
| 2.3 | 692 Hz | 48.3° | 20.3 Hz | 55.0° |
| 2.4 | 637 Hz | 49.7° | 19.6 Hz | 62.5° |

No −180° crossing in either sweep (inner 0.3–2.4 kHz, outer 5–48 Hz),
so the gain margin is beyond the sweep. The inner crossover sits below
its 1 kHz design: the CCM plant holds only away from the line zero
crossings, where the duty saturates and the inductor current goes
discontinuous. The previous tuning put the inner PI zero at the
crossover and gave 21–23° at 220 V; the outer loop crossed at 15–16 Hz
because its design ignored the low-pass and the load pole.

## Known limitations

1. **The compressor is a 3φ RL load** (no back-EMF source). Speed is
   set by the SPWM frequency × slip assumption rather than torque
   balance.
2. **The front end feeds a resistor** (`R_eq = V_link²/P_in_target`),
   so the bus capacitors carry only the 2·f_line ripple, not the
   inverter's 5 kHz current: `P_esr_C009/C010` read ~92 % below PSIM.
3. **The two stages run separately**; they share only `V_link`.

## Validation results (cascade, all OPs)

`python3 run_validation.py`, 10 line cycles, KPIs over the last whole
line cycles of the run. About 1 s of compute per OP.

### What changed on 2026-11-13

The model used to put a 470 µF capacitor (C011) on `rect_p`, between
the bridge and the boost inductor. PSIM's own KPIs rule that out: by
KCL at `rect_p`, `i_D001 = i_L002 + i_C`, and PSIM reports
`I_D001_rms = 4.984 A` against `I_L002_rms = 4.948 A` with a line PF
of 0.979 (OP 2.3). With 470 µF the bridge peak-charges the capacitor:
the simulated line PF was 0.62, `I_in_rms` 8.33 A (PSIM 5.06 A) and
`I_L002` nearly DC, 3.67 A rms (PSIM 4.95 A). That was the 25–30 %
`I_L002_rms` gap of the previous snapshot, and the PF = 0.95 line-side
estimate hid the line-current error. Without C011 the line current
and `I_L002` match PSIM within 1.5 % at OP 2.2 and OP 2.3.

### OP 2.2 (115 V / 60 Hz / 1000 W / 40 °C) — low line

| KPI | Pulsim | PSIM | %err |
|------|-------:|-----:|-----:|
| `V_link_avg` | 380.0 V | — | |
| `I_in_rms` | 8.97 A | 9.04 A | −0.8 % |
| `PF` | 0.997 | — | |
| `I_L002_rms` | 8.94 A | 9.00 A | −0.7 % |
| `I_F500_rms` | 3.39 A | 3.38 A | +0.3 % |
| `P_ohm_L001` | 8.84 W | 8.99 W | −1.6 % |
| `P_ohm_L002` | 4.00 W | 4.05 W | −1.4 % |
| `P_total` | 74.2 W | 70.9 W | +4.7 % |
| `eta_inverter` | 92.6 % | 93.0 % | −0.5 % |

### OP 2.3 (220 V / 50 Hz / 1090 W / 50 °C) — high line, nominal

| KPI | Pulsim | PSIM | %err |
|------|-------:|-----:|-----:|
| `V_link_avg` | 380.1 V | 379.0 V | +0.3 % |
| `I_in_rms` | 5.08 A | 5.06 A | +0.4 % |
| `PF` | 0.992 | 0.979 | +1.3 % |
| `I_L002_rms` | 5.02 A | 4.95 A | +1.4 % |
| `I_F500_rms` | 3.40 A | 3.51 A | −3.4 % |
| `P_ohm_L001` | 2.84 W | 2.81 W | +0.8 % |
| `P_ohm_L002` | 1.26 W | 1.22 W | +2.8 % |
| `P_cond_D001` | 9.95 W | 15.58 W | −36.1 % |
| `P_cond_T1` | 0.32 W | 4.27 W | −92.6 % |
| `P_total` | 47.5 W | 41.8 W | +13.7 % |
| `eta_inverter` | 95.6 % | 95.9 % | −0.3 % |
| `T_J_D001` | 104.4 °C | 102.1 °C | +2.2 % |
| `T_J_D002` | 81.8 °C | 76.1 °C | +7.4 % |
| `T_J_IGBT_IC500` | 66.3 °C | 67.0 °C | −1.0 % |

### OP 2.4 (220 V / 50 Hz / 1023 W / 50 °C) — high line, current-limited

| KPI | Pulsim | PSIM | %err |
|------|-------:|-----:|-----:|
| `V_link_avg` | 380.2 V | 377.9 V | +0.6 % |
| `I_in_rms` | 4.77 A | 6.78 A | −29.7 % |
| `PF` | 0.991 | 0.923 | +7.4 % |
| `I_L002_rms` | 4.71 A | 4.95 A | −4.9 % |
| `P_total` | 44.4 W | 41.8 W | +6.3 % |
| `eta_inverter` | 95.7 % | 95.9 % | −0.3 % |
| `T_J_D002` | 79.5 °C | 80.4 °C | −1.1 % |
| `T_J_IGBT_IC500` | 65.3 °C | 78.8 °C | −17.2 % |

### Reading the comparison

* The front end now reproduces PSIM's line and boost currents at
  OP 2.2 and OP 2.3: `I_in_rms`, `I_L002_rms`, `V_link` and the L001 /
  L002 copper losses within 3 %.
* `P_total` is not a usable target. PSIM's OP 2.3 loss terms sum to
  60.2 W against its stated `P_total` of 41.79 W, and OP 2.4 repeats
  OP 2.3's `P_total`, `I_L002_rms` and `P_ohm_L002` to the last digit.
  Against the sum of PSIM's own terms, OP 2.3 is 21 % low.
* The device-level misses are in the loss models, not the currents:
  the boost MOSFET conduction loss is 92 % low (PSIM's 4.27 W at
  `I_T001_rms = 1.42 A` implies 2.1 Ω), the bridge 36 % low and the SiC
  diode 53 % high.
* OP 2.4 is internally inconsistent in the reference: PSIM's
  `I_in_rms = 6.78 A` at PF 0.923 is 1373 W of line power, while its
  `P_in` field is 1023 W and its `I_L002_rms` equals OP 2.3's. The
  model targets 1023 W, so `I_in_rms` reads 30 % low. PSIM's `P_in`
  field is not the line power at OP 2.3 either (V·I·PF = 1086 W against
  1014.6 W), so the `P_in` row is not compared like for like.
* OP 2.2 boost-MOSFET and SiC-diode junction temperatures are 36–41 %
  below PSIM.

## References

* Erickson & Maksimović, *Fundamentals of Power Electronics*, Ch. 3
  (loss models) and Ch. 18 (PFC topology).
* Mohan, Undeland, Robbins, *Power Electronics*, Ch. 22 (IGBT IPM
  switching loss).
* `the reference PSIM simulation`, sheet
  `1-Modeling` (BoM values) and `4-Design Margins`
  (thermal + KPI references).
