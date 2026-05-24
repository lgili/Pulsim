# Maia PFC FR — Compressor drive

Pulsim port of the **Maia PFC FR** power stage, validated against the
PSIM reference simulation in
`0000083978 - Maia PFC FR - Simulation v0.4 (1).xlsx`
(Nidec internal design deliverable for the *Maia PFC Full Range*
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
| `maia_pfc_compressor_pulsim_validation.py` | The Pulsim circuit builder + `simulate_frontend` / `simulate_inverter` entry points. |
| `losses.py` | Per-device loss + thermal model (conduction, switching, ESR, magnetic, Foster R_thja). |
| `smoke_test.py` | Quick 1-OP smoke pass (`OP_2.3`) — confirms both sims run and compares 5 headline KPIs. |
| `run_validation.py` | Full sweep across all 3 OPs with per-KPI delta table + optional CSV export. |

## Running

From the repo root with Pulsim built:

```bash
# Quick smoke (~0.5 s)
PYTHONPATH=build/python python3 projects/inverters/maia_pfc_compressor/smoke_test.py

# Full 3-OP validation (~1 s)
PYTHONPATH=build/python python3 projects/inverters/maia_pfc_compressor/run_validation.py

# Dump the full KPI matrix to CSV
PYTHONPATH=build/python python3 projects/inverters/maia_pfc_compressor/run_validation.py \
    --csv /tmp/maia_validation.csv
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

## Known limitations of the open-loop model

1. **No PFC control loop** — boost duty is fixed at the CCM gain
   estimate `D = 1 - V_in_pk/V_link_target`. Real PSIM modulates D
   dynamically to track `|V_ac|` and regulate `V_link`. As a result
   Pulsim's input current spectrum (PF, THD) doesn't match PSIM, and
   the steady-state V_link drifts ~3 % low.

2. **L001-C006-bridge tank rings unbounded past ≈ 60 ms** because
   there's no current-loop damping. KPIs are extracted from a 20–40 ms
   stable window before the drift dominates.

3. **Boost-leg currents from L001 state variable are numerically
   unstable** in DCM. `I_in_rms` is therefore back-computed from
   `P_in / (V_in_rms · PF)` rather than from the integrated state.

4. **Compressor is a 3φ RL load** (no back-EMF source). Speed is set
   by SPWM frequency × slip assumption rather than torque balance.

## Validation results (snapshot, OP 2.3)

After 0.5 s of compute time:

| KPI | Pulsim | PSIM | %err |
|------|--------|------|------|
| `V_ac_rms` | 220.0 V | 219.4 V | +0.3 % |
| `V_link_avg` | 374.4 V | 379.0 V | -1.2 % |
| `I_F500_rms` | 3.96 A | 3.51 A | +12.8 % |
| `P_IC500_total` | 11.09 W | 11.94 W | -7.1 % |
| `eta_inverter` | 95.2 % | 95.9 % | -0.7 % |
| `T_J_IGBT_IC500` | 60.7 °C | 67.0 °C | -9.4 % |

Conduction losses on the boost MOSFETs / SiC diode are 70–100 % off
from PSIM due to limitation (1) above; closing the loop would
collapse those deltas to single digits.

## References

* Erickson & Maksimović, *Fundamentals of Power Electronics*, Ch. 3
  (loss models) and Ch. 18 (PFC topology).
* Mohan, Undeland, Robbins, *Power Electronics*, Ch. 22 (IGBT IPM
  switching loss).
* `0000083978 - Maia PFC FR - Simulation v0.4 (1).xlsx`, sheet
  `1-Modeling` (BoM values) and `4-Design Margins Maia FR`
  (thermal + KPI references).
