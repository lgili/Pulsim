# Loss & Thermal Modelling

Pulsim reconstructs device **losses** from the simulated waveforms and
turns them into **junction temperatures** through Foster/Cauer thermal
networks — including several devices sharing one heatsink and the
loss↔temperature feedback that decides whether a converter can run at a
given power. This page explains how the model works and how to use it.

The whole stack lives in `pulsim.losses` and `pulsim.thermal` and is
re-exported at the flat `pulsim` namespace.

---

## At a glance

| Layer | Function | What it gives you |
|---|---|---|
| Losses | `device_loss_summary` | per-device conduction + switching + core loss from one sim |
| T_j (per device) | `device_thermal_summary` | junction-temperature trace per device (Foster) |
| T_j (post-proc) | `compute_temperature` | ΔT_j(t) from a power trace + Z_th |
| Shared heatsink | `shared_heatsink_steady_state` | coupled T_j of N devices on one sink |
| Electro-thermal | `electrothermal_steady_state` | self-consistent T_j with R_ds(T) feedback + **runaway** |
| Live co-sim | `add_foster_network` / `add_shared_heatsink` + observers | T_j(t) solved alongside the circuit |
| Heatsink sizing | `tim_resistance`, `convection_resistance` | R_th from geometry / airflow |

---

## 1. Losses — how it works

### Conduction

By default conduction loss is reconstructed from the actual current
trace as a pure resistance, `P = v·i = v²·g`. Real IGBTs and diodes
follow an **offset + slope** characteristic `V = V_f0 + r·I`, whose
forward-voltage offset dominates at low current. Opt a device into it
with `conduction_specs`:

```python
P = p.device_loss_summary(
    b, res, switch_fn=sw,
    conduction_specs={
        "Q1": {"V_ce0": 0.9, "r_ce": 8e-3},   # IGBT  (aliases V_f0/r_on too)
        "D1": {"V_f0": 0.7, "r_d": 5e-3},     # diode
    },
)
# each entry gains P_cond_model_avg = V_f0·|i| + r_on·i²  (over the real i(t))
```

### Switching

Switching loss is a PSIM/PLECS-style datasheet annotation — the ideal
kernel switches emit no physical switching transient, so the energy is
applied per detected edge:

```python
switch_specs = {"Q1": {"E_on_ref": 0.9e-3, "E_off_ref": 0.6e-3,
                       "V_ref": 400.0, "I_ref": 20.0}}        # linear
diode_specs  = {"D1": {"E_rr_ref": 0.3e-3, "V_R_ref": 400.0}} # or {"Q_rr": …}
```

For better fidelity away from the reference point, give the **nonlinear
datasheet curve** (interpolated at the actual switched current per
event, linearly extrapolated beyond the table):

```python
switch_specs = {"Q1": {
    "E_on_curve":  [(5, 0.4e-3), (10, 1.0e-3), (20, 2.6e-3)],   # E_on vs I @ V_ref
    "E_off_curve": [(5, 0.3e-3), (10, 0.8e-3), (20, 2.1e-3)],
    "V_ref": 400.0}}
diode_specs = {"D1": {
    "E_rr_curve": [(10, 0.5e-6), (50, 4e-6), (100, 12e-6)],     # E_rr vs I_F @ V_R_ref
    "V_R_ref": 400.0}}
```

### Core loss

Inductor core loss uses Steinmetz / iGSE via `core_loss_specs` (material
catalog or an inline `K/alpha/beta` triplet) — see
`device_loss_summary`'s docstring.

---

## 2. Junction temperature — Foster / Cauer

A device's transient thermal impedance `Z_th(t) = Σ R_th_i·(1 − e^(−t/τ_i))`
is a Foster RC ladder. Two ways to use it:

**Post-processing** — convolve a power trace with `Z_th`:

```python
stages = [p.FosterStage(R_th_K_per_W=0.3, tau_s=5e-3),
          p.FosterStage(R_th_K_per_W=0.5, tau_s=50e-3)]
T_j = p.compute_temperature(times, p_loss_trace, stages, T_amb_C=40.0)
```

**End-to-end per device** — wire the loss summary into Foster networks:

```python
summary = p.device_thermal_summary(
    b, res,
    thermal_specs={"Q1": {"stages": stages}},
    conduction_specs={"Q1": {"V_ce0": 0.9, "r_ce": 8e-3}},
    switch_fn=sw, switch_specs=switch_specs,
    T_ambient_C=40.0)
print(summary[0]["T_j_peak"])
```

`p.CauerStage(R_th, C_th)` + `p.add_cauer_thermal_network(...)` is the
physical (per-layer) parametrisation; `p.fit_foster_from_zth(...)` fits a
Foster ladder to a datasheet Z_th(t) curve.

---

## 3. Shared heatsink — coupling N devices

Several power devices on **one** heatsink are thermally coupled: their
dissipations sum at the shared sink, so the sink temperature is driven by
the *total* power and that rise lifts every junction together. A
per-device-independent model misses this — exactly where it matters when
pushing power up.

```python
# A device = its junction→case ladder + a case→sink (TIM) resistance.
igbt = lambda n: p.HeatsinkDevice(
    n, [p.FosterStage(R_th_K_per_W=0.3, tau_s=0.05)],
    R_th_case_to_sink_K_per_W=0.2)
dio  = lambda n: p.HeatsinkDevice(
    n, [p.FosterStage(R_th_K_per_W=0.5, tau_s=0.03)],
    R_th_case_to_sink_K_per_W=0.2)
devs = [igbt(f"Q{i}") for i in range(3)] + [dio(f"D{i}") for i in range(3)]

res = p.shared_heatsink_steady_state(
    devs,
    {**{f"Q{i}": 18.0 for i in range(3)}, **{f"D{i}": 6.0 for i in range(3)}},
    R_th_sink_to_amb_K_per_W=0.5, T_amb_C=40.0)
print(res["T_sink_C"], res["devices"]["Q0"]["T_j_C"])
```

The maths is

$$
T_\text{sink} = T_\text{amb} + R_\text{th,sa}\sum_i P_i,\qquad
T_{j,i} = T_\text{sink} + (R_\text{th,cs}+R_\text{th,jc})\,P_i
$$

— the $\sum_i P_i$ is the coupling. For the transient version, build the
network into a `CircuitBuilder` with `p.add_shared_heatsink(...)` and
drive it with `p.make_heatsink_observer(...)` + `p.simulate`.

---

## 4. Electro-thermal feedback + runaway

Conduction loss climbs with junction temperature (R_ds(T) for MOSFETs,
V_ce(T) for IGBTs), so the steady state is a **self-consistent fixed
point**. A model with R_ds fixed at 25 °C reports an optimistic T_j and
*cannot detect thermal runaway*.

```python
q = p.TempCoLoss(P_cond_ref_W=15.0, P_sw_ref_W=6.0,
                 a_cond_per_C=0.006, a_sw_per_C=0.003)   # tempcos from datasheet
r = p.electrothermal_steady_state(
    devs, {name: q for name in (...)},
    R_th_sink_to_amb_K_per_W=0.5, T_amb_C=60.0)

if r["runaway"]:
    print("thermal runaway:", r["message"])
else:
    print("T_j:", max(d["T_j_C"] for d in r["devices"].values()),
          "feedback gain ρ:", r["feedback_gain"])   # margin = 1 − ρ
```

Because `TempCoLoss` is linear in T_j the fixed point is solved in closed
form, `(I − M·K)·T_j = T_amb + M·(P₀ − K·T_ref)`, with feedback gain
`G = M·K`. `ρ(G) < 1` is stable; **`ρ(G) ≥ 1` is runaway** and is flagged
rather than returned as a plausible-but-wrong number. The live transient
counterpart is `p.make_electrothermal_heatsink_observer(...)`.

---

## 5. Heatsink & TIM sizing

Turn geometry / airflow into the R_th values the API above consumes:

```python
R_cs = p.tim_resistance(area_m2=2.5e-4, thickness_m=1e-4,
                        material="silicone_pad")        # case → sink (TIM)
R_sa = p.convection_resistance(area_m2=0.03, airflow_m_per_s=3.0)  # sink → ambient
```

`p.TIM_CATALOG` lists conductivities for common pads/greases/insulators.
The TIM resistance is exact physics (`thickness/(k·area)`); the
convection coefficient is a first-cut estimate — **prefer the heatsink
datasheet's R_th-vs-airflow curve** for a real design.

---

## Worked example: an IPM at higher power

`projects/inverters/pfc_vsi_drive/thermal_comparison.py` applies the full
stack to a 6-IGBT + 6-diode inverter module and compares it to the
classic per-device-independent estimate. The takeaways:

- at the rated point the two agree within ~1 °C;
- pushing power up, the electro-thermal model runs **hotter** (the
  V_ce(T) feedback the fixed-coefficient model ignores): +8 °C at 4 kW,
  +20 °C at 5 kW — shaving ~500 W off the usable headroom before
  T_j,max;
- under degraded cooling the model **detects thermal runaway**
  (feedback gain ρ ≥ 1), which a fixed-coefficient model cannot.

---

## Reference

- API details: docstrings of `device_loss_summary`, `device_thermal_summary`,
  `shared_heatsink_steady_state`, `electrothermal_steady_state`,
  `tim_resistance`, `convection_resistance`.
- Loss physics: Erickson & Maksimović, *Fundamentals of Power
  Electronics* Ch. 3; Infineon/ON-Semi switching-loss application notes.
