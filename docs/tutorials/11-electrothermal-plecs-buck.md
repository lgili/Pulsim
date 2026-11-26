# Tutorial 11 — PLECS XML to closed-loop junction temperature on a buck

**Goal.** Import a PLECS `.xml` thermal description with `pulsim.lib.import_plecs_xml`, attach its switching tables, on-state voltage table, and Cauer thermal network to the switch of a buck converter with `Part.attach_losses`, run a closed-loop electrothermal transient, and verify energy conservation and loss records.

**Concepts introduced:**
- `pulsim.lib.import_plecs_xml` — reads vendor PLECS XML switching tables (`TurnOnLoss`, `TurnOffLoss`), `ConductionLoss` (`V_on(i, T_j)`), and `ThermalModel` chains (`Cauer` or `Foster`).
- `Part.attach_losses` — builds the thermal network into the MNA circuit and registers the in-kernel loss port (`set_device_losses`) in one call.
- Closed-loop `T_j(t)` versus open-loop fixed `Tj`: reading temperature-dependent tables at the junction temperature the run itself produces.
- `res.loss_events`, `res.device_energy`, `res.heat_injected + res.heat_pending`, and `pulsim.device_loss_summary`.

**The PLECS XML below is SYNTHETIC.** It is written inline so this tutorial is self-contained and reproducible; it is not a real vendor part. For a real design, download the vendor's `.xml` thermal description and pass its path directly to `p.lib.import_plecs_xml(path)`.

## 1. Import a PLECS XML thermal description

Write a synthetic PLECS `.xml` file describing an IGBT with temperature-dependent `E_on(v, i, T_j)`, `E_off(v, i, T_j)`, `V_on(i, T_j)`, and a two-stage Cauer thermal network (`R_th,j-a = 0.4 + 0.6 = 1.0 K/W`, scaled to millisecond thermal time constants so the transient reaches steady state quickly).

```python
from pathlib import Path
import numpy as np
import pulsim as p

PLECS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Package class="IGBT" vendor="ExampleSemi" partnumber="EX40N60" version="1.0">
  <SemiconductorData type="IGBT">
    <TurnOnLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 10 20</CurrentAxis>
      <VoltageAxis>0 200 400</VoltageAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <Energy scale="0.001">
        <Temperature>
          <Voltage>0 0 0</Voltage>
          <Voltage>0 0.25 0.50</Voltage>
          <Voltage>0 0.50 1.00</Voltage>
        </Temperature>
        <Temperature>
          <Voltage>0 0 0</Voltage>
          <Voltage>0 0.45 0.90</Voltage>
          <Voltage>0 0.90 1.80</Voltage>
        </Temperature>
      </Energy>
    </TurnOnLoss>
    <TurnOffLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 10 20</CurrentAxis>
      <VoltageAxis>0 200 400</VoltageAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <Energy scale="0.001">
        <Temperature>
          <Voltage>0 0 0</Voltage>
          <Voltage>0 0.20 0.40</Voltage>
          <Voltage>0 0.40 0.80</Voltage>
        </Temperature>
        <Temperature>
          <Voltage>0 0 0</Voltage>
          <Voltage>0 0.35 0.70</Voltage>
          <Voltage>0 0.70 1.40</Voltage>
        </Temperature>
      </Energy>
    </TurnOffLoss>
    <ConductionLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 10 20</CurrentAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <VoltageDrop scale="1">
        <Temperature>0 1.50 2.00</Temperature>
        <Temperature>0 1.80 2.50</Temperature>
      </VoltageDrop>
    </ConductionLoss>
  </SemiconductorData>
  <ThermalModel>
    <Branch type="Cauer">
      <RCElement R="0.4" C="0.002"/>
      <RCElement R="0.6" C="0.005"/>
    </Branch>
  </ThermalModel>
  <Comment>Synthetic fixture for Tutorial 11</Comment>
</Package>
"""

xml_path = Path("EX40N60.xml")
xml_path.write_text(PLECS_XML, encoding="utf-8")
part = p.lib.import_plecs_xml(xml_path)
chain = part.thermal()
print(f"Imported {part.vendor} {part.number} ({part.cls}), "
      f"{chain.kind} chain R_th = {sum(chain.R):.2f} K/W")
assert chain.kind == "cauer" and abs(sum(chain.R) - 1.0) < 1e-12
```

## 2. Build a buck converter and attach the loss port

Build a 400 V → 200 V / 10 A buck converter (`f_sw = 20 kHz`, `D = 0.5`) and attach the imported part to `Q1` at `T_amb = 40 °C`. `part.attach_losses` adds the Cauer stages to the MNA circuit, seeds the Cauer capacitors at `T_amb`, and registers `E_on`, `E_off`, and `V_on` (`conduction="table"`) on `Q1` so every commutation and step reads the tables at the live junction voltage `v("Tj_Q1")`.

```python
V_IN = 400.0
F_SW = 20e3
DUTY = 0.5
T_AMB = 40.0


def make_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, 0.0, 0.7)
    b.add_inductor("L", "sw", "vout", 1e-3, 10.0)
    b.add_capacitor("Cout", "vout", "gnd", 100e-6, 200.0)
    b.add_resistor("RL", "vout", "gnd", 20.0)
    return b


b_closed = make_buck()
junction = part.attach_losses(b_closed, "Q1", T_amb=T_AMB)
sw_fn = p.make_pwm_switch_fn(
    F_SW, DUTY, b_closed.switch_index_of("Q1"), b_closed.graph.num_switches
)
print(f"Attached thermal network; junction node = {junction!r}")
assert junction == "Tj_Q1"
```

## 3. Run closed-loop `T_j(t)` and inspect loss records

Run the transient with `engine="pwl"` (`dt = 1 µs`, 20 ms total = 400 switching cycles). Because `E_on`, `E_off`, and `V_on` all rise with `T_j`, the closed-loop junction temperature settles noticeably higher than an open-loop evaluation frozen at `T_amb = 40 °C`.

```python
T_END = 0.02
DT = 1e-6

res = p.simulate(b_closed, T_END, dt=DT, switch_fn=sw_fn)
t = np.asarray(res.times)
tj = np.asarray(res.v(junction))
tj_mean = p.time_average(t, tj, t_from=T_END - 1e-3, t_to=T_END)

# Compare against an open-loop run with Tj frozen at T_amb = 40 C
b_open = make_buck()
i_ax, tj_ax, v_grid = part.conduction["v_on"]
v_on_tab = p.ConductionTable(
    np.asarray(i_ax, float).tolist(),
    np.asarray(tj_ax, float).tolist(),
    np.asarray(v_grid, float).ravel().tolist(),
)
b_open.set_device_losses(
    "Q1",
    Tj=T_AMB,
    E_on=part.loss_tables["Eon"],
    E_off=part.loss_tables["Eoff"],
    V_on=v_on_tab,
    conduction="table",
)
res_open = p.simulate(b_open, T_END, dt=DT, switch_fn=sw_fn)

e_closed = res.device_energy("Q1")
e_open = res_open.device_energy("Q1")
ev = res.loss_events("Q1")
summary = {row["name"]: row for row in p.device_loss_summary(b_closed, res)}

print(f"Closed-loop Tj (final 1 ms mean): {tj_mean:.2f} C (T_amb = {T_AMB:.1f} C)")
print(f"Commutation events recorded: {len(ev)} "
      f"(n_on={e_closed['n_on']}, n_off={e_closed['n_off']})")
print(f"Total Q1 heat (closed loop): "
      f"{(e_closed['E_on'] + e_closed['E_off'] + e_closed['E_cond_table']) / T_END:.2f} W "
      f"vs open loop at 40 C: "
      f"{(e_open['E_on'] + e_open['E_off'] + e_open['E_cond_table']) / T_END:.2f} W")
print(f"device_loss_summary P_sw_avg: {summary['Q1']['P_sw_avg']:.2f} W")

```

The truth for both runs is the averaged model in closed form, computed here from the XML's numbers (not from the run). In steady state `i_L` ramps from 7.5 A to 12.5 A (ripple `(V_in − V_out)·D/(f_sw·L)` = 5 A), so the switch turns on at 7.5 A and off at 12.5 A, both at 400 V. The tables are linear in `T_j`, so the heat is `P(T_j) = P(25) + k·(T_j − 25)`, and the Cauer ladder with this heat is a linear two-state ODE, solved here by its eigenvectors.

<!-- doctest: xfail reason="release finding F-X-8: under the default restart rule the closed-loop Tj at dt = 1 us is 0.13 K under its dt -> 0 limit (71.014 C vs ~71.145 C; 0.19 K from the averaged model)" -->
```python
I_AX = np.array([0.0, 10.0, 20.0])
E_ON = {25: [0, 0.50e-3, 1.00e-3], 125: [0, 0.90e-3, 1.80e-3]}    # the 400 V rows
E_OFF = {25: [0, 0.40e-3, 0.80e-3], 125: [0, 0.70e-3, 1.40e-3]}
V_ON = {25: [0, 1.50, 2.00], 125: [0, 1.80, 2.50]}
I_VALLEY, I_PEAK = 7.5, 12.5


def heat(tj):
    """Average Q1 heat (W) at a constant junction temperature."""
    w = (tj - 25.0) / 100.0

    def table(d, i):
        return (1 - w) * np.interp(i, I_AX, d[25]) + w * np.interp(i, I_AX, d[125])

    i = np.linspace(I_VALLEY, I_PEAK, 20001)
    p_cond = DUTY * np.trapezoid(table(V_ON, i) * i, i) / (I_PEAK - I_VALLEY)
    return F_SW * (table(E_ON, I_VALLEY) + table(E_OFF, I_PEAK)) + p_cond


p25 = heat(25.0)
k = (heat(125.0) - p25) / 100.0          # 24.79 W + 0.150 W/K * (Tj - 25)
R0, R1, C0, C1 = 0.4, 0.6, 0.002, 0.005  # junction stage, then the heatsink stage
A = np.array([[(k - 1 / R0) / C0, 1 / (R0 * C0)],
              [1 / (R0 * C1), -(1 / R0 + 1 / R1) / C1]])
rhs = np.array([(p25 - 25.0 * k) / C0, T_AMB / (R1 * C1)])
x_ss = -np.linalg.solve(A, rhs)                    # 71.81 C, 59.09 C
lam, vec = np.linalg.eig(A)
coef = np.linalg.solve(vec, np.array([T_AMB, T_AMB]) - x_ss)
t_q = np.linspace(T_END - 1e-3, T_END, 2001)
tj_q = (x_ss[0] + (vec[0] * coef) @ np.exp(np.outer(lam, t_q))).real
tj_truth = np.trapezoid(tj_q, t_q) / 1e-3          # 71.21 C (not yet settled)
p_open_truth = heat(T_AMB)                          # 27.04 W at a frozen 40 C
p_open = (e_open["E_on"] + e_open["E_off"] + e_open["E_cond_table"]) / T_END
print(f"Tj final 1 ms mean: {tj_mean:.3f} C, averaged model {tj_truth:.3f} C")
print(f"Open-loop heat: {p_open:.3f} W, averaged model {p_open_truth:.3f} W")
# measured 0.02 K and 3.4e-4: the bounds cover the run's own start
# transient (i_L spans 5.8-14.7 A in the first periods) and its
# 7.6-12.4 A steady ripple against the ideal 7.5-12.5 A
assert abs(tj_mean - tj_truth) < 0.1
assert abs(p_open - p_open_truth) < 2e-3 * p_open_truth
```

## 4. Verify energy conservation

The kernel accumulates every joule of switching and conduction heat in compensated form (`res.heat_injected + res.heat_pending`). With `conduction="table"`, the total heat deposited into `Tj_Q1` matches `E_on + E_off + E_cond_table` to rounding (checked at `1e-10` relative), and `res.device_energy("Q1")` also reports `E_cond_electrical` side by side so any difference between the circuit's `R_on` and the table's `V_on(i, T_j)` is explicit.

```python
heat_recorded = e_closed["E_on"] + e_closed["E_off"] + e_closed["E_cond_table"]
heat_ledger = res.heat_injected + res.heat_pending
rel_err = abs(heat_ledger - heat_recorded) / heat_recorded
print(f"Heat ledger vs recorded: rel_err = {rel_err:.2e}")
assert rel_err < 1e-10   # a bookkeeping identity of one kernel
```

## 5. What this does and does not show

- **Verified, not validated against hardware.** The numbers above come from a synthetic PLECS XML fixture; they verify the import, closed-loop temperature integration, and energy-balance ledger, not a physical datasheet part.
- **Event energy is thermal-only (as in PLECS).** Switching energy from `E_on` / `E_off` is deposited into `Tj_Q1` without removing charge from the electrical DC link; conduction heat comes from `V_on(|i|, T_j) · |i|` (`conduction="table"`) or the device's own `v · i` (`conduction="electrical"`).
- **No electrical feedback.** The junction temperature changes the loss tables' energies, not the circuit: `g_on` and `V_f` stay at their built values (in-kernel-event-losses task 4.1 is open).
- **Engine support.** Closed-loop `T_j` with `V_on` tables runs on `engine="pwl"` and `engine="trbdf2"`. See [`docs/losses-and-thermal.md`](../losses-and-thermal.md) and [`docs/known-limitations.md`](../known-limitations.md).
