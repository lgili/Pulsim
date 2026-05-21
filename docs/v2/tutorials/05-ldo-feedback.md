# Tutorial 05 — LDO with op-amp feedback (VCVS + MOSFET)

**Goal.** Build a linear regulator (LDO) using a high-gain VCVS as an error amplifier, a pass MOSFET, and a resistive feedback divider. Verify that the output settles to the reference voltage under a step load change.

**Concepts introduced:**
- `add_vcvs` (Layer 2 V15) — voltage-controlled voltage source as ideal op-amp.
- Negative-feedback loop topology — the virtual-short property.
- Why Newton refresh is automatic once you add a nonlinear device (MOSFET).

**Reference YAML:** [`examples/v2/ldo_with_opamp.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/v2/ldo_with_opamp.yaml).

## Topology

```
                                ┌───── Q1 (pass MOSFET) ────────── vout
   V_in ────┬─────────────┐    │                                     │
            │             │    │ gate                                R1 (top of divider)
            │           ─ │── ─ ─ ─ ─                                 │
            │          │            │                                 fb_node
            │          │ ─ op_amp  ─│                                 │
            │          │            │                                 R2
            │           ─ │── ─ ─ ─ ─                                 │
            │             │    │ +                                    │
            │             │    └── v_ref                              ⏚
            ⏚            ⏚
```

The op-amp (VCVS) compares `v_fb = v_out · R2/(R1+R2)` against `v_ref`. The output drives the MOSFET gate; negative feedback closes around the loop.

For `v_ref = 1.25 V`, `R1 = 2.75 kΩ`, `R2 = 1.25 kΩ` → `V_out = v_ref · (R1+R2)/R2 = 4 V`.

## Python build

```python
import pulsim.v2 as p

V_IN  = 12.0
V_REF = 1.25
GAIN  = 1e5     # op-amp open-loop gain

b = p.CircuitBuilder()
b.add_voltage_source("Vin",  "vin",  "gnd", V_IN)
b.add_voltage_source("Vref", "vref", "gnd", V_REF)

# Op-amp: VCVS with high gain. (in_pos = vref, in_neg = fb_node)
b.add_vcvs("U1",
           in_pos="vref", in_neg="fb_node",
           out_pos="gate", out_neg="gnd",
           gain=GAIN)

# Pass MOSFET (drain = V_in, source = vout, gate driven by op-amp).
b.add_mosfet_level1(
    "Q1", "vin", "vout", "gate",
    K=1e-2, V_T=2.0, lambda_=0.02, kappa=15.0,
)

# Output filter + feedback divider + load.
b.add_capacitor("Cout", "vout", "gnd", 10e-6)
b.add_resistor ("R1",   "vout", "fb_node", 2.75e3)
b.add_resistor ("R2",   "fb_node", "gnd",  1.25e3)
b.add_resistor ("R_L",  "vout", "gnd",     100.0)
```

## Run

```python
res = p.simulate(b, t_end=2e-3, dt=1e-6)
# simulate() auto-detects the MOSFET → enable_nonlinear_refresh=True.

vout_idx = b.node_id_of("vout")
print(f"V_out (final) = {res.states[-1][vout_idx]:.3f} V (target 4.0 V)")
```

Expected:
```
V_out (final) = 3.998 V (target 4.0 V)
```

## What the closed loop is doing

The VCVS enforces `V_gate − 0 = 10⁵ · (V_ref − V_fb)`. With enough gain, even a tiny mismatch between `V_ref` and `V_fb` drives the gate to whatever voltage the MOSFET needs to deliver exactly the right `V_out`. In steady state, `V_fb = V_ref` (the virtual short), so `V_out = V_ref · (R1+R2)/R2`.

The MOSFET operates in saturation; its current is `K · (V_GS − V_T)²` which is a nonlinear function of the gate voltage. Newton has to find the consistent `(V_gate, V_out, I_D)` triple at every time step.

## Loop-stability gotcha

This circuit can ring or oscillate if `Cout` is too small relative to the gain-bandwidth product. The simulation will show:
- Smooth settling (good): exponential approach to 4 V.
- Mild ringing (acceptable): a couple of overshoots.
- Diverging Newton (bad): the gate voltage explodes; `simulate(...)` will throw a convergence error.

If you hit the third case, increase `Cout`, lower `GAIN`, or add a compensation pole (RC in series with the gate or in the feedback path).

## Try this next

- **Step load.** Add a switch in parallel with `R_L` that doubles the load at `t = 1 ms`. Watch the output dip and recover.
- **Bode plot.** Once the [AC analysis proposal](https://github.com/lgili/Pulsim/blob/main/openspec/changes/add-pulsim-v2-ac-analysis/proposal.md) ships, sweep the loop gain from 10 Hz to 1 MHz and verify the crossover frequency.
- **Replace the VCVS with two cascaded VCVSs.** Models a 2-stage op-amp; lets you add a dominant pole between stages.

## Going to [Tutorial 06](06-igbt-boost.md)

The last tutorial covers the IGBT boost convergence story — the one that drove the V13.2 smooth-clamp + V14 rise-time work.
