# Tutorial 03 — Isolated flyback (transformer + commutation)

**Goal.** Drive a 1:1 isolated flyback converter and verify that the secondary-side capacitor settles to the expected output voltage given the duty cycle.

**Concepts introduced:**
- `add_transformer` (Layer 2 V2) — two-winding coupled inductors with mutual coupling `k`.
- Multi-rail topologies (primary-side and secondary-side share only the transformer).
- Why `SwitchedDiode` event detection is critical for hard commutation.

**Reference YAML:** [`examples/flyback.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/flyback.yaml).

## Topology

```
           ┌─────┬───────┐                       ┌────D_out───┬──── vout
           │     │       │                       │            │
   Vin ────┤    L_p     L_s   (k = 0.95)         │            C_out
           │     │       │                       │            │
           │     ▼       │   secondary ground ───┴────────────┴─── gnd_sec
           │   Q1                                              R_L
           │ (LS switch)
           │     │
           ⏚    ⏚ (primary gnd)
```

Two galvanically isolated subnets, connected only by the magnetic coupling between `L_p` and `L_s`. The transformer turns-ratio is set implicitly by `L_p / L_s` (here 1:1).

## Python build

```python
import pulsim as p

V_IN  = 24.0
F_PWM = 100e3
DUTY  = 0.40

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", V_IN)
# add_transformer(name, p_from, p_to, s_from, s_to, L_p, L_s, k).
# The secondary is wound sec_gnd -> sec_pos: while Q1 is ON the
# primary sees +V_in from vin to sw_pri, which drives sec_pos
# NEGATIVE and reverse-biases Dout. That is what makes it a flyback
# (store while ON, release while OFF). Winding it sec_pos -> sec_gnd
# conducts during the ON time -- a forward converter's polarity.
b.add_transformer("T1", "vin", "sw_pri", "sec_gnd", "sec_pos",
                  L_p=200e-6, L_s=200e-6, k=0.95)
b.add_mosfet ("Q1", "sw_pri", "gnd")   # low-side switch on primary
b.add_diode  ("Dout", "sec_pos", "vout", 1e3, 1e-9, V_th=0.7)
b.add_capacitor("Cout", "vout", "sec_gnd", 47e-6)
b.add_resistor ("R_L",  "vout", "sec_gnd", 10.0)
```

> **Note on isolation.** A real flyback's primary and secondary grounds
> are physically separate, and MNA needs every node to have a voltage
> reference — so the secondary must be tied to ground *somehow*.
> Since v2.0 you do not write that tie yourself: `simulate()` runs a
> topology preflight, spots the unreferenced secondary, inserts a
> 1 GΩ tie and tells you exactly which node it anchored to (the
> lowest-numbered node of the subnet — here `sec_gnd`, so the device
> is called `R_auto_iso_sec_gnd`; which node carries the tie does not
> matter, since all it does is fix the subnet's reference). The tie
> goes on the run's own copy of the circuit — your builder is not
> changed. Inspect it with `result.plan.ties` (the circuit that ran is
> `result.circuit`), or pass `auto_regularize=False` to get the
> original named error instead.
>
> If you do write it by hand, make it **large**:
>
> ```python
> b.add_resistor("R_iso", "sec_gnd", "gnd", 1e9)   # reference, not a bond
> ```
>
> Earlier versions of this tutorial suggested 1 µΩ. That is wrong and
> the reason matters: a 1 µΩ tie is a galvanic **bond**. It does not
> give the secondary a reference — it welds it to primary ground,
> destroying the very isolation the circuit exists to provide, and it
> does so silently. A 1 GΩ tie draws nanoamps and leaves the physics
> untouched.

## Switch driver

```python
q1_bit = b.switch_index_of("Q1")
pwm = p.make_pwm_switch_fn(F_PWM, DUTY, q1_bit, b.graph.num_switches)
```

`add_mosfet` adds Q1 *and* its body diode `Q1_body`, so the switching
branches are Q1 (bit 0), `Q1_body` (bit 1) and `Dout` (bit 2) — ask
`switch_index_of` rather than counting. You do NOT drive the diode
bits: event detection flips them when their current tries to reverse.

## Run

```python
import numpy as np

res = p.simulate(b, t_end=10e-3, dt=1e-7, switch_fn=pwm)

# Measure the output across its own load: vout relative to sec_gnd.
vout_idx, sgnd_idx = b.node_id_of("vout"), b.node_id_of("sec_gnd")
t = np.asarray(res.times)
v = np.array([s[vout_idx] - s[sgnd_idx] for s in res.states])
m = t >= 5e-3                       # last half, time-weighted
v_mean = np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0])
print(f"V_out (mean over last half) = {v_mean:.2f} V")
```

Expected output (the run also prints the preflight warning about the
`R_auto_iso_sec_gnd` tie described above):
```
V_out (mean over last half) = 10.73 V
```

For an ideal 1:1 ratio at 40 % duty in continuous conduction,
`V_out = V_in · D / (1 − D) = 16 V`. The run lands well below that
because `k = 0.95` leaves 5 % of each winding (10 µH) as leakage, and
nothing clamps it: the leakage energy rings the primary switch node up
to ≈ 770 V at every turn-off instead of reaching the output. Raise the
coupling and the output climbs to the ideal value — `k = 0.99` gives
14.57 V and `k = 0.999` gives 15.98 V (same run, mean over 9–10 ms).

## Two switch states, four commutations

The flyback alternates between two large-signal states:

1. **Q1 ON, Dout OFF.** Primary inductor energizes from `V_in`. Secondary has no path (D_out blocks the reverse polarity).
2. **Q1 OFF, Dout ON.** Magnetizing energy transfers to the secondary; `D_out` conducts; `C_out` is charged.

Every PWM edge triggers two commutations: Q1 flips, then a few hundred nanoseconds later D_out flips when the secondary voltage polarity inverts. the kernel's substep event detection (Layer 5 V2.1) finds both crossings exactly so the integration doesn't accumulate "diode briefly conducting in reverse" garbage.

## Try this next

- **Add a clamp.** Real flybacks have RCD snubbers or active clamps to handle leakage-inductance spikes. Add a series-R + parallel-C across `Q1` and watch the spike soften.
- **Multi-winding (Layer 2 V16).** Use `add_multi_winding_transformer` to build a 3-output flyback (main + aux bias + monitor) and verify each output settles independently.
- **Saturating core (Phase 4 C.4).** Replace `add_transformer` with `add_saturable_transformer(name, p_from, p_to, s_from, s_to, N_p, N_s, Ae, le, lg, B_sat=…, L_leak_p=…)`: the same T-model, with the magnetising branch generated from the core's geometry and gap, so an overload or a zero-crossing energisation drives the core past `B_sat` and the current runs away the way it does on the bench. Below the knee it reproduces `add_transformer` exactly (`L_p = L_leak_p + L_m`, `M = n·L_m`, `L_s = n²·L_m`).

  Do **not** build it from two `add_saturable_inductor` calls plus a `TransformerCoupling`. Saturable inductors are Newton devices outside the linear history, so that arrangement couples the Jacobian but not the companion history: it converges, and it is wrong.

## Going to [Tutorial 04](04-3phase-vsi.md)

Next we leave SMPS-land and look at a 3-phase inverter for motor drive applications.
