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
b.add_transformer(
    "T1",
    primary_pos="vin", primary_neg="sw_pri",
    secondary_pos="sec_pos", secondary_neg="sec_gnd",
    L_p=200e-6, L_s=200e-6, k=0.95,
)
b.add_mosfet ("Q1", "sw_pri", "gnd")   # low-side switch on primary
b.add_diode  ("Dout", "sec_pos", "vout", 1e3, 1e-9, V_th=0.7)
b.add_capacitor("Cout", "vout", "sec_gnd", 47e-6)
b.add_resistor ("R_L",  "vout", "sec_gnd", 10.0)
```

> **Note on isolation.** A real flyback's primary and secondary grounds
> are physically separate, and MNA needs every node to have a voltage
> reference — so the secondary must be tied to ground *somehow*.
> Since v2.0 you do not write that tie yourself: `simulate()` runs a
> topology preflight, spots the unreferenced secondary, inserts
> `R_auto_iso_sec_gnd` (1 GΩ) and tells you it did. Inspect it with
> `result._preflight`, or pass `auto_regularize=False` to get the
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
pwm = p.make_pwm_switch_fn(
    switch_index=0, frequency=F_PWM, duty=DUTY, phase=0.0,
)
```

Switch index 0 is `Q1`. The diode `Dout` has its own switch index (1), but the user does NOT drive it — event detection flips it automatically when current tries to flow backward.

## Run

```python
res = p.simulate(b, t_end=10e-3, dt=1e-7, switch_fn=pwm)

vout_idx = b.node_id_of("vout")
v = [s[vout_idx] for s in res.states]
k_skip = int(5e-3 / 1e-7)
print(f"V_out (mean over last half) = "
      f"{sum(v[k_skip:]) / (len(v) - k_skip):.2f} V")
```

For ideal 1:1 ratio + 40 % duty: `V_out ≈ V_in · D / (1 − D) ≈ 16 V` (continuous conduction; less in DCM, more with leakage spike).

## Two switch states, four commutations

The flyback alternates between two large-signal states:

1. **Q1 ON, Dout OFF.** Primary inductor energizes from `V_in`. Secondary has no path (D_out blocks the reverse polarity).
2. **Q1 OFF, Dout ON.** Magnetizing energy transfers to the secondary; `D_out` conducts; `C_out` is charged.

Every PWM edge triggers two commutations: Q1 flips, then a few hundred nanoseconds later D_out flips when the secondary voltage polarity inverts. the kernel's substep event detection (Layer 5 V2.1) finds both crossings exactly so the integration doesn't accumulate "diode briefly conducting in reverse" garbage.

## Try this next

- **Add a clamp.** Real flybacks have RCD snubbers or active clamps to handle leakage-inductance spikes. Add a series-R + parallel-C across `Q1` and watch the spike soften.
- **Multi-winding (Layer 2 V16).** Use `add_multi_winding_transformer` to build a 3-output flyback (main + aux bias + monitor) and verify each output settles independently.
- **Saturating core (Layer 2 V17).** Replace `add_transformer` with two `add_saturable_inductor` calls + a `TransformerCoupling` (advanced) — the boost-saturable showcase has the pattern.

## Going to [Tutorial 04](04-3phase-vsi.md)

Next we leave SMPS-land and look at a 3-phase inverter for motor drive applications.
