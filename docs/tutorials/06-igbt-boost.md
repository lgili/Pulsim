# Tutorial 06 — IGBT boost with realistic gate drive

**Goal.** Build a boost converter using a Level-1 IGBT (`add_igbt_level1`) driven by a `PulseVoltageSource` with explicit rise/fall ramps. Understand the V13.2 + V14 convergence story along the way.

**Concepts introduced:**
- `add_igbt_level1` (Layer 2 V14) — linear-conduction IGBT.
- `PulseVoltageSource` with `rise_time` / `fall_time` (Layer 2 V14 extension).
- Why ramped gate drives are essential for Newton convergence on inductive loads.

**Reference YAML:** [`examples/boost_realistic_igbt.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/boost_realistic_igbt.yaml).

## Topology

```
            ┌── L ──┐         ┌── D_out ──┬─── vout ── load
            │       │         │           │
   V_in ────┤      sw ────────┤            C_out
            │       │         │           │
            ⏚      ▼         ⏚          ⏚
                   IGBT (collector=sw, emitter=gnd)
                       gate ←─── pulse source
```

`L` charges from `V_in` when the IGBT is ON; when it opens, the inductor's voltage forward-biases `D_out` and dumps energy into `C_out`.

## Python build

```python
import pulsim as p

V_IN     = 12.0
F_PWM    = 50e3
DUTY     = 0.40
T_RISE   = 100e-9   # 100 ns IGBT-realistic gate rise
T_FALL   = 100e-9

T_PERIOD = 1.0 / F_PWM
PULSE_W  = DUTY * T_PERIOD - T_RISE  # account for rise/fall margins

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", V_IN)
b.add_inductor      ("L",   "vin", "sw",  220e-6)

# IGBT (collector = sw, emitter = gnd, gate = 'g').
b.add_igbt_level1(
    "T1", "sw", "gnd", "g",
    V_CE_sat=2.0, R_CE_sat=0.05, V_T=5.0, kappa=10.0,
)

b.add_diode    ("Dout", "sw", "vout", 1e3, 1e-9, V_th=0.7)
b.add_capacitor("Cout", "vout", "gnd", 47e-6)
b.add_resistor ("R_L",  "vout", "gnd", 20.0)

# Gate drive — pulse source with realistic rise/fall (V14).
b.add_pulse_voltage_source(
    "Vg", "g", "gnd",
    v_initial=0.0, v_pulsed=15.0,    # 15 V gate drive
    t_start=0.0,
    rise_time=T_RISE,
    pulse_width=PULSE_W,
    fall_time=T_FALL,
    period=T_PERIOD,
)
```

## Run

```python
res = p.simulate(b, t_end=2e-3, dt=2e-8)
# Newton refresh auto-enabled thanks to the IGBT.

vout_idx = b.node_id_of("vout")
v_mean = sum(s[vout_idx] for s in res.states[len(res.states)//2:]) / (len(res.states)//2)
boost_ideal = V_IN / (1.0 - DUTY)
print(f"V_out (steady-state mean) = {v_mean:.2f} V"
      f" (ideal boost: {boost_ideal:.2f} V)")
```

Expected:
```
V_out (steady-state mean) = 18.45 V (ideal boost: 20.00 V)
```

The shortfall is due to `V_CE_sat ≈ 2 V` IGBT drop plus the diode V_F ≈ 0.7 V — the **dominant losses** in IGBT-based converters, exactly why IGBTs lose to SiC MOSFETs at high frequencies.

## The convergence story (why this isn't trivial)

A naive boost-with-IGBT setup using:
- `add_pulse_voltage_source` with `rise_time = fall_time = 0` (instant edges)
- No body diode on the switch

… **diverges in Newton** within a few PWM cycles. Here's why:

1. **Step gate transitions** force the IGBT current to change discontinuously. The trapezoidal-rule companion model for `L` says `v_L = (L/dt)·(i_L − i_L_prev) − v_L_prev`; if `i_L` is forced to discontinuously change, the residual is huge and the gate transitioning instantly is fighting that.
2. **No body diode** means there's no path for the inductor current at the exact instant the IGBT opens. The MNA matrix becomes singular for one Newton step (a row with no entries).
3. The Newton iteration **overshoots the V_DS smooth-clamp region** and lands in the spurious `V_DS < 0` polynomial root that the SH1 / IGBT formula has but real devices never visit.

### How v2 solves it

- **V13.2 smooth V_DS clamp.** The MOSFET / IGBT current law has an extra sigmoid that drives `I → 0` for `V_DS < −0.5 V`, eliminating the spurious root.
- **V14 pulse rise/fall ramps.** Set `rise_time = fall_time = 100 ns` (or anything > a few `dt`s). This gives Newton several iterations to "follow" the gate transition smoothly instead of jumping.
- **Body-diode helper (proposal #3.1).** For MOSFETs (not IGBTs — they have no body diode physically), `add_mosfet_level1(..., with_body_diode=True)` adds an anti-parallel diode that keeps the inductor current flowing during dead-time.

### What you can do if it still doesn't converge

1. **Reduce `dt`.** A `dt` of 10 % of the rise time is a good starting heuristic.
2. **Tune `kappa`.** Higher `kappa` makes the sigmoid steeper (more accurate at the boundary, harder to converge). Lower `kappa` (5-10) is safer.
3. **Use `start_from_dc_op = True`.** Computes the DC operating point first so Newton starts from a feasible state, not from `x = 0`.

## Try this next

- **Cascade two boosts.** A 12 V → 20 V → 33 V two-stage. Each stage is its own IGBT + L + D + C cell. Pulse them at slightly different frequencies and watch the interaction.
- **Replace IGBT with SH1 MOSFET.** Same topology, `add_mosfet_level1(..., with_body_diode=True)`. Expect lower V_DS_sat losses and slightly faster convergence.
- **Add a snubber.** RCD across the IGBT (1 Ω + 1 nF + diode) absorbs the inductive-spike transient. The pre-snubber `v_CE_peak` will be lower.

## Where to go from here

You've seen every major v2 feature:
- Passive trapezoidal-rule companion (L, C).
- Switch + event detection.
- Transformers (single + multi-winding).
- Smooth-blend nonlinear devices (diode, MOSFET, IGBT).
- Saturable magnetics (boost saturable showcase).
- Time-varying sources (sine, pulse, PWM) + their DC operating points.
- Op-amp / VCVS feedback loops.
- Steinmetz core-loss post-processing.

Next steps for the curious reader:
- [API reference](../api-reference.md) — Python surface in one page.
- [Gotchas](../gotchas.md) — Newton convergence + body-diode + cache-size corner cases.
- `core/include/pulsim/` — read the layer headers; they're written to be readable.
- `openspec/changes/pulsim-*` — every design decision, in chronological order.
