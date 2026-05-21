# Tutorial 01 — RC charging from a pulse source

**Goal.** Drive an RC integrator with a `PulseVoltageSource` and check that `v_C(t)` matches the analytical `V · (1 − exp(−t/τ))` exponential.

**Concepts introduced:**
- The simplest possible v2 setup: one source, one R, one C, one node.
- The `PulseVoltageSource` (Layer 2 V12) — SPICE-style PULSE waveform.
- Computing the time-constant `τ = R·C` and sanity-checking with three sampled points.

**Reference YAML:** [`examples/v2/rlc_step_response.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/v2/rlc_step_response.yaml) (also contains an inductor — for pure RC see the snippet below).

## Topology

```
     ┌──── R ───┐
     │          ▼
Vstep         vc ────┤├── gnd
     │          │
     └────►─────┘  C
```

- `Vstep`: pulse from 0 V → 10 V at t = 0, held forever (single-shot).
- `R = 1 kΩ`, `C = 1 µF` → `τ = R·C = 1 ms`.

## Python build

```python
import pulsim.v2 as p

V_step = 10.0
R = 1000.0
C = 1.0e-6
tau = R * C

b = p.CircuitBuilder()
b.add_pulse_voltage_source(
    "Vstep", "n0", "gnd",
    v_initial=0.0, v_pulsed=V_step,
    t_start=0.0, pulse_width=10.0,   # held essentially forever
)
b.add_resistor ("R", "n0", "vc", R)
b.add_capacitor("C", "vc", "gnd", C)

dt = 1e-6
res = p.simulate(b, t_end=5 * tau, dt=dt)
```

## Verify against the analytical curve

```python
import math

vc_idx = b.node_id_of("vc")
print(f"τ = {tau*1e3:.1f} ms")
print("k·τ |  v_sim    v_exp")
for k_tau in (1, 2, 3):
    t_check = k_tau * tau
    idx = int(t_check / dt)
    v_sim = res.states[idx][vc_idx]
    v_exp = V_step * (1.0 - math.exp(-t_check / tau))
    print(f"  {k_tau}  | {v_sim:6.3f}   {v_exp:6.3f}")
```

Expected:
```
τ = 1.0 ms
k·τ |  v_sim    v_exp
  1  |  6.321    6.321
  2  |  8.647    8.647
  3  |  9.502    9.502
```

The match is exact to several digits — there is no discretisation error from the trapezoidal companion for first-order responses.

## What just happened

1. `add_pulse_voltage_source` registered a `PulseVoltageSource`-kind branch with `v_initial = 0`, `v_pulsed = 10`. The MNA system is stamped with `V = 0` baseline; the time-varying value lives in the `b_extra` overlay built by `run_transient`.
2. `simulate(...)` called `cache.build(dt=1e-6)`. Since there are no switches, exactly one PWL cache entry is created and KLU-factorized.
3. Each `dt`-step retrieves that single cached entry and performs a back-substitution against the current `b_extra`. With 5000 steps over 5 ms, the whole sim runs in well under a millisecond.

## Try this next

- **Periodic pulse train.** Set `period = 3 * tau` and watch the capacitor charge/discharge cycle. Then add `rise_time = fall_time = 0.1 * tau` for SPICE-style ramps.
- **Inject mid-sim.** Use `t_start = 2 * tau` to delay the step → the capacitor sits at 0 V for two time constants, then charges. Plot to see the dead-time.
- **Move to YAML.** Translate the same circuit to `examples/v2/rlc_step_response.yaml` — the test `core/tests/v2/showcases/test_rlc_step_response.cpp` already covers this end-to-end.
