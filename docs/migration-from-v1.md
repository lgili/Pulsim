# Pulsim v1 → v2 Migration Guide

> **Status (May 2026):** v2 covers ≥ 95 % of v1 workloads. The
> remaining gaps are deliberate (see [§ 7](#7-known-gaps)). If you
> hit one, file an issue — most can be added quickly now that the
> infrastructure is in place.

## 1. Why v2 exists

v1's runtime (`core/include/pulsim/v1/runtime_circuit.hpp`,
~7 800 lines) is a single monolithic header carrying every device,
analysis, and helper. It works but accumulated technical debt:

- Build times grew to 30+ s on a clean repo because every
  translation unit pulls every device.
- Adding a new device meant touching the same 7 k-line file.
- Python bindings were a stamp-collection of overloads.
- The numerical core (PWL state-space, Newton, event tracker) was
  intertwined with the device library.

v2 splits this into focused, header-only modules:

```
core/include/pulsim/v2/
├── numeric/        ← scalar / vector types
├── topology/       ← graph + switch state mask
├── pwl/            ← cache, dc_assemble, device_pool, dc_strategy
├── solver/         ← run_transient, bdf1, options, result
├── analysis/       ← mna_sweep
├── blockchain/     ← chain + 14 control blocks
├── motors/         ← mechanical + dc/pmsm/bldc + adapters
├── thermal/        ← foster, electrothermal adapters
├── switchgear/     ← thyristor, fuse, snubber
├── sources/        ← pwm, sine, pulse, combined fn
├── grid/           ← three-phase helpers
├── magnetic/       ← steinmetz + igse loss models
├── models/         ← ideal_diode + smooth-blend variants
├── builder/        ← CircuitBuilder (the API)
└── yaml/           ← loader
```

The Python side mirrors that split (`python/pulsim/v2_*.py` →
~20 modules). Every helper landed in v2 has zero v1 dependency.

Performance wins (Release build, M2):

- Buck closed-loop, 0.5 s sim, 100 kHz PWM:
  v1 ≈ 12 s wall · v2 ≈ 1.1 s wall · **10–11 ×** speedup.
- Kernel rate ≈ 430 k steps/s (v1 was 24 k under identical load
  before the GIL/switch_fn/ring-buffer overhaul).

## 2. Quick mental model

| Concept            | v1                                | v2                                       |
| ------------------ | --------------------------------- | ---------------------------------------- |
| Top-level package  | `import pulsim`                   | `import pulsim.v2 as p`                  |
| Build a circuit    | `RuntimeCircuit()` + add_*        | `p.CircuitBuilder()` + add_*             |
| Run a transient    | `circuit.simulate(t_end)`         | `p.simulate(b, t_end, dt)`               |
| Control / blocks   | `SignalEvaluator` + closures      | `p.MixedDomainBlockChain` + `chain.add`  |
| DC OP              | `circuit.dc_operating_point()`    | `p.compute_dc_op(b)`                     |
| AC sweep           | `circuit.run_ac_sweep(freqs)`     | `p.run_ac_sweep(b, freqs)`               |
| Frequency response | `circuit.run_fra(...)`            | `p.run_fra(b, frequencies=..., ...)`     |
| Steady-state       | `circuit.run_periodic_shooting()` | `p.run_periodic_shooting(b, t_period)`   |
| Harmonic spectrum  | `circuit.run_harmonic_balance()`  | `p.run_harmonic_balance(b, period)`      |

In every case the v2 API takes the `builder` as the **first arg**
instead of being a method on a circuit object. This is deliberate:
the builder is pure topology, the analyses are free functions, the
results are dataclasses. No hidden state on a "circuit" object.

## 3. Devices — direct mapping

### Sources

| v1                              | v2                                          |
| ------------------------------- | ------------------------------------------- |
| `add_voltage_source(...)`       | `b.add_voltage_source(name, from, to, V)`   |
| `add_current_source(...)`       | `b.add_current_source(name, from, to, I)`   |
| `add_sine_voltage_source(...)`  | `b.add_sine_voltage_source(...)`            |
| `add_pulse_voltage_source(...)` | `b.add_pulse_voltage_source(...)`           |
| `add_pwm_voltage_source(...)`   | `b.add_pwm_voltage_source(...)`             |
| `add_three_phase_source(...)`   | `p.add_three_phase_grid(b, ...)`            |

### Passives

| v1                                  | v2                                                              |
| ----------------------------------- | --------------------------------------------------------------- |
| `add_resistor(name, a, b, R)`       | `b.add_resistor(name, from, to, R)`                             |
| `add_capacitor(...)`                | `b.add_capacitor(...)`                                          |
| `add_inductor(...)`                 | `b.add_inductor(...)`                                           |
| `add_saturable_inductor(...)`       | `b.add_saturable_inductor(name, from, to, L_0, I_sat, ...)`     |
| `add_transformer(...)`              | `b.add_transformer(name, p_from, p_to, s_from, s_to, Lp, Ls, k)`|
| `add_three_phase_rl_load(...)`      | `p.add_three_phase_rl_load(b, name, ...)`                       |

### Switches & semiconductors

| v1                                       | v2                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------- |
| `add_switch(...)`                        | `b.add_switch(name, from, to, g_on, g_off)`                          |
| `add_diode(...)` (PWL)                   | `b.add_diode(name, anode, cathode, g_on, g_off, V_th)`               |
| `add_nonlinear_diode(...)` (smooth)      | `b.add_nonlinear_diode(name, anode, cathode, V_F0, R_d, ...)`        |
| `add_mosfet(...)`                        | `b.add_mosfet(name, drain, source, R_on, R_off)`                     |
| `add_mosfet_with_body_diode(...)`        | `b.add_mosfet_with_body_diode(...)`                                  |
| `add_mosfet_level1(...)` (Shichman)      | `b.add_mosfet_level1(...)`                                           |
| `add_igbt(...)`                          | `b.add_igbt(name, collector, emitter, R_on, R_off)`                  |
| `add_igbt_level1(...)`                   | `b.add_igbt_level1(...)`                                             |
| `add_bridge_rectifier(...)`              | `p.add_bridge_rectifier(b, name, ac_a=, ac_b=, dc_pos=, dc_neg=)`    |

### Three-phase / power-stage helpers

| v1                                | v2                                                |
| --------------------------------- | ------------------------------------------------- |
| `add_three_phase_vsi(...)`        | `p.add_three_phase_vsi(b, name, vdc_pos=, ...)`   |
| `add_three_phase_rl_load(...)`    | `p.add_three_phase_rl_load(b, name, ...)`         |

### Motors

| v1                                            | v2                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| `add_dc_motor(...)`                           | `b.add_dc_motor(...)` (see `v2_motors`)                                         |
| `add_pmsm(...)`                               | `b.add_pmsm(...)`                                                               |
| `add_pmsm_foc(...)` (FOC chain pre-wired)     | `p.wire_pmsm_foc(chain, i_a_channel=, ..., v_dc=, f_pwm=, Kp_i=, Ki_i=)` — see §5 |
| `add_bldc_motor(...)`                         | `b.add_bldc_motor(...)`                                                         |

### Op-amps / VCVS

| v1                  | v2                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------- |
| `add_op_amp(...)`   | `b.add_op_amp_ideal(name, in_pos, in_neg, out, gain=1e5)`                           |
| `add_vcvs(...)`     | `b.add_vcvs(name, in_pos, in_neg, out_pos, out_neg, gain)`                          |

## 4. Analyses

| v1                                     | v2                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `circuit.simulate(t_end, dt)`          | `p.simulate(b, t_end=…, dt=…)`                                                                         |
| `circuit.dc_operating_point()`         | `p.compute_dc_op(b)` (Auto strategy by default)                                                        |
| `circuit.run_ac_sweep(freqs)`          | `p.run_ac_sweep(b, frequencies=…, input_source=…, output_node=…)`                                      |
| `circuit.run_fra(freqs, ...)`          | `p.run_fra(b, frequencies=…, injection_state_idx=…, measurements=[…])`                                 |
| `circuit.run_periodic_shooting(T)`     | `p.run_periodic_shooting(b, t_period=…, dt=…)`                                                         |
| `circuit.run_harmonic_balance(T)`      | `p.run_harmonic_balance(b, period=…, dt=…)` (returns spectrum + THD)                                   |
| `circuit.parameter_sweep(...)`         | `p.run_parameter_sweep(b, ...)` (see `v2_sweep`)                                                       |
| `circuit.run_adaptive(...)`            | `p.simulate(b, ..., adaptive=True)` or `p.run_transient_adaptive(b, ...)`                              |

The new `simulate(..., integrator='auto')` picks BDF1 automatically
when `dt ≥ 1 ms` (typical stiff thermal / slow-plant sims); trap
otherwise. Use `integrator='bdf1'` or `'trap'` to force.

## 5. Control / signal-flow — the biggest API change

v1 used `SignalEvaluator` + closures. v2 uses the
`MixedDomainBlockChain` which executes natively in C++ (so the
controller runs at full kernel rate without per-step Python).

### v1 (closure style)

```python
import pulsim as ps

class PIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp; self.Ki = Ki; self.integral = 0.0
    def step(self, ref, meas, dt):
        err = ref - meas
        self.integral += err * dt
        return self.Kp * err + self.Ki * self.integral

pi = PIController(Kp=0.02, Ki=150.0)
def controller(t, x):
    v_out = x[circuit.node_idx("vout")]
    duty = pi.step(12.0, v_out, dt)
    circuit.set_pwm_duty(duty)
```

### v2 (chain style)

```python
import pulsim.v2 as p

chain = p.MixedDomainBlockChain()
chain.add("lpf", p.FirstOrderLowPass(tau=500e-6),
            inputs=dict(input_value="vout", dt="dt"),
            output="v_filt")
chain.add("pi", p.PIController(Kp=0.020, Ki=150.0,
                                   output_min=0.05, output_max=0.95),
            inputs=dict(setpoint=12.0, measured="channel:v_filt",
                          dt="dt"),
            output="duty")
chain.add("pwm", p.PwmGenerator(frequency=100e3),
            inputs=dict(duty="channel:duty", t="time"),
            output="gate")

observer = chain.make_step_observer(b, dt=DT)
switch_fn = chain.make_pwm_switch_fn("gate",
                                          num_switches=b.graph.num_switches,
                                          switch_idx=0)

p.simulate(b, t_end=T, dt=DT, switch_fn=switch_fn,
              step_observer=observer)
```

The chain compiles to C++ on first use — no Python in the kernel
hot loop. For switching converters this is the difference between
24 k and 430 k kernel steps/sec.

`chain.make_pwm_switch_fn` automatically picks the C++ closure path
when the chain is compiled, falling back to Python for the rare
edge case where `make_step_observer` wasn't called. No `use_kernel=`
flag — the right thing happens by default.

### FOC pre-built helper

v1 had `add_pmsm_foc(...)` that bundled the FOC chain inside the
device. v2 keeps power-stage and control separate but ships
`wire_pmsm_foc` to assemble the standard chain in one call:

```python
vsi = p.add_three_phase_vsi(b, "INV", vdc_pos="dc+", vdc_neg="dc-",
                                out_a="ua", out_b="ub", out_c="uc")
chain = p.MixedDomainBlockChain()
# ... add theta-tracker + current-sensing blocks ...
foc = p.wire_pmsm_foc(chain,
          i_a_channel="i_a", i_b_channel="i_b", i_c_channel="i_c",
          theta_elec_channel="theta_e",
          v_dc=400.0, f_pwm=10e3, Kp_i=0.5, Ki_i=2000.0,
          i_q_ref=5.0)
sw_fn = chain.make_multi_pwm_switch_fn(
    foc.gate_hs_channels,
    num_switches=b.graph.num_switches,
    switch_indices=vsi.high_side_switch_indices)
```

Output channel names (`foc.i_d`, `foc.v_q`, `foc.duty_a`, etc.) are
returned in the `FocWiringResult` for tuning and logging.

## 6. Loss accounting + thermal

| v1                                  | v2                                                              |
| ----------------------------------- | --------------------------------------------------------------- |
| `LossAccumulator`                   | `p.LossAccumulator(label=…)`                                    |
| `EfficiencyCalculator.from_power`   | `p.EfficiencyCalculator.from_power(P_in, P_out)`                |
| `add_foster_network(...)`           | `p.add_foster_network(b, stages, junction_node=…)`              |
| `add_cauer_network(...)`            | `p.add_cauer_thermal_network(b, [p.CauerStage(R, C), ...])`     |
| `ThermalLimitMonitor`               | `p.ThermalLimitMonitor(T_limit_C=…, hysteresis_C=…)`            |
| `JilesAtherton(...)` hysteresis     | `p.JilesAthertonModel(p.JilesAthertonParams(Ms, a, alpha, c, k))` |
| Core-loss B-H loop integration      | `p.compute_bh_loop(current, times, params=…, N_turns=…, A_core=…)` |

Thermal monitors integrate cleanly with `simulate()`:

```python
mon = p.ThermalLimitMonitor(T_limit_C=150.0, hysteresis_C=10.0)
def observe(t, x):
    mon.update(t, x[T_j_idx])
res = p.simulate(b, t_end=10.0, dt=1e-4,
                    step_observer=observe,
                    should_continue=mon.should_continue)
if mon.tripped:
    print(f"Tripped at {mon.trip_time*1e3:.1f} ms, "
          f"T_j peaked at {mon.peak_temperature:.1f} °C")
```

## 7. Known gaps

These v1 features aren't yet in v2. None are blockers for the
common workloads; each is doable when needed.

- **BehavioralMOSFET / IGBT (tanh-smoothed)** — use
  `add_mosfet_level1` instead (Shichman-Hodges; also smooth and
  Newton-friendly).
- **Single-phase induction motor** — domain-specific. The
  three-phase IM is roughly possible by hand-wiring stator + rotor
  circuits; the single-phase split-cap variant needs a dedicated
  device model.
- **Code generation (C99 export) + FMU 2.0** — v2 doesn't yet ship
  the embedded-code generator or the FMU exporter that v1 has. If
  you depend on these, stay on v1 for that flow.
- **Schematic auto-layout** — v1 has `pulsim.schematic`; v2
  doesn't yet. Use a separate tool if you need this.
- **YAML loader breadth** — v2's YAML loader supports 20 device
  kinds vs v1's 52. The missing kinds are mostly control blocks
  (PIController, PwmGenerator, etc.); those live in the chain,
  not the device pool. The YAML schema for `chain:` is on the
  Phase-4 list.

## 8. Practical workflow port — buck CL example

Here's a v1 closed-loop buck transcribed to v2 idiomatic v2:

### v1

```python
import pulsim as ps
circuit = ps.RuntimeCircuit()
circuit.add_voltage_source("Vin", "vin", "gnd", 24.0)
circuit.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                      R_on=1e-3, V_F=0.7)
circuit.add_diode("D_FW", "gnd", "sw", 1e3, 1e-9)
circuit.add_inductor("L1", "sw", "vout", 100e-6)
circuit.add_capacitor("Cout", "vout", "gnd", 47e-6)
circuit.add_resistor("R_load", "vout", "gnd", 5.0)

pi = ps.PIController(Kp=0.02, Ki=150.0, integral_limit=0.95)
def control(t, x):
    v = x[circuit.node_idx("vout")]
    duty = pi.step(12.0, v, 1e-7)
    circuit.set_pwm_duty("Q1", duty)
result = circuit.simulate(t_end=0.5, dt=1e-7, step_observer=control)
```

### v2

```python
import pulsim.v2 as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", 24.0)
b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                R_on=1e-3, R_off=1e9, V_F=0.7)
b.add_diode("D_FW", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
b.add_inductor("L1", "sw", "vout", 100e-6)
b.add_capacitor("Cout", "vout", "gnd", 47e-6)
b.add_resistor("R_load", "vout", "gnd", 5.0)

chain = p.MixedDomainBlockChain()
chain.add("lpf", p.FirstOrderLowPass(tau=500e-6),
            inputs=dict(input_value="vout", dt="dt"), output="v_filt")
chain.add("pi", p.PIController(Kp=0.020, Ki=150.0,
                                   output_min=0.05, output_max=0.95),
            inputs=dict(setpoint=12.0, measured="channel:v_filt",
                          dt="dt"),
            output="duty")
chain.add("pwm", p.PwmGenerator(frequency=100e3),
            inputs=dict(duty="channel:duty", t="time"), output="gate")

observer = chain.make_step_observer(b, dt=1e-7)
sw_fn    = chain.make_pwm_switch_fn("gate",
                                          num_switches=b.graph.num_switches,
                                          switch_idx=0)

result = p.simulate(b, t_end=0.5, dt=1e-7,
                       switch_fn=sw_fn, step_observer=observer)
```

The v2 version is ~3 lines longer but runs roughly 10× faster (in
Release build) AND exposes the controller as named channels that
can be logged, swapped, or replaced without touching the
simulation call.

## 9. What you get for free in v2

Beyond direct equivalents, v2 ships features v1 didn't have:

- **Live scope GUI** (`p.LiveScope`) — real-time pyqtgraph display
  fed by a lock-free C++ ring buffer. STOP button halts the
  kernel between dts.
- **Sub-step state correction** auto-detected when the circuit
  has PWL diodes or smooth nonlinear devices.
- **`simulate(adaptive=True)`** — variable-step driver that grows
  `dt` during slow segments (RL settling, thermal tails).
- **Native ring-buffer streaming** for log-anything-fast pipelines.
- **OpenSpec-tracked changelogs** for the kernel.

## 10. When to file a v2 issue vs stay on v1

Stay on v1 if **today** you need:
- C99 codegen for an embedded target
- FMU 2.0 export
- Schematic auto-layout / netlistsvg
- Hard real-time bound (v2 hasn't profiled WCET yet)

Move to v2 if you need:
- Faster sims (10× on switching converters)
- Cleaner control composition (chain blocks)
- A live scope GUI
- Periodic shooting / FRA / harmonic balance
- Newer thermal helpers (Cauer + monitor)
- Hysteresis loss accounting (Jiles-Atherton)

For anything else, both should work — file an issue if you find a
v1 feature you need on v2.
