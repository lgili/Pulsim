# Components Reference

> **Status:** authoritative — covers every YAML `type:` value the parser
> recognizes, every parameter it reads, and every default it applies.
> Source: `core/src/v1/yaml_parser.cpp` and the `add_*` methods on
> `pulsim.v1.Circuit`. Updated through Phase 28.

This page is the master catalog of every **electrical-stamping** component
Pulsim knows about — the things that contribute rows or columns to the MNA
matrix. For control-domain blocks (PWM generators, PI controllers, Clarke /
Park transforms, PLLs, etc.) see [Control Blocks Reference](control-blocks-reference.md).

## Conventions

Every component is a YAML map with at least `type`, `name`, and `nodes`:

```yaml
- type: <component-type>
  name: <unique-identifier>
  nodes: [<node-a>, <node-b>, ...]
  # parameters either at top level...
  resistance: 10.0
  # ...or nested under params:
  params:
    ic: 0.0
```

- Numeric scalars accept SI multiplier suffixes (`220u`, `47n`, `1.5k`,
  `2.2Meg`, `47e-6`). Lowercase `m = 1e-3`, capital `M = 1e6` only when
  written `M` (use `meg`/`Meg` to disambiguate). Suffixes come from
  `parse_real_string` in the parser.
- The node `0` is ground. Any other unique string becomes a numbered node.
- Aliases listed below all resolve to the same canonical type (e.g. `r`,
  `R`, and `resistor` are interchangeable).
- Surrogate components (marked ⚠ below) stamp a simpler primitive and
  emit a `PULSIM_YAML_W_COMPONENT_SURROGATE` warning at load time so you
  know the model is a first-parity slice, not a full physical model.

## Quick index

| Category | Types |
|---|---|
| Passives | `resistor`, `capacitor`, `inductor`, `snubber_rc`, `transformer`, `coupled_inductor`, `saturable_inductor` |
| Sources | `voltage_source` (DC / PWM / sine / pulse), `current_source` |
| Switching | `diode`, `switch`, `vcswitch`, `mosfet` (`nmos` / `pmos`), `igbt`, `bjt_npn` ⚠, `bjt_pnp` ⚠, `thyristor` ⚠, `triac` ⚠ |
| Protection | `fuse` ⚠, `circuit_breaker` ⚠, `relay` ⚠ |
| Probes (no stamp) | `voltage_probe`, `current_probe`, `power_probe`, `electrical_scope`, `thermal_scope` |

---

## 1. Passive components

### `resistor`

| Alias | Nodes | Linear | Dynamic |
|---|---|---|---|
| `r` | `[n1, n2]` | yes | no |

```yaml
- type: resistor
  name: R1
  nodes: [out, 0]
  value: 10.0          # alias: resistance
```

| Key | Default | Required | Notes |
|---|---|---|---|
| `value` / `resistance` | — | yes | Ohms (Ω). |

C++ struct: `Resistor::Params { resistance = 1000 }`.

### `capacitor`

| Alias | Nodes | Linear | Dynamic |
|---|---|---|---|
| `c` | `[n1, n2]` | yes | yes (trapezoidal companion) |

```yaml
- type: capacitor
  name: Cout
  nodes: [out, 0]
  value: 47u           # alias: capacitance
  ic: 0.0              # initial V(n1) − V(n2), volts
```

| Key | Default | Notes |
|---|---|---|
| `value` / `capacitance` | — | Farads (F), required. |
| `ic` | `0.0` | Initial capacitor voltage at t = tstart (V). Only honored when `simulation.uic: true`. |

Companion conductance is `G_eq = 2C/Δt` (trapezoidal) or `C/Δt` (BDF1).

### `inductor`

| Alias | Nodes | Linear | Dynamic |
|---|---|---|---|
| `l` | `[n1, n2]` | yes | yes |

```yaml
- type: inductor
  name: L1
  nodes: [sw, out]
  value: 220u           # alias: inductance
  ic: 0.0               # initial current, A
```

| Key | Default | Notes |
|---|---|---|
| `value` / `inductance` | — | Henries (H), required. Must be > 0. |
| `ic` | `0.0` | Initial branch current from `n1` to `n2` (A). |

Adds a branch-current row to MNA. Companion `G_eq = Δt/(2L)`.

### `snubber_rc`

| Alias | Nodes |
|---|---|
| `snubber`, `snubberrc` | `[n1, n2]` |

```yaml
- type: snubber_rc
  name: RC_clamp
  nodes: [sw, 0]
  resistance: 4.7       # alias: value
  capacitance: 10n
  ic: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `resistance` / `value` | — | Ω, required. |
| `capacitance` | — | F, required. |
| `ic` | `0.0` | Initial capacitor voltage (V). |

**Topology note:** Despite the “series RC” naming, the runtime stamps R and C
in *parallel* between `n1` and `n2`, named `<name>__R` and `<name>__C`. Use
two discrete `resistor` + `capacitor` if you need a true series RC.

### `transformer`

| Alias | Nodes |
|---|---|
| `t` | `[p1, p2, s1, s2]` (primary +/−, secondary +/−) |

```yaml
- type: transformer
  name: TX
  nodes: [pa, pb, sa, sb]
  turns_ratio: 2.0     # alias: ratio
```

| Key | Default | Notes |
|---|---|---|
| `turns_ratio` / `ratio` | — | `N_p:N_s`. `2.0` = 2:1 step-down, `0.5` = step-up by 2. |
| `magnetizing_inductance` / `lm` | — | Whitelisted by the schema validator, **not currently wired to the device.** Use `coupled_inductor` for magnetizing dynamics. |

Ideal transformer: `V_p = n·V_s`, `I_p + n·I_s = 0`. Adds two branch rows.

### `coupled_inductor`

| Alias | Nodes |
|---|---|
| `coupledinductor` | `[p1, p2, s1, s2]` |

```yaml
- type: coupled_inductor
  name: LK
  nodes: [pa, pb, sa, sb]
  l1: 100u
  l2: 100u
  coupling: 0.98       # aliases: k, mutual_inductance
  ic1: 0.0             # primary initial current
  ic2: 0.0             # secondary initial current
```

| Key | Default | Notes |
|---|---|---|
| `l1` | `1e-3` | Primary L (H), must be > 0. |
| `l2` | `l1` | Secondary L (H). |
| `coupling` / `k` | `0.98` | Coupling coefficient `k = M/√(L1·L2)`, clamped to `\|k\| ≤ 0.999`. |
| `mutual_inductance` | derived | Alternative to `k`: gives M directly; the parser back-solves k. |
| `ic1` / `ic_primary` | `0.0` | Primary initial current (A). |
| `ic2` / `ic_secondary` | `0.0` | Secondary initial current (A). |

Stamps two underlying inductors `<name>__L1`, `<name>__L2` plus the mutual
term `M·dI/dt`. SPICE-style.

### `saturable_inductor`

| Alias | Nodes |
|---|---|
| `saturableinductor`, `sat_inductor` | `[n1, n2]` |

```yaml
- type: saturable_inductor
  name: L_core
  nodes: [in, out]
  inductance: 500u           # unsaturated L₀
  saturation_current: 8.0
  saturation_inductance: 50u
  saturation_exponent: 2.5
  ic: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `inductance` / `value` | — | Small-signal L₀ (H), required, > 0. |
| `saturation_current` | `1.0` | Knee current (A), > 0. |
| `saturation_inductance` | `0.2 · L₀` | Asymptotic L at deep saturation (H). Must satisfy `0 < L_sat ≤ L₀`. |
| `saturation_exponent` | `2.0` | Softness of the L(I) transition. Clamped to `[1, 6]` by the regularizer. |
| `ic` | `0.0` | Initial inductor current (A). |

Effective inductance:
`L_eff(I) = L_sat + (L₀ − L_sat) / (1 + (|I|/I_sat)^exponent)`.

Stamps a regular `inductor` then registers a virtual block that perturbs
its effective L every Newton iteration through the
`effective_inductance_for` callback.

---

## 2. Sources

### `voltage_source`

| Alias | Nodes |
|---|---|
| `v`, `voltagesource`, `vsource`, `source_v` | `[npos, nneg]` |

`voltage_source` is a dispatcher — the `waveform.type` key picks the
backing source object. Four waveform families are supported.

#### DC (default if no waveform block)

```yaml
- type: voltage_source
  name: Vin
  nodes: [vin, 0]
  value: 12.0
```

| Key | Default | Notes |
|---|---|---|
| `value` | `0.0` | DC level (V). |

#### `waveform.type: pwm`

```yaml
- type: voltage_source
  name: Vgate
  nodes: [g, 0]
  waveform:
    type: pwm
    v_high: 5.0
    v_low: 0.0
    frequency: 100e3
    duty: 0.45
    phase: 0.0
    dead_time: 100n
```

| Key | Default | Notes |
|---|---|---|
| `v_high` | `1.0` | High level (V). |
| `v_low` | `0.0` | Low level (V). |
| `frequency` | `10e3` | Carrier frequency (Hz). |
| `duty` | `0.5` | Duty cycle, clamped to `[0, 1]`. |
| `phase` | `0.0` | Initial phase (rad). |
| `dead_time` | `0.0` | Dead time subtracted from on-time (s). |
| `rise_time` / `fall_time` | n/a | Defined in `PWMParams` but not forwarded by the YAML parser. |

#### `waveform.type: sine`

```yaml
- type: voltage_source
  name: Vgrid
  nodes: [grid, 0]
  waveform:
    type: sine
    amplitude: 100.0
    frequency: 60.0
    offset: 0.0
    phase: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `amplitude` | `1.0` | Peak amplitude (V). |
| `frequency` | `50.0` | Hz. |
| `offset` | `0.0` | DC offset (V). |
| `phase` | `0.0` | Initial phase (rad). |

Output: `v(t) = offset + amplitude·sin(2π·frequency·t + phase)`.

#### `waveform.type: pulse`

```yaml
- type: voltage_source
  name: Vstep
  nodes: [in, 0]
  waveform:
    type: pulse
    v_initial: 0.0
    v_pulse: 5.0
    t_delay: 1u
    t_rise: 10n
    t_fall: 10n
    t_width: 5u
    period: 10u           # 0 = single shot
```

| Key | Default | Notes |
|---|---|---|
| `v_initial` | `0.0` | Idle level (V). |
| `v_pulse` | `1.0` | Pulse-high level (V). |
| `t_delay` | `0.0` | Delay before first edge (s). |
| `t_rise` | `1e-9` | Linear rise time (s). |
| `t_fall` | `1e-9` | Linear fall time (s). |
| `t_width` | `1e-6` | Pulse-high duration between rise/fall (s). |
| `period` | `0.0` | `0` = single pulse; `>0` = periodic. |

#### Driving a switch from a pulse / PWM source

If you set `target_component:` (or `target_device:` / `target:`) on a
pulse or PWM voltage source, the parser additionally calls
`Circuit::bind_switch_driver(<source>, <target>)` so the high level of
the waveform forces the named switch closed:

```yaml
- type: voltage_source
  name: Vgate
  nodes: [g, 0]
  target_component: SW1       # closes/opens SW1 from this waveform
  waveform: { type: pwm, frequency: 100e3, duty: 0.45 }
```

#### Waveforms not yet exposed in YAML

`PWL`, `EXP`, `SFFM`, `RAMP`, `TRIANGLE` are present in the C++ tree but
not wired into the YAML dispatcher today. Use the Python API (e.g.
`Circuit.add_pwm_voltage_source(...)`) or a `transfer_function` /
`lookup_table` virtual block to synthesise them.

### `current_source`

| Alias | Nodes |
|---|---|
| `i`, `currentsource`, `isource` | `[npos, nneg]` |

```yaml
- type: current_source
  name: Iload
  nodes: [out, 0]
  value: 1.5
```

| Key | Default | Notes |
|---|---|---|
| `value` | — | DC current (A), flows from `npos` to `nneg`. |

**DC only** — `waveform:` is not parsed for current sources.

---

## 3. Switching devices

### `diode`

| Alias | Nodes |
|---|---|
| `d` | `[anode, cathode]` |

```yaml
- type: diode
  name: D1
  nodes: [sw, vo]
  g_on: 1e3            # alias: ron (Pulsim auto-inverts if you pass ron)
  g_off: 1e-9          # alias: roff
```

| Key | Default | Notes |
|---|---|---|
| `g_on` / `ron` | `1e3` | On conductance (S). Pass `ron` and Pulsim takes `1/ron`. |
| `g_off` / `roff` | `1e-9` | Leakage conductance (S). |

Internal-only knobs (Python/C++ API): `v_threshold = 0.0 V` and
`v_smooth = 0.1 V` (the tanh width in Behavioral mode). Adjust through
`IdealDiode::set_smoothing`. Commute hysteresis (`event_hysteresis_`) is
`1e-9 V` and is not exposed.

### `switch`

| Alias | Nodes |
|---|---|
| `s` | `[n1, n2]` |

Externally commanded switch (does not auto-commute). Drive it from a
PWM/pulse source via `bind_switch_driver` or from
`circuit.set_switch_state(name, closed)` in Python.

```yaml
- type: switch
  name: SW1
  nodes: [in, out]
  g_on: 1e6            # alias: ron
  g_off: 1e-12         # alias: roff
  initial_state: false
```

| Key | Default | Notes |
|---|---|---|
| `g_on` / `ron` | `1e6` | On conductance (S). |
| `g_off` / `roff` | `1e-12` | Off conductance (S). |
| `initial_state` | `false` | `true` = closed at t = 0. Accepts `"closed"`, `"on"`. |

### `vcswitch`

Voltage-controlled switch. tanh-smoothed in Behavioral mode, PWL in Ideal.

| Alias | Nodes |
|---|---|
| `voltagecontrolledswitch` | `[ctrl, t1, t2]` (control node first) |

```yaml
- type: vcswitch
  name: SW
  nodes: [g, sw, 0]
  v_threshold: 2.5
  g_on: 1e3
  g_off: 1e-9
```

| Key | Default | Notes |
|---|---|---|
| `v_threshold` | `2.5` | Threshold on `V(ctrl) − 0` (V). |
| `g_on` | `1e3` | (S) |
| `g_off` | `1e-9` | (S) |
| `hysteresis` | n/a | **YAML-only quirk**: whitelisted but not forwarded. The device's internal tanh width stays at `0.5 V`. Build via the Python `Params` to override. |

### `mosfet` (incl. `nmos`, `pmos`)

| Alias | Nodes |
|---|---|
| `m`, `nmos` | `[gate, drain, source]` |
| `pmos` | `[gate, drain, source]` (sets `is_nmos = false`) |

```yaml
- type: mosfet
  name: M1
  nodes: [g, d, s]
  vth: 2.0
  kp: 0.1
  lambda: 0.01         # alias: lambda_
  g_off: 1e-12
  is_nmos: true        # auto-set to false when type = pmos
  loss:                # optional: switching-loss bookkeeping
    eon: 50e-6
    eoff: 30e-6
  thermal:             # optional: electrothermal port
    enabled: true
    rth: 1.0
    cth: 0.01
    temp_init: 25.0
    temp_ref: 25.0
    alpha: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `vth` | `2.0` | Threshold voltage (V). |
| `kp` | `0.1` | Transconductance `Kp·(W/L)` (A/V²). |
| `lambda` / `lambda_` | `0.01` | Channel-length modulation (1/V). |
| `g_off` | `1e-12` | Cutoff leakage (S). |
| `g_on` | `1e3` | Ideal-mode on conductance (S). Not exposed via YAML — set through the Python `MOSFET::Params`. |
| `is_nmos` | `true` | Set to `false` for PMOS or use the `pmos` alias. |

Behavioral path uses a smooth Shichman-Hodges blend (sigmoid `κ = 50/V`)
so Newton has gradient through the threshold. Ideal path is pure
two-state. See [Device Models](device-models.md) for the equations.

### `igbt`

| Alias | Nodes |
|---|---|
| `q` | `[gate, collector, emitter]` |

```yaml
- type: igbt
  name: Q1
  nodes: [g, c, e]
  vth: 5.0
  g_on: 1e4
  g_off: 1e-12
  v_ce_sat: 1.5
  loss: { eon: 200e-6, eoff: 150e-6, err: 80e-6 }
  thermal: { enabled: true, rth: 0.5, cth: 0.05, temp_init: 25.0 }
```

| Key | Default | Notes |
|---|---|---|
| `vth` | `5.0` | Gate threshold (V). |
| `g_on` | `1e4` | (S) |
| `g_off` | `1e-12` | (S) |
| `v_ce_sat` | `1.5` | C-E saturation voltage (V). Defined in `Params` and parsed; current stamps use `g_on`·`vce` only — `v_ce_sat` lives in the loss model. |

### `bjt_npn` / `bjt_pnp` ⚠

| Alias | Nodes |
|---|---|
| `bjtnpn`, `bjt-npn` / `bjtpnp`, `bjt-pnp` | `[base, collector, emitter]` |

⚠ **Surrogate**: stamps a `mosfet` with `is_nmos = (type == "bjt_npn")`.
Emits `PULSIM_YAML_W_COMPONENT_SURROGATE` at load time. This is a
first-parity slice, not a calibrated BJT model.

| Key | Default | Notes |
|---|---|---|
| `vbe_on` | `2.0` | Mapped to MOSFET `vth`. |
| `beta` | `100` | Validated `> 0`; maps to MOSFET `kp = max(1e-6, beta·1e-3)`. |
| `g_off` | `1e-12` | Off conductance (S). |

Supports `thermal:` block.

### `thyristor` (SCR) / `triac` ⚠

| Alias | Nodes |
|---|---|
| `scr` (for thyristor) | `[gate, anode, cathode]` |
| `triac` | `[gate, mt1, mt2]` |

⚠ Stamps a 2-terminal `IdealSwitch` between pins 1/2 and registers a
virtual block carrying the gate-trigger logic.

| Key | Default | Notes |
|---|---|---|
| `gate_threshold` | `1.0` | (V) |
| `holding_current` | `0.05` | (A) — released when forward current drops below this. |
| `latch_current` | `1.2 · holding_current` | (A) — minimum forward current to latch on. |
| `g_on` | `1e4` | (S) |
| `g_off` | `1e-9` | (S) |
| `initial_state` | `false` | `false`/`"open"` = blocking. |

Thyristor: forward-only. Triac: bidirectional (gate active when
`|V_gate| ≥ gate_threshold`).

Output channels (in `result.channel_values`):
`<name>.trigger`, `<name>.i_est`, `<name>.state`.

---

## 4. Protection devices ⚠

All three are surrogates over `IdealSwitch` + a virtual event controller.

### `fuse` ⚠

| Alias | Nodes |
|---|---|
| `fuse` | `[n1, n2]` |

```yaml
- type: fuse
  name: F1
  nodes: [src, load]
  rating: 5.0
  blow_i2t: 25e-3
  initial_state: closed   # or "blown" / false
```

| Key | Default | Notes |
|---|---|---|
| `rating` | `1.0` | Nominal current rating (A). Documentation only. |
| `blow_i2t` / `i2t` | `rating² · 1e-3` | I²·t threshold to blow (A²·s). |
| `g_on` | `1e4` | (S) |
| `g_off` | `1e-9` | (S) |
| `initial_state` | `true`/`closed` | Accepts `"open"`, `"blown"`, `"tripped"`. |

Output channels: `<name>.i2t` (cumulative stress), `<name>.state` (1/0).

### `circuit_breaker` ⚠

| Alias | Nodes |
|---|---|
| `breaker`, `circuitbreaker` | `[n1, n2]` |

```yaml
- type: circuit_breaker
  name: CB1
  nodes: [bus, load]
  trip_current: 30.0
  trip_time: 50e-3
  initial_state: closed
```

| Key | Default | Notes |
|---|---|---|
| `trip_current` / `trip` | `1.0` | Overcurrent threshold (A). |
| `trip_time` | `0.0` | Inverse-time delay (s). `0` = instantaneous. |
| `g_on` | `1e4` | (S) |
| `g_off` | `1e-9` | (S) |
| `initial_state` | `true` | Accepts boolean or `"open"` / `"tripped"`. |

Output channels: `<name>.trip_timer`, `<name>.state`.

### `relay` ⚠

| Alias | Nodes |
|---|---|
| `relay` | `[coil+, coil−, com, NO, NC]` |

```yaml
- type: relay
  name: K1
  nodes: [cp, cn, com, no, nc]
  pickup_voltage: 5.0
  dropout_voltage: 1.0
  initial_state: false
```

| Key | Default | Notes |
|---|---|---|
| `pickup_current` / `pickup_voltage` | `1.0` | Coil energize threshold. Whichever you specify takes precedence. |
| `dropout_current` / `dropout_voltage` | `0.8 · pickup` | Release threshold. |
| `g_on` / `contact_resistance` / `ron` | `1e4` | (S) |
| `g_off` / `off_resistance` / `roff` | `1e-9` | (S) |
| `initial_state` | `false` | Accepts `"closed"`, `"energized"`, `"on"`. |

Stamps two switches `<name>__no` (between `com` and `NO`) and
`<name>__nc` (between `com` and `NC`) with opposite initial states.

Output channels: `<name>.state` (coil energized), `<name>.no_state`,
`<name>.nc_state`.

---

## 5. Probes & scopes (no stamp)

Probes don't contribute to the MNA — they're observation-only blocks
that emit channels in the trace CSV alongside electrical state.

### `voltage_probe`

```yaml
- type: voltage_probe
  name: Vout_meas
  nodes: [out, 0]
```

Outputs `<name>` = `V(out) − V(0)`.

### `current_probe`

```yaml
- type: current_probe
  name: I_L1
  nodes: [a, b]                # nominal — actual current comes from target
  target_component: L1
```

Reads the branch current of the named device (must be a branch-carrying
element: voltage source, inductor, transformer secondary).

### `power_probe`

```yaml
- type: power_probe
  name: P_load
  nodes: [out, 0]
  target_component: Rload
```

Outputs `<name>` = `V(out) − V(0)` × `branch_current(target)`.

### `electrical_scope`

| Alias | Nodes |
|---|---|
| `scope` | 1 to N |

```yaml
- type: electrical_scope
  name: probe_bus
  nodes: [v_in, v_out, v_sw]
```

Outputs the **mean** of `V(nodes[i])` across all listed pins. Useful as a
multi-pin DC voltmeter or sanity-check tap.

### `thermal_scope`

Same as `electrical_scope` but tagged as `thermal` in the channel
metadata — for reading thermal sub-network nodes when you run with the
electrothermal port.

---

## 6. C++ `Params` structs (Python API)

These are the canonical defaults applied by the C++ device constructors.
When you build a circuit directly from Python (skipping YAML), pass these
structs to the `add_*` methods.

| Struct | Defaults |
|---|---|
| `Resistor::Params` | `resistance = 1000` |
| `Capacitor::Params` | `capacitance = 1e-6`, `initial_voltage = 0` |
| `Inductor::Params` | `inductance = 1e-3`, `initial_current = 0` |
| `VoltageSource::Params` | `voltage = 0` |
| `CurrentSource::Params` | `current = 0` |
| `IdealDiode::Params` | `g_on = 1e3`, `g_off = 1e-9`, `v_threshold = 0`, `v_smooth = 0.1` |
| `IdealSwitch::Params` | `g_on = 1e6`, `g_off = 1e-12`, `initial_state = false` |
| `VoltageControlledSwitch::Params` | `v_threshold = 2.5`, `g_on = 1e3`, `g_off = 1e-9`, `hysteresis = 0.1` |
| `MOSFET::Params` | `vth = 2`, `kp = 0.1`, `lambda = 0.01`, `g_off = 1e-12`, `is_nmos = true`, `g_on = 1e3` |
| `IGBT::Params` | `vth = 5`, `g_on = 1e4`, `g_off = 1e-12`, `v_ce_sat = 1.5` |
| `Transformer::Params` | `turns_ratio = 1`, `magnetizing_inductance = 1e-3` |
| `PWMParams` | `v_high=1, v_low=0, frequency=10e3, duty=0.5, phase=0, dead_time=0, rise_time=0, fall_time=0` |
| `SineParams` | `amplitude=1, offset=0, frequency=50, phase=0` |
| `PulseParams` | `v_initial=0, v_pulse=1, t_delay=0, t_rise=1e-9, t_fall=1e-9, t_width=1e-6, period=0` |
| `RampParams` *(C++ only — no YAML path)* | `v_min=0, v_max=1, frequency=10e3, phase=0, triangle=false` |

---

## 7. Stamp-domain summary

The fourteen YAML types that contribute directly to MNA (G and b):

| # | YAML type | Pins | Linear | Dynamic | PWL-capable |
|---|---|---|---|---|---|
| 1 | `resistor` | 2 | yes | no | — |
| 2 | `capacitor` | 2 | yes | yes | — |
| 3 | `inductor` | 2 | yes | yes | — |
| 4 | `voltage_source` (DC) | 2 | yes | no | — |
| 5 | `voltage_source` (pwm/sine/pulse) | 2 | yes | time-varying | — |
| 6 | `current_source` | 2 | yes | no | — |
| 7 | `diode` | 2 | no | no | yes |
| 8 | `switch` | 2 | yes (PWL) | no | yes |
| 9 | `vcswitch` | 3 | no | no | yes |
| 10 | `mosfet` (incl. `nmos`/`pmos`) | 3 | no | no | yes |
| 11 | `igbt` | 3 | no | no | yes |
| 12 | `transformer` | 4 | yes | no | — |
| 13 | `snubber_rc` | 2 | yes | yes | — |
| 14 | `bjt_npn`/`bjt_pnp` ⚠ | 3 | no | no | yes |

Virtual electrical-domain wrappers (no extra stamps; they perturb a base
device):

| YAML type | Backs |
|---|---|
| `saturable_inductor` | `inductor` + state-dependent L(I) |
| `coupled_inductor` | two `inductor`s + mutual M |
| `thyristor` / `triac` | `switch` + latching event logic |
| `fuse` / `circuit_breaker` | `switch` + trip event logic |
| `relay` | two `switch`es + coil event logic |

---

## See also

- [Control Blocks Reference](control-blocks-reference.md) — every virtual
  control block (PWM, PI, Clarke, Park, PLL, SVM, …).
- [Netlist YAML Format](netlist-format.md) — schema for the top-level
  `simulation:` block and the `components:` list.
- [Device Models](device-models.md) — equations behind MOSFET / IGBT /
  diode and how switching modes pick between PWL and Behavioral paths.
- [Catalog Devices](catalog-devices.md) — vendor MOSFET / IGBT presets
  callable from Python.
- [Magnetic Models](magnetic-models.md) — saturating-inductor BH curve,
  transformer details, Steinmetz core-loss.
- [Electrothermal Workflow](electrothermal-workflow.md) — how the
  `thermal:` block on MOSFET/IGBT works end-to-end.
- [GUI Backend Parity](gui-component-parity.md) — table mapping every
  PulsimGui block to its YAML counterpart.
