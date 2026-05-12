# Control Blocks Reference

> **Status:** authoritative — every virtual control block dispatched
> inside `RuntimeCircuit::execute_mixed_domain_step`. Source:
> `core/include/pulsim/v1/runtime_circuit.hpp`. Aliases / arity from
> `core/src/v1/yaml_parser.cpp`. Updated through Phase 28.

Control blocks are the non-stamping side of the runtime. They produce
**named channels** that drive switches (`pwm_generator`), feed back into
control loops (`pi_controller`), shape signals (`gain`, `lookup_table`),
synchronize to grids (`pll`), or build full vector-control pipelines
(`clarke_transform` → `park_transform` → `inverse_park_transform` →
`svm`). They live alongside electrical components in the same
`components:` list but contribute zero rows to the MNA system.

## How control blocks talk to each other

Every block publishes its outputs as channels indexed by `<name>.<key>`
in `result.channel_values` and `virtual_signal_state_`. Downstream blocks
read those channels through metadata pointers:

```yaml
- type: pwm_generator
  name: PWM
  nodes: [trigger]
  duty_from_channel: PI.output   # ← reads PI controller's channel
  target_component: SW1
```

The most common cross-block metadata keys:

| Key | Used by | Reads |
|---|---|---|
| `target_component` | `pwm_generator`, `relay`, `fuse`, `circuit_breaker`, `thyristor`, `triac`, `current_probe`, `power_probe`, `saturable_inductor`, `coupled_inductor` | the named electrical device |
| `duty_from_channel` | `pwm_generator` | a channel that drives duty cycle |
| `theta_from_channel` | `park_transform`, `inverse_park_transform` | rotation angle θ (rad) |
| `alpha_from_channel`, `beta_from_channel` | `park_transform`, `svm` | stationary-frame components |
| `d_from_channel`, `q_from_channel` | `inverse_park_transform` | synchronous-frame components |

Every block also writes its **primary scalar output** to
`virtual_signal_state_[<name>]`, so simpler blocks (gain, sum) can be
referenced by name only (no suffix needed).

Channels are emitted in the trace CSV under the column prefix `chan:`:
`chan:PI.output`, `chan:CLK.alpha`, `chan:PLL.theta`, etc.

## Quick index

| Category | Blocks |
|---|---|
| Arithmetic | `gain`, `sum`, `subtraction`, `math_block`, `integrator`, `differentiator` |
| Controllers | `pi_controller`, `pid_controller`, `limiter`, `rate_limiter` |
| Comparators / latches | `comparator`, `hysteresis`, `state_machine` |
| Data shaping | `lookup_table`, `transfer_function`, `delay_block`, `sample_hold`, `signal_mux`, `signal_demux` |
| Modulation | `pwm_generator` |
| Three-phase | `clarke_transform`, `inverse_clarke_transform`, `park_transform`, `inverse_park_transform`, `pll`, `svm` |
| Analog | `op_amp` |

The **output clamp protocol** is shared across most arithmetic and
controller blocks: optional keys `output_min`/`output_max`, falling back
to `min`/`max`, then to `rail_low`/`rail_high` (defaults ±1e12). The
`op_amp` block forces its rails to ±15 V by default. Whenever a block
honors this protocol it's labelled **clamp keys** below.

---

## 1. Arithmetic

### `gain`

| Alias | Nodes |
|---|---|
| `gainblock`, `gain-block` | 2 (`input`, `reference`) |

```yaml
- type: gain
  name: G_iout
  nodes: [iout_sense, 0]
  gain: 5.0
  offset: 0.0
  output_min: -10.0
  output_max:  10.0
```

| Key | Default | Notes |
|---|---|---|
| `gain` | `1.0` | Multiplicative. |
| `offset` | `0.0` | Additive (post-gain). |
| Clamp keys | — | Optional. |

Output: `<name>` = `gain · (V(input) − V(reference)) + offset`, clamped.

### `sum`

| Alias | Nodes |
|---|---|
| `adder`, `sumblock` | 2 (`a`, `b`) |

```yaml
- type: sum
  name: ADD1
  nodes: [v_ref, v_fb]
  gain: 1.0
  offset: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `gain` | `1.0` | Multiplicative on the sum. |
| `offset` | `0.0` | Additive. |
| Clamp keys | — | Optional. |

Output: `<name>` = `gain · (V(a) + V(b)) + offset`.

### `subtraction`

| Alias | Nodes |
|---|---|
| `subtract`, `subtractor`, `sub` | 2 (`minuend`, `subtrahend`) |

```yaml
- type: subtraction
  name: ERR
  nodes: [v_ref, v_fb]
```

| Key | Default | Notes |
|---|---|---|
| `gain` | `1.0` | |
| `offset` | `0.0` | |
| Clamp keys | — | |

Output: `<name>` = `gain · (V(minuend) − V(subtrahend)) + offset`.

### `math_block`

| Alias | Nodes |
|---|---|
| `mathblock`, `math` | 2 to N (only `in0` / `in1` are consumed today) |

```yaml
- type: math_block
  name: M1
  nodes: [a, b]
  operation: mul         # add (default) | sub | mul | div
```

| Key | Default | Notes |
|---|---|---|
| `operation` | `add` | One of `add`, `sub`, `mul`, `div`. For `div`, output = 0 when `|in1| < 1e-12`. |

### `integrator`

| Alias | Nodes |
|---|---|
| `integrator` | 2 |

```yaml
- type: integrator
  name: INT1
  nodes: [error, 0]
  output_min: -100.0
  output_max:  100.0
```

| Key | Default | Notes |
|---|---|---|
| Clamp keys | — | Anti-windup is implicit when limits are set. |

Forward-Euler integrator: state `S += V(input) · dt`, output = `clamp(S)`.

### `differentiator`

| Alias | Nodes |
|---|---|
| `differentiator` | 2 |

```yaml
- type: differentiator
  name: D1
  nodes: [signal, 0]
  alpha: 0.3
```

| Key | Default | Notes |
|---|---|---|
| `alpha` | `0.0` | IIR smoothing on the derivative, in `[0, 1]`. 0 = no smoothing. |
| Clamp keys | — | |

Backward-difference: `raw = (signal − prev) / dt`; if `alpha > 0`:
`output = alpha · prev_output + (1 − alpha) · raw`.

---

## 2. Controllers

### `pi_controller`

| Alias | Nodes |
|---|---|
| `pi`, `picontroller`, `pi-controller` | 3 (`in_pos`, `in_neg`, `output_ref`) |

```yaml
- type: pi_controller
  name: PI_iout
  nodes: [iout_ref, iout_meas, 0]
  kp: 2.5
  ki: 800.0
  anti_windup: 1.0
  output_min: 0.0
  output_max: 1.0
```

| Key | Default | Notes |
|---|---|---|
| `kp` | `gain` fallback or `1.0` | Proportional gain. |
| `ki` | `0.0` | Integral gain. |
| `anti_windup` | `1.0` | Treated as boolean (> 0.5 = enabled). Back-calculation. |
| Clamp keys | — | Required for anti-windup to do anything. |

When the unsaturated output exceeds the limit, the integral state is
back-calculated to `(limited − kp·signal) / ki`.

### `pid_controller`

| Alias | Nodes |
|---|---|
| `pidcontroller`, `pid-controller` | 3 |

```yaml
- type: pid_controller
  name: PID
  nodes: [ref, fb, 0]
  kp: 1.5
  ki: 200.0
  kd: 0.01
  output_min: 0.0
  output_max: 1.0
```

| Key | Default | Notes |
|---|---|---|
| `kp` | `gain` or `1.0` | |
| `ki` | `0.0` | |
| `kd` | `0.0` | |
| `anti_windup` | `1.0` | |
| Clamp keys | — | |

Derivative is backward-difference on the **error** signal (not on the
measurement). Same anti-windup as `pi_controller`.

### `limiter`

| Alias | Nodes |
|---|---|
| `limiter` | 2 |

```yaml
- type: limiter
  name: LIM
  nodes: [signal, 0]
  output_min: 0.0
  output_max: 1.0
```

| Key | Default | Notes |
|---|---|---|
| `gain` | `1.0` | Pre-clamp scaler. |
| Clamp keys | — | The actual function. Without them, it's pass-through. |

Output: `clamp(gain · signal, lo, hi)`.

### `rate_limiter`

| Alias | Nodes |
|---|---|
| `ratelimiter`, `rate-limiter` | 2 |

```yaml
- type: rate_limiter
  name: SLEW
  nodes: [signal, 0]
  rising_rate:  1e3        # units per second
  falling_rate: 1e3
```

| Key | Default | Notes |
|---|---|---|
| `rising_rate` | `1e6` | Max d/dt going up. |
| `falling_rate` | `rising_rate` | Max d/dt going down. |
| Clamp keys | — | Applied after rate-limit. |

Output is clamped so `output − previous ∈ [−falling·dt, +rising·dt]`.

---

## 3. Comparators, hysteresis, latches

### `comparator` / `hysteresis`

| Alias | Nodes |
|---|---|
| `comparator` | 3 (`+`, `−`, `output_ref`) |
| `hysteresis` | 2 |

```yaml
- type: hysteresis
  name: ZC
  nodes: [signal, 0]
  threshold: 0.0
  hysteresis: 0.1
  high: 1.0
  low: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `threshold` | `0.0` | |
| `hysteresis` | `0.0` | Total band width (so half-band each side). |
| `high` | `1.0` | Output when on. |
| `low` | `0.0` | Output when off. |

Schmitt-trigger semantics: ON when `signal > threshold + ½·band`,
OFF when `signal < threshold − ½·band`.

### `state_machine`

| Alias | Nodes |
|---|---|
| `statemachine`, `state-machine` | 1 to N |

```yaml
- type: state_machine
  name: LATCH
  nodes: [set_in, reset_in]
  mode: set_reset
  threshold: 0.5
  high: 1.0
  low: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `mode` | `toggle` | One of `toggle`, `level`, `set_reset` / `sr`. |
| `threshold` | `0.5` | Trigger level. |
| `high` / `low` | `1.0` / `0.0` | Output levels. |

Modes:
- `toggle` — flips state on each rising edge above `threshold`.
- `level` — `state = (signal > threshold)` every step.
- `set_reset` — `nodes[0]` is SET, `nodes[1]` is RESET. Reset dominates.

---

## 4. Data shaping

### `lookup_table`

| Alias | Nodes |
|---|---|
| `lookuptable`, `lookup-table`, `lut` | 2 |

```yaml
- type: lookup_table
  name: D_vs_Vin
  nodes: [vin, 0]
  mode: linear            # linear (default) | hold | step | nearest
  x: [10, 20, 30, 40]
  y: [0.9, 0.8, 0.7, 0.6]
```

Alternative formats accepted:

```yaml
table:                    # list of [x, y] pairs (or {x, y} maps)
  - [10, 0.9]
  - [20, 0.8]
mapping:                  # map of x: y
  10: 0.9
  20: 0.8
```

| Key | Default | Notes |
|---|---|---|
| `x`, `y` | — | Parallel arrays of independent / dependent values. |
| `table` / `mapping` | — | Alternatives to `x`/`y`. |
| `mode` | `linear` | `linear`, `hold`/`step` (zero-order), `nearest`. |
| Clamp keys | — | |

Out-of-range inputs are clamped to the table endpoints. Samples are
sorted/de-duplicated on load.

### `transfer_function`

| Alias | Nodes |
|---|---|
| `transferfunction`, `transfer-function` | 2 |

```yaml
- type: transfer_function
  name: HPF
  nodes: [signal, 0]
  num: [1.0, -1.0]        # b0 + b1·z⁻¹ + …
  den: [1.0, -0.9]        # a0 + a1·z⁻¹ + …
```

| Key | Default | Notes |
|---|---|---|
| `num` | — | Numerator coefficients (MSB first). |
| `den` | — | Denominator coefficients; `\|den[0]\| ≥ 1e-15` required. |
| `alpha` | `0.2` | Fallback first-order IIR coefficient (only used if num/den missing/invalid). |
| Clamp keys | — | |

Direct-form II discrete filter. If the parsing fails, falls back to a
single-pole IIR `y[n] = y[n-1] + α · (signal − y[n-1])`.

### `delay_block`

| Alias | Nodes |
|---|---|
| `delayblock`, `delay` | 2 |

```yaml
- type: delay_block
  name: DLY
  nodes: [signal, 0]
  delay: 1e-3
```

| Key | Default | Notes |
|---|---|---|
| `delay` | `0.0` | Transport delay (s). `0` = pass-through. |

Linear-interpolated between bracketing history samples.

### `sample_hold`

| Alias | Nodes |
|---|---|
| `samplehold`, `sample-and-hold` | 2 |

```yaml
- type: sample_hold
  name: ZOH
  nodes: [signal, 0]
  sample_period: 100e-6
```

| Key | Default | Notes |
|---|---|---|
| `sample_period` | `0.0` | Sample interval (s). `≤ 0` means sample every step. |

Zero-order hold: captures `V(signal)` when `(t − last_hold_time) ≥ T_s`.

### `signal_mux`

| Alias | Nodes |
|---|---|
| `signalmux`, `mux` | 2 to N |

```yaml
- type: signal_mux
  name: SEL
  nodes: [a, b, c, d]
  select_index: 2          # picks node `c`
```

| Key | Default | Notes |
|---|---|---|
| `select_index` | `0` | Integer (rounded), clamped into `[0, len(nodes))`. |

### `signal_demux`

| Alias | Nodes |
|---|---|
| `signaldemux`, `demux` | 2 to N |

Today this acts as a **pass-through of `in0`** — the multi-output fan-out
semantics are not yet implemented. Use multiple `gain` taps if you need
real fan-out.

---

## 5. Modulation

### `pwm_generator`

| Alias | Nodes |
|---|---|
| `pwmgenerator`, `pwm` | 1 to 3 (`signal_in` optional, ignored unless `duty_from_input: 1`) |

```yaml
- type: pwm_generator
  name: PWM
  nodes: [trigger]
  frequency: 100e3
  duty_from_channel: PI.output    # channel-driven duty
  duty_min: 0.05
  duty_max: 0.95
  target_component: SW1            # forces SW1 from the PWM signal
```

| Key | Default | Notes |
|---|---|---|
| `frequency` | `1e3` | Carrier frequency (Hz), ≥ 1. |
| `duty` | `0.5` | Static duty cycle. |
| `duty_min` | `0.0` | Lower clamp. |
| `duty_max` | `1.0` | Upper clamp. |
| `duty_from_input` | `0.0` | Boolean (> 0.5). When true, `V(signal_in)` drives duty after the affine map below. |
| `duty_offset` | `0.0` | Affine offset on input-driven duty. |
| `duty_gain` | `1.0` | Affine gain on input-driven duty. |
| `duty_from_channel` (metadata) | — | Name of a channel whose value drives duty. Overrides `duty_from_input`. |
| `target_component` (metadata) | — | Name of a switch (`switch`, `mosfet`, `igbt`, `vcswitch`) to drive. |

Symmetric triangular carrier. Output `<name>` is binary 1/0, plus extra
channels `<name>.duty` (commanded duty, post-clamp) and `<name>.carrier`
(the carrier signal in `[0, 1]`).

---

## 6. Three-phase / vector control (Phase 28)

### `clarke_transform`

| Alias | Nodes |
|---|---|
| `clarke`, `abc_to_alpha_beta` | 3 (`a`, `b`, `c`) |

```yaml
- type: clarke_transform
  name: CLK
  nodes: [a, b, c]
```

Amplitude-invariant Clarke:

| Output channel | Expression |
|---|---|
| `<name>.alpha` | `(2/3)·(vₐ − ½·vᵦ − ½·v_c)` |
| `<name>.beta` | `(vᵦ − v_c) / √3` |
| `<name>.gamma` | `(vₐ + vᵦ + v_c) / 3` |
| `<name>` (primary) | `<name>.alpha` |

### `inverse_clarke_transform`

| Alias | Nodes |
|---|---|
| `inverse_clarke`, `alpha_beta_to_abc` | 3 (`α`, `β`, `γ`) |

| Output channel | Expression |
|---|---|
| `<name>.a` | `α + γ` |
| `<name>.b` | `−½·α + (√3/2)·β + γ` |
| `<name>.c` | `−½·α − (√3/2)·β + γ` |

### `park_transform`

| Alias | Nodes |
|---|---|
| `park`, `alpha_beta_to_dq` | 2 (`α`, `β`) or 3 (`α`, `β`, `γ`) |

```yaml
- type: park_transform
  name: PARK
  nodes: [a, b]                       # or chain through metadata:
  alpha_from_channel: CLK.alpha
  beta_from_channel:  CLK.beta
  theta_from_channel: PLL.theta
```

| Metadata | Default | Notes |
|---|---|---|
| `alpha_from_channel` | nodes[0] | Channel to read α from (overrides positional node). |
| `beta_from_channel` | nodes[1] | Channel to read β from. |
| `theta_from_channel` | `0.0` | Channel for rotation angle θ (rad). |

Park rotation:

| Output channel | Expression |
|---|---|
| `<name>.d` | `cos(θ)·α + sin(θ)·β` |
| `<name>.q` | `−sin(θ)·α + cos(θ)·β` |
| `<name>.zero` | `γ` (pass-through) |

### `inverse_park_transform`

| Alias | Nodes |
|---|---|
| `inverse_park`, `dq_to_alpha_beta` | 2 (`d`, `q`) or 3 (`d`, `q`, `zero`) |

```yaml
- type: inverse_park_transform
  name: IPARK
  nodes: [a, b]                       # placeholder pins
  d_from_channel: PI_d.output
  q_from_channel: PI_q.output
  theta_from_channel: PLL.theta
```

| Metadata | Default | Notes |
|---|---|---|
| `d_from_channel`, `q_from_channel` | nodes[0], nodes[1] | Channels for d / q. |
| `theta_from_channel` | `0.0` | Rotation angle (rad). |

Output:

| Channel | Expression |
|---|---|
| `<name>.alpha` | `cos(θ)·d − sin(θ)·q` |
| `<name>.beta` | `sin(θ)·d + cos(θ)·q` |
| `<name>.gamma` | `zero` (pass-through) |

### `pll`

| Alias | Nodes |
|---|---|
| `phase_locked_loop` | 1 (single-phase input) |

```yaml
- type: pll
  name: PLL
  nodes: [grid]
  kp: 200.0
  ki: 2000.0
  f_nominal_hz: 60.0
```

| Key | Default | Notes |
|---|---|---|
| `kp` | `100.0` | PI proportional gain. |
| `ki` | `1000.0` | PI integral gain. |
| `f_nominal_hz` | `60.0` | Nominal frequency (Hz). |

Single-phase PLL using a q-axis projection error:

```
v_q     = −sin(θ̂) · V(grid)
ω       = ω_nom + k_p · v_q + k_i · ∫v_q dt
dθ̂/dt   = ω,    θ̂ wrapped to [0, 2π)
```

Channels: `<name>.theta`, `<name>.omega`, `<name>.lock_error` (the
instantaneous q-axis error). Locks at a quadrature offset from a pure
sine input — see [Three-Phase Grid Library](three-phase-grid.md#yaml-control-blocks-phase-28)
for the convention.

### `svm`

| Alias | Nodes |
|---|---|
| `space_vector_modulation`, `svpwm` | 1 (trigger, value ignored) |

```yaml
- type: svm
  name: SVM
  nodes: [a]                           # placeholder; α/β come from channels
  v_dc: 200.0
  alpha_from_channel: IPARK.alpha
  beta_from_channel:  IPARK.beta
```

| Key | Default | Notes |
|---|---|---|
| `v_dc` | `1.0` | DC bus voltage (V) — normalizer. |
| `alpha_from_channel`, `beta_from_channel` (metadata) | — | Required channels for the stationary-frame reference. |

Min-max SVPWM with zero-sequence injection. Internal:

```
v_a  =  α
v_b  = −½·α + (√3/2)·β
v_c  = −½·α − (√3/2)·β
cm   = −½·(max + min) of (v_a, v_b, v_c)
d_x  = clamp((v_x + cm + V_dc/2) / V_dc, 0, 1)
```

Output channels: `<name>.d_a`, `<name>.d_b`, `<name>.d_c` (three
half-bridge duties in `[0, 1]`).

---

## 7. Analog elements

### `op_amp`

| Alias | Nodes |
|---|---|
| `opamp`, `op-amp` | 3 (`in+`, `in−`, `output_ref`) |

```yaml
- type: op_amp
  name: U1
  nodes: [vp, vn, 0]
  open_loop_gain: 1e5
  rail_low:  -15.0
  rail_high: +15.0
  offset: 0.0
```

| Key | Default | Notes |
|---|---|---|
| `open_loop_gain` / `gain` | `1e5` | A_OL. |
| `rail_low` | `-15.0` | Negative supply rail (V). |
| `rail_high` | `+15.0` | Positive supply rail (V). |
| `offset` | `0.0` | Input-referred offset (V). |

Output: `clamp(A · (V(in+) − V(in−)) + offset, rail_low, rail_high)`.
Rails are always enforced; `output_min/output_max` are not consulted.

---

## 8. Wiring patterns

### Closed-loop buck (PI on V_out)

```yaml
components:
  # ... electrical: Vin, L1, Cout, Rload, SW1, D1 ...
  - type: subtraction              # error = V_ref − V_out
    name: ERR
    nodes: [v_ref, vout]
  - type: pi_controller
    name: PI
    nodes: [v_ref, vout, 0]
    kp: 0.8
    ki: 2000.0
    output_min: 0.05
    output_max: 0.95
  - type: pwm_generator
    name: PWM
    nodes: [trigger]
    frequency: 100e3
    duty_from_channel: PI
    target_component: SW1
```

### Vector-control open loop

```yaml
components:
  # 3-phase sources Va, Vb, Vc at 120° apart
  - type: pll
    name: PLL
    nodes: [a]
    f_nominal_hz: 60.0
  - type: clarke_transform
    name: CLK
    nodes: [a, b, c]
  - type: park_transform
    name: PARK
    nodes: [a, b]
    alpha_from_channel: CLK.alpha
    beta_from_channel:  CLK.beta
    theta_from_channel: PLL.theta
  - type: inverse_park_transform
    name: IPARK
    nodes: [a, b]
    d_from_channel: PARK.d
    q_from_channel: PARK.q
    theta_from_channel: PLL.theta
  - type: svm
    name: SVM
    nodes: [a]
    v_dc: 200.0
    alpha_from_channel: IPARK.alpha
    beta_from_channel:  IPARK.beta
```

The trace CSV will contain `chan:CLK.alpha`, `chan:CLK.beta`,
`chan:PLL.theta`, `chan:PARK.d`, `chan:PARK.q`, `chan:IPARK.alpha`,
`chan:IPARK.beta`, `chan:SVM.d_a`, `chan:SVM.d_b`, `chan:SVM.d_c` as
first-class observable columns.

### Schmitt trigger that drives a switch

```yaml
- type: hysteresis
  name: SCHMITT
  nodes: [sense, 0]
  threshold: 5.0
  hysteresis: 0.5
  high: 1.0
  low: 0.0
- type: pwm_generator                # repurpose PWM block as level-driver
  name: GATE
  nodes: [trigger]
  frequency: 1e6                     # high enough that level "wins"
  duty_from_channel: SCHMITT
  target_component: SW1
```

---

## 9. Channels not covered here

Some virtual blocks live outside `execute_mixed_domain_step` but still
emit channels:

- **Event-driven components** (`thyristor`, `triac`, `fuse`,
  `circuit_breaker`, `relay`) — see
  [Components Reference](components-reference.md#3-switching-devices)
  for full parameter lists. They write
  `<name>.state`, `<name>.trigger`, `<name>.i_est`, `<name>.i2t`,
  `<name>.trip_timer`, `<name>.no_state`, `<name>.nc_state` etc.
- **Magnetic annotations** (`saturable_inductor`, `coupled_inductor`) —
  emit `<name>.l_eff`, `<name>.i_est`, `<name>.k`, `<name>.mutual`.
- **Probes** (`voltage_probe`, `current_probe`, `power_probe`,
  `electrical_scope`, `thermal_scope`) — emit their main scalar to
  `<name>`.

---

## See also

- [Components Reference](components-reference.md) — every electrical
  device.
- [Netlist YAML Format](netlist-format.md) — schema for the top-level
  `simulation:` block.
- [Three-Phase Grid Library](three-phase-grid.md) — recipes that combine
  the Phase 28 blocks end-to-end with grid sources and inverter stages.
- [KPI Reference](kpi-reference.md) — the metrics you compute *on top of*
  the channels these blocks produce.
- [Examples and Results](examples-and-results.md) — runnable
  worked-example scripts.
