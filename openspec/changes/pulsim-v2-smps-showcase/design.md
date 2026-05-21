# Design — `pulsim-v2-smps-showcase` (Layer 9 V0)

## The buck converter

```
                  L
   Vin ─── Q1 ───┬──╳╳╳╳───┬──── vout
                 │          │
   gnd ─── D1 ───┘     Cout ┴── R_load
                              │
                              └─── gnd
```

- `Vin`: DC source (24 V).
- `Q1`: high-side MOSFET (controlled by switch_fn).
- `D1`: low-side freewheeling diode (auto-commutates).
- `L`: filter inductor (100 µH).
- `Cout`: filter capacitor (47 µF).
- `R_load`: load (5 Ω).

In steady state with duty cycle `D` and ignoring losses:

```
V_out = V_in · D
```

For `V_in = 24 V`, `D = 0.5`: `V_out = 12 V`.

With losses (R_on of MOSFET + winding resistance of L):

```
V_out ≈ V_in · D · η
```

where `η ≈ 0.95-0.99` for a well-designed buck. For our
defaults (`R_on = 1 mΩ`, `R_load = 5 Ω`), expected
V_out ≈ 11.95 V (η ≈ 0.996).

## PWM switch_fn

The buck's MOSFET is the only controlled switch in the
circuit. Its switch_idx is 0 (first switch in
insertion order). The YAML's
`mosfet_with_body_diode` adds 2 branches: the switch
(switch_idx 0) and the body diode (auto-commutating —
NOT controlled by switch_fn). So switch_fn drives bit 0.

```cpp
constexpr Real f_sw = 100e3;     // 100 kHz
constexpr Real T_sw = 1.0 / f_sw; // 10 µs
constexpr Real duty = 0.5;

auto switch_fn = [](Real t) {
    const Real phase = std::fmod(t, T_sw) / T_sw;
    // bit 0 = MOSFET on/off; bit 1 = body diode (left to
    // auto-commutate, mask irrelevant)
    return phase < duty
        ? topology::SwitchStateMask{1}  // Q1 ON
        : topology::SwitchStateMask{0}; // Q1 OFF
};
```

The body diode auto-commutates via V2's `DiodeEventState` —
during Q1's off-time, it conducts to maintain inductor
current; during on-time, it's reverse-biased.

The free-wheeling diode `D1` likewise auto-commutates.

## Verification

The test runs the simulation for 5 ms (~500 PWM cycles).
The LC filter has time constant
`τ ≈ √(L·C) ≈ √(100µ·47µ) ≈ 68 µs`. After ~10·τ ≈ 0.7 ms,
the output is essentially in steady state. We measure mean
+ ripple over the LAST 0.5 ms.

```cpp
// Skip first 4.5 ms (let steady state settle).
const Size k_skip = static_cast<Size>(4.5e-3 / dt);
Real sum = 0;
Real vmax = -1e9, vmin = +1e9;
for (Size k = k_skip; k < result.num_steps(); ++k) {
    const Real v_out = result.states[k][vout_node];
    sum += v_out;
    vmax = std::max(vmax, v_out);
    vmin = std::min(vmin, v_out);
}
const Real v_out_mean = sum / (result.num_steps() - k_skip);
const Real v_out_ripple = vmax - vmin;

REQUIRE(v_out_mean == Approx(12.0).margin(0.5));   // ±0.5V
REQUIRE(v_out_ripple < 1.0);                        // < 1V p-p
```

## Python runner

```python
import math
import pulsim.v2 as p
import numpy as np

# Load circuit + sim options from YAML.
loaded = p.load_yaml_file("examples/v2/buck.yaml")

# Build the cache.
cache = p.PwlStateSpaceCache(loaded.builder.graph,
                              loaded.builder.pool)
cache.build(loaded.options.dt)

# 100 kHz PWM, 50% duty.
T_sw = 1e-5
duty = 0.5

def switch_fn(t):
    phase = math.fmod(t, T_sw) / T_sw
    if phase < duty:
        return p.SwitchStateMask(1)
    else:
        return p.SwitchStateMask(0)

# Run.
result = p.run_transient(
    cache, loaded.builder.graph, loaded.builder.pool,
    loaded.options, switch_fn=switch_fn)

# Output node index (vout).
vout_idx = loaded.builder.node_id_of("vout")

# Steady-state stats (last 10% of samples).
k_start = int(0.9 * result.num_steps())
v_out_samples = np.array([
    result.states[k][vout_idx]
    for k in range(k_start, result.num_steps())
])
print(f"V_out mean: {v_out_samples.mean():.3f} V")
print(f"V_out ripple p-p: {v_out_samples.ptp():.3f} V")
print(f"V_in · D: {24.0 * duty:.3f} V")
```

## Why this is the right validation

End-to-end SMPS showcases catch integration bugs that
per-layer unit tests miss:

- **YAML schema correctness**: any rename in the kernel
  that breaks the YAML schema gets caught here.
- **PWM + diode auto-commutation interaction**: the
  free-wheeling diode and body diode must hand off
  current correctly across switch transitions.
- **History-state correctness across switch state
  changes**: trap companion currents must remain
  continuous through commutation (caps + inductors
  don't snap).
- **Steady-state numerical stability**: 500 PWM cycles
  is enough for any subtle accumulation bug to show
  up.

## What V0 deliberately does NOT do

- **Closed-loop control**: V0 is open-loop fixed-duty.
  Adding a P/PI controller requires a stateful switch_fn
  (carrying integrator state across calls). The current
  `BExtraFn` / `SwitchScheduleFn` are stateless lambdas;
  V1 will explore stateful-callback or controller-block
  abstractions.
- **Boost / flyback / half-bridge** showcases: V0 ships
  buck only as proof-of-concept; the topology family
  expansion is V1.
- **Loss / efficiency analysis**: V0 verifies steady-
  state V_out and ripple. Detailed power-loss
  breakdown (R_on losses, diode conduction loss,
  switching loss) requires power-integration over
  branches — Layer 10 candidate.
- **Frequency sweep** for control-loop design: V0 is
  time-domain only.

## Files

- NEW `core/tests/v2/showcases/test_main.cpp`
- NEW `core/tests/v2/showcases/test_buck_open_loop.cpp`
- MODIFIED `core/CMakeLists.txt` (showcase test target)
- NEW `examples/v2/scripts/run_buck.py`
- NEW `docs/pulsim-v2/layer9-smps-showcase.md`
