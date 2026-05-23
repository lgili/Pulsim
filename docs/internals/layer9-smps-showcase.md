# Layer 9 — SMPS showcase (buck via YAML)

V0 through V8 built every v2 layer with dedicated unit and
integration tests. V9 is the **end-to-end integration
showcase**: load a real circuit from YAML, drive a 100 kHz
PWM, and verify steady-state output against the analytical
formula `V_out = V_in · D`.

This is the "v2 is production-ready" milestone — every
layer participates, every API is exercised through user-
facing entrypoints, against a power-electronics-textbook
target.

## What this showcase exercises

| Layer | Used for |
|-------|----------|
| Layer 0 (numeric / sparse) | Eigen + KLU under the hood |
| Layer 1 (topology) | Graph, BranchKind, SwitchStateMask |
| Layer 2 (devices) | VoltageSource, L, C, R, Diode, MOSFET-w-body-diode |
| Layer 2 V1 (MOSFET helpers) | `add_mosfet_with_body_diode` in YAML |
| Layer 4 (PWL cache) | `PwlStateSpaceCache::build` enumerates 8 segments (3 switches) |
| Layer 5 (run_transient) | dynamic-path trap-companion + event-iteration |
| Layer 5 V2 (diode auto-commutation) | Q1 body diode + D_FW auto-flip |
| Layer 6 (builder) | populated by YAML loader |
| Layer 8 (YAML) | `yaml::load_file` reads `buck.yaml` |

## The buck circuit (`examples/buck.yaml`)

```
       ┌───── Q1 ────────────┬── L ──┬── vout
       │  (MOSFET +          │       │
   24V │   body diode)       │       ├── C_out (47 µF)
       │                     │       │
       │              D_FW   ┴── gnd ├── R_L (5 Ω)
       │              (1k S on,      │
       │              1n S off,      └── gnd
       │              V_th=0.7V)
       └── gnd
```

YAML:

```yaml
circuit:
  devices:
    - {type: voltage_source,             name: Vin, from: vin, to: gnd, V: 24.0}
    - {type: mosfet_with_body_diode,     name: Q1, drain: vin, source: sw}
    - {type: diode,                      name: D_FW, anode: gnd, cathode: sw,
                                          g_on: 1.0e3, g_off: 1.0e-9, V_th: 0.7}
    - {type: inductor,                   name: L1, from: sw, to: vout, L: 100.0e-6}
    - {type: capacitor,                  name: Cout, from: vout, to: gnd, C: 47.0e-6}
    - {type: resistor,                   name: R_L, from: vout, to: gnd, R: 5.0}

simulation:
  t_start: 0.0
  t_end:   5.0e-3
  dt:      1.0e-7
```

## Steady-state math

For a buck converter with duty cycle `D` and assuming no
losses:

```
V_out = V_in · D
```

For our defaults (V_in=24V, D=0.5): `V_out = 12 V`.

With realistic losses (MOSFET R_on = 1 mΩ + small inductor
winding resistance), efficiency η ≈ 99.6 %:

```
V_out ≈ V_in · D · η = 24 · 0.5 · 0.996 ≈ 11.95 V
```

Measured (from the Python runner): **V_out_mean = 11.998 V**
— within 50 mV of the analytical target.

## Output ripple

The LC filter has time constant τ = √(L·C) ≈ 68 µs and
cutoff f_c ≈ 2.3 kHz. At 100 kHz switching frequency, the
filter's attenuation is:

```
A_filter = (f_c / f_sw)² ≈ (2.3 / 100)² ≈ 5e-4
```

So the switching ripple is bounded by:

```
ΔV ≈ V_in · (1−D) · D · A_filter · ... ≈ ~10 mV p-p
```

The test enforces ripple < 1 V p-p (a generous bound).

## Python runner

```python
$ python examples/scripts/run_buck.py

Loading examples/buck.yaml ...
  num_branches = 7
  dt = 1e-07 s
  t_end = 0.005 s
  cache built ✓
Running transient ...
  50001 samples

===== Steady-state V_out =====
  Mean:     11.998 V
  Ripple:   0.012 V (p-p)
  Target:   12.000 V  (= V_in · D)
  Error:    0.002 V
```

If matplotlib is installed, the script also saves a
`run_buck_output.png` plot of V_out vs time.

## C++ test

Two test cases in
`core/tests/v2/showcases/test_buck_open_loop.cpp`:

1. **`D = 50 % → V_out ≈ 12 V`** (the main scenario).
2. **`D = 25 % → V_out ≈ 6 V`** (verifies the
   relationship holds across operating points).

Both run in well under 1 s on a modern Mac.

## Extending the pattern

The same recipe — YAML → builder → cache → PWM switch_fn
→ run_transient — applies to:

- **Boost converter**: swap to `examples/boost.yaml`
  (V1 add-on). The PWM bit-0 still controls the MOSFET;
  topology changes via the YAML.
- **Flyback** (`examples/flyback.yaml`, already
  shipped): same pattern, just with a transformer in the
  topology.
- **Half-bridge / full-bridge inverters**: multiple PWM
  signals via multi-bit masks. The switch_fn returns the
  appropriate combination per time step.
- **Closed-loop control**: replace the open-loop PWM with
  a stateful controller (lambda capturing accumulators).
  V0 deliberately stays open-loop; closed-loop is V1.

## What V0 deliberately does NOT do

- **Closed-loop feedback**: V0 is fixed-duty open-loop.
  Adding a P/PI/PID controller requires a stateful
  switch_fn — the current `SwitchScheduleFn` is a
  `std::function<SwitchStateMask(Real)>` (stateless by
  signature, though lambdas can capture state). V1
  may add a dedicated controller-block abstraction.
- **Boost / flyback / half-bridge showcases**: V0 ships
  buck only as proof-of-concept. The flyback YAML
  exists (from Layer 8) but isn't yet wired to a
  runner.
- **Loss / efficiency analysis**: V0 measures V_out
  only. Power-integration over branches (R_on losses,
  diode conduction losses, switching losses) is a
  candidate for Layer 10.
- **Frequency-domain analysis**: V0 is time-domain only.

## Files

- NEW `core/tests/v2/showcases/test_main.cpp`
- NEW `core/tests/v2/showcases/test_buck_open_loop.cpp`
- MODIFIED `core/CMakeLists.txt`
  (`pulsim_v2_showcase_tests` target + `PULSIM_EXAMPLES_DIR`
  env var)
- NEW `examples/scripts/run_buck.py`
- MODIFIED `python/bindings_v2_kernel.cpp`
  (+ `SwitchStateMask.set/get`, + `Graph.num_switches`)
- NEW `openspec/changes/pulsim-v2-smps-showcase/`
