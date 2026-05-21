# Pulsim v2 — Python example runners

This directory contains **one Python runner per YAML showcase** in
`examples/v2/`. Each script is a self-contained, end-to-end demo:
build the circuit, run the transient, plot the main waveforms.

## Authoring toggle

Every script has a module-level constant near the top:

```python
USE_YAML = True   # ← change to False to see the equivalent Python builder
```

- **`USE_YAML = True`** loads the bundled YAML via `p.load_yaml_file(...)`.
- **`USE_YAML = False`** builds the same circuit programmatically via
  `p.CircuitBuilder().add_*()` calls.

Both paths produce identical results — pick whichever feels more
natural for your workflow. YAML is great for fixed reference circuits;
the Python builder shines when you want to sweep a parameter or build
a circuit dynamically.

## High-level API: `simulate()`

All scripts use the one-call `pulsim.v2.simulate(builder, t_end, dt, **kwargs)`
wrapper (proposal #3.3). It handles:

- Building the `PwlStateSpaceCache`
- Constructing `SimulationOptions`
- Defaulting `switch_fn` to "all switches closed" (override per script)
- Auto-detecting nonlinear devices and enabling Newton refresh
- Forwarding advanced Newton knobs (`tol_newton_dx`, `enable_newton_line_search`, …)

If you need finer control, drop down to the raw `p.run_transient(...)`
call shown in [`docs/v2/api-reference.md`](../../../docs/v2/api-reference.md).

## The 13 examples

| Script | What it simulates | Devices exercised |
|---|---|---|
| `run_rlc_step_response.py` | Series L-R-C step response, ζ ≈ 0.05 underdamped ringing | V12 pulse + L/R/C |
| `run_half_wave_rectifier.py` | 60 Hz sine → diode → R, half-wave rectified output | V11 sine + diode |
| `run_common_source_amplifier.py` | Classic CS amp, Av ≈ −10 V/V at the operating point | V13 SH1 MOSFET + V11 sine |
| `run_buck.py` | Synchronous buck, 24 V → 12 V at 5 Ω, 100 kHz / 50 % | MOSFET (switch) + diode + L/C |
| `run_flyback.py` | Isolated flyback, 48 V → ~24 V via 4:1 transformer | MOSFET + transformer + diode |
| `run_pwm_chopper_realistic_mosfet.py` | V13 SH1 MOSFET in a hard-switched chopper | SH1 MOSFET + V12 pulse |
| `run_boost_realistic_igbt.py` | Boost with V14 IGBT + ramped gate drive | V14 IGBT + V12 pulse + L/D/C |
| `run_boost_realistic_mosfet_v14.py` | Boost with V13 SH1 MOSFET + Newton hardening | V13 MOSFET (the convergence story) |
| `run_boost_saturable_inductor.py` | Boost where L saturates at I_sat = 1.5 A | V17 saturable inductor |
| `run_three_phase_diode_rectifier.py` | 3-φ diode bridge → DC bus | 3× V11 sine + 6× diode |
| `run_three_phase_inverter.py` | 3-φ VSI with SPWM driving Y-RL load | 6 switches + 6 body diodes + SPWM |
| `run_phase_shift_full_bridge.py` | ZVS phase-shift full-bridge isolated DC-DC | 4-switch FB + transformer + bridge rectifier |
| `run_ldo_with_opamp.py` | Linear regulator with op-amp feedback (12 V → 7.5 V) | V15 op-amp + V13 MOSFET |

## Closed-loop showcases + stress tests

These use the `step_observer(t, x)` kernel callback + the
`pulsim.v2_control` block library (PIController, etc.):

| Script | What it simulates | Stress |
|---|---|---|
| `run_buck_closed_loop.py` | Buck w/ PI voltage loop + setpoint step | Cache + commutation |
| `run_boost_closed_loop.py` | Boost w/ PI, shows RHPZ undershoot on setpoint step | RHPZ + Newton |
| `run_flyback_closed_loop.py` | Isolated flyback CL — transformer + commutation | All of the above + iso |
| `run_boost_saturable_closed_loop.py` | Saturable boost CL — i_L pushes 264 A through saturation | V17 Newton + PI |
| `run_three_phase_dq_closed_loop.py` | 3-φ VSI, abc-dq xform, two PI on i_d / i_q | 12 switches + dq + 2 PIs |
| `run_pfc_boost_closed_loop.py` | Single-phase PFC: AC mains → bridge → boost, cascaded V+I PIs | 5 diodes commuting + cascade |
| `run_adversarial_sweep.py` | 9 parametric variants (dt=1ns, Ki=1e6, Kp<0, …) | Find kernel bugs |

The adversarial sweep summary at last run:

```
case                                       dt      Kp        Ki    v_out
baseline (sane)                        100ns   0.050     800.0   10.937
dt = 1 ns (tiny)                         1ns   0.050     800.0    0.205
dt = 50 µs (>Nyquist)                50000ns   0.050     800.0   15.178
Ki = 1e6 (huge)                        100ns   0.050 1000000.0   12.119
Kp=0 (pure I)                          100ns   0.000    1000.0   14.336
Ki=0 (pure P)                          100ns  10.000       0.0   11.930
Kp<0 (positive feedback)               100ns  -0.100    -200.0    1.199
start_from_dc_op=True                  100ns   0.050     800.0   11.912
substep correction ON                  100ns   0.050     800.0   10.937
```

All 9 cases completed without crashes (✓).

## How to run one

From the repo root:

```bash
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"
python3 examples/v2/scripts/run_buck.py
```

Each runner prints a sanity-check summary (steady-state mean,
overshoot, ripple) and saves a 2-subplot PNG to `output/<name>.png`.

## How to run them all

```bash
for f in examples/v2/scripts/run_*.py; do
    echo "=== $f ==="
    PYTHONPATH=python python3 "$f"
done
```

## Where to go from here

- [`docs/v2/getting-started.md`](../../../docs/v2/getting-started.md) — the 3-line RC example.
- [`docs/v2/tutorials/`](../../../docs/v2/tutorials/) — 6 walk-throughs covering the same converters in narrative form.
- [`docs/v2/api-reference.md`](../../../docs/v2/api-reference.md) — full Python surface.
- [`docs/v2/gotchas.md`](../../../docs/v2/gotchas.md) — Newton convergence corner cases.
