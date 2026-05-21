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
