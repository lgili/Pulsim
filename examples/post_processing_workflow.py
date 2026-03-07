"""Post-processing workflow example for Pulsim.

Demonstrates:
- Declaring post-processing jobs via the Python API
- Running all three job kinds: TimeDomain, Spectral, PowerEfficiency
- Consuming structured results (scalar metrics, spectrum, harmonics, efficiency)
- Using various window modes (Time, Cycle)
- Handling undefined metrics and diagnostic codes

Usage:
    PYTHONPATH=build/python python3 examples/post_processing_workflow.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

import pulsim as ps

# ---------------------------------------------------------------------------
# Build a synthetic SimulationResult (replaces a real YAML-driven simulation
# for this standalone example).
# ---------------------------------------------------------------------------

SWITCHING_FREQ = 20_000.0   # Hz — typical buck converter
PERIOD = 1.0 / SWITCHING_FREQ
T_STOP = 10 * PERIOD          # 10 switching cycles
N = 20_000                    # 2000 samples per period

t = np.linspace(0.0, T_STOP, N, endpoint=False)

# Simulated output voltage: DC (12 V) + small ripple + 2nd harmonic content
V_dc = 12.0
V_ripple_amp = 0.3
v_out = (
    V_dc
    + V_ripple_amp * np.sin(2 * np.pi * SWITCHING_FREQ * t)
    + 0.05 * np.sin(2 * np.pi * 2 * SWITCHING_FREQ * t)
)

# Simulated input voltage and current (ideal 24 V source, ~0.6 A)
v_in = np.full(N, 24.0)
i_in = np.full(N, 0.625)   # P_in = 15 W

# Simulated output current through load resistor R = 12/0.5 = 24 Ω → I_out ≈ 0.5 A
i_out = v_out / 24.0

# Build a fake SimulationResult-compatible object
class FakeResult:
    def __init__(self):
        self.time = t.tolist()
        self.virtual_channels = {
            "V(out)": v_out.tolist(),
            "V(in)": v_in.tolist(),
            "I(in)": i_in.tolist(),
            "I(out)": i_out.tolist(),
        }
        self.success = True

result = FakeResult()

# ---------------------------------------------------------------------------
# Declare post-processing jobs
# ---------------------------------------------------------------------------

# Window: last 5 periods (steady state) using time bounds
steady_state_window = ps.PostProcessingWindowSpec(
    mode=ps.PostProcessingWindowMode.Time,
    t_start=5 * PERIOD,
    t_end=T_STOP,
)

# Alternatively, use cycle window: last 5 switching cycles
cycle_window = ps.PostProcessingWindowSpec(
    mode=ps.PostProcessingWindowMode.Cycle,
    cycle_start=5,
    cycle_end=10,
    period=PERIOD,
)

opts = ps.PostProcessingOptions(jobs=[

    # --- Time-domain metrics: output voltage ---
    ps.PostProcessingJob(
        job_id="vout_time_domain",
        kind=ps.PostProcessingJobKind.TimeDomain,
        signals=["V(out)"],
        metrics=["rms", "mean", "min", "max", "p2p", "ripple"],
        window=steady_state_window,
    ),

    # --- Spectral analysis: output voltage spectrum and THD ---
    ps.PostProcessingJob(
        job_id="vout_spectrum",
        kind=ps.PostProcessingJobKind.Spectral,
        signals=["V(out)"],
        fundamental_hz=SWITCHING_FREQ,
        n_harmonics=5,
        window_function=ps.WindowFunction.Hann,
        window=steady_state_window,
    ),

    # --- Power and efficiency ---
    ps.PostProcessingJob(
        job_id="converter_efficiency",
        kind=ps.PostProcessingJobKind.PowerEfficiency,
        input_voltage_signal="V(in)",
        input_current_signal="I(in)",
        output_voltage_signal="V(out)",
        output_current_signal="I(out)",
        window=steady_state_window,
    ),

    # --- Cycle-window example ---
    ps.PostProcessingJob(
        job_id="vout_last5cycles",
        kind=ps.PostProcessingJobKind.TimeDomain,
        signals=["V(out)"],
        metrics=["rms", "ripple"],
        window=cycle_window,
    ),
])

# ---------------------------------------------------------------------------
# Execute post-processing
# ---------------------------------------------------------------------------

pp = ps.run_post_processing(result, opts)

print("=" * 60)
print("Pulsim Post-Processing Example")
print("=" * 60)
print(f"Overall success: {pp.success}")
print(f"Jobs completed: {len(pp.jobs)}")
print(f"Total jobs failed: {sum(1 for jr in pp.jobs if not jr.success)}")
print()

# ---------------------------------------------------------------------------
# Consume time-domain results
# ---------------------------------------------------------------------------

td = pp.jobs[0]
print(f"[{td.job_id}] — Time-Domain Metrics (V(out))")
if td.success:
    for name, m in sorted(td.scalar_metrics.items()):
        print(f"  {name:12s} = {m.value:.6f} {m.unit}")
else:
    print(f"  FAILED: {td.diagnostic} — {td.diagnostic_message}")
if td.undefined_metrics:
    for u in td.undefined_metrics:
        print(f"  UNDEFINED: {u.name} — {u.reason}: {u.reason_message}")
print()

# ---------------------------------------------------------------------------
# Consume spectral results
# ---------------------------------------------------------------------------

spec = pp.jobs[1]
print(f"[{spec.job_id}] — Spectral Analysis (V(out))")
if spec.success:
    print(f"  Fundamental:  {spec.fundamental_hz:.1f} Hz")
    print(f"  THD:          {spec.thd_pct:.3f} %")
    print(f"  Spectrum bins: {len(spec.spectrum_bins)}")
    print("  Harmonics:")
    for h in spec.harmonics:
        print(
            f"    H{h.harmonic_number}: {h.frequency_hz:8.1f} Hz  "
            f"mag={h.magnitude:.6f}  ({h.magnitude_pct_fundamental:.2f}% of fundamental)"
        )
else:
    print(f"  FAILED: {spec.diagnostic} — {spec.diagnostic_message}")
print()

# ---------------------------------------------------------------------------
# Consume power/efficiency results
# ---------------------------------------------------------------------------

eff = pp.jobs[2]
print(f"[{eff.job_id}] — Power and Efficiency")
if eff.success:
    print(f"  P_in:         {eff.average_input_power:.4f} W")
    print(f"  P_out:        {eff.average_output_power:.4f} W")
    print(f"  Efficiency:   {eff.efficiency:.3f} %")
    print(f"  Power factor: {eff.power_factor:.4f}")
else:
    print(f"  FAILED: {eff.diagnostic} — {eff.diagnostic_message}")
print()

# ---------------------------------------------------------------------------
# Cycle-window results
# ---------------------------------------------------------------------------

cw = pp.jobs[3]
print(f"[{cw.job_id}] — Cycle Window (last 5 cycles)")
if cw.success:
    for name, m in sorted(cw.scalar_metrics.items()):
        print(f"  {name:12s} = {m.value:.6f} {m.unit}")
else:
    print(f"  FAILED: {cw.diagnostic} — {cw.diagnostic_message}")
print()

# ---------------------------------------------------------------------------
# Validation against theoretical values
# ---------------------------------------------------------------------------

print("=" * 60)
print("Validation vs. theoretical values")
print("=" * 60)

td_rms = pp.jobs[0].scalar_metrics["rms"].value
# RMS of V_dc + V_ripple*sin(w*t): sqrt(V_dc^2 + (V_ripple/sqrt(2))^2 + ...)
# Approximate: dominant terms
expected_rms_approx = math.sqrt(V_dc**2 + (V_ripple_amp / math.sqrt(2))**2)
print(f"  RMS computed:    {td_rms:.6f} V")
print(f"  RMS theoretical: {expected_rms_approx:.6f} V  (approx)")
print(f"  Error:           {abs(td_rms - expected_rms_approx):.6f} V")
print()

eff_val = pp.jobs[2].efficiency
p_in = 24.0 * 0.625   # 15.0 W
p_out_approx = float(np.mean(v_out[N // 2:] * i_out[N // 2:]))
expected_eff = p_out_approx / p_in * 100.0
print(f"  Efficiency computed: {eff_val:.4f} %")
print(f"  Efficiency expected: {expected_eff:.4f} %  (in-window average)")

print()
print("Example complete.")
