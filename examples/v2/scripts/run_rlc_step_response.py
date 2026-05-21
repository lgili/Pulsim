#!/usr/bin/env python3
"""RLC step response — series L-R-C driven by a 10 V pulse.

Underdamped 2nd-order ringing:

   V_step ── L ── R ── C ── gnd
              (100 µH) (0.1 Ω) (100 µF)

ω_n = 1/√(LC) = 10 000 rad/s ≈ 1591 Hz
ζ   = (R/2)·√(C/L) ≈ 0.05  (lightly damped)
Peak overshoot ≈ 18.5 V at t ≈ 314 µs after the step (which fires
at t = 100 µs).

This is the simplest pedagogical case: no switches, no PWM, no
nonlinear devices. `simulate()` collapses to one line.

Toggle USE_YAML between True (load examples/v2/rlc_step_response.yaml)
and False (build the same circuit from Python directly) to see
both authoring styles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim.v2 as p


# =============================================================================
# Authoring toggle
# =============================================================================
USE_YAML = True   # ← change to False to see the equivalent Python builder


# -----------------------------------------------------------------------------
# Option A — Load the bundled YAML.
# -----------------------------------------------------------------------------
def load_from_yaml() -> tuple[p.CircuitBuilder, float, float, float]:
    yaml_path = Path(__file__).resolve().parent.parent / "rlc_step_response.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"missing YAML: {yaml_path}")
    print(f"Loading {yaml_path}")
    loaded = p.load_yaml_file(str(yaml_path))
    return (loaded.builder,
            loaded.options.t_start,
            loaded.options.t_end,
            loaded.options.dt)


# -----------------------------------------------------------------------------
# Option B — Build the same circuit programmatically.
# -----------------------------------------------------------------------------
def build_from_python() -> tuple[p.CircuitBuilder, float, float, float]:
    """Series L-R-C with a delayed 10 V step on the 'in' node."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source(
        "Vstep", "in", "gnd",
        v_initial=0.0, v_pulsed=10.0,
        t_start=1.0e-4,           # 100 µs delay so baseline is visible
        pulse_width=1.0,          # stays high forever (single-shot)
    )
    b.add_inductor ("L", "in",     "lr_mid", 100e-6)   # 100 µH
    b.add_resistor ("R", "lr_mid", "vc",     0.1)      # 0.1 Ω → ζ ≈ 0.05
    b.add_capacitor("C", "vc",     "gnd",    100e-6)   # 100 µF
    return b, 0.0, 3.0e-3, 5.0e-7


# =============================================================================
# Plot helpers + main
# =============================================================================
def main() -> None:
    if USE_YAML:
        builder, t_start, t_end, dt = load_from_yaml()
    else:
        builder, t_start, t_end, dt = build_from_python()
    print(f"  authoring mode: {'YAML' if USE_YAML else 'Python builder'}")
    print(f"  num_branches:    {builder.num_branches}")
    print(f"  t_end/dt:        {t_end*1e3:.2f} ms / {dt*1e9:.0f} ns")

    # Single-line simulate(): no switches, no nonlinear devices.
    res = p.simulate(builder, t_end=t_end, dt=dt, t_start=t_start)
    print(f"  samples: {res.num_steps()}")

    # Indices for the two waveforms we want to plot.
    vc_idx = builder.node_id_of("vc")
    vin_idx = builder.node_id_of("in")

    times = np.asarray(res.times) * 1e3   # ms
    v_in  = np.array([s[vin_idx] for s in res.states])
    v_c   = np.array([s[vc_idx]  for s in res.states])

    # Quick steady-state check.
    final_window = v_c[int(0.9 * len(v_c)):]
    print(f"  V_C (last 10% mean) = {final_window.mean():.3f} V "
          f"(target 10.000 V)")
    overshoot = v_c.max() - 10.0
    print(f"  V_C peak overshoot   = {overshoot:.3f} V "
          f"(analytical ≈ 8.5 V)")

    # -------------------------------------------------------------------------
    # Plot — two subplots: input step and capacitor ringing.
    # -------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the waveform plot)")
        return

    fig, (ax_in, ax_c) = plt.subplots(2, 1, figsize=(10, 6),
                                       sharex=True)
    ax_in.plot(times, v_in, color="tab:blue", lw=1.0)
    ax_in.set_ylabel("V_in [V]")
    ax_in.set_title("RLC step response — V_step + V_C")
    ax_in.grid(alpha=0.3)

    ax_c.plot(times, v_c, color="tab:orange", lw=1.0)
    ax_c.axhline(10.0, color="k", ls="--", lw=0.8,
                  label="DC target (10 V)")
    ax_c.set_xlabel("time [ms]")
    ax_c.set_ylabel("V_C [V]")
    ax_c.grid(alpha=0.3)
    ax_c.legend(loc="lower right")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "rlc_step_response.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
