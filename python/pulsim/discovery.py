"""Pulsim — discovery helpers.

Interactive helpers that print the v2 catalogue + code snippets so
users can answer questions like:

  - "What devices does v2 ship?"
  - "Which PWM helper do I use for a 3-phase inverter?"
  - "How do I close a loop?"

without diving into header files.

Usage:

    >>> import pulsim as p
    >>> p.catalog()              # show everything, grouped
    >>> p.catalog("sources")     # show only sources
    >>> p.example("buck")        # print a buck-converter snippet
    >>> p.example("PIController")
    >>> p.tour()                 # 60-second walking tour of v2

The catalogue is hand-curated metadata in Python — kept close to
this module so it stays in sync with the public surface. When the
public API changes the catalog should be updated alongside it.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass


__all__ = [
    "catalog",
    "example",
    "tour",
    "list_topologies",
]


# =============================================================================
# Catalogue metadata — hand-curated, keep in sync with the public surface.
# =============================================================================

@dataclass(frozen=True)
class _Entry:
    name: str
    one_liner: str
    when: str = ""        # When to reach for this
    example: str = ""     # Short code snippet


_CATEGORIES: dict[str, list[_Entry]] = {
    "Linear passives": [
        _Entry("add_resistor", "Linear ohm-law resistor",
               "Always; the building block. Use small (1µΩ) values to "
               "tie floating subnets to ground."),
        _Entry("add_capacitor", "Linear capacitor with trapezoidal companion",
               "Output filtering, snubbers, AC coupling, DC bus, …"),
        _Entry("add_inductor", "Linear inductor with trapezoidal companion",
               "Filter chokes, transformer windings, current limit, …"),
        _Entry("add_saturable_inductor",
               "Nonlinear L(i) = L_0 / (1 + (i/I_sat)²) + L_residual",
               "When the core can saturate (most ferrites under heavy "
               "current). Triggers Newton refresh automatically."),
    ],

    "Sources (DC + time-varying)": [
        _Entry("add_voltage_source", "Pure DC V source",
               "Power inputs, DC buses, reference voltages."),
        _Entry("add_current_source", "Pure DC current source",
               "Bias currents, current sinks for sensor amps."),
        _Entry("add_sine_voltage_source",
               "v(t) = v_dc + v_amp · sin(2π·f·t + φ)",
               "AC mains, audio amp test, grid-tie inverter reference."),
        _Entry("add_pulse_voltage_source",
               "SPICE-style PULSE with optional rise/fall ramps",
               "Step responses, gate drives, periodic clocks, "
               "MOSFET/IGBT gate signals."),
        _Entry("add_pwm_voltage_source",
               "Square-wave PWM (freq + duty + phase)",
               "Open-loop SMPS gate drives — use PulseVoltageSource "
               "with rise_time for MOSFET/IGBT plants that need Newton "
               "to follow the gate edge."),
    ],

    "Switches + diodes": [
        _Entry("add_switch", "Binary controlled switch (g_on / g_off)",
               "Any controlled device when you don't need device-level "
               "physics. Driven by switch_fn(t)."),
        _Entry("add_diode",
               "SwitchedDiode — auto-commutated, event-detected",
               "Freewheel diodes, bridge rectifiers, snubbers. v2's "
               "event detection finds zero-crossings within sub-step "
               "and re-solves at the exact commutation time."),
        _Entry("add_nonlinear_diode",
               "Smooth-blend IdealDiode (AD-driven, Newton refresh)",
               "When you need a smooth I-V (Shockley-like) and don't "
               "want event detection — useful inside Newton loops."),
    ],

    "Active devices (nonlinear, Newton)": [
        _Entry("add_mosfet", "Simple MOSFET as a switch (linear R_on/R_off)",
               "Pedagogy + first-cut SMPS. No gate dynamics."),
        _Entry("add_mosfet_with_body_diode",
               "Same + anti-parallel SwitchedDiode (source→drain)",
               "Real MOSFETs — needed for inductive-load freewheeling "
               "during dead-time."),
        _Entry("add_mosfet_level1",
               "Shichman-Hodges Level 1 (smooth cutoff/triode/sat)",
               "When the MOSFET operates as an AMPLIFIER (CS amp, LDO "
               "pass element) — get V-controlled current. Optional "
               "with_body_diode=True for inductive loads."),
        _Entry("add_igbt",
               "Simple IGBT switch (R_on/R_off, no body diode)",
               "Open-loop high-voltage SMPS, motor drives."),
        _Entry("add_igbt_level1",
               "IGBT Level 1 (linear conduction + smooth on/off gating)",
               "Boost converters, hard-switched bridges where you need "
               "realistic on-state losses. Pair with a ramped V12 gate "
               "drive for clean Newton convergence."),
    ],

    "Coupled magnetics": [
        _Entry("add_transformer",
               "Two-winding ideal transformer (L_p, L_s, k)",
               "Flyback, forward, push-pull, full-bridge isolated DC-DC."),
        _Entry("add_multi_winding_transformer",
               "N-winding transformer with N×N coupling matrix",
               "Multi-output flybacks (main + aux bias + monitor)."),
    ],

    "Controlled sources / op-amps": [
        _Entry("add_vcvs",
               "Voltage-controlled voltage source: V_out = gain · (V_in_pos − V_in_neg)",
               "Building block for op-amps, sensor amps, error amps."),
        _Entry("add_op_amp_ideal",
               "Single-ended op-amp wrapper around add_vcvs (defaults to high gain)",
               "Closed-loop analog feedback (LDO error amp, "
               "Type-II compensator, summing junctions)."),
    ],

    "PWM helpers (switch_fn factories)": [
        _Entry("make_pwm_switch_fn",
               "Drive ONE switch with frequency + duty + phase",
               "Open-loop buck, boost, flyback — the simplest case."),
        _Entry("make_dead_time_pwm_pair_fn",
               "Complementary half-bridge with anti-shoot-through dead-time",
               "Synchronous buck/boost, half-bridge inverters."),
        _Entry("make_spwm_pair_fn",
               "Sinusoidal PWM half-bridge (carrier vs sine reference)",
               "Single-phase inverter output, audio class-D amplifier."),
        _Entry("make_three_phase_spwm_fn",
               "6-switch 3-phase VSI with 120° phase shift",
               "Motor drives, grid-tie inverters (use w/ ThreePhaseLegIndices)."),
        _Entry("make_phase_shift_full_bridge_fn",
               "4-switch full-bridge with phase-shift control",
               "ZVS isolated DC-DC, high-power resonant converters."),
        _Entry("make_combined_switch_fn",
               "Bitwise-OR multiple switch_fn callables",
               "When you need to drive several independent legs with "
               "different modulators."),
    ],

    "Control blocks (closed loop)": [
        _Entry("PIController",
               "Discrete PI with trapezoidal integration + conditional anti-windup",
               "Voltage / current loops in SMPS, motor speed loops."),
        _Entry("PIDController",
               "PI + filtered derivative (IIR alpha)",
               "When PI's phase margin isn't enough — adds phase boost "
               "via the D term. Used sparingly in power electronics."),
        _Entry("Comparator",
               "Schmitt trigger with hysteresis band",
               "Hysteresis current controllers, bang-bang control."),
        _Entry("RateLimiter",
               "Asymmetric slew limit on a signal",
               "Soft-start references, gate-drive de-glitching, "
               "anti-windup of integrator outputs."),
        _Entry("FirstOrderLowPass",
               "Single-pole IIR with continuous-time τ",
               "Smooth measurement noise before feeding to a PI; "
               "attenuate LC resonance peaks in feedback paths."),
        _Entry("SampleHold",
               "Periodic zero-order hold (digital controller emulation)",
               "Mimicking discrete digital controllers running at a "
               "rate slower than the simulation dt."),
        _Entry("LookupTable1D",
               "Sorted x → linearly interpolated y",
               "Soft-start ramps, gain schedules, V/f curves."),
        _Entry("MovingAverageFilter",
               "Exponential moving average (α carry-over)",
               "Cheap smoothing when you don't want a continuous-time τ."),
    ],

    "Math / signal blocks (v1 parity)": [
        _Entry("Gain", "y = k · x — stateless scalar multiply",
               "Scale a measurement or reference."),
        _Entry("Sum",
               "N-input weighted sum y = Σ w_i · x_i",
               "Summing junctions, current summing in multi-phase loops."),
        _Entry("Subtract", "y = a − b",
               "Error computation: error = setpoint − measurement."),
        _Entry("MathBlock",
               "Generic op ∈ {add, sub, mul, div, abs, neg, sqrt, pow2}",
               "One-off arithmetic without a custom block."),
        _Entry("Limiter", "Hard clamp y = clamp(x, min, max)",
               "Saturate references, gate drive voltages, etc."),
        _Entry("DelayBlock",
               "FIFO delay: outputs the input from N samples ago",
               "Modeling computational delay, transport lag, "
               "or sample-and-hold offsets."),
    ],

    "Standalone control blocks (v1 parity)": [
        _Entry("Integrator",
               "Discrete integrator gain·dt + anti-windup clamp",
               "When you need an integrator separate from a full PI "
               "(e.g. inside a custom compensator)."),
        _Entry("Differentiator",
               "Backward-difference + IIR filter",
               "Adding D action manually or estimating slopes."),
        _Entry("TransferFunction",
               "Direct-form II discrete H(z) — numerator + denominator",
               "Arbitrary digital filter, lead-lag, biquad, notch."),
        _Entry("StateMachine",
               "Mealy machine — toggle / level / SR-latch",
               "Bang-bang controllers, fault state machines, mode logic."),
        _Entry("OpAmp",
               "Pure-software saturating op-amp (not a circuit element)",
               "When you want a behavioural op-amp inside a Python "
               "block chain (use add_op_amp_ideal for circuit-side)."),
    ],

    "Modulation blocks (v1 parity)": [
        _Entry("PwmGenerator",
               "Software PWM modulator — sawtooth vs duty (output 0/1)",
               "When you need PWM as a SIGNAL (not a SwitchStateMask "
               "bit) — e.g. modulating a voltage-source value."),
        _Entry("SpaceVectorModulator",
               "Classical 7-segment SVM with zero-vector centering",
               "3-phase VSI control — higher modulation index than "
               "naive SPWM."),
    ],

    "Transform blocks (v1 parity)": [
        _Entry("ClarkeTransform",
               "abc → αβ0 (power-invariant convention)",
               "First stage of dq vector control on 3-phase systems."),
        _Entry("InverseClarkeTransform",
               "αβ0 → abc",
               "Final stage when sending duties back to a 3-leg VSI."),
        _Entry("ParkTransform",
               "αβ → dq with rotor angle θ",
               "Rotate the AC stationary frame into a DC rotor frame."),
        _Entry("InverseParkTransform",
               "dq → αβ",
               "Rotate the control output back to the stationary "
               "frame before applying to the inverter."),
    ],

    "Synchronization (v1 parity)": [
        _Entry("PLL",
               "Phase-locked loop on αβ — outputs (θ_hat, ω_hat)",
               "Grid-tie inverter synchronisation, motor sensorless "
               "rotor-angle estimation."),
    ],

    "Routing (v1 parity)": [
        _Entry("SignalMux",
               "Select one of N inputs based on a selector index",
               "Switch between control modes (e.g. soft-start ramp → "
               "PI command at t = T_soft)."),
        _Entry("SignalDemux",
               "Broadcast one input to N outputs",
               "Fan-out hint inside YAML chains (Python users can "
               "just reuse the variable)."),
    ],

    "Auto-tuning + frequency analysis": [
        _Entry("tune_pi_from_bode",
               "Loop-shaping: given plant Bode + target f_c + PM → (Kp, Ki)",
               "After measuring G_vd(jω), let the math pick gains."),
        _Entry("run_ac_sweep",
               "Swept-sine Bode of any plant (linear or nonlinear)",
               "Measure G_vd before tuning, or check loop after."),
        _Entry("loop_gain",
               "Compute L(jω) = C(jω) · G(jω) from a plant Bode + PI",
               "Verify GM/PM analytically before running the loop."),
        _Entry("phase_margin_from_loop / gain_margin_from_loop",
               "Extract PM / GM from a sampled loop gain",
               "Stability checks after auto-tuning."),
    ],

    "Top-level entry": [
        _Entry("simulate",
               "p.simulate(builder, t_end, dt, **kwargs) — one-call API",
               "ALWAYS the first thing to reach for. Hides cache "
               "building, switch_fn defaults, nonlinear-refresh "
               "detection."),
        _Entry("load_yaml_file",
               "Parse a v2 YAML netlist into a LoadedCircuit",
               "When you have a YAML showcase or want a declarative "
               "circuit description."),
        _Entry("scope",
               "One-line waveform plot — auto-detect node indices",
               "After simulate(...) to look at V_out, gate signals, etc."),
        _Entry("plot_bode",
               "Magnitude + phase Bode plot with optional analytical overlay",
               "Render `run_ac_sweep` results without writing matplotlib."),
    ],
}


# =============================================================================
# Code snippets — keyed by short name.
# =============================================================================

_EXAMPLES: dict[str, str] = {
    "rc": """\
        import pulsim as p

        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "n0", "gnd", 5.0)
        b.add_resistor       ("R1", "n0", "vc",  1000.0)
        b.add_capacitor      ("C1", "vc", "gnd", 1e-6)

        res = p.simulate(b, t_end=5e-3, dt=1e-5)
        p.scope(b, res, signals=["vc"])
        """,

    "buck": """\
        import pulsim as p

        b = p.CircuitBuilder()
        b.add_voltage_source        ("Vin", "vin", "gnd", 24.0)
        b.add_mosfet_with_body_diode("Q1",  "vin", "sw")
        b.add_diode                 ("D",   "gnd", "sw", 1e3, 1e-9, V_th=0.7)
        b.add_inductor              ("L",   "sw",  "vout", 100e-6)
        b.add_capacitor             ("Cout","vout","gnd",  47e-6)
        b.add_resistor              ("R_L", "vout","gnd",  10.0)

        pwm = p.make_pwm_switch_fn(
            frequency=100e3, duty=0.5,
            switch_idx=0, num_switches=b.graph.num_switches,
        )
        res = p.simulate(b, t_end=5e-3, dt=1e-7, switch_fn=pwm)
        p.scope(b, res, signals=["vout", "sw"])
        """,

    "boost": """\
        import pulsim as p

        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 12.0)
        b.add_inductor      ("L",   "vin", "sw",  100e-6)
        b.add_mosfet        ("Q1",  "sw",  "gnd")
        b.add_diode         ("D",   "sw",  "vout", 1e3, 1e-9, V_th=0.5)
        b.add_capacitor     ("Cout","vout","gnd",  100e-6)
        b.add_resistor      ("R_L", "vout","gnd",  20.0)

        pwm = p.make_pwm_switch_fn(
            frequency=50e3, duty=0.4,
            switch_idx=0, num_switches=b.graph.num_switches,
        )
        res = p.simulate(b, t_end=3e-3, dt=2e-7, switch_fn=pwm)
        p.scope(b, res, signals=["vout"])
        """,

    "PIController": """\
        import math, pulsim as p

        pi = p.PIController(
            Kp=0.05, Ki=500.0,
            output_min=0.05, output_max=0.95,
        )

        # Closed-loop pattern: step_observer reads V_out, updates duty.
        duty = [0.5]
        def observe(t, x):
            v_out = float(x[vout_idx])
            duty[0] = pi.update(setpoint=12.0, measured=v_out, dt=DT)

        # switch_fn reads the latest duty.
        def switch_fn(t):
            phase = math.fmod(t, T_PWM) / T_PWM
            m = p.SwitchStateMask(n_sw)
            if phase < duty[0]:
                m.set(0, True)
            return m

        res = p.simulate(b, t_end=3e-3, dt=DT,
                         switch_fn=switch_fn, step_observer=observe)
        """,

    "bode": """\
        import math, numpy as np, pulsim as p

        # Sweep a buck plant Bode.
        D_OP, T_PWM = 0.5, 1e-5

        def excite(t, eps, freq):
            return [0.0] * b.pool.state_size(b.graph)

        def make_sw(eps, freq):
            omega = 2*math.pi*freq
            def sw(t):
                duty = D_OP + eps*math.sin(omega*t)
                phase = math.fmod(t, T_PWM)/T_PWM
                m = p.SwitchStateMask(n_sw)
                if phase < duty: m.set(0, True)
                return m
            return sw

        freqs = np.logspace(2, 4, 25)
        result = p.run_ac_sweep(
            b, freqs=freqs, excite_fn=excite,
            switch_fn_factory=make_sw,
            output_idx=b.node_id_of("vout"),
            perturbation_amplitude=0.05,
        )
        p.plot_bode(result, title="Buck control-to-output")
        """,

    "tune": """\
        # After sweeping a plant Bode (see `p.example('bode')`):
        tune = p.tune_pi_from_bode(
            result.freqs, result.H,
            f_crossover=2000.0, phase_margin_deg=60.0,
        )
        print(f"Kp={tune['Kp']:.4f}, Ki={tune['Ki']:.1f}, "
              f"PM={tune['achieved_pm_deg']:.1f}°")

        # Plug it straight into the closed loop:
        pi = tune['controller']
        """,

    "yaml": """\
        import pulsim as p

        loaded = p.load_yaml_file("examples/v2/buck.yaml")
        pwm = p.make_pwm_switch_fn(
            frequency=100e3, duty=0.5,
            switch_idx=0,
            num_switches=loaded.builder.graph.num_switches,
        )
        res = p.simulate(loaded.builder,
                         t_end=loaded.options.t_end,
                         dt=loaded.options.dt,
                         switch_fn=pwm)
        p.scope(loaded.builder, res, signals=["vout"])
        """,
}


# =============================================================================
# Public functions
# =============================================================================

def catalog(category: str | None = None) -> None:
    """Print the pulsim catalogue, optionally filtered by category.

    `category` is matched case-insensitively as a substring of the
    category name. Pass `None` (the default) to print everything.

    >>> p.catalog()
    >>> p.catalog("sources")
    >>> p.catalog("control")
    """
    print("\n" + "=" * 78)
    print(" pulsim — available API")
    print("=" * 78)
    matched_any = False
    needle = category.lower() if category else None
    for cat_name, entries in _CATEGORIES.items():
        if needle is not None and needle not in cat_name.lower():
            continue
        matched_any = True
        print(f"\n# {cat_name}\n" + "-" * len(cat_name))
        for e in entries:
            name = e.name
            print(f"\n  p.{name}")
            for line in textwrap.wrap(e.one_liner, width=70,
                                        initial_indent="    ",
                                        subsequent_indent="    "):
                print(line)
            if e.when:
                for line in textwrap.wrap(f"when: {e.when}", width=70,
                                            initial_indent="    · ",
                                            subsequent_indent="      "):
                    print(line)
    if not matched_any:
        print(f"\n  No category matches '{category}'. Available:")
        for cat_name in _CATEGORIES:
            print(f"    - {cat_name}")
    print()
    print("  Tip: p.example('buck'), p.example('PIController'), "
          "p.example('bode')")
    print("       see also p.tour() for a 1-page walkthrough")
    print()


def example(name: str) -> None:
    """Print a runnable code snippet for `name`.

    Recognised names:

    >>> p.example("rc")              # RC charging
    >>> p.example("buck")            # open-loop buck
    >>> p.example("boost")           # open-loop boost
    >>> p.example("PIController")    # closed-loop PI pattern
    >>> p.example("bode")            # AC sweep
    >>> p.example("tune")            # auto-tune from Bode
    >>> p.example("yaml")            # load YAML netlist

    Pass an unknown name to see the list of available snippets.
    """
    key = name.lower()
    # Try direct match first, then case-insensitive.
    snippet = _EXAMPLES.get(name) or _EXAMPLES.get(key)
    if snippet is None:
        print(f"  No example named '{name}'. Available:")
        for k in _EXAMPLES:
            print(f"    p.example({k!r})")
        return
    print(f"\n--- example: {name} ---")
    print(textwrap.dedent(snippet).strip())
    print()


def tour() -> None:
    """Print a one-page walking tour of pulsim — read top to bottom."""
    print("""
=============================================================================
 pulsim — one-page tour
=============================================================================

1. BUILD A CIRCUIT (fluent Python or YAML)

   import pulsim as p
   b = p.CircuitBuilder()
   b.add_voltage_source("Vin", "n0", "gnd", 5.0)
   b.add_resistor       ("R",  "n0", "vc",  1000.0)
   b.add_capacitor      ("C",  "vc", "gnd", 1e-6)

   # … or:  loaded = p.load_yaml_file("examples/v2/buck.yaml")

2. SIMULATE (one call, everything auto)

   res = p.simulate(b, t_end=5e-3, dt=1e-5)
   #                       (`simulate` auto-detects nonlinear devices
   #                        and turns on Newton refresh as needed)

3. PLOT (no matplotlib boilerplate)

   p.scope(b, res, signals=["vc"])

4. ADD A SWITCHING CONVERTER

   pwm = p.make_pwm_switch_fn(frequency=100e3, duty=0.5,
                                 switch_idx=0,
                                 num_switches=b.graph.num_switches)
   res = p.simulate(b, t_end=5e-3, dt=1e-7, switch_fn=pwm)

5. CLOSE THE LOOP

   pi = p.PIController(Kp=0.05, Ki=200, output_min=0.05, output_max=0.95)
   duty = [0.5]
   def observe(t, x):
       duty[0] = pi.update(setpoint=12.0,
                             measured=float(x[vout_idx]), dt=dt)
   p.simulate(b, ..., step_observer=observe, switch_fn=switch_fn)

6. MEASURE A BODE → AUTO-TUNE

   sweep = p.run_ac_sweep(b, freqs=..., excite_fn=..., output_idx=...)
   gains = p.tune_pi_from_bode(sweep.freqs, sweep.H,
                                 f_crossover=2e3, phase_margin_deg=60)
   pi = gains['controller']   # ready to use

7. DISCOVER MORE

   p.catalog()                # all components, grouped
   p.catalog("sources")       # one category
   p.example("buck")          # code snippet
   p.list_topologies()        # 13 YAML showcases + 6 closed-loop examples

=============================================================================
""")


def list_topologies() -> None:
    """Print the ready-to-run examples in `examples/v2/scripts/`."""
    print("""
Ready-to-run scripts in examples/v2/scripts/
============================================

Open-loop transient (one per YAML in examples/v2/):
  run_rlc_step_response.py            RLC step, ringing
  run_half_wave_rectifier.py          60 Hz sine + diode + R
  run_common_source_amplifier.py      SH1 MOSFET CS amp
  run_buck.py                         100 kHz buck 24 V → 12 V
  run_flyback.py                      Isolated flyback 48 V → 24 V
  run_pwm_chopper_realistic_mosfet.py V13 MOSFET chopper
  run_boost_realistic_igbt.py         IGBT boost w/ ramped gate
  run_boost_realistic_mosfet_v14.py   SH1 MOSFET boost (Newton hardening)
  run_boost_saturable_inductor.py     Saturable L (V17)
  run_three_phase_diode_rectifier.py  3-φ → DC
  run_three_phase_inverter.py         6-switch VSI + SPWM
  run_phase_shift_full_bridge.py      ZVS phase-shift FB
  run_ldo_with_opamp.py               LDO with op-amp feedback

Closed-loop showcases:
  run_buck_closed_loop.py             Buck w/ PI voltage regulation
  run_boost_closed_loop.py            Boost w/ PI — shows RHPZ
  run_flyback_closed_loop.py          Flyback closed loop
  run_boost_saturable_closed_loop.py  V17 + Newton + PI
  run_three_phase_dq_closed_loop.py   abc-dq + 2 PI controllers
  run_pfc_boost_closed_loop.py        Cascaded current + voltage loops

Auto-tune / AC analysis:
  run_ac_sweep_rc.py                  Verify against analytical
  run_ac_sweep_buck.py                Buck G_vd(s) Bode
  run_ac_sweep_nonlinear.py           Saturable boost at 2 OPs
  run_auto_tune_buck.py               Auto-tune PI from Bode (4 steps)

Diagnostic:
  run_adversarial_sweep.py            9 extreme-parameter cases
  debug_buck_bode_offset.py           PWM resolution / Bode story

Run with:
    export PYTHONPATH="$(pwd)/python:$PYTHONPATH"
    python3 examples/v2/scripts/run_buck.py
""")
