"""End-to-end refrigerator compressor — fixed-frequency CC convencional.

Models a typical domestic refrigerator hermetic compressor:

  * Motor   : single-phase induction (PSC topology) on 220 V / 60 Hz line
              — the legacy *compressor convencional* (CC) that's still
              the workhorse of pre-inverter fridges and freezers.
  * Load    : Reciprocating piston compressor with R600a (isobutane,
              the modern domestic-refrigeration standard since the
              HFC phase-out).
  * Working : 6 cm³ displacement, 0.59 bar evaporator, 5.30 bar condenser
              (R600a defaults from `pulsim.refrigerant(R600a)`).

Run::

    python 14_refrigerator_compressor.py

Prints the analytical compression cycle numbers, runs a 200 ms
transient, then reports mechanical state. With no compressor load the
rotor would lock to synchronous speed (188.5 rad/s for 4-pole / 60 Hz)
in a fraction of a second. With the compressor load attached the rotor
settles below sync — the gap is the slip required to develop enough
torque to match the compression demand.

See also:
  * docs/motor-models.md  (the single-phase induction motor section)
  * docs/compressor-and-refrigerant-load.md  (the refrigerant table +
    polytropic physics)
"""

from __future__ import annotations

import math

import pulsim as ps


def build_circuit() -> ps.Circuit:
    ckt = ps.Circuit()

    line = ckt.add_node("line")
    neutral = ckt.add_node("neutral")

    # 220 V (RMS) / 60 Hz line voltage tied to ground at the neutral.
    v_peak = 220.0 * math.sqrt(2.0)
    ckt.add_sine_voltage_source("V_line", line, neutral, v_peak, 60.0)
    ckt.add_voltage_source("V_n", neutral, ps.Circuit.ground(), 0.0)

    # 1φ PSC induction motor — defaults are Embraco-style 1/8 HP
    # compressor parameters (R_s_main = 10 Ω, L_s_main = 50 mH,
    # R_s_aux = 20 Ω, L_s_aux = 80 mH, C_run = 4 µF, etc.).
    motor_params = ps.SinglePhaseInductionMotorParams()
    ckt.add_single_phase_induction_motor(
        "M_compressor", line, neutral, motor_params
    )

    # Reciprocating compressor with R600a — start from the curated
    # refrigerant defaults (polytropic_n = 1.13, P_suction = 0.59 bar,
    # P_discharge = 5.30 bar) and override the per-machine fields.
    comp = ps.compressor_defaults_for(ps.Refrigerant.R600a)
    comp.topology = ps.CompressorTopology.Reciprocating
    comp.displacement_m3 = 6.0e-6  # 6 cm³ per revolution
    comp.b_friction = 5e-4
    comp.tau_coulomb = 0.02
    ckt.attach_compressor_load("M_compressor", comp)

    return ckt


def print_analytical(ckt: ps.Circuit) -> None:
    print("=== Analytical compression cycle (R600a, 6 cm³, 0.59 → 5.30 bar) ===")
    print(f"  Mean compression torque : {ckt.compressor_mean_torque('M_compressor'):.4f} N·m")
    print(f"  Indicated work / cycle  : {ckt.compressor_indicated_work('M_compressor'):.4f} J")

    # Sample the angle-dependent torque profile.
    print("  Instantaneous torque demand at four crank angles (ω = 188 rad/s):")
    for label, theta in [("0°", 0.0), ("90°", math.pi / 2),
                          ("180°", math.pi), ("270°", 3 * math.pi / 2)]:
        tau = ckt.compressor_load_torque("M_compressor", theta, 188.5)
        print(f"    θ = {label:>5} : τ_load = {tau:.4f} N·m")
    print()


def run_transient(ckt: ps.Circuit) -> None:
    # simplify-and-harden-numerical-surface — Phase 2 + Phase 14:
    # use the canonical `from_preset(...)` factory instead of the
    # legacy hand-tune-fields path. `Preset.Fast` is the right pick
    # for this fixed-frequency 60 Hz PSC motor — no high-stiffness
    # dynamics, no nonlinear devices, so the Robust profile would add
    # overhead this circuit doesn't need. See
    # docs/numerical-configuration.md for the preset chooser.
    opts = ps.SimulationOptions.from_preset(ps.Preset.Fast,
                                              dt=5e-5,
                                              tstop=0.20)  # 12 line cycles
    opts.newton_options.num_nodes    = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()

    sim = ps.Simulator(ckt, opts)
    result = sim.run_transient()
    if not result.success:
        raise RuntimeError(f"transient failed: {result.diagnostics}")

    print("=== Transient result at t = 200 ms ===")
    omega = ckt.single_phase_im_omega("M_compressor")
    omega_sync = 2.0 * math.pi * 60.0 / 2  # 4-pole, 60 Hz
    slip = 1.0 - abs(omega) / omega_sync if omega_sync else 0.0
    print(f"  Rotor speed     : {omega:>+8.2f} rad/s  ({omega / (2*math.pi)*60:.1f} RPM)")
    print(f"  Synchronous     : {omega_sync:>8.2f} rad/s  ({omega_sync / (2*math.pi)*60:.1f} RPM)")
    print(f"  Slip            : {slip * 100:>8.2f} %")
    print(f"  Line current    : {ckt.single_phase_im_i_line('M_compressor'):>+8.4f} A (instantaneous)")
    print(f"  Run-cap voltage : {ckt.single_phase_im_V_cap('M_compressor'):>+8.2f} V (instantaneous)")
    print(f"  Shaft torque    : {ckt.single_phase_im_torque('M_compressor'):>+8.4f} N·m (electromagnetic)")
    print()
    print("Note: rotation direction is a PSC wiring choice — real CC")
    print("compressors pick a direction by mechanical winding sense.")


def main() -> None:
    ckt = build_circuit()
    print_analytical(ckt)
    run_transient(ckt)


if __name__ == "__main__":
    main()
