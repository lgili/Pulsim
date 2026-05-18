"""Schematic stress test: Boost PFC → 3-phase VSI → PMSM motor.

Builds an off-line motor drive front-to-back so the netlistsvg backend has
to lay out an AC source + bridge rectifier + boost stage + DC bus + 6-switch
inverter + 3-phase motor on the same canvas. Used to eyeball whether the
renderer survives a topology with cycles + parallel branches + macros that
expand into many atomic devices.

Run with:
    PYTHONPATH=build/python python3 scripts/render_pfc_vsi_pmsm.py
"""

from __future__ import annotations

from pathlib import Path

import pulsim as ps


def build_drive() -> ps.Circuit:
    ckt = ps.Circuit()

    # Input section
    vac_a    = ckt.add_node("vac_a")
    vac_b    = ckt.add_node("vac_b")
    rect_pos = ckt.add_node("rect_pos")
    # Boost stage
    sw_node  = ckt.add_node("sw_node")
    vbus     = ckt.add_node("vbus")
    gate_pwm = ckt.add_node("gate_pwm")
    # 3-phase output
    motor_a  = ckt.add_node("motor_a")
    motor_b  = ckt.add_node("motor_b")
    motor_c  = ckt.add_node("motor_c")
    motor_n  = ckt.add_node("motor_n")
    gnd      = ckt.ground()

    # --- AC mains: 230 Vpk, 50 Hz ---
    sine = ps.SineParams()
    sine.amplitude = 230.0
    sine.frequency = 50.0
    ckt.add_sine_voltage_source("Vac", vac_a, vac_b, sine)

    # --- Single-phase bridge rectifier (expands to 4 diodes DBR__D1..D4) ---
    ckt.add_bridge_rectifier("DBR", vac_a, vac_b, rect_pos, gnd, 0.7, 0.01)

    # --- Boost stage (L + MOSFET + diode + bulk cap) ---
    ckt.add_inductor("L_pfc", rect_pos, sw_node, 1e-3, 0.0)
    ckt.add_mosfet("Q_pfc", gate_pwm, sw_node, gnd)
    ckt.add_diode("D_pfc", sw_node, vbus)
    ckt.add_capacitor("C_bus", vbus, gnd, 470e-6, 400.0)

    # Gate driver for the PFC switch (50 kHz, 50 % duty)
    pulse = ps.PulseParams()
    pulse.v_initial = 0.0
    pulse.v_pulse   = 12.0
    pulse.t_rise    = 50e-9
    pulse.t_fall    = 50e-9
    pulse.t_width   = 10e-6
    pulse.period    = 20e-6
    ckt.add_pulse_voltage_source("V_pfc_gate", gate_pwm, gnd, pulse)

    # --- 3-phase 2-level VSI (expands to 6 MOSFETs + 6 SPWM gate sources) ---
    # f_sw = 10 kHz, m = 0.85, f_mod = 60 Hz electrical output
    ckt.add_three_phase_vsi(
        "VSI", vbus, gnd, motor_a, motor_b, motor_c, 10_000.0, 0.85, 60.0
    )

    # --- 3-phase dynamic PMSM (4-pole, 1.5 kW class) ---
    # Rs = 0.5 Ω, Ld = Lq = 5 mH, psi_pm = 0.1 Wb, p = 2 pole pairs,
    # J = 1e-3 kg·m², b_friction = 1e-4 N·m·s/rad.
    ckt.add_pmsm(
        "PMSM", motor_a, motor_b, motor_c, motor_n,
        0.5, 5e-3, 5e-3, 0.1, 2, 1e-3, 1e-4
    )

    return ckt


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "build" / "schematic_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckt = build_drive()

    print("=" * 64)
    print("Boost PFC + 3-phase VSI + PMSM motor drive — schematic demo")
    print("=" * 64)
    print(f"components: {ckt.num_components()}")
    print(f"nodes:      {ckt.num_nodes()}")
    print()
    print("Top-level component kinds:")
    kinds: dict[str, int] = {}
    for desc in ckt.components():
        kinds[desc.kind] = kinds.get(desc.kind, 0) + 1
    for kind, count in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {kind:24} x {count}")
    print()

    svg_path = out_dir / "pfc_vsi_pmsm.svg"
    png_path = out_dir / "pfc_vsi_pmsm.png"
    ps.schematic.render(ckt, svg_path)
    ps.schematic.render(ckt, png_path)
    print(f"SVG: {svg_path}  ({svg_path.stat().st_size} bytes)")
    print(f"PNG: {png_path}  ({png_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
