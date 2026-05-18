"""Validate the Phase 3 schematic renderer on a Boost PFC topology.

Build a representative Boost PFC converter (AC mains → bridge rectifier →
boost stage → DC bus) entirely in Python and render it to SVG + PNG. The
SVG is what the user inspects to decide if the auto-layout MVP is good
enough to merge.

Run with:
    PYTHONPATH=build/python python3 scripts/render_boost_pfc.py
"""

from __future__ import annotations

from pathlib import Path

import pulsim as ps


def build_boost_pfc() -> ps.Circuit:
    """A single-phase boost PFC.

    Topology:
        Vac (sine 230 Vpk / 50 Hz) ─ across (vac_a, vac_b)
        Bridge rectifier            (vac_a, vac_b) → (rect_pos, gnd)
        Boost inductor L            rect_pos → sw_node
        MOSFET Q                    drain=sw_node, source=gnd, gate=gate_pwm
        Boost diode D               anode=sw_node, cathode=vout
        Output cap C                vout → gnd
        Load R                      vout → gnd
        Gate PWM (50 kHz, 50 %)     gate_pwm → gnd
    """
    ckt = ps.Circuit()

    vac_a    = ckt.add_node("vac_a")
    vac_b    = ckt.add_node("vac_b")
    rect_pos = ckt.add_node("rect_pos")
    sw_node  = ckt.add_node("sw_node")
    vout     = ckt.add_node("vout")
    gate_pwm = ckt.add_node("gate_pwm")
    gnd      = ckt.ground()

    # AC mains (230 Vpk sine, 50 Hz). Negative terminal returns to vac_b
    # (floating relative to ground — that's typical for an off-line PFC
    # input section before the bridge).
    sine = ps.SineParams()
    sine.amplitude = 230.0
    sine.frequency = 50.0
    sine.offset = 0.0
    sine.phase = 0.0
    ckt.add_sine_voltage_source("Vac", vac_a, vac_b, sine)

    # Full-wave bridge rectifier: dc_neg pinned to ground so the boost
    # stage's source rail sits at gnd. V_F0 / R_d are typical Si bridge
    # diode values (0.7 V threshold, 10 mOhm series). Internally this
    # macro expands to 4 atomic diodes named DBR__D1..D4.
    ckt.add_bridge_rectifier("DBR", vac_a, vac_b, rect_pos, gnd, 0.7, 0.01)

    # Boost stage proper.
    ckt.add_inductor("L1", rect_pos, sw_node, 1e-3, 0.0)         # 1 mH boost L
    ckt.add_mosfet("Q1", gate_pwm, sw_node, gnd)                  # NMOS, defaults
    ckt.add_diode("D1", sw_node, vout)                            # boost diode

    # Output filter + load.
    ckt.add_capacitor("Cout", vout, gnd, 470e-6, 0.0)             # 470 uF bulk cap
    ckt.add_resistor("Rload", vout, gnd, 100.0)                   # 100 Ohm load

    # Gate driver — 50 kHz, 50 % duty.
    pulse = ps.PulseParams()
    pulse.v_initial = 0.0
    pulse.v_pulse   = 12.0
    pulse.t_delay   = 0.0
    pulse.t_rise    = 50e-9
    pulse.t_fall    = 50e-9
    pulse.t_width   = 10e-6      # 50 % @ 20 us period = 50 kHz
    pulse.period    = 20e-6
    ckt.add_pulse_voltage_source("Vgate", gate_pwm, gnd, pulse)

    return ckt


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "build" / "schematic_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckt = build_boost_pfc()
    layout = ps.schematic.compute_layout(ckt)

    print("=" * 64)
    print("Boost PFC schematic demo")
    print("=" * 64)
    print(f"components: {len(layout.components)}")
    print(f"wires:      {len(layout.wires)}")
    print(f"junctions:  {len(layout.junctions)}")
    print(f"canvas:     {layout.canvas.width:.1f} x {layout.canvas.height:.1f} {layout.canvas.unit}")
    print()
    print("Placements:")
    for name, p in layout.components.items():
        print(f"  {name:6} {p.kind:18} center=({p.x:7.1f}, {p.y:7.1f}) rot={p.rotation:3d}°")
    print()

    svg_path = out_dir / "boost_pfc.svg"
    png_path = out_dir / "boost_pfc.png"
    ps.schematic.render_layout(layout, svg_path)
    ps.schematic.render_layout(layout, png_path)
    print(f"SVG: {svg_path}  ({svg_path.stat().st_size} bytes)")
    print(f"PNG: {png_path}  ({png_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
