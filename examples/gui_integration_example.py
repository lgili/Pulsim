#!/usr/bin/env python3
"""GUI integration example using the current Pulsim Python API.

This script demonstrates:
1) Building a circuit from GUI-style metadata
2) Strict YAML parser loading for frontend payload validation
3) Transient simulation via SimulationOptions + Simulator
4) Accessing mixed-domain virtual component channels for plotting
"""

from __future__ import annotations

import textwrap

import pulsim as ps


def build_gui_style_circuit() -> ps.Circuit:
    """Build a buck-like test circuit the same way a GUI backend adapter would."""
    ckt = ps.Circuit()

    n_in = ckt.add_node("vin")
    n_out = ckt.add_node("vout")
    gnd = ckt.ground()

    # Electrical stage (stable PWM-fed RC load)
    # For GUI smoke/integration this is preferred over a hard-switching nonlinear stage.
    ckt.add_resistor("Rin", n_in, n_out, 2.2)
    ckt.add_capacitor("C1", n_out, gnd, 220e-6, 0.0)
    ckt.add_resistor("Rload", n_out, gnd, 8.0)

    # Frontend payload can still map this to time-varying sources;
    # keep the smoke path deterministic and robust for CI.
    ckt.add_voltage_source("Vin", n_in, gnd, 10.0)

    # GUI-side control/instrumentation metadata (mixed-domain virtual components)
    ckt.add_virtual_component(
        "voltage_probe",
        "probe_vout",
        [n_out, gnd],
        {},
        {"label": "Vout", "unit": "V"},
    )
    ckt.add_virtual_component(
        "current_probe",
        "probe_iin",
        [n_in, gnd],
        {},
        {"label": "iin", "target_component": "Vin"},
    )

    return ckt


def run_transient(ckt: ps.Circuit) -> ps.SimulationResult:
    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 2.5e-3
    opts.dt = 2e-7
    opts.dt_min = 5e-9
    opts.dt_max = 2e-6
    opts.step_mode = ps.StepMode.Fixed
    opts.integrator = ps.Integrator.Trapezoidal
    opts.enable_events = False

    # Keep GUI backend path explicit (native runtime as default)
    opts.transient_backend = ps.TransientBackendMode.Native

    sim = ps.Simulator(ckt, opts)
    return sim.run_transient(ckt.initial_state())


def strict_yaml_parse_demo() -> tuple[ps.Circuit, ps.SimulationOptions]:
    """Validate GUI-generated YAML payload using strict parser mode."""
    yaml_text = textwrap.dedent(
        """
        schema: pulsim-v1
        version: 1
        components:
          - type: voltage_source
            name: Vin
            nodes: [vin, 0]
            value: 24
          - type: resistor
            name: Rload
            nodes: [vout, 0]
            value: 8
          - type: capacitor
            name: Cout
            nodes: [vout, 0]
            value: 220u
            ic: 0
        simulation:
          t_start: 0
          t_stop: 500u
          dt: 1u
          step_mode: fixed
        """
    ).strip()

    parser_opts = ps.YamlParserOptions()
    parser_opts.strict = True
    parser = ps.YamlParser(parser_opts)
    ckt, sim_opts = parser.load_string(yaml_text)

    if parser.warnings:
        print("YAML warnings:")
        for w in parser.warnings:
            print("  -", w)

    return ckt, sim_opts


def print_summary(result: ps.SimulationResult, ckt: ps.Circuit) -> None:
    print("\n=== Simulation summary ===")
    print("success:", result.success)
    print("message:", result.message)
    print("steps:", result.total_steps)
    print("newton_total:", result.newton_iterations_total)
    print("events:", len(result.events))
    print("time_points:", len(result.time))

    if result.time and result.states:
        names = ckt.signal_names()
        vout_idx = names.index("v(vout)") if "v(vout)" in names else None
        if vout_idx is not None:
            print("vout_final:", result.states[-1][vout_idx])

    if result.virtual_channels:
        print("virtual channels:")
        for ch, values in result.virtual_channels.items():
            tail = values[-1] if values else None
            print(f"  - {ch}: samples={len(values)} last={tail}")

    bt = result.backend_telemetry
    print("backend:", bt.selected_backend, "family:", bt.solver_family, "form:", bt.formulation_mode)
    print("backend_used_sundials:", bt.sundials_used)


def main() -> None:
    print("Pulsim version:", ps.__version__)
    print("Backend capabilities:", ps.backend_capabilities())

    ckt = build_gui_style_circuit()
    result = run_transient(ckt)
    print_summary(result, ckt)

    ckt_yaml, sim_opts_yaml = strict_yaml_parse_demo()
    sim_yaml = ps.Simulator(ckt_yaml, sim_opts_yaml)
    res_yaml = sim_yaml.run_transient(ckt_yaml.initial_state())
    print("\n=== Strict YAML smoke ===")
    print("success:", res_yaml.success, "steps:", res_yaml.total_steps)


if __name__ == "__main__":
    main()
