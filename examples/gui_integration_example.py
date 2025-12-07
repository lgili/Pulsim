#!/usr/bin/env python3
"""
GUI Integration Example for Pulsim

This example demonstrates how to use Pulsim's GUI integration features:
1. Component metadata for building palettes and property editors
2. Circuit validation with detailed diagnostics
3. Simulation control (pause/resume/stop)
4. Progress callbacks for progress bars
5. Schematic position storage for layout persistence
6. Streaming configuration for memory-efficient long simulations

Run this script to see examples of each feature in action.
"""

import pulsim
import threading
import time
import json


def example_component_metadata():
    """
    Example 1: Component Metadata for GUI Palettes

    Use ComponentRegistry to build component palettes organized by category
    and generate property editors with correct field types.
    """
    print("\n" + "=" * 60)
    print("Example 1: Component Metadata for GUI Palettes")
    print("=" * 60)

    registry = pulsim.ComponentRegistry.instance()

    # List all categories
    print("\nAvailable component categories:")
    for category in registry.all_categories():
        print(f"  - {category}")

    # Build a palette structure
    print("\nComponent palette structure:")
    for category in registry.all_categories():
        print(f"\n  [{category}]")
        for comp_type in registry.types_in_category(category):
            meta = registry.get(comp_type)
            print(f"    - {meta.display_name} ({meta.symbol_id})")

    # Show property editor fields for a resistor
    print("\n\nProperty editor for Resistor:")
    meta = registry.get(pulsim.ComponentType.Resistor)
    print(f"  Description: {meta.description}")
    print(f"  Pins:")
    for pin in meta.pins:
        print(f"    - {pin.name}: {pin.description}")
    print(f"  Parameters:")
    for param in meta.parameters:
        unit_str = f" [{param.unit}]" if param.unit else ""
        required_str = " (required)" if param.required else " (optional)"
        print(f"    - {param.display_name}{unit_str}{required_str}")
        print(f"      Type: {param.type.name}")
        if param.min_value is not None:
            print(f"      Min: {param.min_value}")
        if param.max_value is not None:
            print(f"      Max: {param.max_value}")


def example_circuit_validation():
    """
    Example 2: Circuit Validation with Detailed Diagnostics

    Use validate_circuit() to get structured diagnostics for GUI error display.
    """
    print("\n" + "=" * 60)
    print("Example 2: Circuit Validation with Detailed Diagnostics")
    print("=" * 60)

    # Create a circuit with errors and warnings
    circuit = pulsim.Circuit()

    # This circuit has issues:
    # - No ground reference
    # - Floating node
    circuit.add_resistor("R1", "a", "b", 1000.0)
    circuit.add_resistor("R2", "c", "d", 2000.0)  # Disconnected

    result = pulsim.validate_circuit(circuit)

    print(f"\nValidation result: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"Has errors: {result.has_errors()}")
    print(f"Has warnings: {result.has_warnings()}")

    if result.has_errors():
        print("\nErrors (must be fixed):")
        for error in result.errors():
            print(f"  [{error.code.name}] {error.message}")
            if error.component_name:
                print(f"    Component: {error.component_name}")
            if error.node_name:
                print(f"    Node: {error.node_name}")
            # Get more details about the error code
            desc = pulsim.diagnostic_code_description(error.code)
            print(f"    Help: {desc}")

    if result.has_warnings():
        print("\nWarnings (may cause issues):")
        for warning in result.warnings():
            print(f"  [{warning.code.name}] {warning.message}")

    # Now create a valid circuit
    print("\n--- Creating a valid circuit ---")
    valid_circuit = pulsim.Circuit()
    valid_circuit.add_voltage_source("V1", "in", "0", 5.0)
    valid_circuit.add_resistor("R1", "in", "out", 1000.0)
    valid_circuit.add_resistor("R2", "out", "0", 1000.0)

    result = pulsim.validate_circuit(valid_circuit)
    print(f"Validation result: {'VALID' if result.is_valid else 'INVALID'}")


def example_simulation_control():
    """
    Example 3: Simulation Control (Pause/Resume/Stop)

    Use SimulationController for thread-safe control of long simulations.
    """
    print("\n" + "=" * 60)
    print("Example 3: Simulation Control (Pause/Resume/Stop)")
    print("=" * 60)

    # Create a circuit for longer simulation
    circuit = pulsim.Circuit()
    circuit.add_voltage_source("V1", "in", "0", 5.0)
    circuit.add_resistor("R1", "in", "out", 1000.0)
    circuit.add_capacitor("C1", "out", "0", 1e-6)  # 1uF

    # Create controller
    controller = pulsim.SimulationController()

    print(f"\nInitial state: {controller.state.name}")

    # Setup simulation options for longer run
    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 0.01  # 10ms
    opts.dt = 1e-7     # 100ns timestep

    # Track progress updates
    progress_updates = []

    def progress_callback(progress):
        progress_updates.append(progress.progress_percent)
        # In a real GUI, you would update the progress bar here
        if len(progress_updates) % 10 == 0:
            print(f"  Progress: {progress.progress_percent:.1f}% "
                  f"(t={progress.current_time*1000:.3f}ms, "
                  f"steps={progress.steps_completed})")

    # Run simulation in a thread (like you would in a GUI)
    sim = pulsim.Simulator(circuit, opts)

    def run_simulation():
        sim.run_transient_with_progress(
            callback=None,
            event_callback=None,
            control=controller,
            progress_callback=progress_callback,
            min_interval_ms=50,  # Update every 50ms
            min_steps=100
        )

    print("\nStarting simulation in background thread...")
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.start()

    # Wait a bit, then pause
    time.sleep(0.1)
    if controller.is_running():
        print("\nRequesting pause...")
        controller.request_pause()
        time.sleep(0.05)
        print(f"State after pause request: {controller.state.name}")

    # Resume after a brief pause
    time.sleep(0.1)
    if controller.is_paused():
        print("\nRequesting resume...")
        controller.request_resume()

    # Wait for completion
    sim_thread.join(timeout=5.0)

    print(f"\nFinal state: {controller.state.name}")
    print(f"Total progress updates received: {len(progress_updates)}")


def example_progress_callbacks():
    """
    Example 4: Progress Callbacks for Progress Bars

    Configure progress callbacks to update GUI progress indicators.
    """
    print("\n" + "=" * 60)
    print("Example 4: Progress Callbacks for Progress Bars")
    print("=" * 60)

    circuit = pulsim.Circuit()
    circuit.add_voltage_source("V1", "in", "0", 5.0)
    circuit.add_resistor("R1", "in", "out", 1000.0)
    circuit.add_capacitor("C1", "out", "0", 1e-6)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 0.005  # 5ms
    opts.dt = 1e-7

    # Configure progress callback throttling in options
    opts.progress_min_interval_ms = 100   # Max 10 updates/second
    opts.progress_min_steps = 50          # At least 50 steps between updates
    opts.progress_include_memory = False  # Don't track memory (faster)

    print("\nProgress callback configuration:")
    print(f"  Min interval: {opts.progress_min_interval_ms}ms")
    print(f"  Min steps: {opts.progress_min_steps}")
    print(f"  Track memory: {opts.progress_include_memory}")

    progress_data = []

    def detailed_progress_callback(p):
        progress_data.append({
            'percent': p.progress_percent,
            'time': p.current_time,
            'steps': p.steps_completed,
            'newton_iters': p.newton_iterations,
            'convergence_warning': p.convergence_warning,
            'elapsed': p.elapsed_seconds,
            'remaining': p.estimated_remaining_seconds
        })

    sim = pulsim.Simulator(circuit, opts)
    result = sim.run_transient_with_progress(
        callback=None,
        event_callback=None,
        control=None,
        progress_callback=detailed_progress_callback,
        min_interval_ms=100,
        min_steps=50
    )

    print(f"\nSimulation completed!")
    print(f"Total progress callbacks: {len(progress_data)}")

    if progress_data:
        print("\nSample progress data:")
        for i, p in enumerate(progress_data[::max(1, len(progress_data)//5)]):
            print(f"  {p['percent']:5.1f}% | t={p['time']*1000:.2f}ms | "
                  f"steps={p['steps']:6d} | elapsed={p['elapsed']:.3f}s")


def example_schematic_positions():
    """
    Example 5: Schematic Position Storage

    Store and retrieve component positions for GUI layout persistence.
    """
    print("\n" + "=" * 60)
    print("Example 5: Schematic Position Storage")
    print("=" * 60)

    # Create circuit and set positions
    circuit = pulsim.Circuit()
    circuit.add_voltage_source("V1", "in", "0", 5.0)
    circuit.add_resistor("R1", "in", "out", 1000.0)
    circuit.add_capacitor("C1", "out", "0", 1e-6)

    # Set positions (as if placed by user in schematic editor)
    circuit.set_position("V1", pulsim.SchematicPosition(x=50, y=100, orientation=0))
    circuit.set_position("R1", pulsim.SchematicPosition(x=150, y=100, orientation=0))
    circuit.set_position("C1", pulsim.SchematicPosition(x=250, y=150, orientation=90))

    print("\nComponent positions set:")
    for name in ["V1", "R1", "C1"]:
        pos = circuit.get_position(name)
        if pos:
            print(f"  {name}: x={pos.x}, y={pos.y}, rotation={pos.orientation}deg")

    # Export to JSON (includes positions)
    json_str = pulsim.circuit_to_json(circuit, include_positions=True)

    print("\nJSON export (with positions):")
    json_obj = json.loads(json_str)
    print(json.dumps(json_obj, indent=2)[:500] + "...")

    # Import from JSON
    loaded_circuit = pulsim.parse_netlist_string(json_str)

    print("\nPositions after import:")
    for name in ["V1", "R1", "C1"]:
        if loaded_circuit.has_position(name):
            pos = loaded_circuit.get_position(name)
            print(f"  {name}: x={pos.x}, y={pos.y}, rotation={pos.orientation}deg")
        else:
            print(f"  {name}: no position")


def example_streaming_configuration():
    """
    Example 6: Streaming Configuration for Long Simulations

    Configure memory-efficient result storage for long simulations.
    """
    print("\n" + "=" * 60)
    print("Example 6: Streaming Configuration for Long Simulations")
    print("=" * 60)

    circuit = pulsim.Circuit()
    circuit.add_voltage_source("V1", "in", "0", 5.0)
    circuit.add_resistor("R1", "in", "out", 1000.0)
    circuit.add_capacitor("C1", "out", "0", 1e-6)

    # Without streaming: many points
    opts_full = pulsim.SimulationOptions()
    opts_full.tstart = 0.0
    opts_full.tstop = 0.01  # 10ms
    opts_full.dt = 1e-7
    opts_full.streaming_decimation = 1  # Store all points

    sim_full = pulsim.Simulator(circuit, opts_full)
    result_full = sim_full.run_transient()

    print(f"\nWithout decimation:")
    print(f"  Points stored: {result_full.num_points()}")

    # With decimation: fewer points
    opts_decimated = pulsim.SimulationOptions()
    opts_decimated.tstart = 0.0
    opts_decimated.tstop = 0.01
    opts_decimated.dt = 1e-7
    opts_decimated.streaming_decimation = 100  # Store every 100th point

    sim_decimated = pulsim.Simulator(circuit, opts_decimated)
    result_decimated = sim_decimated.run_transient()

    print(f"\nWith decimation (every 100th point):")
    print(f"  Points stored: {result_decimated.num_points()}")

    # With rolling buffer
    opts_rolling = pulsim.SimulationOptions()
    opts_rolling.tstart = 0.0
    opts_rolling.tstop = 0.01
    opts_rolling.dt = 1e-7
    opts_rolling.streaming_rolling_buffer = True
    opts_rolling.streaming_max_points = 1000

    sim_rolling = pulsim.Simulator(circuit, opts_rolling)
    result_rolling = sim_rolling.run_transient()

    print(f"\nWith rolling buffer (max 1000 points):")
    print(f"  Points stored: {result_rolling.num_points()}")

    print("\nStreaming is useful for:")
    print("  - Long simulations where you only need recent data")
    print("  - Memory-constrained environments")
    print("  - Real-time displays that show a time window")


def example_enhanced_results():
    """
    Example 7: Enhanced Simulation Results

    Access detailed signal info, solver settings, and events from results.
    """
    print("\n" + "=" * 60)
    print("Example 7: Enhanced Simulation Results")
    print("=" * 60)

    circuit = pulsim.Circuit()
    circuit.add_voltage_source("V1", "in", "0", 5.0)
    circuit.add_resistor("R1", "in", "out", 1000.0)
    circuit.add_capacitor("C1", "out", "0", 1e-6)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 0.005
    opts.dt = 1e-7
    opts.integration_method = pulsim.IntegrationMethod.Trapezoidal

    sim = pulsim.Simulator(circuit, opts)
    result = sim.run_transient()

    print("\nSignal information (for plot labels and axes):")
    for info in result.signal_info:
        print(f"  - {info.name}: type={info.type}, unit={info.unit}")

    print("\nSolver information:")
    print(f"  Method: {result.solver_info.method}")
    print(f"  Abstol: {result.solver_info.abstol}")
    print(f"  Reltol: {result.solver_info.reltol}")
    print(f"  Adaptive: {result.solver_info.adaptive_timestep}")

    print("\nPerformance metrics:")
    print(f"  Total steps: {result.total_steps}")
    print(f"  Avg Newton iterations: {result.average_newton_iterations:.2f}")
    print(f"  Convergence failures: {result.convergence_failures}")
    print(f"  Timestep reductions: {result.timestep_reductions}")

    print(f"\nEvents recorded: {result.num_events()}")
    if result.num_events() > 0:
        print("Sample events:")
        for event in result.events[:5]:
            print(f"  t={event.time:.6f}s: {event.description}")


def main():
    """Run all GUI integration examples."""
    print("=" * 60)
    print("Pulsim GUI Integration Examples")
    print("=" * 60)
    print("\nThis script demonstrates GUI integration features.")
    print("Each example shows how to use a specific feature.")

    example_component_metadata()
    example_circuit_validation()
    example_simulation_control()
    example_progress_callbacks()
    example_schematic_positions()
    example_streaming_configuration()
    example_enhanced_results()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
