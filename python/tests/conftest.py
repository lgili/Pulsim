"""Pytest configuration and fixtures for Pulsim tests."""

import pytest
import sys
import os

# Add the build directory to path if running tests before installation
# This allows testing the compiled module directly
build_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build'),
]
for path in build_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path) and abs_path not in sys.path:
        sys.path.insert(0, abs_path)


@pytest.fixture
def simple_rc_circuit():
    """Create a simple RC circuit for testing."""
    import pulsim as sl

    circuit = sl.Circuit()
    circuit.add_voltage_source("V1", "in", "0", 5.0)
    circuit.add_resistor("R1", "in", "out", 1000.0)
    circuit.add_capacitor("C1", "out", "0", 1e-6, ic=0.0)
    return circuit


@pytest.fixture
def buck_converter_circuit():
    """Create a buck converter circuit for testing."""
    import pulsim as sl

    circuit = sl.Circuit()

    # Input voltage source
    circuit.add_voltage_source("Vdc", "vcc", "0", 48.0)

    # PWM control signal
    circuit.add_voltage_source("Vctrl", "ctrl", "0", 5.0)

    # High-side switch
    sw_params = sl.SwitchParams()
    sw_params.ron = 0.01
    sw_params.roff = 1e9
    sw_params.vth = 2.5
    circuit.add_switch("S1", "vcc", "sw", "ctrl", "0", sw_params)

    # Freewheeling diode
    diode_params = sl.DiodeParams()
    diode_params.ideal = True
    circuit.add_diode("D1", "0", "sw", diode_params)

    # LC filter
    circuit.add_inductor("L1", "sw", "out", 100e-6)
    circuit.add_capacitor("C1", "out", "0", 100e-6)

    # Load
    circuit.add_resistor("Rload", "out", "0", 10.0)

    return circuit


@pytest.fixture
def simulation_options():
    """Default simulation options for testing."""
    import pulsim as sl

    opts = sl.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 1e-3
    opts.dt = 1e-6
    opts.dtmax = 10e-6
    opts.abstol = 1e-12
    opts.reltol = 1e-3
    opts.use_ic = True
    return opts
