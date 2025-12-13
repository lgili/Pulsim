"""Tests for circuit simulation."""

import pulsim as sl


class TestSimulationOptions:
    """Test simulation options configuration."""

    def test_default_options(self):
        """Default simulation options."""
        opts = sl.SimulationOptions()
        assert opts.tstart == 0.0
        assert opts.dt > 0
        assert opts.abstol > 0
        assert opts.reltol > 0

    def test_custom_options(self):
        """Custom simulation options."""
        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-3
        opts.dt = 1e-6
        opts.dtmax = 10e-6
        opts.abstol = 1e-9
        opts.reltol = 1e-4
        opts.max_newton_iterations = 100
        opts.use_ic = True

        assert opts.tstop == 1e-3
        assert opts.use_ic is True


class TestDCAnalysis:
    """Test DC operating point analysis."""

    def test_resistive_divider(self):
        """Simple resistive voltage divider."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "out", 1000.0)
        circuit.add_resistor("R2", "out", "0", 1000.0)

        sim = sl.Simulator(circuit)
        status, iterations = sim.dc_operating_point()

        assert status == int(sl.SolverStatus.Success)
        # V(out) should be 5V (voltage divider)

    def test_rc_dc(self):
        """RC circuit DC analysis - capacitor is open."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "out", 1000.0)
        circuit.add_capacitor("C1", "out", "0", 1e-6)

        sim = sl.Simulator(circuit)
        status, _ = sim.dc_operating_point()

        assert status == int(sl.SolverStatus.Success)


class TestTransientSimulation:
    """Test transient simulation."""

    def test_rc_step_response(self):
        """RC circuit step response."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "out", 1000.0)
        circuit.add_capacitor("C1", "out", "0", 1e-6, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 5e-3  # 5 time constants
        opts.dt = 1e-6
        opts.dtmax = 10e-6
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert result.final_status == sl.SolverStatus.Success
        assert len(result.time) > 10
        assert result.total_steps > 0

        # Check signal names
        assert "V(in)" in result.signal_names
        assert "V(out)" in result.signal_names

    def test_rl_step_response(self):
        """RL circuit step response."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "out", 100.0)
        circuit.add_inductor("L1", "out", "0", 10e-3, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-3
        opts.dt = 1e-7
        opts.dtmax = 5e-6
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert result.final_status == sl.SolverStatus.Success

    def test_simulation_to_dict(self):
        """Test result conversion to dictionary."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        opts = sl.SimulationOptions()
        opts.tstop = 1e-4
        opts.dt = 1e-6

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        d = result.to_dict()

        assert "time" in d
        assert "signals" in d
        assert "V(in)" in d["signals"]


class TestSimulateFunction:
    """Test the convenience simulate() function."""

    def test_simulate_simple(self):
        """Simple simulation using simulate()."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        opts = sl.SimulationOptions()
        opts.tstop = 1e-4

        result = sl.simulate(circuit, opts)

        assert result.final_status == sl.SolverStatus.Success


class TestPowerLosses:
    """Test power loss tracking."""

    def test_conduction_losses(self):
        """Test conduction loss calculation."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "vcc", "0", 12.0)
        circuit.add_voltage_source("Vctrl", "ctrl", "0", 5.0)  # Switch ON
        circuit.add_resistor("R1", "out", "0", 1.0)

        params = sl.SwitchParams()
        params.ron = 0.1  # 100mOhm
        params.roff = 1e9
        params.vth = 2.5
        params.initial_state = True

        circuit.add_switch("S1", "vcc", "out", "ctrl", "0", params)

        opts = sl.SimulationOptions()
        opts.tstop = 1e-3
        opts.dt = 1e-6

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert result.final_status == sl.SolverStatus.Success

        losses = sim.power_losses()
        assert losses.conduction_loss > 0
        assert losses.total_loss() >= losses.conduction_loss
