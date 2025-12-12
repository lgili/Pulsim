"""Tests for PulsimCore High-Performance API."""

import sys
import math
import os
import glob
import pytest

# Ensure build path is FIRST in sys.path before importing pulsim
# This is necessary because pytest may add python/ to path, which has pulsim source
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
build_patterns = [
    os.path.join(project_root, 'build', 'cp*', 'python'),
    os.path.join(project_root, 'build', 'python'),
]

# Find the build path
build_path = None
for pattern in build_patterns:
    for path in glob.glob(pattern):
        if os.path.exists(path):
            build_path = path
            break
    if build_path:
        break

# Remove any entries that might shadow the build path (like python/)
if build_path:
    source_python = os.path.join(project_root, 'python')
    if source_python in sys.path:
        sys.path.remove(source_python)
    # Add build path at the front
    if build_path in sys.path:
        sys.path.remove(build_path)
    sys.path.insert(0, build_path)

# Remove any cached pulsim modules to ensure we get the right one
for mod in list(sys.modules.keys()):
    if mod.startswith('pulsim'):
        del sys.modules[mod]

# Import pulsim
try:
    import pulsim as ps
except ImportError:
    ps = None

# Skip all tests if module not available
if ps is None:
    pytest.skip("pulsim module not available", allow_module_level=True)


class TestDevices:
    """Test device creation and properties."""

    def test_resistor(self):
        """Test Resistor device."""
        r = ps.Resistor(1000.0, "R1")
        assert r.resistance() == 1000.0
        assert r.name() == "R1"

    def test_resistor_default_name(self):
        """Test Resistor with default name."""
        r = ps.Resistor(500.0)
        assert r.resistance() == 500.0
        assert r.name() == ""

    def test_capacitor(self):
        """Test Capacitor device."""
        c = ps.Capacitor(1e-6, 0.0, "C1")
        assert c.capacitance() == 1e-6
        assert c.name() == "C1"

    def test_capacitor_with_initial_voltage(self):
        """Test Capacitor with initial voltage."""
        c = ps.Capacitor(100e-9, 5.0, "C2")
        assert c.capacitance() == 100e-9
        assert c.name() == "C2"

    def test_inductor(self):
        """Test Inductor device."""
        l = ps.Inductor(1e-3, 0.0, "L1")
        assert l.inductance() == 1e-3
        assert l.name() == "L1"

    def test_voltage_source(self):
        """Test VoltageSource device."""
        vs = ps.VoltageSource(12.0, "V1")
        assert vs.voltage() == 12.0
        assert vs.name() == "V1"

    def test_current_source(self):
        """Test CurrentSource device."""
        cs = ps.CurrentSource(0.001, "I1")
        assert cs.current() == 0.001
        assert cs.name() == "I1"


class TestEnums:
    """Test enumerations."""

    def test_device_type(self):
        """Test DeviceType enum."""
        assert ps.DeviceType.Resistor is not None
        assert ps.DeviceType.Capacitor is not None
        assert ps.DeviceType.Inductor is not None
        assert ps.DeviceType.VoltageSource is not None
        assert ps.DeviceType.Diode is not None
        assert ps.DeviceType.MOSFET is not None

    def test_solver_status(self):
        """Test SolverStatus enum."""
        assert ps.SolverStatus.Success is not None
        assert ps.SolverStatus.MaxIterationsReached is not None
        assert ps.SolverStatus.SingularMatrix is not None

    def test_dc_strategy(self):
        """Test DCStrategy enum."""
        assert ps.DCStrategy.Direct is not None
        assert ps.DCStrategy.GminStepping is not None
        assert ps.DCStrategy.SourceStepping is not None
        assert ps.DCStrategy.PseudoTransient is not None
        assert ps.DCStrategy.Auto is not None

    def test_rlc_damping(self):
        """Test RLCDamping enum."""
        assert ps.RLCDamping.Underdamped is not None
        assert ps.RLCDamping.Critical is not None
        assert ps.RLCDamping.Overdamped is not None

    def test_simd_level(self):
        """Test SIMDLevel enum."""
        assert ps.SIMDLevel.None_ is not None
        assert ps.SIMDLevel.SSE2 is not None
        assert ps.SIMDLevel.AVX is not None
        assert ps.SIMDLevel.AVX512 is not None
        assert ps.SIMDLevel.NEON is not None


class TestConfiguration:
    """Test configuration classes."""

    def test_tolerances_defaults(self):
        """Test default tolerances."""
        tols = ps.Tolerances.defaults()
        assert tols.voltage_abstol > 0
        assert tols.current_abstol > 0
        assert tols.residual_tol > 0

    def test_tolerances_modifiable(self):
        """Test tolerance modification."""
        tols = ps.Tolerances()
        tols.voltage_abstol = 1e-6
        assert tols.voltage_abstol == 1e-6

    def test_newton_options(self):
        """Test Newton solver options."""
        opts = ps.NewtonOptions()
        assert opts.max_iterations > 0
        assert opts.initial_damping > 0
        opts.max_iterations = 100
        assert opts.max_iterations == 100

    def test_gmin_config(self):
        """Test Gmin stepping configuration."""
        cfg = ps.GminConfig()
        assert cfg.initial_gmin > 0
        assert cfg.final_gmin < cfg.initial_gmin
        # reduction_factor > 1 because we divide by it to reduce gmin
        assert cfg.reduction_factor > 1.0
        assert cfg.required_steps() > 0

    def test_source_stepping_config(self):
        """Test source stepping configuration."""
        cfg = ps.SourceSteppingConfig()
        assert cfg.initial_scale >= 0
        assert cfg.final_scale <= 1.0
        assert cfg.max_steps > 0

    def test_pseudo_transient_config(self):
        """Test pseudo-transient configuration."""
        cfg = ps.PseudoTransientConfig()
        assert cfg.initial_dt > 0
        assert cfg.max_dt >= cfg.initial_dt
        assert cfg.min_dt <= cfg.initial_dt

    def test_dc_convergence_config(self):
        """Test DC convergence configuration."""
        cfg = ps.DCConvergenceConfig()
        assert cfg.strategy == ps.DCStrategy.Auto
        cfg.strategy = ps.DCStrategy.GminStepping
        assert cfg.strategy == ps.DCStrategy.GminStepping


class TestIntegrationConfig:
    """Test integration method configuration."""

    def test_bdf_order_config(self):
        """Test BDF order configuration."""
        cfg = ps.BDFOrderConfig()
        assert cfg.min_order >= 1
        assert cfg.max_order >= cfg.min_order
        assert cfg.initial_order >= cfg.min_order
        assert cfg.initial_order <= cfg.max_order

    def test_timestep_config_defaults(self):
        """Test default timestep configuration."""
        cfg = ps.TimestepConfig.defaults()
        assert cfg.dt_min > 0
        assert cfg.dt_max > cfg.dt_min
        assert cfg.safety_factor > 0
        assert cfg.safety_factor <= 1.0

    def test_timestep_config_conservative(self):
        """Test conservative timestep configuration."""
        cfg = ps.TimestepConfig.conservative()
        default = ps.TimestepConfig.defaults()
        assert cfg.safety_factor <= default.safety_factor

    def test_timestep_config_aggressive(self):
        """Test aggressive timestep configuration."""
        cfg = ps.TimestepConfig.aggressive()
        default = ps.TimestepConfig.defaults()
        assert cfg.safety_factor >= default.safety_factor


class TestAnalyticalSolutions:
    """Test analytical solution classes for validation."""

    def test_rc_analytical(self):
        """Test RC circuit analytical solution."""
        # 1k Ohm, 1uF capacitor, 0V to 5V step
        rc = ps.RCAnalytical(1000, 1e-6, 0.0, 5.0)
        tau = rc.tau()
        assert abs(tau - 1e-3) < 1e-9  # tau = R*C = 1ms

        # At t=0, voltage should be initial
        assert abs(rc.voltage(0.0) - 0.0) < 1e-9

        # At t=tau, voltage should be ~63.2% of final
        v_tau = rc.voltage(tau)
        expected = 5.0 * (1 - math.exp(-1))  # ~3.16V
        assert abs(v_tau - expected) < 0.01

        # At t=5*tau, voltage should be ~99.3% of final
        v_5tau = rc.voltage(5 * tau)
        assert abs(v_5tau - 5.0) < 0.05

    def test_rl_analytical(self):
        """Test RL circuit analytical solution."""
        # 1k Ohm, 1mH inductor, 10V source
        rl = ps.RLAnalytical(1000, 1e-3, 10.0, 0.0)
        tau = rl.tau()
        assert abs(tau - 1e-6) < 1e-12  # tau = L/R = 1us

        # Final current should be V/R
        i_final = rl.I_final()
        assert abs(i_final - 0.01) < 1e-9  # 10V / 1kOhm = 10mA

    def test_rlc_underdamped(self):
        """Test underdamped RLC circuit."""
        # For underdamped: zeta = R/(2*sqrt(L/C)) < 1
        # With R=10, L=1mH, C=1uF: zeta = 10/(2*sqrt(1e-3/1e-6)) = 10/(2*31.6) = 0.158 < 1
        rlc = ps.RLCAnalytical(10, 1e-3, 1e-6, 10.0, 0.0, 0.0)
        assert rlc.damping_type() == ps.RLCDamping.Underdamped

        omega_0 = rlc.omega_0()
        expected_omega_0 = 1.0 / math.sqrt(1e-3 * 1e-6)  # ~31623 rad/s
        assert abs(omega_0 - expected_omega_0) < 1

    def test_rlc_overdamped(self):
        """Test overdamped RLC circuit."""
        # R=1000, L=1mH, C=1uF -> overdamped
        rlc = ps.RLCAnalytical(1000, 1e-3, 1e-6, 10.0, 0.0, 0.0)
        assert rlc.damping_type() == ps.RLCDamping.Overdamped

    def test_waveform_generation(self):
        """Test waveform generation from analytical."""
        rc = ps.RCAnalytical(1000, 1e-6, 0.0, 5.0)
        waveform = rc.waveform(0.0, 5e-3, 1e-4)  # 0 to 5ms, 100us step
        assert len(waveform) > 0


class TestValidation:
    """Test validation framework."""

    def test_compare_waveforms_identical(self):
        """Test waveform comparison with identical data."""
        # Format: list of (time, value) tuples
        sim = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0), (0.3, 4.0), (0.4, 5.0)]
        ana = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0), (0.3, 4.0), (0.4, 5.0)]
        result = ps.compare_waveforms("test", sim, ana)
        assert result.passed
        assert result.max_error < 1e-10

    def test_compare_waveforms_small_error(self):
        """Test waveform comparison with small error."""
        sim = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0), (0.3, 4.0), (0.4, 5.0)]
        ana = [(0.0, 1.001), (0.1, 2.001), (0.2, 3.001), (0.3, 4.001), (0.4, 5.001)]
        result = ps.compare_waveforms("test", sim, ana, 0.01)  # 1% threshold
        assert result.passed

    def test_compare_waveforms_large_error(self):
        """Test waveform comparison with large error."""
        sim = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0), (0.3, 4.0), (0.4, 5.0)]
        ana = [(0.0, 1.5), (0.1, 2.5), (0.2, 3.5), (0.3, 4.5), (0.4, 5.5)]
        result = ps.compare_waveforms("test", sim, ana, 0.01)  # 1% threshold
        assert not result.passed


class TestHighPerformance:
    """Test high-performance features."""

    def test_simd_detection(self):
        """Test SIMD level detection."""
        level = ps.detect_simd_level()
        assert isinstance(level, ps.SIMDLevel)

    def test_simd_vector_width(self):
        """Test SIMD vector width."""
        width = ps.simd_vector_width()
        assert width >= 1
        assert width <= 8  # Max for AVX512 doubles

    def test_linear_solver_config(self):
        """Test linear solver configuration."""
        cfg = ps.LinearSolverConfig()
        assert cfg.pivot_tolerance > 0
        cfg.reuse_symbolic = True
        assert cfg.reuse_symbolic


class TestBenchmark:
    """Test benchmark framework."""

    def test_benchmark_timing(self):
        """Test benchmark timing structure."""
        timing = ps.BenchmarkTiming()
        timing.name = "test"
        timing.iterations = 100
        assert timing.name == "test"
        assert timing.iterations == 100

    def test_benchmark_result(self):
        """Test benchmark result structure."""
        result = ps.BenchmarkResult()
        result.circuit_name = "test_circuit"
        result.num_nodes = 10
        result.num_devices = 5
        assert result.circuit_name == "test_circuit"
        assert result.num_nodes == 10


class TestExport:
    """Test export functions."""

    def test_export_validation_csv(self):
        """Test CSV export of validation results."""
        sim = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0)]
        ana = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0)]
        result = ps.compare_waveforms("test", sim, ana)
        csv = ps.export_validation_csv([result])
        assert "test" in csv
        assert "," in csv

    def test_export_validation_json(self):
        """Test JSON export of validation results."""
        sim = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0)]
        ana = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0)]
        result = ps.compare_waveforms("test", sim, ana)
        json_str = ps.export_validation_json([result])
        assert "test" in json_str


class TestUtilities:
    """Test utility functions."""

    def test_solver_status_to_string(self):
        """Test solver status to string conversion."""
        s = ps.solver_status_to_string(ps.SolverStatus.Success)
        assert "Success" in s or "success" in s.lower()


class TestCircuit:
    """Test Circuit builder class."""

    def test_circuit_creation(self):
        """Test circuit creation."""
        ckt = ps.Circuit()
        assert ckt.num_nodes() == 0
        assert ckt.num_devices() == 0

    def test_add_nodes(self):
        """Test node addition."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        n2 = ckt.add_node("n2")
        assert gnd == -1
        assert n1 == 0
        assert n2 == 1
        assert ckt.num_nodes() == 2

    def test_add_resistor(self):
        """Test resistor addition."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        ckt.add_resistor("R1", n1, gnd, 1000.0)
        assert ckt.num_devices() == 1

    def test_add_capacitor(self):
        """Test capacitor addition."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        ckt.add_capacitor("C1", n1, gnd, 1e-6, 0.0)
        assert ckt.num_devices() == 1

    def test_voltage_source_branch(self):
        """Test voltage source creates branch."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        ckt.add_voltage_source("V1", n1, gnd, 5.0)
        assert ckt.num_branches() == 1
        assert ckt.system_size() == 2  # 1 node + 1 branch

    def test_assemble_dc(self):
        """Test DC matrix assembly."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        ckt.add_voltage_source("V1", n1, gnd, 5.0)
        ckt.add_resistor("R1", n1, gnd, 1000.0)

        G, b = ckt.assemble_dc()
        assert G.shape == (2, 2)
        assert b.shape == (2,)
        assert b[1] == 5.0  # VS voltage


class TestDCSolver:
    """Test DC solver functionality."""

    def test_solve_dc_resistor_divider(self):
        """Test DC solve with resistor divider."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        n2 = ckt.add_node("n2")

        ckt.add_voltage_source("V1", n1, gnd, 10.0)
        ckt.add_resistor("R1", n1, n2, 1000.0)
        ckt.add_resistor("R2", n2, gnd, 1000.0)

        result = ps.solve_dc(ckt)
        assert result.success()
        assert abs(result.solution[0] - 10.0) < 0.01  # n1 = 10V
        assert abs(result.solution[1] - 5.0) < 0.01   # n2 = 5V

    def test_dc_operating_point_linear(self):
        """Test dc_operating_point with linear circuit."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")

        ckt.add_voltage_source("V1", n1, gnd, 5.0)
        ckt.add_resistor("R1", n1, gnd, 1000.0)

        result = ps.dc_operating_point(ckt)
        assert result.success
        assert result.strategy_used == ps.DCStrategy.Direct
        assert abs(result.newton_result.solution[0] - 5.0) < 0.01

    def test_dc_operating_point_diode(self):
        """Test dc_operating_point with diode."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        n2 = ckt.add_node("n2")

        ckt.add_voltage_source("V1", n1, gnd, 5.0)
        ckt.add_resistor("R1", n1, n2, 1000.0)
        ckt.add_diode("D1", n2, gnd)

        config = ps.DCConvergenceConfig()
        config.strategy = ps.DCStrategy.Auto
        result = ps.dc_operating_point(ckt, config)
        assert result.success
        # n2 should be near 0V (diode forward)
        assert result.newton_result.solution[1] < 0.1


class TestTransient:
    """Test transient simulation."""

    def test_rc_transient(self):
        """Test RC circuit transient response."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        n2 = ckt.add_node("n2")

        ckt.add_voltage_source("V1", n1, gnd, 5.0)
        ckt.add_resistor("R1", n1, n2, 1000.0)
        ckt.add_capacitor("C1", n2, gnd, 1e-6, 0.0)

        # Run transient for 5*tau
        tau = 1e-3  # 1ms
        times, states, success, msg = ps.run_transient(ckt, 0.0, 5*tau, tau/100)

        assert success
        assert len(times) > 0
        assert len(states) == len(times)

        # Check at t=tau: V should be ~63.2% of final
        idx_tau = int(tau / (tau/100))
        v_cap = states[idx_tau][1]
        expected = 5.0 * (1 - math.exp(-1))
        assert abs(v_cap - expected) < 0.1

    def test_transient_with_initial_state(self):
        """Test transient with DC initial state."""
        import numpy as np

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("n1")
        n2 = ckt.add_node("n2")

        ckt.add_voltage_source("V1", n1, gnd, 5.0)
        ckt.add_resistor("R1", n1, n2, 1000.0)
        ckt.add_resistor("R2", n2, gnd, 1000.0)

        # Get DC solution first
        dc_result = ps.solve_dc(ckt)
        assert dc_result.success()

        # Run short transient from DC state
        times, states, success, msg = ps.run_transient(
            ckt, 0.0, 1e-6, 1e-7, dc_result.solution)

        assert success
        # State should stay at DC: n1=5V, n2=2.5V (resistor divider)
        assert abs(states[-1][1] - 2.5) < 0.01


class TestNonlinearDevices:
    """Test nonlinear device bindings."""

    def test_ideal_diode(self):
        """Test IdealDiode creation."""
        d = ps.IdealDiode(1e3, 1e-9, "D1")
        assert d.name() == "D1"

    def test_ideal_switch(self):
        """Test IdealSwitch creation and control."""
        sw = ps.IdealSwitch(1e6, 1e-12, False, "S1")
        assert not sw.is_closed()
        sw.close()
        assert sw.is_closed()
        sw.open()
        assert not sw.is_closed()

    def test_mosfet(self):
        """Test MOSFET creation."""
        params = ps.MOSFETParams()
        params.vth = 2.0
        params.kp = 0.01
        m = ps.MOSFET(params, "M1")
        assert m.name() == "M1"
        assert m.params().vth == 2.0

    def test_igbt(self):
        """Test IGBT creation."""
        params = ps.IGBTParams()
        params.vth = 5.0
        igbt = ps.IGBT(params, "Q1")
        assert igbt.name() == "Q1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
