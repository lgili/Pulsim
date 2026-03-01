"""Validation tests for thermal simulation API.

Tests Foster network thermal modeling, ThermalSimulator, and
junction temperature calculations.

Tolerance: 1% for analytical Zth(t) curves.
"""

import pytest
import numpy as np

# Try to import pulsim - skip tests if import fails
try:
    import pulsim as ps
    PULSIM_AVAILABLE = True
except ImportError:
    PULSIM_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not PULSIM_AVAILABLE,
    reason="pulsim module not available"
)


# =============================================================================
# Test Constants
# =============================================================================

# Typical MOSFET thermal parameters
RTH_JC = 0.5    # Junction-to-case (K/W)
RTH_CS = 0.3    # Case-to-sink (K/W)
RTH_SA = 1.0    # Sink-to-ambient (K/W)
RTH_JA = RTH_JC + RTH_CS + RTH_SA  # Total = 1.8 K/W

TAU_1 = 0.001   # 1ms for junction
TAU_2 = 0.010   # 10ms for case
TAU_3 = 0.100   # 100ms for heatsink

T_AMBIENT = 25.0  # 25C ambient


# =============================================================================
# FosterStage Tests
# =============================================================================

class TestFosterStage:
    """Test individual Foster network stage."""

    def test_stage_creation(self):
        """Test FosterStage creation and properties."""
        stage = ps.FosterStage(RTH_JC, TAU_1)

        assert stage.Rth == RTH_JC
        assert stage.tau == TAU_1

        # Cth = tau / Rth
        expected_Cth = TAU_1 / RTH_JC
        assert abs(stage.Cth() - expected_Cth) < 1e-10

    def test_stage_zth(self):
        """Test Zth(t) calculation for single stage."""
        stage = ps.FosterStage(1.0, 0.01)  # 1 K/W, 10ms

        # At t=0, Zth = 0
        assert stage.Zth(0.0) == 0.0

        # At t=tau, Zth = Rth * (1 - e^-1) ≈ 0.632 * Rth
        zth_tau = stage.Zth(0.01)
        expected = 1.0 * (1.0 - np.exp(-1.0))
        assert abs(zth_tau - expected) < 1e-6

        # At t=5*tau, Zth ≈ Rth (steady state)
        zth_5tau = stage.Zth(0.05)
        expected_ss = 1.0 * (1.0 - np.exp(-5.0))
        assert abs(zth_5tau - expected_ss) < 1e-4

    def test_stage_delta_T(self):
        """Test temperature rise calculation."""
        stage = ps.FosterStage(1.0, 0.01)

        # 100W applied for 1 tau
        dT = stage.delta_T(100.0, 0.01)
        expected = 100.0 * 1.0 * (1.0 - np.exp(-1.0))
        assert abs(dT - expected) < 1e-4


# =============================================================================
# FosterNetwork Tests
# =============================================================================

class TestFosterNetwork:
    """Test Foster thermal network."""

    def test_network_creation_from_lists(self):
        """Test creation from Rth and tau lists."""
        Rth_list = [RTH_JC, RTH_CS, RTH_SA]
        tau_list = [TAU_1, TAU_2, TAU_3]

        network = ps.FosterNetwork(Rth_list, tau_list, "TestMOSFET")

        assert network.num_stages() == 3
        assert network.name() == "TestMOSFET"
        assert abs(network.total_Rth() - RTH_JA) < 1e-10

    def test_network_stages(self):
        """Test individual stage access."""
        network = ps.FosterNetwork([1.0, 2.0], [0.01, 0.1])

        assert network.num_stages() == 2
        assert network.stage(0).Rth == 1.0
        assert network.stage(0).tau == 0.01
        assert network.stage(1).Rth == 2.0
        assert network.stage(1).tau == 0.1

    def test_network_total_zth(self):
        """Test total Zth(t) is sum of stages."""
        network = ps.FosterNetwork([1.0, 2.0], [0.01, 0.1])

        t = 0.05  # 50ms

        # Calculate expected Zth
        zth1 = 1.0 * (1.0 - np.exp(-t / 0.01))
        zth2 = 2.0 * (1.0 - np.exp(-t / 0.1))
        expected = zth1 + zth2

        actual = network.Zth(t)
        assert abs(actual - expected) < 1e-10

    def test_network_steady_state(self):
        """Test steady-state temperature rise."""
        network = ps.FosterNetwork([RTH_JC, RTH_CS, RTH_SA], [TAU_1, TAU_2, TAU_3])

        P = 100.0  # 100W
        expected = P * RTH_JA

        assert abs(network.delta_T_ss(P) - expected) < 1e-10

    def test_zth_curve_generation(self):
        """Test Zth(t) curve generation."""
        network = ps.FosterNetwork([1.0], [0.1])

        curve = network.Zth_curve(0.0, 1.0, 11)

        assert len(curve) == 11
        assert curve[0][0] == 0.0  # t=0
        assert curve[-1][0] == 1.0  # t=1

        # At t=0, Zth should be 0
        assert curve[0][1] == 0.0

        # At t=1s >> tau, Zth should approach Rth=1.0
        assert curve[-1][1] > 0.99


# =============================================================================
# CauerNetwork Tests
# =============================================================================

class TestCauerNetwork:
    """Test Cauer thermal network."""

    def test_cauer_creation(self):
        """Test CauerNetwork creation."""
        Rth_list = [0.5, 0.3, 1.0]
        Cth_list = [0.002, 0.033, 0.1]  # J/K

        network = ps.CauerNetwork(Rth_list, Cth_list, "CauerTest")

        assert network.num_stages() == 3
        assert network.name() == "CauerTest"
        assert abs(network.total_Rth() - 1.8) < 1e-10

    def test_cauer_total_capacitance(self):
        """Test total thermal capacitance."""
        network = ps.CauerNetwork([1.0, 2.0], [0.1, 0.2])

        assert abs(network.total_Cth() - 0.3) < 1e-10

    def test_cauer_stages(self):
        """Test stage access."""
        network = ps.CauerNetwork([1.0, 2.0], [0.1, 0.2])

        assert network.stage(0).Rth == 1.0
        assert network.stage(0).Cth == 0.1
        assert abs(network.stage(0).tau() - 0.1) < 1e-10


# =============================================================================
# ThermalSimulator Tests
# =============================================================================

class TestThermalSimulator:
    """Test ThermalSimulator transient calculations."""

    def test_simulator_initial_state(self):
        """Test initial state is ambient temperature."""
        network = ps.FosterNetwork([1.0], [0.1])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        assert sim.ambient() == T_AMBIENT
        assert sim.Tj() == T_AMBIENT
        assert sim.time() == 0.0

    def test_simulator_step_response(self):
        """Test single step response against analytical solution."""
        # Simple single-stage network
        Rth = 1.0
        tau = 0.1
        network = ps.FosterNetwork([Rth], [tau])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        P = 100.0  # Apply 100W
        dt = 0.1   # Step for 1 tau

        sim.step(P, dt)

        # Expected: Tj = Tamb + P * Rth * (1 - e^(-dt/tau))
        expected_dT = P * Rth * (1.0 - np.exp(-dt / tau))
        expected_Tj = T_AMBIENT + expected_dT

        assert abs(sim.Tj() - expected_Tj) < 0.1  # 0.1C tolerance

    def test_simulator_steady_state(self):
        """Test that temperature approaches steady state."""
        network = ps.FosterNetwork([1.0], [0.1])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        P = 100.0

        # Step for 10 time constants (should be at steady state)
        for _ in range(100):
            sim.step(P, 0.01)  # 10ms steps, total 1s >> tau

        expected_ss = T_AMBIENT + P * 1.0
        assert abs(sim.Tj() - expected_ss) < 0.5  # 0.5C tolerance

    def test_simulator_cooldown(self):
        """Test cooldown after power removal."""
        network = ps.FosterNetwork([1.0], [0.1])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        # Heat up to steady state
        P = 100.0
        for _ in range(100):
            sim.step(P, 0.01)

        T_hot = sim.Tj()

        # Remove power and cool down
        for _ in range(100):
            sim.step(0.0, 0.01)

        # Should be back near ambient
        assert sim.Tj() < T_hot
        assert abs(sim.Tj() - T_AMBIENT) < 1.0

    def test_simulator_waveform(self):
        """Test simulation with power waveform."""
        network = ps.FosterNetwork([1.0], [0.1])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        # Pulsed power: 100W for 100ms, then 0W for 100ms
        times = [0.0, 0.1, 0.2]
        powers = [0.0, 100.0, 0.0]

        temps = sim.simulate(times, powers)

        assert len(temps) == 3
        assert temps[0] == T_AMBIENT  # Start at ambient
        assert temps[1] > T_AMBIENT   # Heated up
        assert temps[2] < temps[1]    # Cooled down

    def test_simulator_reset(self):
        """Test reset to ambient."""
        network = ps.FosterNetwork([1.0], [0.1])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        # Heat up
        sim.step(100.0, 0.1)
        assert sim.Tj() > T_AMBIENT

        # Reset
        sim.reset()
        assert sim.Tj() == T_AMBIENT
        assert sim.time() == 0.0

    def test_simulator_steady_state_temperature(self):
        """Test steady_state_temperature() calculation."""
        network = ps.FosterNetwork([RTH_JC, RTH_CS, RTH_SA], [TAU_1, TAU_2, TAU_3])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        P = 50.0  # 50W
        T_ss = sim.steady_state_temperature(P)

        expected = T_AMBIENT + P * RTH_JA
        assert abs(T_ss - expected) < 1e-6


# =============================================================================
# ThermalLimitMonitor Tests
# =============================================================================

class TestThermalLimitMonitor:
    """Test thermal limit monitoring."""

    def test_monitor_ok(self):
        """Test OK zone detection."""
        monitor = ps.ThermalLimitMonitor(125.0, 150.0)

        assert monitor.is_ok(100.0)
        assert monitor.is_ok(124.9)
        assert not monitor.is_ok(125.0)

        assert monitor.check(100.0) == 0

    def test_monitor_warning(self):
        """Test warning zone detection."""
        monitor = ps.ThermalLimitMonitor(125.0, 150.0)

        assert monitor.is_warning(125.0)
        assert monitor.is_warning(140.0)
        assert monitor.is_warning(149.9)
        assert not monitor.is_warning(100.0)
        assert not monitor.is_warning(150.0)

        assert monitor.check(130.0) == 1

    def test_monitor_exceeded(self):
        """Test maximum exceeded detection."""
        monitor = ps.ThermalLimitMonitor(125.0, 150.0)

        assert monitor.is_exceeded(150.0)
        assert monitor.is_exceeded(175.0)
        assert not monitor.is_exceeded(149.9)

        assert monitor.check(160.0) == 2

    def test_monitor_set_limits(self):
        """Test setting new limits."""
        monitor = ps.ThermalLimitMonitor(100.0, 125.0)

        assert monitor.T_warning() == 100.0
        assert monitor.T_max() == 125.0

        monitor.set_limits(110.0, 140.0)

        assert monitor.T_warning() == 110.0
        assert monitor.T_max() == 140.0


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestThermalFactories:
    """Test factory functions for creating thermal networks."""

    def test_mosfet_thermal_model(self):
        """Test MOSFET thermal model creation."""
        network = ps.create_mosfet_thermal_model(RTH_JC, RTH_CS, RTH_SA, "IRFP460")

        assert network.num_stages() == 3
        assert network.name() == "IRFP460"
        assert abs(network.total_Rth() - RTH_JA) < 1e-10

    def test_simple_thermal_model(self):
        """Test simple single-stage model."""
        network = ps.create_simple_thermal_model(2.0, 0.5, "Simple")

        assert network.num_stages() == 1
        assert network.name() == "Simple"
        assert network.total_Rth() == 2.0

    def test_datasheet_4param_model(self):
        """Test 4-parameter datasheet model."""
        network = ps.create_from_datasheet_4param(
            0.1, 0.001,
            0.3, 0.01,
            0.5, 0.1,
            1.0, 1.0,
            "DS_Model"
        )

        assert network.num_stages() == 4
        assert abs(network.total_Rth() - 1.9) < 1e-10


# =============================================================================
# Analytical Validation Tests
# =============================================================================

class TestThermalAnalytical:
    """Validate thermal simulation against analytical solutions."""

    def test_single_stage_vs_analytical(self):
        """Compare single-stage Zth against analytical formula."""
        Rth = 1.5
        tau = 0.05
        network = ps.FosterNetwork([Rth], [tau])

        # Generate Zth curve
        num_points = 101
        t_end = 0.5  # 10 time constants
        curve = network.Zth_curve(0.0, t_end, num_points)

        # Compare against analytical: Zth(t) = Rth * (1 - exp(-t/tau))
        max_error = 0.0
        for t, zth_sim in curve:
            zth_analytical = Rth * (1.0 - np.exp(-t / tau))
            error = abs(zth_sim - zth_analytical)
            max_error = max(max_error, error)

        # Should be essentially exact (numerical precision)
        assert max_error < 1e-10

    def test_multi_stage_vs_analytical(self):
        """Compare multi-stage network against analytical sum."""
        Rth_list = [0.5, 1.0, 2.0]
        tau_list = [0.001, 0.01, 0.1]
        network = ps.FosterNetwork(Rth_list, tau_list)

        times = np.linspace(0, 1.0, 101)

        max_error = 0.0
        for t in times:
            # Analytical: sum of individual stages
            zth_analytical = sum(
                R * (1.0 - np.exp(-t / tau))
                for R, tau in zip(Rth_list, tau_list)
            )
            zth_sim = network.Zth(t)
            error = abs(zth_sim - zth_analytical)
            max_error = max(max_error, error)

        assert max_error < 1e-10

    def test_simulator_vs_analytical_step(self):
        """Compare simulator step response against analytical."""
        Rth = 1.0
        tau = 0.1
        P = 100.0

        network = ps.FosterNetwork([Rth], [tau])
        sim = ps.ThermalSimulator(network, T_AMBIENT)

        # Simulate with small steps
        dt = 0.001  # 1ms steps
        times = np.arange(0, 1.0, dt)

        temps_sim = [T_AMBIENT]
        for t in times[1:]:
            sim.step(P, dt)
            temps_sim.append(sim.Tj())

        # Compare with analytical
        # Tj(t) = Tamb + P * Rth * (1 - exp(-t/tau))
        max_rel_error = 0.0
        for i, t in enumerate(times):
            if t == 0:
                continue
            dT_analytical = P * Rth * (1.0 - np.exp(-t / tau))
            T_analytical = T_AMBIENT + dT_analytical

            rel_error = abs(temps_sim[i] - T_analytical) / (T_analytical - T_AMBIENT + 1e-10)
            max_rel_error = max(max_rel_error, rel_error)

        # Should be within 1%
        assert max_rel_error < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
