"""Tests for thermal simulation."""

import pytest
import pulsim as sl


class TestThermalModel:
    """Test thermal model construction."""

    def test_foster_network(self):
        """Test Foster network construction."""
        foster = sl.FosterNetwork()
        foster.stages = []

        stage1 = sl.ThermalRCStage()
        stage1.rth = 0.1
        stage1.cth = 0.01

        stage2 = sl.ThermalRCStage()
        stage2.rth = 0.2
        stage2.cth = 0.02

        foster.stages = [stage1, stage2]

        assert len(foster.stages) == 2
        assert foster.rth_total() == pytest.approx(0.3)  # 0.1 + 0.2

    def test_thermal_rc_stage_tau(self):
        """Test RC stage time constant."""
        stage = sl.ThermalRCStage()
        stage.rth = 0.5  # K/W
        stage.cth = 0.1  # J/K

        # tau = R * C = 0.5 * 0.1 = 0.05 s
        assert stage.tau() == pytest.approx(0.05)

    def test_thermal_model_construction(self):
        """Test thermal model construction."""
        model = sl.ThermalModel()
        model.device_name = "Q1"
        # Use Simple type to use rth_jc directly
        model.type = sl.ThermalNetworkType.Simple
        model.rth_jc = 0.5
        model.rth_cs = 0.3
        model.rth_sa = 1.0
        model.tj_max = 175.0
        model.tj_warn = 150.0

        assert model.device_name == "Q1"
        # rth_ja = rth_jc + rth_cs + rth_sa = 0.5 + 0.3 + 1.0 = 1.8
        assert model.rth_ja() == pytest.approx(1.8)


class TestThermalSimulator:
    """Test thermal simulator."""

    def test_thermal_simulator_basic(self):
        """Basic thermal simulation."""
        model = sl.create_mosfet_thermal("M1", rth_jc=0.5, rth_cs=0.3, rth_sa=1.0)

        thermal = sl.ThermalSimulator()
        thermal.add_model(model)
        thermal.set_ambient(25.0)
        thermal.initialize()

        assert thermal.ambient() == 25.0
        assert thermal.junction_temp("M1") == pytest.approx(25.0)

    def test_thermal_step(self):
        """Test thermal stepping with power input."""
        model = sl.create_mosfet_thermal("M1", rth_jc=0.5)

        thermal = sl.ThermalSimulator()
        thermal.add_model(model)
        thermal.set_ambient(25.0)
        thermal.initialize()

        # Apply 10W for 1ms
        for _ in range(100):
            thermal.step(1e-5, {"M1": 10.0})

        tj = thermal.junction_temp("M1")
        # Temperature should have risen
        assert tj > 25.0

    def test_temperature_adjustment(self):
        """Test Rds_on and Vth temperature adjustment."""
        thermal = sl.ThermalSimulator()

        # Rds_on increases with temperature
        rds_25 = 0.05  # 50mOhm at 25°C
        rds_100 = thermal.adjust_rds_on(rds_25, 100.0)
        assert rds_100 > rds_25

        # Vth decreases with temperature (negative TC)
        vth_25 = 3.0  # 3V at 25°C
        vth_100 = thermal.adjust_vth(vth_25, 100.0)
        assert vth_100 < vth_25


class TestFosterNetworkFitting:
    """Test Foster network curve fitting."""

    def test_fit_foster_network(self):
        """Fit Foster network from Zth curve."""
        # Typical Zth curve datasheet points (time, Zth)
        zth_curve = [
            (1e-5, 0.01),
            (1e-4, 0.05),
            (1e-3, 0.15),
            (1e-2, 0.35),
            (1e-1, 0.48),
            (1.0, 0.50),
        ]

        foster = sl.fit_foster_network(zth_curve, num_stages=4)

        assert len(foster.stages) == 4
        # Total Rth should be close to final Zth
        assert foster.rth_total() == pytest.approx(0.5, rel=0.2)

    def test_create_mosfet_thermal(self):
        """Test MOSFET thermal model creation."""
        model = sl.create_mosfet_thermal("Q1", rth_jc=0.8, rth_cs=0.5, rth_sa=2.0)

        assert model.device_name == "Q1"
        assert model.rth_jc == pytest.approx(0.8)
        assert model.rth_cs == pytest.approx(0.5)
        assert model.rth_sa == pytest.approx(2.0)
        assert model.type == sl.ThermalNetworkType.Foster
        assert len(model.foster.stages) > 0
