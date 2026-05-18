"""Integration tests for the auto-parasitics pre-flight pipeline.

The `boost-pfc-auto-parasitics` change (Pulsim 0.10.0a12) hooks
`Circuit.auto_configure_parasitics()` into `Simulator.run_transient()`
by default, so cold-start users hit convergent boost-class circuits
without knowing about C_oss, Tustin damping, or PWL Ideal feasibility.

These tests pin three behaviors:
  1. **Detection**: an L-in-series-with-MOSFET boost cell shows up as
     a TopologyIssue with Severity.Critical and an action sized for
     the user's `max_overshoot_frac` target.
  2. **Application**: after `run_transient()`, the MOSFET's C_oss has
     actually changed (auto-parasitics is not just diagnostic).
  3. **Opt-out**: a user-set `C_oss = X` is respected; a user-set
     `enabled = False` disables the analyzer entirely.
"""
from __future__ import annotations

import math
import pytest
import pulsim as ps


# Common boost-PFC fixture --------------------------------------------------

def _build_boost(C_oss_user: float = 0.0,
                  Eon_25: float = 60e-6,
                  inductor_initial_A: float = 5.0,
                  cap_initial_V: float = 400.0):
    """Build a minimal DC-fed boost: Vdc → L → switch/diode → C_out → R_load.

    Args
    ----
    C_oss_user : float
        If > 0, set MOSFETParams.C_oss explicitly (simulates a user override).
    Eon_25 : float
        If > 0, auto-promotes MOSFET to PWL Ideal (which is where
        auto-parasitics decides to act). Default 60 µJ.
    """
    c = ps.Circuit()
    n_bulk = c.add_node("bulk"); n_sw = c.add_node("sw")
    n_bus = c.add_node("bus"); n_gate = c.add_node("gate")
    gnd = ps.Circuit.ground()

    c.add_voltage_source("Vdc", n_bulk, gnd, 310.0)

    ind = ps.InductorParams()
    ind.inductance = 100e-6
    ind.initial_current = inductor_initial_A
    c.add_inductor("L_boost", n_bulk, n_sw, ind)

    mp = ps.MOSFETParams()
    mp.vth = 4.0; mp.kp = 50.0; mp.g_on = 50.0; mp.g_off = 1e-12
    mp.is_nmos = True
    mp.Eon_25 = Eon_25; mp.Eoff_25 = 2 * Eon_25
    mp.I_ref = 5.0; mp.V_ref = 400.0
    mp.R_th_ja = 1.5
    if C_oss_user > 0:
        mp.C_oss = C_oss_user
    c.add_mosfet("M_pfc", n_gate, n_sw, gnd, mp)
    c.add_pwm_voltage_source("Vg", n_gate, gnd, 12.0, 0.0, 100e3, 0.25)

    dbst = ps.RealisticDiodeParams()
    dbst.V_F0 = 0.85; dbst.R_d = 30e-3; dbst.g_on = 1.0/dbst.R_d
    dbst.Qrr = 60e-9
    c.add_diode("D_boost", n_sw, n_bus, dbst)

    cap_o = ps.CapacitorParams()
    cap_o.capacitance = 220e-6; cap_o.initial_voltage = cap_initial_V
    c.add_capacitor("C_out", n_bus, gnd, cap_o)

    load = ps.ResistorParams(); load.resistance = 291.0
    c.add_resistor("R_load", n_bus, gnd, load)

    return c


def _short_opts():
    """Minimum-cost transient options for a 50 µs smoke."""
    opts = ps.SimulationOptions()
    opts.tstart = 0.0; opts.tstop = 50e-6; opts.dt = 100e-9
    opts.dt_min = 1e-9; opts.dt_max = 200e-9
    opts.auto_parasitics.verbose = False  # silence stderr in pytest
    return opts


# -------------- detection (dry-run / no-simulate path) ----------------------

class TestDetection:
    def test_boost_pair_is_detected_critical(self):
        """A 100 µH boost L feeding a MOSFET in PWL Ideal mode with the
        in-ctor 10 nF default C_oss must show up as Critical (overshoot
        far above V_bus)."""
        c = _build_boost()
        # Dry-run: don't mutate, just report.
        report = c.auto_configure_parasitics(dry_run=True)
        assert len(report.issues) >= 1
        # Find the MOSFET issue.
        mosfet_issue = next((i for i in report.issues
                              if i.switch_name == "M_pfc"), None)
        assert mosfet_issue is not None
        assert mosfet_issue.inductor_name == "L_boost"
        assert mosfet_issue.L_henry == pytest.approx(100e-6, rel=1e-9)
        assert mosfet_issue.severity == ps.TopologyIssueSeverity.Critical

    def test_dry_run_does_not_mutate(self):
        """`dry_run=True` returns the recommendation without changing the
        device's C_oss. Sanity-check via a second invocation that produces
        the same predicted overshoot."""
        c = _build_boost()
        r1 = c.auto_configure_parasitics(dry_run=True)
        r2 = c.auto_configure_parasitics(dry_run=True)
        # Two dry-runs over the same circuit produce identical reports.
        assert len(r1.issues) == len(r2.issues)
        for i in range(len(r1.issues)):
            assert (r1.issues[i].predicted_overshoot
                    == pytest.approx(r2.issues[i].predicted_overshoot))


# -------------- application (real-mutation path) ----------------------------

class TestApplication:
    def test_simulator_runs_auto_config_by_default(self):
        """Constructing a Simulator on a boost circuit triggers the
        pre-flight; the result must carry a non-empty topology_report
        with at least one applied action."""
        c = _build_boost()
        opts = _short_opts()
        sim = ps.Simulator(c, opts)
        r = sim.run_transient(c.initial_state())
        assert r.success
        assert len(r.topology_report.issues) >= 1
        # Some action should have been taken (M_pfc + D_boost both get
        # auto-sized C_oss / C_j).
        assert r.topology_report.num_actions() >= 1

    def test_v_sw_overshoot_drops_after_auto_config(self):
        """The whole point: after auto-config, V_sw stays bounded. We
        verify by comparing against the *uncon­figured* baseline.

        Baseline: in-ctor default C_oss = 10 nF, with auto-parasitics OFF.
        Configured: auto-parasitics ON (default).

        Boost with L=100 µH, I_peak=6 A: pure-LC overshoot = 6·√(100µH/10nF)
        = 600 V. After auto-config the C_oss should be ~90 nF (sized for
        50 % overshoot of 400 V = 200 V target).
        """
        # Baseline: disable auto-parasitics so we see what the in-ctor
        # 10 nF default produces.
        c_base = _build_boost()
        opts_base = _short_opts()
        opts_base.auto_parasitics.enabled = False
        sim_base = ps.Simulator(c_base, opts_base)
        r_base = sim_base.run_transient(c_base.initial_state())
        v_sw_idx_base = c_base.get_node("sw")
        import numpy as np
        x_base = np.asarray(r_base.states)
        v_sw_peak_base = float(x_base[:, v_sw_idx_base].max())

        # Configured: default ON.
        c_cfg = _build_boost()
        opts_cfg = _short_opts()
        sim_cfg = ps.Simulator(c_cfg, opts_cfg)
        r_cfg = sim_cfg.run_transient(c_cfg.initial_state())
        v_sw_idx_cfg = c_cfg.get_node("sw")
        x_cfg = np.asarray(r_cfg.states)
        v_sw_peak_cfg = float(x_cfg[:, v_sw_idx_cfg].max())

        # The configured run's peak should be at least 25 % lower than
        # the baseline — generous tolerance because the diode still
        # clips and exact numbers depend on phase of the gate inside the
        # short window.
        assert v_sw_peak_cfg < 0.75 * v_sw_peak_base, (
            f"Auto-config did not reduce V_sw peak: "
            f"baseline = {v_sw_peak_base:.1f} V, "
            f"configured = {v_sw_peak_cfg:.1f} V"
        )


# -------------- opt-out (per-device + global) -------------------------------

class TestOptOut:
    def test_user_set_C_oss_is_respected(self):
        """A user who explicitly sets `MOSFETParams.C_oss = X` opts out
        per-device — the analyzer records the issue but takes no action."""
        c = _build_boost(C_oss_user=50e-9)
        report = c.auto_configure_parasitics(dry_run=False)
        # Find the M_pfc action.
        action = next((a for a in report.actions
                        if a.device_name == "M_pfc"), None)
        assert action is not None
        assert action.kind == ps.ParasiticActionKind.None_, (
            f"Expected None_, got {action.kind} with rationale: {action.rationale}"
        )
        assert "user" in action.rationale.lower()

    def test_global_opt_out_via_options(self):
        """Setting `opts.auto_parasitics.enabled = False` disables the
        analyzer entirely. result.topology_report.issues is empty."""
        c = _build_boost()
        opts = _short_opts()
        opts.auto_parasitics.enabled = False
        sim = ps.Simulator(c, opts)
        r = sim.run_transient(c.initial_state())
        assert r.success
        assert len(r.topology_report.issues) == 0
        assert "disabled" in r.topology_report.summary


# -------------- non-switching topologies ------------------------------------

class TestNoFalsePositives:
    def test_rl_load_no_switch_no_action(self):
        """An RL load with no switch should produce zero issues — the
        analyzer must never act on circuits without commutation."""
        c = ps.Circuit()
        n1 = c.add_node("n1"); gnd = ps.Circuit.ground()
        c.add_voltage_source("V", n1, gnd, 5.0)
        ind = ps.InductorParams(); ind.inductance = 1e-3
        c.add_inductor("L", n1, gnd, ind)
        report = c.auto_configure_parasitics()
        assert len(report.issues) == 0
        assert report.num_actions() == 0
