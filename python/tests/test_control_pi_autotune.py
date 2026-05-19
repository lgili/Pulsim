"""Pin the contract for `pulsim.auto_tune_pi_buck()` and verify it
plays nicely with the C++-backed `pulsim.PIController` in a real
per-period closed-loop buck simulation.

Why these tests exist
---------------------

`add_virtual_component("pi_controller", ...)` is buggy when the host
circuit has a `voltage_source` reference node and is driven by
`execute_mixed_domain_step(x, t)` AFTER `run_transient(zeros)` —
the voltage source's value isn't propagated into the returned
state, so the PI sees `signal = 0 − vout` (inverted sign) and
saturates.

The recommended path is the standalone `ps.PIController` (C++) with
gains from `ps.auto_tune_pi_buck()`. These tests pin that contract.
"""

import numpy as np
import pulsim as ps


def test_auto_tune_pi_buck_returns_finite_positive_gains():
    """Smoke: sensible Kp/Ki for a canonical 24V→12V/2A/25kHz buck."""
    gains = ps.auto_tune_pi_buck(
        Vin=24.0, L=330e-6, C=220e-6, R=6.0, fsw=25e3,
        target_pm_deg=60.0,
    )
    assert gains.kp > 0
    assert gains.ki > 0
    assert gains.fc_hz > 0
    assert gains.fc_hz < 25e3 / 5  # Shannon cap
    assert gains.f0_hz > 0
    assert gains.Q > 0
    # PM estimate should be at least the target (we err on the
    # conservative side by placing fc sub-resonance with a far zero).
    assert gains.estimated_pm_deg >= gains.target_pm_deg


def test_auto_tune_pi_buck_rejects_invalid_inputs():
    """Negative / zero topology values raise ValueError."""
    for bad in [{"Vin": 0}, {"Vin": -1}, {"L": 0}, {"C": -1e-6},
                {"R": 0}, {"fsw": 0}]:
        kwargs = dict(Vin=24, L=330e-6, C=220e-6, R=6.0, fsw=25e3)
        kwargs.update(bad)
        try:
            ps.auto_tune_pi_buck(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")

    # PM out of range
    for pm in [-10, 0, 180, 200]:
        try:
            ps.auto_tune_pi_buck(Vin=24, L=330e-6, C=220e-6, R=6.0, fsw=25e3,
                                 target_pm_deg=pm)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for target_pm_deg={pm}")


def test_auto_tune_pi_buck_with_explicit_bandwidth_clamps_to_shannon():
    """User asks for fc above fsw/5 → clamped to fsw/5."""
    gains = ps.auto_tune_pi_buck(
        Vin=24.0, L=330e-6, C=220e-6, R=6.0, fsw=25e3,
        target_bandwidth_hz=1e6,  # absurdly high
    )
    assert gains.fc_hz == 25e3 / 5


def test_buck_closed_loop_tracks_reference():
    """End-to-end: auto-tune + PIController closes the loop on a real
    buck simulation, vout tracks Vref within 5 %."""
    Vin, Vref, fsw = 24.0, 12.0, 25000.0
    L, C, R = 330e-6, 220e-6, 6.0

    gains = ps.auto_tune_pi_buck(Vin=Vin, L=L, C=C, R=R, fsw=fsw,
                                 target_pm_deg=60.0)

    ckt = ps.Circuit()
    gnd = ckt.ground()
    n_sw = ckt.add_node("sw"); n_out = ckt.add_node("out")
    pwm = ps.PWMParams()
    pwm.v_high = Vin; pwm.v_low = 0; pwm.frequency = fsw
    pwm.duty = Vref / Vin   # open-loop seed
    ckt.add_pwm_voltage_source("Vpwm", n_sw, gnd, pwm)
    ckt.add_inductor("L1", n_sw, n_out, L)
    ckt.add_capacitor("C1", n_out, gnd, C)
    ckt.add_resistor("Rload", n_out, gnd, R)

    pi = ps.PIController(Kp=gains.kp, Ki=gains.ki,
                         output_min=0.05, output_max=0.95)

    T_sw = 1.0 / fsw
    opts = ps.SimulationOptions()
    opts.tstart = 0; opts.tstop = T_sw; opts.dt = T_sw / 40
    opts.dt_min = opts.dt; opts.dt_max = opts.dt
    opts.adaptive_timestep = False
    opts.integrator = ps.Integrator.BDF1
    opts.switching_mode = ps.SwitchingMode.Behavioral
    opts.newton_options.max_iterations = 140
    opts.newton_options.num_nodes = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()

    # Seed x0 from DC OP — critical to avoid the run_transient(zeros) bug.
    sim_dc = ps.Simulator(ckt, opts)
    dc = sim_dc.dc_operating_point()
    assert dc.success
    x = dc.newton_result.solution.copy()

    t = 0.0
    vout_history = []
    n_periods = 1500  # 60 ms ≈ 4× LC time constant
    for _ in range(n_periods):
        opts.tstart = t; opts.tstop = t + T_sw
        sim = ps.Simulator(ckt, opts)
        result = sim.run_transient(x)
        assert result.success
        states = np.array(result.states)
        x = states[-1]
        t += T_sw
        v_out_avg = float(np.mean(states[:, n_out]))
        new_duty = pi.update(error=Vref - v_out_avg, t=t)
        ckt.set_pwm_duty("Vpwm", new_duty)
        vout_history.append(v_out_avg)

    # Last 100 periods should average within 5 % of reference
    last_mean = float(np.mean(vout_history[-100:]))
    err_pct = abs(last_mean - Vref) / Vref * 100.0
    assert err_pct < 5.0, (
        f"buck closed-loop vout = {last_mean:.3f} V (target {Vref} V, "
        f"err {err_pct:.2f} %) — expected within 5 %. "
        f"Tuned gains: Kp={gains.kp:.4f}, Ki={gains.ki:.2f}, "
        f"fc={gains.fc_hz:.0f} Hz, est_PM={gains.estimated_pm_deg:.1f}°"
    )


def test_picontroller_standalone_anti_windup():
    """The C++-backed PIController has back-calculation anti-windup."""
    pi = ps.PIController(Kp=0.1, Ki=10.0, output_min=0.0, output_max=1.0)

    # Saturate at the high limit for many steps
    for _ in range(100):
        out = pi.update(error=100.0, t=0.001)
        assert out == 1.0

    # Now drive a small negative error — output should respond quickly,
    # not stay glued to 1.0 (the anti-windup unwinds the integrator).
    out = pi.update(error=-0.5, t=0.101)
    assert out < 1.0, (
        f"PIController anti-windup is broken — output stayed at {out} "
        f"after a negative error following long saturation."
    )
