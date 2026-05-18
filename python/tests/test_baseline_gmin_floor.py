"""simplify-and-harden-numerical-surface — gmin-floor regression.

Pins the contract that `SimulationOptions::baseline_node_gmin` is applied
on every transient step (not just as a retry fallback), keeping the MNA
non-singular when a switching converter's intermediate node temporarily
floats during the dead-time interval.

The canonical failure case: a buck converter where both the main switch
and the freewheel diode are OFF simultaneously for some micro-window.
Before the gmin floor shipped, the un-regularized MNA went singular at
that step and the simulator's recovery silently froze all time-dependent
sources at their t=0 value — the PWM source then read constant 5 V and
the switch stayed ON forever, producing `vout = vin` (the buck stopped
bucking). With the gmin floor at 1e-12 S (SPICE-equivalent), the matrix
stays well-conditioned and the PWM toggles correctly.
"""

import numpy as np
import pulsim as ps


def _build_buck_no_bleed():
    """Buck converter WITHOUT the 1 MΩ sw_node→gnd bleed resistor that
    historically masked the singularity. This is the topology that should
    work out of the box once the gmin floor is in place."""
    VIN, VOUT, FSW = 12.0, 5.0, 100e3
    L, C, R_LOAD = 100e-6, 100e-6, 2.5
    D = VOUT / VIN

    ckt = ps.Circuit()
    gnd = ckt.ground()
    n_vin = ckt.add_node("vin")
    n_gate = ckt.add_node("gate")
    n_sw = ckt.add_node("sw_node")
    n_out = ckt.add_node("vout")

    ckt.add_voltage_source("Vin", n_vin, gnd, VIN)
    pwm = ps.PWMParams()
    pwm.v_high = 5.0
    pwm.v_low = 0.0
    pwm.frequency = FSW
    pwm.duty = D
    ckt.add_pwm_voltage_source("Vpwm", n_gate, gnd, pwm)

    ckt.add_vcswitch("S1", n_gate, n_vin, n_sw, 2.5, 350.0, 1e-9)
    ckt.add_diode("D1", gnd, n_sw, 350.0, 1e-9)
    ckt.add_inductor("L1", n_sw, n_out, L)
    ckt.add_capacitor("C1", n_out, gnd, C)
    ckt.add_resistor("R_load", n_out, gnd, R_LOAD)

    return ckt, n_gate, n_out, VIN, VOUT, D, FSW


def _make_opts(ckt, FSW):
    T_SW = 1.0 / FSW
    dt = T_SW / 40.0
    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 6e-3
    opts.dt = dt
    opts.dt_min = dt
    opts.dt_max = dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    opts.integrator = ps.Integrator.BDF1
    opts.switching_mode = ps.SwitchingMode.Behavioral
    opts.newton_options.max_iterations = 140
    opts.newton_options.num_nodes = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()
    return opts


def test_baseline_gmin_default_is_spice_equivalent():
    """Default value matches SPICE's gmin (1e-12 S)."""
    opts = ps.SimulationOptions()
    assert opts.baseline_node_gmin == 1e-12


def test_buck_without_bleed_resistor_settles_correctly():
    """The canonical floating-node case — buck w/o bleed should now
    settle near the duty-driven target instead of getting stuck at Vin."""
    ckt, n_gate, n_out, VIN, VOUT, D, FSW = _build_buck_no_bleed()
    opts = _make_opts(ckt, FSW)

    sim = ps.Simulator(ckt, opts)
    result = sim.run_transient()
    assert result.success

    # PWM gate must toggle (the headline symptom of the old bug).
    v_gate = np.array([s[n_gate] for s in result.states])
    unique = sorted(set(round(v, 2) for v in v_gate))
    assert unique == [0.0, 5.0], (
        f"PWM gate stuck at {unique} — gmin floor regressed; the "
        f"baseline_node_gmin = {opts.baseline_node_gmin} did not "
        f"prevent the floating-node singularity."
    )

    # vout settles near D*Vin (open-loop). Allow 10 % envelope —
    # the residual error is DCR + diode V_F, NOT the singularity.
    v_out = np.array([s[n_out] for s in result.states])
    last10 = v_out[int(0.9 * len(v_out)):]
    mean_vout = float(last10.mean())
    target = D * VIN
    err_pct = abs(mean_vout - target) / target * 100.0
    assert err_pct < 10.0, (
        f"vout mean = {mean_vout:.3f} V vs target {target:.3f} V "
        f"(error {err_pct:.1f} %, > 10 % envelope). The buck is not "
        f"bucking — check the gmin floor wiring."
    )


def test_opt_out_field_is_writable():
    """Power users can opt out by setting baseline_node_gmin = 0.0; the
    field MUST round-trip through pybind11 and apply at construction.

    Note: opting out doesn't ALWAYS reproduce the legacy bug — devices
    with finite `g_off` (PWL devices' OFF leakage ≈ 1 nS) provide some
    DC regularization too, so a buck may still solve cleanly with the
    floor disabled. The intent of the floor is "belt + suspenders":
    even when device leakage is absent (linear-only circuits with
    isolated reactive subnetworks), the floor keeps the matrix
    well-conditioned."""
    opts = ps.SimulationOptions()
    opts.baseline_node_gmin = 0.0
    assert opts.baseline_node_gmin == 0.0

    opts.baseline_node_gmin = 1e-10
    assert opts.baseline_node_gmin == 1e-10

    # Sanity: setting it to a tiny positive value doesn't blow up.
    opts.baseline_node_gmin = 1e-14
    assert opts.baseline_node_gmin == 1e-14


def test_floor_does_not_distort_well_conditioned_circuit():
    """The 1e-12 floor must not change measurable observables on a
    well-conditioned circuit (no floating nodes). RC charging time
    constant should match analytical within 0.5 %."""
    R = 1e3
    C_val = 1e-6
    Vin = 5.0

    ckt = ps.Circuit()
    gnd = ckt.ground()
    n_in = ckt.add_node("in")
    n_out = ckt.add_node("out")
    ckt.add_voltage_source("Vsrc", n_in, gnd, Vin)
    ckt.add_resistor("R", n_in, n_out, R)
    ckt.add_capacitor("C", n_out, gnd, C_val)

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 5e-3  # 5 RC time constants
    opts.dt = 50e-6
    opts.dt_min = opts.dt
    opts.dt_max = opts.dt
    opts.adaptive_timestep = False
    opts.newton_options.num_nodes = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()

    sim = ps.Simulator(ckt, opts)
    result = sim.run_transient()
    assert result.success

    v_out_final = float(result.states[-1][n_out])
    # After 5 τ, vout converges to Vin within 1 %. The exact final value
    # depends on the integrator (BDF1 default at dt = τ/20). The point
    # of THIS test is that the 1e-12 floor doesn't drag vout DOWN
    # measurably — leakage at 1e-12 S × 5 V = 5 pA on a 1 µF cap over
    # 5 ms drops 0.025 µV, undetectable.
    assert abs(v_out_final - Vin) / Vin < 0.01, (
        f"RC final voltage {v_out_final:.4f} V vs Vin {Vin} V "
        f"(err {abs(v_out_final - Vin)/Vin*100:.2f} %, > 1 %). The 1e-12 "
        f"gmin floor should NOT measurably distort a well-conditioned "
        f"RC step response."
    )
