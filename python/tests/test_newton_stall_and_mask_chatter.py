"""Two engine defects, characterised: the Newton stall at h_min and
the diode-mask chatter inside Newton.

Both were queued as "the absolute tolerances are too tight at a stiff
step". Measurement says they are two different defects, and that the
absolute step tolerance is LOAD-BEARING.

DEFECT 1 — TR-BDF2 stalls at h_min after a hard switching edge.
`test_saturable_transformer.py::test_engines_agree_below_saturation`
(a strict xfail) aborts with

    Newton failed at t = 4e-06 with h already at h_min —
    failed to converge after 50 iterations
    (||dx||_inf = 8.787e-08 worst at node T1.pm,
     ||residual||_inf = 7.629e-06 worst at the branch equation of
     current through inductor T1.m)

T1.m is a LINEAR inductor row, stamped in conductance form and so
scaled by 2L/h: at h_min that residual is a current error of ~1e-13 A.
The circuit is 48 V, so the step is nine converged digits. Both norms
are at their noise floor and the run aborts anyway, because
`tol_newton_dx` and `tol_newton_res` are ABSOLUTE
(solver/options.hpp:83-87). `nonlinear_solve.hpp` already carries a
scale-relative RESIDUAL acceptance — gated behind the absolute step
test, so it never fires here.

Three ways to make the step test scale-aware were implemented and
measured; all three are refuted, which is why this is pinned:

* dx scaled by the RHS norm (the scale the residual branch uses):
  far too loose. |b_constant| reaches ~1e7 at a landing step, so the
  gate becomes 1e-9·1e7 = 1e-2 — a ten-millivolt step accepted as
  converged. The flyback then returned 3.53 V against the fixed
  engine's 15.31 V, and the TR-BDF2 controller ground on a Lauritzen
  recovery until max_steps (10 M steps; the suite went from 168 s to
  602 s), because every accepted-but-unconverged iterate perturbs the
  LTE estimate the step controller runs on.
* dx scaled by the SOLUTION norm: too tight. The stall sits at
  8.787e-08 on a 48 V circuit and 1e-9·48 = 4.8e-08 still rejects it.
* dx per unknown, SPICE-style (|dx_i| <= 1e-6·|x_i| + abstol): clears
  the stall, and breaks something that matters. The DC
  operating-point cascade depends on the absolute test to FAIL:
  `test_dc_operating_point.py::test_gmin_stepping_solves_what_the_direct_solve_cannot`
  pins that the direct solve cannot do a stiff diode chain, so `auto`
  falls through to gmin stepping. With the relative test the direct
  solve "converges" — residual 1.07e-11, genuinely a root — to a
  DIFFERENT root, 10.67 V away from the homotopy's answer, and the
  cascade stops there. Five tests (3 Python, 2 C++) pin that
  fall-through. An absolute step test is what keeps Newton from
  stopping on a spurious branch of a multi-root system, so making it
  relative trades an abort for a silent wrong operating point.

What that leaves: the stall is real and the fix is NOT a tolerance
change. It needs the step controller not to drive h to h_min here
(why is a landing step of ~1e-12 s attempted at all?), or a
row-equilibrated Jacobian so the 2L/h rows do not set the scale.

DEFECT 2 — the fixed engine fails at 10 ns in the PSFB rectifier
whenever a Newton device is in the primary, and the cause is DIODE-MASK
CHATTER INSIDE the Newton loop, not tolerances. Measured below: with
the four rectifier diodes as PWL `add_diode`, a saturable or
hysteretic primary inductor aborts at t = 1.5e-6 with ||dx|| = 0.76 V
at node vout and ||residual|| = 5.4e-10 — a bouncing iterate at a
near-zero residual. Replace those four diodes with smooth Shockley
junctions and the SAME circuit runs and agrees with the linear
reference to 0.13 %. A plain linear inductor also runs, because
without a Newton device the diodes are decided by the OUTER event
iteration; with one, `make_combined_diode_mosfet_refresh` re-decides
them every Newton iteration and at a stiff step no single mask is
self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

V_BUS, F_PWM, PHASE = 100.0, 100e3, 1.0
N_PRIMARY, L_M_PATH, A_CORE, L_GAP = 12, 0.10, 3.5e-4, 1.5e-3


def _psfb(primary, rectifier="pwl"):
    """The plant of examples/scripts/run_psfb_hysteretic_inloop.py,
    with the primary inductor and the rectifier diode kind swapped."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vbus", "vbus", "gnd", V_BUS)
    for name, frm, to in [("HS_A", "vbus", "mid_a"), ("LS_A", "mid_a", "gnd"),
                          ("HS_B", "vbus", "mid_b"), ("LS_B", "mid_b", "gnd")]:
        b.add_switch(name, frm, to, g_on=1e3, g_off=1e-9)
    for name, a, c in [("D_HS_A", "mid_a", "vbus"), ("D_LS_A", "gnd", "mid_a"),
                       ("D_HS_B", "mid_b", "vbus"), ("D_LS_B", "gnd", "mid_b")]:
        b.add_diode(name, a, c, 1e3, 1e-9)
    if primary == "hysteretic":
        p.add_hysteretic_inductor(
            b, name="L_leak", from_node="mid_a", to_node="pri_pos",
            params=p.reference_material("ferrite_n87"), N_turns=N_PRIMARY,
            l_m=L_M_PATH, A_core=A_CORE, l_gap=L_GAP)
    elif primary == "saturable":
        b.add_saturable_inductor("L_leak", "mid_a", "pri_pos", 37.7e-6, 30.0, 1e-6)
    else:
        b.add_inductor("L_leak", "mid_a", "pri_pos", 37.7e-6)
    b.add_transformer("T1", p_from="pri_pos", p_to="mid_b",
                      s_from="sec_pos", s_to="sec_neg",
                      L_p=40e-6, L_s=10e-6, k=0.99)
    quad = [("D1", "sec_pos", "rect_pos"), ("D2", "sec_neg", "rect_pos"),
            ("D3", "rect_neg", "sec_pos"), ("D4", "rect_neg", "sec_neg")]
    for name, a, c in quad:
        if rectifier == "shockley":
            b.add_shockley_diode(name, a, c, I_S=1e-12)
        else:
            b.add_diode(name, a, c, 1e3, 1e-9)
    b.add_inductor("L_out", "rect_pos", "vout", 100e-6)
    b.add_capacitor("C_out", "vout", "rect_neg", 47e-6)
    b.add_resistor("R_L", "vout", "rect_neg", 10.0)
    return b


def _run(primary, dt, rectifier="pwl"):
    b = _psfb(primary, rectifier)
    sw = p.make_phase_shift_full_bridge_fn(
        switching_frequency=F_PWM, phase_shift=PHASE,
        leg_a_hs_idx=0, leg_a_ls_idx=1, leg_b_hs_idx=2, leg_b_ls_idx=3,
        num_switches=b.graph.num_switches, dead_time=100e-9)
    res = p.simulate(b, t_end=0.2e-3, dt=dt, switch_fn=sw, engine="pwl")
    return float(np.asarray(res.v("vout"))[-1])


def test_at_a_moderate_step_every_primary_agrees():
    """The control: at 50 ns all three run and land together, so the
    Newton devices are not the problem per se."""
    v = {k: _run(k, 50e-9) for k in ("linear", "saturable", "hysteretic")}
    assert v["saturable"] == pytest.approx(v["linear"], rel=0.01), v
    assert v["hysteretic"] == pytest.approx(v["linear"], rel=0.05), v


@pytest.mark.xfail(strict=True,
                   reason="the PWL diode mask is re-decided inside the Newton "
                          "loop once a Newton device is present, and at 10 ns "
                          "no single mask is self-consistent: the iterate "
                          "bounces by 0.76 V at node vout while the residual "
                          "sits at 5.4e-10. See the module docstring; the "
                          "smooth-diode control below is the proof.")
@pytest.mark.parametrize("primary", ["saturable", "hysteretic"])
def test_a_newton_device_survives_a_stiff_step(primary):
    v = _run(primary, 10e-9)
    assert v == pytest.approx(_run("linear", 10e-9), rel=0.02), v


def test_the_control_smooth_diodes_make_the_same_circuit_converge():
    """THE PROOF that the mask is the culprit, not the step size and not
    the flux device: swap only the rectifier's four PWL diodes for
    smooth Shockley junctions — nothing else — and the saturable
    primary runs at 10 ns and agrees with the linear reference."""
    v_lin = _run("linear", 10e-9, rectifier="shockley")
    v_sat = _run("saturable", 10e-9, rectifier="shockley")
    assert v_sat == pytest.approx(v_lin, rel=0.005), (v_sat, v_lin)
