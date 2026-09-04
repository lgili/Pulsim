"""engine='auto' is the DEFAULT, and it ROUTES.

v2.0 breaking change. `auto` does not mean "the variable-step
engine" — it means "pick the engine". The measurement that shaped
this: flipping the default to the variable engine outright broke
190 of the suite's tests (66 observers with no cadence, 51
`start_from_dc_op`, 15 nonlinear circuits, 13 MMC arms, …). With
routing, zero break.

The rule a user has to hold in their head is one line:

    dt given  ->  fixed step, exactly as before
    no dt     ->  the engine picks, and takes the variable-step
                  path whenever the circuit qualifies

so the new engine is opt-in by OMISSION, and `result.engine_used`
always says which one ran (plus `engine_route_reason` when `auto`
did not pick the variable one).
"""

import numpy as np
import pytest

import pulsim as p


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    return b


def _rc_unsupported():
    """A circuit the variable-step engine genuinely cannot serve.

    NO DEVICE KIND blocks the router any more — see
    `test_no_device_kind_blocks_the_router` below. Nonlinear diodes
    and MOSFETs were always served (each stage becomes a Newton
    solve), and the five devices that carry state each have a
    derived BDF2 second stage now. The saturable inductor was the
    last holdout and used to sit here.

    What remains is structural rather than physical: a C BLOCK
    samples its `fn` on a fixed dt grid, so a variable step has no
    grid to fire it on. It is attached to the BUILDER, which is what
    lets it stand in for the old fixture.
    """
    b = _rc()

    def _tick(t, dt, inp, out, state):
        out[0] = 0.0

    p.add_c_block(b, inputs=[], outputs=[("i", "n1", "gnd")],
                  dt=1e-4, fn=_tick)
    return b


def test_default_is_auto():
    import inspect
    sig = inspect.signature(p.simulate)
    assert sig.parameters["engine"].default == "auto"


def test_no_dt_linear_takes_the_variable_engine():
    res = p.simulate(_rc(), t_end=1e-3)      # no engine, no dt
    assert res.engine_used == "trbdf2"
    assert res.engine_route_reason is None
    t = np.asarray(res.times)
    ref = 5.0 * (1.0 - np.exp(-t / 1e-3))
    assert np.abs(np.asarray(res.v("n1")) - ref).max() < 1e-3


def test_explicit_dt_still_means_fixed_step():
    """The rule that keeps every existing script's answer.

    Reading a user's dt as the variable engine's step CEILING
    changed answers silently — measured 4.985 -> 4.968 V on a
    closed-loop buck. An explicit dt is a request, not a hint.
    """
    res = p.simulate(_rc(), t_end=1e-3, dt=1e-6)
    assert res.engine_used == "pwl"
    assert "explicit dt" in res.engine_route_reason
    # bit-identical to naming the engine outright
    ref = p.simulate(_rc(), t_end=1e-3, dt=1e-6, engine="pwl")
    assert np.array_equal(np.asarray(res.states),
                           np.asarray(ref.states))


def test_nonlinear_devices_are_served_not_routed_away():
    """Newton per stage: a real diode no longer sends a run to the
    fixed engine."""
    from pulsim import _pulsim as _k
    b = _rc()
    b.add_nonlinear_diode("D", "n1", "gnd", _k.IdealDiodeParams())
    res = p.simulate(b, t_end=1e-3)          # no dt
    assert res.engine_used == "trbdf2"


def test_unsupported_circuit_routes_and_says_why():
    res = p.simulate(_rc_unsupported(), t_end=1e-3, dt=1e-6)
    assert res.engine_used == "pwl"
    assert "C block" in res.engine_route_reason


def test_unsupported_and_no_dt_asks_for_one():
    with pytest.raises(ValueError) as e:
        p.simulate(_rc_unsupported(), t_end=1e-3)
    msg = str(e.value)
    assert "C block" in msg            # the blocker, named
    assert "pass dt=" in msg           # and what to do about it


def test_trbdf2_refuses_where_auto_routes():
    """Naming the engine means you want to KNOW."""
    with pytest.raises(ValueError, match="C block"):
        p.simulate(_rc_unsupported(), t_end=1e-3, dt=1e-6,
                    engine="trbdf2")


def test_no_device_kind_blocks_the_router():
    """The statement the fixture above no longer makes. Every device
    that carries state has a derived TR-BDF2 second stage, so no
    CIRCUIT routes away from the variable engine — only kwargs and
    builder-level features do. A device blocker creeping back would
    silently cost that device's users the adaptive engine, which is
    exactly the kind of regression nothing else here would catch."""
    from pulsim import _pulsim as _k
    b = _rc()
    b.add_saturable_inductor("Lsat", "n1", "gnd", L_0=1e-3,
                              I_sat=2.0, L_residual=1e-4)
    b.add_nonlinear_diode("D", "n1", "gnd", _k.IdealDiodeParams())
    b.add_lauritzen_diode("Dl", "n1", "gnd", tau=1e-7, T_M=1e-8)
    b.add_igbt_level1("Q", "in", "e", "g", 1.5, 0.05, 5.0,
                      tau_tail=1e-6, k_tail=0.3)
    b.add_resistor("Re", "e", "gnd", 1.0)
    b.add_voltage_source("Vg", "g", "gnd", 0.0)
    b.add_pmsm_mna("M", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=0.5, L_d=1e-3, L_q=3e-3, psi_pm=0.05,
                   pole_pairs=4, J=1e-3, B=1e-4)
    res = p.simulate(b, t_end=1e-4)      # no dt — router's choice
    assert res.engine_used == "trbdf2", res.engine_route_reason


def test_trbdf2_reads_dt_as_the_step_ceiling():
    res = p.simulate(_rc(), t_end=1e-3, dt=1e-5, engine="trbdf2")
    assert res.engine_used == "trbdf2"
    assert np.diff(np.asarray(res.times)).max() <= 1e-5 * 1.001


def test_routed_run_still_warns_about_ignored_kwargs():
    """Validation ran against the engine the caller ASKED for, so
    a kwarg the destination ignores would slip through silently."""
    with pytest.warns(UserWarning, match="ignored by the "
                                          "fixed-step engine"):
        p.simulate(_rc(), t_end=1e-3, dt=1e-6, rtol=1e-6)


def test_engine_used_is_stamped_on_every_path():
    assert p.simulate(_rc(), t_end=1e-3, dt=1e-6,
                       engine="pwl").engine_used == "pwl"
    assert p.simulate(_rc(), t_end=1e-3,
                       engine="trbdf2").engine_used == "trbdf2"
    assert p.simulate(_rc(), t_end=1e-3, rtol=1e-6,
                       engine="dsed").engine_used == "dsed"


def test_unknown_engine_lists_the_valid_ones():
    with pytest.raises(ValueError, match="trbdf2"):
        p.simulate(_rc(), t_end=1e-3, engine="bogus")
