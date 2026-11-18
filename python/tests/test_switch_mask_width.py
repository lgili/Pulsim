"""A caller's switch mask of the wrong width is refused on every engine.

Before: `combine_masks` / `SwitchStateMask.overlay` took the USER mask's
width, so a `switch_fn` returning a mask narrower than
`graph.num_switches` silently dropped the missing bits (a branch-form
diode's bit, or every diode's on dsed's overlay), and `.set(i)` past the
width wrote a bit nothing reads.
"""

import pytest

import pulsim as p

GON, GOFF = 1e3, 1e-9


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("R", "vout", "gnd", 50.0)
    return b


def _short(t):
    return p.SwitchStateMask(1)          # the circuit has 2 switch bits


def _wide(t):
    return p.SwitchStateMask(3)


def test_runs_from_this_worktree():
    assert "pulsim_wt/phase0" in p.__file__


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
@pytest.mark.parametrize("fn", [_short, _wide])
def test_simulate_refuses_a_mask_of_the_wrong_width(engine, fn):
    b = _buck()
    assert b.graph.num_switches == 2
    width = fn(0.0).size
    with pytest.raises(ValueError, match=rf"width {width}; this circuit has 2 switch bits"):
        p.simulate(b, t_end=1e-4, dt=1e-6, engine=engine, switch_fn=fn)


def test_simulate_runs_with_the_right_width():
    # Control: the same circuit and engine run with a full-width mask.
    b = _buck()
    on = p.SwitchStateMask(2)
    on.set(b.switch_index_of("S"), True)
    res = p.simulate(b, t_end=1e-4, dt=1e-6, engine="pwl", switch_fn=lambda t: on)
    assert res.v("vout")[-1] > 1.0


def test_native_schedule_of_the_wrong_width_is_refused():
    b = _buck()
    sched = p.make_pwm_switch_fn(1e5, 0.3, 0, 1, 0.0)    # 1-bit masks
    for engine in ("pwl", "trbdf2", "dsed"):
        with pytest.raises(ValueError, match="width 1; this circuit has 2"):
            p.simulate(b, t_end=1e-4, dt=1e-6, engine=engine, switch_fn=sched)


def test_run_transient_refuses_a_short_mask():
    b = _buck()
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(dt=1e-6)
    opts = p.SimulationOptions(t_start=0.0, t_end=1e-5, dt=1e-6)
    with pytest.raises(ValueError, match="width 0; this circuit has 2"):
        p.run_transient(cache, b.graph, b.pool, opts,
                        switch_fn=lambda t: p.SwitchStateMask(0))


def test_set_and_get_past_the_width_raise():
    m = p.SwitchStateMask(1)
    with pytest.raises(IndexError, match="bit 5 is out of range for a mask of width 1"):
        m.set(5, True)
    with pytest.raises(IndexError, match="out of range"):
        m.get(1)
    m.set(0, True)
    assert m.get(0)
