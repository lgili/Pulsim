"""Optimisation x control graph (integration batch 8): a gate that follows a
controller cannot be shot as a periodic orbit of the circuit's true state
(the controller's state is not in it and the period runs run no
controller), so it is refused by name on every engine and by the
evaluator, instead of returning the plant's orbit under a frozen duty."""
from __future__ import annotations

import warnings

import pytest

import pulsim as p
from pulsim.design import Evaluator, SteadySpec

T = 1e-5


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    return b


def _pi_loop(b):
    vout = b.node_id_of("vout")
    pi = p.PIController(Kp=0.01, Ki=50.0, output_min=0.05, output_max=0.95)
    return p.control.bind_pi_to_switch(
        b, pi=pi, measured=lambda x: x[vout], setpoint=12.0,
        switch="S", freq=1 / T)


def _graph_gate(b):
    ch = p.MixedDomainBlockChain()
    ch.add("pwm", p.PwmGenerator(frequency=1 / T), inputs=dict(
        duty=0.5, t="time"), output="g")
    ch.make_step_observer(b, dt=T / 100)
    return ch.make_pwm_switch_fn("g", num_switches=b.graph.num_switches,
                                 switch_idx=b.switch_index_of("S"))


ENGINES = [("exact", {}), ("dsed", {}), ("pwl", {"dt": T / 100}), ("trbdf2", {})]


@pytest.mark.parametrize("engine,kw", ENGINES)
@pytest.mark.parametrize("gate", [lambda b: _pi_loop(b).switch_fn, _graph_gate],
                         ids=["bind_pi_to_switch", "graph_pwm_switch_fn"])
def test_a_closed_loop_gate_is_refused_by_name(gate, engine, kw):
    b = _buck()
    sf = gate(b)
    with pytest.raises(p.UnsupportedError) as e:
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine=engine, **kw)
    assert e.value.feature == "closed_loop"
    assert "frozen duty" in str(e.value)
    ok, why = p.periodic_steady_state_supported(b, T, sf, engine=engine, **kw)
    assert not ok and "frozen duty" in why


def test_the_open_loop_twin_is_still_solved_on_pwl():
    # control: the same plant with a plain PWM gate converges
    b = _buck()
    sf = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                              b.graph.num_switches)
    r = p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100)
    assert r.stable and r._ctx.loop == "fd-broyden"


def test_the_evaluator_refuses_a_closed_loop_gate():
    def build(d):
        b = _buck()
        return b, _pi_loop(b).switch_fn
    for steady in (SteadySpec(method="pss"), SteadySpec(method="transient")):
        ev = Evaluator(build, period=T, loads=("RL",), output="vout",
                       steady=steady)
        try:
            rec = ev.evaluate({"x": 1.0})
        except p.UnsupportedError as e:
            assert e.feature == "closed_loop"
            continue
        assert rec.status != "ok"
        assert "bind_pi_to_switch" in rec.reason


@pytest.mark.parametrize("engine,kw", ENGINES)
def test_a_hand_written_gate_around_duty_get_is_refused(engine, kw):
    b = _buck()
    duty_get, _obs, _hist = p.control.bind_pi_to_duty_callable(
        b, pi=p.PIController(Kp=0.01, Ki=50.0, output_min=0.05,
                             output_max=0.95),
        measured=lambda x: 0.0, setpoint=12.0, freq=1 / T)
    n, i = b.graph.num_switches, b.switch_index_of("S")

    def sf(t):   # the docstring's pattern: a closure over duty_get
        m = p.SwitchStateMask(n)
        m.set(i, p.control.pwm_on(t, T, duty_get()))
        return m
    with pytest.raises(p.UnsupportedError, match="bind_pi_to_duty_callable"):
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine=engine, **kw)


def _drive_source(b):
    b.add_resistor("Rd", "drv", "gnd", 1e3)
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="vout"), output="y")
    ch.drive_source(b, "y", ("v", "drv", "gnd"))


def _c_block(b):
    # a C block's output source holds the block's last output; the period
    # runs never tick the block (integration batch 10: was not refused)
    b.add_resistor("Rr", "ref", "gnd", 1e3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p.add_c_block(b, inputs=[("v", "vout")], outputs=[("v", "ref", "gnd")],
                      dt=T, fn=lambda t, dt, i, o, s: o.__setitem__(0, 0.5 * i[0]),
                      name="vref")


SOURCE_ROUTES = [(_drive_source, "drive_source"), (_c_block, "add_c_block")]


@pytest.mark.parametrize("engine,kw", ENGINES)
@pytest.mark.parametrize("form", ["builder", "circuit"])
@pytest.mark.parametrize("route,name", SOURCE_ROUTES,
                         ids=[n for _, n in SOURCE_ROUTES])
def test_a_controller_driven_source_is_refused(route, name, form, engine, kw):
    b = _buck()
    route(b)
    sf = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                              b.graph.num_switches)
    src = b.build() if form == "circuit" else b
    with pytest.raises(p.UnsupportedError, match=name) as e:
        p.periodic_steady_state(src, period=T, switch_fn=sf, engine=engine,
                                **kw)
    assert e.value.feature == "closed_loop"


@pytest.mark.parametrize("form", ["builder", "circuit"])
@pytest.mark.parametrize("route,name", SOURCE_ROUTES,
                         ids=[n for _, n in SOURCE_ROUTES])
def test_the_evaluator_refuses_a_controller_driven_source(route, name, form):
    def build(d):
        b = _buck()
        route(b)
        sf = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                  b.graph.num_switches)
        return (b.build() if form == "circuit" else b), sf
    ev = Evaluator(build, period=T, loads=("RL",), output="vout",
                   steady=SteadySpec(method="pss"))
    rec = ev.evaluate({"x": 1.0})
    assert rec.status == "refused"
    assert name in rec.reason and "evaluator runs no controller" in rec.reason


@pytest.mark.parametrize("k", ["discontinuity_restart",
                               "enable_substep_state_correction"])
def test_pwl_shooting_refuses_switching_off_the_restart_rule(k):
    b = _buck()
    sf = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                              b.graph.num_switches)
    with pytest.raises(p.UnsupportedError) as e:
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100, options={k: False})
    assert e.value.feature == k


def test_a_threaded_circuit_sweep_refuses_a_shared_controller():
    b = _buck()
    loop = _pi_loop(b)
    with pytest.raises(ValueError, match="share one controller"):
        p.sweep(b.build(), params={"RL.R": [4.0, 5.0]},
                kpi_fn=lambda r, d: {}, parallel=True, t_end=1e-4,
                dt=1e-7, switch_fn=loop.switch_fn,
                step_observer=loop.step_observer, verbose=False)


def test_a_combined_schedule_with_a_closed_loop_part_is_refused():
    # since G23 the loop's gate declares its edges, so a combination of it
    # has edges too: only the closed-loop mark the combination carries
    # (SwitchSchedule.follows_controller) keeps it refused (before G23:
    # refused as edgeless)
    b = _buck()
    sf = p.make_combined_switch_fn(b.graph.num_switches,
                                   [_pi_loop(b).switch_fn])
    assert sf.has_edges and sf.follows_controller
    # it survives a second combination
    assert p.make_combined_switch_fn(b.graph.num_switches,
                                     [sf]).follows_controller
    with pytest.raises(p.UnsupportedError, match="frozen duty"):
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100)
    open_loop = p.make_combined_switch_fn(b.graph.num_switches, [
        p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                             b.graph.num_switches)])
    assert not open_loop.follows_controller


@pytest.mark.parametrize("combine", [False, True])
def test_sensitivities_refuse_a_closed_loop_gate(combine):
    from pulsim.sensitivity import simulate_sensitivities
    b = _buck()
    sf = _pi_loop(b).switch_fn
    if combine:
        sf = p.make_combined_switch_fn(b.graph.num_switches, [sf])
    with pytest.raises(p.UnsupportedError) as e:
        simulate_sensitivities(b, 10 * T, sensitivities=["RL.R"],
                               switch_fn=sf)
    assert e.value.feature == "closed_loop"


@pytest.mark.parametrize("key", ["closed_loops", "step_observer"])
def test_the_evaluator_refuses_a_controller_in_its_options(key):
    b = _buck()
    loop = _pi_loop(b)
    val = [loop] if key == "closed_loops" else loop.step_observer
    with pytest.raises(p.UnsupportedError, match=key) as e:
        Evaluator(lambda d: _buck(), period=T, loads=("RL",), output="vout",
                  options={key: val})
    assert e.value.feature == "closed_loop"


def test_shooting_refuses_a_chain_overlay_as_b_extra_fn():
    # Integration batch 10 review: the check read a `_closed_loop` mark
    # that nothing sets; a chain's overlay is recognised by type
    b = _buck()
    b.add_resistor("Rd", "drv", "gnd", 1e3)
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="vout"), output="y")
    ch.drive_source(b, "y", ("v", "drv", "gnd"))
    obs = ch.make_step_observer(b, dt=T / 100)
    bex = obs._b_extra_fn                     # handed over as a plain option
    sf = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                              b.graph.num_switches)
    b2 = _buck()
    b2.add_resistor("Rd", "drv", "gnd", 1e3)
    b2.add_voltage_source("drv_src", "drv", "gnd", 0.0)
    with pytest.raises(p.UnsupportedError, match="driven-source overlay") as e:
        p.periodic_steady_state(b2, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100, options={"b_extra_fn": bex})
    assert e.value.feature == "closed_loop"
