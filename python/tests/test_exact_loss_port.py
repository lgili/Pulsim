"""add-in-kernel-event-losses 3.2 / 3.3: the loss port on the exact engine.

The same solver::LossTracker as pwl and trbdf2, fed from the event-driven
scheduler's step hooks (dsed/loss_port.hpp): the switching energy by the
reading rule on each event's two limits, the conduction energy as the exact
integral of v·i along each mode's closed form, the impulsive energy of a
re-initialisation booked to the switch that caused it. The closed-form truth
(the RL chopper of test_device_losses_variable_step) is there; here the buck
and the VSI against trbdf2 at tight tolerance, the energy balance, the
impulse, the run left untouched, and the pieces refused by name. Loss data
here is synthetic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim._result_views import pl_integral, pl_integral_product, step_start_states

from .test_device_losses_kernel import _switch_names, _vsi

_RNG = np.random.default_rng(7)
_AX = dict(v_axis=[0.0, 400.0, 800.0], i_axis=[0.0, 30.0, 60.0], tj_axis=[25.0, 150.0])
E_ON = p.LossTable(**_AX, energy=_RNG.uniform(1e-3, 5e-3, (3, 3, 2)))
E_OFF = p.LossTable(**_AX, energy=_RNG.uniform(1e-3, 5e-3, (3, 3, 2)))
VIN, RL, LL, CC = 24.0, 10.0, 100e-6, 47e-6


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", LL)
    b.add_capacitor("C", "vout", "gnd", CC)
    b.add_resistor("RL", "vout", "gnd", RL)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn, ["Q1", "D1"]


def _vsi_case():
    b, fn = _vsi()
    return b, fn, _switch_names(b)


def _run(make, t_end, engine, losses=True, **kw):
    b, fn, names = make()
    if losses:
        for n in names:
            b.set_device_losses(n, Tj=125.0, E_on=E_ON, E_off=E_OFF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return b, names, p.simulate(b, t_end, switch_fn=fn, engine=engine, **kw)


# t_end off every gate edge: an edge AT t_end is booked by trbdf2 and not
# fired by the exact engine (it belongs to the run after it).
@pytest.mark.parametrize("make,t_end", [(_buck, 202.5e-6), (_vsi_case, 2.03e-3)],
                         ids=["buck", "vsi"])
def test_per_device_energies_agree_with_trbdf2_at_tight_tolerance(make, t_end):
    _, names, ex = _run(make, t_end, "dsed")
    _, _, tr = _run(make, t_end, "trbdf2", rtol=1e-10, atol=1e-12)
    for n in names:
        a, c = ex.device_energy(n), tr.device_energy(n)
        assert (a["n_on"], a["n_off"]) == (c["n_on"], c["n_off"]), n
        assert a["n_on"] + a["n_off"] > 30, n
        for k in ("E_on", "E_off", "E_cond_electrical"):
            assert a[k] == pytest.approx(c[k], rel=1e-4), (n, k)


def test_the_buck_energy_balance_closes():
    """Source work = load + device conduction + stored energy (the devices are
    conductance PWL devices: their loss is their ∫v·i). The source and load
    integrals are the samples' lines from each step's right limit (1e-8 s
    steps); the device energies are the port's exact integrals."""
    b, names, r = _run(_buck, 202.5e-6, "dsed", dt=1e-8)   # dt = the step cap
    t = np.asarray(r.times)
    x = np.asarray(r.states)
    sw = np.asarray(r.switch_states, dtype=bool)
    xs, ms = step_start_states(r, x, sw)
    col = {b.graph.node_name(i): i for i in range(b.graph.num_nodes)}
    q = b.switch_index_of("Q1")

    def i_q(xx, mm):
        g = np.where(mm[:, q], 1e3, 1e-9)
        return g * (VIN - xx[:, col["sw"]])

    e_src = VIN * pl_integral(t, i_q(x, sw), i_q(xs, ms))
    vo, vo_s = x[:, col["vout"]], xs[:, col["vout"]]
    e_load = pl_integral_product(t, vo, vo, vo_s, vo_s) / RL
    i_l = np.asarray(r.i("L"))
    stored = 0.5 * CC * vo[-1] ** 2 + 0.5 * LL * i_l[-1] ** 2
    losses = sum(r.device_energy(n)["E_cond_electrical"] for n in names)
    assert e_src == pytest.approx(e_load + losses + stored, rel=1e-6)


def test_a_capacitor_loop_impulse_is_booked_to_the_switch_that_closed_it():
    """½·C1C2/(C1+C2)·ΔV² of the loop the ideal switch closes at 1 µs is its
    E_impulse, as on trbdf2. It opens at 2 µs: the step after the projection
    ends at an event, and its integral starts from the projected state
    (reviewer: it started from the pre-projection one and was refused)."""
    c1, c2, v1, v2 = 1e-6, 3e-6, 10.0, 2.0
    out = {}
    for engine in ("dsed", "trbdf2"):
        b = p.CircuitBuilder()
        b.add_capacitor("C1", "x", "gnd", c1, v1)
        b.add_capacitor("C2", "y", "gnd", c2, v2)
        b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
        k = b.switch_index_of("S")

        def fn(t, k=k, n=b.graph.num_switches):
            m = p.SwitchStateMask(n)
            m.set(k, 1e-6 <= t < 2e-6)
            return m
        tab = p.LossTable(v_axis=[10.0], i_axis=[1.0], tj_axis=[25.0],
                          energy=np.full((1, 1, 1), 1e-9))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.set_device_losses("S", Tj=25.0, E_on=tab, E_off=tab)
            r = p.simulate(b, t_end=5e-6, engine=engine, switch_fn=fn, auto_regularize=False)
        out[engine] = r.device_energy("S")
    dw = 0.5 * c1 * c2 / (c1 + c2) * (v1 - v2) ** 2
    e = out["dsed"]
    assert (e["n_on"], e["n_off"]) == (1, 1)
    assert e["E_impulse"] == pytest.approx(dw, rel=1e-8)
    assert e["E_impulse"] == pytest.approx(out["trbdf2"]["E_impulse"], rel=1e-9)


def test_a_record_only_port_leaves_the_exact_run_bit_identical():
    _, _, r0 = _run(_buck, 202.5e-6, "dsed", losses=False)
    _, _, r1 = _run(_buck, 202.5e-6, "dsed")
    assert np.asarray(r0.times).tobytes() == np.asarray(r1.times).tobytes()
    assert np.asarray(r0.states).tobytes() == np.asarray(r1.states).tobytes()
    assert r1.heat_injected == 0.0 and r1.device_energy("Q1")["n_on"] > 10


def _refused(b, fn=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(p.UnsupportedError) as ei:
            p.simulate(b, 1e-4, switch_fn=fn, engine="dsed", **kw)
    return ei.value


# a junction / heat node runs (the temperature loop, test_exact_junction_loop)
@pytest.mark.parametrize("what", ["table", "sine", "bdf2", "t_start"])
def test_what_the_exact_port_cannot_honour_is_refused_by_name(what):
    b, fn, _ = _buck()
    kw = {}
    if what == "table":
        b.set_device_losses("Q1", Tj=25.0, conduction="table",
                            V_on=p.ConductionTable([10.0], [25.0, 125.0], [1.0, 1.2]))
    else:
        b.set_device_losses("Q1", Tj=25.0, E_on=E_ON)
        if what == "sine":
            b.add_sine_voltage_source("Vs", "s", "gnd", 0.0, 1.0, 50.0)
            b.add_resistor("Rs", "s", "gnd", 1.0)
        elif what == "bdf2":   # a circuit without diodes (they refuse bdf2 first)
            b = p.CircuitBuilder()
            b.add_voltage_source("V", "in", "gnd", 10.0)
            b.add_switch("Q1", "in", "o", 1e3, 1e-9)
            b.add_resistor("R", "o", "gnd", 10.0)
            b.set_device_losses("Q1", Tj=25.0, E_on=E_ON)
            fn = p.make_pwm_switch_fn(10e3, 0.5, 0, 1)
            kw["integrator"] = "bdf2"
        else:
            kw["t_start"] = 1e-5
    e = _refused(b, fn, **kw)
    assert (e.engine, e.feature, e.device) == ("dsed", "device_losses", "Q1")
