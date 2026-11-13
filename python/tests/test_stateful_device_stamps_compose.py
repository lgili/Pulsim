"""Adding one stateful device must not delete another's physics.

`run_transient` builds its Newton refresh by wrapping the caller's
callback once per stateful device kind — Coss, then Lauritzen diode,
then MNA-native PMSM, then IGBT tail, then saturable inductor. Each
wrapper is supposed to capture the ACCUMULATED callback so the chain
composes.

The saturable wrapper captured the RAW user callback instead, and it
is installed LAST. So it overwrote the chain with its own base: a
circuit containing a saturable inductor lost the Coss, Lauritzen,
PMSM and IGBT-tail stamps entirely. No error, no warning — those
devices simply simulated as though the feature were absent.

Measured before the fix, with a saturable inductor sitting on its
own isolated node, electrically connected to nothing in the power
stage: an IGBT's turn-off energy fell from 16.87 mJ to 0.045 mJ,
exactly the value it takes with no tail declared at all. A device
that is not even in the circuit's current path erased another
device's physics.

These tests pin the composition itself, which nothing else did —
every existing test exercises at most one stateful device kind.
"""

import numpy as np

import pulsim as p

ISOLATED_SAT = "an electrically disconnected saturable inductor"


def _igbt_eoff(with_sat, *, tail=True, others=False, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 600.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0,
                               2e-6, 0.0, 1e-8, 1e-8)
    b.add_inductor("Lload", "dc", "c", 500e-6, i0=100.0)
    b.add_diode("Dfw", "c", "dc", 1e3, 1e-9, 0.7)
    tail_kw = dict(tau_tail=1e-6, k_tail=0.30) if tail else {}
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0, **tail_kw)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)
    b.add_capacitor("Cs", "c", "gnd", 2e-9)
    if with_sat:
        b.add_voltage_source("Vs", "s", "gnd", 1.0)
        b.add_resistor("Rs", "s", "sm", 10.0)
        b.add_saturable_inductor("Lsat", "sm", "gnd", 1e-3, 5.0, 5e-5)
    if others:
        # Two more stateful kinds, isolated from the power stage. (An
        # isolated MNA PMSM is left out: at dt = 1 ns it sends every
        # step of the whole circuit to a dt/2 retry, flag on or off,
        # which moves E_off by 4e-5 for reasons unrelated to the chain.
        # Its BackwardEuler stamp has its own oracle in
        # core/tests/layer5_v1/test_backward_euler_stage.cpp.)
        b.add_voltage_source("Vo", "o", "gnd", 5.0)
        b.add_resistor("Ro", "o", "on", 100.0)
        b.add_nonlinear_capacitor("Cn", "on", "gnd", 1e-9, 25.0, 0.5)
        b.add_lauritzen_diode("Dl", "on", "gnd")
    n = b.graph.num_switches
    res = p.simulate(b, t_end=6e-6, dt=1e-9, engine="pwl",
                     switch_fn=lambda _t: p.SwitchStateMask(n), **kw)
    if kw.get("discontinuity_restart"):
        # the zero start and every Dfw flip restart
        assert res.n_restart_steps >= 2, res.n_restart_steps
    t = np.asarray(res.times)
    v = np.asarray(res.v("c"))
    i = np.asarray(res.i("Rsense"))
    m = t > 1.95e-6
    return float(np.trapezoid(np.maximum(v[m] * i[m], 0.0), t[m]))


def test_a_saturable_inductor_does_not_erase_the_igbt_tail():
    alone = _igbt_eoff(with_sat=False)
    with_sat = _igbt_eoff(with_sat=True)
    no_tail = _igbt_eoff(with_sat=False, tail=False)
    # The tail is worth ~380x the tail-free loss here, so "the
    # stamp ran" and "the stamp did not run" are unmistakable.
    assert alone > 100 * no_tail, (alone, no_tail)
    assert with_sat == alone or abs(with_sat - alone) < 1e-6 * alone, (
        f"{ISOLATED_SAT} changed E_off from {alone} to {with_sat}")


def test_the_saturable_inductor_itself_still_works_alongside():
    """The converse: the composition must not lose the saturable
    stamp either. Its own flux closure has to hold with an IGBT
    tail present in the same circuit."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0,
                              v_amplitude=50.0, frequency=1e3, phase=0.0)
    b.add_saturable_inductor("Ls", "a", "gnd", 1e-3, 5.0, 5e-5)
    # An IGBT with a tail, on its own branch, so both wrappers run.
    b.add_voltage_source("Vdc", "dc", "gnd", 100.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0,
                               5e-3, 0.0, 1e-8, 1e-8)
    b.add_igbt_level1("Q1", "dc", "e", "g", 1.5, 0.05, 5.0,
                      tau_tail=1e-6, k_tail=0.30)
    b.add_resistor("Re", "e", "gnd", 10.0)
    n = b.graph.num_switches
    res = p.simulate(b, t_end=40e-3, dt=2e-7, engine="pwl",
                     switch_fn=lambda _t: p.SwitchStateMask(n))
    t = np.asarray(res.times)
    i = np.asarray(res.i("Ls"))

    def cyc(k):
        m = (t >= k * 1e-3) & (t < (k + 1) * 1e-3)
        return float(i[m].mean())

    assert np.abs(i).max() > 25.0, np.abs(i).max()
    first, last = cyc(0), cyc(39)
    assert abs(last - first) / 40 < 1e-7 * abs(first), (first, last)


def test_the_chain_composes_on_a_restart_step():
    """add-discontinuity-restart-rule: the restart half-steps hand every
    wrapper a BackwardEuler stage through the same refresh_stage. With
    four stateful kinds in the chain (IGBT tail, saturable inductor,
    Coss, Lauritzen diode) the IGBT tail must still give the E_off it
    gives alone, restart steps included."""
    kw = dict(discontinuity_restart=True)
    alone = _igbt_eoff(with_sat=False, **kw)
    all_four = _igbt_eoff(with_sat=True, others=True, **kw)
    no_tail = _igbt_eoff(with_sat=False, tail=False, **kw)
    assert alone > 100 * no_tail, (alone, no_tail)
    assert abs(all_four - alone) < 1e-6 * alone, (alone, all_four)
