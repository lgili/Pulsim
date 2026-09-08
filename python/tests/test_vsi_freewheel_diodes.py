"""The three-phase VSI carries its freewheel diodes (v2.0).

`add_three_phase_vsi` used to build six bare switches. For an INDUCTIVE
load that is not an inverter: at the first PWM turn-off the load current
has nowhere to go, and what the user gets is not a clear error but a
Newton failure a few hundred microseconds in, naming some capacitor node.

Measured on a compressor drive while building a real model: the run
aborted at t = 168 us; adding the six antiparallel diodes by hand moved
the abort 75x further into the simulation. The diodes are part of the
device, exactly as a MOSFET's body diode is (audit C.1), so they are on
by default and `with_freewheel_diodes=False` is the opt-out.
"""

from __future__ import annotations

import numpy as np

import pulsim as p


def _bridge(with_diodes: bool):
    """A DC bus, the bridge, and an inductive load on one leg."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "bp", "bn", 300.0)
    b.add_resistor("Rgnd", "bn", "gnd", 1e-3)
    vsi = p.add_three_phase_vsi(
        b, "INV", vdc_pos="bp", vdc_neg="bn",
        out_a="ua", out_b="ub", out_c="uc",
        R_on=0.05, with_freewheel_diodes=with_diodes)
    # One leg drives an R-L to the negative rail: the classic clamped
    # inductive switching cell.
    b.add_inductor("Lload", "ua", "mid", 2e-3)
    b.add_resistor("Rload", "mid", "bn", 5.0)
    b.add_resistor("Rb", "ub", "bn", 1e6)
    b.add_resistor("Rc", "uc", "bn", 1e6)
    return b, vsi


def _chop(vsi, n_sw, fsw=5e3, duty=0.5):
    hi = list(vsi.high_side_switch_indices)

    def fn(t):
        m = p.SwitchStateMask(n_sw)
        if ((t * fsw) % 1.0) < duty:
            m.set(hi[0], True)
        return m
    return fn


def test_the_bridge_ships_its_freewheel_diodes():
    b, vsi = _bridge(True)
    kinds = {c["name"]: c["kind"] for c in b.components()}
    assert len(vsi.diode_names) == 6
    assert all(kinds[d] == "diode" for d in vsi.diode_names)
    # Same leg order as the switches, so a caller can pair them.
    assert vsi.diode_names[0].endswith("Dha")
    assert vsi.diode_names[1].endswith("Dla")


def test_an_inductive_leg_commutates_and_the_diode_takes_the_current():
    """The behaviour the diodes exist for: when the high side opens, the
    inductor current continues through the LOW-side diode instead of
    being interrupted."""
    b, vsi = _bridge(True)
    r = p.simulate(b, t_end=4e-3, dt=2e-6,
                   switch_fn=_chop(vsi, b.graph.num_switches), engine="pwl")
    t = np.asarray(r.times)
    i_l = np.asarray(r.i("Lload"))
    i_dl = np.asarray(r.i(vsi.diode_names[1]))      # the low-side diode
    m = t >= 2e-3
    assert np.all(np.isfinite(i_l))
    # The load current is continuous and positive — it never collapses.
    assert float(i_l[m].min()) > 1.0, float(i_l[m].min())
    # And the freewheel diode genuinely carries it part of the time.
    assert float(np.abs(i_dl[m]).max()) > 0.5 * float(i_l[m].max())


def test_without_them_the_answer_is_silently_absurd():
    """The opt-out is honest, and this is what the OLD default produced.

    A bridge with no antiparallel path does not refuse an inductive
    load — it answers, and the answer is nonsense. Measured on the same
    circuit, over the last 2 ms:

        with diodes:    v(ua) in [-1.7, 298.7] V,  i_L in [25.6, 33.6] A
        without:        v(ua) in [-42418, +42418] V, i_L in [-21.0, +21.1] A

    42 kV on a 300 V bus, and a load current that reverses, with no
    warning. That is the reason the diodes are now on by default: the
    failure mode was not an error message, it was a plausible-looking
    run with an impossible answer in it.
    """
    b, vsi = _bridge(False)
    r = p.simulate(b, t_end=4e-3, dt=2e-6,
                   switch_fn=_chop(vsi, b.graph.num_switches), engine="pwl")
    t = np.asarray(r.times)
    m = t >= 2e-3
    v_leg = np.abs(np.asarray(r.v("ua"))[m])
    # Nothing in a 300 V bridge can reach ten times the rail.
    assert float(v_leg.max()) > 3000.0, float(v_leg.max())

    b2, vsi2 = _bridge(True)
    r2 = p.simulate(b2, t_end=4e-3, dt=2e-6,
                    switch_fn=_chop(vsi2, b2.graph.num_switches), engine="pwl")
    t2 = np.asarray(r2.times)
    v2 = np.abs(np.asarray(r2.v("ua"))[t2 >= 2e-3])
    assert float(v2.max()) < 400.0, float(v2.max())
