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

import warnings

import numpy as np
import pytest

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
    # release: the stored 42 kV is the rule-off engine's answer (the premise
    # of this test); the default's is the next test
    r = p.simulate(b, t_end=4e-3, dt=2e-6, discontinuity_restart=False,
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


def test_without_them_the_default_engine_records_every_cut():
    """The same diode-less bridge under the release default (restart rule):
    the leg voltage stays on the rail (299.985 V) and the load current is cut
    from 13.49 A to the leakage at each turn-off — an impulse per turn-off,
    recorded by name: 20 cut-set records of 'Lload' by the leg's switches,
    0.0270 V*s and 0.182 J each at most (L*i and L*i^2/2), 3.50 J in all."""
    b, vsi = _bridge(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=4e-3, dt=2e-6,
                       switch_fn=_chop(vsi, b.graph.num_switches), engine="pwl")
    t = np.asarray(r.times)
    assert float(np.abs(np.asarray(r.v("ua"))[t >= 2e-3]).max()) < 300.0
    recs = list(r.reinit_records)
    assert len(recs) == 20
    assert all(x.kind == "cut_set" and x.states == ["Lload"]
               and x.devices == ["INV__HSa", "INV__LSa"] for x in recs)
    # truth: the R-L charge from 0 through R_on + R_load + R_gnd for the on-time,
    # i(t_on) = V/R (1 - exp(-R t_on / L)); the gate has no edge oracle, so an
    # edge is taken at a grid point next to it (F-SB-2): each of the two edges
    # moves t_on by at most one dt (13.022 / 13.256 / 13.488 A for -dt, 0, +dt)
    R, L = 5.0 + 0.05 + 1e-3, 2e-3
    lo, hi = (300.0 / R * (1.0 - np.exp(-R * t_on / L)) for t_on in (1e-4 - 2e-6, 1e-4 + 2e-6))
    for x in recs:   # each record is L*i and L*i^2/2 of the current it cut
        i_x = abs(x.lambda_) / L
        assert lo * (1 - 1e-3) <= i_x <= hi * (1 + 1e-3), (i_x, lo, hi)
        assert x.dW == pytest.approx(0.5 * L * i_x ** 2, rel=1e-6)
    # release default (split): the opaque edges are bisected (F-SB-2), t_on = 1e-4 exactly: 13.256 A
    mid = 300.0 / R * (1.0 - np.exp(-R * 1e-4 / L))
    # measured 2.6e-5 off (trapezoid at h = tau/200 + the R_off leakage); the +-dt values are 1.75e-2 away
    assert float(np.asarray(r.i("Lload")).max()) == pytest.approx(mid, rel=1e-4)


def test_floating_neutral_vsi_runs_on_the_exact_engine():
    """add-ai-usage-guide 12.6: the exact engine refused a VSI whose star
    neutral floats, with a message listing long-fixed causes and a design
    note no wheel ships. It runs now; truth for a balanced R-L star:
    KCL at the neutral (the phase currents sum to 0) and so
    v_n = (v_a + v_b + v_c) / 3 at every sample."""
    import numpy as np

    from pulsim import _dsed_dispatch

    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                              out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    spwm = p.make_three_phase_spwm_fn(
        10e3, 50.0, 0.8, p.ThreePhaseLegIndices(*v.switch_indices),
        b.graph.num_switches, 1e-6)
    r = p.simulate(b, 2e-3, engine="dsed", switch_fn=spwm)
    assert r.engine_used == "dsed"
    i_sum = r.i("La") + r.i("Lb") + r.i("Lc")
    assert np.max(np.abs(i_sum)) < 1e-9 * np.max(np.abs(r.i("La")))
    v_n = (r.v("a") + r.v("b") + r.v("c")) / 3.0
    assert np.max(np.abs(r.v("n") - v_n)) < 1e-9 * 600.0
    msg = _dsed_dispatch._BRIDGE_FALLBACK_MESSAGE
    assert "notes/" not in msg and "floating capacitors" not in msg


@pytest.mark.parametrize("dt", [2e-6, 2e-7])
def test_a_cut_that_destroys_energy_warns_by_energy_whatever_dt(dt):
    """Orchestrator decision 2026-11-23: the hazard of the diode-less bridge is
    the energy each cut destroys, not the voltage one step happens to show
    (45 x the rail at dt = 2e-6, under the voltage check's factor 100). One
    warning per run, same text at both steps. Truth: each record removes
    L i^2 / 2 of the current it cut, i = V/R (1 - exp(-R t_on / L)) with
    t_on = 1e-4 +- dt (F-SB-2), so the total of the 20 cuts lies in
    20 L i(t_on -+ dt)^2 / 2."""
    import re
    b, vsi = _bridge(False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = p.simulate(b, t_end=4e-3, dt=dt,
                       switch_fn=_chop(vsi, b.graph.num_switches), engine="pwl")
    msgs = [str(w.message) for w in caught if "destroyed" in str(w.message)]
    assert len(msgs) == 1, msgs
    m = re.search(r"(\d+) instantaneous current cut\(s\) of 'Lload' destroyed (\S+) J", msgs[0])
    assert m, msgs[0]
    R, L = 5.0 + 0.05 + 1e-3, 2e-3
    lo, hi = (300.0 / R * (1.0 - np.exp(-R * t_on / L)) for t_on in (1e-4 - dt, 1e-4 + dt))
    n = int(m.group(1))
    assert n == len(r.reinit_records) == 20
    assert n * 0.5 * L * lo ** 2 * (1 - 1e-3) <= float(m.group(2)) <= n * 0.5 * L * hi ** 2 * (1 + 1e-3)
    assert [d.code for d in r.diagnostics["messages"]].count("impulse_energy") == 1


def test_a_locator_residual_does_not_warn():
    """The threshold (1e-6 of the largest stored energy) is far above a
    locator's rounding-level record (F-Y-4: 2e-15 J) and far below a real cut
    (L i^2 / 2 = the stored energy itself): the helper on two synthetic records."""
    from types import SimpleNamespace as NS
    b, _ = _bridge(True)
    L, i_max = 2e-3, 10.0                         # stored: L i^2 / 2 = 0.1 J
    res = NS(i=lambda name: np.array([0.0, i_max]) if name == "Lload" else np.zeros(2),
             v=lambda node: np.zeros(2))
    rec = lambda dW: NS(kind="cut_set", dW=dW, cause="", devices=["X"], states=["Lload"])  # noqa: E731
    for dW, warns in ((2e-15, False), (0.5 * L * i_max ** 2, True)):
        res.reinit_records = [rec(dW)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p._warn_impulse_energy(res, b, "record")
        assert bool([w for w in caught if "destroyed" in str(w.message)]) is warns, dW
    with warnings.catch_warnings(record=True) as caught:   # 'error' refuses instead
        warnings.simplefilter("always")
        p._warn_impulse_energy(res, b, "error")
    assert not caught


@pytest.mark.parametrize("dt,warns", [(2e-6, False), (2e-7, True)])
def test_the_kernel_object_reads_the_cut_set_records_too(dt, warns):
    """The C++ twin of the voltage check (`sim::Simulation::check_voltage_`)
    reads the restart rule's records as simulate() does: the largest cut is
    the spike |lambda| / dt = L i / dt one step would hold, judged against
    100 x the 300 V rail. Truth: i(t_on) = V/R (1 - exp(-R t_on / L)),
    t_on = 1e-4 +- dt, so L i / dt is ~1.3e4 V at dt = 2e-6 (no warning)
    and ~1.3e5 V at dt = 2e-7 (warning, naming that peak)."""
    import re
    b, vsi = _bridge(False)
    sim = p.Simulation(b.build(), p.RunOptions(t_end=4e-3, dt=dt, engine="pwl"))
    sim.set_switch_fn(_chop(vsi, b.graph.num_switches))
    res = sim.run()
    assert res.reinit_records
    R, L = 5.0 + 0.05 + 1e-3, 2e-3
    lo, hi = (L * 300.0 / R * (1.0 - np.exp(-R * t_on / L)) / dt for t_on in (1e-4 - dt, 1e-4 + dt))
    assert bool(hi < 100 * 300.0) is not warns and bool(lo > 100 * 300.0) is warns   # the premise
    d = [x for x in sim.diagnostics if x["code"] == "implausible_voltage"]
    assert bool(d) is warns, sim.diagnostics
    if warns:
        peak = float(re.search(r"reached (\S+) V", d[0]["message"]).group(1))
        assert lo * (1 - 1e-3) <= peak <= hi * (1 + 1e-3), (peak, lo, hi)
