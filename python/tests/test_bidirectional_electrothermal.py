"""Bidirectional electro-thermal coupling — audit C.2.

Pulsim already closed half the loop: the transient electro-thermal
observer recomputes each device's injected power from its current
junction temperature, so loss follows temperature. But the
electrical side never saw the temperature — the R_on in the MNA
matrix stayed whatever it was when the circuit was built.

    electrical -> lumped power -> thermal -> lumped power   (closed)
    thermal -> device parameters in the matrix              (OPEN)

WHAT THAT COSTS, honestly. On a well-designed buck: almost
nothing. Doubling a 5 mOhm MOSFET's R_on from 25 C to 125 C moves
v_out by -0.22 %, because the device drop is tiny next to the
load. That is NOT the motivation.

The motivation is the case where the answer IS the coupling. Two
identical MOSFETs in parallel sharing 100 A, one mounted worse
(R_th 2.0 vs 1.2 K/W), Rds_on doubling by 125 C:

                        frozen        coupled
    current imbalance     0.0 %         6.9 %
    hottest junction     65.0 C        72.1 C

The frozen model reports the imbalance as EXACTLY ZERO. That is
structural blindness, not a small error — current sharing between
paralleled devices is entirely a temperature effect. It also
under-reads the hottest junction by 7 C, against a 125 C limit.
"""

import numpy as np
import pytest

import pulsim as p

I_TOT = 100.0
R25 = 5e-3
ALPHA = 0.008          # Rds_on roughly doubles by 125 C
RTH_A, RTH_B = 1.2, 2.0
T_AMB = 40.0
# Thermally invisible next to RTH_A / RTH_B, so the analytic
# fixed point (independent paths to ambient) stays exact; a
# literal 0 is refused by add_shared_heatsink because it leaves
# T_sink unconstrained.
R_SA = 1e-6


def _analytic_fixed_point():
    """The coupled answer, solved directly. The simulation has to
    land on this."""
    def ron(t):
        return R25 * (1.0 + ALPHA * (t - 25.0))

    ta, tb = T_AMB, T_AMB
    ia = ib = I_TOT / 2
    for _ in range(500):
        ra, rb = ron(ta), ron(tb)
        ia = I_TOT * rb / (ra + rb)
        ib = I_TOT - ia
        ta = T_AMB + ia * ia * ra * RTH_A
        tb = T_AMB + ib * ib * rb * RTH_B
    return {"i_a": ia, "i_b": ib, "T_a": ta, "T_b": tb}


def _parallel_pair(coupled: bool, t_end=4.0, dt=2e-3,
                    update_every_s=2e-2):
    """Two paralleled devices fed 100 A, each with its own thermal
    path to ambient. `coupled=False` freezes R_on at 25 C, which is
    what the transient path could do before this module."""
    b = p.CircuitBuilder()
    b.add_current_source("Isrc", "gnd", "bus", I_TOT)
    b.add_resistor("Rq_A", "bus", "gnd", R25)
    b.add_resistor("Rq_B", "bus", "gnd", R25)

    devs = [
        p.HeatsinkDevice("A", [p.FosterStage(R_th_K_per_W=RTH_A,
                                              tau_s=0.05)],
                          R_th_case_to_sink_K_per_W=0.0),
        p.HeatsinkDevice("B", [p.FosterStage(R_th_K_per_W=RTH_B,
                                              tau_s=0.05)],
                          R_th_case_to_sink_K_per_W=0.0),
    ]
    hs = p.add_shared_heatsink(b, devs,
                                R_th_sink_to_amb_K_per_W=R_SA,
                                T_amb_C=T_AMB)

    tempco = [
        p.TempCoResistance(branch="Rq_A",
                            junction_node=hs.junction_nodes["A"],
                            R_ref_ohms=R25, a_per_C=ALPHA),
        p.TempCoResistance(branch="Rq_B",
                            junction_node=hs.junction_nodes["B"],
                            R_ref_ohms=R25, a_per_C=ALPHA),
    ]

    if not coupled:
        # The old shape: powers computed from the (frozen) circuit,
        # temperature never written back.
        tempco = [
            p.TempCoResistance(branch=t.branch,
                                junction_node=t.junction_node,
                                R_ref_ohms=t.R_ref_ohms,
                                a_per_C=0.0)
            for t in tempco
        ]

    # `simulate` builds the PWL cache itself and hands it to the hook,
    # the same way the MMC Thevenin driver is wired; the hook returns
    # the (observer, b_extra) pair to compose in.
    res = p.simulate(
        b, t_end=t_end, dt=dt, engine="pwl",
        on_cache=lambda cache: p.make_bidirectional_observer(
            b, cache, hs, tempco, update_every_s=update_every_s))
    x = np.asarray(res.states[-1])
    ja = b.node_id_of(hs.junction_nodes["A"])
    jb = b.node_id_of(hs.junction_nodes["B"])
    v_bus = float(np.asarray(res.v("bus"))[-1])
    return {
        "T_a": float(x[ja]), "T_b": float(x[jb]),
        "i_a": abs(float(np.asarray(res.i("Rq_A"))[-1])),
        "i_b": abs(float(np.asarray(res.i("Rq_B"))[-1])),
        "v_bus": v_bus,
    }


# ---------------------------------------------------------------
# The property.
# ---------------------------------------------------------------

def test_it_lands_on_the_analytic_fixed_point():
    """The coupled transient must settle where the coupled algebra
    says it does. This is the whole claim."""
    want = _analytic_fixed_point()
    got = _parallel_pair(coupled=True)
    assert got["T_a"] == pytest.approx(want["T_a"], rel=0.02), (got,
                                                                 want)
    assert got["T_b"] == pytest.approx(want["T_b"], rel=0.02), (got,
                                                                 want)
    assert got["i_a"] == pytest.approx(want["i_a"], rel=0.02), (got,
                                                                 want)
    assert got["i_b"] == pytest.approx(want["i_b"], rel=0.02), (got,
                                                                 want)


def test_the_frozen_model_reports_zero_imbalance():
    """The baseline this module exists to fix, pinned so it cannot
    later be mistaken for a modelling subtlety."""
    got = _parallel_pair(coupled=False)
    assert got["i_a"] == pytest.approx(got["i_b"], rel=1e-9)
    assert got["i_a"] == pytest.approx(I_TOT / 2, rel=1e-6)


def test_the_coupling_produces_the_imbalance():
    coupled = _parallel_pair(coupled=True)
    imb = abs(coupled["i_a"] - coupled["i_b"]) / (I_TOT / 2)
    assert imb == pytest.approx(0.069, abs=0.015), imb
    # The better-cooled device takes MORE current: it stays cooler,
    # so its R_on stays lower. (For a negative tempco this reverses
    # and stops being self-limiting — see `runaway_margin`.)
    assert coupled["i_a"] > coupled["i_b"]
    assert coupled["T_a"] < coupled["T_b"]


def test_the_frozen_model_under_reads_the_hottest_junction():
    """7 C against a 125 C limit is the difference between a design
    that passes and one that does not."""
    frozen = _parallel_pair(coupled=False)
    coupled = _parallel_pair(coupled=True)
    hot_frozen = max(frozen["T_a"], frozen["T_b"])
    hot_coupled = max(coupled["T_a"], coupled["T_b"])
    assert hot_coupled - hot_frozen == pytest.approx(7.1, abs=1.5)


# ---------------------------------------------------------------
# The tempco law and its guards.
# ---------------------------------------------------------------

def test_the_tempco_law():
    tc = p.TempCoResistance(branch="R", junction_node="Tj",
                             R_ref_ohms=5e-3, a_per_C=ALPHA)
    assert tc.at(25.0) == pytest.approx(5e-3)
    assert tc.at(125.0) == pytest.approx(9e-3)


def test_a_linear_tempco_cannot_go_negative():
    """Far below T_ref the straight line crosses zero. A negative
    resistance is a different circuit, not a cold one."""
    tc = p.TempCoResistance(branch="R", junction_node="Tj",
                             R_ref_ohms=5e-3, a_per_C=ALPHA)
    assert tc.at(-1000.0) > 0.0


@pytest.mark.parametrize("kw", [
    {"R_ref_ohms": 0.0},
    {"R_ref_ohms": -1.0},
    {"R_min_ohms": 0.0},
])
def test_bad_tempco_parameters_are_refused(kw):
    base = dict(branch="R", junction_node="Tj", R_ref_ohms=5e-3,
                a_per_C=ALPHA)
    base.update(kw)
    with pytest.raises(ValueError):
        p.TempCoResistance(**base)


def test_an_unknown_junction_node_is_refused():
    b = p.CircuitBuilder()
    b.add_resistor("Rq", "a", "gnd", 1.0)
    hs = p.add_shared_heatsink(
        b, [p.HeatsinkDevice("A", [p.FosterStage(R_th_K_per_W=1.0,
                                                  tau_s=0.05)])],
        R_th_sink_to_amb_K_per_W=R_SA)
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(1e-3)
    with pytest.raises(KeyError, match="not on this heatsink"):
        p.make_bidirectional_observer(
            b, cache, hs,
            [p.TempCoResistance(branch="Rq",
                                 junction_node="not_a_node",
                                 R_ref_ohms=1.0, a_per_C=0.0)],
            update_every_s=1e-2)


def test_update_every_s_has_no_default():
    """Refactorising is the expensive half of the loop and the right
    cadence follows from the user's own thermal time constants.
    Choosing one silently is how a coupling becomes wrong."""
    b = p.CircuitBuilder()
    b.add_resistor("Rq", "a", "gnd", 1.0)
    hs = p.add_shared_heatsink(
        b, [p.HeatsinkDevice("A", [p.FosterStage(R_th_K_per_W=1.0,
                                                  tau_s=0.05)])],
        R_th_sink_to_amb_K_per_W=R_SA)
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(1e-3)
    tc = [p.TempCoResistance(branch="Rq",
                              junction_node=hs.junction_nodes["A"],
                              R_ref_ohms=1.0, a_per_C=0.0)]
    with pytest.raises(TypeError):
        p.make_bidirectional_observer(b, cache, hs, tc)
    with pytest.raises(ValueError, match="update_every_s"):
        p.make_bidirectional_observer(b, cache, hs, tc,
                                       update_every_s=0.0)


# ---------------------------------------------------------------
# Runaway.
# ---------------------------------------------------------------

def test_runaway_margin_is_the_loop_gain():
    """G = I^2 * R_ref * a * R_th. At G >= 1 there is no fixed point
    at all, and a transient run would just climb — a simulation
    artefact rather than an answer."""
    tc = p.TempCoResistance(branch="Rq", junction_node="Tj",
                             R_ref_ohms=R25, a_per_C=ALPHA)
    ok = p.runaway_margin([tc], {"Rq": 50.0}, {"Rq": RTH_A})
    assert ok["Rq"]["gain"] == pytest.approx(50.0 ** 2 * R25
                                              * ALPHA * RTH_A)
    assert ok["worst"]["stable"]

    bad = p.runaway_margin([tc], {"Rq": 250.0}, {"Rq": 4.0})
    assert not bad["worst"]["stable"]


def test_a_negative_tempco_is_stable_for_a_single_device():
    """Which is why the paralleling case is the one that bites: the
    single-device loop is fine, the sharing is not."""
    tc = p.TempCoResistance(branch="Rq", junction_node="Tj",
                             R_ref_ohms=R25, a_per_C=-0.004)
    out = p.runaway_margin([tc], {"Rq": 200.0}, {"Rq": 5.0})
    assert out["worst"]["gain"] < 0.0
    assert out["worst"]["stable"]


# ---------------------------------------------------------------
# The hook's own guards. It refactors the fixed-step engine's PWL
# cache mid-run; on an engine with no such cache it would never
# fire and the loop it was closing would silently stay open.
# ---------------------------------------------------------------

def _hooked_circuit():
    b = p.CircuitBuilder()
    b.add_current_source("Isrc", "gnd", "bus", I_TOT)
    b.add_resistor("Rq", "bus", "gnd", R25)
    hs = p.add_shared_heatsink(
        b, [p.HeatsinkDevice("A", [p.FosterStage(R_th_K_per_W=RTH_A,
                                                  tau_s=0.05)])],
        R_th_sink_to_amb_K_per_W=R_SA, T_amb_C=T_AMB)
    tc = [p.TempCoResistance(branch="Rq",
                              junction_node=hs.junction_nodes["A"],
                              R_ref_ohms=R25, a_per_C=ALPHA)]
    hook = lambda cache: p.make_bidirectional_observer(  # noqa: E731
        b, cache, hs, tc, update_every_s=2e-2)
    return b, hook


def test_on_cache_is_refused_on_dsed():
    b, hook = _hooked_circuit()
    with pytest.raises(ValueError, match="on_cache"):
        p.simulate(b, t_end=1e-3, engine="dsed", on_cache=hook)


def test_on_cache_blocks_the_trbdf2_router():
    """`engine='auto'` must not route a hooked run onto TR-BDF2."""
    b, hook = _hooked_circuit()
    why = p._trbdf2_blockers(
        b, dt=None, step_observer=None, closed_loops=[],
        on_cache=hook, live_stream=None, progress=False,
        start_from_dc_op=False, strict_event_iterations=False,
        switch_fn=None, controller_period=None, max_dt_halvings=None,
        store_every=None, mmc_arms=None,
        enable_substep_state_correction=None,
        inductor_freeze_di_max=None, inductor_abs_clamp=None)
    assert any("on_cache" in w for w in why), why


def test_a_hook_returning_garbage_is_refused_by_name():
    b, _ = _hooked_circuit()
    with pytest.raises(TypeError, match="on_cache"):
        p.simulate(b, t_end=1e-3, dt=1e-3, engine="pwl",
                   on_cache=lambda cache: 42)
