"""add-complementarity-mode-selection tasks 2 and 3: the C++ selector against the prototype.

* task 2: the C++ Lemke (``pulsim._pulsim._lcp_lemke``) against the numpy oracle on the 218 LCPs
  the prototype posed on the five families, and on random PSD instances;
* task 3: the C++ port model on the conductance form (g_on = 1e3, g_off = 1e-9, R_on = 1/g_on),
  posed at every state where the prototype decided a mode: the port LCP equals the prototype's
  (same circuit with the resistors made explicit) and the selected mode equals the prototype's.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
import pulsim  # noqa: E402
from tests.oracles import lcp_reference as L  # noqa: E402
from tests.oracles import lcp_pulsim_bridge as B  # noqa: E402

K = pulsim._pulsim
INSTANCES = json.load(open(os.path.join(os.path.dirname(__file__), "oracles", "lcp_instances.json")))


def test_cpp_lemke_reproduces_the_oracle_on_the_family_lcps():
    max_piv = 0
    for o in INSTANCES:
        M, q = np.array(o["M"]), np.array(o["q"])
        z, w, piv, status = K._lcp_lemke(M, q)
        zr, wr, pivr, str_ = L.lemke(M, q)
        assert (piv, status) == (pivr, str_)
        sc = max(1.0, np.abs(q).max(), np.abs(zr).max())
        np.testing.assert_allclose(z, zr, rtol=0, atol=1e-9 * sc)
        np.testing.assert_allclose(w, wr, rtol=0, atol=1e-9 * sc)
        n, ze, we = K._lcp_enumerate(M, q)
        assert n >= 1          # degenerate ties: counts depend on the singular-minor solver
        max_piv = max(max_piv, piv)
    assert max_piv == 6


def test_cpp_lemke_and_enumeration_on_random_definite_instances():
    rng = np.random.default_rng(7)
    for n in range(100):
        k = 1 + n % 8
        A, S = rng.standard_normal((k, k)), rng.standard_normal((k, k))
        M = A @ A.T + (S - S.T) + 1e-2 * np.eye(k)
        q = 3.0 * rng.standard_normal(k)
        z, w, piv, status = K._lcp_lemke(M, q)
        zr, _, pivr, _ = L.lemke(M, q)
        assert piv == pivr
        cnt, ze, _ = K._lcp_enumerate(M, q)
        assert cnt == 1
        np.testing.assert_allclose(z, ze, atol=1e-8 * max(1.0, np.abs(ze).max()))


def test_cpp_lemke_secondary_ray_and_trivial():
    M = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert K._lcp_lemke(M, np.array([-1.0, -1.0]))[3] == "secondary ray"
    z, w, piv, status = K._lcp_lemke(M, np.array([0.0, 2.0]))
    assert (piv, status) == (0, "trivial") and not z.any()


# Families with a leak-only island in some previous mode (the DC side of a bridge with every
# diode OFF): the island's common-mode impedance (~2.5e8 ohm) is rounding-limited in BOTH
# computations, so those port matrices are compared only where no island exists (max|H| < 1e8).
@pytest.mark.parametrize("name", list(B.FAMILIES))
def test_port_model_equals_the_prototype_port_lcp(name):
    fam = B.FAMILIES[name]
    _, rec = B.prototype_selections(fam)
    cc = fam.proto(conductance=True)
    P = B.PulsimPorts(fam)
    h = B.H_E.get(name, 1e-9)
    compared = 0
    for t, y, sw, prev, _ in rec:
        r = P.select(t, y, sw, prev, h)
        if np.abs(r["H"]).max() > 1e8:
            continue
        Hp, qp = L.hybrid_port_lcp(cc, y, t, sw, prev, h)
        np.testing.assert_allclose(r["H"], Hp, rtol=0, atol=1e-6 * np.abs(Hp).max())
        np.testing.assert_allclose(r["q"], qp, rtol=0, atol=1e-6 * max(1.0, np.abs(qp).max()))
        compared += 1
    assert compared >= min(len(rec), 5) // 2


# At t = 0 the boost's switch closes on an inductor at rest. With R_on = 1 mOhm switch and diode
# the diode shares the switch's current (both paths are 1 mOhm into a capacitor at 0 V), so the
# conductance circuit selects ON; the ideal reference (R_on = 0) sees an exact tie and takes OFF.
MODEL_DIFFERENCE = {"boost_50": 1, "boost_500": 1}


@pytest.mark.parametrize("name", list(B.FAMILIES))
def test_selected_mode_equals_the_prototype_mode_at_every_event(name):
    fam = B.FAMILIES[name]
    _, rec = B.prototype_selections(fam)
    P = B.PulsimPorts(fam)
    h = 1e-7 * B.T_REF[name]                     # the default h_e (DECISIONS.md)
    bad = []
    for t, y, sw, prev, sel in rec:
        r = P.select(t, y, sw, prev, h)
        assert r["status"] in ("solution", "trivial")
        if tuple(r["mode"]) != sel:
            bad.append(t)
    assert len(bad) == MODEL_DIFFERENCE.get(name, 0), bad
    if bad:
        assert bad == [0.0]


def test_at_rest_zero_drop_diode_is_one_lcp_two_pivots_and_off():
    """Prototype case (i): the buck at rest, V_f = 0, gate off: 1 LCP, 2 pivots, diode OFF."""
    fam = B.FAMILIES["buck_rest"]
    P = B.PulsimPorts(fam)
    r = P.select(0.0, np.zeros(2), (False,), (True,), 1e-7 * B.T)
    assert r["mode"] == (False,) and r["pivots"] == 2 and r["ties"] == 0


def test_hysteresis_port_half_wave_rectifier():
    """D8 / task 3.4: a diode with V_th = 0.5 V > V_f = 0 from a 100 V 50 Hz sine into 10 ohm is held
    OFF while v_ak < V_th (prev OFF) and released above it; ON stays ON while i > 0."""
    b = pulsim.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_diode("D1", "a", "k", 1e3, 1e-9, 0.5, 0.0)
    b.add_resistor("R", "k", "gnd", 10.0)
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(1e-5)
    ad = K._NativeCircuitBuilderAdapter(b, cache)
    m = pulsim.SwitchStateMask(b.graph.num_switches)
    ad.set_mask(m)
    x = np.zeros(b.pool.state_size(b.graph))
    t_on = np.arcsin(0.5 / 100.0) / (2 * np.pi * 50)       # 15.9155 us
    h = 1e-7 * 0.02
    below = ad._mode_select_at(m, t_on - 1e-6, x, h)
    above = ad._mode_select_at(m, t_on + 1e-6, x, h)
    assert below["mode"] == (False,) and below["released"] == 0
    assert above["mode"] == (True,) and above["released"] == 1
    m.set(b.switch_index_of("D1"), True)
    on = ad._mode_select_at(m, t_on - 1e-6, x, h)             # ON, v = 0.49 V: still i > 0
    assert on["mode"] == (True,)
    assert t_on == pytest.approx(15.9155e-6, rel=1e-5)
