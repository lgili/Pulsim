"""Dependent states on the exact engine: a mode whose algebraic block is singular (a capacitor
loop, an inductor cut-set) holds the states on K x = k; an inconsistent state is projected at
mode entry, energy-weighted, and the impulse is recorded with the fixed engine's ReinitRecord
fields (add-exact-propagator-strategy DECISIONS, dependent states)."""
import numpy as np
import pytest

import pulsim as p
from pulsim._result_views import with_right_limits

ENGINES = ["dsed", "exact"]


def _cap_loop(R=1e3):
    # two equal 1.5 uF capacitors at 8 V and 0 V, tied by a 0 V source: a capacitor loop
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1.5e-6, 8.0)
    b.add_capacitor("C2", "b", "gnd", 1.5e-6, 0.0)
    b.add_voltage_source("V0", "a", "b", 0.0)
    b.add_resistor("R", "a", "gnd", R)
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_two_capacitors_share_their_charge(engine):
    r = p.simulate(_cap_loop(), 1e-3, engine=engine, dt=1e-6)
    t = np.asarray(r.times)
    va, vb = np.asarray(r.v("a")), np.asarray(r.v("b"))
    assert va[0] == pytest.approx(4.0, abs=1e-12)
    assert vb[0] == pytest.approx(4.0, abs=1e-12)
    # then both discharge together through R: tau = R (C1 + C2) = 3 ms
    np.testing.assert_allclose(va, 4.0 * np.exp(-t / 3e-3), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(vb, va, rtol=0, atol=1e-12)
    (rec,) = r.reinit_records
    assert rec.kind == "capacitor_loop"
    assert rec.t == 0.0 and rec.sample == 0
    assert len(rec.states) == 2
    assert rec.lambda_ == pytest.approx(6e-6, rel=1e-12)      # 6 uC moved
    assert rec.dW == pytest.approx(24e-6, rel=1e-12)          # 24 uJ lost
    assert rec.projection_gap < 1e-12


def _cut_set_at_start():
    # a 0 A current source in series with an inductor carrying 2 A: an inductor cut-set
    b = p.CircuitBuilder()
    b.add_current_source("I0", "gnd", "n1", 0.0)
    b.add_inductor("L", "n1", "n2", 1e-3, 2.0)
    b.add_resistor("R", "n2", "gnd", 10.0)
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_an_inductor_cut_set_takes_the_current_to_zero(engine):
    r = p.simulate(_cut_set_at_start(), 1e-4, engine=engine, dt=1e-6)
    assert np.abs(np.asarray(r.i("L"))).max() < 1e-12
    (rec,) = r.reinit_records
    assert rec.kind == "cut_set"
    assert abs(rec.lambda_) == pytest.approx(1e-3 * 2.0, rel=1e-12)   # |Lambda| = L i
    assert rec.dW == pytest.approx(0.5 * 1e-3 * 2.0 ** 2, rel=1e-12)


def _switched_cut_set(R=10.0):
    # the switch opens with g_off = 0: the inductor's only path is cut
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "a", 1e6, 0.0)
    b.add_resistor("R", "a", "b", R)
    b.add_inductor("L", "b", "gnd", 1e-3)
    fn = p.make_pwm_switch_fn(1e3, 0.5, b.switch_index_of("S"), b.graph.num_switches)
    return b, fn


@pytest.mark.parametrize("engine", ENGINES)
def test_a_switch_that_cuts_an_inductor_records_the_flux(engine):
    b, fn = _switched_cut_set()
    r = p.simulate(b, 2e-3, engine=engine, dt=1e-6, switch_fn=fn)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    recs = r.reinit_records
    assert [rec.t for rec in recs] == pytest.approx([5e-4, 1.5e-3], abs=1e-15)
    for rec in recs:
        assert rec.kind == "cut_set"
        k = rec.sample
        assert t[k] == pytest.approx(rec.t, abs=1e-15)
        # the stored sample is the left limit: the on-state current, closed form
        i_on = 10.0 / (1e-6 + 10.0) * (1 - np.exp(-5e-4 * (1e-6 + 10.0) / 1e-3))
        assert i[k] == pytest.approx(i_on, rel=1e-9)
        assert abs(rec.lambda_) == pytest.approx(1e-3 * i[k], rel=1e-9)
        assert rec.dW == pytest.approx(0.5 * 1e-3 * i[k] ** 2, rel=1e-9)
    # the right limit, one ulp after each record: the cut current
    two = with_right_limits(r)
    t2, i2 = np.asarray(two.times), np.asarray(two.i("L"))
    for rec in recs:
        j = int(np.flatnonzero(t2 == t[rec.sample])[0])
        assert abs(i2[j + 1]) < 1e-12
    off = (np.mod(t, 1e-3) > 5e-4 + 1e-9)
    assert np.abs(i[off]).max() < 1e-12


def test_the_rank_decision_survives_rounding():
    # 20 last-bit perturbations of R: one mode sequence, the same records
    seqs = set()
    for k in range(20):
        R = np.nextafter(10.0, np.inf) if k == 0 else 10.0 * (1 + (k - 10) * 2.0 ** -52)
        b, fn = _switched_cut_set(R)
        r = p.simulate(b, 2e-3, engine="exact", dt=1e-6, switch_fn=fn)
        seqs.add((tuple(np.round(np.asarray(r.event_times), 12)),
                  tuple((rec.kind, len(rec.states)) for rec in r.reinit_records)))
    assert len(seqs) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_a_user_injection_on_a_constraint_is_refused(engine):
    # the flow keeps K x' = 0: a b_extra_fn current into the loop would break K x = k unseen
    b = _cap_loop()
    zero = [0.0] * b.pool.state_size(b.graph)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 1e-3, engine=engine, dt=1e-6, b_extra_fn=lambda _t: zero,
                   allow_implicit_segments=True)
    assert (ei.value.engine, ei.value.feature) == ("dsed", "extraction")
    assert "dependent states" in str(ei.value)
# Integration batch 10: on_impulse holds on the exact engine (before, 'error'
# and 'warn' were ignored there: dsed refused every dependent mode until
# this branch, so it never recorded).
@pytest.mark.parametrize("engine", ENGINES)
def test_on_impulse_error_refuses_the_projection(engine):
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_cap_loop(), 1e-4, engine=engine, dt=1e-6, on_impulse="error")
    assert (ei.value.engine, ei.value.feature) == ("dsed", "impulse")
    assert "a capacitor loop" in str(ei.value) and "6e-06 C" in str(ei.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_on_impulse_warn_on_the_exact_engine(engine):
    with pytest.warns(UserWarning, match="impulsive re-initialisation"):
        r = p.simulate(_cap_loop(), 1e-4, engine=engine, dt=1e-6, on_impulse="warn")
    assert len(r.reinit_records) == 1
