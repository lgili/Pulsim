"""engine='exact' / 'dsed' accepts a full-MNA initial_state (a DC operating point) by projecting it
onto the reduced state (add-exact-propagator-strategy DECISIONS, refusal causes)."""
import numpy as np
import pytest

import pulsim as p


def _rlc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "a", 5.0)
    b.add_inductor("L", "a", "out", 1e-4)
    b.add_capacitor("C", "out", "gnd", 1e-5)
    b.add_resistor("RL", "out", "gnd", 20.0)
    return b


@pytest.mark.parametrize("engine", ["dsed", "exact"])
def test_full_mna_dc_start_is_the_equilibrium(engine):
    b = _rlc()
    x = np.asarray(p.compute_dc_op(b), float)
    assert x.size == b.pool.state_size(b.graph) > 2      # a full vector, not the 2 states
    r = p.simulate(_rlc(), 1e-3, engine=engine, initial_state=x, dt=1e-5)
    # closed form: v(out) = 10 * 20 / 25 = 8 V and i(L) = 0.4 A, held for the whole run
    assert np.abs(np.asarray(r.v("out")) - 8.0).max() < 1e-9
    assert np.abs(np.asarray(r.i("L")) - 0.4).max() < 1e-9


def test_full_mna_start_off_equilibrium_matches_trbdf2():
    b = _rlc()
    x = np.zeros(b.pool.state_size(b.graph))
    x[b.node_id_of("out")] = 3.0                        # C at 3 V, L at rest, algebraically
    x[b.node_id_of("in")] = x[b.node_id_of("a")] = 10.0  # consistent (pwl starts first order otherwise)
    r = p.simulate(_rlc(), 1e-3, engine="dsed", initial_state=x, dt=1e-6)
    # trbdf2 at rtol 1e-10 (pwl is first order here: 3.8e-4 at dt 1e-8, 1.9e-4 at 5e-9)
    ref = p.simulate(_rlc(), 1e-3, engine="trbdf2", initial_state=x, rtol=1e-10, atol=1e-13)
    t = np.asarray(r.times)
    assert np.asarray(r.v("out"))[0] == pytest.approx(3.0, abs=1e-12)
    err = np.abs(np.asarray(r.v("out")) - np.interp(t, np.asarray(ref.times), np.asarray(ref.v("out"))))
    assert err.max() < 5e-6                             # measured 7.7e-7


def test_a_wrong_length_vector_is_still_an_error():
    with pytest.raises(ValueError, match="initial_state has shape"):
        p.simulate(_rlc(), 1e-4, engine="dsed", initial_state=np.zeros(3))


def test_equal_sizes_with_different_readings_are_refused():
    # n_mna == n_state == 2: the full vector is (v_a, v_b), the reduced state (v_C1, v_C2) with
    # v_C2 = v_a - v_b; the two readings differ, so neither is guessed (reviewer finding)
    def build():
        b = p.CircuitBuilder()
        b.add_current_source("I", "gnd", "a", 1e-3)
        b.add_capacitor("C1", "a", "gnd", 1e-6)
        b.add_capacitor("C2", "a", "b", 1e-6)
        b.add_resistor("R", "b", "gnd", 1e3)
        return b
    b = build()
    assert b.pool.state_size(b.graph) == 2
    with pytest.raises(ValueError, match="readings differ"):
        p.simulate(build(), 1e-4, engine="dsed", initial_state=np.array([2.0, 0.5]))
