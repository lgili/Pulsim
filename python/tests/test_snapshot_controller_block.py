"""The step controller's snapshot block (add-unified-device-interface
DECISIONS #24; the object is refactor-unified-transient-stepper's
``step::ControllerState``: a kind and a flat vector in that controller's
layout): pickle version 4 carries it as a 15th field, written only
when the block is filled, so a snapshot without one stays the 14-field
version 4 earlier readers load. 13- and 14-field version-4 pickles and
version 3 (12 or 13 fields) still load, with an empty block, which means
"restart the controller", as before the block existed."""
import pickle

import pytest

import pulsim as p
from pulsim import _pulsim as _k


def _snap():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "c", 10.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    return p.simulate(b, t_end=1e-5, dt=1e-7, engine="pwl").final_snapshot


def _load(state):
    s = _k.SolverSnapshot.__new__(_k.SolverSnapshot)
    s.__setstate__(state)
    return s


def test_an_empty_block_writes_the_14_field_version_4():
    s = _snap()
    state = s.__getstate__()
    assert state[0] == "SolverSnapshot" and state[1] == 4 and len(state) == 14
    assert s.controller_state is None
    back = pickle.loads(pickle.dumps(s))
    assert back.controller_state is None and back.t == s.t


def test_older_version_4_and_version_3_pickles_still_load():
    state = _snap().__getstate__()
    for t in (state[:13], state, state + (None,)):
        s = _load(t)
        assert s.controller_state is None and s.valid
    v3 = ("SolverSnapshot", 3) + tuple(state[2:12])
    assert _load(v3).controller_state is None
    assert _load(v3 + (False,)).controller_state is None


def test_a_filled_block_round_trips_as_a_15th_field():
    # kind 2 = LTE: [h, h_mem_gate0, h_mem_gate1, pending_gate_slot,
    # diode_slot, gate_parity, h_mem_diode...]
    v = [2e-7, 1e-7, 3e-7, -1.0, -1.0, 1.0, 5e-8, 6e-8]
    s = _load(_snap().__getstate__() + ((2, v),))
    assert s.controller_state == (2, v)
    assert len(s.__getstate__()) == 15
    back = pickle.loads(pickle.dumps(s))
    assert back.controller_state == (2, v)


@pytest.mark.parametrize("block", [
    (0, [1.0]),          # kind None is not a filled block
    (3, [1.0]),          # unknown kind
    (2, [1.0], "x"),     # three values
])
def test_a_malformed_block_is_refused(block):
    with pytest.raises(RuntimeError, match="controller-state field"):
        _load(_snap().__getstate__() + (block,))
