"""The step controller's snapshot block (add-unified-device-interface
DECISIONS #24): pickle version 4 carries it as a 15th field, written only
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
    state = _snap().__getstate__() + (
        ("trbdf2", 1, ["h", "h_mem_gate"], [[2e-7], [1e-7, 3e-7]]),)
    s = _load(state)
    assert s.controller_state == ("trbdf2", 1,
                                  {"h": [2e-7], "h_mem_gate": [1e-7, 3e-7]})
    out = s.__getstate__()
    assert len(out) == 15
    back = pickle.loads(pickle.dumps(s))
    assert back.controller_state == s.controller_state


@pytest.mark.parametrize("block", [
    ("", 1, ["h"], [[1.0]]),          # no engine
    ("trbdf2", 0, ["h"], [[1.0]]),    # version 0
    ("trbdf2", 1, ["h", "x"], [[1.0]]),  # names and values differ
    ("trbdf2", 1, ["h"]),             # three values
])
def test_a_malformed_block_is_refused(block):
    with pytest.raises(RuntimeError, match="controller-state field"):
        _load(_snap().__getstate__() + (block,))
