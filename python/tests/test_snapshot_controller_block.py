"""One snapshot pickle (batch 14a, seam A).

Version 4 grew four layouts on parallel branches: the PR's 14 fields
(controllers'-state marker at 12, grid at 13), the device interface's
13 / 14 / 15 fields (grid at 12, marker at 13, the step controller's
``step::ControllerState`` block at 14), and the event queue's 15 fields
(marker at 12, tick phase at 13-14). Version 5 is the one layout that
carries all of them, always written (18 fields): marker 12, grid 13,
controller block 14 (or None), tick_origin 15, tick_next_k 16. Every
older form still loads; the two 15-field version-4 variants are told
apart by field 12 (a bool = the marker, a tuple = the grid)."""
import math
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


GRID = (0.0, 100, 1e-7, False)
CTRL = (2, [2e-7, 1e-7, 3e-7, -1.0, -1.0, 1.0, 5e-8, 6e-8])
TICKS = ([0.0, 5e-5], [6, 3])


def _same_core(a, b):
    assert a.t == b.t and list(a.x) == list(b.x) and a.valid == b.valid
    assert list(a.dev_state) == list(b.dev_state)
    assert list(a.stamped_mask) == list(b.stamped_mask)


def _grid(s):
    return (s.grid_origin, s.grid_step, s.grid_dt, s.post_edge_derivatives)


def test_version_5_writes_every_block():
    s = _snap()
    state = s.__getstate__()
    assert state[:2] == ("SolverSnapshot", 5) and len(state) == 18
    assert state[12] is False and len(state[13]) == 4
    assert state[14] is None and s.controller_state is None
    back = pickle.loads(pickle.dumps(s))
    _same_core(back, s)
    assert _grid(back) == _grid(s) and back.controller_state is None


def test_a_filled_controller_and_tick_phase_round_trip():
    base = _snap().__getstate__()
    s = _load(base[:14] + (CTRL,) + TICKS)
    assert s.controller_state == CTRL
    assert list(s.tick_origin) == TICKS[0] and list(s.tick_next_k) == TICKS[1]
    back = pickle.loads(pickle.dumps(s))
    assert back.controller_state == CTRL
    assert list(back.tick_origin) == TICKS[0]
    assert list(back.tick_next_k) == TICKS[1]


def test_every_older_form_loads_its_blocks():
    s0 = _snap()
    st = s0.__getstate__()
    core = tuple(st[2:12])                          # t .. legacy payload
    v4 = ("SolverSnapshot", 4) + core
    forms = {
        "v3-12": (("SolverSnapshot", 3) + core, dict()),
        "v3-13": (("SolverSnapshot", 3) + core + (False,), dict()),
        "v4-13 grid": (v4 + (GRID,), dict(grid=GRID)),
        "v4-14 PR": (v4 + (False, GRID), dict(grid=GRID)),
        "v4-14 device interface": (v4 + (GRID, False), dict(grid=GRID)),
        "v4-15 device interface": (v4 + (GRID, False, CTRL),
                                   dict(grid=GRID, ctrl=CTRL)),
        "v4-15 event queue": (v4 + (False,) + TICKS, dict(ticks=TICKS)),
        # the device interface's last version 4 (52f82c08): marker, grid,
        # controller
        "v4-15 device interface 52f82c08": (v4 + (False, GRID, CTRL),
                                            dict(grid=GRID, ctrl=CTRL)),
        "v5": (st[:13] + (GRID, CTRL) + TICKS,
               dict(grid=GRID, ctrl=CTRL, ticks=TICKS)),
    }
    for name, (state, want) in forms.items():
        s = _load(state)
        _same_core(s, s0)
        if "grid" in want:
            assert _grid(s) == want["grid"], name
        else:
            assert math.isnan(s.grid_origin), name
        assert s.controller_state == want.get("ctrl"), name
        o, k = want.get("ticks", ([], []))
        assert list(s.tick_origin) == o and list(s.tick_next_k) == k, name
        assert s._control_state is None, name
        # and every form writes back as version 5 with the same blocks
        again = _load(s.__getstate__())
        assert again.controller_state == s.controller_state, name
        assert list(again.tick_next_k) == list(s.tick_next_k), name


def test_the_marker_is_read_from_each_variant():
    st = _snap().__getstate__()
    core = tuple(st[2:12])
    v4 = ("SolverSnapshot", 4) + core
    for state in (("SolverSnapshot", 3) + core + (True,), v4 + (True, GRID),
                  v4 + (GRID, True), v4 + (GRID, True, None),
                  v4 + (True,) + TICKS, v4 + (True, GRID, CTRL),
                  st[:12] + (True,) + st[13:]):
        assert _load(state)._control_state is not None, len(state)


def test_a_pulsim_2_tuple_still_loads():
    s0 = _snap()
    st = (s0.t, list(s0.x), [], [], [], [], [], [], [], True)
    s = _load(st)
    assert s.t == s0.t and s.valid


@pytest.mark.parametrize("block", [
    (0, [1.0]),          # kind None is not a filled block
    (3, [1.0]),          # unknown kind
    (2, [1.0], "x"),     # three values
])
def test_a_malformed_block_is_refused(block):
    st = _snap().__getstate__()
    with pytest.raises(RuntimeError, match="controller-state field"):
        _load(st[:14] + (block,) + st[15:])


@pytest.mark.parametrize("edit", [
    lambda st: st[:13] + ("grid",) + st[14:],            # grid not a tuple
    lambda st: st[:15] + (["a"],) + st[16:],             # origins not numbers
    lambda st: st[:2] + ("t",) + st[3:],                 # t not a number
])
def test_a_malformed_tuple_raises_the_named_error(edit):
    with pytest.raises(RuntimeError, match="SolverSnapshot: malformed"):
        _load(edit(_snap().__getstate__()))


def test_tick_phase_lengths_must_agree():
    st = _snap().__getstate__()
    with pytest.raises(RuntimeError, match="differ in length"):
        _load(st[:15] + ([0.0], []))
