"""PLECS-style 3-D switching-loss tables E(v, i, Tj) — audit C.1.

Pulsim already annotated switching loss PSIM-style: one energy
curve E(I) at a reference bus voltage, scaled linearly by the
actual blocking voltage. That model has no temperature axis, and
switching energy is strongly temperature-dependent.

Measured against a 600 V / 100 A Si IGBT's own datasheet, reading
the 25 C / 300 V curve and asking for a 600 V bus at a 125 C
junction:

    I (A)    what it gives    datasheet    error
      25        2.60 mJ        4.40 mJ    -40.9 %
     100       11.00 mJ       18.00 mJ    -38.9 %
     200       24.00 mJ       38.60 mJ    -37.8 %

About 40 % of the switching loss missing — and it separates
cleanly:

    voltage       linear 300 -> 600 V scaling      -5.2 %
    temperature   25 C table read at 125 C        -35.6 %

So the linear voltage scaling was a decent approximation all
along. What was missing was the junction temperature, and that is
what these tables add — as a real axis, the way a datasheet's own
tables are shaped, rather than as a correction factor.
"""

import numpy as np
import pytest

import pulsim as p

# A realistic 600 V / 100 A Si IGBT (Infineon-class), E_on [J].
_I = [25.0, 50.0, 100.0, 150.0, 200.0]
DATASHEET = {
    (300.0, 25.0): list(zip(_I, [1.3e-3, 2.7e-3, 5.5e-3,
                                  8.6e-3, 12.0e-3])),
    (300.0, 125.0): list(zip(_I, [2.1e-3, 4.3e-3, 8.5e-3,
                                   13.2e-3, 18.3e-3])),
    (600.0, 25.0): list(zip(_I, [2.8e-3, 5.8e-3, 11.6e-3,
                                  18.1e-3, 25.2e-3])),
    (600.0, 125.0): list(zip(_I, [4.4e-3, 9.1e-3, 18.0e-3,
                                   27.9e-3, 38.6e-3])),
}


def _table():
    return p.LossTable.from_curves(DATASHEET)


# ---------------------------------------------------------------
# It reproduces the datasheet.
# ---------------------------------------------------------------

def test_grid_points_come_back_exactly():
    """Interpolation must be an identity on the tabulated points.
    If this drifts, every number downstream is off by whatever the
    interpolator invented."""
    t = _table()
    for (v, tj), curve in DATASHEET.items():
        for i, e in curve:
            assert float(t(v, i, tj)) == pytest.approx(e, rel=1e-12)


def test_the_temperature_axis_is_what_was_missing():
    """The measurement that motivates the whole module."""
    t = _table()
    cold = float(t(600.0, 100.0, 25.0))
    hot = float(t(600.0, 100.0, 125.0))
    assert hot / cold == pytest.approx(18.0 / 11.6, rel=1e-9)
    # Reading the cold table at a hot junction loses ~36 %.
    assert (cold / hot - 1.0) == pytest.approx(-0.356, abs=0.01)


def test_it_interpolates_and_extrapolates():
    t = _table()
    mid = float(t(450.0, 75.0, 75.0))
    assert 0.0 < mid < float(t(600.0, 100.0, 125.0))
    # Past the last current, linear extrapolation rather than a
    # clamp — a converter run past its datasheet should not read
    # as if it were sitting on the last tabulated point.
    far = float(t(600.0, 250.0, 125.0))
    assert far > float(t(600.0, 200.0, 125.0))


def test_it_is_vectorised():
    t = _table()
    got = t([300.0, 600.0], [100.0, 100.0], [25.0, 125.0])
    assert got.shape == (2,)
    assert got[0] == pytest.approx(5.5e-3)
    assert got[1] == pytest.approx(18.0e-3)


def test_energy_never_comes_back_negative():
    """Linear extrapolation below the first tabulated current runs
    negative on paper; a negative switching energy is never an
    answer."""
    t = _table()
    assert float(t(300.0, 0.0, 25.0)) >= 0.0
    assert float(t(300.0, -50.0, 25.0)) >= 0.0


# ---------------------------------------------------------------
# Refusals — a mistranscribed table must be named, not absorbed.
# ---------------------------------------------------------------

def test_a_transposed_table_is_refused():
    """Datasheet tables are usually transcribed current-major, so
    getting the axis order wrong is the likely mistake."""
    with pytest.raises(ValueError, match=r"v x i x Tj"):
        p.LossTable(v_axis=[300.0, 600.0], i_axis=_I,
                     tj_axis=[25.0, 125.0],
                     energy=np.zeros((5, 2, 2)))


def test_an_unsorted_axis_is_refused():
    with pytest.raises(ValueError, match="ascending"):
        p.LossTable(v_axis=[600.0, 300.0], i_axis=[100.0],
                     tj_axis=[25.0], energy=np.zeros((2, 1, 1)))


def test_negative_energy_is_refused():
    with pytest.raises(ValueError, match=">= 0"):
        p.LossTable(v_axis=[300.0], i_axis=[100.0],
                     tj_axis=[25.0],
                     energy=np.full((1, 1, 1), -1.0))


def test_an_incomplete_grid_is_refused_rather_than_filled():
    """Filling a missing corner would invent a datasheet number."""
    partial = dict(DATASHEET)
    del partial[(600.0, 125.0)]
    with pytest.raises(ValueError, match="incomplete"):
        p.LossTable.from_curves(partial)


def test_curves_need_not_share_sample_points():
    """Datasheet curves rarely do."""
    t = p.LossTable.from_curves({
        (300.0, 25.0): [(20.0, 1.0e-3), (100.0, 5.5e-3)],
        (300.0, 125.0): [(50.0, 4.3e-3), (150.0, 13.2e-3)],
    })
    assert float(t(300.0, 100.0, 25.0)) == pytest.approx(5.5e-3,
                                                          rel=1e-9)


# ---------------------------------------------------------------
# The Tj source.
# ---------------------------------------------------------------

def test_a_table_without_a_junction_temperature_is_refused():
    """A 3-D table is indexed by Tj; picking one silently would
    hide exactly the error the table exists to fix."""
    from pulsim.losses import _switch_switching_loss

    times = np.linspace(0.0, 1e-3, 11)
    closed = np.zeros(11, dtype=bool)
    closed[5:] = True
    with pytest.raises(ValueError, match="Tj"):
        _switch_switching_loss(closed, times,
                                np.full(11, 600.0),
                                np.full(11, 100.0),
                                {"E_on_table": _table()})


@pytest.mark.parametrize("tj_spec", [
    125.0,                                   # scalar
    lambda _t: 125.0,                        # callable
])
def test_tj_accepts_a_scalar_or_a_callable(tj_spec):
    from pulsim.losses import _switch_switching_loss

    times = np.linspace(0.0, 1e-3, 11)
    closed = np.zeros(11, dtype=bool)
    closed[5:] = True
    out = _switch_switching_loss(closed, times,
                                  np.full(11, 600.0),
                                  np.full(11, 100.0),
                                  {"E_on_table": _table(),
                                   "Tj": tj_spec})
    # One turn-on edge at 600 V / 100 A / 125 C.
    assert out["E_sw_on_total"] == pytest.approx(18.0e-3, rel=1e-6)


def test_a_wrong_length_tj_array_is_refused():
    from pulsim.loss_tables import resolve_tj

    with pytest.raises(ValueError, match="shape"):
        resolve_tj(np.zeros(5), np.zeros(11))


def test_a_tj_array_tracks_the_time_grid():
    """The shape a thermal simulation delivers."""
    from pulsim.losses import _switch_switching_loss

    times = np.linspace(0.0, 1e-3, 11)
    closed = np.zeros(11, dtype=bool)
    closed[5:] = True
    cold = _switch_switching_loss(
        closed, times, np.full(11, 600.0), np.full(11, 100.0),
        {"E_on_table": _table(), "Tj": np.full(11, 25.0)})
    hot = _switch_switching_loss(
        closed, times, np.full(11, 600.0), np.full(11, 100.0),
        {"E_on_table": _table(), "Tj": np.full(11, 125.0)})
    assert hot["E_sw_on_total"] > 1.5 * cold["E_sw_on_total"]


def test_the_table_needs_no_v_ref():
    """Voltage is an axis of the table, not a scaling factor, so
    V_ref is neither required nor consulted. Demanding it would be
    asking for a number with nowhere to go."""
    from pulsim.losses import _switch_switching_loss

    times = np.linspace(0.0, 1e-3, 11)
    closed = np.zeros(11, dtype=bool)
    closed[5:] = True
    # No V_ref at all.
    out = _switch_switching_loss(
        closed, times, np.full(11, 300.0), np.full(11, 100.0),
        {"E_on_table": _table(), "Tj": 25.0})
    assert out["E_sw_on_total"] == pytest.approx(5.5e-3, rel=1e-6)
    # And a stray one is ignored rather than applied as a scaling.
    out2 = _switch_switching_loss(
        closed, times, np.full(11, 300.0), np.full(11, 100.0),
        {"E_on_table": _table(), "Tj": 25.0, "V_ref": 999.0})
    assert out2["E_sw_on_total"] == out["E_sw_on_total"]


def test_the_curve_path_still_demands_a_v_ref():
    """The old form is unchanged: without a table, voltage is a
    scaling factor and it needs its anchor."""
    from pulsim.losses import _switch_switching_loss

    times = np.linspace(0.0, 1e-3, 11)
    closed = np.zeros(11, dtype=bool)
    closed[5:] = True
    with pytest.raises(ValueError, match="V_ref"):
        _switch_switching_loss(
            closed, times, np.full(11, 300.0), np.full(11, 100.0),
            {"E_on_curve": [(100.0, 5.5e-3)]})
