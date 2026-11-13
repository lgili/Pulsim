"""add-in-kernel-event-losses 1.3: the kernel loss tables read exactly
what `LossTable.__call__` reads, inside and outside the grid."""

import numpy as np
import pytest

import pulsim
from pulsim import _pulsim
from pulsim.loss_tables import LossTable


def _table(rng):
    v = np.cumsum(rng.uniform(50, 200, 4))
    i = np.cumsum(rng.uniform(1, 30, 5))
    tj = np.array([25.0, 75.0, 125.0, 150.0])
    e = rng.uniform(1e-4, 5e-3, (v.size, i.size, tj.size))
    return LossTable(v, i, tj, e)


def test_kernel_table_equals_python_on_10000_points():
    rng = np.random.default_rng(7)
    t = _table(rng)
    k = t.to_kernel()
    # half of every axis range again past both ends: extrapolation + clamp
    def draw(ax, n):
        span = ax[-1] - ax[0]
        return rng.uniform(ax[0] - span, ax[-1] + span, n)
    n = 10_000
    v, i, tj = draw(t.v_axis, n), draw(t.i_axis, n), draw(t.tj_axis, n)
    want = t(v, i, tj)
    got = np.array([k(a, b, c) for a, b, c in zip(v, i, tj)])
    outside = ((v < t.v_axis[0]) | (v > t.v_axis[-1])
               | (i < t.i_axis[0]) | (i > t.i_axis[-1])
               | (tj < t.tj_axis[0]) | (tj > t.tj_axis[-1]))
    assert outside.sum() > 1000 and (~outside).sum() > 100
    assert (want == 0).sum() > 10          # the clamp is exercised
    err = np.abs(got - want) / np.maximum(np.abs(want), 1e-6)
    assert err.max() <= 1e-12, err.max()


def test_temperature_axis_extrapolates():
    t = LossTable([400.0], [10.0], [25.0, 125.0], [[[1e-3, 2e-3]]])
    k = t.to_kernel()
    assert k(400.0, 10.0, 75.0) == pytest.approx(1.5e-3, rel=1e-12)
    assert k(400.0, 10.0, 175.0) == pytest.approx(2.5e-3, rel=1e-12)
    assert k(400.0, 10.0, 175.0) == pytest.approx(
        float(t(400.0, 10.0, 175.0)), rel=1e-12)


def test_conduction_table_equals_a_collapsed_loss_table():
    rng = np.random.default_rng(3)
    i = np.array([0.0, 5.0, 20.0, 50.0])
    tj = np.array([25.0, 125.0])
    von = rng.uniform(0.5, 2.0, (i.size, tj.size))
    k = _pulsim.ConductionTable(i.tolist(), tj.tolist(), von.ravel().tolist())
    ref = LossTable([0.0], i, tj, von[None])
    q_i = rng.uniform(-60, 120, 2000)
    q_t = rng.uniform(-50, 250, 2000)
    got = np.array([k(a, b) for a, b in zip(q_i, q_t)])
    want = ref(0.0, q_i, q_t)
    assert np.max(np.abs(got - want) / np.maximum(want, 1e-6)) <= 1e-12


@pytest.mark.parametrize("args, msg", [
    (([], [1.0], [1.0], []), "v axis is empty"),
    (([1.0], [2.0, 1.0], [1.0], [1.0, 1.0]), "i axis must be strictly"),
    (([1.0], [1.0], [1.0, 2.0], [1.0]), "energy has 1 entries"),
    (([1.0], [1.0], [1.0], [-1.0]), "energy must be >= 0"),
])
def test_kernel_refusals_name_the_axis(args, msg):
    with pytest.raises(ValueError, match=msg):
        _pulsim.LossTable3D(*args)


def test_the_tables_are_exported():
    assert pulsim.LossTable3D is _pulsim.LossTable3D
    assert pulsim.ConductionTable is _pulsim.ConductionTable
