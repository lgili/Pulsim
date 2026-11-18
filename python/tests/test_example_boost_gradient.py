"""examples/optimization/boost_gradient.py: what its docstring says is
checked. Synthetic, ideal devices; the printed values are observations."""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from . import _sens_reference as ref

EXAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "examples"
           / "optimization" / "boost_gradient.py")


@pytest.fixture(scope="module")
def bg():
    spec = importlib.util.spec_from_file_location("boost_gradient", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["boost_gradient"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("boost_gradient", None)


def test_the_gradient_at_the_start_is_a_central_difference(bg):
    ev = bg.make_evaluator()
    f = bg.objective(ev)
    x0 = {"L": bg.L0, "C": bg.C0}
    _v, g = f(x0)
    for name in ("L", "C"):
        def val(x, name=name):
            return f({**x0, name: x})[0]
        fd, err, _h = ref.central_difference(val, x0[name],
                                             rel_steps=(1e-3, 1e-4))
        assert abs(g[name] - fd) <= 1e-6 * abs(fd), (name, g[name], fd, err)


def test_the_optimiser_meets_both_ripple_targets(bg, capsys):
    if importlib.util.find_spec("scipy") is None:
        # the example's own fallback is what runs; the grid test checks it
        bg.main([])
        assert "on the kernel batch" in capsys.readouterr().out
        return
    ev = bg.make_evaluator()
    start = ev.evaluate({"L": bg.L0, "C": bg.C0})
    res, n = bg.optimise(ev)
    assert res.success, res.message
    best = ev.evaluate({"L": res.x[0] * bg.L0, "C": res.x[1] * bg.C0})
    assert best.status == "ok"
    assert best.kpis["cost"] < start.kpis["cost"]
    assert best.kpis["ripple_v"] == pytest.approx(bg.RIPPLE_V, rel=0.05)
    assert best.kpis["ripple_i.L"] == pytest.approx(bg.RIPPLE_I, rel=0.05)
    assert n <= 60


def test_the_grid_fallback_runs_in_the_kernel_and_equals_the_python_path(bg):
    best, recs, ev = bg.grid(n=4, workers=2)
    assert ev.last_batch["kernel"] == len(recs) == 16
    assert ev.last_batch["python"] == 0
    py = [ev.evaluate(r.design) for r in recs]
    assert recs == py
    assert best.kpis["cost"] == min(r.kpis["cost"] for r in recs
                                    if r.status == "ok")
    assert np.isfinite(best.kpis["efficiency"])
