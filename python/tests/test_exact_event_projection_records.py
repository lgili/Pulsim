"""F-R-3 / F-SB-3: the exact engine records every event projection, rule on or off.

At D1's turn-off in ``diode_half_wave_rl_load`` the exact engine projects out the g_off
transient (tau = L * g_off = 1 ns, far under its dt/100 parasitic window): V(out)'s right
limit is the settled -3.1e-6 V where the truth starts at -0.31385 V and decays. The stored
rows were right (left limits to 4e-14 V; the right limit is the settled value), but with no
record of the projection the scoreboard's rule (1) could not see it and graded the right
limit as a wrong limit: 0.0304 of the range (``limits``), where the time-weighted part reads
4.4e-6. ``res.times`` itself is unchanged: strictly increasing, one row per stored sample,
the left limit at an event; right limits stay in ``right_limit_*`` (opt-in views).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import v1_reader  # noqa: E402

import pulsim  # noqa: E402

CELL = "circuits/diodes/diode_half_wave_rl_load.yaml"


def _run(rule=False):
    """The netlist's dt (the scoreboard's call); with the rule, no dt (the auto route's)."""
    case = v1_reader.load(ROOT / CELL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if rule:
            return pulsim.simulate(case.build()[0], case.t_end, engine="dsed", output_rtol=1e-5)
        return case.run("dsed")


@pytest.mark.parametrize("rule", [False, True], ids=["rule_off", "rule_on"])
def test_every_event_projection_is_recorded(rule):
    res = _run(rule)
    t = np.asarray(res.times)
    assert np.all(np.diff(t) > 0)                       # the contract: strictly increasing
    ep = res.diagnostics["event_projections"]
    assert ep["n_projections"] >= 3                     # D1's three turn-offs at least
    assert not ep["capped"] and len(ep["projections"]) == ep["n_projections"]
    # under the rule the amplitude is over the recovered vector: 0.31385 V each (V(out) ->
    # settled); rule off it is max |dx| over the state (no recovery on the default path:
    # it would evaluate the sources and a user's b_extra_fn), with no column
    big = [p for p in ep["projections"] if p["amplitude"] > (0.3 if rule else 0.0)]
    assert len(big) >= 3
    for p_ in big:
        assert (p_["column"] is None) == (not rule)
        assert p_["tau"] > 0 and p_["tau"] < 1e-8      # the g_off cluster, 1 ns
        # at a stored event row (the left limit), and that row has a right limit
        k = int(np.argmin(np.abs(t - p_["t"])))
        assert abs(t[k] - p_["t"]) <= 4 * np.spacing(p_["t"])
        assert k in set(np.asarray(res.right_limit_index).tolist())
    if not rule:
        assert "output_grid" not in res.diagnostics     # rule off: no rule keys
    else:
        assert res.diagnostics["output_grid"]["n_projections"] == ep["n_projections"]


@pytest.mark.reference
def test_scoreboard_rule_1_sees_the_projection():
    """The cell's exact grade: 0.0304 (limits, unreported projection) before; the rule (1)
    charge of the collapsed 1 ns transient now (measured 4.8e-6 on this Mac)."""
    pytest.importorskip("mpmath")
    import differential_runner as dr
    import exact_scoreboard as es
    import exact_truth as et
    sol = et.Solver(et.Circuit(es.ROOT / CELL), es.truth_dps(CELL)).run()
    case = v1_reader.load(es.ROOT / CELL)
    cols = dr.observables_of(case)
    g, _, _ = es.graded_columns(sol, cols)
    res = _run()
    r = es.grade(sol, dr.traces(res, cols), g, es.spans(sol, g), case.dt, 1e-3, self_score=False,
                 parasitic=es.PARASITIC["dsed"] * case.dt)
    assert r["n_projected"] >= 1
    assert r["exact"] < 1e-4                            # was 0.0304
