"""Release finding F-Y-1: sensitivities on engine 'pwl' must run under the
release defaults — the restart rule ON and event splitting ON
(``discontinuity_restart=True, enable_substep_state_correction=True``).

The six pwl tests of ``test_sensitivity_step.py`` and the pwl sensitivity
cell of ``test_engine_conformance.py`` are run again in that configuration
(their own truths: central differences of the same scheme, the closed
form), plus the RC closed form at the tolerance the restart rule earns: the
SDIRK start removes plain pwl's O(dt) first-step error.
"""

import math
import warnings

import numpy as np
import pytest

import pulsim as p

from . import test_engine_conformance as conf
from . import test_sensitivity_step as st

RELEASE = dict(discontinuity_restart=True, enable_substep_state_correction=True)


@pytest.fixture
def release(monkeypatch):
    def run(b, t_end, engine, **kw):
        if engine == "pwl":
            kw = {**RELEASE, **kw}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return p.simulate(b, t_end, engine=engine, **kw)

    monkeypatch.setattr(st, "_run", run)
    return run


def test_rc_matches_central_differences(release):
    st.test_rc_on_pwl_matches_central_differences_and_the_closed_form()


def test_rc_matches_the_closed_form_to_second_order(release):
    b = st._rc()
    r = release(b, st.TAU, "pwl", dt=1e-6, sensitivities=["R.R", "C.C"])
    S = r.sensitivity.states[-1, b.node_id_of("out")]
    dR, dC = st._rc_closed_form(st.TAU)
    assert abs(S[0] / dR - 1) < 1e-6 and abs(S[1] / dC - 1) < 1e-6, (S, dR, dC)


def test_linear_counters(release):
    """Three SDIRK restart steps (two stages each) open the run."""
    b = st._rc()
    r = release(b, 2e-4, "pwl", dt=1e-6, sensitivities=["R.R", "C.C"])
    n_steps = len(r.sensitivity.times) - 1
    assert n_steps == 200
    assert r.sensitivity.counters == {"n_sens_solves": 2 * (n_steps + 3),
                                      "n_sens_iterations": 0, "n_sens_discarded": 0}


def test_shockley_rectifier_matches_central_differences(release):
    names = list(st.NAMES.values())
    b, r = st._pwl_rect(sens=names)
    out, a = b.node_id_of("out"), b.node_id_of("a")
    S = r.sensitivity.states[-1]
    for j, (k, nm) in enumerate(st.NAMES.items()):
        def fin(v, k=k):
            bb, rr = st._pwl_rect(**{k: v})
            x = np.asarray(rr.states[-1])
            return np.array([x[bb.node_id_of("out")], x[bb.node_id_of("a")]])
        fd, est, _h = st.ref.central_difference(fin, st.P0[k], rel_steps=(1e-4, 1e-5))
        got = np.array([S[out, j], S[a, j]])
        assert st.ref.rel_err(got, fd) < 1e-6, (nm, got, fd, est)
    c = r.sensitivity.counters   # 4000 steps, the first three with two stages
    assert c["n_sens_solves"] == len(names) * 4003
    assert 0 < c["n_sens_iterations"] <= 2 * c["n_sens_solves"]


def test_states_and_factorizations_identical_with_sensitivities(release):
    st.test_states_and_factorizations_identical_with_sensitivities()


@pytest.mark.parametrize("k", ["K", "V_T"])
def test_mosfet_level1_matches_central_differences(release, k):
    st.test_mosfet_level1_on_pwl_matches_central_differences(k)


def test_conformance_cell():
    r = conf._sim("pwl", sensitivities=["R.R"], **RELEASE)
    assert r.sensitivity.params == ["R.R"]
    assert r.sensitivity.states.shape[2] == 1
    assert math.isfinite(float(r.sensitivity.states[-1].max()))
