"""The fixed-step restart rule and step split resolve their defaults in the
run plan (update-number-changing-release blockers 1.5 and 1.8).

``discontinuity_restart`` / ``enable_substep_state_correction`` = None mean
"this run's default": ``pulsim._plan.DEFAULT_*`` (False today; the release
sets both True). The split's default follows the rule's resolved value; under
``mmc_arms`` the rule's default is not applied, with one diagnostic. An
explicit True / False does what it did before.
"""

import numpy as np
import pytest

import pulsim as p
from pulsim import _plan

T = 1e-5


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    return b


def _run(**kw):
    b = _buck()
    sf = p.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"),
                              b.graph.num_switches)
    return p.simulate(b, 20 * T, 7e-8, engine="pwl", switch_fn=sf, **kw)


@pytest.fixture
def release_defaults(monkeypatch):
    monkeypatch.setattr(_plan, "DEFAULT_DISCONTINUITY_RESTART", True)
    monkeypatch.setattr(_plan, "DEFAULT_SUBSTEP_STATE_CORRECTION", True)


def _same(a, b):
    np.testing.assert_allclose(np.asarray(a.states), np.asarray(b.states),
                               rtol=0, atol=0)


def test_todays_defaults_are_both_off():
    assert _plan.DEFAULT_DISCONTINUITY_RESTART is False
    assert _plan.DEFAULT_SUBSTEP_STATE_CORRECTION is False
    _same(_run(), _run(discontinuity_restart=False,
                       enable_substep_state_correction=False))


def test_explicit_restart_alone_keeps_the_split_off_today():
    # the split's default is DEFAULT_SUBSTEP_STATE_CORRECTION and the rule
    _same(_run(discontinuity_restart=True),
          _run(discontinuity_restart=True, enable_substep_state_correction=False))


def test_release_defaults_turn_both_on(release_defaults):
    r = _run()
    _same(r, _run(discontinuity_restart=True,
                  enable_substep_state_correction=True))
    assert r.n_restart_steps > 0


def test_restart_off_with_the_split_left_at_its_default_runs(release_defaults):
    # was refused ("requires discontinuity_restart") once the split defaulted on
    _same(_run(discontinuity_restart=False),
          _run(discontinuity_restart=False,
               enable_substep_state_correction=False))


def test_explicit_split_without_the_rule_is_still_refused(release_defaults):
    with pytest.raises(p.UnsupportedError, match="requires"):
        _run(discontinuity_restart=False, enable_substep_state_correction=True)


def _arm_run(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "p", "gnd", 600.0)
    b.add_resistor("Rload", "p", "a", 10.0)
    arm = p.add_mmc_thevenin_arm(b, "arm", "a", "gnd", n_sm=4, c_sm=2e-3,
                                 r_on=1e-3, v_c_init=0.0,
                                 n_on=lambda t: 0 if int(t / 1e-4) % 2 == 0 else 4)
    return p.simulate(b, t_end=1e-3, dt=1e-6, mmc_arms=[arm], **kw), arm


def test_mmc_arms_take_the_rule_off_by_default(release_defaults):
    r, arm = _arm_run()
    assert "mmc_arms" in (r.engine_route_reason or "")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(_plan, "DEFAULT_DISCONTINUITY_RESTART", False)
        m.setattr(_plan, "DEFAULT_SUBSTEP_STATE_CORRECTION", False)
        r0, arm0 = _arm_run()
    _same(r, r0)
    assert arm.v_c == arm0.v_c


def test_mmc_arms_with_an_explicit_rule_stay_refused(release_defaults):
    with pytest.raises(ValueError, match="mmc_arms.*discontinuity_restart=True"):
        _arm_run(discontinuity_restart=True)
