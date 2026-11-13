"""add-branch-form-ideal-switch task 1.2: the dense references reproduce the prototype numbers.

``tests/oracles/ideal_switch_reference.py`` ports ``probes/branch_form.py`` and
``probes/sdirk_restart_ideal.py``; these tests pin it to ``probes/sdirk_restart_ideal.out``, so the
engine tests of groups 3, 5 and 6 compare against a reference that is itself checked.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
from tests.oracles import ideal_switch_reference as R  # noqa: E402


def test_cutset_trapezoid_rings_and_sdirk_restart_does_not():
    ring = R.cutset("none")
    assert np.all(ring["i_L"][100:106] == 0.0)
    assert ring["v_a"][100:106] == pytest.approx([-397.307628, 397.307628] * 3, rel=1e-8)
    s = R.cutset("sdirk3")
    assert np.all(s["i_L"][100:106] == 0.0)
    assert s["v_a"][100] == pytest.approx(959.018799, rel=1e-8)
    assert np.all(s["v_a"][101:106] == 0.0)
    flux = 1e-3 * s["iL_minus"]
    assert flux == pytest.approx(9.93096483e-3, rel=1e-8)
    # the SDIRK quadrature of the node voltage over the first step removes exactly L i(t-)
    assert s["area"] == pytest.approx(-flux, rel=1e-12)
    assert 0.5e-3 * s["iL_minus"] ** 2 == pytest.approx(49.3120e-3, rel=1e-5)


def test_capacitor_loop_conserves_charge_and_marks_the_pulse():
    ring = R.capacitor_loop("none")
    assert np.allclose(ring[:, :2], 4.0, rtol=0, atol=1e-12)
    assert ring[:, 2] == pytest.approx([120, -120, 120, -120, 120], rel=1e-9)
    s = R.capacitor_loop("sdirk3")
    assert np.allclose(s[:, :2], 4.0, rtol=0, atol=1e-12)
    assert s[0, 2] == pytest.approx(-289.7056, rel=1e-6)
    assert np.all(np.abs(s[1:, 2]) < 1e-13)
    assert 1e-6 * (10.0 - s[0, 0]) == pytest.approx(6.0e-6, rel=1e-12)
    assert 0.5e-6 * 100 + 0.5 * 3e-6 * 4 - 0.5 * 4e-6 * s[-1, 0] ** 2 == pytest.approx(24.0e-6, rel=1e-9)


def test_branch_form_reproduces_the_conductance_form_with_the_same_values():
    h = 1e-3 / 20
    o_b, _ = R.rl_switch(True, 1e-3, 1e-9, True, h, 200)
    o_c, _ = R.rl_switch(False, 1e-3, 1e-9, True, h, 200)
    assert np.max(np.abs(o_b - o_c)) < 1e-10


@pytest.mark.parametrize("N, err", [(50, -2.442e-6), (100, -6.591e-7)])
def test_ideal_dcm_buck_sdirk3(N, err):
    vC, iL, nf, nev = R.run_buck(N, "sdirk3")
    assert vC - R.VC_REF == pytest.approx(err, rel=2e-3)
    assert iL == 0.0
    assert nf == 606
