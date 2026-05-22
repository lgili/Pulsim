"""Unit tests for `pulsim.snubber` — the snubber-sizing advisor.

These tests pin the numerical sizing rules so the closed-form math stays
honest as the surrounding solver evolves. They do *not* run a simulation —
the advisor is pure math (no I/O, no MNA assembly), which is the whole
point: users can call it before constructing the circuit to decide whether
PWL Ideal is viable for their (L, I_peak, V_bus, dt, f_sw) operating point.

Background
==========
The boost PFC in PWL Ideal mode shows V_sw overshoot ≈ I·√(L/C_oss) at
the OFF edge (Tustin captures the LC ring undamped). For the default
`MOSFETParams.C_oss = 10 nF` with a 100 µH boost L and 5 A peak, that's
500 V on top of a 400 V bus — clearly unphysical, and the advisor must
communicate this clearly.
"""
from __future__ import annotations

import io
import pytest

from pulsim import snubber


# -------------- predict_overshoot --------------------------------------------

class TestPredictOvershoot:
    def test_default_boost_10nF(self):
        # Canonical "the runtime default is wrong for this topology" case.
        v = snubber.predict_overshoot(L=100e-6, C=10e-9, I_peak=5.0)
        assert v == pytest.approx(500.0, rel=1e-3)

    def test_scaling_with_L(self):
        v1 = snubber.predict_overshoot(L=100e-6, C=10e-9, I_peak=5.0)
        v2 = snubber.predict_overshoot(L=400e-6, C=10e-9, I_peak=5.0)
        # √(L) scaling → factor 2.
        assert v2 == pytest.approx(2.0 * v1, rel=1e-6)

    def test_scaling_with_C(self):
        v1 = snubber.predict_overshoot(L=100e-6, C=10e-9, I_peak=5.0)
        v2 = snubber.predict_overshoot(L=100e-6, C=40e-9, I_peak=5.0)
        # √(1/C) scaling → factor 0.5.
        assert v2 == pytest.approx(0.5 * v1, rel=1e-6)

    def test_proportional_to_I(self):
        v1 = snubber.predict_overshoot(L=100e-6, C=10e-9, I_peak=2.0)
        v2 = snubber.predict_overshoot(L=100e-6, C=10e-9, I_peak=4.0)
        assert v2 == pytest.approx(2.0 * v1, rel=1e-6)

    def test_rejects_zero_or_negative(self):
        with pytest.raises(ValueError):
            snubber.predict_overshoot(L=0.0, C=10e-9, I_peak=5.0)
        with pytest.raises(ValueError):
            snubber.predict_overshoot(L=100e-6, C=-1e-9, I_peak=5.0)


# -------------- recommend_C_oss ----------------------------------------------

class TestRecommendCoss:
    def test_target_overshoot_is_hit(self):
        """The closed-form formula should land exactly on the target.
        C = (I/V_overshoot_max)² · L  →  predicted V_overshoot = target."""
        rec = snubber.recommend_C_oss(L=100e-6, I_peak=5.0, V_bus=400.0,
                                        max_overshoot_frac=0.5)
        # 50 % of 400 = 200 V target.
        assert rec.V_overshoot_predicted == pytest.approx(200.0, rel=1e-6)
        # C_oss = (5/200)² · 100e-6 = 6.25e-8 = 62.5 nF.
        assert rec.C_oss == pytest.approx(62.5e-9, rel=1e-6)

    def test_strict_target_infeasible_at_high_fsw(self):
        """20 % overshoot at L=100µH, 100 kHz boost gives t_rise > t_off →
        the advisor must flag this as infeasible. This pins the verdict
        that PWL Ideal can't deliver bounded overshoot AND boost transfer
        at the same time at hard-switched-converter frequencies."""
        rec = snubber.recommend_C_oss(L=100e-6, I_peak=5.0, V_bus=400.0,
                                        max_overshoot_frac=0.2,
                                        f_sw=100e3, duty_off=0.75)
        assert not rec.feasible
        assert "t_rise > OFF interval" in rec.notes

    def test_loose_target_feasible(self):
        """50 % overshoot at the same operating point — feasible because
        C_oss shrinks enough that t_rise (5 µs) < OFF interval (7.5 µs)."""
        rec = snubber.recommend_C_oss(L=100e-6, I_peak=5.0, V_bus=400.0,
                                        max_overshoot_frac=0.5,
                                        f_sw=100e3, duty_off=0.75)
        assert rec.feasible
        assert rec.t_rise_to_V_bus < 0.75 / 100e3

    def test_loss_formula(self):
        """P_loss = ½·C·V²·f_sw (hard-switched). Verify against the
        closed form so refactors can't sneak a factor of 2 wrong again."""
        rec = snubber.recommend_C_oss(L=100e-6, I_peak=5.0, V_bus=400.0,
                                        max_overshoot_frac=0.5,
                                        f_sw=100e3)
        expected = 0.5 * 62.5e-9 * 400.0 ** 2 * 100e3
        assert rec.P_loss_estimate == pytest.approx(expected, rel=1e-6)
        # Sanity: this should be 500 W — clearly too much for a 550 W
        # converter, hence "Behavioral mode" is the right call here.
        assert rec.P_loss_estimate == pytest.approx(500.0, rel=1e-3)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            snubber.recommend_C_oss(L=-1, I_peak=5, V_bus=400)
        with pytest.raises(ValueError):
            snubber.recommend_C_oss(L=100e-6, I_peak=0, V_bus=400)
        with pytest.raises(ValueError):
            snubber.recommend_C_oss(L=100e-6, I_peak=5, V_bus=-1)
        with pytest.raises(ValueError):
            snubber.recommend_C_oss(L=100e-6, I_peak=5, V_bus=400,
                                    max_overshoot_frac=0.0)


# -------------- recommend_rc_snubber ----------------------------------------

class TestRecommendRCSnubber:
    def test_critical_damping_R(self):
        """For critical damping (ζ = 1), R = 2·√(L/C)."""
        rec = snubber.recommend_rc_snubber(L=100e-6, C=10e-9)
        # 2·√(100e-6/10e-9) = 2·√(1e4) = 200 Ω.
        assert rec.R == pytest.approx(200.0, rel=1e-6)
        assert rec.zeta == 1.0

    def test_underdamped(self):
        """ζ = 0.5 → R halves."""
        rec = snubber.recommend_rc_snubber(L=100e-6, C=10e-9, zeta=0.5)
        assert rec.R == pytest.approx(100.0, rel=1e-6)

    def test_dissipation_formula(self):
        """Same ½·C·V²·f_sw as C_oss switching loss."""
        rec = snubber.recommend_rc_snubber(L=100e-6, C=10e-9,
                                             V_bus=400.0, f_sw=100e3)
        expected = 0.5 * 10e-9 * 400.0 ** 2 * 100e3
        assert rec.P_dissipation_estimate == pytest.approx(expected, rel=1e-6)
        # = 80 W — large fraction of a 550 W converter, hence the
        # advisor's recommendation to drop to Behavioral.
        assert rec.P_dissipation_estimate == pytest.approx(80.0, rel=1e-3)


# -------------- advise() (integration) --------------------------------------

class TestAdvise:
    def test_advise_prints_to_file(self):
        buf = io.StringIO()
        snubber.advise(L=100e-6, I_peak=5.0, V_bus=400.0, file=buf)
        text = buf.getvalue()
        assert "Boost-class PWL snubber-sizing report" in text
        # Should cite both the recommended C_oss AND the runtime's 10 nF
        # default so the user sees the trade-off.
        assert "10 nF" in text
        # And state a verdict.
        assert "VERDICT" in text

    def test_advise_default_boost_flags_behavioral(self):
        """At the canonical 100 µH / 5 A / 400 V / 100 kHz boost op-point,
        the verdict must be 'drop to Behavioral mode'."""
        buf = io.StringIO()
        snubber.advise(L=100e-6, I_peak=5.0, V_bus=400.0, file=buf)
        text = buf.getvalue()
        assert "NOT recommended" in text or "Behavioral" in text
