"""Regression: accessing 1.5-era helper symbols on the ``pulsim``
module must raise ``AttributeError`` with an actionable migration
hint pointing at the right 1.6 replacement (GUI integration findings
T3.1).

These tests pin the *content* of the hint dictionary — if a future
refactor accidentally drops one of these entries, the hint
disappears and downstream binders silently get a bare
``AttributeError`` again. The hints are the only "1.5 → 1.6
migration" UX we have in-source; the rest lives in
``docs/migration-guide.md``.
"""
from __future__ import annotations

import pytest

import pulsim as p


class TestSweepHelpersRetiredIn16:
    """``pulsim.sweep.Distribution`` / ``.Cartesian`` / ``.metrics``
    were retired when ``pulsim.sweep`` collapsed from a subpackage
    to a single function. Direct ``pulsim.Distribution`` access
    (common in GUI converters that import top-level) must fail loudly.
    """

    def test_distribution_hint(self):
        with pytest.raises(AttributeError) as exc:
            _ = p.Distribution
        msg = str(exc.value)
        assert "Distribution" in msg
        assert "monte_carlo" in msg, (
            f"hint should redirect at p.monte_carlo: {msg!r}")
        assert "rng.normal" in msg or "lambda" in msg, (
            f"hint should show the new lambda(rng) shape: {msg!r}")
        assert "migration-guide" in msg

    def test_cartesian_hint(self):
        with pytest.raises(AttributeError) as exc:
            _ = p.Cartesian
        msg = str(exc.value)
        assert "params=" in msg, (
            f"hint should mention the new params= kwarg: {msg!r}")
        assert "p.sweep" in msg

    def test_metrics_hint(self):
        with pytest.raises(AttributeError) as exc:
            _ = p.metrics
        msg = str(exc.value)
        assert "kpi_fn" in msg, (
            f"hint should redirect at the new kpi_fn contract: "
            f"{msg!r}")


class TestNoParamsStructIn16:
    """1.5 had ``PmsmParams`` / ``ThreePhaseVsiParams`` / ``BldcParams``
    POD structs paired with a ``circuit.add_pmsm(params)`` shape. 1.6
    moved to module-level functions with direct kwargs and no
    parallel struct."""

    def test_PmsmParams_hint(self):
        with pytest.raises(AttributeError) as exc:
            _ = p.PmsmParams
        msg = str(exc.value)
        assert "no params struct" in msg
        assert "add_pmsm" in msg
        assert "R_s=" in msg or "kwargs" in msg

    def test_ThreePhaseVsiParams_hint(self):
        with pytest.raises(AttributeError) as exc:
            _ = p.ThreePhaseVsiParams
        msg = str(exc.value)
        assert "add_three_phase_vsi" in msg

    def test_BldcParams_hint(self):
        with pytest.raises(AttributeError) as exc:
            _ = p.BldcParams
        msg = str(exc.value)
        assert "add_bldc" in msg


class TestSweepStillUsableViaTheNewContract:
    """Sanity: the new contract for ``p.sweep`` and ``p.monte_carlo``
    still works — the hints aren't masking a regression."""

    def test_sweep_function_callable(self):
        # `p.sweep` resolves to the function (not the retired
        # subpackage), so accessing it does NOT raise.
        assert callable(p.sweep)

    def test_monte_carlo_function_callable(self):
        assert callable(p.monte_carlo)
