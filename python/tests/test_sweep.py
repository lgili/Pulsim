"""Phase 9 of `add-monte-carlo-parameter-sweep`: sweep tests.

Pins:
  - Distribution inverse-CDFs are correct (uniform/normal/loguniform/triangular).
  - Cartesian sampling enumerates the full product.
  - LHS / Sobol produce `n_samples` rows in the unit hypercube.
  - Reproducibility: identical seed → identical samples.
  - Metric library extracts the right scalars from a SimulationResult.
  - End-to-end sweep on a passive RC circuit succeeds and produces
    a non-empty SweepResult.
"""

from __future__ import annotations

import math

import pytest

pulsim = pytest.importorskip("pulsim")
np = pytest.importorskip("numpy")
sw = pulsim.sweep


# -----------------------------------------------------------------------------
# Phase 2 — distributions
# -----------------------------------------------------------------------------

def test_uniform_inverse_cdf_at_quartiles():
    d = sw.Distribution.uniform(low=0.0, high=10.0)
    assert d.inverse_cdf(np.array([0.0, 0.25, 0.5, 0.75, 1.0])).tolist() == \
           pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0])


def test_normal_inverse_cdf_zero_mean_returns_zero_at_median():
    d = sw.Distribution.normal(mean=0.0, std=1.0)
    assert abs(float(d.inverse_cdf(np.array([0.5]))[0])) < 1e-9


def test_loguniform_quantile_is_log_linear():
    d = sw.Distribution.loguniform(low=1.0, high=100.0)
    # log10(quantile) at p = 0.5 should be (log10(1) + log10(100)) / 2 = 1.
    val = float(d.inverse_cdf(np.array([0.5]))[0])
    assert math.isclose(math.log10(val), 1.0, abs_tol=1e-9)


def test_triangular_quantile_at_mode():
    d = sw.Distribution.triangular(low=0.0, mode=2.0, high=10.0)
    # CDF at mode: c = (mode - low) / (high - low) = 0.2.
    val = float(d.inverse_cdf(np.array([0.2]))[0])
    assert math.isclose(val, 2.0, abs_tol=1e-9)


def test_uniform_rejects_inverted_range():
    with pytest.raises(ValueError):
        sw.Distribution.uniform(low=10.0, high=0.0)


# -----------------------------------------------------------------------------
# Phase 2 — sampling strategies
# -----------------------------------------------------------------------------

def test_cartesian_enumerates_full_product():
    samples = sw.sample(
        {
            "a": [1.0, 2.0],
            "b": [10.0, 20.0, 30.0],
        },
        n_samples=0,        # ignored for cartesian
        strategy="cartesian",
    )
    assert len(samples) == 6
    seen = {(s["a"], s["b"]) for s in samples}
    assert seen == {(1.0, 10.0), (1.0, 20.0), (1.0, 30.0),
                    (2.0, 10.0), (2.0, 20.0), (2.0, 30.0)}


def test_monte_carlo_seeded_produces_identical_samples():
    spec = {"x": sw.Distribution.uniform(0.0, 1.0)}
    a = sw.sample(spec, n_samples=64, strategy="monte_carlo", seed=42)
    b = sw.sample(spec, n_samples=64, strategy="monte_carlo", seed=42)
    assert a == b


def test_lhs_produces_n_samples_in_unit_hypercube():
    spec = {
        "x": sw.Distribution.uniform(0.0, 1.0),
        "y": sw.Distribution.uniform(0.0, 1.0),
    }
    samples = sw.sample(spec, n_samples=64, strategy="lhs", seed=7)
    assert len(samples) == 64
    for s in samples:
        assert 0.0 <= s["x"] <= 1.0
        assert 0.0 <= s["y"] <= 1.0


def test_sobol_quasi_random_samples_have_low_discrepancy():
    spec = {"x": sw.Distribution.uniform(0.0, 1.0)}
    samples = sw.sample(spec, n_samples=64, strategy="sobol", seed=0)
    xs = sorted(s["x"] for s in samples)
    # Sobol of 64 points in 1D should fill (0, 1) more uniformly than IID.
    # Max gap between consecutive sorted samples is a low-discrepancy
    # proxy: ≤ ~3/N for Sobol.
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    assert max(gaps) < 3.0 / 64.0 * 2     # generous safety factor


# -----------------------------------------------------------------------------
# Phase 4 — metrics
# -----------------------------------------------------------------------------

def test_metric_steady_state_uses_last_10pct_window_by_default():
    """Build a tiny RC and verify steady_state metric returns the
    asymptotic value (≈ Vin = 5)."""
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 5.0)
    ckt.add_resistor("R1", in_, out, 1e3)
    ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)

    opts = pulsim.SimulationOptions()
    opts.tstop = 1e-2          # 10 time constants → fully settled
    opts.dt = 1e-5
    opts.dt_max = 1e-5
    opts.adaptive_timestep = False
    opts.newton_options.num_nodes = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()

    sim = pulsim.Simulator(ckt, opts)
    dc = sim.dc_operating_point()
    run = sim.run_transient(dc.newton_result.solution)
    assert run.success

    ss = sw.steady_state("out")(run, ckt)
    assert math.isclose(ss, 5.0, rel_tol=0.01)


# -----------------------------------------------------------------------------
# Phase 1 / 9 — end-to-end sweep
# -----------------------------------------------------------------------------

def test_sweep_end_to_end_on_rc_circuit():
    """Sweep R across 3 values; confirm steady-state metric tracks."""
    def make_rc(R):
        ckt = pulsim.Circuit()
        in_ = ckt.add_node("in")
        out = ckt.add_node("out")
        ckt.add_voltage_source("V1", in_, ckt.ground(), 5.0)
        ckt.add_resistor("R1", in_, out, R)
        ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)
        return ckt

    def make_opts():
        opts = pulsim.SimulationOptions()
        opts.tstop = 5e-3
        opts.dt = 1e-5
        opts.dt_max = 1e-5
        opts.adaptive_timestep = False
        return opts

    result = sw.run(
        circuit_factory=make_rc,
        parameters={"R": [500.0, 1000.0, 2000.0]},
        metrics=[sw.steady_state("out")],
        n_samples=0,                    # cartesian ignores
        strategy="cartesian",
        sim_options_factory=make_opts,
    )
    assert result.n_samples == 3
    assert result.n_succeeded == 3
    assert result.n_failed == 0
    # Every sample's steady state should be ≈ 5 V (DC analysis on RC).
    for k in range(3):
        ss = result.metrics[k]["steady_state[out]"]
        assert math.isclose(ss, 5.0, abs_tol=0.5)


def test_sweep_records_failure_reasons():
    """A factory that throws should be captured in `failed`, not crash
    the sweep."""
    def bad_factory(R):
        if R < 0:
            raise ValueError(f"negative R: {R}")
        ckt = pulsim.Circuit()
        a = ckt.add_node("a")
        b = ckt.add_node("b")
        ckt.add_voltage_source("V1", a, ckt.ground(), 1.0)
        ckt.add_resistor("R1", a, b, R)
        ckt.add_resistor("R2", b, ckt.ground(), 1.0)
        return ckt

    result = sw.run(
        circuit_factory=bad_factory,
        parameters={"R": [-1.0, 1.0]},
        metrics=[sw.steady_state("b", t_window=(0.0, 1e-4))],
        strategy="cartesian",
    )
    assert result.n_samples == 2
    assert result.n_failed >= 1
    assert any("negative R" in (f or "") for f in result.failed)


def test_sweep_to_pandas_returns_wide_frame():
    pd = pytest.importorskip("pandas")

    def make_rc(R):
        ckt = pulsim.Circuit()
        in_ = ckt.add_node("in")
        out = ckt.add_node("out")
        ckt.add_voltage_source("V1", in_, ckt.ground(), 1.0)
        ckt.add_resistor("R1", in_, out, R)
        ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)
        return ckt

    result = sw.run(
        circuit_factory=make_rc,
        parameters={"R": [1000.0, 2000.0]},
        metrics=[sw.steady_state("out")],
        strategy="cartesian",
        sim_options_factory=lambda: _quick_opts(),
    )
    df = result.to_pandas()
    assert len(df) == 2
    assert "R" in df.columns
    assert "steady_state[out]" in df.columns


def _quick_opts():
    opts = pulsim.SimulationOptions()
    opts.tstop = 5e-3
    opts.dt = 1e-5
    opts.dt_max = 1e-5
    opts.adaptive_timestep = False
    return opts
