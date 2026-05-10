"""Phase 5 of `add-frequency-domain-analysis`: Python bindings + plotting
helpers integration test.

Pins the Python-facing contract:
- `pulsim.Simulator.linearize_around` returns a `LinearSystem`
- `pulsim.Simulator.run_ac_sweep` returns an `AcSweepResult` with valid
  Bode data
- `pulsim.Simulator.run_fra` returns a `FraResult` consistent with
  `run_ac_sweep` within ≤ 1 dB / ≤ 5°
- `pulsim.export_ac_csv` writes parseable CSV
- `pulsim.bode_plot` and `pulsim.nyquist_plot` produce matplotlib output
  when the user has matplotlib installed (skipped otherwise)
"""

from __future__ import annotations

import csv
import math
import os
import tempfile

import pytest

pulsim = pytest.importorskip("pulsim")


def _build_rc_circuit():
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 1.0)
    ckt.add_resistor("R1", in_, out, 1e3)
    ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 1e-6
    opts.dt = 1e-7
    opts.dt_min = 1e-15
    opts.dt_max = 1e-7
    opts.adaptive_timestep = False
    opts.newton_options.num_nodes = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()
    return ckt, opts


def test_linearize_around_returns_pwl_descriptor_form():
    ckt, opts = _build_rc_circuit()
    sim = pulsim.Simulator(ckt, opts)
    dc = sim.dc_operating_point()
    assert dc.success

    sys = sim.linearize_around(dc.newton_result.solution, 0.0)
    assert sys.ok
    assert sys.method == "piecewise_linear_segment"
    assert sys.state_size == ckt.system_size()
    assert sys.input_size == 1
    assert sys.output_size == sys.state_size


def test_ac_sweep_rc_corner_matches_analytical():
    ckt, opts = _build_rc_circuit()
    sim = pulsim.Simulator(ckt, opts)

    ac = pulsim.AcSweepOptions()
    ac.f_start = 1.0
    ac.f_stop = 1e6
    ac.points_per_decade = 30
    ac.scale = pulsim.AcSweepScale.Logarithmic
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]

    result = sim.run_ac_sweep(ac)
    assert result.success
    assert len(result.frequencies) > 100
    assert result.total_factorizations == len(result.frequencies)

    f_corner = 1.0 / (2.0 * math.pi * 1e3 * 1e-6)
    m = result.measurements[0]
    i_corner = min(
        range(len(result.frequencies)),
        key=lambda i: abs(math.log10(result.frequencies[i]) - math.log10(f_corner)),
    )
    # Analytical: -3.01 dB / -45° at the RC corner.
    assert abs(m.magnitude_db[i_corner] - (-3.0103)) < 0.2
    assert abs(m.phase_deg[i_corner] - (-45.0)) < 2.0


def test_fra_agrees_with_ac_sweep_within_tolerance():
    ckt, opts = _build_rc_circuit()
    sim = pulsim.Simulator(ckt, opts)

    f_corner = 1.0 / (2.0 * math.pi * 1e3 * 1e-6)

    ac = pulsim.AcSweepOptions()
    ac.f_start = f_corner / 10
    ac.f_stop = f_corner * 10
    ac.points_per_decade = 4
    ac.scale = pulsim.AcSweepScale.Logarithmic
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]
    ac_result = sim.run_ac_sweep(ac)
    assert ac_result.success

    fra = pulsim.FraOptions()
    fra.f_start = ac.f_start
    fra.f_stop = ac.f_stop
    fra.points_per_decade = ac.points_per_decade
    fra.scale = ac.scale
    fra.perturbation_source = "V1"
    fra.perturbation_amplitude = 1e-2
    fra.measurement_nodes = ["out"]
    fra.n_cycles = 6
    fra.discard_cycles = 2
    fra.samples_per_cycle = 64
    fra_result = sim.run_fra(fra)
    assert fra_result.success
    assert len(fra_result.frequencies) == len(ac_result.frequencies)

    ac_m = ac_result.measurements[0]
    fra_m = fra_result.measurements[0]
    for i in range(len(ac_result.frequencies)):
        assert abs(fra_m.magnitude_db[i] - ac_m.magnitude_db[i]) < 1.0, (
            f"frequency {ac_result.frequencies[i]} Hz: |Δ mag| > 1 dB"
        )
        assert abs(fra_m.phase_deg[i] - ac_m.phase_deg[i]) < 5.0, (
            f"frequency {ac_result.frequencies[i]} Hz: |Δ phase| > 5°"
        )


def test_export_ac_csv_writes_parseable_file():
    ckt, opts = _build_rc_circuit()
    sim = pulsim.Simulator(ckt, opts)

    ac = pulsim.AcSweepOptions()
    ac.f_start = 10.0
    ac.f_stop = 1e4
    ac.points_per_decade = 5
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]
    result = sim.run_ac_sweep(ac)
    assert result.success

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fp:
        path = fp.name
    try:
        pulsim.export_ac_csv(result, path, format="magphase")
        with open(path) as fp:
            reader = csv.reader(fp)
            rows = list(reader)
        assert rows[0] == ["frequency_hz", "mag_db_out", "phase_deg_out"]
        assert len(rows) - 1 == len(result.frequencies)
    finally:
        os.unlink(path)


def test_bode_plot_produces_matplotlib_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    ckt, opts = _build_rc_circuit()
    sim = pulsim.Simulator(ckt, opts)

    ac = pulsim.AcSweepOptions()
    ac.f_start = 10.0
    ac.f_stop = 1e4
    ac.points_per_decade = 5
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]
    result = sim.run_ac_sweep(ac)
    assert result.success

    fig, (ax_mag, ax_phase) = pulsim.bode_plot(result, title="RC low-pass")
    assert fig is not None
    assert ax_mag is not None
    assert ax_phase is not None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
        path = fp.name
    try:
        fig.savefig(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_bode_plot_rejects_failed_result():
    failed = pulsim.AcSweepResult()
    # Default-constructed: success = False.
    assert not failed.success
    with pytest.raises(ValueError, match="not successful"):
        pulsim.bode_plot(failed)


def test_phase6_export_json_roundtrip():
    """Phase 6.2: export_ac_json produces a JSON document that captures
    every public field of an AcSweepResult.
    """
    import json

    ckt, opts = _build_rc_circuit()
    sim = pulsim.Simulator(ckt, opts)
    ac = pulsim.AcSweepOptions()
    ac.f_start = 10.0
    ac.f_stop = 1e3
    ac.points_per_decade = 3
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]
    result = sim.run_ac_sweep(ac)
    assert result.success

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fp:
        path = fp.name
    try:
        pulsim.export_ac_json(result, path)
        with open(path) as fp:
            doc = json.load(fp)
        assert doc["kind"] == "AcSweepResult"
        assert doc["success"] is True
        assert len(doc["frequencies"]) == len(result.frequencies)
        assert len(doc["measurements"]) == 1
        assert doc["measurements"][0]["node"] == "out"
        assert "magnitude_db" in doc["measurements"][0]
        assert "total_factorizations" in doc
    finally:
        os.unlink(path)


def test_phase6_load_ac_result_csv_roundtrips_via_plot():
    """Phase 6.4: writing a CSV and loading it back yields a plot-shaped
    container that bode_plot can render."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    ckt, opts = _build_rc_circuit()
    sim = pulsim.Simulator(ckt, opts)
    ac = pulsim.AcSweepOptions()
    ac.f_start = 10.0
    ac.f_stop = 1e3
    ac.points_per_decade = 4
    ac.perturbation_source = "V1"
    ac.measurement_nodes = ["out"]
    result = sim.run_ac_sweep(ac)
    assert result.success

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fp:
        path = fp.name
    try:
        pulsim.export_ac_csv(result, path, format="magphase")
        loaded = pulsim.load_ac_result_csv(path, format="magphase")
        assert loaded.success
        assert len(loaded.frequencies) == len(result.frequencies)
        assert len(loaded.measurements) == 1
        assert loaded.measurements[0].node == "out"
        # bode_plot accepts the loaded container — quacks like a result.
        fig, _ = pulsim.bode_plot(loaded, title="from CSV")
        assert fig is not None
    finally:
        os.unlink(path)


def test_phase4_multi_input_perturbation_sources():
    """Phase 4: passing perturbation_sources (list) returns one
    AcMeasurement per (source, node) pair, each carrying both labels."""
    ckt = pulsim.Circuit()
    a = ckt.add_node("a")
    b = ckt.add_node("b")
    out = ckt.add_node("out")
    ckt.add_voltage_source("Va", a, ckt.ground(), 1.0)
    ckt.add_voltage_source("Vb", b, ckt.ground(), 1.0)
    ckt.add_resistor("Ra", a, out, 1e3)
    ckt.add_resistor("Rb", b, out, 1e3)
    ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 1e-6
    opts.dt = 1e-7
    opts.dt_min = 1e-12
    opts.dt_max = 1e-7
    opts.adaptive_timestep = False
    opts.newton_options.num_nodes = ckt.num_nodes()
    opts.newton_options.num_branches = ckt.num_branches()

    sim = pulsim.Simulator(ckt, opts)
    ac = pulsim.AcSweepOptions()
    ac.f_start = 10.0
    ac.f_stop = 1e3
    ac.points_per_decade = 3
    ac.perturbation_sources = ["Va", "Vb"]
    ac.measurement_nodes = ["out"]
    result = sim.run_ac_sweep(ac)
    assert result.success
    assert len(result.measurements) == 2
    sources_seen = {m.perturbation_source for m in result.measurements}
    assert sources_seen == {"Va", "Vb"}
