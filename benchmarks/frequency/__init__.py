"""Frequency-domain analysis helpers for benchmark validation.

This package wraps Pulsim's existing `Simulator.run_fra` (Frequency Response
Analysis: per-frequency injection + transient + Goertzel DFT) and
`Simulator.run_ac_sweep` (linearized small-signal solve) into a Bode-plot
workflow that compares the measured response against an analytical model.

This is the design-grade feature that distinguishes a transient-only tool
from a compensator-design tool — PLECS Smart Control's headline.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class BodePoint:
    """A single (f, |H|_dB, ∠H_deg) measurement."""
    freq_hz: float
    magnitude_db: float
    phase_deg: float


@dataclass
class BodeMeasurement:
    """Result of a single AC sweep over a frequency range."""
    points: List[BodePoint] = field(default_factory=list)
    perturbation_source: str = ""
    observable: str = ""
    wall_seconds: float = 0.0

    def frequencies(self) -> List[float]:
        return [p.freq_hz for p in self.points]

    def magnitudes_db(self) -> List[float]:
        return [p.magnitude_db for p in self.points]

    def phases_deg(self) -> List[float]:
        return [p.phase_deg for p in self.points]


@dataclass
class BodeComparison:
    """Result of compare_to_analytical: per-point and aggregate errors."""
    db_errors: List[float] = field(default_factory=list)
    phase_errors_deg: List[float] = field(default_factory=list)
    max_db_err: float = 0.0
    max_phase_err_deg: float = 0.0


def run_fra_sweep(
    yaml_path: Path,
    perturbation_source: str,
    observable: str,
    f_start: float,
    f_stop: float,
    points_per_decade: int = 10,
    amplitude: float = 0.01,
    settle_cycles: int = 5,
    measure_cycles: int = 5,
    samples_per_cycle: int = 64,
    use_initial_conditions: bool = True,
) -> BodeMeasurement:
    """Run an FRA sweep on a Pulsim YAML netlist.

    Returns the measured Bode response of `observable` to perturbations on
    `perturbation_source`. Uses `Simulator.run_fra` (transient injection +
    Goertzel), which is what we want for switching converters whose
    small-signal behaviour is not strictly linear.

    `observable` should be a node name (without the V() wrapper) that
    matches one of the circuit's node names.
    """
    import pulsim
    import time

    # Load + parse the YAML — load() returns (circuit, options)
    parser_opts = pulsim.YamlParserOptions()
    parser_opts.strict = False
    parser = pulsim.YamlParser(parser_opts)
    circuit, options = parser.load(str(yaml_path))
    if parser.errors:
        raise RuntimeError("YAML parse errors: " + "; ".join(str(e) for e in parser.errors))

    # Wire the Newton options' node/branch counts (matches the benchmark backend)
    options.newton_options.num_nodes = int(circuit.num_nodes())
    options.newton_options.num_branches = int(circuit.num_branches())

    sim = pulsim.Simulator(circuit, options)

    # FRA options
    fra_opts = pulsim.FraOptions()
    fra_opts.f_start = f_start
    fra_opts.f_stop = f_stop
    fra_opts.points_per_decade = points_per_decade
    fra_opts.perturbation_source = perturbation_source
    fra_opts.perturbation_amplitude = amplitude
    fra_opts.perturbation_phase = 0.0
    fra_opts.measurement_nodes = [observable]
    fra_opts.n_cycles = measure_cycles
    fra_opts.discard_cycles = settle_cycles
    fra_opts.samples_per_cycle = samples_per_cycle

    t0 = time.perf_counter()
    result = sim.run_fra(fra_opts)
    wall = time.perf_counter() - t0

    if not result.success:
        raise RuntimeError(f"FRA failed: {result.failure_reason}")

    points: List[BodePoint] = []
    if result.measurements:
        m = result.measurements[0]
        for f, db, phase in zip(result.frequencies, m.magnitude_db, m.phase_deg):
            points.append(BodePoint(freq_hz=f, magnitude_db=db, phase_deg=phase))

    return BodeMeasurement(
        points=points,
        perturbation_source=perturbation_source,
        observable=observable,
        wall_seconds=wall,
    )


def compare_to_analytical(
    measured: BodeMeasurement,
    model_fn: Callable[[float], complex],
) -> BodeComparison:
    """Score measured Bode points against an analytical transfer function.

    `model_fn(f_hz) -> complex H(jω)` should return the complex transfer
    function value at frequency f. Magnitude is compared in dB; phase in
    degrees. Both errors are reported per-point and as max-over-window.
    """
    db_errors: List[float] = []
    phase_errors: List[float] = []

    for pt in measured.points:
        h = model_fn(pt.freq_hz)
        mag = abs(h)
        if mag <= 1e-30:
            db_model = -300.0
        else:
            db_model = 20.0 * math.log10(mag)
        phase_model_deg = math.degrees(math.atan2(h.imag, h.real))

        db_errors.append(pt.magnitude_db - db_model)
        # Phase wrapping: bring both into the same 360° window
        dphase = pt.phase_deg - phase_model_deg
        while dphase > 180:
            dphase -= 360
        while dphase < -180:
            dphase += 360
        phase_errors.append(dphase)

    return BodeComparison(
        db_errors=db_errors,
        phase_errors_deg=phase_errors,
        max_db_err=max((abs(e) for e in db_errors), default=0.0),
        max_phase_err_deg=max((abs(e) for e in phase_errors), default=0.0),
    )


def extract_margins(measured: BodeMeasurement) -> Dict[str, Optional[float]]:
    """Locate gain margin (dB), phase margin (deg), and crossover frequency.

    - Crossover: where |H| = 0 dB (linear interpolation between adjacent
      points if the curve crosses).
    - Phase margin: 180° + ∠H at crossover.
    - Gain margin: -|H|_dB at the frequency where ∠H = -180°.

    Returns Nones when the curve doesn't cross the relevant axes.
    """
    out: Dict[str, Optional[float]] = {
        "crossover_hz": None,
        "phase_margin_deg": None,
        "gain_margin_db": None,
    }

    if len(measured.points) < 2:
        return out

    # crossover (|H| crosses 0 dB going downward)
    for i in range(1, len(measured.points)):
        a, b = measured.points[i - 1], measured.points[i]
        if (a.magnitude_db >= 0.0 and b.magnitude_db < 0.0) or \
           (a.magnitude_db < 0.0 and b.magnitude_db >= 0.0):
            # Linear-in-log-f interpolation
            if b.magnitude_db != a.magnitude_db:
                t = -a.magnitude_db / (b.magnitude_db - a.magnitude_db)
            else:
                t = 0.0
            t = max(0.0, min(1.0, t))
            log_f_cross = math.log10(a.freq_hz) + t * (math.log10(b.freq_hz) - math.log10(a.freq_hz))
            f_cross = 10 ** log_f_cross
            phase_cross = a.phase_deg + t * (b.phase_deg - a.phase_deg)
            out["crossover_hz"] = f_cross
            out["phase_margin_deg"] = 180.0 + phase_cross
            break

    # gain margin (|H| at frequency where ∠H = -180°)
    for i in range(1, len(measured.points)):
        a, b = measured.points[i - 1], measured.points[i]
        # Look for -180° crossing (going down)
        if (a.phase_deg >= -180.0 and b.phase_deg < -180.0) or \
           (a.phase_deg < -180.0 and b.phase_deg >= -180.0):
            if b.phase_deg != a.phase_deg:
                t = (-180.0 - a.phase_deg) / (b.phase_deg - a.phase_deg)
            else:
                t = 0.0
            t = max(0.0, min(1.0, t))
            mag_at_180 = a.magnitude_db + t * (b.magnitude_db - a.magnitude_db)
            out["gain_margin_db"] = -mag_at_180
            break

    return out


__all__ = [
    "BodePoint",
    "BodeMeasurement",
    "BodeComparison",
    "run_fra_sweep",
    "compare_to_analytical",
    "extract_margins",
]
