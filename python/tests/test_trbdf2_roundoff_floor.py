"""TR-BDF2 error estimate at its own roundoff floor.

benchmarks/circuits/switching/buck_boost/sepic.yaml at rtol 1e-8 /
atol 1e-12 ran to max_steps: with the switch and the diode both off, sw
and mid are held only by g_off, the estimate's roundoff floor (~1e-10 A,
independent of h) sat above its scale (1.5e-11 A), every step was
rejected down to h_min, and there the KCL noise turned the OFF diode 5 V
forward. Zero-time diode "events" that changed no bit were recorded and
retaken three times, then decided from noise.

Counters, not wall time.
"""

import sys
import warnings
from pathlib import Path

import pytest

import pulsim

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "benchmarks" / "tools")]
pytest.importorskip("yaml")
import v1_reader  # noqa: E402

SEPIC = ROOT / "benchmarks/circuits/switching/buck_boost/sepic.yaml"


def test_sepic_tight_tolerance_finishes_in_bounded_steps():
    """Hit max_steps (1e7) at t = 0.776 ms before this change; takes
    28,539 accepted steps after it. Bound: twice 24,926 samples, the
    target set for this fix."""
    c = v1_reader.load(str(SEPIC))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = c.run("trbdf2", rtol=1e-8, atol=1e-12)
    st = r._trbdf2_stats
    assert r.times[-1] == pytest.approx(c.t_end, rel=1e-9)
    assert st["n_accept"] <= 2 * 24926
    assert st["n_roundoff_floor_accepts"] > 0   # the mechanism ran
    assert abs(float(r.v("out")[-1]) - 17.3157) < 1e-4


def test_diode_bridge_rtol_1e_6_finishes():
    """The capacitor-input bridge of probes/trbdf2_baseline.py at rtol
    1e-6 hit max_steps (1e7) at t = 10.2 ms before this change; it takes
    1,971 accepted steps after it."""
    b = pulsim.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "s", "a", 0.5)
    b.add_inductor("Ls", "a", "ac1", 1e-3)
    b.add_diode("D1", "ac1", "dp", 1e3, 1e-9)
    b.add_diode("D2", "gnd", "dp", 1e3, 1e-9)
    b.add_diode("D3", "dn", "ac1", 1e3, 1e-9)
    b.add_diode("D4", "dn", "gnd", 1e3, 1e-9)
    b.add_capacitor("C", "dp", "dn", 470e-6)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    r = pulsim.simulate(b, t_end=40e-3, engine="trbdf2",
                        rtol=1e-6, atol=1e-9)
    assert r.times[-1] == pytest.approx(40e-3, rel=1e-9)
    assert r._trbdf2_stats["n_accept"] <= 4000
