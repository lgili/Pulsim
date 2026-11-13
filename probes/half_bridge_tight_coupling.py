"""Probe: tightly coupled half bridge (projects/converters/half_bridge).

Reference (examples audit, 2026-11-13): at k = 0.999 and 0.99999,
engine='trbdf2' raised "refactorize failed (numerically singular)" at
h ~ 5e-14 s; at k = 0.99999 the fixed step read 4.17 V at dt = 2e-8
against 4.99 V at dt = 1e-8 (ideal n·D·V_g = 5.0 V) with no warning.
Owner: core/include/pulsim/solver/trbdf2_transient.hpp (step-size floor
and refactorization on a nearly singular coupled pair) and the fixed
engine's missing convergence warning (run_transient.hpp).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/converters/half_bridge"))
import pulsim as p  # noqa: E402
from half_bridge_model import HalfBridgeParams  # noqa: E402
from half_bridge_pulsim_validation import (  # noqa: E402
    _alternating_switch_fn, _build_builder, simulate_half_bridge,
)

bp = HalfBridgeParams()
for k in (0.999, 0.99999):
    for dt in (2e-8, 1e-8):
        t, v = simulate_half_bridge(bp, t_end=3e-3, dt=dt, k_coupling=k)
        print(f"k={k}: pwl dt={dt:.0e}: V_o = {p.time_average(t, v, t_from=2e-3):.3f} V "
              f"(ideal {bp.n * bp.D * bp.V_g:.2f})")
    b = _build_builder(bp, k_coupling=k)
    sw = _alternating_switch_fn(b.graph.num_switches, bp.f_sw, bp.D, bp.t_dead,
                                b.switch_index_of("S1"), b.switch_index_of("S2"))
    try:
        r = p.simulate(b, t_end=3e-3, switch_fn=sw, engine="trbdf2")
        v = np.asarray(r.v("vout"))
        print(f"k={k}: trbdf2: V_o = {p.time_average(np.asarray(r.times), v, t_from=2e-3):.3f} V")
    except Exception as e:  # report the refusal by name
        print(f"k={k}: trbdf2: {type(e).__name__}: {str(e)[:160]}")
