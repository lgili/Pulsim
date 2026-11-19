"""The trbdf2 engine's time array never goes backward (an engine invariant).

A diode event re-decided at the same t (the chatter-safe zero-time cascade)
recorded a second post-event sample without dropping the first, pending one:
when the second probe step was the shorter, the record went backward (CI
Linux, the PSFB example: t[63648] 2.0e-12 s before t[63647]). The kernel now
raises ``pulsim.InternalError`` on a backward sample instead of returning it.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

from .test_bridge_ground_truth import _bridge

_PSFB = Path(__file__).resolve().parents[2] / "examples" / "phase_shift_full_bridge.yaml"
_KEYS = ("V", "L", "L_p", "L_s", "C", "R", "k")


def _psfb(values):
    """The PSFB example with {(device, key): value} substituted."""
    blocks = re.split(r"(?m)^(?=    - type:)", _PSFB.read_text(encoding="utf-8"))
    for i, blk in enumerate(blocks):
        name = re.search(r"(?m)^      name: (\S+)", blk)
        for (dev, key), v in values.items():
            if name and name.group(1) == dev:
                blk, n = re.subn(rf"(?m)^(      {key}:\s*)\S+", rf"\g<1>{v!r}", blk)
                assert n == 1, (dev, key)
        blocks[i] = blk
    return p.load_yaml_string("".join(blocks))


def _psfb_defaults():
    vals = {}
    for blk in re.split(r"(?m)^(?=    - type:)", _PSFB.read_text(encoding="utf-8")):
        name = re.search(r"(?m)^      name: (\S+)", blk)
        for key in _KEYS:
            m = re.search(rf"(?m)^      {key}:\s*(\S+)", blk)
            if name and m:
                vals[(name.group(1), key)] = float(m.group(1))
    return vals


def _run_psfb(values):
    lc = _psfb(values)
    si, n = lc.builder.switch_index_of, len(lc.switch_names)
    fn = p.make_phase_shift_full_bridge_fn(
        100e3, np.pi / 2, si("HS_A"), si("LS_A"), si("HS_B"), si("LS_B"), n, 100e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(lc, engine="trbdf2", switch_fn=fn)


# A 1e-15 rounding draw that went backward on macOS arm64 (a post-event sample
# 8e-16 s before the pending one at t = 1.81995 ms).
BACKWARD_DRAW = {
    ("Vbus", "V"): 99.99999999999997, ("L_leak", "L"): 9.999999999999993e-07,
    ("T1", "L_p"): 3.999999999999997e-05, ("T1", "L_s"): 1.0000000000000003e-05,
    ("T1", "k"): 0.9899999999999999, ("L_out", "L"): 0.0001000000000000001,
    ("C_out", "C"): 0.00010000000000000007, ("R_load", "R"): 10.000000000000007,
    ("R_iso", "R"): 0.0009999999999999992,
}


def test_psfb_backward_draw_records_increasing_times():
    t = np.asarray(_run_psfb(BACKWARD_DRAW).times)
    assert np.all(np.diff(t) >= 0)


def _monotone_or_grind(run):
    """The times of a run; a grind refusal (a separate, known PSFB defect: an
    unresolved 1e-15 s mode) is not this invariant's business."""
    try:
        r = run()
    except p.SimulationAborted as e:
        if "grinding" not in str(e):
            raise
        t = np.asarray(e.partial.times)
    else:
        t = np.asarray(r.times)
    d = np.diff(t)
    assert np.all(d >= 0), (int(np.argmax(d < 0)), d.min())


def test_psfb_times_increase_under_rounding_fuzz():
    # A seed whose three draws finish on macOS arm64 (a grind costs ~40 s).
    base = _psfb_defaults()
    rng = np.random.default_rng(20261120)
    for _ in range(3):
        vals = {k: v * (1 + rng.uniform(-1, 1) * 1e-13) for k, v in base.items()}
        _monotone_or_grind(lambda: _run_psfb(vals))


@pytest.mark.parametrize("form,mode", [("branch", None), ("conductance", None),
                                       ("branch", "lcp")])
def test_floating_bridge_times_increase_under_rounding_fuzz(form, mode):
    rng = np.random.default_rng(20261119)
    extra = {"mode_selection": mode} if mode else {}
    for _ in range(4):
        d = rng.uniform(-1, 1, 6) * 1e-13
        _monotone_or_grind(lambda: p.simulate(
            _bridge(d, form), t_end=40e-3, engine="trbdf2", rtol=1e-7,
            atol=1e-10, **extra))
