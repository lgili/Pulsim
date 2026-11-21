"""The floating-bus diode bridge against its exact reference (benchmarks/BRIDGE_GROUND_TRUTH.md,
benchmarks/tools/bridge_ground_truth.py: mpmath, 50 digits, every commutation bisected to 1e-38 s,
the 30-40 ms bus average 311.343881428885841697 V, unique mode after each of 11 commutations).

The exact engine follows the reference's event list: the same mode sequence, and every
commutation within 1e-10 s of its exact instant. The float64 resolution limits the reference
names (D3's turn-on readable only 1e-10 s after it, its turn-off 1e-12 s) bound the times.
The bus average then agrees to 4.5e-10 V (1.4e-12 relative), on the default selection and on
'lcp' alike. It is averaged over exactly [30, 40] ms: the first stored sample falls 0.88 us
into the window, and the trapezoid over the samples alone read 2.0e-4 V high."""
import warnings

import numpy as np
import pytest

import pulsim as p
from tests.test_trbdf2_rounding_fuzz import _bridge

TRUTH_BUS = 311.343881428885841697
# (t [s], mode D1..D4 after it): the reference's 11 commutations (BRIDGE_GROUND_TRUTH.md)
TRUTH_EVENTS = [
    (0.004453708135472069740740, (0, 0, 0, 0)),
    (0.010005784513217728980424, (0, 0, 1, 0)),
    (0.013234043039166346336001, (0, 1, 1, 0)),
    (0.016037029154868641651128, (0, 0, 1, 0)),
    (0.019993816884401272433491, (0, 0, 0, 0)),
    (0.023578074673711604264767, (1, 0, 0, 1)),
    (0.026164880107693963425638, (0, 0, 0, 0)),
    (0.030006040329416432127840, (0, 0, 1, 0)),
    (0.033448621701173689951408, (0, 1, 1, 0)),
    (0.036121739554460249850922, (0, 0, 1, 0)),
    (0.039993904164045823079728, (0, 0, 0, 0)),
]


def _bus_average(r, a=0.03, b=0.04):
    t = np.asarray(r.times)
    v = np.asarray(r.v("dp")) - np.asarray(r.v("dn"))
    m = (t > a) & (t < b)
    tt = np.r_[a, t[m], b]
    vv = np.r_[np.interp(a, t, v), v[m], np.interp(b, t, v)]
    return np.trapezoid(vv, tt) / (b - a)


@pytest.mark.parametrize("mode", [None, "lcp"])
def test_exact_engine_matches_the_bridge_ground_truth(mode):
    kw = {} if mode is None else {"mode_selection": mode}
    b = _bridge([0.0] * 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 0.04, engine="exact", output_rtol=0, **kw)  # release: the subject is the engine, not the default output rule's samples
    s = np.asarray(r.switch_states).astype(int)[:, [b.switch_index_of(f"D{k}") for k in range(1, 5)]]
    assert tuple(s[0]) == (1, 0, 0, 1)
    changes = np.flatnonzero(np.any(s[1:] != s[:-1], axis=1)) + 1
    assert [tuple(s[i]) for i in changes] == [m for _, m in TRUTH_EVENTS]
    # each change sits in the first sample after its event: the event itself is in event_times
    et = np.asarray(r.event_times)
    for t_true, _ in TRUTH_EVENTS:
        assert np.min(np.abs(et - t_true)) < 1e-10, t_true     # measured <= 7.5e-12 s
    assert _bus_average(r) == pytest.approx(TRUTH_BUS, abs=1e-7)   # measured -4.5e-10 V
