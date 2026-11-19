# Local identity gate (this machine only, like bit_identity.py): ten paralleled bridges MUST run on
# engine='exact' over 20 ms, default selection and mode_selection='lcp', bus within 1e-3 V of
# pwl+split (dt 1e-7). The CI test (test_exact_route.py::test_ten_paralleled_bridges_agree_with_pwl)
# also accepts the law check's named refusal until a diode leaving at its own located zero is decided
# by one rule (DECISIONS session 9); flip the CI test back to this strict form when that lands.
# usage: PYTHONPATH=python:. python3 ten_banks_strict.py   (from the repo root; exit 0 = pass)
import sys
import warnings

import numpy as np

sys.path.insert(0, "python")
from tests.test_exact_route import _paralleled_bridges, _ten_bridges_pwl  # noqa: E402

import pulsim as p  # noqa: E402

warnings.simplefilter("ignore")
tg = np.linspace(0.0, 0.02, 2001)
ref = np.interp(tg, *_ten_bridges_pwl())
bad = 0
for kw in ({}, {"mode_selection": "lcp"}):
    try:
        r = p.simulate(_paralleled_bridges(10), t_end=0.02, engine="exact", **kw)
        err = np.abs(np.interp(tg, r.times, np.asarray(r.v("p")) - np.asarray(r.v("n"))) - ref).max()
        ok = err <= 1e-3
        print(kw or "default", f"bus vs pwl+split {err:.3e} V", "PASS" if ok else "FAIL")
    except Exception as e:  # the strict gate: a refusal fails it
        ok = False
        print(kw or "default", "FAIL", type(e).__name__, str(e)[:160])
    bad += not ok
sys.exit(1 if bad else 0)
