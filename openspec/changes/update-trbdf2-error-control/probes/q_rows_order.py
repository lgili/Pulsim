"""TR-BDF2 error vs rtol on every stateful kind, against its truth.

    PYTHONPATH=<worktree>/python python q_rows_order.py [--integrator esdirk43] [--rtols ...]

One line per (case, rtol): error against the truth and accepted steps. The
truths are those of python/tests/test_esdirk43_stateful_kinds.py and
test_esdirk43_engine.py (closed forms; SciPy DOP853 at rtol 1e-13; the
pinned SciPy induction-machine run), plus the IGBT tail energy and the
Lauritzen recovery charge in closed form. atol = rtol * 1e-3 throughout.
"""
import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
# an explicit PYTHONPATH (another build of pulsim) wins; the tests come from this tree
sys.path.insert(len(sys.path) if os.environ.get("PYTHONPATH") else 0, str(ROOT / "python"))
import pulsim as p  # noqa: E402
from tests import test_esdirk43_engine as E  # noqa: E402
from tests import test_esdirk43_stateful_kinds as K  # noqa: E402
from tests import test_induction_machine_mna as IM  # noqa: E402


def laur_wave(r):
    return K._laur_err(r)


def laur_qrr(r):
    # recovered charge over the 5 us after the edge, closed form of the
    # reverse current -(q0/TM)exp(-A t) - tau IS/TM
    t = np.asarray(r.times)
    i = np.asarray(r.v("in")) - np.asarray(r.v("a"))
    m = t > K.TD
    tp = np.concatenate([[0.0], t[m] - K.TD])
    ip = np.concatenate([[np.nan], i[m]])
    q0, A, T = K._laur_q0(), 1 / K.TM + 1 / K.TAU, tp[-1]
    cf = -(q0 / K.TM) * (1 - math.exp(-A * T)) / A - K.TAU * K.IS / K.TM * T
    ip[0] = -(q0 / K.TM) - K.TAU * K.IS / K.TM   # the right limit at the edge
    return abs(np.trapezoid(ip, tp) - cf) / abs(cf)


def igbt_wave(r):
    return K._igbt_err(r)


def igbt_energy(r):
    # tail energy in the IGBT over 10 us after the gate edge:
    # i = k I0 e^{-t/tau}, v = 100 - 10 i
    t = np.asarray(r.times)
    i = (100.0 - np.asarray(r.v("a"))) / 10.0
    i0 = i[np.searchsorted(t, K.TDI) - 1]
    m = t > K.TDI
    tp = np.concatenate([[0.0], t[m] - K.TDI])
    pw = np.concatenate([[np.nan], ((100.0 - 10.0 * i) * i)[m]])
    a = K.KT * i0
    pw[0] = (100.0 - 10.0 * a) * a
    T = tp[-1]
    cf = 100.0 * a * K.TT * (1 - math.exp(-T / K.TT)) - 10.0 * a * a * K.TT / 2 * (
        1 - math.exp(-2 * T / K.TT))
    return abs(np.trapezoid(pw, tp) - cf) / abs(cf)


CASES = {
    "lauritzen_wave": (K._laur, K.TD + 5e-6, dict(dt=1e-6), laur_wave),
    "lauritzen_qrr": (K._laur, K.TD + 5e-6, dict(dt=1e-6), laur_qrr),
    "igbt_wave": (K._igbt, K.TDI + 10e-6, dict(dt=2e-6), igbt_wave),
    "igbt_energy": (K._igbt, K.TDI + 10e-6, dict(dt=2e-6), igbt_energy),
    "pmsm_locked": (K._pmsm_locked, K.TD + 5e-3, {}, K._pmsm_err),
    "im_dol_w": (lambda: IM._machine(p.CircuitBuilder(), J=IM.J_DOL), IM.T_REF[-1],
                 dict(dt=2e-5),
                 lambda r: IM._dol_error(np.asarray(r.times), r.v("w"), r.i("M_a"))[0]),
    "sat_lc_end": (E._sat_lc, E.TD + 4 * E.T0, {}, E._sat_end_err),
    "nlcap_rc": (E._cap_rc, E.TDC + 3e-6, {}, E._cap_err),
}

PMSM_FREE_REF = {}


def pmsm_free(r):
    if not PMSM_FREE_REF:
        ref = p.simulate(K._pmsm_free(), 0.1, engine="trbdf2", integrator="esdirk43",
                         rtol=1e-11, atol=1e-14)
        PMSM_FREE_REF["w"] = ref.v("w")[-1]
    return abs(r.v("w")[-1] - PMSM_FREE_REF["w"]) / abs(PMSM_FREE_REF["w"])


CASES["pmsm_free_w"] = (K._pmsm_free, 0.1, {}, pmsm_free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrator", default=None)
    ap.add_argument("--rtols", default="1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10")
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    kw0 = dict(engine="trbdf2")
    if a.integrator:
        kw0["integrator"] = a.integrator
    for name in a.cases.split(","):
        build, t_end, kw, err = CASES[name]
        rows = []
        for rt in (float(x) for x in a.rtols.split(",")):
            try:
                r = p.simulate(build(), t_end, rtol=rt, atol=rt * 1e-3, **kw0, **kw)
                e, n = err(r), r._trbdf2_stats["n_accept"]
            except Exception as ex:  # noqa: BLE001 - recorded
                e, n = float("nan"), f"{type(ex).__name__}"
            rows.append((rt, e, n))
            print(f"{name:16s} rtol {rt:.0e}  err {e:.3e}  accepted {n}", flush=True)
        ok = [(rt, e) for rt, e, _ in rows if np.isfinite(e) and e > 0]
        if len(ok) >= 2:
            s = np.polyfit(np.log10([x for x, _ in ok]), np.log10([y for _, y in ok]), 1)[0]
            print(f"{name:16s} slope d log(err) / d log(rtol) = {s:.2f}", flush=True)


if __name__ == "__main__":
    main()
