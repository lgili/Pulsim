"""Slice 3 (DECISIONS #19): every dsed trajectory of the mode-selection surveys (shadow_survey +
extra_circuits; settle / default guard / 'lcp'), dumped from one build and compared bit for bit
(np.array_equal on times and states) against another.

    PYTHONPATH=<old build>/python python3 survey_bit_identity.py save OLD.npz
    PYTHONPATH=<new build>/python python3 survey_bit_identity.py check OLD.npz > survey_bit_identity.out
"""
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
import pulsim as p  # noqa: E402
import pulsim._dsed_dispatch as DD  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extra_circuits as E  # noqa: E402
import shadow_survey as S  # noqa: E402


def dump():
    out = {}
    cases = list(S.CASES) + [("X:" + lab, (lambda mk: (lambda: mk()[:3]))(m)) for lab, m in E.CASES]
    for label, make in cases:
        for tag, mode, guard in (("settle", None, False), ("default", None, True), ("lcp", "lcp", True)):
            DD.MODE_GUARD = guard
            try:
                b, fn, t_end = make()
                kw = {} if fn is None else {"switch_fn": fn}
                if mode:
                    kw["mode_selection"] = mode
                r = p.simulate(b, t_end=t_end, engine="dsed", **kw)
                out[f"{label}|{tag}|t"] = np.asarray(r.times)
                out[f"{label}|{tag}|x"] = np.asarray(r.states)
                out[f"{label}|{tag}|ev"] = np.asarray(r.event_times)
            except Exception as e:  # noqa: BLE001
                out[f"{label}|{tag}|err"] = np.array(f"{type(e).__name__}: {str(e)[:200]}")
            finally:
                DD.MODE_GUARD = True
    return out


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    print("pulsim from", p.__file__)
    new = dump()
    if cmd == "save":
        np.savez(path, **new)
        print("saved", len(new), "arrays")
        return 0
    old = np.load(path)
    keys = sorted({k.rsplit("|", 1)[0] for k in list(old.files) + list(new)})
    same = changed = 0
    for k in keys:
        def g(z, s, k=k):
            key = f"{k}|{s}"
            return (z[key] if key in z else None) if isinstance(z, dict) else (z[key] if key in z.files else None)
        if g(old, "err") is not None or g(new, "err") is not None:
            print("ERROR  ", k, g(old, "err"), "->", g(new, "err"))
            changed += 1
        elif np.array_equal(g(old, "t"), g(new, "t")) and np.array_equal(g(old, "x"), g(new, "x")):
            same += 1
        else:
            changed += 1
            print("CHANGED", k, "events", len(g(old, "ev")), "->", len(g(new, "ev")),
                  "end state", g(old, "x")[-1][:3], "->", g(new, "x")[-1][:3])
    print(f"bit-identical {same}, changed {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
