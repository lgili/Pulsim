"""Task 3.2 / 3.3: the band of h_e/T_ref over which the C++ port model + selector (conductance
form, g_on = 1e3, g_off = 1e-9) returns the prototype's mode at every prototype event, and the
tie statistics at h_e = 1e-6 T_ref (the design's estimate) and 1e-7 T_ref (the default taken).

    PYTHONPATH=<worktree>/python python3 h_e_band.py      (needs the built extension; ~1 min)
"""
import sys
import warnings

warnings.simplefilter("ignore")
from tests.oracles import lcp_pulsim_bridge as B  # noqa: E402

SCALES = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def main():
    cache = {}
    print("mismatches against the prototype's mode (events) per h_e/T_ref; 'x' = an error")
    print(f"{'family':16s} {'events':>6s} " + " ".join(f"{s:>7.0e}" for s in SCALES))
    for name, fam in B.FAMILIES.items():
        c, rec = B.prototype_selections(fam)
        P = B.PulsimPorts(fam)
        cache[name] = (rec, P)
        row = []
        for sc in SCALES:
            h = sc * B.T_REF[name]
            bad = 0
            try:
                for t, y, sw, prev, sel in rec:
                    if tuple(P.select(t, y, sw, prev, h)["mode"]) != sel:
                        bad += 1
                row.append(f"{bad:7d}")
            except Exception:
                row.append(f"{'x':>7s}")
        print(f"{name:16s} {len(rec):6d} " + " ".join(row))
    print()
    for rel in (1e-6, 1e-7):
        tie_table(cache, rel)


def tie_table(cache, rel):
    print(f"h_e = {rel:.0e} T_ref: first-pass ties, their currents, widenings, ties left (OFF),")
    print("and ties decided differently with tol_w 10x smaller (tol = 1e-10)")
    for name, (rec, P) in cache.items():
        h = rel * B.T_REF[name]
        ties = wid = left = diff = 0
        cur = []
        for t, y, sw, prev, sel in rec:
            r = P.select(t, y, sw, prev, h)
            r2 = P.select(t, y, sw, prev, h, tol=1e-10)
            ties += r["first_pass_ties"]
            cur += list(r["tie_currents"])
            wid += r["widenings"]
            left += r["ties"]
            if r["first_pass_ties"] and tuple(r["mode"]) != tuple(r2["mode"]):
                diff += 1
        cmax = f"{max(cur):.2e} A" if cur else "-"
        print(f"{name:16s} ties={ties:3d} max|i_tie|={cmax:>11s} widenings={wid:3d} left={left:3d} "
              f"decided differently at tol/10: {diff}")
    print()


if __name__ == "__main__":
    sys.exit(main())
