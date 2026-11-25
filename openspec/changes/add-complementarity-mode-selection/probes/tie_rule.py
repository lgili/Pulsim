"""Slice 2, task 1: why the three-phase C||R bridge goes wrong on the conductance form, and which
tie rule is right on every family. Prototype only (numpy), exact trajectories.

    PYTHONPATH=<worktree>/python:<worktree>/python/tests/.. python3 tie_rule.py > tie_rule.out

Each family runs in the prototype twice: with ideal elements (R_on = 0, g_off = 0: the island is a
gauge) and in the conductance form (R_on = 1e-3 and a 1e9 ohm leak made explicit, i.e. exactly
pulsim's resistor-pair devices). Each run uses one tie rule:

  off      : D5 as slice 1 shipped it -- a port still tied after widening is OFF
  basis    : a remaining tie keeps the LCP's own basis (ON iff i_d is basic, i.e. i > w)
  checked  : ties OFF if that mode is complementary on the same LCP data (OFF gaps >= -tol_w,
             ON currents >= -tol_i after the principal pivot); otherwise the LCP's own basis

'wrong' is the prototype's independent judge: the selected mode's own exact flow violates one of
its predicates (an ON diode's current or an OFF diode's gap goes negative) right after the event.
"""
import inspect
import sys
import time

import numpy as np

from tests.oracles import lcp_pulsim_bridge as B
from tests.oracles import lcp_reference as L

LAST = {}
_src = inspect.getsource(L._lcp_at).replace(
    "    return st, piv",
    "    LAST.update(i=i_ph, w=w_v, zs=zs, Hs=Hs, qv=qv, sel=list(sel), k=k, tol_w=tol_w, tol_i=tol_i)\n"
    "    return st, piv")
_ns = dict(vars(L))
_ns["LAST"] = LAST
exec(compile(_src, "_lcp_at_probe", "exec"), _ns)
LCP_AT = _ns["_lcp_at"]
RULE = {"name": "off"}


def select_modes_rule(c, y, t, sw_on, prev_on, h, widen=(100.0, 1e4)):
    """select_modes with the remaining-tie decision replaced by RULE (the 'off' rule is D5)."""
    k = len(c.D)
    if k == 0:
        return (), dict(pivots=0, pivots_total=0, status="no diodes", ties=0, widen=0)
    st, piv = LCP_AT(c, y, t, sw_on, prev_on, h)
    first_lcp = dict(LAST)                  # the LCP at h_e: the first-order answer
    own = [1 if first_lcp["i"][j] * first_lcp["zs"] > first_lcp["w"][j] else 0 for j in range(k)]
    first, pivots, nw = piv, piv, 0
    for f in widen:
        if -1 not in st:
            break
        st2, p2 = LCP_AT(c, y, t, sw_on, prev_on, h * f)
        pivots += p2
        nw += 1
        st = [s2 if s == -1 else s for s, s2 in zip(st, st2)]
    ties = sum(1 for s in st if s == -1)
    cand = [0 if s == -1 else s for s in st]
    if RULE["name"] == "basis" and nw:
        cand = own                         # the h_e LCP's own solution whenever it had a tie
    elif RULE["name"] == "checked" and nw and not complementary(first_lcp, cand):
        RULE["kept"] = RULE.get("kept", 0) + 1
        cand = own
    return tuple(s == 1 for s in cand), dict(pivots=first, pivots_total=pivots, status="solution",
                                             ties=ties, widen=nw)


def ppt(H, q, beta):
    """Principal pivot of w = H z + q exchanging z_beta and w_beta."""
    k = len(q)
    b = list(beta)
    a = [j for j in range(k) if j not in beta]
    H2, q2 = H.copy(), q.copy()
    if not b:
        return H2, q2
    Bi = np.linalg.inv(H[np.ix_(b, b)])
    H2[np.ix_(b, b)] = Bi
    H2[np.ix_(b, a)] = -Bi @ H[np.ix_(b, a)]
    H2[np.ix_(a, b)] = H[np.ix_(a, b)] @ Bi
    H2[np.ix_(a, a)] = H[np.ix_(a, a)] - H[np.ix_(a, b)] @ Bi @ H[np.ix_(b, a)]
    q2[b] = -Bi @ q[b]
    q2[a] = q[a] - H[np.ix_(a, b)] @ Bi @ q[b]
    return H2, q2


def solve_fixed(lcp, st):
    """The LCP of `lcp` with every decided port (st >= 0) fixed in its mode, solved on the ties alone.
    Returns (st_ties, i, w, coordinate flags, H in those coordinates) -- st_ties: 1/0/-1 per port."""
    k, sel, zs = lcp["k"], lcp["sel"], lcp["zs"]
    is_i = [sel[j] < k for j in range(k)]            # coordinate z_j is i_j (else w_j)
    want = [is_i[j] if st[j] < 0 else (st[j] == 0) for j in range(k)]   # OFF: z = i = 0; ON: z = w = 0
    beta = [j for j in range(k) if want[j] != is_i[j]]
    H, q = ppt(lcp["Hs"], lcp["qv"], beta)
    T = [j for j in range(k) if st[j] < 0]
    z = np.zeros(k)
    if T:
        zt, wt, _, status = L.lemke(H[np.ix_(T, T)], q[T])
        if zt is None:
            raise L.ModeError("widened LCP: " + status)
        z[T] = zt
    w_ = H @ z + q
    i_ph = np.array([(z[j] if want[j] else w_[j]) / zs for j in range(k)])
    w_v = np.array([w_[j] if want[j] else z[j] for j in range(k)])
    # a decided port whose complement this LCP violates was not decided robustly (two diodes on one
    # high-impedance island split a tiny gap differently at each step): free it and solve again
    bad = [j for j in range(k) if (st[j] == 0 and w_v[j] < -lcp["tol_w"]) or (st[j] == 1 and i_ph[j] < -lcp["tol_i"])]
    if bad:
        RULE["unfixed"] = RULE.get("unfixed", 0) + len(bad)
        return solve_fixed(lcp, [-1 if j in bad else s for j, s in enumerate(st)])
    out = []
    for j in range(k):
        if st[j] >= 0:
            out.append(st[j])
        elif w_v[j] > lcp["tol_w"]:
            out.append(0)
        elif i_ph[j] > lcp["tol_i"]:
            out.append(1)
        else:
            out.append(-1)
    return out, i_ph, w_v, want, H


def resistance(H, want, i_ph, j, zs, st):
    """dw_j/di_j (ohm) with the other ports in the solution's mode (H in coordinates `want`)."""
    k = len(want)
    on = [False if m == j else (st[m] == 1 if st[m] >= 0 else i_ph[m] > 0) for m in range(k)]
    target = [not o for o in on]                     # z_m = i_m for OFF ports (and j)
    beta = [m for m in range(k) if target[m] != want[m]]
    H2, _ = ppt(H, np.zeros(k), beta)
    return H2[j, j] * zs                             # scaled (V per V-scaled current) -> ohm


def judge(c, y, sw, mode, h):
    """Prototype exact-flow check of `mode` at 100 h and 1e4 h (its own predicates >= -gscale)."""
    try:
        m = L.Mode(c, sw, mode)
    except L.ModeError:
        return 0
    if m.P.shape[0] == 0:
        return 1
    y0, _ = L.project(c, m, y)
    gscale = 1e-9 * max(1.0, np.abs(m.P).max() * np.abs(y0).max())
    for tau in (100 * h, 1e4 * h):
        if (m.P @ m.flow(y0, tau)).min() < -gscale:
            return 0
    return 1


def select_slice2(c, y, t, sw_on, prev_on, h, widen=(100.0, 1e4)):
    """Slice-2 D5: widening with decided ports fixed; a remaining tie that pins an island is ON when
    the mode with the ties OFF fails the exact-flow judge."""
    k = len(c.D)
    if k == 0:
        return (), dict(pivots=0, pivots_total=0, status="no diodes", ties=0, widen=0)
    st, piv = LCP_AT(c, y, t, sw_on, prev_on, h)
    lcp0 = dict(LAST)
    i_ph, w_v, want = lcp0["i"], lcp0["w"], [s < k for s in lcp0["sel"]]
    first, pivots, nw = piv, piv, 0
    last = (st, i_ph, w_v, want, ppt(lcp0["Hs"], lcp0["qv"], [])[0])
    for f in widen:
        if -1 not in st:
            break
        LCP_AT(c, y, t, sw_on, prev_on, h * f)         # poses the widened LCP (its own answer unused)
        nw += 1
        try:
            st, i_ph, w_v, want, H = solve_fixed(dict(LAST), st)
        except (L.ModeError, np.linalg.LinAlgError):      # as the kernel: keep the earlier decision
            RULE["failed_widenings"] = RULE.get("failed_widenings", 0) + 1
            break
        last = (st, i_ph, w_v, want, H)
    st, i_ph, w_v, want, H = last
    ties = [j for j in range(k) if st[j] < 0]
    mode = [1 if s == 1 else 0 for s in st]
    if ties:
        pins = [j for j in ties if i_ph[j] > 0
                and i_ph[j] * resistance(H, want, i_ph, j, LAST["zs"], st) > LAST["tol_w"]]
        if pins and not judge(c, y, sw_on, tuple(bool(m_) for m_ in mode), h):
            for j in pins:
                mode[j] = 1
            RULE["kept"] = RULE.get("kept", 0) + 1
    return tuple(m_ == 1 for m_ in mode), dict(pivots=first, pivots_total=pivots, status="solution",
                                               ties=sum(1 for s in st if s < 0), widen=nw)


def complementary(lcp, mode, tol=1e-9):
    """Is `mode` (1 ON) a complementary solution of the LCP `lcp` (w = Hs z + qv, z = coords[sel])?"""
    k, sel, Hs, qv = lcp["k"], lcp["sel"], lcp["Hs"], lcp["qv"]
    # z_j is i_j when sel[j] < k; it is basic (nonzero allowed) iff that coordinate is the mode's free one
    a = [j for j in range(k) if (sel[j] < k) == bool(mode[j])]
    z = np.zeros(k)
    if a:
        try:
            z[a] = np.linalg.solve(Hs[np.ix_(a, a)], -qv[a])
        except np.linalg.LinAlgError:
            return False
    w = Hs @ z + qv
    sc = max(1.0, np.abs(qv).max())
    return z.min() >= -tol * sc and w.min() >= -tol * sc


def run(fam, conductance, rule):
    RULE.clear()
    RULE["name"] = rule
    c = fam.proto(conductance=conductance)
    L.select_modes = select_slice2 if rule == "slice2" else select_modes_rule
    try:
        r = fam.run(c)
    finally:
        L.select_modes = ORIG_SELECT
    return r


ORIG_SELECT = L.select_modes


def main():
    print(f"{'family':18s} {'form':11s} {'rule':8s} {'events':>6s} {'ties':>4s} {'wrong':>5s} "
          f"{'immed':>5s} {'maxpiv':>6s} {'pins':>4s} {'unfx':>4s} {'fail':>4s}  ON-set sizes")
    for name, fam in B.FAMILIES.items():
        for cond in (False, True):
            for rule in ("off", "slice2"):
                t0 = time.time()
                try:
                    r = run(fam, cond, rule)
                except Exception as e:  # noqa: BLE001
                    print(f"{name:18s} {'conductance' if cond else 'ideal':11s} {rule:8s} FAILS "
                          f"{type(e).__name__}: {str(e)[:90]}")
                    continue
                ev = [e for e in r.events if e[1] == "diode"]
                sizes = sorted({sum(e[3]) for e in ev})
                print(f"{name:18s} {'conductance' if cond else 'ideal':11s} {rule:8s} {len(r.events):6d} "
                      f"{r.ties:4d} {r.wrong_mode:5d} {r.immediate:5d} {max(r.pivots):6d} {RULE.get('kept', 0):4d} {RULE.get('unfixed', 0):4d} {RULE.get('failed_widenings', 0):4d}  {sizes}"
                      f"  [{time.time() - t0:.1f} s]")
                sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
