"""DECISIONS #30: the all-OFF gaps of a series-pair turn-off, posed in the all-OFF mask vs pivoted
from the previous mode's port matrix (double, and the same pivot in mpmath from the double H).
    PYTHONPATH=python python3 pair_split.py bridge1_vf0 0.005101
"""
import sys
import warnings

import mpmath as mp
import numpy as np
warnings.simplefilter('ignore'); sys.path.insert(0, 'python')
from tests.oracles import lcp_pulsim_bridge as B
mp.mp.dps = 50
name = sys.argv[1]; t0 = float(sys.argv[2])
fam = B.FAMILIES[name]; _, rec = B.prototype_selections(fam); P = B.PulsimPorts(fam); h_e = 1e-7 * B.T_REF[name]
for t, y, sw, prev, sel in rec:
    if abs(t - t0) > 5e-5: continue
    for fac in (1, 100, 1e4):
        h = h_e * fac
        off = P.select(t, y, sw, (False,) * 4, h)       # posed in the all-OFF mask: u = the OFF gaps
        r = P.select(t, y, sw, prev, h)                  # posed in the previous mode
        H, q = np.array(r['H']), np.array(r['q'])
        b = [j for j in range(4) if prev[j]]             # ON ports pivoted to OFF
        a = [j for j in range(4) if not prev[j]]
        Hbb = H[np.ix_(b, b)]
        # OFF coords for b: gap_b = -Hbb^-1 q_b (i_b = 0 there)
        g_dbl = -np.linalg.solve(Hbb, q[b])
        Hm = mp.matrix(Hbb.tolist()); g_mp = -(Hm ** -1) * mp.matrix(q[b].tolist())
        print(f'x{fac:g}: OFF gaps direct (all-OFF mask) {np.array(off["u"])[b]}  hybrid pivot dbl {g_dbl}  pivot exact-on-rounded {[float(v) for v in g_mp]}  cond(Hbb) {np.linalg.cond(Hbb):.2e}')
