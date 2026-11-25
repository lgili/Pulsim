"""add-complementarity-mode-selection: the LCP mode selection on five circuit families.

Run: nice -n 15 python3 lcp_cases.py        (numpy, scipy, mpmath; about one minute)

Every diode and switch is ideal (R_on = 0, g_off = 0) unless stated. At every event ONE LCP decides
the diode modes (lcp_core.select_modes); the run counts
  pivots         Lemke pivots per event (0 = the previous mode is still consistent)
  ties           diodes still at i = w = 0 after widening (decided OFF; both modes consistent)
  widened        LCP re-solves at 100 h_e / 1e4 h_e to decide a tie (the next term of the RCP)
  wrong          a diode event after which the LCP kept the mode whose predicate was violated
  immediate      a newly selected mode violated at its first instant (would need a re-selection)
  rcp            disagreements between the LCP at h_e and at h_e/10 (the rational-complementarity check)
"wrong", "immediate" and a zero-time loop are the failure signatures of a mode selector; all must be 0.
Oracles: 40-digit orbits (DCM buck: ../../add-branch-form-ideal-switch/probes/ideal_dcm_buck.py),
closed-form commutation angles (bridges), mpmath orbit of the CCM boost.

Today's kernel on the same circuits (resistor-pair diodes, value law, 64-round settle, chatter
guard) is measured by today_dsed.py.
"""
import time
import numpy as np
import mpmath as mp
from lcp_core import Circuit, simulate, pwm, pwm_edges

mp.mp.dps = 30
np.set_printoptions(linewidth=140)


def stats(r):
    p = np.array(r.pivots)
    pt = np.array(r.pivots_total)
    hist = {int(v): int(np.sum(p == v)) for v in np.unique(p)}
    return (f"events={len(r.events)} LCPs={len(p)} pivots/event max={p.max()} mean={p.mean():.2f} hist={hist} "
            f"(with tie re-solves max={pt.max()} mean={pt.mean():.2f}) "
            f"ties={r.ties} widened={r.widen} wrong={r.wrong_mode} immediate={r.immediate} rcp={r.rcp_disagree} "
            f"impulses={len(r.impulses)}")


# ---------------------------------------------------------------------------------------------
print("=" * 100)
print("(i) at rest, V_f = 0 (the V_th = 0 diode of dsed's 'did not settle after 64 rounds')")
T = 1e-5


def buck(R=50.0, L=100e-6, Vf=0.0):
    c = Circuit()
    c.add_Vdc("Vin", "vin", "0", 24.0); c.add_S("S", "vin", "sw"); c.add_D("D1", "0", "sw", V_f=Vf)
    c.add_L("L", "sw", "vout", L); c.add_C("C", "vout", "0", 47e-6); c.add_R("vout", "0", R)
    return c


c = buck()
r = simulate(c, 1e-6, gate=lambda t: (False,))
print("   buck at rest, gate off, 1 us :", stats(r), "| final mode", r.mode.key, "state", r.y[:2])
r = simulate(c, 2e-6 + 40 * T, gate=pwm(T, 0.3, t0=2e-6), edges=pwm_edges(T, 0.3, 2e-6 + 40 * T, t0=2e-6),
             h_e=1e-6 * T)
print(f"   gate held off to 2 us, then 40 T of PWM: v_C = {r.y[0]:.9f} V;", stats(r))

# ---------------------------------------------------------------------------------------------
print("=" * 100)
print("(ii) single-phase full bridge, ideal diodes, C filter (100 V 50 Hz, 1000 uF || 100 ohm), 10 cycles")


def bridge1(Vf, C=1e-3, R=100.0):
    c = Circuit()
    c.add_Vsin("Vs", "a", "0", 100.0, 50.0)
    c.add_D("D1", "a", "p", V_f=Vf); c.add_D("D3", "0", "p", V_f=Vf)
    c.add_D("D2", "n", "a", V_f=Vf); c.add_D("D4", "n", "0", V_f=Vf)
    c.add_C("C", "p", "n", C); c.add_R("p", "n", R)
    return c


def rect_ref(A, P, w, R, C, Vf):
    """Periodic C-filtered rectifier: conduction psi in [psi_on, psi_off] on A sin(psi) - 2 Vf,
    decay until the next pulse, repetition angle P. Returns (psi_on, psi_off) in radians."""
    A, R, C, Vf, w, P = map(mp.mpf, (A, R, C, Vf, w, P))
    poff = mp.findroot(lambda s: C * A * w * mp.cos(s) + (A * mp.sin(s) - 2 * Vf) / R, mp.pi * 0.55)
    voff = A * mp.sin(poff) - 2 * Vf
    pon = mp.findroot(lambda s: A * mp.sin(s) - 2 * Vf - voff * mp.exp(-(s + P - poff) / (w * R * C)), mp.pi / 2 - 0.3)
    return pon, poff


w = 2 * np.pi * 50
pon, poff = rect_ref(100, np.pi, w, 100, 1e-3, 0.7)
print(f"   reference (mpmath): conduction from {float(pon) * 180 / np.pi:.9f} deg to {float(poff) * 180 / np.pi:.9f} deg of each half cycle")
for Vf in (0.7, 0.0):
    c = bridge1(Vf)
    t1 = time.time()
    r = simulate(c, 0.2, h_e=1e-9)
    el = time.time() - t1
    ev = [e for e in r.events if e[1] == "diode"]
    print(f"   V_f = {Vf}: {stats(r)}  [{el:.1f} s]")
    if Vf == 0.7:
        # after the first turn-off every half cycle is the reference orbit
        errs = []
        first_off = next(i for i, e in enumerate(ev) if sum(e[3]) == 0)
        for e in ev[first_off:]:
            ang = (w * e[0]) % np.pi
            ref = float(poff) if sum(e[3]) == 0 else float(pon)
            errs.append(abs(ang - ref) / w)
        print(f"      impulses (t, J): {[(t_, f'{dw:.1e}') for t_, dw in r.impulses]}")
        print(f"      max |t_event - t_ref| over {len(errs)} events after the first turn-off: {max(errs):.2e} s;"
              f" modes seen {sorted(set(e[3] for e in ev))}")
# emulated value law at the at-rest start of the V_f = 0 bridge and at a zero crossing with all diodes off
c = bridge1(0.0)

# ---------------------------------------------------------------------------------------------
print("=" * 100)
print("(iii) DCM buck (24 V, 100 uH, 47 uF, 50 ohm, D = 0.3, 100 kHz), ideal S and D, V_f = 0, 4000 periods from rest")
VC_REF, TAU_REF = 8.9963970815477, 0.79983739563545
c = buck()
N = 4000
t1 = time.time()
r = simulate(c, N * T, gate=pwm(T, 0.3), edges=pwm_edges(T, 0.3, N * T), h_e=1e-6 * T,
             sample_times=[(N - 1) * T], confirm=True)
el = time.time() - t1
vC = r.samples[-1][1][0]
off = [e[0] for e in r.events if e[1] == "diode" and e[0] > (N - 1) * T]
print(f"   {stats(r)}  [{el:.1f} s]")
print(f"   v_C at the rising edge of period {N - 1}: {vC:.13f} V  (40-digit orbit {VC_REF}; error {vC - VC_REF:+.1e})")
print(f"   diode turn-off tau/T: {(off[0] - (N - 1) * T) / T:.14f} (orbit {TAU_REF}; error {((off[0] - (N - 1) * T) / T - TAU_REF):+.1e})")
print(f"   idle-interval inductor current at the end: {r.y[1]:+.1e} A; masks visited: {sorted(r.cache.keys())}")
print("   L*(1 +- d), 40 periods each (dsed aborted 'is chattering' for three of these six with V_th = 1e-6):")
vend = {}
for d in (1e-5, 1e-4, 1e-2):
    for s in (+1, -1):
        cc = buck(L=100e-6 * (1 + s * d))
        rr = simulate(cc, 40 * T, gate=pwm(T, 0.3), edges=pwm_edges(T, 0.3, 40 * T), h_e=1e-6 * T)
        vend[(d, s)] = rr.y[0]
        print(f"      d = {s * d:+.0e}: v_C(40T) = {rr.y[0]:.12f} V  wrong={rr.wrong_mode} immediate={rr.immediate} max pivots={max(rr.pivots)}")
fd = (vend[(1e-5, 1)] - vend[(1e-5, -1)]) / (2e-5 * 100e-6)
print(f"      central difference d v_C(40T) / dL at d = 1e-5: {fd:.6f} V/H")

# ---------------------------------------------------------------------------------------------
print("=" * 100)
print("(iv) boost (12 V, 100 uH, 47 uF, D = 0.5, 100 kHz), ideal S (sw-gnd) and D (sw-out)")


def boost(R, Vf=0.0):
    c = Circuit()
    c.add_Vdc("Vin", "vin", "0", 12.0); c.add_L("L", "vin", "sw", 100e-6)
    c.add_S("S", "sw", "0"); c.add_D("D1", "sw", "out", V_f=Vf)
    c.add_C("C", "out", "0", 47e-6); c.add_R("out", "0", R)
    return c


def boost_ccm_orbit(R):
    Vin, L, C, D, Tm, Rm = map(mp.mpf, ("12", "100e-6", "47e-6", "0.5", "1e-5", str(R)))
    A1 = mp.matrix([[-1 / (Rm * C), 0, 0], [0, 0, Vin / L], [0, 0, 0]])      # (v_C, i_L, 1), S on
    A2 = mp.matrix([[-1 / (Rm * C), 1 / C, 0], [-1 / L, 0, Vin / L], [0, 0, 0]])
    M = mp.expm(A2 * (1 - D) * Tm) * mp.expm(A1 * D * Tm)
    # fixed point of the affine map on (v, i)
    Mx = mp.matrix([[M[0, 0], M[0, 1]], [M[1, 0], M[1, 1]]])
    g = mp.matrix([M[0, 2], M[1, 2]])
    x = mp.lu_solve(mp.eye(2) - Mx, g)
    return x


xo = boost_ccm_orbit(50)
for R, N in ((50.0, 20000), (500.0, 4000)):
    c = boost(R)
    t1 = time.time()
    r = simulate(c, N * T, gate=pwm(T, 0.5), edges=pwm_edges(T, 0.5, N * T), h_e=1e-6 * T,
                 sample_times=[(N - 1) * T])
    el = time.time() - t1
    yv = r.samples[-1][1]
    dio = [e for e in r.events if e[1] == "diode"]
    print(f"   R = {R:g} ohm: {stats(r)} diode-located events={len(dio)}  [{el:.1f} s]")
    print(f"      state at the rising edge of period {N - 1}: v_C = {yv[0]:.12f} V, i_L = {yv[1]:.12f} A")
    if R == 50.0:
        print(f"      CCM orbit (mpmath): v_C = {mp.nstr(xo[0], 13)} V, i_L = {mp.nstr(xo[1], 13)} A;"
              f" errors {yv[0] - float(xo[0]):+.1e} V, {yv[1] - float(xo[1]):+.1e} A")

# ---------------------------------------------------------------------------------------------
print("=" * 100)
print("(v-a) three-phase diode bridge, L_s = 1 mH per phase, ideal DC current load 10 A, V_f = 0 (commutation overlap)")
Vm, f, Ls, Id = 100.0, 50.0, 1e-3, 10.0


def bridge3(Ls=None, Id=None, C=None, R=None, Vf=0.0):
    c = Circuit()
    for ph, ang in (("a", 0.0), ("b", -2 * np.pi / 3), ("c", 2 * np.pi / 3)):
        src = "s" + ph if Ls else ph
        c.add_Vsin("V" + ph, src, "0", Vm, f, phase=ang)
        c.add_D("Dt" + ph, ph, "p", V_f=Vf); c.add_D("Db" + ph, "n", ph, V_f=Vf)
    if Ls:
        i0 = {"a": 0.0, "b": -Id, "c": Id}
        for ph in "abc":
            c.add_L("L" + ph, "s" + ph, ph, Ls, i0=i0[ph])
    if Id:
        c.add_Idc("Id", "p", "n", Id)
    if C:
        c.add_C("C", "p", "n", C); c.add_R("p", "n", R)
    return c


c = bridge3(Ls=Ls, Id=Id)
t1 = time.time()
r = simulate(c, 0.06, h_e=1e-9)
el = time.time() - t1
mu = np.arccos(1 - 2 * w * Ls * Id / (np.sqrt(3) * Vm))
print(f"   {stats(r)}  [{el:.1f} s]; overlap angle mu = {mu * 180 / np.pi:.6f} deg (1 - cos mu = 2 w L_s I_d / (sqrt3 Vm))")
ev = [e for e in r.events if e[1] == "diode"]
errs, kinds = [], {}
for t_e, _, _, d in ev:
    th = (w * t_e - np.pi / 6) % (np.pi / 3)
    non = sum(d)
    ref = 0.0 if non == 3 else mu
    if th > np.pi / 3 - 1e-6:
        th -= np.pi / 3
    errs.append(abs(th - ref) / w)
    kinds[non] = kinds.get(non, 0) + 1
print(f"   diode events by number of conducting diodes after the event {kinds};"
      f" max |t_event - t_ref| = {max(errs):.2e} s over {len(errs)} events")

print("(v-b) three-phase diode bridge, no L_s, C filter (1000 uF || 20 ohm), V_f = 0.7, C starts at 0 V")
c = bridge3(C=1e-3, R=20.0, Vf=0.7)
t1 = time.time()
r = simulate(c, 0.1, h_e=1e-9)
el = time.time() - t1
pon3, poff3 = rect_ref(np.sqrt(3) * Vm, np.pi / 3, w, 20.0, 1e-3, 0.7)
print(f"   {stats(r)}  [{el:.1f} s]")
print(f"   impulse at t = 0 (capacitor loop closed by the diodes on an uncharged C): "
      f"{[(t, round(dw, 6)) for t, dw in r.impulses]} J")
ev = [e for e in r.events if e[1] == "diode"]
errs = []
for t_e, _, _, d in ev[4:]:
    # line-voltage argument: v_xy = sqrt3 Vm sin(theta + 30 deg - k 60 deg)
    psi = (w * t_e + np.pi / 6) % (np.pi / 3) + np.pi / 3
    ref = float(poff3) if sum(d) == 0 else float(pon3)
    errs.append(abs(psi - ref) / w)
print(f"   reference: conduction from {float(pon3) * 180 / np.pi:.6f} to {float(poff3) * 180 / np.pi:.6f} deg of the line voltage;"
      f" max |t_event - t_ref| = {max(errs):.2e} s over {len(errs)} events; ON-sets seen {sorted(set(sum(e[3]) for e in ev))}")
