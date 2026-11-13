"""No-regression baseline of engine='trbdf2' for the TR-BDF2 accuracy track.

    python trbdf2_baseline.py --record   # untouched build: writes trbdf2_baseline.json
    python trbdf2_baseline.py            # after a change: before/after table
    python trbdf2_baseline.py --save=F   # also write this run's rows/hashes to F
    python trbdf2_baseline.py --against=F  # compare with F's rows/hashes instead

Needs PYTHONPATH=<worktree>/python. Errors are against exact piecewise-LTI
references where one exists (buck CCM/DCM via expm + root finding, pulse RC,
LC tank analytic), else against a reference run frozen in the json: pwl at
a fine dt (dsed refuses the V_th = 0 diodes of these circuits at t = 0).
`tw` is the time-weighted mean of the output (np.trapezoid) against the
reference run's. Counters come from res._trbdf2_stats. The two pwl runs
must stay bit-identical (sha256 of the state arrays).
"""
import hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import pulsim as p

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "trbdf2_baseline.json"

# ---- buck (audit circuit, core_review numerics/p2_buck.py) ----
Vin, Lb, Cb, gon, goff, T = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 1e-5
T0 = 2e-6


def buck(R, D, nper, engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", Vin)
    b.add_switch("S", "vin", "sw", gon, goff)
    b.add_diode("D1", "gnd", "sw", gon, goff)
    b.add_inductor("L", "sw", "vout", Lb)
    b.add_capacitor("C", "vout", "gnd", Cb)
    b.add_resistor("RL", "vout", "gnd", R)
    n = b.graph.num_switches
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        ph = (t - T0) / T
        fr = ph - np.floor(ph + 1e-9)
        m.set(k, bool(t >= T0 - 1e-13 and fr < D - 1e-9))
        return m
    return p.simulate(b, t_end=T0 + nper * T, engine=engine, switch_fn=fn, **kw)


def buck_exact(R, D, nper):
    """Piecewise-LTI expm propagation, diode events by brentq (p2_buck.py)."""
    from scipy.optimize import brentq

    def AB(gs, gd):
        k = 1 / (gs + gd)
        return (np.array([[-k / Lb, -1 / Lb], [1 / Cb, -1 / (R * Cb)]]),
                np.array([gs * Vin * k / Lb, 0.0]))

    def prop(A, B, x, dt):
        M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = B
        E = expm(M * dt); return E[:2, :2] @ x + E[:2, 2]

    def seg(x, rem, gs):
        vsw = (gs * Vin - x[0]) / (gs + goff)
        d_on = -vsw >= 0
        A, B = AB(gs, gon if d_on else goff)
        f = lambda dt: -((gs * Vin - prop(A, B, x, dt)[0]) / (gs + (gon if d_on else goff)))
        f0, f1 = f(1e-15), f(rem)
        if f0 * f1 < 0:
            dtx = brentq(f, 1e-15, rem, xtol=1e-18, rtol=1e-15)
            x = prop(A, B, x, dtx)
            A2, B2 = AB(gs, goff if d_on else gon)
            return prop(A2, B2, x, rem - dtx)
        return prop(A, B, x, rem)
    x = np.zeros(2)
    for _ in range(nper):
        x = seg(x, D * T, gon)
        x = seg(x, (1 - D) * T, goff)
    return x  # [iL, vC] at T0 + nper*T


# ---- other circuits ----
def boost(engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", gon, goff)
    b.add_diode("D1", "sw", "vout", gon, goff)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    n = b.graph.num_switches; k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n); fr = t / T - np.floor(t / T + 1e-9)
        m.set(k, bool(fr < 0.5 - 1e-9)); return m
    return p.simulate(b, t_end=200 * T, engine=engine, switch_fn=fn, **kw)


def vsi_rl(engine, **kw):
    # half bridge, split DC (two sources), sine-PWM 10 kHz / 50 Hz, RL load
    b = p.CircuitBuilder()
    b.add_voltage_source("Vp", "p", "gnd", 100.0)
    b.add_voltage_source("Vn", "gnd", "n", 100.0)
    b.add_switch("S1", "p", "a", gon, goff)
    b.add_switch("S2", "a", "n", gon, goff)
    b.add_diode("D1", "a", "p", gon, goff)
    b.add_diode("D2", "n", "a", gon, goff)
    b.add_resistor("R", "a", "m", 5.0)
    b.add_inductor("L", "m", "gnd", 5e-3)
    n = b.graph.num_switches; k1 = b.switch_index_of("S1"); k2 = b.switch_index_of("S2")
    fc = 10e3

    def fn(t):
        m = p.SwitchStateMask(n)
        ref = 0.8 * np.sin(2 * np.pi * 50 * t)
        ph = t * fc - np.floor(t * fc)
        car = 4 * abs(ph - 0.5) - 1  # triangle in [-1, 1]
        up = bool(ref > car)
        m.set(k1, up); m.set(k2, not up); return m
    return p.simulate(b, t_end=20e-3, engine=engine, switch_fn=fn, **kw)


def bridge_c(engine, **kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "s", "a", 0.5)
    b.add_inductor("Ls", "a", "ac1", 1e-3)
    b.add_diode("D1", "ac1", "dp", gon, goff)
    b.add_diode("D2", "gnd", "dp", gon, goff)
    b.add_diode("D3", "dn", "ac1", gon, goff)
    b.add_diode("D4", "dn", "gnd", gon, goff)
    b.add_capacitor("C", "dp", "dn", 470e-6)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    return p.simulate(b, t_end=40e-3, engine=engine, **kw)


def shockley_rect(engine, **kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("Rs", "s", "a", 10.0)
    b.add_shockley_diode("D", "a", "out")
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("RL", "out", "gnd", 1e3)
    return p.simulate(b, t_end=5e-3, engine=engine, **kw)


def mosfet_hb(engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", 48.0)
    b.add_mosfet_level1("M1", "vdc", "sw", "g1", K=20.0, V_T=3.0)
    b.add_mosfet_level1("M2", "sw", "gnd", "g2", K=20.0, V_T=3.0)
    b.add_pulse_voltage_source("Vg1", "g1", "sw", 0.0, 12.0, 0.2e-6, 4.6e-6, T)
    b.add_pulse_voltage_source("Vg2", "g2", "gnd", 0.0, 12.0, 5.2e-6, 4.6e-6, T)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", 5.0)
    return p.simulate(b, t_end=50 * T, engine=engine, **kw)


def sat_ind(engine, **kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 20.0, 50.0)
    b.add_resistor("R", "s", "a", 2.0)
    b.add_saturable_inductor("Ls", "a", "gnd", 50e-3, 2.0, 5e-3)
    return p.simulate(b, t_end=40e-3, engine=engine, **kw)


PR, PC, PT0, PW = 1e3, 1e-9, 3.3e-6, 50e-6


def pulse_rc(engine, **kw):  # core_review numerics/p13b.py
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, PT0, PW, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", PR)
    b.add_capacitor("C", "a", "gnd", PC)
    return p.simulate(b, t_end=100e-6, engine=engine, **kw)


def pulse_rc_exact(t):
    tau = PR * PC; v = np.zeros_like(t); m = t > PT0
    v[m] = 10 * (1 - np.exp(-(t[m] - PT0) / tau))
    m2 = t > PT0 + PW
    v[m2] = 10 * (1 - np.exp(-PW / tau)) * np.exp(-(t[m2] - PT0 - PW) / tau)
    return v


TL, TC, TR = 1e-3, 1e-6, 0.1


def lc_tank(engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", TR)
    b.add_inductor("L", "a", "b", TL)
    b.add_capacitor("C", "b", "gnd", TC)
    return p.simulate(b, t_end=5e-3, engine=engine, **kw)


def lc_exact(t):
    A = np.array([[-TR / TL, -1 / TL, 10 / TL], [1 / TC, 0, 0], [0, 0, 0]])
    return (expm(A * t) @ np.array([0.0, 0.0, 1.0]))[:2]  # [i, v]


# name -> (runner(engine, **kw), signals, reference engine kwargs)
CIRCUITS = {
    "buck_ccm": (lambda e, **k: buck(1.0, 0.5, 400, e, **k), [("i", "L"), ("v", "vout")], None),
    "buck_dcm": (lambda e, **k: buck(50.0, 0.3, 400, e, **k), [("i", "L"), ("v", "vout")], None),
    "boost": (boost, [("i", "L"), ("v", "vout")], ("pwl", dict(dt=T / 2000))),
    "vsi_rl": (vsi_rl, [("i", "L")], ("pwl", dict(dt=5e-8))),
    "bridge_c": (bridge_c, [("i", "Ls"), ("v", "dp")], ("pwl", dict(dt=2e-7))),
    "shockley_rect": (shockley_rect, [("v", "out")], ("pwl", dict(dt=1e-8))),
    "mosfet_hb": (mosfet_hb, [("i", "L"), ("v", "out")], ("pwl", dict(dt=T / 4000))),
    "sat_ind": (sat_ind, [("i", "Ls")], ("pwl", dict(dt=2e-7))),
    "pulse_rc": (pulse_rc, [("v", "a")], None),
    "lc_tank": (lc_tank, [("i", "L"), ("v", "b")], None),
}
PWL_RUNS = {
    "pwl_buck_ccm": lambda: buck(1.0, 0.5, 40, "pwl", dt=T / 200),
    "pwl_shockley": lambda: shockley_rect("pwl", dt=1e-6),
    "dsed_buck_ccm": lambda: buck(1.0, 0.5, 40, "dsed"),
}


def sig(res, s):
    return np.asarray(res.i(s[1]) if s[0] == "i" else res.v(s[1]))


def tw(res, s):
    t = np.asarray(res.times); y = sig(res, s)
    return float(np.trapezoid(y, t) / (t[-1] - t[0]))


def reference(name):
    run, sigs, ref = CIRCUITS[name]
    if name.startswith("buck"):
        R, D = (1.0, 0.5) if name == "buck_ccm" else (50.0, 0.3)
        xe = buck_exact(R, D, 400)
        # dsed cannot settle the V_th = 0 DCM diode at t = 0; pwl at T/2000 there
        r = run("dsed") if name == "buck_ccm" else run("pwl", dt=T / 2000)
        return {"final": [float(xe[0]), float(xe[1])], "tw": tw(r, sigs[-1])}
    if name == "pulse_rc":
        return {"final": None, "tw": None}
    if name == "lc_tank":
        r = run("dsed")
        return {"final": [float(v) for v in lc_exact(5e-3)], "tw": tw(r, sigs[-1])}
    eng, kw = ref
    r = run(eng, **kw)
    return {"final": [float(sig(r, s)[-1]) for s in sigs], "tw": tw(r, sigs[-1])}


def measure(name, ref, **kw):
    run, sigs, _ = CIRCUITS[name]
    t1 = time.perf_counter()
    r = run("trbdf2", **kw)
    wall = time.perf_counter() - t1
    t = np.asarray(r.times); h = np.diff(t)
    st = dict(r._trbdf2_stats)
    if name == "pulse_rc":
        err = float(np.max(np.abs(sig(r, sigs[0]) - pulse_rc_exact(t))))
        twe = None
    else:
        err = float(max(abs(sig(r, s)[-1] - f) for s, f in zip(sigs, ref["final"])))
        twe = float(abs(tw(r, sigs[-1]) - ref["tw"]))
    return {"err": err, "tw_err": twe, "n_accept": st["n_accept"], "n_reject": st["n_reject"],
            "n_solves": st["n_solves"], "n_gate": st["n_gate_events"], "n_diode": st["n_diode_events"],
            "h_min": float(h.min()), "wall_s": wall,
            "extra": {k: st[k] for k in ("n_derivative_refreshes", "n_refresh_probes",
                                         "h_post_event_min", "h_max_basis") if k in st}}


CASES = [(n, {}) for n in CIRCUITS] + [("buck_ccm", dict(rtol=1e-4, atol=1e-7)),
                                       ("buck_ccm", dict(rtol=1e-6, atol=1e-9)),
                                       ("buck_ccm", dict(rtol=1e-8, atol=1e-11))]


def key(n, kw):
    return n + ("" if not kw else "@rtol=%g" % kw["rtol"])


def pwl_hash(f):
    r = f()
    a = np.ascontiguousarray(np.asarray(r.states)); t = np.ascontiguousarray(np.asarray(r.times))
    return hashlib.sha256(a.tobytes() + t.tobytes()).hexdigest()


def main():
    record = "--record" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = json.loads(OUT.read_text()) if OUT.exists() else None
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    cmp = json.loads(Path(opt["against"]).read_text()) if "against" in opt else base
    if record:
        refs = {n: reference(n) for n in CIRCUITS}
    else:
        refs = base["refs"]
    rows = {}
    for n, kw in CASES:
        if only and n not in only:
            continue
        try:
            rows[key(n, kw)] = measure(n, refs[n], **kw)
        except Exception as ex:  # report, do not hide
            rows[key(n, kw)] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    hashes = {n: pwl_hash(f) for n, f in PWL_RUNS.items()}
    if record:
        OUT.write_text(json.dumps({"refs": refs, "rows": rows, "pwl_sha256": hashes}, indent=1))
        print("recorded", OUT)
    hdr = f"{'case':22s} {'err':>19s} {'tw_err':>19s} {'accept':>13s} {'solves':>13s} {'reject':>9s} {'h_min':>19s}"
    print(hdr)
    for k, r in rows.items():
        b = (cmp or {}).get("rows", {}).get(k, {}) if not record else {}
        if "error" in r:
            print(f"{k:22s} {r['error']}")
            continue

        def c(f, fmt):
            new = r.get(f); old = b.get(f)
            ns = "-" if new is None else format(new, fmt)
            return ns if old is None else f"{format(old, fmt)}>{ns}"
        print(f"{k:22s} {c('err', '.2e'):>19s} {c('tw_err', '.2e'):>19s} {c('n_accept', 'd'):>13s} "
              f"{c('n_solves', 'd'):>13s} {c('n_reject', 'd'):>9s} {c('h_min', '.1e'):>19s}  wall {r['wall_s']:.2f}s {r['extra'] or ''}")
    if "save" in opt:
        Path(opt["save"]).write_text(json.dumps({"rows": rows, "pwl_sha256": hashes}, indent=1))
    if cmp:
        for n, hsh in hashes.items():
            old = cmp["pwl_sha256"].get(n)
            print(f"{n}: bit-identical={'(no record)' if old is None else hsh == old}")


if __name__ == "__main__":
    main()
