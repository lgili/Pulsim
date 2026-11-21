# Baseline / acceptance probe for add-exact-propagator-strategy (needs PYTHONPATH=<worktree>/python).
# Run on the untouched build first, then after each step: errors against the 50-digit references,
# step/event counters, the failure cases of the proposal, wall time (reported, never asserted).
# `python dsed_baseline.py OUT.npz` also saves pwl / trbdf2 trajectories for the bit-identity check;
# `python dsed_baseline.py OUT.npz BEFORE.npz` compares them with np.array_equal.
import sys, time, warnings
import numpy as np, pulsim as p
warnings.simplefilter("ignore")
print("pulsim from", p.__file__)
Vin, L0, C, gon, goff, T = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 1e-5
CCM_REF = (11.8015131102418, 12.2288664641139)
DCM_VC, DCM_TAU = 8.99624708728943, 0.799830698444468

def buck(R, vth, L=L0, D=0.5, t0=0.0, edges=True):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", Vin); b.add_switch("S", "vin", "sw", gon, goff)
    b.add_diode("D1", "gnd", "sw", gon, goff, vth)
    b.add_inductor("L", "sw", "vout", L); b.add_capacitor("C", "vout", "gnd", C); b.add_resistor("RL", "vout", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"), b.graph.num_switches, t0)
    return b, fn

def run(label, f):
    t1 = time.perf_counter()
    try:
        out = f()
        print(f"{label}: {out}  [{(time.perf_counter() - t1) * 1e3:.1f} ms]")
        return True
    except Exception as e:
        print(f"{label}: FAILS {type(e).__name__}: {str(e)[:160]}")
        return False

def counters(r):
    return f"steps={r.n_accept} exact={r.n_exact_steps} events={r.n_events} samples={len(r.times)}" + (
        f" eig={r.exact['n_eig']} pade={r.exact['n_pade']} pred_evals={r.exact['n_predicate_evals']}"
        if getattr(r, "exact", None) else "")

def ccm():
    b, fn = buck(1.0, 1e-6)
    r = p.simulate(b, t_end=40 * T, engine="dsed", switch_fn=fn, dt=T / 10)
    iL, vC = np.asarray(r.i("L"))[-1], np.asarray(r.v("vout"))[-1]
    return f"err iL={iL - CCM_REF[0]:+.2e} vC={vC - CCM_REF[1]:+.2e}  {counters(r)}"

def dcm(vth, nper=4000):
    b, fn = buck(50.0, vth, D=0.3)
    r = p.simulate(b, t_end=nper * T, engine="dsed", switch_fn=fn)
    t = np.asarray(r.times); vC = np.asarray(r.v("vout"))
    k = int(np.argmin(abs(t - (nper - 1) * T)))
    # switch_states is the STAMPED mask (the one over (t[k-1], t[k]]): the
    # event sample is the last one stamped ON.
    sw = np.asarray(r.switch_states)[:, b.switch_index_of("D1")]
    off = [t[i - 1] for i in range(1, len(t)) if sw[i - 1] and not sw[i] and t[i - 1] > (nper - 1) * T]
    tau = (off[0] - (nper - 1) * T) / T if off else float("nan")
    return f"vC(rising edge)-ref={vC[k] - DCM_VC:+.2e}  diode-off tau/T-ref={tau - DCM_TAU:+.2e} (= {(tau - DCM_TAU) * T:+.1e} s)  {counters(r)}"

def rest(vth):
    b, _ = buck(50.0, vth)
    n = b.graph.num_switches
    r = p.simulate(b, t_end=1e-6, engine="dsed", switch_fn=lambda t: p.SwitchStateMask(n))
    return f"v(sw) last={np.asarray(r.v('sw'))[-1]:.6g} iL last={np.asarray(r.i('L'))[-1]:.3e}"

def vend(L):
    b, fn = buck(50.0, 1e-6, L=L, D=0.3)
    return float(np.asarray(p.simulate(b, t_end=40 * T, engine="dsed", switch_fn=fn).v("vout"))[-1])

def rlc_critical(eps):
    import mpmath as mp
    mp.mp.dps = 40
    L, Cc = 100e-6, 100e-9
    R = 2 * np.sqrt(L / Cc) * (1 + eps)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0); b.add_resistor("R", "in", "a", R)
    b.add_inductor("L", "a", "o", L); b.add_capacitor("C", "o", "gnd", Cc)
    tp = 2 * np.pi * np.sqrt(L * Cc)
    r = p.simulate(b, t_end=tp, engine="dsed")
    M = mp.matrix([[-mp.mpf(R) / L, -1 / mp.mpf(L), 10 / mp.mpf(L)], [1 / mp.mpf(Cc), 0, 0], [0, 0, 0]])
    x = mp.expm(M * mp.mpf(tp)) * mp.matrix([0, 0, 1])
    i, v = np.asarray(r.i("L"))[-1], np.asarray(r.v("o"))[-1]
    # relative error in the energy norm (sqrt(L)*i, sqrt(C)*v), and component-wise
    rel = np.hypot(np.sqrt(L) * (i - float(x[0])), np.sqrt(Cc) * (v - float(x[1]))) / np.hypot(
        np.sqrt(L) * float(x[0]), np.sqrt(Cc) * float(x[1]))
    rel_c = max(abs(i - float(x[0])) / abs(float(x[0])), abs(v - float(x[1])) / abs(float(x[1])))
    tiers = getattr(r, "exact", {}).get("masks") if getattr(r, "exact", None) else None
    return f"rel err (energy)={rel:.1e} (max component {rel_c:.1e}) tiers={[(m['tier'], round(m['kappa'])) for m in tiers] if tiers else None}"

def sync_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0); b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_switch("Q2", "sw", "gnd", 1e3, 1e-9); b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6); b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_dead_time_pwm_pair_fn(100e3, 0.5, b.switch_index_of("Q1"), b.switch_index_of("Q2"), b.graph.num_switches, 0.0)
    return b, fn

def perf():
    b, fn = sync_buck()
    walls = []
    for _ in range(5):
        t1 = time.perf_counter(); r = p.simulate(b, t_end=20e-3, switch_fn=fn, engine="dsed"); walls.append(time.perf_counter() - t1)
    return f"vout={np.asarray(r.v('vout'))[-1]:.9f} {counters(r)} wall min={min(walls) * 1e3:.2f} ms median={np.median(walls) * 1e3:.2f} ms"

run("CCM 40T dt=T/10 (ref 1e-10)", ccm)
for vth in (1e-6, 1e-9, 0.0):
    run(f"DCM 4000T V_th={vth:g}", lambda: dcm(vth))
run("(3) rest V_th=0", lambda: rest(0.0))
run("    rest V_th=1e-9 (control)", lambda: rest(1e-9))
for d in (1e-5, 1e-4, 1e-2):
    for s in (+1, -1):
        run(f"(2) DCM V_th=1e-6 L*(1{'+' if s > 0 else '-'}{d:g})", lambda: f"vC(40T)={vend(L0 * (1 + s * d)):.12f}")
run("    FD dvC/dL at 1e-5 (ref 1758.1254)", lambda: f"{(vend(L0 * (1 + 1e-5)) - vend(L0 * (1 - 1e-5))) / (2e-5 * L0):.7f}")
for vth in (0.0, 1e-9):
    def held(vth=vth):
        b, fn = buck(50.0, vth, D=0.3, t0=2e-6)
        return f"vC(t0+40T)={np.asarray(p.simulate(b, t_end=2e-6 + 40 * T, engine='dsed', switch_fn=fn).v('vout'))[-1]:.6f} (13.309497)"
    run(f"    gate held off to 2us, V_th={vth:g}", held)
for eps in (1e-3, 1e-9, 1e-13, 0.0):
    run(f"(5) RLC R/Rc-1={eps:g}", lambda: rlc_critical(eps))
run("sync buck 2000T (perf row 22)", perf)

if len(sys.argv) > 1:
    arrs = {}
    for eng in ("pwl", "trbdf2"):
        b, fn = buck(1.0, 1e-6)
        r = p.simulate(b, t_end=20 * T, engine=eng, switch_fn=fn, **({"dt": T / 50} if eng == "pwl" else {}))
        arrs[f"{eng}_buck"] = np.asarray(r.states)
        b, fn = sync_buck()
        r = p.simulate(b, t_end=20 * T, engine=eng, switch_fn=fn, **({"dt": T / 50} if eng == "pwl" else {}))
        arrs[f"{eng}_sync"] = np.asarray(r.states)
    np.savez(sys.argv[1], **arrs)
    if len(sys.argv) > 2:
        old = np.load(sys.argv[2])
        for k in arrs:
            print(f"bit-identical {k}: {np.array_equal(old[k], arrs[k])} shape={arrs[k].shape}")
