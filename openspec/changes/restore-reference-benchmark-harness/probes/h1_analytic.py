"""Harness probe: three analytic circuits x three engines, time-weighted error on a uniform comparison grid."""
import warnings, numpy as np, pulsim as p
warnings.simplefilter("ignore")
def metric(t, y, ref, t_end, M=2001):
    tg = np.linspace(0.0, t_end, M); yi = np.interp(tg, t, y); r = ref(tg)
    span = r.max() - r.min()
    return np.sqrt(np.trapezoid((yi - r)**2, tg) / t_end) / span, np.abs(yi - r).max()
def rc():
    b = p.CircuitBuilder(); b.add_voltage_source("V", "in", "gnd", 10.0); b.add_resistor("R", "in", "o", 1e3); b.add_capacitor("C", "o", "gnd", 1e-6)
    return b, "o", 5e-3, 1e-5, (lambda t: 10*(1-np.exp(-t/1e-3)))
def rlc():
    R, L, C = 1.0, 1e-3, 10e-6; a = R/(2*L); wd = np.sqrt(1/(L*C) - a*a)
    b = p.CircuitBuilder(); b.add_voltage_source("V", "in", "gnd", 10.0); b.add_resistor("R", "in", "a", R); b.add_inductor("L", "a", "o", L); b.add_capacitor("C", "o", "gnd", C)
    return b, "o", 10e-3, 1e-6, (lambda t: 10*(1-np.exp(-a*t)*(np.cos(wd*t)+a/wd*np.sin(wd*t))))
def rl():
    b = p.CircuitBuilder(); b.add_voltage_source("V", "in", "gnd", 10.0); b.add_resistor("R", "in", "o", 10.0); b.add_inductor("L", "o", "gnd", 1e-3)
    return b, "o", 1e-3, 1e-6, (lambda t: 10*np.exp(-t/1e-4))
for name, mk in (("RC step", rc), ("RLC underdamped", rlc), ("RL step v_L", rl)):
    for eng in ("pwl", "trbdf2", "dsed"):
        b, node, t_end, dt, ref = mk()
        kw = dict(dt=dt) if eng == "pwl" else (dict(rtol=1e-6) if eng == "trbdf2" else {})
        try:
            r = p.simulate(b, t_end, engine=eng, **kw)
            t = np.asarray(r.times); y = np.asarray(r.v(node))
            e, emax = metric(t, y, ref, t_end)
            print(f"{name:16s} {eng:6s} n={len(t):6d} rms/span={e:.2e} max|err|={emax:.2e} V")
        except Exception as ex:
            print(f"{name:16s} {eng:6s} {type(ex).__name__}: {str(ex)[:120]}")
# metric sanity: ramp on a clustered grid
t = np.concatenate([np.linspace(0, 0.01, 1000), np.linspace(0.01, 1, 11)[1:]]); y = t
print("ramp clustered: unweighted mean=%.5f  time-weighted=%.5f" % (y.mean(), np.trapezoid(y, t)/(t[-1]-t[0])))
