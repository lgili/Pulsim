# Probe of the CURRENT kernel (needs PYTHONPATH=<worktree>/python).
# What the current kernel does with g_off = 0 and with a very large g_on.
import numpy as np, pulsim as p, warnings
warnings.simplefilter("ignore")
def rl(g_on, g_off, closed=True, engine='pwl', **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "a", g_on, g_off)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    n = b.graph.num_switches; k = b.switch_index_of("S")
    def fn(t):
        m = p.SwitchStateMask(n); m.set(k, bool(closed) if not callable(closed) else bool(closed(t))); return m
    return p.simulate(b, t_end=10e-3, switch_fn=fn, engine=engine, **kw)
for g_on, g_off, closed, lab in [(1e3, 1e-9, True, "baseline"), (1e3, 0.0, True, "g_off=0, closed"), (1e3, 0.0, False, "g_off=0, open (floating a,b)"),
                                 (1e3, 0.0, lambda t: t < 5e-3, "g_off=0, opens at 5 ms"), (1e12, 1e-9, True, "g_on=1e12"), (float('inf'), 1e-9, True, "g_on=inf")]:
    for eng, kw in [('pwl', dict(dt=5e-5)), ('trbdf2', {})]:
        try:
            r = rl(g_on, g_off, closed, eng, **kw); i = np.asarray(r.i("L")); v = np.asarray(r.v("b"))
            print(f"{lab:32s} {eng:7s}: i_L(end)={i[-1]:+.6e} (ideal 9.999546) finite={np.isfinite(i).all()} max|v_b|={np.abs(v).max():.3e}")
        except Exception as e:
            print(f"{lab:32s} {eng:7s}: {type(e).__name__}: {str(e)[:170]}")
