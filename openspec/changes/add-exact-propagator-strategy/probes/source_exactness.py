"""Ramp 0 -> 10 V over 1 ms into an RC (R 1k, C 1u), and the 50 Hz sine into 1 ohm + 1 mH at h = T/20,
on dsed: error against the closed forms, step counts, tiers."""
import warnings
import numpy as np
import pulsim as p

R, C, TR = 1e3, 1e-6, 1e-3
tau = R * C
def ramp_ref(t):
    a = 10.0 / TR
    t1 = np.minimum(t, TR)
    v = a * (t1 - tau * (1 - np.exp(-t1 / tau)))
    late = t > TR
    vT = a * (TR - tau * (1 - np.exp(-TR / tau)))
    v[late] = 10.0 + (vT - 10.0) * np.exp(-(t[late] - TR) / tau)
    return v

for integ in ("auto", "rk45"):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", 0.0, 10.0, 0.0, 1.0, 0.0, TR, 0.0)
    b.add_resistor("R", "a", "o", R)
    b.add_capacitor("C", "o", "gnd", C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=5e-3, engine="dsed", integrator=integ)
    t = np.asarray(r.times); v = np.asarray(r.v("o"))
    print(f"ramp RC integrator={integ}: max|v - ref| = {np.abs(v - ramp_ref(t)).max():.3e} V, "
          f"samples {len(t)}, n_exact_steps {getattr(r, 'n_exact_steps', None)}, "
          f"tiers {[m['tier'] for m in (r.exact or {}).get('masks', [])]}, "
          f"reasons {[m['reason'] for m in (r.exact or {}).get('masks', [])]}, "
          f"sample near 1 ms: {bool(np.any(np.abs(t - TR) < 1e-12))}")

w = 2 * np.pi * 50; Rl, L = 1.0, 1e-3
b = p.CircuitBuilder()
b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
b.add_resistor("R", "a", "b", Rl)
b.add_inductor("L", "b", "gnd", L)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = p.simulate(b, t_end=0.1, engine="dsed", integrator="rk45", dt=1e-2)   # dt is the ceiling; h = 1e-3 = T/20
t = np.asarray(r.times); i = np.asarray(r.i("L"))
Z = Rl + 1j * w * L; ph = np.angle(Z)
ref = 100 / abs(Z) * (np.sin(w * t - ph) + np.sin(ph) * np.exp(-Rl / L * t))
print(f"sine RL at h = {np.diff(t).max():.1e} s (T/20): max|i - ref| = {np.abs(i - ref).max():.3e} A over {len(t)} samples, "
      f"n_exact_steps {r.n_exact_steps}, tiers {[m['tier'] for m in r.exact['masks']]}")
