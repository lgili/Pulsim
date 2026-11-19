"""Independent reference for the flux-map PMSM locked-rotor step.

Run once (needs SciPy); the numbers it prints are pinned in
`python/tests/test_pmsm_flux_map.py`. Nothing here imports pulsim.

The machine (declared SYNTHETIC test parameters, not a real motor) has
the analytic co-energy

    W'(i_d, i_q) = psi·i_d + ½·L_q2·i_q² + L_s·I_s²·log cosh(r/I_s),
    r = |(i_d, i_q)|,

so lambda = grad W' saturates with the current magnitude and couples the
axes (d lambda_d / d i_q = d lambda_q / d i_d != 0). With the rotor locked
at theta_e = 0 and constant dq voltages, the stator law in the rotor
frame is

    v_dq = R_s·i_dq + H(i_dq)·d i_dq/dt,      H = Hessian of W',

with v_dq ramped linearly from 0 over T_RISE (a consistent start: a
voltage step at t = 0 would start the trapezoidal rule from an
inconsistent derivative, a first-order error of its own),

integrated here as a 2-state ODE by Radau and DOP853 at rtol 1e-12 (they
must agree). The device under test sees only a table of lambda sampled
from the same W' and fits its own spline, so the comparison covers the
fit, the abc transformation, the stamp and the time discretisation.

    python python/tests/_machine_reference/flux_map_reference.py
"""

from __future__ import annotations

import math

PSI, L_Q2, L_S, I_S, R_S = 0.05, 2e-3, 1e-3, 10.0, 0.5
V_D, V_Q = 6.0, 8.0          # steady state i = (12, 16) A, |i| = 20 A = 2·I_s
T_RISE = 50e-6               # linear voltage ramp 0 -> V over [0, T_RISE]
INSTANTS = (0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3, 16e-3)


def iso(i_d: float, i_q: float):
    """f = L_s·I_s·tanh(x)/r and g = f'(r)/r, x = r/I_s (series near 0)."""
    x = math.hypot(i_d, i_q) / I_S
    if x < 1e-3:
        f = L_S * (1 - x * x / 3 + 2 * x**4 / 15)
        g = L_S / I_S**2 * (-2 / 3 + 8 * x * x / 15)
    else:
        th = math.tanh(x)
        f = L_S * th / x
        g = L_S / I_S**2 * ((1 - th * th) - th / x) / (x * x)
    return f, g


def flux(i_d: float, i_q: float):
    f, _ = iso(i_d, i_q)
    return PSI + f * i_d, L_Q2 * i_q + f * i_q


def hessian(i_d: float, i_q: float):
    f, g = iso(i_d, i_q)
    return (f + g * i_d * i_d, g * i_d * i_q, L_Q2 + f + g * i_q * i_q)


def rhs(t, y):
    i_d, i_q = y
    hdd, hdq, hqq = hessian(i_d, i_q)
    ramp = min(t / T_RISE, 1.0)
    a, b = ramp * V_D - R_S * i_d, ramp * V_Q - R_S * i_q
    det = hdd * hqq - hdq * hdq
    return [(hqq * a - hdq * b) / det, (hdd * b - hdq * a) / det]


def main() -> None:
    from scipy.integrate import solve_ivp

    out = {}
    for method in ("Radau", "DOP853"):
        first = solve_ivp(rhs, (0.0, T_RISE), [0.0, 0.0], method=method,
                          rtol=1e-12, atol=1e-12)   # the ramp's kink is a
        sol = solve_ivp(rhs, (T_RISE, INSTANTS[-1]), first.y[:, -1],
                        method=method, t_eval=INSTANTS, rtol=1e-12,
                        atol=1e-12)                 # segment boundary
        out[method] = sol.y.T
    for k, t in enumerate(INSTANTS):
        r, d = out["Radau"][k], out["DOP853"][k]
        assert max(abs(r - d)) < 1e-9, (t, r, d)
        print(f"    ({t!r}, {float(r[0])!r}, {float(r[1])!r}),")


if __name__ == "__main__":
    main()
