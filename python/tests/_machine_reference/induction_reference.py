"""Independent references for the MNA-native induction machine tests.

Run once (needs SciPy); the numbers it prints are pinned in
`python/tests/test_induction_machine_mna.py`. Nothing here imports
pulsim: the same machine equations (design.md, stationary frame,
amplitude-invariant Clarke) are integrated as a 5-state ODE in
(lambda_s_alpha, lambda_s_beta, lambda_r_alpha, lambda_r_beta, omega)
by two different SciPy integrators at rtol 1e-12, and the two must
agree; the steady-state numbers come from the per-phase equivalent
circuit in closed form.

    python python/tests/_machine_reference/induction_reference.py
"""

from __future__ import annotations

import math

# Declared test machine (design.md, not a real machine).
RS, LLS, RR, LLR, LM, PP = 0.5, 2e-3, 0.4, 2e-3, 60e-3, 2
V_RMS, F = 100.0, 60.0
# Free acceleration (direct-on-line start).
J, B, T_LOAD, T_END = 0.01, 0.0, 0.0, 0.2
INSTANTS = (0.02, 0.05, 0.1, 0.15, 0.2)


def equivalent_circuit(slip: float) -> tuple[float, float]:
    """Stator current [A rms] and torque [N·m] at `slip`."""
    w = 2 * math.pi * F
    zr = RR / slip + 1j * w * LLR
    zm = 1j * w * LM
    z = RS + 1j * w * LLS + zm * zr / (zm + zr)
    i_s = V_RMS / z
    i_r = i_s * zm / (zm + zr)
    torque = 3 * abs(i_r) ** 2 * RR / slip / (w / PP)
    return abs(i_s), torque


def free_acceleration(method: str) -> dict[str, list[float]]:
    from scipy.integrate import solve_ivp

    lr = LLR + LM
    sig_ls = LLS + LM * LLR / lr
    kr, tau_r = LM / lr, lr / RR
    vp = V_RMS * math.sqrt(2)
    w = 2 * math.pi * F
    s3 = math.sqrt(3)

    def v_ab(t):
        va = vp * math.sin(w * t)
        vb = vp * math.sin(w * t - 2 * math.pi / 3)
        vc = vp * math.sin(w * t + 2 * math.pi / 3)
        return (2 * va - vb - vc) / 3, (vb - vc) / s3

    def rhs(t, x):
        lsa, lsb, lra, lrb, om = x
        isa = (lsa - kr * lra) / sig_ls
        isb = (lsb - kr * lrb) / sig_ls
        va, vb = v_ab(t)
        we = PP * om
        torque = 1.5 * PP * kr * (lra * isb - lrb * isa)
        return [va - RS * isa, vb - RS * isb,
                -lra / tau_r + LM / tau_r * isa - we * lrb,
                -lrb / tau_r + LM / tau_r * isb + we * lra,
                (torque - B * om - T_LOAD) / J]

    sol = solve_ivp(rhs, (0.0, T_END), [0.0] * 5, method=method,
                    rtol=1e-12, atol=1e-13, t_eval=INSTANTS,
                    max_step=1e-4)
    lsa, lsb, lra, lrb, om = sol.y
    isa = (lsa - kr * lra) / sig_ls
    isb = (lsb - kr * lrb) / sig_ls
    ia = isa
    ib = -0.5 * isa + s3 / 2 * isb
    return {"omega": list(om), "i_a": list(ia), "i_b": list(ib),
            "psi_ra": list(lra)}


if __name__ == "__main__":
    for s in (1.0, 0.05, 1e-12):
        print(f"slip {s:g}: I = {equivalent_circuit(s)[0]:.10g} A rms, "
              f"T = {equivalent_circuit(s)[1]:.10g} N·m")
    a = free_acceleration("Radau")
    b = free_acceleration("DOP853")
    for k in a:
        dev = max(abs(x - y) for x, y in zip(a[k], b[k]))
        print(k, [f"{x:.12g}" for x in a[k]], f"Radau-DOP853 {dev:.2e}")
