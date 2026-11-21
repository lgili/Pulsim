"""Independent reference for the saturating three-leg core.

The circuit of python/tests/test_permeance_magnetic.py::sat_three_leg:
a sine source v(t) = V*sin(2*pi*f*t) through R into a winding of N turns
on leg a (linear permeance Pa); from the top yoke T, leg b (saturable,
GappedCore material law) and leg c (linear Pc) return to the bottom yoke
(the reference). One state, the flux of leg a:

    dPhi_a/dt = (v(t) - R*i)/N,   i = F_x/N,   F_x = F_T + Phi_a/Pa,
    Phi_b(F_T) + Pc*F_T = Phi_a   (flux continuity at T, solved by Newton)

Phi_b(F) = mu0*A*[F/l + Ms*tanh(F/(l*H0))], Ms = B_sat/mu0,
H0 = Ms/(mu_r - 1): written here from the material law, not from pulsim.
Classical RK4 at h and h/2; the difference bounds the error. Prints the
pinned table the test uses. Pure Python + math, no pulsim import.
"""

import math

MU0 = 4e-7 * math.pi
V, F_HZ, R, N = 10.0, 100.0, 2.0, 50
PA, PC = 4e-6, 1e-6
A, L, MUR, BSAT = 50e-6, 40e-3, 3000.0, 0.3
MS = BSAT / MU0
H0 = MS / (MUR - 1)


def phi_b(F):
    return MU0 * A * (F / L + MS * math.tanh(F / (L * H0)))


def p_b(F):
    return MU0 * A / L * (1 + (MUR - 1) / math.cosh(F / (L * H0)) ** 2)


def f_t(phi_a, guess):
    F = guess
    for _ in range(100):
        r = phi_b(F) + PC * F - phi_a
        d = p_b(F) + PC
        dF = r / d
        F -= dF
        if abs(dF) <= 1e-15 * max(1.0, abs(F)):
            break
    return F


def state(t, phi_a, guess):
    FT = f_t(phi_a, guess)
    Fx = FT + phi_a / PA
    i = Fx / N
    return FT, i, (V * math.sin(2 * math.pi * F_HZ * t) - R * i) / N


def run(h, t_out):
    t, y, g = 0.0, 0.0, 0.0
    out = {}
    k_end = round(t_out[-1] / h)
    marks = {round(tt / h): tt for tt in t_out}
    for k in range(k_end + 1):
        if k in marks:
            FT, i, _ = state(t, y, g)
            out[marks[k]] = (i, phi_b(FT), PC * FT)
        if k == k_end:
            break
        g, _, k1 = state(t, y, g)
        _, _, k2 = state(t + h / 2, y + h / 2 * k1, g)
        _, _, k3 = state(t + h / 2, y + h / 2 * k2, g)
        _, _, k4 = state(t + h, y + h * k3, g)
        y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = (k + 1) * h
    return out


if __name__ == "__main__":
    t_out = [k * 1e-3 for k in range(1, 11)]
    a = run(1e-7, t_out)
    b = run(5e-8, t_out)
    worst = max(abs(a[t][j] - b[t][j]) / max(abs(b[s][j]) for s in t_out)
                for t in t_out for j in range(3))
    print(f"# RK4 h=1e-7 vs h=5e-8: worst difference {worst:.2e} of the span")
    print("REF = [  # t, i [A], Phi_b [Wb], Phi_c [Wb]")
    for t in t_out:
        i, pb, pc = b[t]
        print(f"    ({t:.3e}, {i:.15e}, {pb:.15e}, {pc:.15e}),")
    print("]")
