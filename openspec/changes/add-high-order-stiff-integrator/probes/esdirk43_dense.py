"""Dense output of order 4 for ESDIRK4(3)6L[2]SA, in stage-value form (DECISIONS 3).

x(t0 + θh) = x0 + Σ_{i=2..6} w_i(θ)·(X_i − x0)

i.e. the continuous weights b(θ) = Σ w_i(θ)·a_i (rows 2..6 of A). Being a
combination of stage values it never needs a stage derivative, so it is
defined for the algebraic unknowns of a DAE and is independent of the
explicit first stage's derivative there (the only part of k_1 a DAE does
not determine). Order 4 (ODE sense) needs Σb=θ, Σb·c=θ²/2, Σb·c²=θ³/3,
Σb·c³=θ⁴/4, Σb·A·c²=θ⁴/12; with stage order 2 (A·c = c²/2, all six rows)
the other three order-4 conditions follow. 5 conditions, 5 unknowns.

Run: python3 esdirk43_dense.py  (sympy). Prints the polynomial
coefficients that core/include/pulsim/solver/step/esdirk43.hpp carries.
"""
import sympy as sp

R = sp.Rational
g = R(1, 4)
A = sp.zeros(6, 6)
A[1, 0] = R(1, 4); A[1, 1] = g
A[2, 0] = R(8611, 62500); A[2, 1] = R(-1743, 31250); A[2, 2] = g
A[3, 0] = R(5012029, 34652500); A[3, 1] = R(-654441, 2922500); A[3, 2] = R(174375, 388108); A[3, 3] = g
A[4, 0] = R(15267082809, 155376265600); A[4, 1] = R(-71443401, 120774400)
A[4, 2] = R(730878875, 902184768); A[4, 3] = R(2285395, 8070912); A[4, 4] = g
A[5, :] = sp.Matrix([[R(82889, 524892), 0, R(15625, 83664), R(69875, 102672), R(-2260, 8211), g]])
c = A * sp.ones(6, 1)
c2 = c.applyfunc(lambda x: x**2)
c3 = c.applyfunc(lambda x: x**3)
Ac = A * c
assert all(Ac[i] == c2[i] / 2 for i in range(6)), "stage order 2"
th = sp.symbols("theta")
vecs = [sp.ones(6, 1), c, c2, c3, A * c2]
rhs = [th, th**2 / 2, th**3 / 3, th**4 / 4, th**4 / 12]
W = sp.symbols("w2:7")
b_th = sum((W[i] * A[i + 1, :].T for i in range(5)), sp.zeros(6, 1))
sol = sp.solve([sp.expand((b_th.T * v)[0] - r) for v, r in zip(vecs, rhs)], W, dict=True)[0]
w = [sp.expand(sol[x]) for x in W]
b_full = sum((w[i] * A[i + 1, :].T for i in range(5)), sp.zeros(6, 1))
# all eight order-4 conditions of b(θ), identically in θ
AAc = A * Ac
cAc = sp.Matrix([c[i] * Ac[i] for i in range(6)])
for v, r in zip([sp.ones(6, 1), c, c2, Ac, c3, cAc, A * c2, AAc],
                [th, th**2/2, th**3/3, th**3/6, th**4/4, th**4/8, th**4/12, th**4/24]):
    assert sp.expand((b_full.T * v)[0] - r) == 0
assert [wi.subs(th, 1) for wi in w] == [0, 0, 0, 0, 1], "w(1) = e6: continuous, ends on X_6"
assert all(wi.subs(th, 0) == 0 for wi in w)
print("w_i(θ) = Σ_k p_ik θ^k, k = 1..4 (stage i = 2..6):")
for i, wi in enumerate(w):
    p = sp.Poly(wi, th)
    print(f"  stage {i+2}:", [str(p.coeff_monomial(th**k)) for k in range(1, 5)])
# fifth-order residual of the continuous weights at θ = 1/2 (what limits the dense order)
c4 = c.applyfunc(lambda x: x**4)
print("order-5 bushy residual Σb(θ)c⁴ − θ⁵/5 at θ=1/2:", float((b_full.T * c4)[0].subs(th, R(1, 2)) - R(1, 2)**5 / 5))
