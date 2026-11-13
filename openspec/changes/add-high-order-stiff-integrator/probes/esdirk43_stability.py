# sympy. Run from this directory (it reads esdirk43_lc.py).
from fractions import Fraction as F
import sympy as sp
exec(open("esdirk43_lc.py").read().split("An = np.array")[0].replace("import numpy as np", ""))
Ah = sp.Matrix([[sp.Rational(A[i][j].numerator, A[i][j].denominator) for j in range(1, 6)] for i in range(1, 6)])
a1 = sp.Matrix([sp.Rational(A[i][0].numerator, A[i][0].denominator) for i in range(1, 6)])
Y = -Ah.inv()*a1                      # stage values per unit y_n as z -> -inf
print("R(inf)   =", Y[4], float(Y[4]))
# embedded: yhat = y + h*bh.K ; as z->inf: K_1 = z*y (stage 1 explicit), K_i = z*Y_i ; yhat/y -> 1 + z*(bh1 + sum bh_i Y_i) -> needs bh1 + sum bh_i Y_i = 0 for a finite limit
bhs = [sp.Rational(x.numerator, x.denominator) for x in bh]
lead = bhs[0] + sum(bhs[i+1]*Y[i] for i in range(5))
print("embedded: coefficient of z as z->inf =", sp.nsimplify(lead), float(lead))
bs = [sp.Rational(x.numerator, x.denominator) for x in b]
print("main: coefficient of z =", sp.simplify(bs[0] + sum(bs[i+1]*Y[i] for i in range(5))))
z = sp.symbols('z'); Afull = sp.Matrix(6, 6, lambda i, j: sp.Rational(A[i][j].numerator, A[i][j].denominator))
one = sp.ones(6, 1)
Rhat = 1 + z*(sp.Matrix([bhs]) * (sp.eye(6) - z*Afull).inv() * one)[0]
print("Rhat(-inf) =", sp.limit(sp.simplify(Rhat), z, -sp.oo), float(sp.limit(sp.simplify(Rhat), z, -sp.oo)))
Rm = 1 + z*(sp.Matrix([bs]) * (sp.eye(6) - z*Afull).inv() * one)[0]
print("R(-inf) =", sp.limit(sp.simplify(Rm), z, -sp.oo))
