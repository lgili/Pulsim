"""Truth for the excess-loss element (add-permeance-magnetic-domain 4.2/4.3).

40-digit mpmath: the mean of |cos|^1.5 (the ideal law's sine constant),
the mean power of the regularised law F = c*u/(u^2+eps^2)^(1/4) under
u = U*cos(wt), and the regularisation error against its leading term
(3/4)*(eps/U)^2. Run: python excess_loss_truth.py
"""
import mpmath as mp

mp.mp.dps = 40
K = mp.gamma(mp.mpf(5) / 4) / (mp.sqrt(mp.pi) * mp.gamma(mp.mpf(7) / 4))


def p_reg(c, eps, U):
    def f(t):
        return c * (U * mp.cos(t)) ** 2 / ((U * mp.cos(t)) ** 2 + eps ** 2) ** 0.25

    h = mp.pi / 2
    return mp.quad(f, [0, h - mp.mpf("1e-2"), h - mp.mpf("1e-4"), h,
                       h + mp.mpf("1e-4"), h + mp.mpf("1e-2"), mp.pi]) / mp.pi


if __name__ == "__main__":
    print("mean|cos|^1.5 =", K)
    c, eps, U = mp.mpf("0.01"), mp.mpf("0.01"), mp.mpf(10)
    print("4.3 ideal       c*U^1.5*K =", c * U ** 1.5 * K)
    print("4.3 regularised          =", p_reg(c, eps, U))
    for r in ("1e-2", "1e-3", "1e-4"):
        r = mp.mpf(r)
        d = (K - p_reg(1, r, 1)) / K
        print(f"eps/U = {float(r):.0e}: delta = {float(d):.6e}, (3/4)(eps/U)^2 = {float(0.75 * r * r):.6e}")
