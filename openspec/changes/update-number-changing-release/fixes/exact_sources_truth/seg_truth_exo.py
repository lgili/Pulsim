# ruff: noqa: E731, E741
"""50-digit truths for the C++ resonance / ramp cases (no pulsim):
x̄' = As x̄, truth = mp.quad of l·x̄(s), (l·x̄)², x̄ᵀQx̄, (x̄ᵀQx̄)² with x̄(s) = expm(As s) x̄0."""
import mpmath as mp
mp.mp.dps = 50
H = mp.mpf("1e-5")
W0 = mp.mpf(300000)
l = mp.matrix([1.5, -0.7, 0.4, -0.2, 0.3])
Q = mp.matrix([[1, .5, .25, 0, .1], [.5, -2, .1, .2, 0], [.25, .1, 3, 0, -.3], [0, .2, 0, 1, .05], [.1, 0, -.3, .05, .5]])
Q = mp.matrix([[mp.mpf(repr(float(Q[i, j]))) for j in range(5)] for i in range(5)])
l = mp.matrix([mp.mpf(repr(float(v))) for v in l])
x0 = mp.matrix([mp.mpf("0.2"), mp.mpf("-0.1"), mp.mpf("0.6"), mp.mpf("0.8"), 1])

def As_of(w, damp, osc=True):
    # lossless (or damped) LC [i, v] driven by the oscillator (s, c) at w
    A = mp.zeros(5, 5)
    A[0, 1], A[1, 0] = W0, -W0
    A[0, 0] = A[1, 1] = -damp
    if osc:
        A[0, 2] = mp.mpf(2e5)          # coupling of sin into i'
        A[1, 3] = mp.mpf(-1e5)         # of cos into v'
        A[2, 3], A[3, 2] = w, -w
    A[1, 4] = mp.mpf(5e4)              # DC
    return A

def ramp_As():
    # RC-like [x1, x2] driven by a level (state 2) with a slope (state 3): a Jordan block at 0
    A = mp.zeros(5, 5)
    A[0, 0], A[0, 1] = mp.mpf(-2e5), mp.mpf(1e5)
    A[1, 0], A[1, 1] = mp.mpf(5e4), mp.mpf(-3e5)
    A[0, 2] = mp.mpf(2e5)              # the level drives x1
    A[2, 3] = mp.mpf(1e5)              # level' = slope (slope state scaled by 1e5)
    A[1, 4] = mp.mpf(1e4)
    return A

def truth(As, s0, s1):
    x = lambda s: mp.expm(As * s) * x0
    y1 = lambda s: (l.T * x(s))[0]
    y2 = lambda s: (x(s).T * Q * x(s))[0]
    pts = mp.linspace(s0, s1, 9)
    return [mp.quad(y1, pts), mp.quad(lambda s: y1(s) ** 2, pts),
            mp.quad(y2, pts), mp.quad(lambda s: y2(s) ** 2, pts)]

cases = {
    "resonant": (As_of(W0, 0), 0, H),
    "detuned 1e-8": (As_of(W0 * (1 + mp.mpf("1e-8")), 0), 0, H),
    "detuned 1e-3": (As_of(W0 * (1 + mp.mpf("1e-3")), 0), 0, H),
    "far": (As_of(W0 * 3, 0), 0, H),
    "resonant, damped, piece": (As_of(W0, mp.mpf(1e3)), 0.25 * H, 0.9 * H),
    "ramp": (ramp_As(), 0, H),
    "ramp, piece": (ramp_As(), 0.3 * H, 0.7 * H),
}
for k, (As, s0, s1) in cases.items():
    t = truth(As, mp.mpf(s0), mp.mpf(s1))
    print(k, ", ".join(mp.nstr(v, 20) for v in t))
