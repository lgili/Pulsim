import numpy as np
exec(open("esdirk43_lc.py").read().split("# ---- generic stiffly-accurate")[0].split("An = np.array")[0])
An = np.array([[float(x) for x in r] for r in A]); bn = np.array([float(x) for x in b])
gm = 2-np.sqrt(2); d = gm/2; w = np.sqrt(2)/4
At = np.array([[0, 0, 0], [d, d, 0], [w, w, d]]); bt = At[-1]
def R(z, A, b): return 1 + z*b@np.linalg.solve(np.eye(len(b)) - z*A, np.ones(len(b)))
print("steps/period | TR-BDF2 |R|-1, phase err | ESDIRK4(3) |R|-1, phase err   (per step)")
for N in [100, 50, 20, 10]:
    th = 2*np.pi/N; r1 = R(1j*th, At, bt); r2 = R(1j*th, An, bn)
    print(f"{N:4d} | {abs(r1)-1:+.2e} {np.angle(r1)-th:+.2e} | {abs(r2)-1:+.2e} {np.angle(r2)-th:+.2e}")
# trapezoidal for reference: |R| = 1 exactly, phase err = 2*atan(th/2) - th
for N in [100, 50, 20, 10]:
    th = 2*np.pi/N; print(f"trap N={N}: phase err {2*np.arctan(th/2)-th:+.2e}")
