# 50-digit truth for test_exact_route.py::test_ten_paralleled_bridges_first_mode_matches_a_50_digit_exponential
# (exact engine session 9). n paralleled three-phase bridges (sine 100 V, 50 Hz, phase 6 deg per bank,
# L = 0.2 mH, diodes g_on 1 kS / g_off 1 nS, bus C = 2 mF || R = 5 ohm, n tied to ground by R_g),
# the first mode (every bank: Dt_a, Db_b, Dt_c ON) held from rest. The KCL is written out by hand
# (n = R_g * sum(i)), not taken from the reduction under test; mpmath expm of the augmented system.
# usage: python3 ten_banks_50_digits.py N [t_end=1e-4] [log10 R_g = 6]
#   N = 10 prints: truth vc(1e-4) = 11.725547151164887... iL0a = 0.12972284951463489...
import sys, mpmath as mp
mp.mp.dps=50
n=int(sys.argv[1]); te=mp.mpf(sys.argv[2]) if len(sys.argv)>2 else mp.mpf("1e-4")
gon,goff=mp.mpf(1000),mp.mpf("1e-9"); L=mp.mpf("0.2e-3"); C=mp.mpf("2e-3"); R=mp.mpf(5); Rg=mp.mpf(10)**int(sys.argv[3]) if len(sys.argv)>3 else mp.mpf(10)**6
w=2*mp.pi*50
# S0 mode: per bank Dt a ON, Db b ON, Dt c ON
def mode_g(m):
    gt=[];gb=[]
    for k in range(n):
        for ph in range(3):
            gt.append(gon if m[k*6+ph*2] else goff); gb.append(gon if m[k*6+ph*2+1] else goff)
    return gt,gb
m=[1,0,0,1,1,0]*n
gt,gb=mode_g(m)
ang=[mp.radians(a+6*k) for k in range(n) for a in (0,-120,120)]
N3=3*n
def rhs(i, vc, vs):
    # solve n_node: inflow at n from p-side: ipn = C dvc + vc/R unknown; use KCL at p and n combined
    # x_k = (i_k + gt p + gb nn)/(gt+gb), p = nn+vc
    # KCL p: sum gt(x_k - p) = ipn ; KCL n: ipn = sum gb(nn - x_k) + nn/Rg
    # sum over both: sum gt(x-p) - sum gb(nn-x) - nn/Rg = 0 -> sum (gt+gb) x - gt p - gb nn ... = sum i_k (since (gt+gb)x - gt p - gb nn = i_k)
    # => sum i_k = nn/Rg  -> nn = Rg*sum i_k
    nn = Rg*mp.fsum(i); p = nn+vc
    x=[(i[k]+gt[k]*p+gb[k]*nn)/(gt[k]+gb[k]) for k in range(N3)]
    ipn = mp.fsum(gt[k]*(x[k]-p) for k in range(N3))
    dvc=(ipn - vc/R)/C
    di=[(vs[k]-x[k])/L for k in range(N3)]
    return di, dvc
# Build A (states: i_0..i_{N3-1}, vc) and Bs, Bc (sources 100 sin(wt+ang) = 100(sin wt cos a + cos wt sin a))
ns=N3+1
M=mp.zeros(ns+2)
def col(i,vc,vs):
    di,dvc=rhs(i,vc,vs); return di+[dvc]
z=[mp.mpf(0)]*N3
for j in range(ns):
    i=list(z); vc=mp.mpf(0)
    if j<N3: i[j]=mp.mpf(1)
    else: vc=mp.mpf(1)
    c=col(i,vc,z)
    for r_ in range(ns): M[r_,j]=c[r_]
cs=col(z,0,[100*mp.cos(a) for a in ang]); cc=col(z,0,[100*mp.sin(a) for a in ang])
for r_ in range(ns): M[r_,ns]=cs[r_]; M[r_,ns+1]=cc[r_]
M[ns,ns+1]=w; M[ns+1,ns]=-w
E=mp.expm(M*te); x0=mp.matrix([0]*ns+[0,1]); x=E*x0
print("truth vc(%s) ="%te, mp.nstr(x[ns-1],20), " iL0a =", mp.nstr(x[0],20))
