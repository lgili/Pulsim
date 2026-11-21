"""Which engine is right on an ideal circuit with dependent states: the
R_on -> 0 limit of the branch form (no dependent states for R_on > 0)."""
import contextlib, sys
sys.path[:0] = ["benchmarks/tools", "benchmarks"]
import differential_runner as dr, ideal_switch_flip_evidence as ev, reference_status as rs, v1_reader, pulsim
from pathlib import Path

@contextlib.contextmanager
def small_ron(r):
    cb = pulsim.CircuitBuilder
    sw, di = cb.add_switch, cb.add_diode
    def add_switch(self, name, a, b, g_on=1e3, g_off=1e-9, *args, **kw):
        if args or kw: return sw(self, name, a, b, g_on, g_off, *args, **kw)
        return self.add_ideal_switch(name, a, b, r, g_off)
    def add_diode(self, name, a, c, g_on=None, g_off=None, V_th=0.0, V_f=0.0, *args, **kw):
        if kw.get("form", "conductance") != "conductance" or args:
            return di(self, name, a, c, g_on, g_off, V_th, V_f, *args, **kw)
        kw.pop("form", None)
        return di(self, name, a, c, None, 1e-9 if g_off is None else g_off, V_th, 0.0, form="branch", R_on=r, **kw)
    cb.add_switch, cb.add_diode = add_switch, add_diode
    try: yield
    finally: cb.add_switch, cb.add_diode = sw, di

rel = sys.argv[1]
case = v1_reader.load(Path("benchmarks") / rel)
cols = dr.observables_of(case)
with ev.ideal_form():
    R1 = dr.traces(case.run("trbdf2", rtol=1e-8, atol=1e-12), cols)
    with rs.extra_kwargs({"discontinuity_restart": True}):
        R2 = dr.traces(case.run("pwl", dt=case.dt / 16), cols)
print("ideal: pwl+restart dt/16 vs trbdf2 1e-8:", dr.compare(R2, R1, cols, 1e-3)[0])
for r in (1e-3, 1e-4, 1e-5, 1e-6):
    with small_ron(r):
        try:
            T = dr.traces(case.run("trbdf2", rtol=1e-8, atol=1e-12), cols)
            et = (dr.compare(T, R1, cols, 1e-3)[0], dr.compare(T, R2, cols, 1e-3)[0])
        except Exception as e:
            et = str(e)[:80]
        try:
            P = dr.traces(case.run("pwl", dt=case.dt / 16), cols)
            ep = (dr.compare(P, R1, cols, 1e-3)[0], dr.compare(P, R2, cols, 1e-3)[0])
        except Exception as e:
            ep = str(e)[:80]
    print(f"R_on={r:g}: trbdf2 vs (R1 trbdf2-ideal, R2 pwl-ideal) = {et}; pwl dt/16 vs (R1, R2) = {ep}", flush=True)
