"""Stage-2 fallback probe: 3-phase diode rectifier, 24 V-peak phases,
1 mF bus from 0 V (tau = 5 us). Prints bus peak and max |error| vs a
tight trbdf2 reference, flag off/on, at dt/tau = 10 and 1.
classic=True: low side on a floating rail (bus = line-to-line).
Run: PYTHONPATH=python python3 rectifier_stage2.py
"""
import warnings

import numpy as np
import pulsim as p

warnings.simplefilter("ignore")
def build(classic):
    b=p.CircuitBuilder()
    for n,ph in (("a",0.0),("b",2.0943951),("c",4.1887902)):
        b.add_sine_voltage_source("V"+n,n,"gnd",0.0,24.0,100.0,ph)
        b.add_diode("D"+n+"h",n,"out",200.0,1e-6)
        b.add_diode("D"+n+"l","neg" if classic else "gnd",n,200.0,1e-6)
    lo = "neg" if classic else "gnd"
    b.add_capacitor("Cout","out",lo,1e-3)
    b.add_resistor("Rload","out",lo,200.0)
    if classic: b.add_resistor("Rref","neg","gnd",1e6)
    return b
for classic in (False, True):
    ref=p.simulate(build(classic),t_end=40e-3,engine="trbdf2",rtol=1e-8,atol=1e-10)
    tr=np.asarray(ref.times); vr=np.asarray(ref.v("out"))-(np.asarray(ref.v("neg")) if classic else 0)
    for dt in (50e-6,5e-6):
        for kw in ({},dict(discontinuity_restart=True)):
            r=p.simulate(build(classic),t_end=40e-3,dt=dt,engine="pwl",**kw)
            t=np.asarray(r.times); v=np.asarray(r.v("out"))-(np.asarray(r.v("neg")) if classic else 0)
            print(classic, dt, bool(kw), "max %.4f"%v.max(), "err %.4f"%np.abs(v-np.interp(t,tr,vr)).max(), "ref max %.4f"%vr.max())
