"""Why ll16_flyback_closed_loop is wrong from t = 0 (found 2026-11-13).

The v1 reader starts ll16 from compute_dc_op. A voltage source across
two series inductors (Vin - Lleak - Lm, as in ll16's primary) has no DC
operating point: the inductors are shorts. compute_dc_op returns one
anyway, with the inductor current set by the source over the inductors'
DC regularisation (about 2e-12 ohm): 2.4e13 A. ll16's transient then
starts from that state on every engine. Minimal reproducer:

    python ll16_dc_op.py
"""
import numpy as np

import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("V", "vin", "gnd", 48.0)
b.add_inductor("L1", "vin", "d", 2e-6)
b.add_inductor("L2", "d", "gnd", 158e-6)
x = np.asarray(p.compute_dc_op(b))
print("max |x| =", float(np.abs(x).max()), "(expected: a refusal)")
