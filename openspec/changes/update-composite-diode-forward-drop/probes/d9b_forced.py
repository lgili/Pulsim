"""D9b: forced 10 A source->drain through the composites, gate OFF and gate ON.
today = shipped composite; future = add_switch + add_diode(V_th=0, V_f=0.7) (PWL) emulation."""
import warnings, numpy as np, pulsim as p
warnings.simplefilter("ignore")
def run(build, gate_on, sw_name="Q"):
    b = p.CircuitBuilder(); b.add_current_source("I", "s", "gnd", 10.0); b.add_resistor("Rb", "s", "gnd", 1e6)
    build(b); n = b.graph.num_switches
    m = p.SwitchStateMask(n)
    if gate_on: m.set(b.switch_index_of(sw_name), True)
    r = p.simulate(b, t_end=1e-4, dt=1e-6, engine="pwl", switch_fn=lambda t: m)
    v = abs(float(np.asarray(r.v("s"))[-1]))
    rows = {e["name"]: round(e.get("P_avg", 0.0), 4) for e in p.device_loss_summary(b, r)}
    return v, rows
today  = lambda b: b.add_mosfet("Q", "gnd", "s", 1e-3, 1e9, True, 0.7)
def future(b):
    b.add_switch("Q", "gnd", "s", 1e3, 1e-9); b.add_diode("Q_body", "s", "gnd", 1e3, 1e-9, 0.0, 0.7)
for tag, f in (("today ", today), ("future", future)):
    for g in (False, True):
        v, rows = run(f, g)
        print(f"add_mosfet {tag} gate={'ON ' if g else 'OFF'} |v_SD|={v:.4f} V  P={v*10:.3f} W  {rows}")
# Level-1 MOSFET body diode (hard-coded g_on=1e3, V_th=0.5), gate OFF: drain=gnd, source=s
def l1(b):
    b.add_voltage_source("Vg", "g", "s", 0.0); b.add_mosfet_level1("M", "gnd", "s", "g", 5.0, 3.0)
b = p.CircuitBuilder(); b.add_current_source("I", "s", "gnd", 10.0); b.add_resistor("Rb", "s", "gnd", 1e6); l1(b)
n = b.graph.num_switches
try:
    r = p.simulate(b, t_end=1e-4, dt=1e-6, engine="pwl")
    print(f"add_mosfet_level1 today gate OFF |v_SD|={abs(float(np.asarray(r.v('s'))[-1])):.4f} V  switches={n}")
except Exception as e: print("level1", type(e).__name__, str(e)[:160])
