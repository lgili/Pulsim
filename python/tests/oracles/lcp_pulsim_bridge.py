"""Test-only bridge: the prototype's circuits and event states on pulsim's conductance form.

For each family of probes/lcp_cases.py this builds
  * the prototype circuit (ideal elements) and its exact LCP-driven run, recording every mode
    selection the prototype makes (held state y, time, gates, previous and selected mode), and
  * the same circuit as a pulsim CircuitBuilder with resistor-pair devices (g_on = 1e3,
    g_off = 1e-9, R_on = 1/g_on), and the full-MNA vector of a prototype state,
so the C++ port model + selector (adapter._mode_select_at) can be posed at the very states the
prototype decided, and its answer compared with the prototype's mode (task 3).

Also the prototype circuit with the conductance form's resistors made explicit (diode and switch
R_on = 1e-3, a 1e9 ohm leak across each), whose port LCP must equal pulsim's (task 3.1).
"""
import numpy as np

import pulsim as p

from . import lcp_reference as L

GON, GOFF = 1e3, 1e-9
T = 1e-5


class Family:
    """One circuit, described once and built both ways."""

    def __init__(self, name, parts, run):
        self.name, self.parts, self.run = name, parts, run

    def proto(self, conductance=False):
        c = L.Circuit()
        ron = 1.0 / GON if conductance else 0.0
        for kind, *a in self.parts:
            if kind == "V":
                c.add_Vdc(*a)
            elif kind == "Vsin":
                name, n1, n2, amp, f, ph = a
                c.add_Vsin(name, n1, n2, amp, f, phase=ph)
            elif kind == "I":
                c.add_Idc(*a)
            elif kind == "R":
                c.add_R(a[1], a[2], a[3])
            elif kind == "C":
                c.add_C(*a)
            elif kind == "L":
                c.add_L(*a)
            elif kind == "S":
                c.add_S(a[0], a[1], a[2], R_on=ron)
                if conductance:
                    c.add_R(a[1], a[2], 1.0 / GOFF)
            elif kind == "D":
                c.add_D(a[0], a[1], a[2], V_f=a[3], R_on=ron)
                if conductance:
                    c.add_R(a[1], a[2], 1.0 / GOFF)
        return c

    def builder(self, V_th=0.0):
        b = p.CircuitBuilder()
        def g(n):
            return "gnd" if n == "0" else n
        for kind, *a in self.parts:
            if kind == "V":
                b.add_voltage_source(a[0], g(a[1]), g(a[2]), a[3])
            elif kind == "Vsin":
                name, n1, n2, amp, f, ph = a
                b.add_sine_voltage_source(name, g(n1), g(n2), 0.0, amp, f, ph)
            elif kind == "I":       # prototype: I flows a -> b THROUGH the source
                b.add_current_source(a[0], g(a[2]), g(a[1]), a[3])
            elif kind == "R":
                b.add_resistor(a[0], g(a[1]), g(a[2]), a[3])
            elif kind == "C":
                b.add_capacitor(a[0], g(a[1]), g(a[2]), a[3])
            elif kind == "L":
                b.add_inductor(a[0], g(a[1]), g(a[2]), a[3])
            elif kind == "S":
                b.add_switch(a[0], g(a[1]), g(a[2]), GON, GOFF)
            elif kind == "D":
                b.add_diode(a[0], g(a[1]), g(a[2]), GON, GOFF, V_th, a[3])
        return b


def _buck_parts():
    return [("V", "Vin", "vin", "0", 24.0), ("S", "S", "vin", "sw"), ("D", "D1", "0", "sw", 0.0),
            ("L", "L", "sw", "vout", 100e-6), ("C", "C", "vout", "0", 47e-6), ("R", "R", "vout", "0", 50.0)]


def _boost_parts(R):
    return [("V", "Vin", "vin", "0", 12.0), ("L", "L", "vin", "sw", 100e-6), ("S", "S", "sw", "0"),
            ("D", "D1", "sw", "out", 0.0), ("C", "C", "out", "0", 47e-6), ("R", "R", "out", "0", R)]


def _bridge1_parts(Vf):
    return [("Vsin", "Vs", "a", "0", 100.0, 50.0, 0.0),
            ("D", "D1", "a", "p", Vf), ("D", "D3", "0", "p", Vf), ("D", "D2", "n", "a", Vf),
            ("D", "D4", "n", "0", Vf), ("C", "C", "p", "n", 1e-3), ("R", "R", "p", "n", 100.0)]


def _bridge3_parts(Ls=None, Id=None, C=None, R=None, Vf=0.0):
    out = []
    for ph, ang in (("a", 0.0), ("b", -2 * np.pi / 3), ("c", 2 * np.pi / 3)):
        src = "s" + ph if Ls else ph
        out += [("Vsin", "V" + ph, src, "0", 100.0, 50.0, ang),
                ("D", "Dt" + ph, ph, "p", Vf), ("D", "Db" + ph, "n", ph, Vf)]
    if Ls:
        out += [("L", "L" + ph, "s" + ph, ph, Ls) for ph in "abc"]
    if Id:
        out += [("I", "Id", "p", "n", Id)]
    if C:
        out += [("C", "C", "p", "n", C), ("R", "R", "p", "n", R)]
    return out


def _pwm_run(D, n, t0=0.0, h_e=1e-6 * T):
    def run(c):
        return L.simulate(c, t0 + n * T, gate=L.pwm(T, D, t0=t0), edges=L.pwm_edges(T, D, t0 + n * T, t0=t0),
                          h_e=h_e, confirm=False)
    return run


def _free_run(t_end, h_e=1e-9, x0=None):
    def run(c):
        y0 = None
        if x0 is not None:
            y0 = np.concatenate([np.asarray(x0, float), c.s_at(0.0)])
        return L.simulate(c, t_end, h_e=h_e, y0=y0, confirm=False)
    return run


FAMILIES = {
    "buck_rest": Family("buck_rest", _buck_parts(), _pwm_run(0.3, 10, t0=2e-6)),
    "dcm_buck": Family("dcm_buck", _buck_parts(), _pwm_run(0.3, 40)),
    "boost_50": Family("boost_50", _boost_parts(50.0), _pwm_run(0.5, 40)),
    "boost_500": Family("boost_500", _boost_parts(500.0), _pwm_run(0.5, 40)),
    "bridge1_vf07": Family("bridge1_vf07", _bridge1_parts(0.7), _free_run(0.04)),
    "bridge1_vf0": Family("bridge1_vf0", _bridge1_parts(0.0), _free_run(0.04)),
    "bridge3_overlap": Family("bridge3_overlap", _bridge3_parts(Ls=1e-3, Id=10.0),
                              _free_run(0.04, x0=[0.0, -10.0, 10.0])),
    "bridge3_cfilter": Family("bridge3_cfilter", _bridge3_parts(C=1e-3, R=20.0, Vf=0.7), _free_run(0.04)),
}
H_E = {"buck_rest": 1e-6 * T, "dcm_buck": 1e-6 * T, "boost_50": 1e-6 * T, "boost_500": 1e-6 * T}
# T_ref of design D5: the PWM period, or the mains period of the bridges
T_REF = {k: (T if k in H_E else 0.02) for k in FAMILIES}


def prototype_selections(fam):
    """Run the prototype; return [(t, y, sw, prev, selected)] for every LCP-decided event."""
    rec = []
    orig = L.select_modes

    def spy(c, y, t, sw_on, prev_on, h, widen=(100.0, 1e4)):
        d, info = orig(c, y, t, sw_on, prev_on, h, widen)
        rec.append((t, y.copy(), tuple(sw_on), tuple(prev_on), tuple(d)))
        return d, info

    L.select_modes = spy
    try:
        c = fam.proto()
        fam.run(c)
    finally:
        L.select_modes = orig
    return c, rec


class PulsimPorts:
    """pulsim side of one family: adapter + full-MNA mapping of a prototype state."""

    def __init__(self, fam, V_th=0.0):
        self.b = b = fam.builder(V_th)
        self.cache = p.PwlStateSpaceCache(b.graph, b.pool)
        self.cache.build_lazy(1e-5)
        self.ad = p._pulsim._NativeCircuitBuilderAdapter(b, self.cache)
        self.n = b.pool.state_size(b.graph)
        def g(nm):
            return -1 if nm == "0" else b.node_id_of(nm)
        self.caps = [(g(a[1]), g(a[2])) for k, *a in fam.parts if k == "C"]
        self.inds = [b.pool.branch_var_id_for_inductor(b.branch_id_of(a[0]), b.graph)
                     for k, *a in fam.parts if k == "L"]
        sw_names = [a[0] for k, *a in fam.parts if k == "S"]
        d_names = [a[0] for k, *a in fam.parts if k == "D"]
        self.sw_idx = [b.switch_index_of(nm) for nm in sw_names]
        self.d_idx = [b.switch_index_of(nm) for nm in d_names]
        self.n_nodes = b.graph.num_nodes

    def x_full(self, y):
        x = np.zeros(self.n)
        if self.caps:                        # node voltages with v_a - v_b = v_C (least squares)
            A = np.zeros((len(self.caps), self.n_nodes))
            for r, (a, k) in enumerate(self.caps):
                if a >= 0:
                    A[r, a] += 1.0
                if k >= 0:
                    A[r, k] -= 1.0
            x[:self.n_nodes] = np.linalg.lstsq(A, y[:len(self.caps)], rcond=None)[0]
        for j, row in enumerate(self.inds):
            x[row] = y[len(self.caps) + j]
        return x

    def mask(self, sw, d):
        m = p.SwitchStateMask(self.b.graph.num_switches)
        for i, on in zip(self.sw_idx, sw):
            m.set(i, bool(on))
        for i, on in zip(self.d_idx, d):
            m.set(i, bool(on))
        return m

    def select(self, t, y, sw, prev, h_e, tol=1e-9):
        m = self.mask(sw, prev)
        self.ad.set_mask(m)
        return self.ad._mode_select_at(m, t, self.x_full(y), h_e, tol)
