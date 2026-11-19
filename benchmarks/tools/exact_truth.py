"""Exact reference for piecewise-linear switched benchmark circuits.

Independent of pulsim (never imports it): parses a ``schema: pulsim-v1``
benchmark YAML with PyYAML and solves it to working precision with mpmath,
generalising ``bridge_ground_truth.py``.

Model (the simulator's own laws, ``models/switched_diode.hpp`` and the
branch form of ``builder/circuit_builder.hpp``):

* ``form='conductance'`` (the netlist as written): a switch / diode stamps
  g_on when ON and g_off when OFF; a diode ON carries g_on (v - V_f).
  Diode: ON -> OFF iff g_on (v - V_f) <= 0, OFF -> ON iff v >= V_on.
* ``form='ideal'`` (the ideal twin of ``ideal_switch_flip_evidence.py``):
  R_on = 0 -- an ON switch / diode is a 0 V branch with its own current,
  OFF keeps the netlist's g_off; a diode turns off when its branch current
  is <= 0 and on when v >= V_on (V_f = 0).

Method, per mode (switch and diode states): MNA  E x' = A x + B w(t), w the
source exosystem [1, sin wt, cos wt, ...], w' = S w. E, A are exact
rationals (the engine's doubles, exactly). With M = lam E - A, Eh = M^-1 E
is split EXACTLY (fractions) into its core and nilpotent parts,
T^-1 Eh T = diag(Cs, N): the slow coordinates z' = F z + G w carry the
state; the fast ones are a fixed function of w (no floating point in any
rank decision). A mode change keeps z = Cs^-1 [T^-1 M^-1 (E x)]_top, i.e.
it depends only on the charges and fluxes E x: the distributional solution
(charge / flux conservation), the impulsive energy is reported. The
augmented y = [z; w] is propagated exactly through the eigen-decomposition
of H = [[F, G], [0, S]] (mpmath), so every observable is a sum of
exponentials; window integrals (means, RMS, energies) are exact sums.
Events: gate / source edges exactly from the schedule (fractions); diode
commutations by bracketing on a scan and bisection to ~1e-(dps-12)
relative; at each event the candidate modes (devices at or past their
boundary, closed under the flips they force) are enumerated and the
forward-consistent ones (leading Taylor sign of each device's signal)
counted. Several: reported, the one with fewest flips is followed (a
``forced`` map follows another).

    uv run --no-project --with mpmath --with numpy --with pyyaml python \\
        benchmarks/tools/exact_truth.py solve <circuit.yaml> [--form ideal]
    ... exact_truth.py validate          # the closed-form validation table
    ... exact_truth.py compare <circuit.yaml> <trace.csv>   # harness metric
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import itertools
import json
import math
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import mpmath as mp
import yaml

GROUND = {"0", "gnd", "GND"}
CONTROL = {"pi_controller", "pwm_generator", "pll", "park_transform", "clarke_transform",
           "gain", "svm", "sample_hold", "rate_limiter"}


class Unsupported(Exception):
    """A construct the exact tool does not model (named in the message)."""


# ------------------------------------------------------------------ parsing --

def value(raw) -> float:
    """v1 numeric literal -> the double the harness passes to the engine."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    for suf, mul in (("meg", 1e6), ("f", 1e-15), ("p", 1e-12), ("n", 1e-9), ("u", 1e-6),
                     ("µ", 1e-6), ("m", 1e-3), ("k", 1e3), ("g", 1e9)):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * mul
    return float(s)


def node(n) -> str:
    s = str(n)
    return "gnd" if s in GROUND else s


def fr_of(t) -> Fr:
    """Exact Fraction of an mpf (or pass a Fraction through)."""
    if isinstance(t, Fr):
        return t
    sign, man, exp, _ = mp.mpf(t)._mpf_
    q = Fr(int(man)) * Fr(2) ** exp
    return -q if sign else q


def mpf(q):
    """Fraction / float / int -> mpf at the current precision."""
    if isinstance(q, Fr):
        return mp.mpf(q.numerator) / q.denominator
    return mp.mpf(q)


def on_windows(w: dict, thr: float):
    """v1 gate semantics (same arithmetic as v1_reader._on_windows): True /
    False, or (f, duty, start) with ON while frac((t - start) f) < duty."""
    kind = w.get("type")
    if kind == "dc":
        return value(w["value"]) > thr
    if kind == "pwm":
        hi, lo = value(w.get("v_high", 1.0)), value(w.get("v_low", 0.0))
        f = value(w.get("frequency", 10e3))
        T = 1.0 / f
        t_on = min(max(value(w.get("duty", 0.5)) * T - value(w.get("dead_time", 0.0)), 0.0), T)
        start = (-value(w.get("phase", 0.0)) * T) % T
        if (hi > thr) == (lo > thr):
            return hi > thr
        if hi > thr:
            return f, t_on / T, start
        return f, 1.0 - t_on / T, (start + t_on) % T
    if kind == "pulse":
        v0, v1 = value(w.get("v_initial", 0.0)), value(w.get("v_pulse", 1.0))
        td, tr, tf = (value(w.get(k, 0.0)) for k in ("t_delay", "t_rise", "t_fall"))
        pw, per = value(w.get("t_width", 0.0)), value(w.get("period", 0.0))
        if (v0 > thr) == (v1 > thr):
            return v0 > thr
        if v0 > thr or per <= 0.0:
            raise Unsupported("pulse gate that starts high or does not repeat")
        x = (thr - v0) / (v1 - v0)
        on, off = td + tr * x, td + tr + pw + tf * (1.0 - x)
        return 1.0 / per, (off - on) / per, on % per
    raise Unsupported(f"gate waveform {kind!r}")


class Schedule:
    """ON while frac((t - start)/T) < duty, exact in fractions (T = the
    engine's double 1/f)."""

    def __init__(self, spec):
        self.const = spec if isinstance(spec, bool) else None
        if self.const is None:
            f, duty, start = spec
            self.T, self.duty, self.start = Fr(1.0 / f), Fr(duty), Fr(start)

    def state(self, t: Fr) -> bool:
        if self.const is not None:
            return self.const
        x = (t - self.start) / self.T
        return x - math.floor(x) < self.duty

    def next_edge(self, t: Fr):
        if self.const is not None or self.duty <= 0 or self.duty >= 1:
            return None
        k = math.floor((t - self.start) / self.T)
        c = [self.start + (k + d) * self.T for d in (self.duty, 1, 1 + self.duty)]
        return min(x for x in c if x > t)


class Wave:
    """A source waveform: value(t) and its piece (const, sines) at t."""

    def __init__(self, w: dict):
        self.kind = w.get("type")
        self.w = w
        if self.kind == "dc":
            self.dc = Fr(value(w["value"]))
        elif self.kind == "sine":
            self.off, self.amp = Fr(value(w.get("offset", 0.0))), value(w["amplitude"])
            self.f, self.ph = value(w["frequency"]), value(w.get("phase", 0.0))
        elif self.kind == "pwm":
            spec = on_windows(dict(w, v_high=1.0, v_low=0.0), 0.5)
            self.hi, self.lo = Fr(value(w.get("v_high", 1.0))), Fr(value(w.get("v_low", 0.0)))
            self.sch = Schedule(spec)
        elif self.kind == "pulse":
            g = {k: Fr(value(w.get(k, d))) for k, d in (
                ("v_initial", 0.0), ("v_pulse", 1.0), ("t_delay", 0.0), ("t_rise", 0.0),
                ("t_fall", 0.0), ("t_width", 0.0), ("period", 0.0))}
            self.p = g
        else:
            raise Unsupported(f"voltage_source waveform {self.kind!r}")

    def freqs(self):
        return [self.f] if self.kind == "sine" else []

    def _pulse_pos(self, t: Fr):
        p = self.p
        if t < p["t_delay"]:
            return None
        e = t - p["t_delay"]
        if p["period"] > 0:
            e -= math.floor(e / p["period"]) * p["period"]
        return e

    def ramps(self) -> bool:
        return self.kind == "pulse" and (self.p["t_rise"] > 0 or self.p["t_fall"] > 0)

    def piece_at(self, t: Fr):
        """(c0, c1): the value is c0 + c1 t on [t, next break) (right limit)."""
        if self.kind != "pulse" or not self.ramps():
            return self.const_at(t), Fr(0)
        p, e = self.p, self._pulse_pos(t)
        v0, v1, tr, pw, tf = (p[k] for k in ("v_initial", "v_pulse", "t_rise", "t_width", "t_fall"))
        if e is not None and e < tr:
            sl = (v1 - v0) / tr
            return v0 + sl * e - sl * t, sl
        if e is not None and tr + pw <= e < tr + pw + tf:
            sl = (v0 - v1) / tf
            return v1 + sl * (e - tr - pw) - sl * t, sl
        return self.const_at(t), Fr(0)

    def const_at(self, t: Fr) -> Fr:
        """Piecewise-constant value on [t, next break) (right limit)."""
        if self.kind == "dc":
            return self.dc
        if self.kind == "sine":
            return self.off
        if self.kind == "pwm":
            return self.hi if self.sch.state(t) else self.lo
        p, e = self.p, self._pulse_pos(t)
        if e is None or e >= p["t_rise"] + p["t_width"]:
            return p["v_initial"]
        return p["v_pulse"]

    def value(self, t, side=1):
        """Exact value (mp) at mp time t; side=-1 the left limit at a break."""
        if self.kind == "sine":
            return mpf(self.off) + self.amp * mp.sin(2 * mp.pi * self.f * t + self.ph)
        if self.kind == "dc":
            return mpf(self.dc)
        if self.kind == "pwm":
            return mpf(self._state_side(t, side))
        p = self.p
        tt = fr_of(t)
        e = self._pulse_pos(tt)
        if side < 0 and e is not None and e == 0 and tt > p["t_delay"]:
            e = p["period"]
        if e is None:
            return mpf(p["v_initial"])
        v0, v1, tr, pw, tf = (p[k] for k in ("v_initial", "v_pulse", "t_rise", "t_width", "t_fall"))
        if e < tr:
            return mpf(v0 + (v1 - v0) * e / tr)
        if e < tr + pw:
            return mpf(v1)
        if e < tr + pw + tf:
            return mpf(v1 + (v0 - v1) * (e - tr - pw) / tf)
        return mpf(v0)

    def _state_side(self, t, side):
        tt = fr_of(t)
        # an event time is the edge rounded to dps digits: snap it, or the right
        # limit reads the old level (the stored gate trace lost its edges)
        eps = Fr(1, 10 ** (mp.mp.dps - 5)) * max(abs(tt), Fr(1, 10 ** 9))
        e = self.sch.next_edge(tt - eps)
        if e is not None and abs(e - tt) <= eps:
            tt = e
        if side < 0:
            tt -= Fr(1, 10 ** (mp.mp.dps + 20))
        return self.hi if self.sch.state(tt) else self.lo

    def next_break(self, t: Fr):
        if self.kind == "pwm":
            return self.sch.next_edge(t)
        if self.kind != "pulse":
            return None
        p = self.p
        ends = [Fr(0), p["t_rise"], p["t_rise"] + p["t_width"],
                p["t_rise"] + p["t_width"] + p["t_fall"]]
        if t < p["t_delay"]:
            return p["t_delay"]
        if p["period"] <= 0:
            c = [p["t_delay"] + x for x in ends if p["t_delay"] + x > t]
            return min(c) if c else None
        k = math.floor((t - p["t_delay"]) / p["period"])
        c = [p["t_delay"] + (k + j) * p["period"] + x for j in (0, 1) for x in ends]
        return min(x for x in c if x > t)


class Circuit:
    """A benchmark netlist mapped to MNA (exact rationals)."""

    def __init__(self, path, form="conductance"):
        if isinstance(path, dict):          # an in-memory netlist (validation cases)
            d, self.path = path, Path(path["benchmark"]["id"] + ".yaml")
        else:
            self.path = Path(path)
            d = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if d.get("schema") != "pulsim-v1":
            raise Unsupported(f"schema {d.get('schema')!r}")
        sim, bm = d.get("simulation") or {}, d.get("benchmark") or {}
        self.id = bm.get("id", self.path.stem)
        self.form = form
        self.t_end = Fr(value(sim["tstop"]))
        self.dt = value(sim["dt"])
        if value(sim.get("tstart", 0.0)) != 0.0:
            raise Unsupported("tstart != 0")
        self.uic = bool(sim.get("uic", False))
        comps = d.get("components") or []
        obs = [o["column"] for o in bm.get("observables", []) if "column" in o]
        val = (bm.get("validation") or {}).get("observable")
        self._build(comps)
        self.observables = obs or ([val] if val else self.default_observables())

    # -- netlist ------------------------------------------------------------
    def _build(self, comps):
        for c in comps:
            t = c["type"]
            if t in CONTROL:
                raise Unsupported(f"closed loop / control block ({t})")
            if t == "diode" and "is" in c:
                raise Unsupported(f"Shockley diode {c['name']}")
            if t in ("mosfet", "igbt"):
                raise Unsupported(f"{t} (Newton device) {c['name']}")
            if t == "saturable_inductor":
                raise Unsupported("saturable_inductor")
            if t not in ("resistor", "capacitor", "inductor", "voltage_source", "current_source",
                         "diode", "switch", "vcswitch", "transformer", "coupled_inductor",
                         "snubber_rc"):
                raise Unsupported(f"component type {t!r}")
        # gate-only nodes: a source to ground whose node feeds only vcswitch controls
        pins = {}
        for c in comps:
            ns = [node(n) for n in c.get("nodes", [])]
            for i, n in enumerate(ns):
                role = "ctrl" if c["type"] == "vcswitch" and i == 0 else c["type"]
                pins.setdefault(n, []).append((role, c["name"]))
        self.gate_src = {}
        for c in comps:
            if c["type"] == "voltage_source":
                a, k = (node(n) for n in c["nodes"])
                if k == "gnd" and a != "gnd" and all(r == "ctrl" or nm == c["name"] for r, nm in pins[a]):
                    self.gate_src[a] = c
        nodes = []
        for c in comps:
            for n in c.get("nodes", []):
                n = node(n)
                if c["type"] == "vcswitch" and n == node(c["nodes"][0]):
                    pass
                if n != "gnd" and n not in nodes and n not in self.gate_src:
                    nodes.append(n)
        self.res, self.caps, self.inds, self.mut, self.vsrc, self.isrc = [], [], [], [], [], []
        self.xfmr, self.devs = [], []
        self.waves = {}
        for c in comps:
            t, nm = c["type"], c["name"]
            ns = [node(n) for n in c.get("nodes", [])]
            g_on = c.get("g_on")
            if g_on is None and c.get("ron") is not None:
                g_on = 1.0 / value(c["ron"])
            g_off = c.get("g_off")
            if g_off is None and c.get("roff") is not None:
                g_off = 1.0 / value(c["roff"])
            if t == "resistor":
                self.res.append((nm, ns[0], ns[1], Fr(1.0 / value(c.get("value", c.get("resistance"))))))
            elif t == "capacitor":
                self.caps.append((nm, ns[0], ns[1], Fr(value(c.get("value", c.get("capacitance")))),
                                  Fr(value(c.get("ic", 0.0)))))
            elif t == "inductor":
                self.inds.append((nm, ns[0], ns[1], Fr(value(c.get("value", c.get("inductance")))),
                                  Fr(value(c.get("ic", 0.0)))))
            elif t == "coupled_inductor":
                l1 = value(c.get("l1", 1e-3))
                l2, k = value(c.get("l2", l1)), value(c.get("coupling", 0.98))
                i1 = len(self.inds)
                self.inds.append((nm + ".p", ns[0], ns[1], Fr(l1), Fr(0)))
                self.inds.append((nm + ".s", ns[2], ns[3], Fr(l2), Fr(0)))
                self.mut.append((i1, i1 + 1, Fr(k * math.sqrt(l1 * l2))))
            elif t == "snubber_rc":
                mid = f"{nm}__mid"
                nodes.append(mid)
                self.res.append((nm + "__R", ns[0], mid, Fr(1.0 / value(c["resistance"]))))
                self.caps.append((nm + "__C", mid, ns[1], Fr(value(c["capacitance"])),
                                  Fr(value(c.get("ic", 0.0)))))
            elif t == "voltage_source":
                wv = Wave(c["waveform"])
                if ns[0] in self.gate_src and self.gate_src[ns[0]] is c:
                    self.waves[ns[0]] = wv
                    continue
                self.vsrc.append((nm, ns[0], ns[1], wv))
            elif t == "current_source":
                self.isrc.append((nm, ns[0], ns[1], Fr(value(c["value"]))))
            elif t == "transformer":
                n = Fr(1.0 / value(c.get("turns_ratio", c.get("ratio"))))
                self.xfmr.append((nm, *ns[:4], n))
            elif t == "diode":
                self.devs.append(dict(name=nm, a=ns[0], k=ns[1], kind="D",
                                      g_on=Fr(value(g_on if g_on is not None else 1e3)),
                                      g_off=Fr(value(g_off if g_off is not None else 1e-9)),
                                      V_th=Fr(0), V_f=Fr(0)))
            elif t in ("switch", "vcswitch"):
                pos = ns[1:] if t == "vcswitch" else ns
                gon = value(g_on if g_on is not None else (1e3 if t == "vcswitch" else 1e6))
                goff = value(g_off if g_off is not None else (1e-9 if t == "vcswitch" else 1e-12))
                if t == "switch":
                    spec = bool(c.get("initial_state", c.get("initial_closed", False)))
                else:
                    src = self.gate_src.get(ns[0])
                    if src is None:
                        raise Unsupported(f"vcswitch {nm} control node driven by the circuit")
                    spec = on_windows(src["waveform"], value(c.get("v_threshold", 2.5)))
                self.devs.append(dict(name=nm, a=pos[0], k=pos[1], kind="S", g_on=Fr(gon),
                                      g_off=Fr(goff), sched=Schedule(spec)))
        self.nodes = nodes
        self.ideal = self.form == "ideal"
        # unknowns: nodes, inductor currents, vsource currents, transformer (ip, is), ideal branches
        self.ix = {f"V({n})": i for i, n in enumerate(nodes)}
        k = len(nodes)
        for nm, *_ in self.inds:
            self.ix[f"I({nm})"] = k
            k += 1
        for nm, *_ in self.vsrc:
            self.ix[f"Is({nm})"] = k
            k += 1
        for nm, *_ in self.xfmr:
            self.ix[f"Ip({nm})"], self.ix[f"Is({nm})"] = k, k + 1
            k += 2
        if self.ideal:
            for dv in self.devs:
                dv["ib"] = k
                self.ix[f"I({dv['name']})"] = k
                k += 1
        self.n = k
        self.diodes = [i for i, dv in enumerate(self.devs) if dv["kind"] == "D"]
        self.switches = [i for i, dv in enumerate(self.devs) if dv["kind"] == "S"]
        fs = sorted({f for _, _, _, wv in self.vsrc for f in wv.freqs()})
        self.freqs = fs
        # exosystem w = [1, sin w1 t, cos w1 t, ..., (t if a source ramps)]
        self.ramp = any(wv.ramps() for _, _, _, wv in self.vsrc)
        self.m = 1 + 2 * len(fs) + (1 if self.ramp else 0)
        self.S = mp.zeros(self.m, self.m)
        for j, f in enumerate(fs):
            w = 2 * mp.pi * f
            self.S[1 + 2 * j, 2 + 2 * j], self.S[2 + 2 * j, 1 + 2 * j] = w, -w
        if self.ramp:
            self.S[self.m - 1, 0] = 1
        self.E = self._E()

    def default_observables(self):
        """What differential_runner.observables_of reads off the builder:
        every V(node) (gate nodes included) and I(inductor)."""
        out = [f"V({n})" for n in self.nodes] + [f"V({n})" for n in self.gate_src
                                                   if n in self.waves]
        out += [f"I({nm})" for nm, *_ in self.inds if "." not in nm]
        return out

    def _vi(self, n):
        return None if n == "gnd" else self.ix[f"V({n})"]

    def _E(self):
        E = [[Fr(0)] * self.n for _ in range(self.n)]
        for _, a, b, C, _ic in self.caps:
            for r, sr in ((a, 1), (b, -1)):
                for c, sc in ((a, 1), (b, -1)):
                    if r != "gnd" and c != "gnd":
                        E[self._vi(r)][self._vi(c)] += sr * sc * C
        for nm, _a, _b, L, _ in self.inds:
            i = self.ix[f"I({nm})"]
            E[i][i] += L
        for p, s, M in self.mut:
            ip, is_ = (self.ix[f"I({self.inds[q][0]})"] for q in (p, s))
            E[ip][is_] += M
            E[is_][ip] += M
        return E

    def charges(self):
        """E x0 for the netlist's initial conditions (charges, fluxes)."""
        q = [Fr(0)] * self.n
        for _, a, b, C, ic in self.caps:
            if a != "gnd":
                q[self._vi(a)] += C * ic
            if b != "gnd":
                q[self._vi(b)] -= C * ic
        for p, (nm, _a, _b, L, ic) in enumerate(self.inds):
            q[self.ix[f"I({nm})"]] += L * ic
        for p, s, M in self.mut:
            q[self.ix[f"I({self.inds[p][0]})"]] += M * self.inds[s][4]
            q[self.ix[f"I({self.inds[s][0]})"]] += M * self.inds[p][4]
        return q

    # -- a mode ----------------------------------------------------------------
    def stamp(self, mode):
        """(A rational n x n, B mp n x m builder) for mode = tuple of device bits.
        Rows: KCL (current leaving the node) at node rows, then branch rows."""
        n = self.n
        A = [[Fr(0)] * n for _ in range(n)]
        Bc = [dict() for _ in range(n)]       # row -> {w index: Fraction or mp}

        def cond(a, b, g):
            for r, sr in ((a, 1), (b, -1)):
                for c, sc in ((a, 1), (b, -1)):
                    if r != "gnd" and c != "gnd":
                        A[self._vi(r)][self._vi(c)] -= sr * sc * g

        def branch(a, b, col):           # current col leaves a, enters b
            if a != "gnd":
                A[self._vi(a)][col] -= 1
            if b != "gnd":
                A[self._vi(b)][col] += 1

        def vrow(row, a, b, s=1):
            if a != "gnd":
                A[row][self._vi(a)] += s
            if b != "gnd":
                A[row][self._vi(b)] -= s

        def addB(row, j, v):
            Bc[row][j] = Bc[row].get(j, 0) + v

        for _, a, b, g in self.res:
            cond(a, b, g)
        for nm, a, b, _L, _ in self.inds:
            i = self.ix[f"I({nm})"]
            branch(a, b, i)
            vrow(i, a, b)
        for nm, a, b, wv in self.vsrc:
            i = self.ix[f"Is({nm})"]
            branch(a, b, i)
            vrow(i, a, b)
            c0, c1 = wv.piece_at(self._tsrc)
            addB(i, 0, -c0)
            if c1:
                addB(i, self.m - 1, -c1)
            if wv.kind == "sine":
                j = 1 + 2 * self.freqs.index(wv.f)
                addB(i, j, -mp.mpf(wv.amp) * mp.cos(wv.ph))
                addB(i, j + 1, -mp.mpf(wv.amp) * mp.sin(wv.ph))
        for _, a, b, cur in self.isrc:        # cur flows out of a into the circuit
            if a != "gnd":
                addB(self._vi(a), 0, cur)
            if b != "gnd":
                addB(self._vi(b), 0, -cur)
        for nm, p1, p2, s1, s2, nr in self.xfmr:
            ip, is_ = self.ix[f"Ip({nm})"], self.ix[f"Is({nm})"]
            branch(p1, p2, ip)
            branch(s1, s2, is_)
            vrow(ip, s1, s2)              # v_s - n v_p = 0
            vrow(ip, p1, p2, -nr)
            A[is_][ip] += 1               # i_p + n i_s = 0
            A[is_][is_] += nr
        for dv, on in zip(self.devs, mode):
            a, k = dv["a"], dv["k"]
            if not self.ideal:
                g = dv["g_on"] if on else dv["g_off"]
                cond(a, k, g)
                if on and dv.get("V_f"):
                    for nd, s in ((a, 1), (k, -1)):   # -g V_f leaving a
                        if nd != "gnd":
                            addB(self._vi(nd), 0, s * g * dv["V_f"])
                continue
            ib = dv["ib"]
            branch(a, k, ib)
            if on:                         # v_a - v_k - V_f = 0
                vrow(ib, a, k)
                if dv.get("V_f"):
                    addB(ib, 0, -dv["V_f"])
            else:                          # g_off v - i = 0
                vrow(ib, a, k, dv["g_off"])
                A[ib][ib] -= 1
        B = mp.zeros(n, self.m)
        for r, dct in enumerate(Bc):
            for j, v in dct.items():
                B[r, j] = mpf(v) if isinstance(v, Fr) else v
        return A, B

    _tsrc = Fr(0)

    def src_key(self, t: Fr):
        return tuple(wv.piece_at(t) for _, _, _, wv in self.vsrc)

    def next_break(self, t: Fr):
        c = [dv["sched"].next_edge(t) for dv in self.devs if dv["kind"] == "S"]
        c += [wv.next_break(t) for _, _, _, wv in self.vsrc]
        c = [x for x in c if x is not None]
        return min(c) if c else None

    def switch_bits(self, t: Fr):
        return tuple(dv["sched"].state(t) for dv in self.devs if dv["kind"] == "S")

    def full_mode(self, sw, dio):
        out, si, di = [], iter(sw), iter(dio)
        for dv in self.devs:
            out.append(next(si) if dv["kind"] == "S" else next(di))
        return tuple(out)


# ---------------------------------------------------- exact rational algebra --

def f_rref(M):
    M = [row[:] for row in M]
    rows, cols = len(M), len(M[0]) if M else 0
    piv, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return M, piv


def f_mul(A, B):
    Bt = list(zip(*B))
    return [[sum((a * b for a, b in zip(row, col) if a and b), Fr(0)) for col in Bt] for row in A]


def f_inv(A):
    n = len(A)
    R, piv = f_rref([row + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(A)])
    if piv[:n] != list(range(n)) or len(piv) < n or piv[n - 1] != n - 1:
        raise ZeroDivisionError("singular")
    return [row[n:] for row in R]


def f_null(A):
    n = len(A[0])
    R, piv = f_rref(A)
    free = [c for c in range(n) if c not in piv]
    out = []
    for fc in free:
        v = [Fr(0)] * n
        v[fc] = Fr(1)
        for i, pc in enumerate(piv):
            v[pc] = -R[i][fc]
        out.append(v)
    return out


def f_eye(n):
    return [[Fr(int(i == j)) for j in range(n)] for i in range(n)]


def to_mp(A):
    return mp.matrix([[mpf(x) for x in row] for row in A]) if A and A[0] else mp.zeros(len(A), 0)


# -------------------------------------------------------------- one mode ----

class Struct:
    """Exact core / nilpotent split of one mode's pencil (rationals)."""

    def __init__(self, E, A):
        n = len(E)
        for lam in (Fr(10) ** k for k in (4, 6, 2, 8, 3, 5, 7, 9, 1)):
            M = [[lam * e - a for e, a in zip(re, ra)] for re, ra in zip(E, A)]
            try:
                Mi = f_inv(M)
                break
            except ZeroDivisionError:
                continue
        else:
            raise Unsupported("singular pencil (ideal elements leave a current or a "
                              "voltage undetermined)")
        Eh = f_mul(Mi, E)
        P, rk = f_eye(n), n
        for k in range(n + 1):
            r = len(f_rref(P)[1]) if P else 0
            if k > 0 and r == rk:
                break
            rk = r
            P = f_mul(P, Eh)
        self.index = k - 1
        Pk = P                      # Eh^(index+1) has the same range / kernel
        _, piv = f_rref(Pk)
        R = [[Pk[i][c] for c in piv] for i in range(n)]
        for j in range(len(piv)):            # unit columns, exactly: z in physical scale
            mx = max(abs(R[i][j]) for i in range(n))
            for i in range(n):
                R[i][j] /= mx
        Kb = f_null(Pk)
        K = [[v[i] for v in Kb] for i in range(n)]
        r = len(piv)
        T = [R[i] + K[i] for i in range(n)]
        Ti = f_inv(T)
        Bl = f_mul(Ti, f_mul(Eh, T))
        assert all(Bl[i][j] == 0 for i in range(n) for j in range(n) if (i < r) != (j < r))
        Cs = [row[:r] for row in Bl[:r]]
        Nn = [row[r:] for row in Bl[r:]]
        Csi = f_inv(Cs) if r else []
        TM = f_mul(Ti, Mi)
        self.r, self.lam = r, lam
        self.F = [[(lam if i == j else 0) - Csi[i][j] for j in range(r)] for i in range(r)]
        self.ProjE = f_mul(Csi, TM[:r]) if r else []
        self.TMb = TM[r:]
        nk = n - r
        Pn = f_inv([[lam * Nn[i][j] - (i == j) for j in range(nk)] for i in range(nk)]) if nk else []
        self.P, self.J = Pn, f_mul(Pn, Nn) if nk else []
        self.R, self.K = R, K
        self.E, self.A = E, A


class Dyn:
    """Propagator of one mode with one source piece: y = [z; w], y' = H y."""

    def __init__(self, ck, st: Struct, B):
        r, m, n = st.r, ck.m, ck.n
        S = ck.S
        G = to_mp(st.ProjE) * B if r else mp.zeros(0, m)
        Q = mp.zeros(n - r, m)
        if n - r:
            Bb = to_mp(st.TMb) * B
            Pm, Jm = to_mp(st.P), to_mp(st.J)
            Jk, Sk = mp.eye(n - r), mp.eye(m)
            for _ in range(n - r + 1):
                Q -= Jk * Pm * Bb * Sk
                Jk, Sk = Jm * Jk, S * Sk
        N = r + m
        H = mp.zeros(N, N)
        Fm = to_mp(st.F)
        for i in range(r):
            for j in range(r):
                H[i, j] = Fm[i, j]
            for j in range(m):
                H[i, r + j] = G[i, j]
        for i in range(m):
            for j in range(m):
                H[r + i, r + j] = S[i, j]
        X = mp.zeros(n, N)
        Rm, Km = to_mp(st.R), to_mp(st.K)
        KQ = Km * Q if n - r else mp.zeros(n, m)
        for i in range(n):
            for j in range(r):
                X[i, j] = Rm[i, j]
            for j in range(m):
                X[i, r + j] = KQ[i, j]
        self.r, self.m, self.N, self.H, self.X, self.st = r, m, N, H, X, st
        # residual of the DAE on the whole manifold: E X H - A X - [0 B]
        Res = to_mp(st.E) * X * H - to_mp(st.A) * X
        for i in range(n):
            for j in range(m):
                Res[i, r + j] -= B[i, j]
        sc = max([mp.mpf(1)] + [abs(X[i, j]) for i in range(n) for j in range(N)])
        self.dae_residual = max([mp.mpf(0)] + [abs(Res[i, j]) for i in range(n) for j in range(N)]) / sc
        # H = [[F, G], [0, S]] is block triangular: eigenpairs of F (balanced),
        # and per source eigenpair (mu, u) of S the vector [(mu - F)^-1 G u; u]
        # (mu = 0 for the constant, +-i w for a sine). Normalised columns.
        lamF, VF = [], mp.zeros(r, r)
        if r:
            d = balance(Fm)
            Fb = mp.matrix(r, r)
            for i in range(r):
                for j in range(r):
                    Fb[i, j] = Fm[i, j] * d[j] / d[i]
            lamF, Vb = mp.eig(Fb)
            VF = mp.diag(d) * Vb
        pairs = [(mp.mpf(0), [mp.mpf(1)] + [0] * (m - 1))]
        for j in range(len(ck.freqs)):
            w = S[1 + 2 * j, 2 + 2 * j]
            for sg in (1, -1):
                u = [0] * m
                u[1 + 2 * j], u[2 + 2 * j] = mp.mpf(1), sg * mp.mpc(0, 1)
                pairs.append((sg * mp.mpc(0, w), u))
        lam = list(lamF) + [mu for mu, _ in pairs]
        V = mp.zeros(N, N)
        for k in range(r):
            for i in range(r):
                V[i, k] = VF[i, k]
        res = mp.mpf(0) if ck.ramp else mp.inf   # a ramp (t' = 1) is a Jordan block: expm
        for k, (mu, u) in enumerate(pairs):
            um = mp.matrix(u)
            col = mp.matrix(r, 1)
            if r:
                Gu = G * um
                tiny = mp.mpf(10) ** (-mp.mp.dps // 3)
                if max([abs(x) for x in Gu] + [0]) == 0:
                    pass                 # the source does not drive z: col = 0
                elif any(abs(mu - x) <= tiny * max(abs(mu), abs(x), tiny) for x in lamF):
                    res = mp.mpf(0)      # a mode resonant with this source
                else:
                    col = mp.lu_solve(mu * mp.eye(r) - Fm, Gu)
            for i in range(r):
                V[i, r + k] = col[i]
            for i in range(m):
                V[r + i, r + k] = um[i]
        for k in range(N):                      # unit columns (max norm)
            s_ = max(abs(V[i, k]) for i in range(N))
            if s_ > 0:
                for i in range(N):
                    V[i, k] /= s_
        # a repeated eigenvalue is fine when its eigenvectors are independent
        # (two decoupled charges); a defective H (a mode resonant with a
        # source, e.g. an ideal inductor across a DC source) is refused
        try:
            Vi = mp.inverse(V)
            cond = mp.mnorm(V, 1) * mp.mnorm(Vi, 1)
        except ZeroDivisionError:
            cond = mp.inf
        if res == 0 or not cond < mp.mpf(10) ** (mp.mp.dps // 2):
            # defective H (a mode resonant with a source: an ideal inductor
            # across a DC source ramps): propagate this mode by mp.expm
            self.cond, self.lam, self.V, self.Vi = mp.mpf(1), None, None, None
        else:
            self.cond = cond
            self.lam, self.V, self.Vi = lam, V, Vi
        self.ProjE = to_mp(st.ProjE) if r else None

    def y0(self, Ex, wv):
        z = self.ProjE * mp.matrix([mpf(q) if isinstance(q, Fr) else q for q in Ex]) \
            if self.r else []
        return mp.matrix([z[i] for i in range(self.r)] + list(wv))


def balance(H, sweeps=30):
    """Power-of-two diagonal d (Parlett-Reinsch) making D^-1 H D balanced:
    exact (no rounding), it only conditions the eigenvectors."""
    N = H.rows
    d = [mp.mpf(1)] * N
    for _ in range(sweeps):
        done = True
        for i in range(N):
            c = sum(abs(H[j, i] * d[i] / d[j]) for j in range(N) if j != i)
            r = sum(abs(H[i, j] * d[j] / d[i]) for j in range(N) if j != i)
            if c == 0 or r == 0:
                continue
            f = mp.mpf(2) ** int(mp.nint(mp.log(r / c, 2) / 2))
            if f != 1 and (c * f + r / f) < 0.95 * (c + r):
                d[i] *= f
                done = False
        if done:
            break
    return d


def w_at(ck, t):
    out = [mp.mpf(1)]
    for f in ck.freqs:
        a = 2 * mp.pi * f * t
        out += [mp.sin(a), mp.cos(a)]
    if ck.ramp:
        out.append(mp.mpf(t))
    return out


def expo(mu, h):
    """(e^{mu h} - 1)/mu, the limit h at mu = 0."""
    if abs(mu * h) < mp.mpf(10) ** (-mp.mp.dps // 2):
        return h * (1 + mu * h / 2)
    return mp.expm1(mu * h) / mu


class Seg:
    def __init__(self, dyn, t0, y0, tau):
        self.dyn, self.t0, self.tau = dyn, t0, tau
        self.x = dyn.lam is None                  # expm segment (defective mode)
        self.c = None if self.x else dyn.Vi * y0
        self.y0 = y0
        self._W = {}

    def lin_row(self, row, ta, tb):
        """Integral of row.y over [ta, tb] (local times), exact."""
        if not self.x:
            return self.lin(self.coef(row), ta, tb)
        return (row * self._int(ta, tb)[0])[0]

    def _int(self, ta, tb):
        """(integral of y, integral of y y^T) over [ta, tb]: augmented and Van
        Loan matrix exponentials (exact to working precision)."""
        k = (ta, tb)
        if k not in self._W:
            d, h = self.dyn, tb - ta
            ya = self.y(ta)
            N = d.N
            A1 = mp.zeros(N + 1, N + 1)
            for i in range(N):
                for j in range(N):
                    A1[i, j] = d.H[i, j]
                A1[i, N] = ya[i]
            E1 = mp.expm(A1 * h)
            iy = mp.matrix([E1[i, N] for i in range(N)])
            C = mp.zeros(2 * N, 2 * N)
            for i in range(N):
                for j in range(N):
                    C[i, j] = -d.H[i, j]          # Van Loan: [[-A, Q], [0, A^T]], A = H
                    C[i, N + j] = ya[i] * ya[j]
                    C[N + i, N + j] = d.H[j, i]
            E2 = mp.expm(C * h)
            G2 = mp.matrix(N, N)
            F3 = mp.matrix(N, N)
            for i in range(N):
                for j in range(N):
                    G2[i, j] = E2[i, N + j]
                    F3[i, j] = E2[N + i, N + j]
            self._W[k] = (iy, F3.T * G2)
        return self._W[k]

    def y(self, tau):
        d = self.dyn
        if self.x:
            return mp.expm(d.H * tau) * self.y0
        e = [mp.exp(lk * tau) * self.c[k] for k, lk in enumerate(d.lam)]
        return mp.matrix([mp.re(sum(d.V[i, k] * e[k] for k in range(d.N))) for i in range(d.N)])

    def coef(self, row):
        """row (1 x N on y) -> a_k with row.y(tau) = Re sum a_k e^{lam_k tau}."""
        d = self.dyn
        rv = row * d.V
        return [rv[0, k] * self.c[k] for k in range(d.N)]

    def lin(self, a, ta, tb):
        lam = self.dyn.lam
        return mp.re(sum(a[k] * mp.exp(lam[k] * ta) * expo(lam[k], tb - ta) for k in range(len(a))))

    def quad(self, Qy, ta, tb):
        """Integral of y^T Qy y over [ta, tb] (tau, local), exact."""
        d = self.dyn
        if self.x:
            W = self._int(ta, tb)[1]
            return sum(Qy[i, j] * W[j, i] for i in range(d.N) for j in range(d.N))
        e = [mp.exp(lk * ta) * self.c[k] for k, lk in enumerate(d.lam)]
        Yh = d.V * mp.diag(e)
        M = Yh.T * Qy * Yh
        h = tb - ta
        return mp.re(sum(M[k, j] * expo(d.lam[k] + d.lam[j], h)
                         for k in range(d.N) for j in range(d.N)))


# -------------------------------------------------------------- the solver --

class Solver:
    def __init__(self, ck: Circuit, dps=50, forced=None, scan=None):
        self.ck, self.dps, self.forced = ck, dps, forced or {}
        mp.mp.dps = dps
        ck.S = mp.zeros(ck.m, ck.m)
        for j, f in enumerate(ck.freqs):
            w = 2 * mp.pi * f
            ck.S[1 + 2 * j, 2 + 2 * j], ck.S[2 + 2 * j, 1 + 2 * j] = w, -w
        if ck.ramp:
            ck.S[ck.m - 1, 0] = 1
        self.structs, self.dyns = {}, {}
        self.Em = to_mp(ck.E)
        self.tol = mp.mpf(10) ** (-(dps - 12))
        self.scan = mp.mpf(scan) if scan else mp.mpf(ck.dt) / 2

    # -- modes --------------------------------------------------------------
    def dyn(self, mode, skey):
        k = (mode, skey)
        if k not in self.dyns:
            ck = self.ck
            if mode not in self.structs:
                A, _ = ck.stamp(mode)
                try:
                    self.structs[mode] = Struct(ck.E, A)
                except Unsupported as ex:
                    self.structs[mode] = ex
            st = self.structs[mode]
            if isinstance(st, Exception):
                self.dyns[k] = st
            else:
                ck._tsrc = self._t_for_key[skey]
                _, B = ck.stamp(mode)
                try:
                    self.dyns[k] = Dyn(ck, st, B)
                except Unsupported as ex:
                    self.dyns[k] = ex
        return self.dyns[k]

    def dev_rows(self, dyn, mode):
        """Per device: (row on y of its signal g that must stay > 0, or None
        for a switch)."""
        ck, rows = self.ck, []
        X = dyn.X
        for dv, on in zip(ck.devs, mode):
            if dv["kind"] == "S":
                rows.append(None)
                continue
            rx = mp.zeros(1, ck.n)
            off = mp.mpf(0)
            if ck.ideal and on:
                rx[0, dv["ib"]] = 1
            else:
                for nd, s in ((dv["a"], 1), (dv["k"], -1)):
                    if nd != "gnd":
                        rx[0, ck._vi(nd)] += s
                off = dv["V_f"] if on else dv["V_th"]
                if not on:
                    rx = -rx
                    off = -off
            ry = rx * X
            ry[0, dyn.r] -= mpf(off) if isinstance(off, Fr) else off
            rows.append(ry)
        return rows

    def taylor_sign(self, dyn, row, y):
        """Sign of row.y(t) for t -> 0+ (first significant Taylor term)."""
        v = y
        rmax = max(abs(row[0, j]) for j in range(dyn.N))
        tiny = mp.mpf(10) ** (-mp.mp.dps + 5)
        # order 0: a device at an event sits on its boundary to the precision
        # of the projection between modes (~15 digits lost when g_off and g_on
        # meet): below 10^-(dps/2) of its scale it is zero and the derivatives
        # decide (the bridge's rail turn-on re-selected the old mode forever)
        tol0 = mp.mpf(10) ** (-(mp.mp.dps // 2))
        for k in range(10):
            g = (row * v)[0]
            sc = sum(abs(row[0, j] * v[j]) for j in range(dyn.N))
            ref = rmax * max(abs(v[j]) for j in range(dyn.N))
            if abs(g) > (tol0 if k == 0 else self.tol) * sc and abs(g) > tiny * ref:
                return 1 if g > 0 else -1
            v = dyn.H * v
        return 0

    def jump_source_work(self, mode, xm, xp):
        """Impulsive work of the voltage sources in a state jump: integrating
        E x' = A x + B w across it, E (x+ - x-) = A s with s the impulse weights;
        a source of voltage v delivers -v s_I. A is singular when a node holds
        only capacitors (its voltage impulse is free): the minimum-norm s, and
        None when that freedom reaches a source current or A s misses."""
        ck = self.ck
        A, b = to_mp(ck.stamp(mode)[0]), self.Em * (xp - xm)
        U, S, V = mp.svd_r(A)
        cut = max(S) * mp.mpf(10) ** (-mp.mp.dps // 2)
        keep = [i for i in range(len(S)) if S[i] > cut]
        ub = U.T * b
        s = mp.zeros(A.rows, 1)
        for i in keep:
            s += V[i, :].T * (ub[i] / S[i])
        if mp.norm(A * s - b) > cut * (1 + mp.norm(b)):
            return None
        w = mp.mpf(0)
        for src in ck.vsrc:
            nm, na, nb = src[:3]
            k = ck.ix.get(f"Is({nm})")
            if k is not None:
                if any(abs(V[i, k]) > cut for i in range(len(S)) if i not in keep):
                    return None
                v = sum(sg * xp[ck._vi(nd)] for nd, sg in ((na, 1), (nb, -1)) if nd != "gnd")
                w -= v * s[k]
        return w

    def select(self, t, Ex, sw, prev_dio, skey, k_event):
        """Enumerate candidate diode modes at (t, Ex): returns (dio, info)."""
        ck = self.ck
        nd = len(ck.diodes)
        wv = w_at(ck, t)

        def check(dio):
            mode = ck.full_mode(sw, dio)
            dy = self.dyn(mode, skey)
            if isinstance(dy, Exception):
                return None, None, None, str(dy)
            y = dy.y0(Ex, wv)
            signs = []
            for row in self.dev_rows(dy, mode):
                if row is not None:
                    signs.append(self.taylor_sign(dy, row, y))
            return signs, dy, y, None

        base = tuple(prev_dio)
        signs, *_ = check(base)
        cand = set()
        if signs is None:
            cand = set(range(nd))
        else:
            cand = {i for i, s in enumerate(signs) if s <= 0}
        tried, results = set(), {}
        while True:
            new = False
            for sub in itertools.product((False, True), repeat=len(cand)):
                dio = list(base)
                for i, f in zip(sorted(cand), sub):
                    if f:
                        dio[i] = not dio[i]
                dio = tuple(dio)
                if dio in tried:
                    continue
                tried.add(dio)
                s, dy, y, why = check(dio)
                results[dio] = (s, why)
                if s is not None:
                    bad = {i for i, x in enumerate(s) if x <= 0}
                    if bad - cand and len(cand | bad) <= 12:
                        cand |= bad
                        new = True
            if not new:
                break
        strict = [d for d, (s, _) in results.items() if s is not None and all(x > 0 for x in s)]
        weak = [d for d, (s, _) in results.items()
                if s is not None and all(x > 0 or (x == 0 and not d[i]) for i, x in enumerate(s))]
        pool = strict or weak
        info = dict(t=t, n_candidates=len(results), n_strict=len(strict), n_weak=len(weak),
                    singular=sum(1 for s, w in results.values() if w), boundary=sorted(cand))
        if not pool:
            raise RuntimeError(f"no consistent diode mode at t={mp.nstr(t, 20)}: "
                               f"{ {k: v for k, v in results.items()} }")
        pool.sort(key=lambda d: (sum(a != b for a, b in zip(d, base)), d))
        info["consistent"] = pool
        pick = self.forced.get(k_event, 0)
        return pool[min(pick, len(pool) - 1)], info

    # -- DC operating point -------------------------------------------------
    def dc_start(self, sw, skey):
        """v1 without uic: the DC point (C open, L short) at t = 0; E x0."""
        ck = self.ck
        nd = len(ck.diodes)
        ok = []
        for dio in itertools.product((False, True), repeat=nd):
            mode = ck.full_mode(sw, dio)
            ck._tsrc = Fr(0)
            A, B = ck.stamp(mode)
            try:
                Ai = f_inv(A)
            except ZeroDivisionError:
                continue
            b = [-sum((B[i, j] * w for j, w in enumerate(w_at(ck, mp.mpf(0)))), mp.mpf(0))
                 for i in range(ck.n)]
            x = to_mp(Ai) * mp.matrix(b)
            good = True
            for dv, on in zip(ck.devs, mode):
                if dv["kind"] != "D":
                    continue
                if ck.ideal and on:
                    s = x[dv["ib"]]
                else:
                    v = sum((s * x[ck._vi(n)] for n, s in ((dv["a"], 1), (dv["k"], -1))
                             if n != "gnd"), mp.mpf(0))
                    s = v if on else -v
                if s <= -self.tol * (1 + abs(s)):
                    good = False
                if on and s <= 0:
                    good = False
            if good:
                ok.append((dio, x))
        if not ok:
            return None, 0
        dio, x = ok[0]
        Em = to_mp(ck.E)
        return list(Em * x), len(ok)

    # -- run -------------------------------------------------------------------
    def run(self, verbose=False):
        ck = self.ck
        t = mp.mpf(0)
        tF = Fr(0)
        t_end = mpf(ck.t_end)
        self._t_for_key = {}
        sw = ck.switch_bits(tF)
        skey = ck.src_key(tF)
        self._t_for_key.setdefault(skey, tF)
        self.notes = []
        Ex = [mpf(q) for q in ck.charges()]
        n_dc = None
        if not ck.uic:
            ex, n_dc = self.dc_start(sw, skey)
            if ex is None:
                self.notes.append("no DC operating point: zero start (v1_reader's fallback)")
            else:
                Ex = ex
        dio, info = self.select(t, Ex, sw, (False,) * len(ck.diodes), skey, 0)
        info.update(cause="start", prev=None, mode=ck.full_mode(sw, dio))
        self.events = [info]
        self.segs = []
        self.jumps = []
        mode = ck.full_mode(sw, dio)
        dy = self.dyn(mode, skey)
        y = dy.y0(Ex, w_at(ck, t))
        while t < t_end:
            nb = ck.next_break(tF)
            stop_F = min(x for x in (nb, ck.t_end) if x is not None)
            H = mpf(stop_F) - t
            seg = Seg(dy, t, y, H)
            rows = self.dev_rows(dy, mode)
            hit = self.find_root(seg, rows, H)
            tau = hit[0] if hit else H
            seg.tau, seg.mode = tau, mode
            self.segs.append(seg)
            yE = seg.y(tau)
            xm = dy.X * yE
            Ex = list(self.Em * xm)
            if hit:
                t = t + tau
                tF = Fr(int(t * 10 ** (self.dps + 5)), 10 ** (self.dps + 5))
                cause = f"{ck.devs[hit[1]]['name']} root"
            else:
                tF = stop_F
                t = mpf(stop_F)
                cause = "schedule"
                if tF >= ck.t_end:
                    break
            sw = ck.switch_bits(tF)
            skey = ck.src_key(tF)
            self._t_for_key.setdefault(skey, tF)
            prev = mode
            dio, info = self.select(t, Ex, sw, [mode[i] for i in ck.diodes], skey, len(self.events))
            mode = ck.full_mode(sw, dio)
            dy = self.dyn(mode, skey)
            y = dy.y0(Ex, w_at(ck, t))
            xp = dy.X * y
            dE = self.stored(xp) - self.stored(xm)
            if abs(dE) > mp.mpf(10) ** (-(mp.mp.dps // 2)) * (1 + abs(self.stored(xm))):  # projection noise is ~15 digits
                self.jumps.append(dict(t=t, dE=dE, x_minus=xm, x_plus=xp, mode=mode,
                                       w_src=self.jump_source_work(mode, xm, xp)))
            info.update(cause=cause, prev=prev, mode=mode)
            if (mode == prev and len(self.events) > 20
                    and all(e["t"] == t and e["mode"] == mode for e in self.events[-20:])):
                raise RuntimeError(f"no progress at t={mp.nstr(t, 20)}: a root keeps re-firing")
            self.events.append(info)
            if verbose:
                print(f"  {len(self.events) - 1:4d} t={mp.nstr(t, 20):>24s} {cause:14s} "
                      f"{flips(ck, prev, mode)} cand {info['n_candidates']} strict {info['n_strict']}")
        self.n_dc = n_dc
        return self

    def find_root(self, seg, rows, H):
        if seg.x:
            return self.find_root_x(seg, rows, H)
        dy = seg.dyn
        A = []
        for d, row in enumerate(rows):
            if row is None:
                continue
            a = seg.coef(row)
            sc = sum(abs(x) for x in a)
            yscale = sum(abs(row[0, j] * seg.y0[j]) for j in range(dy.N))
            if sc <= self.tol * (1 + yscale) * 1e3:
                continue                    # identically zero: pinned on its boundary
            A.append((d, a, [a[k] * dy.lam[k] for k in range(dy.N)]))
        if not A:
            return None
        lam = dy.lam
        sc_all = mp.mpf(10) ** (-(self.dps - 8))

        eps = mp.mpf(10) ** (-(mp.mp.dps // 2))  # projection noise (~half the digits), as in taylor_sign

        def ev(tau):
            # (g, g', rounding noise of g): a crossing must exceed the noise of
            # the sum it is read from (a zero-current start cancels to ~eps)
            e = [mp.exp(x * tau) for x in lam]
            return [(mp.re(sum(a[k] * e[k] for k in range(dy.N))),
                     mp.re(sum(al[k] * e[k] for k in range(dy.N))),
                     eps * sum(abs(a[k] * e[k]) for k in range(dy.N))) for _, a, al in A]

        def g1(i, tau, deriv=False):
            a = A[i][2] if deriv else A[i][1]
            return mp.re(sum(a[k] * mp.exp(lam[k] * tau) for k in range(dy.N)))

        # adaptive scan: resolve every oscillation while its term is alive
        # (|a_k| e^{Re lam_k tau} above 10^-(dps-8) of the signal's scale)
        osc = []
        for _, a, _ in A:
            mag = sum(abs(x) for x in a)
            for k in range(dy.N):
                if abs(mp.im(lam[k])) > 0 and abs(a[k]) > 0:
                    re = mp.re(lam[k])
                    r = abs(a[k]) / (sc_all * mag)
                    dead = mp.inf if re >= 0 else (mp.log(r) / -re if r > 1 else mp.mpf(0))
                    osc.append((dead, mp.pi / (4 * abs(mp.im(lam[k])))))

        def step_at(tau):
            return min([min(self.scan, H)] + [h for dead, h in osc if tau < dead])

        grid = [mp.mpf(0)]
        x = mp.mpf("1e-24")
        first = step_at(0)
        while x < min(first, H):
            grid.append(x)
            x *= 4
        tau = grid[-1]
        prev = ev(grid[0])
        a_ = grid[0]
        pts = iter(grid[1:])
        while True:
            b = next(pts, None)
            if b is None:
                b = min(tau + step_at(tau), H) if tau < H else None
                if b is None:
                    return None
            tau = b
            cur = ev(b)
            best = None
            for i in range(len(A)):
                gb, db, nb = cur[i]
                da = prev[i][1]
                lo = None
                if gb < -nb:
                    lo = (a_, b)
                elif da < 0 < db:
                    x0, x1 = a_, b
                    while x1 - x0 > self.tol * (x1 + mp.mpf("1e-30")):
                        xm = (x0 + x1) / 2
                        if g1(i, xm, True) < 0:
                            x0 = xm
                        else:
                            x1 = xm
                    if g1(i, x1) < -nb:
                        lo = (a_, x1)
                if lo is not None:
                    x0, x1 = lo
                    while x1 - x0 > self.tol * (x1 + mp.mpf("1e-30")):
                        xm = (x0 + x1) / 2
                        if g1(i, xm) > 0:
                            x0 = xm
                        else:
                            x1 = xm
                    if best is None or x1 < best[0]:
                        best = (x1, A[i][0])
            if best is not None:
                return best
            if b >= H:
                return None
            prev, a_ = cur, b

    def find_root_x(self, seg, rows, H):
        """find_root for an expm segment: march y by a fixed-step propagator,
        bisect on direct matrix exponentials."""
        dy = seg.dyn
        eps = mp.mpf(10) ** (-(mp.mp.dps // 2))  # projection noise (~half the digits), as in taylor_sign
        act = []
        for d, row in enumerate(rows):
            if row is None:
                continue
            v, mag = seg.y0, mp.mpf(0)
            for _ in range(4):
                mag = max(mag, abs((row * v)[0]))
                v = dy.H * v
            yscale = sum(abs(row[0, j] * seg.y0[j]) for j in range(dy.N))
            if mag > self.tol * (1 + yscale) * 1e3:
                act.append((d, row, row * dy.H))
        if not act:
            return None

        def ev(y):
            return [((r * y)[0], (rd * y)[0], eps * sum(abs(r[0, j] * y[j]) for j in range(dy.N)))
                    for _, r, rd in act]

        grid = [mp.mpf(0)]
        x = mp.mpf("1e-24")
        h = min(self.scan, H)
        while x < h:
            grid.append(x)
            x *= 4
        a_, ya = mp.mpf(0), seg.y0
        prev = ev(ya)
        Phi = mp.expm(dy.H * h)
        k = 1
        while True:
            if k < len(grid):
                b = grid[k]
                yb = seg.y(b)
            else:
                b = min(a_ + h, H)
                yb = Phi * ya if b == a_ + h else seg.y(b)
            k += 1
            cur = ev(yb)
            best = None
            for i, (d, r, rd) in enumerate(act):
                gb, db, nb = cur[i]
                lo = None
                if gb < -nb:
                    lo = (a_, b)
                elif prev[i][1] < 0 < db:
                    x0, x1 = a_, b
                    while x1 - x0 > self.tol * (x1 + mp.mpf("1e-30")):
                        xm = (x0 + x1) / 2
                        if (rd * seg.y(xm))[0] < 0:
                            x0 = xm
                        else:
                            x1 = xm
                    if (r * seg.y(x1))[0] < -nb:
                        lo = (a_, x1)
                if lo is not None:
                    x0, x1 = lo
                    while x1 - x0 > self.tol * (x1 + mp.mpf("1e-30")):
                        xm = (x0 + x1) / 2
                        if (r * seg.y(xm))[0] > 0:
                            x0 = xm
                        else:
                            x1 = xm
                    if best is None or x1 < best[0]:
                        best = (x1, d)
            if best is not None:
                return best
            if b >= H:
                return None
            prev, a_, ya = cur, b, yb

    # -- post-processing ------------------------------------------------------
    def stored(self, x):
        ck = self.ck
        e = mp.mpf(0)
        for _, a, b, C, _ic in ck.caps:
            v = (x[ck._vi(a)] if a != "gnd" else 0) - (x[ck._vi(b)] if b != "gnd" else 0)
            e += mpf(C) * v * v / 2
        iL = [x[ck.ix[f"I({nm})"]] for nm, *_ in ck.inds]
        for k, (_, _a, _b, L, _) in enumerate(ck.inds):
            e += mpf(L) * iL[k] ** 2 / 2
        for p, s, M in ck.mut:
            e += mpf(M) * iL[p] * iL[s]
        return e

    def seg_at(self, t, side=1):
        import bisect
        starts = [s.t0 for s in self.segs]
        i = bisect.bisect_right(starts, t) - 1
        if side < 0:
            while i > 0 and self.segs[i].t0 >= t:
                i -= 1
        return self.segs[max(i, 0)]

    def obs_row(self, col):
        """(1 x n row on x, gate wave or None) of an observable column."""
        ck = self.ck
        kind, _, rest = col.partition("(")
        name = rest.rstrip(")")
        r = mp.zeros(1, ck.n)
        if kind.upper() == "V":
            parts = [node(p.strip()) for p in name.split(",")]
            waves = []
            for nd, s in zip(parts, (1, -1)):
                if nd == "gnd":
                    continue
                if nd in ck.waves:
                    waves.append((ck.waves[nd], s))
                else:
                    r[0, ck._vi(nd)] += s
            return r, waves
        r[0, ck.ix[f"I({name})"]] = 1
        return r, []

    def sample(self, cols, ts):
        """{col: [mp values]} at mp times ts; at a duplicated time the first is
        the left limit, the second the right limit."""
        rows = {c: self.obs_row(c) for c in cols}
        out = {c: [] for c in cols}
        last, prev, phi, cseg, ca = None, None, None, None, None  # prev = (seg, tau, y); phi = (dyn, h, expm(H h))
        for t in ts:
            side = 1 if (last is not None and t == last) else -1
            last = t
            seg = self.seg_at(t, side)
            tau = t - seg.t0
            if not seg.x:
                # eigen path: per column Re sum a_k e^{lam_k tau} (N terms, not N^2)
                if seg is not cseg:
                    cseg = seg
                    ca = {c: seg.coef(r * seg.dyn.X) for c, (r, _) in rows.items()}
                e = [mp.exp(lk * tau) for lk in seg.dyn.lam]
                for c, (r, waves) in rows.items():
                    v = mp.re(mp.fdot(ca[c], e)) + sum(s * wv.value(t, side) for wv, s in waves)
                    out[c].append(v)
                continue
            if prev and prev[0] is seg and tau > prev[1]:
                # expm per point is the cost of a fine grid: step from the last sample
                h = tau - prev[1]
                if phi is None or phi[0] is not seg.dyn or abs(h - phi[1]) > mp.mpf(10) ** (5 - mp.mp.dps) * h:
                    phi = (seg.dyn, h, mp.expm(seg.dyn.H * h))
                y = phi[2] * prev[2]
            else:
                y = seg.y(tau)
            prev = (seg, tau, y)
            x = seg.dyn.X * y
            for c, (r, waves) in rows.items():
                v = (r * x)[0] + sum(s * wv.value(t, side) for wv, s in waves)
                out[c].append(v)
        return out

    def at_float(self, cols, t, delta=0.0, rel=1e-13, gaps=False):
        """The truth at float times ``t`` for the ``exact`` measure:
        ``{col: (len(t), m) array}``, one column per segment whose interval
        reaches within ``delta`` of t_i, each segment's own formula continued
        to t_i (NaN pads). Away from events m = 1; near one, both one-sided
        branches are candidates, so a left- or right-limit sample is compared
        with its own side. Evaluated in float; a value whose cancellation
        bound (4 eps sum|a_k e^{lam_k tau}|) exceeds ``rel`` x the column's
        peak is recomputed in mpmath. ``gaps=True`` also returns the
        (len(t), m) time distance from t_i to each candidate's interval
        (0 for the segment holding t_i)."""
        import numpy as np
        t = np.asarray(t, dtype=float)
        segs = self.segs
        t0 = np.array([float(s.t0) for s in segs])
        t1 = np.array([float(s.t0 + s.tau) for s in segs])
        j_lo = np.clip(np.searchsorted(t1, t - delta, side="left"), 0, len(segs) - 1)
        j_hi = np.clip(np.searchsorted(t0, t + delta, side="right") - 1, j_lo, len(segs) - 1)
        m = int((j_hi - j_lo).max()) + 1 if t.size else 1
        rows = {c: self.obs_row(c) for c in cols}
        out = {c: np.full((t.size, m), np.nan) for c in cols}
        bnd = {c: np.zeros((t.size, m)) for c in cols}
        coef = {}
        for o in range(m):
            jj = j_lo + o
            ok = jj <= j_hi
            for j in np.unique(jj[ok]):
                idx = np.flatnonzero(ok & (jj == j))
                seg = segs[j]
                hi = float(seg.t0)
                tau = (t[idx] - hi) - float(seg.t0 - mp.mpf(hi))
                for c, (r, waves) in rows.items():
                    if seg.x:
                        xs = self._x_states(seg, t[idx])
                        v = np.array([float((r * x)[0]) for x in xs])
                        b = np.zeros(idx.size)
                    else:
                        if (j, c) not in coef:
                            a = seg.coef(r * seg.dyn.X)
                            coef[j, c] = (np.array([complex(x) for x in a]),
                                          np.array([complex(x) for x in seg.dyn.lam]))
                        a, lam = coef[j, c]
                        with np.errstate(over="ignore", invalid="ignore"):
                            ex = np.exp(np.outer(tau, lam)) * a
                            v = ex.sum(axis=1).real
                            b = 4 * np.finfo(float).eps * np.abs(ex).sum(axis=1)
                    if waves:
                        v = v + np.array([self._wave_part(waves, seg, x) for x in t[idx]])
                    out[c][idx, o], bnd[c][idx, o] = v, b
        valid = (j_lo[:, None] + np.arange(m)[None, :]) <= j_hi[:, None]
        single = j_lo == j_hi
        for c, (r, waves) in rows.items():
            v = np.abs(out[c][single, 0])
            v = v[np.isfinite(v)]
            peak = float(v.max()) if v.size else 1.0
            bad = valid & (~np.isfinite(out[c]) | (bnd[c] > rel * max(peak, 1e-300)))
            for i, o in zip(*np.nonzero(bad)):
                seg = segs[j_lo[i] + o]
                tau = mp.mpf(float(t[i])) - seg.t0
                if seg.x:
                    v = (r * (seg.dyn.X * seg.y(tau)))[0]
                else:
                    a = seg.coef(r * seg.dyn.X)
                    v = mp.re(mp.fsum(ak * mp.exp(lk * tau) for ak, lk in zip(a, seg.dyn.lam)))
                out[c][i, o] = float(v) + (self._wave_part(waves, seg, t[i]) if waves else 0.0)
        if not gaps:
            return out
        jj = np.minimum(j_lo[:, None] + np.arange(m)[None, :], len(segs) - 1)
        gap = np.maximum(np.maximum(t0[jj] - t[:, None], t[:, None] - t1[jj]), 0.0)
        return out, np.where(valid, gap, np.nan)

    def _x_states(self, seg, ts):
        """x at float times ts on an expm segment: stepped in time order,
        one expm(H h) per distinct exact gap h (an engine grid has few)."""
        import numpy as np
        key = ts.tobytes()
        if getattr(self, "_xs_seg", None) is seg and self._xs_key == key:
            return self._xs_val
        out = [None] * len(ts)
        y, tp, cache = None, None, {}
        for q in np.argsort(ts, kind="stable"):
            tq = mp.mpf(float(ts[q]))
            if y is None:
                y = seg.y(tq - seg.t0)
            elif tq != tp:
                h = tq - tp
                if h not in cache:
                    cache[h] = mp.expm(seg.dyn.H * h)
                y = cache[h] * y
            tp = tq
            out[q] = seg.dyn.X * y
        self._xs_seg, self._xs_key, self._xs_val = seg, key, out
        return out

    def _wave_part(self, waves, seg, x):
        """Source-node part of a column on ``seg``, its time clamped to the
        segment (a continuation reads the source's one-sided value)."""
        x = mp.mpf(float(x))
        a, b = seg.t0, seg.t0 + seg.tau
        tt, side = (a, 1) if x <= a else ((b, -1) if x >= b else (x, -1))
        return float(sum(s * wv.value(tt, side) for wv, s in waves))

    def window(self, cols, w0, w1):
        """Exact time-weighted mean and RMS of each column over [w0, w1]."""
        rows = {c: self.obs_row(c) for c in cols}
        res = {}
        for c, (r, waves) in rows.items():
            if waves:
                res[c] = None
                continue
            s1 = s2 = mp.mpf(0)
            for seg in self.segs:
                a, b = max(seg.t0, w0), min(seg.t0 + seg.tau, w1)
                if b <= a:
                    continue
                ry = r * seg.dyn.X
                s1 += seg.lin_row(ry, a - seg.t0, b - seg.t0)
                s2 += seg.quad(ry.T * ry, a - seg.t0, b - seg.t0)
            T = w1 - w0
            res[c] = dict(mean=s1 / T, rms=mp.sqrt(s2 / T))
        return res

    def energy(self, w0=None, w1=None):
        """Exact energy per element over [w0, w1]: absorbed by resistors and
        devices, delivered by sources; stored-energy change; balance."""
        ck = self.ck
        w0 = mp.mpf(0) if w0 is None else w0
        w1 = mpf(ck.t_end) if w1 is None else w1
        acc = {}

        def vrow(a, b):
            r = mp.zeros(1, ck.n)
            for nd, s in ((a, 1), (b, -1)):
                if nd != "gnd":
                    r[0, ck._vi(nd)] += s
            return r

        first = last = None
        for seg in self.segs:
            a, b = max(seg.t0, w0), min(seg.t0 + seg.tau, w1)
            if b <= a:
                continue
            X, r, mode = seg.dyn.X, seg.dyn.r, seg.mode

            def q(ru, rv):
                u, v = ru * X, rv * X
                M = (u.T * v + v.T * u) / 2
                return seg.quad(M, a - seg.t0, b - seg.t0)

            const = mp.zeros(1, X.cols)
            const[0, r] = 1
            for nm, na, nb, g in ck.res:
                vr = vrow(na, nb)
                acc[nm] = acc.get(nm, 0) + mpf(g) * q(vr, vr)
            for dv, on in zip(ck.devs, mode):
                vr = vrow(dv["a"], dv["k"])
                if ck.ideal:
                    ir = mp.zeros(1, ck.n)
                    ir[0, dv["ib"]] = 1
                    e = q(vr, ir)
                else:
                    g = mpf(dv["g_on"] if on else dv["g_off"])
                    e = g * q(vr, vr)
                    if on and dv.get("V_f"):
                        u = vr * X
                        e -= g * mpf(dv["V_f"]) * seg.lin_row(u, a - seg.t0, b - seg.t0)
                acc[dv["name"]] = acc.get(dv["name"], 0) + e
            for nm, na, nb, _ in ck.vsrc:
                ir = mp.zeros(1, ck.n)
                ir[0, ck.ix[f"Is({nm})"]] = 1
                acc["src:" + nm] = acc.get("src:" + nm, 0) - q(vrow(na, nb), ir)
            for nm, na, nb, cur in ck.isrc:
                u = vrow(na, nb) * X
                acc["src:" + nm] = acc.get("src:" + nm, 0) + mpf(cur) * seg.lin_row(
                    u, a - seg.t0, b - seg.t0)
            if first is None:
                first = (seg, a)
            last = (seg, b)
        Es = [self.stored(s.dyn.X * s.y(tt - s.t0)) for s, tt in (first, last)]
        jumps = sum((j["dE"] for j in self.jumps if w0 < j["t"] <= w1), mp.mpf(0))
        src = sum((v for k, v in acc.items() if k.startswith("src:")), mp.mpf(0))
        diss = sum((v for k, v in acc.items() if not k.startswith("src:")), mp.mpf(0))
        # stored energy in the scale: a circuit at rest has only rounding in every term
        scale = max(abs(src), abs(diss), abs(Es[0]), abs(Es[1]), abs(jumps), mp.mpf(10) ** -300)
        bal = (src - diss - (Es[1] - Es[0] - jumps)) / scale
        return dict(per=acc, source=src, dissipated=diss, dE=Es[1] - Es[0], jump_dE=jumps,
                    impulsive_loss=-jumps, balance_rel=bal)


def flips(ck, a, b):
    if a is None:
        return " ".join(f"{dv['name']}{'+' if y else '-'}" for dv, y in zip(ck.devs, b))
    return " ".join(f"{dv['name']}{'+' if y else '-'}"
                    for dv, x, y in zip(ck.devs, a, b) if x != y) or "-"


# ------------------------------------------------------------ outputs ------

def out_grid(sol, dt=None):
    """Netlist dt grid with every event two-sided (left then right limit)."""
    ck = sol.ck
    dt = Fr(dt if dt is not None else ck.dt)
    n = int(ck.t_end / dt)
    ts = [mpf(k * dt) for k in range(n + 1)]
    if ts[-1] < mpf(ck.t_end):
        ts.append(mpf(ck.t_end))
    for e in sol.events[1:]:
        ts += [e["t"], e["t"]]
        # fast modes start at events (a g_off node rings in ns): geometric points after it
        ts += [e["t"] + mpf(dt) / 2 ** k for k in range(1, 11) if e["t"] + mpf(dt) / 2 ** k < mpf(ck.t_end)]
    for s in sol.segs[1:]:
        ts += [s.t0, s.t0]
    ts.sort()
    out, prev, cnt = [], None, 0
    for t in ts:
        cnt = cnt + 1 if t == prev else 1
        if cnt <= 2:
            out.append(t)
        prev = t
    return out


def write_csv(path, ts, vals, cols):
    import numpy as np
    t = [float(x) for x in ts]
    # the right limit one ulp after the left: np.interp needs increasing times
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = float(np.nextafter(t[i - 1], math.inf))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("time," + ",".join(cols) + "\n")
        for i in range(len(t)):
            f.write(f"{t[i]!r}," + ",".join(repr(float(vals[c][i])) for c in cols) + "\n")


def solve(path, form="conductance", dps=50, forced=None, verbose=False):
    ck = Circuit(path, form)
    return Solver(ck, dps, forced).run(verbose)


# ------------------------------------------------------------- validation --
# Each case: the tool on an in-memory netlist against a closed form written
# by hand (its own state equations, mp.expm / mp.findroot), sharing no code
# with the MNA path. Element values are the doubles the netlist gives.

def _net(cid, comps, tstop, dt, obs=(), uic=True):
    return dict(schema="pulsim-v1", simulation=dict(tstop=tstop, dt=dt, uic=uic),
                benchmark=dict(id=cid, observables=[{"column": c} for c in obs]),
                components=comps)


def _R(n, a, b, v):
    return dict(type="resistor", name=n, nodes=[a, b], value=v)


def _Vdc(n, a, v):
    return dict(type="voltage_source", name=n, nodes=[a, 0], waveform=dict(type="dc", value=v))


def _at(sol, col, t):
    return sol.sample([col], [mp.mpf(t)])[col][0]


def _g(R):
    """1/R as the engine stamps it: the double 1.0 / R, exactly."""
    return 1 / mpf(Fr(1.0 / R))


def _rel(a, b):
    return abs(a - b) / max(abs(b), mp.mpf(10) ** -40)


def case_rc(dps):
    R, C, V = _g(1e3), mp.mpf(1e-6), mp.mpf(10)
    sol = Solver(Circuit(_net("rc", [_Vdc("V", "in", 10.0), _R("R", "in", "out", 1e3),
                                     dict(type="capacitor", name="C", nodes=["out", 0], value=1e-6)],
                              5e-3, 1e-5, ["V(out)"])), dps).run()
    tau, T = R * C, mp.mpf(5e-3)
    out = [("v(1 ms)", _at(sol, "V(out)", 1e-3), V * (1 - mp.exp(-mp.mpf(1e-3) / tau))),
           ("v(3 ms)", _at(sol, "V(out)", 3e-3), V * (1 - mp.exp(-mp.mpf(3e-3) / tau))),
           ("mean 0-5 ms", sol.window(["V(out)"], 0, T)["V(out)"]["mean"],
            V * (1 - tau / T * (1 - mp.exp(-T / tau))))]
    return out, sol


def case_rl(dps):
    R, L, V = _g(10.0), mp.mpf(1e-3), mp.mpf(10)
    sol = Solver(Circuit(_net("rl", [_Vdc("V", "in", 10.0), _R("R", "in", "a", 10.0),
                                     dict(type="inductor", name="L1", nodes=["a", 0], value=1e-3)],
                              5e-4, 1e-6, ["I(L1)"])), dps).run()
    f = lambda t: V / R * (1 - mp.exp(-mp.mpf(t) * R / L))  # noqa: E731
    return [("i(50 us)", _at(sol, "I(L1)", 5e-5), f(5e-5)),
            ("i(300 us)", _at(sol, "I(L1)", 3e-4), f(3e-4))], sol


def case_rlc(dps):
    R, L, C, V = mp.mpf(2.0), mp.mpf(1e-3), mp.mpf(10e-6), mp.mpf(10)
    sol = Solver(Circuit(_net("rlc", [_Vdc("V", "in", 10.0), _R("R", "in", "a", 2.0),
                                      dict(type="inductor", name="L1", nodes=["a", "b"], value=1e-3),
                                      dict(type="capacitor", name="C", nodes=["b", 0], value=10e-6)],
                              2e-3, 1e-6, ["I(L1)"])), dps).run()
    al = R / (2 * L)
    wd = mp.sqrt(1 / (L * C) - al ** 2)
    f = lambda t: V / (wd * L) * mp.exp(-al * t) * mp.sin(wd * t)  # noqa: E731
    return [(f"i({t * 1e6:g} us)", _at(sol, "I(L1)", t), f(mp.mpf(t))) for t in (1e-4, 7e-4, 1.9e-3)], sol


def case_sine_rc(dps):
    R, C, A, fr = _g(1e3), mp.mpf(1e-6), mp.mpf(5), mp.mpf(1e3)
    sol = Solver(Circuit(_net("sine_rc", [
        dict(type="voltage_source", name="V", nodes=["in", 0],
             waveform=dict(type="sine", amplitude=5.0, frequency=1e3)),
        _R("R", "in", "out", 1e3), dict(type="capacitor", name="C", nodes=["out", 0], value=1e-6)],
        3e-3, 1e-6, ["V(out)"])), dps).run()
    w, tau = 2 * mp.pi * fr, R * C
    ph = mp.atan(w * tau)

    def f(t):
        return A / mp.sqrt(1 + (w * tau) ** 2) * (mp.sin(w * t - ph) + mp.sin(ph) * mp.exp(-t / tau))
    return [(f"v({t * 1e3:g} ms)", _at(sol, "V(out)", t), f(mp.mpf(t))) for t in (7e-4, 2.3e-3)], sol


def bridge_net():
    def D(n, a, k):
        return dict(type="diode", name=n, nodes=[a, k], g_on=1e3, g_off=1e-9)
    return _net("bridge", [
        dict(type="voltage_source", name="Vac", nodes=["s", 0],
             waveform=dict(type="sine", amplitude=325.0, frequency=50.0)),
        _R("Rs", "s", "a", 0.5), dict(type="inductor", name="Ls", nodes=["a", "ac1"], value=1e-3),
        D("D1", "ac1", "dp"), D("D2", 0, "dp"), D("D3", "dn", "ac1"), D("D4", "dn", 0),
        dict(type="capacitor", name="C", nodes=["dp", "dn"], value=470e-6),
        _R("RL", "dp", "dn", 100.0), _R("Rtie", "dn", 0, 1e6)], 0.04, 4e-6, ["V(dp,dn)"])


def commutations(sol, t_min=1e-12):
    """Mode changes after the start-up (the bridge script's count)."""
    return sum(1 for e in sol.events[1:] if e["prev"] != e["mode"] and e["t"] > t_min)


def case_bridge(dps):
    """bridge_ground_truth.py's numbers. That script takes 1/R exactly; the
    engine (and this tool on a netlist) stamps the double 1.0/R, which moves
    the bus by -2.0e-16 V (311.343881428885841499): the exact 1/R here."""
    ck = Circuit(bridge_net())
    R = {"Rs": 0.5, "RL": 100.0, "Rtie": 1e6}
    ck.res = [(nm, a, b, 1 / Fr(R[nm])) for nm, a, b, _ in ck.res]
    sol = Solver(ck, dps).run()
    bus = sol.window(["V(dp,dn)"], mp.mpf("0.03"), mp.mpf("0.04"))["V(dp,dn)"]["mean"]
    ev = [e["t"] for e in sol.events[1:] if e["prev"] != e["mode"]]
    return [("bus mean 30-40 ms", bus, mp.mpf("311.343881428885841697")),
            ("commutations", mp.mpf(commutations(sol)), mp.mpf(11)),
            ("event 1 (D1/D4 tie)", ev[0], mp.mpf("0.004453708135472069740740")),
            ("event 11", ev[10], mp.mpf("0.039993904164045823079728"))], sol


def _buck(L, dps):
    """Conductance-form buck; closed-form periodic orbit from its own state
    equations ([iL, vC, 1], v_sw from KCL) by mp.expm, the DCM turn-off by
    mp.findroot; the tool runs 3 periods from that orbit."""
    Vin, gon, goff, C, R, f, D = 24.0, 1e3, 1e-9, 100e-6, 10.0, 50e3, 0.4
    m = {k: mp.mpf(v) for k, v in dict(Vin=Vin, gon=gon, goff=goff, C=C, L=L).items()}
    m["R"] = _g(R)
    duty = on_windows(dict(type="pwm", v_high=10.0, v_low=0.0, frequency=f, duty=D), 5.0)[1]
    T, DT = mpf(Fr(1.0 / f)), mpf(Fr(1.0 / f) * Fr(duty))     # the schedule's own edges

    def Amat(gs, gd):
        s = gs + gd
        return mp.matrix([[-1 / (s * m["L"]), -1 / m["L"], gs * m["Vin"] / (s * m["L"])],
                          [1 / m["C"], -1 / (m["R"] * m["C"]), 0], [0, 0, 0]])

    A1, A2, A3 = Amat(m["gon"], m["goff"]), Amat(m["goff"], m["gon"]), Amat(m["goff"], m["goff"])

    def orbit(td):
        P = mp.expm(A3 * (T - DT - td)) * mp.expm(A2 * td) * mp.expm(A1 * DT)
        x = mp.lu_solve(mp.eye(2) - P[:2, :2], P[:2, 2])
        return x

    def resid(td):   # diode current zero: i_L = g_off Vin at DT + td
        x = orbit(td)
        y = mp.expm(A2 * td) * mp.expm(A1 * DT) * mp.matrix([x[0], x[1], 1])
        return y[0] - m["goff"] * m["Vin"]

    if resid(T - DT) > 0:
        td, x = T - DT, orbit(T - DT)          # CCM
    else:
        td = mp.findroot(resid, (mp.mpf(0), T - DT), solver="anderson")
        x = orbit(td)
    comps = [_Vdc("Vin", "vin", Vin),
             dict(type="voltage_source", name="Vg", nodes=["g", 0],
                  waveform=dict(type="pwm", v_high=10.0, v_low=0.0, frequency=f, duty=D)),
             dict(type="vcswitch", name="S", nodes=["g", "vin", "sw"], v_threshold=5.0,
                  g_on=gon, g_off=goff),
             dict(type="diode", name="D", nodes=[0, "sw"], g_on=gon, g_off=goff),
             dict(type="inductor", name="L1", nodes=["sw", "out"], value=L, ic=float(x[0])),
             dict(type="capacitor", name="C", nodes=["out", 0], value=C, ic=float(x[1])),
             _R("R", "out", 0, R)]
    ck = Circuit(_net("buck", comps, 3 / f, 1e-7, ["V(out)", "I(L1)"]))
    # start exactly on the orbit, not on its float64 rounding
    ck.inds[0] = ck.inds[0][:4] + (fr_of(x[0]),)
    ck.caps[0] = ck.caps[0][:4] + (fr_of(x[1]),)
    sol = Solver(ck, dps).run()
    tE = 3 * T
    out = [("i_L(3T)", _at(sol, "I(L1)", tE), x[0]), ("v_C(3T)", _at(sol, "V(out)", tE), x[1])]
    offs = [e["t"] for e in sol.events[1:] if e["cause"].startswith("D ")]
    if offs:
        out.append(("diode turn-off (1st period)", offs[0], DT + td))
    return out, sol


def case_buck_ccm(dps):
    return _buck(1e-3, dps)


def case_buck_dcm(dps):
    return _buck(10e-6, dps)


def case_halfwave_rl(dps):
    """Ideal diode, R-L load: extinction angle from
    sin(b - phi) + sin(phi) e^{-b/(w tau)} = 0."""
    Vm, f, R, L = 100.0, 50.0, 10.0, 50e-3
    sol = Solver(Circuit(_net("halfwave", [
        dict(type="voltage_source", name="V", nodes=["in", 0],
             waveform=dict(type="sine", amplitude=Vm, frequency=f)),
        dict(type="diode", name="D", nodes=["in", "a"], g_on=1e3, g_off=1e-9),
        _R("R", "a", "b", R), dict(type="inductor", name="L1", nodes=["b", 0], value=L)],
        0.02, 1e-5, ["I(L1)"]), form="ideal"), dps).run()
    w = 2 * mp.pi * mp.mpf(f)
    R = _g(R)
    ph = mp.atan(w * mp.mpf(L) / R)
    wt = w * mp.mpf(L) / R
    b = mp.findroot(lambda b: mp.sin(b - ph) + mp.sin(ph) * mp.exp(-b / wt), (mp.pi, 2 * mp.pi),
                    solver="anderson")
    Z = mp.sqrt(mp.mpf(R) ** 2 + (w * mp.mpf(L)) ** 2)
    t1 = mp.mpf("0.004")
    i1 = mp.mpf(Vm) / Z * (mp.sin(w * t1 - ph) + mp.sin(ph) * mp.exp(-t1 * mp.mpf(R) / mp.mpf(L)))
    ext = next(e["t"] for e in sol.events[1:] if e["cause"] == "D root" and e["t"] > 1e-12)
    return [("extinction time", ext, b / w), ("i(4 ms)", _at(sol, "I(L1)", t1), i1)], sol


def case_cap_jump(dps):
    """Two capacitors closed by an ideal switch: charge conservation and the
    impulsive loss C1 C2 / (C1 + C2) dv^2 / 2."""
    C1, C2, g, t0 = mp.mpf(1e-6), mp.mpf(2e-6), mp.mpf(1e-9), mp.mpf(1e-6)
    sol = Solver(Circuit(_net("capjump", [
        dict(type="voltage_source", name="Vg", nodes=["g", 0],
             waveform=dict(type="pulse", v_initial=0.0, v_pulse=10.0, t_delay=1e-6, t_width=1.0,
                           period=2.0)),
        dict(type="vcswitch", name="S", nodes=["g", "n1", "n2"], v_threshold=5.0, g_on=1e3,
             g_off=1e-9),
        dict(type="capacitor", name="C1", nodes=["n1", 0], value=1e-6, ic=10.0),
        dict(type="capacitor", name="C2", nodes=["n2", 0], value=2e-6)],
        2e-6, 1e-8, ["V(n1)"]), form="ideal"), dps).run()
    d0 = 10 * mp.exp(-g * (1 / C1 + 1 / C2) * t0)
    loss = C1 * C2 / (C1 + C2) * d0 ** 2 / 2
    return [("V after", _at(sol, "V(n1)", 2e-6), 10 * C1 / (C1 + C2)),
            ("impulsive loss", impulsive_loss(sol), loss)], sol


def case_src_jump(dps):
    """An ideal switch closing 10 V onto 1 uF: the source's impulsive work
    C V dv, the loss C dv^2 / 2 (dv: the jump, v(t0) from the OFF leakage)."""
    C, V, g, t0 = mp.mpf(1e-6), mp.mpf(10), mp.mpf(1e-9), mp.mpf(1e-6)
    sol = Solver(Circuit(_net("srcjump", [
        _Vdc("V", "a", 10.0),
        dict(type="voltage_source", name="Vg", nodes=["g", 0],
             waveform=dict(type="pulse", v_initial=0.0, v_pulse=10.0, t_delay=1e-6, t_width=1.0,
                           period=2.0)),
        dict(type="vcswitch", name="S", nodes=["g", "a", "n"], v_threshold=5.0, g_on=1e3,
             g_off=1e-9),
        dict(type="capacitor", name="C", nodes=["n", 0], value=1e-6)],
        2e-6, 1e-8, ["V(n)"]), form="ideal"), dps).run()
    dv = V * mp.exp(-g / C * t0)
    return [("V after", _at(sol, "V(n)", 2e-6), V),
            ("source impulsive work", sum(j["w_src"] for j in sol.jumps), C * V * dv),
            ("impulsive loss", impulsive_loss(sol), C * dv ** 2 / 2)], sol


def impulsive_loss(sol):
    """Energy lost in the state jumps: sources' impulsive work - stored change."""
    if any(j["w_src"] is None for j in sol.jumps):
        return None
    return sum((j["w_src"] - j["dE"] for j in sol.jumps), mp.mpf(0))


def case_ramp(dps):
    """Ideal switch closing 10 V onto 1 mH (a defective mode: the expm path):
    i = i(t0) + V (t - t0) / L; i(t0) from the OFF leakage g_off."""
    V, L, g, t0 = mp.mpf(10), mp.mpf(1e-3), mp.mpf(1e-9), mp.mpf(1e-6)
    sol = Solver(Circuit(_net("ramp", [
        _Vdc("V", "in", 10.0),
        dict(type="voltage_source", name="Vg", nodes=["g", 0],
             waveform=dict(type="pulse", v_initial=0.0, v_pulse=10.0, t_delay=1e-6, t_width=1.0,
                           period=2.0)),
        dict(type="vcswitch", name="S", nodes=["g", "in", "a"], v_threshold=5.0, g_on=1e3,
             g_off=1e-9),
        dict(type="inductor", name="L1", nodes=["a", 0], value=1e-3)],
        5e-6, 1e-8, ["I(L1)"]), form="ideal"), dps).run()
    tau = L * g                    # OFF: L in series with 1/g_off, i = g V (1 - e^{-t/(L g)})
    i0 = g * V * (1 - mp.exp(-t0 / tau))
    t1 = mp.mpf(5e-6)
    return [("i(5 us)", _at(sol, "I(L1)", t1), i0 + V * (t1 - t0) / L),
            ("i mean 1-5 us", sol.window(["I(L1)"], t0, t1)["I(L1)"]["mean"],
             i0 + V * (t1 - t0) / (2 * L))], sol


def case_rc_edge(dps):
    """RC driven by a pulse with a 100 us rising edge: superposition of two
    ramps, v = a [r(t) - r(t - tr)], r(t) = t - tau (1 - e^{-t/tau})."""
    R, C, tr, V = _g(1e3), mp.mpf(1e-6), mp.mpf(1e-4), mp.mpf(5)
    sol = Solver(Circuit(_net("rc_edge", [
        dict(type="voltage_source", name="Vp", nodes=["in", 0],
             waveform=dict(type="pulse", v_initial=0.0, v_pulse=5.0, t_rise=1e-4, t_fall=1e-4,
                           t_width=1.0, period=3.0)),
        _R("R", "in", "out", 1e3), dict(type="capacitor", name="C", nodes=["out", 0], value=1e-6)],
        2e-3, 1e-6, ["V(out)"])), dps).run()
    tau, a = R * C, V / tr

    def r(t):
        return t - tau * (1 - mp.exp(-t / tau)) if t > 0 else mp.mpf(0)
    return [(f"v({t * 1e6:g} us)", _at(sol, "V(out)", t), a * (r(mp.mpf(t)) - r(mp.mpf(t) - tr)))
            for t in (5e-5, 1.5e-3)], sol


CASES = [("ideal L ramp (expm path)", case_ramp), ("RC ramp edge", case_rc_edge), ("RC step", case_rc), ("RL step", case_rl), ("RLC step", case_rlc),
         ("RC sine", case_sine_rc), ("floating bridge", case_bridge),
         ("buck CCM orbit", case_buck_ccm), ("buck DCM orbit", case_buck_dcm),
         ("ideal half-wave RL", case_halfwave_rl), ("capacitor jump", case_cap_jump),
         ("source jump", case_src_jump)]


def validate(dps=50, names=None):
    rows = []
    for name, fn in CASES:
        if names and name not in names:
            continue
        t0 = time.time()
        mp.mp.dps = dps                      # before any closed form is evaluated
        checks, sol = fn(dps)
        en = sol.energy()
        for what, got, ref in checks:
            rows.append((name, what, got, ref, _rel(got, ref)))
        rows.append((name, "energy balance", en["balance_rel"], mp.mpf(0), abs(en["balance_rel"])))
        print(f"  {name}: {time.time() - t0:.1f} s", file=sys.stderr)
    return rows


# ------------------------------------------------------------ per circuit --

TRUTH_DIR = Path(__file__).resolve().parents[1] / "truth"


def _harness_metrics():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import metrics  # noqa: E402 - pure numpy, no pulsim
    return metrics


def err_rms_symmetric(a, b, abs_scale=0.0):
    """metrics.err_rms_symmetric of feat/core-a-ideal-switch (the evidence's
    `u` / `e cross`): both traces lose their initial sample(s), union grid,
    linear interpolation, span = the larger peak-to-peak (floored)."""
    import numpy as np

    def cut(t, y):
        k = int(np.searchsorted(t, t[0], side="right"))
        return t[k:], y[k:]
    (ta, ya), (tb, yb) = cut(*a), cut(*b)
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    g = np.union1d(ta, tb)
    g = g[(g >= lo) & (g <= hi)]
    va, vb = np.interp(g, ta, ya), np.interp(g, tb, yb)
    span = max(float(np.ptp(va)), float(np.ptp(vb)), abs_scale)
    return float(np.sqrt(np.trapezoid((va - vb) ** 2, g) / (g[-1] - g[0])) / span)


def _floats(sol, cols, ts):
    import numpy as np
    v = sol.sample(cols, ts)
    t = np.array([float(x) for x in ts])
    for i in range(1, len(t)):          # right limit one ulp after the left one
        if t[i] <= t[i - 1]:
            t[i] = np.nextafter(t[i - 1], math.inf)
    return t, {c: np.array([float(x) for x in v[c]]) for c in cols}


def truth(path, form="conductance", dps=(30, 60), out_dir=TRUTH_DIR, write=True, verbose=False,
          store=8):
    """Solve one benchmark circuit at two precisions; return the record of
    EXACT_TRUTH.md and write <id>[.ideal].csv.gz + .json under out_dir."""
    import numpy as np
    rel = str(Path(path)).split("benchmarks/")[-1]
    rec = dict(circuit=rel, form=form, supported=True)
    try:
        ck = Circuit(path, form)
    except Unsupported as ex:
        return dict(rec, supported=False, why=str(ex))
    cols = ck.observables
    rec["observables"] = cols
    sols, t0 = {}, time.time()
    for d in dps:
        try:
            sols[d] = Solver(Circuit(path, form), d).run(verbose)
        except Unsupported as ex:
            return dict(rec, supported=False, why=str(ex))
        except RuntimeError as ex:
            if dps[0] < 45:
                # the order-0 sign test sits at 10^-(dps/2) of the terms' scale: a
                # g_off spike (1e7 V on a 0.05 V signal) needs more digits
                return truth(path, form, tuple(d + 15 for d in dps), out_dir, write, verbose, store)
            return dict(rec, supported=False, why=f"mode selection failed: {str(ex)[:200]}")
    rec["dps"] = list(dps)
    rec["wall_s"] = round(time.time() - t0, 1)
    lo, hi = sols[dps[0]], sols[dps[-1]]
    mp.mp.dps = dps[-1]
    ev = [e for e in hi.events[1:] if e["prev"] != e["mode"]]
    rec["events"] = [dict(t=mp.nstr(e["t"], 25), change=flips(ck, e["prev"], e["mode"]),
                          cause=e["cause"], strict=e["n_strict"], candidates=e["n_candidates"],
                          consistent=len(e["consistent"])) for e in hi.events[1:]]
    rec["n_commutations"] = len(ev)
    rec["n_events"] = len(hi.events) - 1
    amb = [e for e in hi.events if len(e["consistent"]) > 1]
    rec["ambiguous"] = [dict(t=mp.nstr(e["t"], 20), modes=[flips(ck, None, ck.full_mode(
        [e["mode"][i] for i in ck.switches], dd)) for dd in e["consistent"]]) for e in amb]
    rec["weak_only"] = sum(1 for e in hi.events if e["n_strict"] == 0)
    rec["start_mode"] = flips(ck, None, hi.events[0]["mode"])
    rec["dc_start_modes"] = hi.n_dc
    rec["notes"] = hi.notes
    rec["jumps"] = [dict(t=mp.nstr(j["t"], 20), dE=mp.nstr(j["dE"], 12)) for j in hi.jumps]
    rec["impulsive_loss_J"] = mp.nstr(impulsive_loss(hi), 15) if impulsive_loss(hi) is not None else None
    rec["max_dae_residual"] = mp.nstr(max(d.dae_residual for d in hi.dyns.values()
                                          if isinstance(d, Dyn)), 3)
    rec["max_eigvec_cond"] = mp.nstr(max(d.cond for d in hi.dyns.values() if isinstance(d, Dyn)), 3)
    T = mpf(ck.t_end)
    wins = {"full": (mp.mpf(0), T), "last20": (T * mp.mpf("0.8"), T)}
    rec["windows"] = {}
    for wn, (a, b) in wins.items():
        wh, wl = hi.window(cols, a, b), lo.window(cols, a, b)
        rec["windows"][wn] = {c: None if wh[c] is None else dict(
            mean=mp.nstr(wh[c]["mean"], 25), rms=mp.nstr(wh[c]["rms"], 25),
            d_prec=mp.nstr(abs(wh[c]["mean"] - wl[c]["mean"]), 3)) for c in cols}
    en = hi.energy()
    rec["energy_J"] = {k: mp.nstr(v, 15) for k, v in en["per"].items()}
    rec["energy_balance_rel"] = mp.nstr(en["balance_rel"], 3)
    # precision doubling on the output grid
    # every float sample from the low run: the check below bounds it far under 1 ulp
    mp.mp.dps = dps[0]
    g1 = out_grid(lo, ck.dt / store)          # the stored trace: dt/store, events two-sided
    th, yh = _floats(lo, cols, g1)
    mp.mp.dps = dps[-1]
    # precision doubling on the netlist grid, away from events: a grid point on a
    # gate edge falls on either side of it depending on the precision
    near = sorted(e["t"] for e in hi.events[1:])
    gap = mpf(Fr(ck.dt)) * mp.mpf(10) ** -9
    gr = [mpf(k * Fr(ck.dt)) for k in range(int(ck.t_end / Fr(ck.dt)) + 1)]
    gr = [t for t in gr if not near or abs(near[min(bisect.bisect(near, t), len(near) - 1)] - t) > gap
          and abs(near[max(bisect.bisect(near, t) - 1, 0)] - t) > gap]
    mp.mp.dps = dps[0]
    vl = lo.sample(cols, [mpf(t) for t in gr])
    mp.mp.dps = dps[-1]
    vh = hi.sample(cols, gr)
    same_grid = len(lo.events) == len(hi.events)
    if same_grid:
        dmax = max(max(abs(a - b) for a, b in zip(vl[c], vh[c])) /
                   max(max(abs(x) for x in vh[c]), mp.mpf(10) ** -30) for c in cols)
        rec["precision_doubling_rel"] = mp.nstr(dmax, 3)
    else:
        rec["precision_doubling_rel"] = "event lists differ"
    tl = [e["t"] for e in lo.events[1:] if e["prev"] != e["mode"]]
    rec["event_time_diff_s"] = (mp.nstr(max([abs(a["t"] - b) for a, b in zip(ev, tl)] + [0]), 3)
                                if len(tl) == len(ev) else f"{len(tl)} vs {len(ev)} commutations")
    # stored-trace uncertainty and the uniform-grid floor (harness measures)
    met = _harness_metrics()
    mp.mp.dps = dps[0]
    g4 = out_grid(lo, ck.dt / (4 * store))
    t4, y4 = _floats(lo, cols, g4)
    n = int(ck.t_end / Fr(ck.dt))
    gu = [mpf(k * Fr(ck.dt)) for k in range(n + 1)]
    tu, yu = _floats(lo, cols, gu)
    u_store, floor = 0.0, 0.0
    for c in cols:
        sc = max(1e-3 * float(np.abs(y4[c]).max()), 1e-9)
        u_store = max(u_store, err_rms_symmetric((th, yh[c]), (t4, y4[c]), abs_scale=sc))
        floor = max(floor, met.err_rms(tu, yu[c], (t4, y4[c]), abs_scale=sc, exclude_t0=True))
    rec["u_stored_trace"] = u_store
    rec["uniform_dt_floor"] = floor
    rec["dt"] = ck.dt
    rec["stored_dt"] = ck.dt / store
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(rel).stem + (".ideal" if form == "ideal" else "")
        with gzip.open(out_dir / f"{stem}.csv.gz", "wt", encoding="utf-8", newline="") as f:
            f.write("time," + ",".join(cols) + "\n")
            # values to 13 digits: the interpolation error of the stored grid (u stored) is far larger
            for i in range(len(th)):
                f.write(f"{float(th[i])!r}," + ",".join(f"{float(yh[c][i]):.13g}" for c in cols) + "\n")
        (out_dir / f"{stem}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def compare(circuit, trace_csv, form="conductance", dps=30):
    """(gate err_rms, symmetric err) per column of an engine trace against
    the truth, the truth evaluated exactly on the union of the trace's grid
    and its own dt/32 grid (so its interpolation adds nothing measurable)."""
    import numpy as np
    sol = Solver(Circuit(circuit, form), dps).run()
    op = gzip.open if str(trace_csv).endswith(".gz") else open
    with op(trace_csv, "rt", encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        data = np.loadtxt(f, delimiter=",", ndmin=2)
    t = data[:, 0]
    met = _harness_metrics()
    g = sorted({mp.mpf(float(x)) for x in t} | set(out_grid(sol, sol.ck.dt / 32)))
    tg, yg = _floats(sol, [c for c in head[1:] if c in sol.ck.observables], g)
    out = {}
    for j, c in enumerate(head[1:], 1):
        if c not in yg:
            continue
        sc = max(1e-3 * float(np.abs(yg[c]).max()), 1e-9)
        out[c] = (met.err_rms(t, data[:, j], (tg, yg[c]), abs_scale=sc, exclude_t0=True),
                  err_rms_symmetric((t, data[:, j]), (tg, yg[c]), abs_scale=sc))
    return out


REPORT = Path(__file__).resolve().parents[1] / "EXACT_TRUTH.md"


def _tol(entry, switched):
    """The harness tolerances (differential_runner.default_tol + triage tol:)."""
    d = {e: (1e-3 if not switched else (2e-2 if e == "pwl" else 1e-3)) for e in ("pwl", "trbdf2", "dsed")}
    d.update(entry.get("tol") or {})
    return d


def report(out=REPORT, truth_dir=TRUTH_DIR, dps=50):
    """EXACT_TRUTH.md from the records in truth_dir (everything below an
    existing '## Notes' heading is kept)."""
    root = Path(__file__).resolve().parents[1]
    tri = yaml.safe_load((root / "manifests" / "v2_triage.yaml").read_text(encoding="utf-8"))["circuits"]
    L = ["# Exact truth for the PWL benchmark circuits", "",
         "Generated by `benchmarks/tools/exact_truth.py report` (do not edit above `## Notes`).",
         "The tool is independent of pulsim: it parses each benchmark YAML itself and solves the",
         "simulator's piecewise-linear laws exactly in mpmath (method: the module docstring).",
         "`conductance` is the netlist as written; `ideal` is the ideal twin of",
         "`ideal_switch_flip_evidence.py` (R_on = 0 switches and diodes, V_f = 0, g_off kept).",
         "Truth waveforms: `benchmarks/truth/<circuit>[.ideal].csv.gz` (time + the harness's",
         "observables, every event two-sided: the right limit one ulp after the left) with a",
         "`.json` record. They do not replace the harness's references.", "",
         "## Validation (closed forms, independent of the MNA path)", "",
         "| case | check | tool | closed form | rel. diff |", "|---|---|---|---|---|"]
    for name, what, got, ref, rd in validate(dps):
        L.append(f"| {name} | {what} | {mp.nstr(got, 22)} | {mp.nstr(ref, 22)} | {mp.nstr(rd, 2)} |")
    L += ["", f"{dps} digits. Energy balance: (source energy - dissipation - stored-energy change +",
          "impulsive loss) / largest term, exact window integrals.", ""]
    # coverage
    rows, refused = [], []
    for rel, e in sorted(tri.items()):
        if e["status"] != "runs":
            continue
        try:
            Circuit(root / rel)
            rows.append(rel)
        except Unsupported as ex:
            refused.append((rel, str(ex)))
    L += ["## Coverage of the harness's 'runs' circuits", "",
          f"Parsed and in scope: {len(rows)} of {len(rows) + len(refused)}. Refused by name:", "",
          "| circuit | reason |", "|---|---|"]
    L += [f"| {r} | {w} |" for r, w in refused]
    # per circuit
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(truth_dir.glob("*.json"))]
    L += ["", "## Solved circuits", "",
          "`u stored`: harness symmetric distance of the stored trace (dt/8 grid) to the exact",
          "waveform on a dt/32 grid: the stored truth's own uncertainty as a reference. `floor`:",
          "the gate measure (`metrics.err_rms`) of the EXACT waveform sampled on the netlist's",
          "uniform dt grid (no event samples) against the truth: what a perfect fixed-step trace",
          "scores; a pwl cell near it disagrees through the measure, not the engine. `prec`: max",
          "difference of the 30- and 60-digit runs (45 / 75 where 30 cannot select the mode) on the netlist grid, relative to the column's",
          "peak. `modes`: events at which more than one mode is consistent (followed: the one",
          "with the fewest flips); `weak`: events where only a mode with a device pinned at its",
          "boundary (v = 0 identically) is consistent. `impulsive loss`: energy lost in the state",
          "jumps, the sources' impulsive work minus the stored-energy change (E dx = A s).", "",
          "| circuit | form | commutations | modes non-unique | weak | impulsive loss [J] | prec | "
          "energy bal. | u stored | floor | tol pwl / trbdf2 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in recs:
        if not r.get("supported"):
            L.append(f"| {r['circuit']} | {r['form']} | refused: {r.get('why', '')[:90]} |||||||||")
            continue
        e = tri.get(r["circuit"], {})
        ck = Circuit(root / r["circuit"])
        sw = bool(ck.devs) or any(wv.kind in ("pwm", "pulse") for _, _, _, wv in ck.vsrc)
        t = _tol(e, sw)
        L.append(f"| {r['circuit']} | {r['form']} | {r['n_commutations']} | {len(r['ambiguous'])} | "
                 f"{r['weak_only']} | {r['impulsive_loss_J']} | {r['precision_doubling_rel']} | "
                 f"{r['energy_balance_rel']} | {r['u_stored_trace']:.2g} | {r['uniform_dt_floor']:.3g} | "
                 f"{t['pwl']:g} / {t['trbdf2']:g} |")
    L += ["", "Stored-reference error (the harness's trbdf2 rtol 1e-8 reference against this truth):",
          "not computed here (it needs one engine run per circuit; this tool never runs the",
          "engines). With a trace CSV `time,<cols>`:",
          "`exact_truth.py compare <circuit.yaml> <trace.csv> [--form ideal]` prints its gate and",
          "symmetric errors against the truth.", ""]
    keep = ""
    if out.exists():
        old = out.read_text(encoding="utf-8")
        if "\n## Notes" in old:
            keep = old[old.index("\n## Notes") + 1:]
    out.write_text("\n".join(L) + "\n" + (keep or "## Notes\n"), encoding="utf-8")


# ------------------------------------------------------------------ CLI -----

def _main_solve(a):
    t0 = time.time()
    sol = solve(a.circuit, a.form, a.dps, verbose=a.verbose)
    ck = sol.ck
    print(f"[{ck.id} form={a.form} dps={a.dps}: {len(sol.segs)} segments, "
          f"{len(sol.events) - 1} events, {time.time() - t0:.1f} s]")
    amb = [e for e in sol.events if len(e["consistent"]) > 1]
    print(f"ambiguous events: {len(amb)}; jumps: {len(sol.jumps)}")
    w1 = mpf(ck.t_end)
    for c, r in sol.window(ck.observables, mp.mpf(0), w1).items():
        if r:
            print(f"  {c}: mean {mp.nstr(r['mean'], 20)} rms {mp.nstr(r['rms'], 20)}")
    en = sol.energy()
    print("energy balance rel:", mp.nstr(en["balance_rel"], 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("circuit")
    s.add_argument("--form", default="conductance", choices=("conductance", "ideal"))
    s.add_argument("--dps", type=int, default=50)
    s.add_argument("--verbose", action="store_true")
    tr = sub.add_parser("truth")
    tr.add_argument("circuits", nargs="+")
    tr.add_argument("--form", default="conductance", choices=("conductance", "ideal"))
    tr.add_argument("--dps", default="30,60")
    cp = sub.add_parser("compare", help="harness error of a trace CSV (time,col,...) against "
                        "the exact truth evaluated on the union grid")
    cp.add_argument("circuit")
    cp.add_argument("trace")
    cp.add_argument("--form", default="conductance", choices=("conductance", "ideal"))
    cp.add_argument("--dps", type=int, default=30)
    rp = sub.add_parser("report")
    rp.add_argument("--dps", type=int, default=50)
    v = sub.add_parser("validate")
    v.add_argument("--dps", type=int, default=50)
    v.add_argument("--case", action="append")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if a.cmd == "solve":
        _main_solve(a)
    elif a.cmd == "truth":
        for c in a.circuits:
            r = truth(c, a.form, tuple(int(x) for x in a.dps.split(",")))
            keys = ("supported", "why", "n_commutations", "ambiguous", "precision_doubling_rel",
                    "energy_balance_rel", "u_stored_trace", "uniform_dt_floor", "wall_s")
            print(json.dumps({k: r.get(k) for k in keys if k in r}))
    elif a.cmd == "compare":
        for c, r in compare(a.circuit, a.trace, a.form, a.dps).items():
            print(f"{c}: gate err_rms {r[0]:.4g}  symmetric {r[1]:.4g}")
    elif a.cmd == "report":
        report(dps=a.dps)
    elif a.cmd == "validate":
        print("| case | check | tool | closed form | rel. diff |\n|---|---|---|---|---|")
        for name, what, got, ref, rd in validate(a.dps, a.case):
            print(f"| {name} | {what} | {mp.nstr(got, 22)} | {mp.nstr(ref, 22)} | {mp.nstr(rd, 2)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
