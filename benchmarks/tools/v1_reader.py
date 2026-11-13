"""Harness-only reader for the ``schema: pulsim-v1`` benchmark circuits.

The 2.0 kernel loader refuses v1 on purpose; this reader lives in the
benchmark harness, is never imported by ``pulsim``, and maps only what the
126 files under ``benchmarks/circuits`` use (design.md, decision 1).
``load(path)`` returns a :class:`BenchCase`; ``case.run(engine)`` builds a
fresh circuit and simulates it. A construct with no exact 2.0 equivalent
raises :class:`Unmapped` naming it -- the triage manifest records it as
``skipped`` with that text.

Semantics copied from the v1 kernel (git 990995e4, core/src/v1/):
- pwm source: ``phase`` in RADIANS advancing the carrier
  (``t + phase/2π·T``), on-time ``duty·T − dead_time``;
- vcswitch: ON while v(ctrl) > v_threshold (the tanh ``hysteresis`` is a
  Newton smoothing, ignored here); defaults g_on 1e3, g_off 1e-9;
- diode: binary, g_on 1e3 / g_off 1e-9; with ``is`` / ``n`` it is Shockley;
- transformer: ``turns_ratio`` = N_p/N_s, so 2.0 ``n`` = 1/turns_ratio;
- ``uic: false`` (or absent) starts from the DC operating point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

import pulsim

GROUND = {"0", "gnd", "GND"}
CONTROL_TYPES = {"pi_controller", "pwm_generator"}


class Unmapped(Exception):
    """A v1 construct the reader cannot map exactly to 2.0."""


def value(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    for suf, mul in (("meg", 1e6), ("f", 1e-15), ("p", 1e-12), ("n", 1e-9), ("u", 1e-6),
                     ("µ", 1e-6), ("m", 1e-3), ("k", 1e3), ("g", 1e9)):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * mul
    return float(s)


def _node(n: Any) -> str:
    s = str(n)
    return "gnd" if s in GROUND else s


@dataclass
class BenchCase:
    path: Path
    id: str
    t_end: float
    dt: float
    t_start: float
    uic: bool
    observables: List[str]
    validation: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    _components: List[dict] = field(default_factory=list, repr=False)

    def build(self):
        """Fresh ``(builder, switch_fn, step_observer)`` for one run."""
        return _build(self._components, [])

    def run(self, engine: str = "pwl", *, dt: Optional[float] = None, **kw):
        b, switch_fn, observer = self.build()
        args: Dict[str, Any] = dict(engine=engine, t_start=self.t_start)
        if switch_fn is not None:
            args["switch_fn"] = switch_fn
        if observer is not None:
            args["step_observer"] = observer
        if engine == "pwl":
            args["dt"] = dt if dt is not None else self.dt
        elif dt is not None:
            args["dt"] = dt
        x0 = None if self.uic else self._dc_start(b, switch_fn)
        if x0 is not None:
            args["initial_state"] = x0
        args.update(kw)
        return pulsim.simulate(b, self.t_end, **args)


    def _dc_start(self, b, switch_fn):
        """v1 without ``uic`` started from the DC operating point. Return it
        only when it gives a reactive state a non-zero value (otherwise it is
        the zero start every engine already takes); dsed refuses a full
        state vector, which is reported as its refusal."""
        names = b.state_var_names()
        op = pulsim.compute_dc_op(b, mask=switch_fn(self.t_start) if switch_fn else None)
        idx = {n: i for i, n in enumerate(names)}

        def v(n):
            return 0.0 if _node(n) == "gnd" else op[idx[f"V({_node(n)})"]]
        for c in self._components:
            if c["type"] in ("capacitor", "snubber_rc"):
                if abs(v(c["nodes"][0]) - v(c["nodes"][1])) > 1e-12:
                    return op
            elif c["type"] in ("inductor", "coupled_inductor"):
                if any(abs(op[i]) > 1e-12 for n, i in idx.items()
                       if n.startswith(f"I({c['name']}")):
                    return op
        return None


def observable(res, column: str):
    """Values of a v1 observable column (``V(node)``, ``I(name)``)."""
    kind, _, rest = column.partition("(")
    name = rest.rstrip(")")
    if kind.upper() == "V":
        if "," in name:
            a, c = (_node(x.strip()) for x in name.split(","))
            return _v(res, a) - _v(res, c)
        return _v(res, _node(name))
    if kind.upper() == "I":
        return res.i(name)
    raise Unmapped(f"observable {column!r}")


def _v(res, n):
    import numpy as np
    return np.zeros(len(res.times)) if n == "gnd" else np.asarray(res.v(n))


def load(path) -> BenchCase:
    path = Path(path)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    if d.get("schema") != "pulsim-v1":
        raise Unmapped(f"schema {d.get('schema')!r}")
    sim = d.get("simulation") or {}
    bm = d.get("benchmark") or {}
    comps = d.get("components") or []
    case = BenchCase(
        path=path,
        id=bm.get("id", path.stem),
        t_end=value(sim["tstop"]),
        dt=value(sim["dt"]),
        t_start=value(sim.get("tstart", 0.0)),
        uic=bool(sim.get("uic", False)),
        observables=[o["column"] for o in bm.get("observables", []) if "column" in o],
        validation=dict(bm.get("validation") or {}),
        _components=comps,
    )
    if case.t_start != 0.0:
        raise Unmapped(f"tstart {case.t_start}")
    # Build once now so an unmappable circuit fails at load time.
    _, _, _ = _build(comps, case.notes)
    return case


# ---------------------------------------------------------------- mapping ---

def _waveform_of(comps, ctrl: str):
    """The v1 waveform driving node ``ctrl`` (a source from ctrl to ground)."""
    for c in comps:
        if c["type"] == "voltage_source":
            a, k = (_node(n) for n in c["nodes"])
            if a == ctrl and k == "gnd":
                return c["waveform"]
    return None


def _on_windows(w: dict, thr: float):
    """(frequency, duty, phase_s) of the interval where the waveform > thr,
    or True / False for a constant gate."""
    kind = w.get("type")
    if kind == "dc":
        return value(w["value"]) > thr
    if kind == "pwm":
        hi, lo = value(w.get("v_high", 1.0)), value(w.get("v_low", 0.0))
        f = value(w.get("frequency", 10e3))
        T = 1.0 / f
        t_on = min(max(value(w.get("duty", 0.5)) * T - value(w.get("dead_time", 0.0)), 0.0), T)
        # v1 carrier: t_mod = (t + phase/2π·T) mod T, high while t_mod < t_on
        start = (-value(w.get("phase", 0.0)) / (2 * math.pi) * T) % T
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
            raise Unmapped("pulse gate that starts high or does not repeat")
        x = (thr - v0) / (v1 - v0)
        on, off = td + tr * x, td + tr + pw + tf * (1.0 - x)
        return 1.0 / per, (off - on) / per, on % per
    raise Unmapped(f"gate waveform {kind!r}")


def _build(comps: List[dict], notes: List[str]):
    b = pulsim.CircuitBuilder()
    gated: List[tuple] = []          # (name, gate spec)
    loops: List[tuple] = []          # (switch name, pi comp, pwm comp)
    newton = False
    by_name = {c["name"]: c for c in comps}
    pwm_targets = {c.get("target_component"): c for c in comps if c["type"] == "pwm_generator"}

    for c in comps:
        t, name = c["type"], c["name"]
        nodes = [_node(n) for n in c.get("nodes", [])]
        g_on = c.get("g_on")
        if g_on is None and c.get("ron") is not None:
            g_on = 1.0 / value(c["ron"])
        g_off = c.get("g_off")
        if g_off is None and c.get("roff") is not None:
            g_off = 1.0 / value(c["roff"])

        if t == "resistor":
            b.add_resistor(name, *nodes, value(c.get("value", c.get("resistance"))))
        elif t in ("capacitor", "inductor"):
            v = value(c.get("value", c.get("capacitance", c.get("inductance"))))
            ic = value(c.get("ic", 0.0))
            add = b.add_capacitor if t == "capacitor" else b.add_inductor
            # an explicit 0 IC is the default; passing it makes dsed refuse
            add(name, *nodes, v, ic) if ic != 0.0 else add(name, *nodes, v)
        elif t == "voltage_source":
            w = c["waveform"]
            kind = w.get("type")
            if kind == "dc":
                b.add_voltage_source(name, *nodes, value(w["value"]))
            elif kind == "sine":
                b.add_sine_voltage_source(name, *nodes, value(w.get("offset", 0.0)),
                                          value(w["amplitude"]), value(w["frequency"]),
                                          value(w.get("phase", 0.0)))
            elif kind == "pulse":
                b.add_pulse_voltage_source(
                    name, *nodes, value(w.get("v_initial", 0.0)), value(w["v_pulse"]),
                    value(w.get("t_delay", 0.0)), value(w["t_width"]), value(w.get("period", 0.0)),
                    value(w.get("t_rise", 0.0)), value(w.get("t_fall", 0.0)))
            elif kind == "pwm":
                f = value(w.get("frequency", 10e3))
                spec = _on_windows(dict(w, v_high=1.0, v_low=0.0), 0.5)
                if isinstance(spec, bool):
                    b.add_voltage_source(name, *nodes, value(w.get("v_high" if spec else "v_low", 0.0)))
                else:
                    _, duty, start = spec
                    b.add_pwm_voltage_source(name, *nodes, value(w.get("v_high", 1.0)),
                                             value(w.get("v_low", 0.0)), f, duty, start)
            else:
                raise Unmapped(f"voltage_source waveform {kind!r}")
        elif t == "current_source":
            b.add_current_source(name, *nodes, value(c["value"]))
        elif t == "diode":
            if "is" in c:
                b.add_shockley_diode(name, *nodes, I_S=value(c["is"]), n=value(c.get("n", 1.0)))
                newton = True
            else:
                b.add_diode(name, *nodes, value(g_on if g_on is not None else 1e3),
                            value(g_off if g_off is not None else 1e-9))
        elif t in ("switch", "vcswitch"):
            pos = nodes[1:] if t == "vcswitch" else nodes
            b.add_switch(name, *pos, value(g_on if g_on is not None else (1e3 if t == "vcswitch" else 1e6)),
                         value(g_off if g_off is not None else (1e-9 if t == "vcswitch" else 1e-12)))
            if name in pwm_targets:
                pwm = pwm_targets[name]
                pi = by_name.get(pwm.get("duty_from_channel"))
                if pi is None or pi["type"] != "pi_controller":
                    raise Unmapped(f"pwm_generator duty from {pwm.get('duty_from_channel')!r}")
                loops.append((name, pi, pwm))
            elif t == "switch":
                closed = bool(c.get("initial_state", c.get("initial_closed", False)))
                gated.append((name, closed))
            else:
                w = _waveform_of(comps, nodes[0])
                if w is None:
                    raise Unmapped(f"vcswitch {name} control node {nodes[0]!r} is driven by the circuit")
                gated.append((name, _on_windows(w, value(c.get("v_threshold", 2.5)))))
        elif t == "mosfet":
            if c.get("is_nmos", True) is False:
                raise Unmapped("p-channel mosfet (no 2.0 Level-1 PMOS)")
            if "loss" in c or "thermal" in c:
                raise Unmapped("mosfet loss/thermal block")
            d, s = nodes[1], nodes[2]
            b.add_mosfet_level1(name, d, s, nodes[0], K=value(c.get("kp", 1.0)),
                                V_T=value(c.get("vth", 2.0)), lambda_=value(c.get("lambda", 0.0)),
                                with_body_diode=False)
            newton = True
        elif t == "igbt":
            b.add_igbt_level1(name, nodes[1], nodes[2], nodes[0],
                              V_CE_sat=value(c.get("v_ce_sat", 1.5)),
                              R_CE_sat=1.0 / value(g_on if g_on is not None else 1e3),
                              V_T=value(c.get("vth", 5.0)))
            newton = True
        elif t == "transformer":
            b.add_ideal_transformer(name, *nodes, 1.0 / value(c.get("turns_ratio", c.get("ratio"))))
        elif t == "coupled_inductor":
            l1 = value(c.get("l1", 1e-3))
            b.add_transformer(name, *nodes, l1, value(c.get("l2", l1)), value(c.get("coupling", 0.98)))
        elif t == "snubber_rc":
            mid = f"{name}__mid"
            b.add_resistor(f"{name}__R", nodes[0], mid, value(c["resistance"]))
            ic = value(c.get("ic", 0.0))
            args = (f"{name}__C", mid, nodes[1], value(c["capacitance"]))
            b.add_capacitor(*args, ic) if ic != 0.0 else b.add_capacitor(*args)
        elif t == "saturable_inductor":
            raise Unmapped("saturable_inductor (v1 exponent law has no 2.0 equivalent)")
        elif t in CONTROL_TYPES:
            pass  # consumed through the switch it targets
        else:
            raise Unmapped(f"component type {t!r}")

    for c in comps:
        if c["type"] == "pwm_generator" and c.get("target_component") not in by_name:
            raise Unmapped(f"pwm_generator {c['name']} has no target")
        if c["type"] == "pi_controller" and not any(lp[1] is c for lp in loops):
            raise Unmapped(f"pi_controller {c['name']} does not drive a switch")

    n = b.graph.num_switches
    parts = []
    for name, spec in gated:
        k = b.switch_index_of(name)
        if spec is True:
            parts.append(pulsim.make_pwm_switch_fn(1.0, 1.0, k, n))
        elif spec is not False:
            f, duty, start = spec
            parts.append(pulsim.make_pwm_switch_fn(f, duty, k, n, start))
            notes.append(f"{name}: schedule f={f:g} Hz duty={duty:.6g} start={start:.6g} s")
    observer = None
    if loops:
        if len(loops) > 1:
            raise Unmapped("more than one closed loop")
        sw, pi, pwm = loops[0]
        ref, fb = (_node(x) for x in pi["nodes"][:2])
        wref = _waveform_of(comps, ref)
        if wref is None or wref.get("type") != "dc":
            raise Unmapped("pi_controller reference is not a DC source")
        lo = max(value(pi.get("output_min", 0.0)), value(pwm.get("duty_min", 0.0)))
        hi = min(value(pi.get("output_max", 1.0)), value(pwm.get("duty_max", 1.0)))
        ctrl = pulsim.PIController(Kp=value(pi.get("kp", 0.0)), Ki=value(pi.get("ki", 0.0)),
                                   output_min=lo, output_max=hi)
        fb_id = b.node_id_of(fb)
        loop = pulsim.bind_pi_to_switch(b, pi=ctrl, measured=lambda x, i=fb_id: x[i],
                                        setpoint=value(wref["value"]), switch=sw,
                                        freq=value(pwm["frequency"]))
        notes.append(f"{sw}: bind_pi_to_switch kp={ctrl.Kp} ki={ctrl.Ki} f={value(pwm['frequency']):g} "
                     "(PI updated once per PWM period; v1 updated it every step)")
        parts.append(loop.switch_fn)
        observer = loop.step_observer
    if newton:
        notes.append("Newton device present")
    switch_fn: Optional[Callable] = None
    if gated or loops:  # diodes ignore the mask; switches need an explicit one
        switch_fn = (parts[0] if len(parts) == 1 and observer is None
                     else pulsim.make_combined_switch_fn(n, parts) if parts
                     else pulsim.make_pwm_switch_fn(1.0, 0.0, 0, n))
    return b, switch_fn, observer
