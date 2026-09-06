"""Pulsim — SPICE netlist importer.

Parses a subset of SPICE (Berkeley / LTspice / ngspice flavour) and
emits a populated `pulsim.CircuitBuilder`. Goal is to let users
bring existing designs into v2 without rewriting them.

Supported elements (Phase E.10):
  * `R` — resistor:       ``R<name> <n+> <n-> <value>``
  * `C` — capacitor:      ``C<name> <n+> <n-> <value>``
  * `L` — inductor:       ``L<name> <n+> <n-> <value>``
  * `V` — voltage source: ``V<name> <n+> <n-> <DC|SINE|PULSE>(...)``
  * `D` — diode:          ``D<name> <anode> <cathode> <model> [area]``
    with its ``.MODEL`` card: ``D(IS N RS BV IBV ...)`` becomes the
    Shockley junction it describes (RS as a series resistor); an
    LTspice ``D(Ron Roff Vfwd)`` card becomes the PWL diode.
  * `M` — MOSFET:         ``M<name> <D> <G> <S> [<B>] <model> [W= L= M=]``
    with a LEVEL-1 ``NMOS`` card (Shichman–Hodges: K = KP·W_eff/(2·L_eff),
    V_T = VTO, λ = LAMBDA, RD/RS as series resistors) or an LTspice
    ``VDMOS`` card (K = KP/2, body diode from Is/N/Rb/BV, Rds).
  * `X` — subcircuit instance, flattened: ports bound, inner nodes and
    designators scoped as ``X1.X2.R1``; ``PARAMS:`` defaults and
    ``k=v`` on the X line honoured; local ``.MODEL`` cards shadow
    outer ones; ``.GLOBAL`` nodes and ``0``/``gnd`` are not scoped.
  * `K` — coupled L (deferred — magnetics need more work)
  * `Q` / `J` — BJT / JFET (deferred)
  * `.param` — parameter assignments (numeric only, no expressions)
  * `.model` / `.subckt` … `.ends` / `.global` / `.options DEFW DEFL`
  * `*` / `;` — comments
  * `.end` — terminator (ignored)

What is REFUSED by name rather than approximated (v2.0, audit C.5):
a ``D`` or ``M`` with no model name (SPICE refuses it too), a model
that is referenced but not defined (``.INC``/``.LIB`` are not
followed), a MOSFET card that is not LEVEL 1 or VDMOS, a PMOS or
``pchan`` device, a bulk tied away from the source with a body
effect, a diode with a high-injection knee (IKF) or a PWL card with
reverse breakdown, an ``AKO:`` model. Before this the importer built a
0.7 V PWL diode for every ``D`` and a 10 mΩ switch for every ``M``
whatever the card said — measured against ``D(IS=1e-9 N=1.5 RS=0.02)``
that is a 1 mV forward drop instead of 0.89 V at 2.4 A: conduction
loss under-reported by two orders of magnitude, with no warning.
Parameters that are dropped knowingly (TT/CJO junction dynamics,
gate capacitances, RG) are listed in one summary warning; pass
``strict=True`` to make any such drop a refusal.

Suffix units recognised (case-insensitive):
  * `T` 1e12,  `G` 1e9,  `MEG` 1e6,  `K` 1e3
  * `M` 1e-3,  `U` 1e-6 (µ), `N` 1e-9,  `P` 1e-12, `F` 1e-15

Typical usage:

    from pulsim import spice_to_builder
    b = spice_to_builder('''
        * RC low-pass filter
        V1 vin gnd DC 5
        R1 vin vout 1k
        C1 vout gnd 100n
        .end
    ''')
    res = p.simulate(b, t_end=10e-3, dt=1e-6)

For behavioural sources, expressions beyond ``{param}``, BJTs or
coupled inductors, fall back to the YAML loader or hand-build the
CircuitBuilder.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "SpiceElement",
    "SpiceModel",
    "SpiceSubckt",
    "parse_spice_value",
    "parse_spice_netlist",
    "spice_to_builder",
]


# =============================================================================
# SPICE value parsing — handles engineering suffixes
# =============================================================================

# Order matters: MEG (1e6) must be checked before M (1e-3).
_UNIT_SUFFIXES = [
    ("MEG", 1e6),
    ("T", 1e12),
    ("G", 1e9),
    ("K", 1e3),
    ("M", 1e-3),
    ("U", 1e-6),
    ("N", 1e-9),
    ("P", 1e-12),
    ("F", 1e-15),
]


def parse_spice_value(token: str,
                          params: Optional[Dict[str, float]] = None
                          ) -> float:
    """Parse a SPICE value (with engineering suffix) into a float.

    Accepts:
      * "100" → 100.0
      * "1k", "1K", "1Kohm" → 1000.0
      * "100n", "100nF" → 100e-9
      * "10MEG" → 1e7
      * "{Lval}" → params["Lval"] (curly-brace param reference)
      * "1.5e-3" → 0.0015 (no suffix, scientific notation)
    """
    s = str(token).strip()
    if not s:
        return 0.0
    # Curly-brace param reference: {name}.
    if s.startswith("{") and s.endswith("}"):
        if params is None:
            raise ValueError(
                f"parse_spice_value: param reference {s!r} but no "
                f"params provided")
        name = s[1:-1].strip()
        if name not in params:
            raise KeyError(f"undefined SPICE param {name!r}")
        return float(params[name])

    # Strip trailing unit text (e.g. "1kohm" → "1k").
    s_upper = s.upper()
    # Try suffixes in order.
    for suffix, mul in _UNIT_SUFFIXES:
        if s_upper.endswith(suffix) or \
            any(s_upper.endswith(suffix + u) for u in
                  ("OHM", "OHMS", "F", "H", "V", "A")):
            # Strip the suffix + any trailing unit.
            for u in ("OHM", "OHMS", "F", "H", "V", "A", ""):
                if s_upper.endswith(suffix + u):
                    numpart = s_upper[: -(len(suffix) + len(u))]
                    try:
                        return float(numpart) * mul
                    except ValueError:
                        continue
    # Plain number (possibly scientific).
    # Strip optional trailing unit characters.
    for u in ("OHM", "OHMS", "F", "H", "V", "A"):
        if s_upper.endswith(u):
            s_upper = s_upper[: -len(u)]
            break
    return float(s_upper)


# =============================================================================
# Element dataclasses + tokeniser
# =============================================================================

class _Scope(dict):
    """A parameter scope. SPICE names are case-insensitive."""

    def __init__(self, *args, **kw):
        super().__init__()
        self.update(*args, **kw)

    def __setitem__(self, k, v):
        super().__setitem__(str(k).upper(), v)

    def __getitem__(self, k):
        return super().__getitem__(str(k).upper())

    def __contains__(self, k):
        return super().__contains__(str(k).upper())

    def get(self, k, default=None):
        return super().get(str(k).upper(), default)

    def update(self, *args, **kw):
        for k, v in dict(*args, **kw).items():
            self[k] = v


_NUM_UNIT_RE = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)([A-Za-z]*)$")


def _num(token: str, scope: Optional[Dict[str, float]]) -> float:
    """A SPICE number: engineering suffix, trailing unit letters
    (``50ns``, ``10Vdc``), ``{param}``, or a bare parameter name."""
    s = str(token).strip()
    if s.startswith("{") and s.endswith("}"):
        name = s[1:-1].strip()
        if scope is None or name not in scope:
            raise KeyError(f"undefined SPICE param {name!r}")
        return float(scope[name])
    if scope is not None and s in scope and not _NUM_UNIT_RE.match(s):
        return float(scope[s])
    try:
        return parse_spice_value(s, scope)
    except ValueError:
        m = _NUM_UNIT_RE.match(s)
        if not m:
            raise
        num, word = m.group(1), m.group(2).upper()
        mul = 1.0
        for suf, k in _UNIT_SUFFIXES:
            if word.startswith(suf):
                mul = k
                break
        return float(num) * mul


@dataclass
class SpiceModel:
    """A ``.MODEL <name> <type>(<params>)`` card. Bare words inside
    the parentheses (LTspice's ``pchan``…) land in ``flags``."""
    name: str
    type: str
    params: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    raw_line: str = ""


@dataclass
class SpiceSubckt:
    """``.SUBCKT <name> <ports...> [PARAMS: k=v ...]`` … ``.ENDS``.
    The body is kept as raw lines and parsed per instance, in that
    instance's parameter scope; nested definitions are scoped to it."""
    name: str
    ports: List[str] = field(default_factory=list)
    param_defaults: Dict[str, str] = field(default_factory=dict)
    body: List[str] = field(default_factory=list)
    subckts: Dict[str, "SpiceSubckt"] = field(default_factory=dict)


@dataclass
class SpiceElement:
    """One parsed SPICE element. `kind` is the first letter (R, C,
    L, V, D, M, …); `name` is the rest of the designator on its own
    line; `designator` is the full, flattened name (``XA.X1.R1`` for
    an element inside a subcircuit instance). `nodes` are the
    flattened node names; `tokens` the remainder (value + model +
    options); `model` the resolved ``.MODEL`` card, if any; `scope`
    the parameters in effect where the element was written."""
    kind: str
    name: str
    nodes: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    raw_line: str = ""
    designator: str = ""
    model: Optional[SpiceModel] = None
    model_name: Optional[str] = None
    scope: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.designator:
            self.designator = f"{self.kind}{self.name}"


def _strip_comment(line: str) -> str:
    # Inline comment marker: ';' splits the line; SPICE convention.
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()


def _join_continuations(text: str) -> List[str]:
    lines: List[str] = []
    buffer: Optional[str] = None
    for raw in text.splitlines():
        ln = raw.rstrip()
        if not ln or ln.startswith("*"):
            continue
        if ln.startswith("+"):
            if buffer is None:
                continue
            buffer += " " + ln[1:].strip()
        else:
            if buffer is not None:
                lines.append(buffer)
            buffer = ln
    if buffer is not None:
        lines.append(buffer)
    return lines


_MODEL_RE = re.compile(r"^\.MODEL\s+(?P<name>\S+)\s+(?P<rest>.*)$", re.IGNORECASE | re.DOTALL)


def _parse_model_card(line: str, scope: Dict[str, float]) -> SpiceModel:
    m = _MODEL_RE.match(line.strip())
    if not m:
        raise ValueError(f"unparseable .MODEL card: {line!r}")
    name, rest = m.group("name"), m.group("rest").strip()
    if rest.upper().startswith("AKO"):
        raise ValueError(
            f".MODEL {name}: AKO (inherit another model's parameters) is not "
            "supported; expand the base model's parameters into this card.")
    tm = re.match(r"^([A-Za-z]+)\s*(?:\((.*)\)|(.*))$", rest, re.DOTALL)
    if not tm:
        raise ValueError(f"unparseable .MODEL card: {line!r}")
    mtype = tm.group(1).upper()
    body = tm.group(2) if tm.group(2) is not None else (tm.group(3) or "")
    body = re.sub(r"\s*=\s*", "=", body.strip())
    params: Dict[str, float] = {}
    flags: List[str] = []
    for tok in re.split(r"[\s,]+", body):
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                params[k.upper()] = _num(v, scope)
            except (ValueError, KeyError) as e:
                raise ValueError(f".MODEL {name}: {k}={v!r} is not a number ({e}).") from e
        else:
            flags.append(tok.lower())
    return SpiceModel(name=name, type=mtype, params=params, flags=flags, raw_line=line)


_NODE_COUNT = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "Q": 3, "J": 3,
               "K": 0, "E": 4, "G": 4, "F": 2, "H": 2}


def _parse_element_line(ln: str) -> SpiceElement:
    tokens = ln.split()
    designator = tokens[0]
    kind = designator[0].upper()
    name = designator[1:]
    if kind == "X":
        plain = [t for t in tokens[1:] if "=" not in t and not t.upper().startswith("PARAMS")]
        assigns = [t for t in tokens[1:] if "=" in t]
        if not plain:
            raise ValueError(f"{designator}: no subcircuit name on the X line: {ln!r}")
        return SpiceElement(kind=kind, name=name, nodes=plain[:-1],
                            tokens=[plain[-1]] + assigns, raw_line=ln)
    if kind == "M":
        plain = [t for t in tokens[1:] if "=" not in t]
        n_nodes = len(plain) - 1
        if n_nodes not in (3, 4):
            raise ValueError(
                f"{designator}: a MOSFET line is 'M<name> D G S [B] <model> "
                f"[W= L= M=]'; got {ln!r}")
    else:
        n_nodes = _NODE_COUNT.get(kind, 2)
    nodes = tokens[1:1 + n_nodes]
    rest = tokens[1 + n_nodes:]
    return SpiceElement(kind=kind, name=name, nodes=nodes, tokens=rest, raw_line=ln)


def _model_token(el: SpiceElement) -> Optional[str]:
    for t in el.tokens:
        if "=" in t:
            continue
        try:
            parse_spice_value(t)
            continue                      # an area factor / multiplier
        except ValueError:
            return t
    return None


def _map_node(n: str, prefix: str, port_map: Dict[str, str], globals_: set) -> str:
    nu = n.upper()
    if nu in ("0", "GND") or nu in globals_:
        return n
    if nu in port_map:
        return port_map[nu]
    return prefix + n


def _assigns(tokens: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s,]+)", " ".join(tokens)):
        out[kv.group(1)] = kv.group(2)
    return out


def _expand(sub: SpiceSubckt, scope: _Scope, models_chain: List[Dict[str, SpiceModel]],
            subckt_chain: List[Dict[str, SpiceSubckt]], prefix: str,
            port_map: Dict[str, str], globals_: set, stack: Tuple[str, ...],
            out: List[SpiceElement]) -> None:
    """Flatten one subcircuit body (or the top level) into `out`."""
    for ln in sub.body:
        if ln.upper().startswith(".PARAM"):
            for k, v in _assigns(ln.split()[1:]).items():
                scope[k] = _num(v, scope)
    local_models: Dict[str, SpiceModel] = {}
    for ln in sub.body:
        if ln.upper().startswith(".MODEL"):
            m = _parse_model_card(ln, scope)
            local_models[m.name.upper()] = m
    chain = [local_models] + models_chain
    schain = [sub.subckts] + subckt_chain
    for ln in sub.body:
        if ln.startswith("."):
            continue
        el = _parse_element_line(ln)
        if el.kind == "X":
            sname = el.tokens[0]
            inst = prefix + el.designator
            target = next((d[sname.upper()] for d in schain if sname.upper() in d), None)
            if target is None:
                known = sorted(k for d in schain for k in d)
                raise ValueError(
                    f"{inst}: subcircuit {sname!r} is not defined in this netlist "
                    f"(known: {known}). A subcircuit from an external .lib/.inc "
                    "must be pasted in; the importer does not follow include "
                    "paths.")
            if sname.upper() in stack:
                raise ValueError(f"{inst}: subcircuit {sname!r} instantiates itself.")
            if len(el.nodes) != len(target.ports):
                raise ValueError(
                    f"{inst}: {len(el.nodes)} nodes for subcircuit {sname!r}, "
                    f"which has {len(target.ports)} ports {target.ports}.")
            # Values on the X line are evaluated in the CALLER's scope;
            # the subcircuit's defaults in the same scope, then overridden.
            passed = {k: _num(v, scope) for k, v in _assigns(el.tokens[1:]).items()}
            inner = _Scope(scope)
            for k, raw in target.param_defaults.items():
                inner[k] = _num(raw, scope)
            inner.update(passed)
            mapped = [_map_node(n, prefix, port_map, globals_) for n in el.nodes]
            new_map = {pt.upper(): n for pt, n in zip(target.ports, mapped)}
            _expand(target, inner, chain, schain, inst + ".", new_map, globals_,
                    stack + (sname.upper(),), out)
            continue
        el.designator = prefix + el.designator
        el.nodes = [_map_node(n, prefix, port_map, globals_) for n in el.nodes]
        el.scope = scope
        if el.kind in ("D", "M", "Q", "J"):
            mn = _model_token(el)
            el.model_name = mn
            if mn is not None:
                el.model = next((d[mn.upper()] for d in chain if mn.upper() in d), None)
        out.append(el)


def parse_spice_netlist(text: str, *, return_models: bool = False):
    """Parse a SPICE netlist string into a list of `SpiceElement`s
    plus the global parameters — and, with ``return_models=True``,
    the global ``.MODEL`` cards keyed by upper-cased name.

    ``.SUBCKT`` blocks are collected and every ``X`` instance is
    flattened (see :func:`spice_to_builder`); the elements come back
    with flattened designators and node names, and with their model
    card resolved through the subcircuit scopes. Multi-line
    continuation via `+` at the start of a line is supported.
    """
    lines = _join_continuations(text)
    top = SpiceSubckt(name="")
    params_raw: List[Tuple[str, str]] = []
    model_lines: List[str] = []
    globals_: set = set()
    options: Dict[str, str] = {}
    stack: List[SpiceSubckt] = []
    for ln in lines:
        ln = _strip_comment(ln)
        if not ln:
            continue
        u = ln.upper()
        if u.startswith(".SUBCKT"):
            toks = ln.split()
            if len(toks) < 2:
                raise ValueError(f".SUBCKT without a name: {ln!r}")
            ports = [t for t in toks[2:] if "=" not in t and not t.upper().startswith("PARAMS")]
            sub = SpiceSubckt(name=toks[1], ports=ports, param_defaults=_assigns(toks[2:]))
            stack.append(sub)
            continue
        if u.startswith(".ENDS"):
            if not stack:
                raise ValueError(".ENDS without a matching .SUBCKT")
            sub = stack.pop()
            (stack[-1].subckts if stack else top.subckts)[sub.name.upper()] = sub
            continue
        if stack:
            stack[-1].body.append(ln)
            continue
        if u.startswith(".PARAM"):
            params_raw.extend(_assigns(ln.split()[1:]).items())
            continue
        if u.startswith(".MODEL"):
            model_lines.append(ln)
            continue
        if u.startswith(".GLOBAL"):
            globals_.update(t.upper() for t in ln.split()[1:])
            continue
        if u.startswith(".OPTION"):
            for k, v in _assigns(ln.split()[1:]).items():
                if k.upper() in ("DEFW", "DEFL"):
                    options[k.upper()] = v
            continue
        if u.startswith(".INC") or u.startswith(".LIB"):
            warnings.warn(
                f"spice_import: {ln.split()[0]} is not followed — models and "
                "subcircuits defined in that file are unknown here; paste them "
                "into the netlist.", stacklevel=2)
            continue
        if u.startswith("."):
            # .END, .TRAN, .OP, … — analysis directives the simulator owns.
            continue
        top.body.append(ln)
    if stack:
        raise ValueError(f".SUBCKT {stack[-1].name} has no .ENDS")

    scope = _Scope()
    for k, v in params_raw:
        scope[k] = _num(v, scope)
    for k, v in options.items():
        scope[f"__{k}__"] = _num(v, scope)
    models: Dict[str, SpiceModel] = {}
    for ln in model_lines:
        m = _parse_model_card(ln, scope)
        models[m.name.upper()] = m

    elements: List[SpiceElement] = []
    _expand(top, scope, [models], [top.subckts], "", {}, globals_, (), elements)
    if return_models:
        return elements, scope, models
    return elements, scope


# =============================================================================
# SPICE → CircuitBuilder
# =============================================================================

def _normalize_node(name: str) -> str:
    """SPICE uses '0' for ground; v2 uses 'gnd'. Otherwise lowercase
    the node name (SPICE is case-insensitive but our v2 builder is
    case-sensitive)."""
    if name == "0":
        return "gnd"
    return name.lower()


def _parse_source_spec(tokens: List[str],
                       params: Dict[str, float],
                       ) -> Dict[str, Any]:
    """Parse the value spec for a V/I source. Returns a dict:
        {kind: "dc" | "sine" | "pulse", value: ...}

    Accepts:
      * "5"            → DC 5 V
      * "DC 5"         → DC 5 V
      * "SIN(0 10 50)" or "SINE(0 10 50 [td [df [phi]]])"
                       → SINE with DC offset, amplitude, freq, etc.
      * "PULSE(0 5 0 1u 1u 5u 10u)" → PULSE source
    """
    if not tokens:
        raise ValueError("source value tokens are empty")
    first = tokens[0].upper()

    flat = " ".join(tokens)
    m_sine = re.match(r"SINE?\s*\(\s*([^)]+)\)", flat, re.IGNORECASE)
    if m_sine:
        vals = [_num(t, params) for t in m_sine.group(1).split()]
        if len(vals) < 3:
            raise ValueError("SINE spec needs at least V_dc, V_amp, frequency")
        return {"kind": "sine", "v_dc": vals[0], "v_amplitude": vals[1],
                "frequency": vals[2], "phase": vals[5] if len(vals) >= 6 else 0.0}
    m_pulse = re.match(r"PULSE\s*\(\s*([^)]+)\)", flat, re.IGNORECASE)
    if m_pulse:
        vals = [_num(t, params) for t in m_pulse.group(1).split()]
        if len(vals) < 7:
            raise ValueError("PULSE spec needs V1 V2 td tr tf pw period")
        return {"kind": "pulse", "v_low": vals[0], "v_high": vals[1],
                "delay": vals[2], "rise_time": vals[3], "fall_time": vals[4],
                "pulse_width": vals[5], "period": vals[6]}
    if first == "DC":
        return {"kind": "dc", "value": _num(tokens[1], params)}
    return {"kind": "dc", "value": _num(tokens[0], params)}


_V_T = 0.025852   # thermal voltage at 300 K, as in the kernel's Shockley diode

_DIODE_KNOWN = {"IS", "N", "RS", "BV", "IBV", "CJO", "CJ0", "VJ", "M", "TT", "EG", "XTI",
                "FC", "TNOM", "KF", "AF", "IKF", "ISR", "NR", "NBV", "IBVL", "NBVL",
                "TBV1", "TBV2", "TRS1", "TRS2", "TIKF", "LEVEL"}
_DIODE_PWL = {"RON", "ROFF", "VFWD", "VREV", "RREV", "ILIMIT", "REVILIMIT", "EPSILON",
              "REVEPSILON"}
_L1_KNOWN = {"LEVEL", "VTO", "KP", "LAMBDA", "GAMMA", "PHI", "RD", "RS", "RSH", "CBD",
             "CBS", "IS", "JS", "PB", "CGSO", "CGDO", "CGBO", "TOX", "NSUB", "W", "L",
             "LD", "WD", "TNOM", "KF", "AF", "FC", "UO", "U0", "NSS", "TPG", "XJ", "MJ",
             "MJSW", "CJ", "CJSW"}
_L23_IGNORED = {"VMAX", "THETA", "ETA", "DELTA", "NFS", "UCRIT", "UEXP", "NEFF", "KAPPA",
                "XQC", "UTRA"}
_VDMOS_KNOWN = {"VTO", "KP", "LAMBDA", "RD", "RS", "RG", "RDS", "RB", "IS", "N", "BV",
                "IBV", "TT", "CJO", "CGDMIN", "CGDMAX", "A", "CGS", "M", "VJ", "MTRIODE",
                "SUBTHRES", "KSUBTHRES", "EG", "XTI", "TNOM", "NCHAN", "RQ", "VQ",
                "TRS1", "TRS2", "TRD1", "TRD2", "TRG1", "TRG2", "TRB1", "TRB2", "TKSUBTHRES1",
                "TKSUBTHRES2", "TVTO", "BEX", "TRDS1", "TRDS2", "TNOM", "TTAU", "MU", "TT",
                "VDS", "RON", "QG", "TEMP"}


def _inst_assigns(el: SpiceElement) -> Dict[str, float]:
    return {k.upper(): _num(v, el.scope) for k, v in _assigns(el.tokens).items()}


def _spice_diode(b, el: SpiceElement, nodes: List[str], notes: List[str],
                 drops: List[str]) -> None:
    d = el.designator
    if el.model_name is None:
        raise ValueError(
            f"{d}: no model name on the diode line ({el.raw_line!r}). SPICE "
            "itself refuses a D without a model, and this importer no longer "
            "substitutes a generic 0.7 V PWL diode for it — that hid a "
            "two-orders-of-magnitude conduction-loss error. Add a .MODEL card, "
            "or build the diode with add_diode()/add_shockley_diode() yourself.")
    mdl = el.model
    if mdl is None:
        raise ValueError(
            f"{d}: references model {el.model_name!r}, which is not defined in "
            "this netlist (nor in an enclosing subcircuit). Paste the .MODEL "
            "card in — .INC/.LIB files are not followed — or drop the "
            "reference and build the diode yourself.")
    if mdl.type != "D":
        raise ValueError(f"{d}: model {mdl.name!r} is of type {mdl.type}, not D.")
    p = mdl.params
    area = 1.0
    extra = [t for t in el.tokens if "=" not in t]
    if len(extra) >= 2:
        area = _num(extra[1], el.scope)
    inst = _inst_assigns(el)
    area *= inst.get("AREA", 1.0) * inst.get("M", 1.0)

    if any(k in p for k in _DIODE_PWL):
        # LTspice's piecewise-linear diode: Ron / Roff / Vfwd (+ reverse
        # breakdown and softening knobs).
        unknown = sorted(k for k in p if k not in _DIODE_PWL | {"TT", "CJO", "CJ0", "EPSILON"})
        if unknown:
            raise ValueError(
                f"{d}: PWL diode model {mdl.name!r} mixes {unknown} with "
                "Ron/Roff/Vfwd; this importer maps only the PWL set.")
        if "VREV" in p or "RREV" in p or "REVILIMIT" in p:
            raise ValueError(
                f"{d}: PWL diode model {mdl.name!r} sets reverse breakdown "
                "(Vrev/Rrev); Pulsim's PWL diode has no breakdown branch and "
                "a Zener-like use would be silently wrong.")
        if "ILIMIT" in p or "EPSILON" in p or "REVEPSILON" in p:
            drops.append(f"{d}: PWL diode softening knobs "
                         f"{sorted(k for k in p if k in ('ILIMIT', 'EPSILON', 'REVEPSILON'))} dropped")
        Ron = float(p.get("RON", 1.0))
        Roff = float(p.get("ROFF", 1.0 / 1e-12))
        Vfwd = float(p.get("VFWD", 0.0))
        import pulsim as _v2
        b.add_nonlinear_diode(d, nodes[0], nodes[1],
                              _v2.IdealDiodeParams(V_F0=Vfwd, R_d=Ron / area, G_off=area / Roff))
        return

    unknown = sorted(k for k in p if k not in _DIODE_KNOWN)
    if unknown:
        raise ValueError(
            f"{d}: model {mdl.name!r} has parameters this importer does not "
            f"understand: {unknown}.")
    if float(p.get("IKF", 0.0)) > 0:
        raise ValueError(
            f"{d}: model {mdl.name!r} sets IKF={p['IKF']:g} (high-injection "
            "knee): above IKF the forward law doubles its slope, which is "
            "exactly the conduction drop being mapped, and Pulsim's Shockley "
            "diode has no such knee. Refused rather than mapped without it.")
    if float(p.get("ISR", 0.0)) > 0:
        raise ValueError(
            f"{d}: model {mdl.name!r} sets ISR (recombination current); the "
            "low-current law is not a single exponential and is not mapped.")
    I_S = float(p.get("IS", 1e-14)) * area
    n = float(p.get("N", 1.0))
    RS = float(p.get("RS", 0.0)) / area
    BV = float(p.get("BV", 0.0))
    BV_p = 0.0
    if BV > 0:
        # SPICE: BV is where the reverse current equals IBV; the kernel's
        # knee is a mirrored exponential with current ~I_S there.
        IBV = float(p.get("IBV", 1e-3))
        BV_p = BV - n * _V_T * math.log(IBV / I_S) if IBV > I_S else BV
        if BV_p <= 0:
            raise ValueError(f"{d}: BV={BV} with IBV={IBV} and IS={I_S} gives no positive knee.")
        if abs(BV_p - BV) > 0.02 * BV:
            notes.append(f"{d}: BV {BV:g} V at IBV={IBV:g} A is a knee at {BV_p:.4g} V in the kernel's law")
    dyn = {k: p[k] for k in ("CJO", "CJ0", "TT") if float(p.get(k, 0.0)) > 0}
    if dyn:
        drops.append(
            f"{d}: model {mdl.name!r} carries {dyn} (junction capacitance / "
            "transit time): the Shockley diode built here has no reverse "
            "recovery and no junction charge, so its switching loss is "
            "under-reported. Use add_lauritzen_diode for recovery")
    if any(k in p for k in ("EG", "XTI", "TNOM")):
        notes.append(f"{d}: EG/XTI/TNOM present; Tj = TNOM assumed (I_S is not temperature-scaled)")
    anode = nodes[0]
    if RS > 0:
        mid = f"{d}.a"
        b.add_resistor(f"{d}.rs", nodes[0], mid, RS)
        anode = mid
    b.add_shockley_diode(d, anode, nodes[1], I_S=I_S, n=n, BV=BV_p)


def _spice_mosfet(b, el: SpiceElement, nodes: List[str], notes: List[str],
                  drops: List[str]) -> None:
    d = el.designator
    if el.model_name is None:
        raise ValueError(
            f"{d}: no model name on the MOSFET line ({el.raw_line!r}). SPICE "
            "refuses an M without a model, and this importer no longer "
            "substitutes a generic 10 mΩ switch for it. Add a .MODEL card, or "
            "build the switch with add_mosfet()/add_mosfet_level1() yourself.")
    mdl = el.model
    if mdl is None:
        raise ValueError(
            f"{d}: references model {el.model_name!r}, which is not defined in "
            "this netlist (a vendor MOSFET is usually a .SUBCKT in a .lib file "
            "— paste it in so it can be flattened). The importer will not "
            "substitute a generic switch for it.")
    drain, gate, source = nodes[0], nodes[1], nodes[2]
    bulk = nodes[3] if len(nodes) == 4 else None
    inst = _inst_assigns(el)
    mult = float(inst.get("M", 1.0))
    p = mdl.params
    if mdl.type == "VDMOS":
        _vdmos(b, el, d, drain, gate, source, inst, mult, mdl, notes, drops)
        return
    if mdl.type not in ("NMOS", "PMOS"):
        raise ValueError(f"{d}: model {mdl.name!r} is of type {mdl.type}, not NMOS/PMOS/VDMOS.")
    if mdl.type == "PMOS":
        raise ValueError(
            f"{d}: PMOS model {mdl.name!r} — Pulsim's level-1 device is "
            "n-channel and its law has no polarity flag; swapping drain and "
            "source does not invert the gate sense, and no node permutation "
            "gives V_SG > |VTO|. Mirror the circuit, or use the PWL switch.")
    level = int(p.get("LEVEL", 1))
    if level != 1:
        raise ValueError(
            f"{d}: model {mdl.name!r} is LEVEL {level}; only the Shichman–Hodges "
            "LEVEL 1 law maps onto add_mosfet_level1. A level-2/3 or BSIM card "
            "cannot be reduced to it without inventing numbers.")
    unknown = sorted(k for k in p if k not in _L1_KNOWN and k not in _L23_IGNORED)
    if unknown:
        raise ValueError(
            f"{d}: model {mdl.name!r} carries {unknown}, which the LEVEL-1 "
            "mapping has no place for.")
    ignored = sorted(k for k in p if k in _L23_IGNORED)
    if ignored:
        drops.append(f"{d}: {ignored} are level-2/3 parameters that SPICE ignores at LEVEL 1; ignored here too")
    if bulk is not None and bulk != source:
        if float(p.get("GAMMA", 0.0)) != 0.0:
            raise ValueError(
                f"{d}: bulk tied to {bulk!r}, not the source, with GAMMA="
                f"{p['GAMMA']:g}: the body effect shifts V_T with V_SB and "
                "Pulsim's level-1 device has no bulk terminal. Tie the bulk to "
                "the source or drop GAMMA.")
        notes.append(f"{d}: bulk {bulk!r} ignored (GAMMA = 0, no body effect)")
    if "VTO" not in p:
        raise ValueError(
            f"{d}: model {mdl.name!r} has no VTO (has {sorted(p)}); a threshold "
            "of SPICE's default 0 V is never a power device.")
    if "KP" in p:
        KP = float(p["KP"])
    elif ("UO" in p or "U0" in p) and "TOX" in p:
        uo = float(p.get("UO", p.get("U0")))
        KP = uo * 1e-4 * 3.9 * 8.854e-12 / float(p["TOX"])
        notes.append(f"{d}: KP = UO·ε_ox/TOX = {KP:.4g} A/V²")
    else:
        KP = 2e-5
        notes.append(f"{d}: model has no KP (nor UO+TOX); SPICE's default KP = 2e-5 A/V² used")
    defw = float(el.scope.get("__DEFW__", 100e-6))
    defl = float(el.scope.get("__DEFL__", 100e-6))
    W = float(inst.get("W", p.get("W", defw)))
    L = float(inst.get("L", p.get("L", defl)))
    Leff = L - 2.0 * float(p.get("LD", 0.0))
    Weff = W - 2.0 * float(p.get("WD", 0.0))
    if Leff <= 0 or Weff <= 0:
        raise ValueError(f"{d}: effective W/L not positive (W={W}, L={L}, LD/WD in the model).")
    K = KP * Weff / (2.0 * Leff) * mult
    RD = (float(p.get("RD", 0.0)) + float(p.get("RSH", 0.0)) * float(inst.get("NRD", 0.0))) / mult
    RS = (float(p.get("RS", 0.0)) + float(p.get("RSH", 0.0)) * float(inst.get("NRS", 0.0))) / mult
    explicit_body = False
    I_S_body = 0.0
    if "IS" in p:
        explicit_body = True
        I_S_body = float(p["IS"]) * mult
    elif "JS" in p:
        area = float(inst.get("AD", 0.0)) + float(inst.get("AS", 0.0))
        if area <= 0:
            raise ValueError(f"{d}: model gives JS (per-area) but the line has no AD/AS.")
        explicit_body = True
        I_S_body = float(p["JS"]) * area * mult
    caps = sorted(k for k in p if k in ("CBD", "CBS", "CGSO", "CGDO", "CGBO", "CJ", "CJSW"))
    if caps:
        drops.append(f"{d}: capacitances {caps} dropped: no switching transient in the level-1 device")
    dn, sn = drain, source
    if RD > 0:
        dn = f"{d}.d"
        b.add_resistor(f"{d}.rd", drain, dn, RD)
    if RS > 0:
        sn = f"{d}.s"
        b.add_resistor(f"{d}.rs", sn, source, RS)
    b.add_mosfet_level1(d, dn, sn, gate, K=K, V_T=float(p["VTO"]),
                        lambda_=float(p.get("LAMBDA", 0.0)),
                        with_body_diode=not explicit_body)
    if explicit_body:
        b.add_shockley_diode(f"{d}_body", sn, dn, I_S=I_S_body, n=1.0)
        notes.append(f"{d}: body diode from IS/JS ({I_S_body:.3g} A) built explicitly")


def _vdmos(b, el, d, drain, gate, source, inst, mult, mdl, notes, drops) -> None:
    """LTspice's VDMOS: 'the DC model is the same as a level 1
    monolithic MOSFET except that the length and width default to
    one so that transconductance can be directly specified'."""
    p = mdl.params
    if "pchan" in mdl.flags:
        raise ValueError(
            f"{d}: VDMOS model {mdl.name!r} is p-channel (pchan); Pulsim's "
            "level-1 device is n-channel. Mirror the circuit or use the PWL switch.")
    unknown = sorted(k for k in p if k not in _VDMOS_KNOWN)
    unknown_flags = sorted(f for f in mdl.flags if f not in ("nchan",))
    if unknown or unknown_flags:
        raise ValueError(
            f"{d}: VDMOS model {mdl.name!r} carries {unknown + unknown_flags}, "
            "which this importer does not understand.")
    if float(p.get("MTRIODE", 1.0)) != 1.0:
        raise ValueError(
            f"{d}: VDMOS model {mdl.name!r} sets mtriode={p['MTRIODE']:g}, which "
            "changes the triode-region law; the level-1 device has no such knob.")
    if "VTO" not in p or "KP" not in p:
        raise ValueError(f"{d}: VDMOS model {mdl.name!r} needs VTO and KP (has {sorted(p)}).")
    W = float(inst.get("W", 1.0))
    L = float(inst.get("L", 1.0))
    K = float(p["KP"]) / 2.0 * W / L * mult
    RD = float(p.get("RD", 0.0)) / mult
    RS = float(p.get("RS", 0.0)) / mult
    RDS = float(p.get("RDS", 0.0))
    if "RG" in p:
        drops.append(f"{d}: RG={p['RG']:g} dropped: the gate is a node reference (ideal gate)")
    if any(k in p for k in ("SUBTHRES", "KSUBTHRES")):
        drops.append(f"{d}: subthreshold conduction (subthres/ksubthres) dropped")
    caps = sorted(k for k in p if k in ("CJO", "TT", "CGDMIN", "CGDMAX", "A", "CGS", "M", "VJ"))
    if caps:
        drops.append(f"{d}: {caps} (gate / junction charge, transit time) dropped: no switching "
                     "transient and no reverse recovery in the level-1 device")
    dn, sn = drain, source
    if RD > 0:
        dn = f"{d}.d"
        b.add_resistor(f"{d}.rd", drain, dn, RD)
    if RS > 0:
        sn = f"{d}.s"
        b.add_resistor(f"{d}.rs", sn, source, RS)
    b.add_mosfet_level1(d, dn, sn, gate, K=K, V_T=float(p["VTO"]),
                        lambda_=float(p.get("LAMBDA", 0.0)), with_body_diode=False)
    if RDS > 0:
        b.add_resistor(f"{d}.rds", dn, sn, RDS / mult)
    # The body diode the card describes (LTspice defaults: Is=1e-14, N=1).
    I_S = float(p.get("IS", 1e-14)) * mult
    n = float(p.get("N", 1.0))
    RB = float(p.get("RB", 0.0)) / mult
    BV = float(p.get("BV", 0.0))
    BV_p = 0.0
    if BV > 0:
        IBV = float(p.get("IBV", 1e-3))
        BV_p = BV - n * _V_T * math.log(IBV / I_S) if IBV > I_S else BV
    an = sn
    if RB > 0:
        an = f"{d}.b"
        b.add_resistor(f"{d}_body.rb", sn, an, RB)
    b.add_shockley_diode(f"{d}_body", an, dn, I_S=I_S, n=n, BV=BV_p)
    notes.append(f"{d}: VDMOS → level-1 K = KP/2 = {K:.4g} A/V², body diode Is={I_S:.3g} A n={n:g}")


def spice_to_builder(text_or_path, *, strict: bool = False) -> Any:
    """Parse a SPICE netlist string OR file path and return a
    populated `pulsim.CircuitBuilder`.

    Auto-detects path-vs-text: if the input contains a newline it's
    treated as raw text; otherwise as a path.

    Every ``.MODEL`` card is applied (diode → Shockley junction with
    RS, LEVEL-1 NMOS / VDMOS → Shichman–Hodges device with RD/RS and
    the body diode the card gives) or refused by name; see the module
    docstring. Parameters that are dropped knowingly are listed in
    one summary warning; ``strict=True`` turns any drop into a
    refusal.
    """
    if "\n" not in str(text_or_path):
        from pathlib import Path
        text = Path(text_or_path).read_text()
    else:
        text = str(text_or_path)

    elements, params, _models = parse_spice_netlist(text, return_models=True)

    # Lazy import to avoid a hard module-load dependency on the kernel.
    import pulsim as _v2
    b = _v2.CircuitBuilder()
    notes: List[str] = []
    drops: List[str] = []

    for elem in elements:
        kind = elem.kind
        designator = elem.designator
        nodes = [_normalize_node(n) for n in elem.nodes]
        tok = elem.tokens
        scope = elem.scope

        if kind == "R":
            if len(tok) < 1:
                raise ValueError(f"{designator}: missing value")
            b.add_resistor(designator, nodes[0], nodes[1], _num(tok[0], scope))
        elif kind == "C":
            if len(tok) < 1:
                raise ValueError(f"{designator}: missing value")
            b.add_capacitor(designator, nodes[0], nodes[1], _num(tok[0], scope))
        elif kind == "L":
            if len(tok) < 1:
                raise ValueError(f"{designator}: missing value")
            b.add_inductor(designator, nodes[0], nodes[1], _num(tok[0], scope))
        elif kind == "V":
            spec = _parse_source_spec(tok, scope)
            if spec["kind"] == "dc":
                b.add_voltage_source(designator, nodes[0], nodes[1], spec["value"])
            elif spec["kind"] == "sine":
                b.add_sine_voltage_source(
                    designator, nodes[0], nodes[1],
                    v_dc=spec["v_dc"], v_amplitude=spec["v_amplitude"],
                    frequency=spec["frequency"], phase=spec.get("phase", 0.0))
            elif spec["kind"] == "pulse":
                b.add_pulse_voltage_source(
                    designator, nodes[0], nodes[1],
                    v_low=spec["v_low"], v_high=spec["v_high"],
                    delay=spec["delay"], rise_time=spec["rise_time"],
                    fall_time=spec["fall_time"], pulse_width=spec["pulse_width"],
                    period=spec["period"])
        elif kind == "I":
            spec = _parse_source_spec(tok, scope)
            if spec["kind"] != "dc":
                raise NotImplementedError(
                    "Time-varying current source from SPICE not yet supported by the importer")
            b.add_current_source(designator, nodes[0], nodes[1], spec["value"])
        elif kind == "D":
            _spice_diode(b, elem, nodes, notes, drops)
        elif kind == "M":
            _spice_mosfet(b, elem, nodes, notes, drops)
        elif kind == "X":
            raise ValueError(
                f"{designator}: an unexpanded subcircuit instance survived "
                "flattening — an importer bug; please report the netlist.")
        elif kind == "Q":
            raise NotImplementedError(f"BJT ({designator}) not yet supported")
        elif kind == "J":
            raise NotImplementedError(f"JFET ({designator}) not yet supported")
        elif kind == "K":
            raise NotImplementedError(f"Coupled inductor ({designator}) not yet supported")
        else:
            raise ValueError(
                f"unknown SPICE element kind: {kind!r} on line: {elem.raw_line}")
    if drops and strict:
        raise ValueError(
            "spice_import (strict=True): model parameters would be dropped:\n  "
            + "\n  ".join(drops))
    if drops or notes:
        warnings.warn(
            "spice_import: model mapping notes:\n  " + "\n  ".join(drops + notes),
            stacklevel=2)
    return b
