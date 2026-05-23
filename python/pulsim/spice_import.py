"""Pulsim — SPICE netlist importer.

Parses a subset of SPICE (Berkeley / LTspice / ngspice flavour) and
emits a populated `pulsim.CircuitBuilder`. Goal is to let users
bring existing designs into v2 without rewriting them.

Supported elements (Phase E.10):
  * `R` — resistor:       ``R<name> <n+> <n-> <value>``
  * `C` — capacitor:      ``C<name> <n+> <n-> <value>``
  * `L` — inductor:       ``L<name> <n+> <n-> <value>``
  * `V` — voltage source: ``V<name> <n+> <n-> <DC|SINE|PULSE>(...)``
  * `D` — diode:          ``D<name> <anode> <cathode> [model]``
  * `M` — MOSFET:         ``M<name> <D> <G> <S> [<B>] <model>``
  * `K` — coupled L (deferred — magnetics need more work)
  * `Q` / `J` — BJT / JFET (deferred)
  * `.param` — parameter assignments (numeric only, no expressions)
  * `*` / `;` — comments
  * `.end` — terminator (ignored)
  * `.subckt` / `.ends` — subcircuit support (deferred)

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

For complex netlists with subcircuits, models, expressions, or
behavioural sources, fall back to the YAML loader or hand-build the
CircuitBuilder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "SpiceElement",
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
# Element dataclass + tokeniser
# =============================================================================

@dataclass
class SpiceElement:
    """One parsed SPICE element. `kind` is the first letter (R, C,
    L, V, D, M, …); `name` is the rest of the designator (e.g.
    "R1" → kind='R', name='1'). `nodes` is the list of node names
    in order; `tokens` is the remainder (value + model + options)."""
    kind: str
    name: str
    nodes: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    raw_line: str = ""


def _strip_comment(line: str) -> str:
    # Inline comment marker: ';' splits the line; SPICE convention.
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()


def parse_spice_netlist(text: str) -> Tuple[List[SpiceElement],
                                                  Dict[str, float]]:
    """Parse a SPICE netlist string into a list of `SpiceElement`s
    plus a dict of `.param` assignments.

    Multi-line continuation via `+` at the start of a line is
    supported.
    """
    lines: List[str] = []
    raw_lines = text.splitlines()
    buffer: Optional[str] = None
    for raw in raw_lines:
        ln = raw.rstrip()
        if not ln or ln.startswith("*"):
            continue
        # Continuation line.
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

    elements: List[SpiceElement] = []
    params: Dict[str, float] = {}
    for ln in lines:
        ln = _strip_comment(ln)
        if not ln:
            continue
        u = ln.upper()
        # Dot commands.
        if u.startswith(".END") or u.startswith(".ENDS"):
            continue
        if u.startswith(".PARAM"):
            for m in re.finditer(
                    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+)", ln):
                params[m.group(1)] = parse_spice_value(m.group(2),
                                                            params)
            continue
        if u.startswith("."):
            # Other dot commands (.MODEL, .TRAN, .OP, …) — ignored
            # by the importer; the simulator's run_transient/etc.
            # handles those.
            continue
        # Element line.
        tokens = ln.split()
        if not tokens:
            continue
        designator = tokens[0]
        kind = designator[0].upper()
        name = designator[1:]
        # Number of node terminals varies by element. Use a hint
        # table.
        node_count = {
            "R": 2, "C": 2, "L": 2,
            "V": 2, "I": 2,
            "D": 2,
            "M": 4,         # NMOS / PMOS: D G S B (B optional)
            "Q": 3,         # BJT: C B E
            "J": 3,         # JFET: D G S
            "K": 0,         # coupled inductor pair, no nodes (refs)
            "X": -1,        # subcircuit instance — variable
        }.get(kind, 2)

        if node_count < 0:
            # X (subcircuit): all but last token are nodes.
            elements.append(SpiceElement(
                kind=kind, name=name,
                nodes=tokens[1:-1] if len(tokens) > 1 else [],
                tokens=tokens[-1:] if len(tokens) > 1 else [],
                raw_line=ln))
            continue
        # MOSFETs are tricky — body terminal is OPTIONAL. Auto-shrink
        # if there are < node_count + 1 tokens (designator + nodes +
        # at least 1 model token).
        if kind == "M" and len(tokens) - 1 - 1 < 4:
            node_count = 3
        nodes = tokens[1 : 1 + node_count]
        rest = tokens[1 + node_count :]
        elements.append(SpiceElement(
            kind=kind, name=name, nodes=nodes, tokens=rest,
            raw_line=ln))
    return elements, params


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

    # Detect SINE / PULSE specifier.
    flat = " ".join(tokens)
    m_sine = re.match(
        r"SINE?\s*\(\s*([^)]+)\)", flat, re.IGNORECASE)
    if m_sine:
        params_inside = m_sine.group(1).split()
        vals = [parse_spice_value(t, params) for t in params_inside]
        if len(vals) < 3:
            raise ValueError(
                "SINE spec needs at least V_dc, V_amp, frequency")
        return {
            "kind": "sine",
            "v_dc": vals[0],
            "v_amplitude": vals[1],
            "frequency": vals[2],
            "phase": vals[5] if len(vals) >= 6 else 0.0,
        }
    m_pulse = re.match(
        r"PULSE\s*\(\s*([^)]+)\)", flat, re.IGNORECASE)
    if m_pulse:
        params_inside = m_pulse.group(1).split()
        vals = [parse_spice_value(t, params) for t in params_inside]
        if len(vals) < 7:
            raise ValueError(
                "PULSE spec needs V1 V2 td tr tf pw period")
        return {
            "kind": "pulse",
            "v_low": vals[0], "v_high": vals[1],
            "delay": vals[2], "rise_time": vals[3],
            "fall_time": vals[4], "pulse_width": vals[5],
            "period": vals[6],
        }
    # Else: DC (with optional "DC " prefix).
    if first == "DC":
        return {"kind": "dc",
                  "value": parse_spice_value(tokens[1], params)}
    # Bare number.
    return {"kind": "dc",
              "value": parse_spice_value(tokens[0], params)}


def spice_to_builder(text_or_path) -> Any:
    """Parse a SPICE netlist string OR file path and return a
    populated `pulsim.CircuitBuilder`.

    Auto-detects path-vs-text: if the input contains a newline it's
    treated as raw text; otherwise as a path.
    """
    if "\n" not in str(text_or_path):
        # Treat as file path.
        from pathlib import Path
        text = Path(text_or_path).read_text()
    else:
        text = str(text_or_path)

    elements, params = parse_spice_netlist(text)

    # Lazy import to avoid a hard module-load dependency on the kernel.
    import pulsim as _v2
    b = _v2.CircuitBuilder()

    for elem in elements:
        kind = elem.kind
        name = elem.name
        designator = kind + name
        nodes = [_normalize_node(n) for n in elem.nodes]
        tok = elem.tokens

        if kind == "R":
            if len(tok) < 1:
                raise ValueError(f"R{name}: missing value")
            R = parse_spice_value(tok[0], params)
            b.add_resistor(designator, nodes[0], nodes[1], R)
        elif kind == "C":
            if len(tok) < 1:
                raise ValueError(f"C{name}: missing value")
            C = parse_spice_value(tok[0], params)
            b.add_capacitor(designator, nodes[0], nodes[1], C)
        elif kind == "L":
            if len(tok) < 1:
                raise ValueError(f"L{name}: missing value")
            L = parse_spice_value(tok[0], params)
            b.add_inductor(designator, nodes[0], nodes[1], L)
        elif kind == "V":
            spec = _parse_source_spec(tok, params)
            if spec["kind"] == "dc":
                b.add_voltage_source(designator, nodes[0], nodes[1],
                                        spec["value"])
            elif spec["kind"] == "sine":
                b.add_sine_voltage_source(
                    designator, nodes[0], nodes[1],
                    v_dc=spec["v_dc"],
                    v_amplitude=spec["v_amplitude"],
                    frequency=spec["frequency"],
                    phase=spec.get("phase", 0.0),
                )
            elif spec["kind"] == "pulse":
                # v2's add_pulse_voltage_source signature uses
                # (v_low, v_high, delay, rise, fall, width, period).
                b.add_pulse_voltage_source(
                    designator, nodes[0], nodes[1],
                    v_low=spec["v_low"], v_high=spec["v_high"],
                    delay=spec["delay"],
                    rise_time=spec["rise_time"],
                    fall_time=spec["fall_time"],
                    pulse_width=spec["pulse_width"],
                    period=spec["period"],
                )
        elif kind == "I":
            spec = _parse_source_spec(tok, params)
            if spec["kind"] != "dc":
                raise NotImplementedError(
                    "Time-varying current source from SPICE "
                    "not yet supported by the importer")
            b.add_current_source(designator, nodes[0], nodes[1],
                                    spec["value"])
        elif kind == "D":
            # Basic diode — `D<n> <anode> <cathode> [model]`. The
            # model is ignored in this importer; we use v2's
            # default add_diode with reasonable defaults. Users
            # who need a specific model should set them manually.
            b.add_diode(designator, nodes[0], nodes[1],
                          g_on=1e3, g_off=1e-9, V_th=0.7)
        elif kind == "M":
            # Default to add_mosfet (PWL switch + body diode).
            # SPICE order is D G S [B] <model>. We use D (drain) and
            # S (source) as the channel nodes; ignore G (gate)
            # because v2's PWL mosfet doesn't model gate dynamics
            # — gate is driven via switch_fn externally.
            drain = nodes[0]; source = nodes[2]
            b.add_mosfet_with_body_diode(designator,
                                              drain, source,
                                              R_on=1e-2, R_off=1e9,
                                              V_F=0.7)
        elif kind == "X":
            raise NotImplementedError(
                f"Subcircuit instances (X{name}) not yet supported")
        elif kind == "Q":
            raise NotImplementedError(
                f"BJT (Q{name}) not yet supported")
        elif kind == "J":
            raise NotImplementedError(
                f"JFET (J{name}) not yet supported")
        elif kind == "K":
            raise NotImplementedError(
                f"Coupled inductor (K{name}) not yet supported")
        else:
            raise ValueError(
                f"unknown SPICE element kind: {kind!r} on line: "
                f"{elem.raw_line}")
    return b
