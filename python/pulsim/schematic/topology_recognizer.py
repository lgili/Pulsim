"""Deterministic topology recognizer — Tier 1 of the schematic V2 stack.

Recognizes canonical power-electronics topologies from the structural
graph of a circuit alone, without LLM calls or user input. Each
detector function returns a confidence score in [0, 1]:

- ``1.0``  — perfect structural match (right count + right connectivity)
- ``0.5–0.9``  — partial match (missing one optional component, e.g. no
  explicit load resistor, OR extra unidentified components)
- ``< 0.5``  — treat as "no match"; the dispatcher tries the next
  detector.

The dispatcher iterates every detector, picks the topology with the
highest confidence, returns ``None`` when all scores are below the
``MIN_CONFIDENCE`` threshold.

This module is pure Python with no extra dependencies (it does not
need ``networkx`` — the structural analysis is light enough to roll
by hand and keep deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


# -----------------------------------------------------------------------------
# Public types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RecognizedTopology:
    """The recognizer's verdict for one circuit.

    Attributes
    ----------
    name:
        Canonical topology name (see ``KNOWN_TOPOLOGIES``).
    confidence:
        Score in [0, 1]. Tier 1 produces 0.5–1.0; Tier 2 (LLM) can
        emit lower confidences for partial matches.
    role_map:
        ``{component_name: role_name}`` — used by the template
        instantiator to position each component at its canonical
        slot (e.g. ``"L_main"``, ``"switch_high"``, ``"freewheel_diode"``).
    source:
        Where the verdict came from. ``"heuristic"`` for this module,
        ``"llm"`` for the LLM classifier, ``"cache"`` when re-read
        from the local cache file.
    """

    name: str
    confidence: float
    role_map: dict[str, str] = field(default_factory=dict)
    source: Literal["heuristic", "llm", "cache"] = "heuristic"


#: Closed set of topologies the recognizer can return. The LLM
#: classifier validates its output against this set.
KNOWN_TOPOLOGIES: frozenset[str] = frozenset({
    # SMPS
    "buck", "boost", "buck_boost", "flyback", "forward",
    "half_bridge", "full_bridge",
    # Filters / passives
    "rc_filter", "rl_filter", "rlc_filter",
    # Rectifiers
    "half_wave_rectifier", "full_wave_bridge_rectifier",
})

#: Detectors below this confidence are not returned by ``recognize``.
MIN_CONFIDENCE: float = 0.5


# -----------------------------------------------------------------------------
# Circuit view — small read-only adapter used by every detector
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _Comp:
    """Lightweight component record used internally by the detectors."""
    name: str
    kind: str
    nodes: tuple[int, ...]
    branch_id: int


class _CircuitView:
    """Indexed read-only view of a circuit for structural queries.

    Built once at the top of ``recognize()``; every detector reads
    from it instead of re-walking the builder API.
    """

    __slots__ = (
        "components", "ground", "by_kind", "by_node",
    )

    def __init__(self, circuit: Any) -> None:
        ground = circuit.ground
        if callable(ground):
            ground = ground()
        self.ground: int = int(ground)

        comps: list[_Comp] = []
        for raw in circuit.components():
            # `raw` may be a dict (new flat binding) or a SimpleNamespace
            # (after the adapter shim).
            if isinstance(raw, dict):
                name = raw.get("name", "")
                kind = raw.get("kind", "unknown")
                nodes = tuple(raw.get("nodes", ()))
                bid = raw.get("branch_id", 0)
            else:
                name = getattr(raw, "name", "")
                kind = getattr(raw, "kind", "unknown")
                nodes = tuple(getattr(raw, "nodes", ()))
                bid = getattr(raw, "branch_id", 0)
            comps.append(_Comp(name=name, kind=kind, nodes=nodes,
                                branch_id=int(bid)))
        self.components: list[_Comp] = comps

        # kind → list of components of that kind, preserving order
        by_kind: dict[str, list[_Comp]] = {}
        for c in comps:
            by_kind.setdefault(c.kind, []).append(c)
        self.by_kind: dict[str, list[_Comp]] = by_kind

        # node id → components touching that node (either terminal)
        by_node: dict[int, list[_Comp]] = {}
        for c in comps:
            for n in c.nodes:
                by_node.setdefault(int(n), []).append(c)
        self.by_node: dict[int, list[_Comp]] = by_node

    # ---- convenience queries -------------------------------------------------

    def count(self, kind: str) -> int:
        return len(self.by_kind.get(kind, ()))

    def of_kind(self, kind: str) -> list[_Comp]:
        return self.by_kind.get(kind, [])

    def touches_ground(self, comp: _Comp) -> bool:
        return self.ground in comp.nodes

    def other_terminal(self, comp: _Comp, node: int) -> Optional[int]:
        """For a 2-terminal device, return the terminal that is NOT
        ``node``. Returns ``None`` for devices with != 2 terminals or
        when ``node`` is not a terminal."""
        if len(comp.nodes) != 2 or node not in comp.nodes:
            return None
        a, b = comp.nodes
        return int(b) if int(a) == node else int(a)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

_SOURCE_KINDS: frozenset[str] = frozenset({
    "voltage_source", "pwm_voltage_source",
    "sine_voltage_source", "pulse_voltage_source",
})
_SWITCH_KINDS: frozenset[str] = frozenset({
    "switch", "mosfet_level1", "igbt_level1",
})
_DIODE_KINDS: frozenset[str] = frozenset({
    "diode", "nonlinear_diode",
})


def _source_count(view: _CircuitView, sine: bool = False) -> int:
    """Count voltage sources, optionally restricted to sinusoidal."""
    if sine:
        return view.count("sine_voltage_source")
    return sum(view.count(k) for k in _SOURCE_KINDS)


def _switch_count(view: _CircuitView) -> int:
    return sum(view.count(k) for k in _SWITCH_KINDS)


def _diode_count(view: _CircuitView) -> int:
    return sum(view.count(k) for k in _DIODE_KINDS)


def _first_of_any(view: _CircuitView,
                  kinds: frozenset[str]) -> Optional[_Comp]:
    for k in kinds:
        for c in view.of_kind(k):
            return c
    return None


def _all_of_any(view: _CircuitView,
                kinds: frozenset[str]) -> list[_Comp]:
    out: list[_Comp] = []
    for k in kinds:
        out.extend(view.of_kind(k))
    return out


def _connects(comp: _Comp, n1: int, n2: int) -> bool:
    """True iff `comp` has both `n1` and `n2` as terminals (order-free)."""
    if len(comp.nodes) < 2:
        return False
    s = {int(x) for x in comp.nodes}
    return n1 in s and n2 in s


# -----------------------------------------------------------------------------
# Detectors — one per topology
# -----------------------------------------------------------------------------

DetectorFn = Callable[[_CircuitView], tuple[float, dict[str, str]]]


def detect_buck(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Canonical buck — Vin · switch_high · freewheel_diode · L · Cout · Rload.

    Structural constraints:
      * exactly 1 source
      * exactly 1 switch
      * exactly 1 diode
      * exactly 1 inductor
      * exactly 1 capacitor
      * 0–1 resistor (the load — optional but typical)
      * the switch's high-side connects source(+) to switching node `sw`
      * the diode's anode is ground and cathode is `sw` (freewheel)
      * the inductor goes from `sw` to `vout`
      * the capacitor is between `vout` and ground
    """
    if (_source_count(view) != 1 or _switch_count(view) != 1
            or _diode_count(view) != 1 or view.count("inductor") != 1
            or view.count("capacitor") != 1):
        return 0.0, {}

    src = _first_of_any(view, _SOURCE_KINDS)
    sw  = _first_of_any(view, _SWITCH_KINDS)
    d   = _first_of_any(view, _DIODE_KINDS)
    L   = view.of_kind("inductor")[0]
    C   = view.of_kind("capacitor")[0]
    R   = view.of_kind("resistor")[0] if view.count("resistor") >= 1 else None

    # Source touches ground (its "negative" terminal is gnd).
    if not view.touches_ground(src):
        return 0.0, {}
    # Source's other terminal is Vin+.
    vin_plus = view.other_terminal(src, view.ground)
    if vin_plus is None:
        return 0.0, {}

    # Switch connects vin+ to switching node `sw`.
    if vin_plus not in sw.nodes:
        return 0.0, {}
    sw_node = view.other_terminal(sw, vin_plus)
    if sw_node is None:
        return 0.0, {}

    # Diode: anode = gnd, cathode = sw_node (freewheel, conducts when sw opens).
    # In our model nodes[0] = from = anode, nodes[1] = to = cathode.
    if len(d.nodes) != 2:
        return 0.0, {}
    d_anode, d_cathode = int(d.nodes[0]), int(d.nodes[1])
    if d_anode != view.ground or d_cathode != sw_node:
        return 0.0, {}

    # Inductor between sw_node and vout.
    if sw_node not in L.nodes:
        return 0.0, {}
    vout = view.other_terminal(L, sw_node)
    if vout is None or vout == view.ground:
        return 0.0, {}

    # Capacitor between vout and ground.
    if not _connects(C, vout, view.ground):
        return 0.0, {}

    role_map: dict[str, str] = {
        src.name:  "source",
        sw.name:   "switch_high",
        d.name:    "freewheel_diode",
        L.name:    "inductor_main",
        C.name:    "output_capacitor",
    }
    confidence = 1.0
    if R is not None and _connects(R, vout, view.ground):
        role_map[R.name] = "load_resistor"
    elif R is None:
        # No load resistor — slightly less confident (the canonical
        # form has one, but it's not strictly required).
        confidence = 0.9
    else:
        # Resistor present but not between vout & ground → suspicious.
        confidence = 0.7

    return confidence, role_map


def detect_boost(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Canonical boost — Vin · L · switch_low · diode · Cout · Rload."""
    if (_source_count(view) != 1 or _switch_count(view) != 1
            or _diode_count(view) != 1 or view.count("inductor") != 1
            or view.count("capacitor") != 1):
        return 0.0, {}

    src = _first_of_any(view, _SOURCE_KINDS)
    sw  = _first_of_any(view, _SWITCH_KINDS)
    d   = _first_of_any(view, _DIODE_KINDS)
    L   = view.of_kind("inductor")[0]
    C   = view.of_kind("capacitor")[0]
    R   = view.of_kind("resistor")[0] if view.count("resistor") >= 1 else None

    if not view.touches_ground(src):
        return 0.0, {}
    vin_plus = view.other_terminal(src, view.ground)
    if vin_plus is None:
        return 0.0, {}

    # Inductor goes from vin+ to switching node.
    if vin_plus not in L.nodes:
        return 0.0, {}
    sw_node = view.other_terminal(L, vin_plus)
    if sw_node is None or sw_node == view.ground:
        return 0.0, {}

    # Switch is LOW-side: between sw_node and ground.
    if not _connects(sw, sw_node, view.ground):
        return 0.0, {}

    # Diode: anode = sw_node, cathode = vout.
    if len(d.nodes) != 2:
        return 0.0, {}
    d_anode, d_cathode = int(d.nodes[0]), int(d.nodes[1])
    if d_anode != sw_node:
        return 0.0, {}
    vout = d_cathode
    if vout == view.ground or vout == vin_plus:
        return 0.0, {}

    # Output cap between vout and ground.
    if not _connects(C, vout, view.ground):
        return 0.0, {}

    role_map: dict[str, str] = {
        src.name: "source",
        L.name:   "inductor_main",
        sw.name:  "switch_low",
        d.name:   "output_diode",
        C.name:   "output_capacitor",
    }
    confidence = 1.0
    if R is not None and _connects(R, vout, view.ground):
        role_map[R.name] = "load_resistor"
    elif R is None:
        confidence = 0.9
    else:
        confidence = 0.7

    return confidence, role_map


def detect_buck_boost(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Inverting buck-boost: Vin · switch_high · L_low · diode · Cout · Rload.

    Topology: Vin+ → switch → sw_node → L → gnd; sw_node → diode →
    vout (NEGATIVE w.r.t. ground); Cout, Rload between vout and gnd.
    """
    if (_source_count(view) != 1 or _switch_count(view) != 1
            or _diode_count(view) != 1 or view.count("inductor") != 1
            or view.count("capacitor") != 1):
        return 0.0, {}

    src = _first_of_any(view, _SOURCE_KINDS)
    sw  = _first_of_any(view, _SWITCH_KINDS)
    d   = _first_of_any(view, _DIODE_KINDS)
    L   = view.of_kind("inductor")[0]
    C   = view.of_kind("capacitor")[0]
    R   = view.of_kind("resistor")[0] if view.count("resistor") >= 1 else None

    if not view.touches_ground(src):
        return 0.0, {}
    vin_plus = view.other_terminal(src, view.ground)
    if vin_plus is None:
        return 0.0, {}

    # Switch high-side
    if vin_plus not in sw.nodes:
        return 0.0, {}
    sw_node = view.other_terminal(sw, vin_plus)
    if sw_node is None:
        return 0.0, {}

    # Inductor between sw_node and ground (low-side L)
    if not _connects(L, sw_node, view.ground):
        return 0.0, {}

    # Diode: cathode at vout, anode at sw_node (the inverted-output flavor).
    # We accept either polarity — buck-boost variants differ.
    if len(d.nodes) != 2:
        return 0.0, {}
    if sw_node not in d.nodes:
        return 0.0, {}
    vout = view.other_terminal(d, sw_node)
    if vout is None or vout == view.ground or vout == vin_plus:
        return 0.0, {}

    if not _connects(C, vout, view.ground):
        return 0.0, {}

    role_map: dict[str, str] = {
        src.name: "source",
        sw.name:  "switch_high",
        L.name:   "inductor_main",
        d.name:   "output_diode",
        C.name:   "output_capacitor",
    }
    confidence = 0.95  # never quite 1.0 — pattern is rare enough that we leave
                      # buck/boost slightly higher priority.
    if R is not None and _connects(R, vout, view.ground):
        role_map[R.name] = "load_resistor"
    elif R is None:
        confidence = 0.85

    return confidence, role_map


def detect_flyback(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Flyback: source · transformer · primary-switch · secondary-diode · Cout · Rload.

    Our internal transformer representation splits into two coupled
    inductors. So the structural fingerprint is:
      * 1 source
      * 1 switch
      * 1 diode
      * 2 inductors (primary + secondary)
      * 1 capacitor (output)
      * 0–1 resistor
    """
    if (_source_count(view) != 1 or _switch_count(view) != 1
            or _diode_count(view) != 1 or view.count("inductor") != 2
            or view.count("capacitor") != 1):
        return 0.0, {}

    src = _first_of_any(view, _SOURCE_KINDS)
    sw  = _first_of_any(view, _SWITCH_KINDS)
    d   = _first_of_any(view, _DIODE_KINDS)
    inductors = view.of_kind("inductor")
    C   = view.of_kind("capacitor")[0]
    R   = view.of_kind("resistor")[0] if view.count("resistor") >= 1 else None

    if not view.touches_ground(src):
        return 0.0, {}

    # Identify primary (touches the primary loop with source + switch)
    # vs secondary (touches the diode + cap loop).
    primary: Optional[_Comp] = None
    secondary: Optional[_Comp] = None
    for L in inductors:
        sw_overlap = bool(set(L.nodes) & set(sw.nodes))
        d_overlap  = bool(set(L.nodes) & set(d.nodes))
        if sw_overlap and not d_overlap:
            primary = L
        elif d_overlap and not sw_overlap:
            secondary = L

    if primary is None or secondary is None:
        return 0.0, {}

    role_map: dict[str, str] = {
        src.name:        "source",
        sw.name:         "switch_primary",
        d.name:          "rectifier_diode",
        primary.name:    "transformer_primary",
        secondary.name:  "transformer_secondary",
        C.name:          "output_capacitor",
    }
    confidence = 0.9   # transformer detection is heuristic; LLM tier
                      # can refine.
    if R is not None:
        role_map[R.name] = "load_resistor"
    return confidence, role_map


def detect_half_bridge(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Half-bridge: 2 switches in series between Vdc and gnd.

    The midpoint of the leg is where the load connects.
    """
    if _source_count(view) != 1 or _switch_count(view) != 2:
        return 0.0, {}
    src = _first_of_any(view, _SOURCE_KINDS)
    switches = _all_of_any(view, _SWITCH_KINDS)
    if src is None or len(switches) != 2:
        return 0.0, {}
    if not view.touches_ground(src):
        return 0.0, {}
    vdc = view.other_terminal(src, view.ground)
    if vdc is None:
        return 0.0, {}

    # One switch must connect vdc to a midpoint; the other midpoint to gnd.
    high: Optional[_Comp] = None
    low:  Optional[_Comp] = None
    for s in switches:
        if vdc in s.nodes:
            high = s
        elif view.ground in s.nodes:
            low = s
    if high is None or low is None or high is low:
        return 0.0, {}
    mid_high = view.other_terminal(high, vdc)
    mid_low  = view.other_terminal(low, view.ground)
    if mid_high is None or mid_low is None or mid_high != mid_low:
        return 0.0, {}

    role_map = {
        src.name:  "source",
        high.name: "switch_high",
        low.name:  "switch_low",
    }
    return 1.0, role_map


def detect_full_bridge(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Full-bridge / H-bridge — 4 switches in two legs across Vdc."""
    if _source_count(view) != 1 or _switch_count(view) != 4:
        return 0.0, {}
    src = _first_of_any(view, _SOURCE_KINDS)
    switches = _all_of_any(view, _SWITCH_KINDS)
    if src is None or len(switches) != 4:
        return 0.0, {}
    if not view.touches_ground(src):
        return 0.0, {}
    vdc = view.other_terminal(src, view.ground)
    if vdc is None:
        return 0.0, {}

    # Group switches by which rail they touch.
    high_switches = [s for s in switches if vdc in s.nodes]
    low_switches  = [s for s in switches if view.ground in s.nodes]
    if len(high_switches) != 2 or len(low_switches) != 2:
        return 0.0, {}

    # Pair each high with the low sharing its midpoint.
    mids_high = {view.other_terminal(s, vdc): s for s in high_switches}
    mids_low  = {view.other_terminal(s, view.ground): s for s in low_switches}
    if set(mids_high) != set(mids_low) or len(mids_high) != 2:
        return 0.0, {}

    legs = sorted(mids_high.keys())
    role_map = {src.name: "source"}
    role_map[mids_high[legs[0]].name] = "switch_a_high"
    role_map[mids_low[legs[0]].name]  = "switch_a_low"
    role_map[mids_high[legs[1]].name] = "switch_b_high"
    role_map[mids_low[legs[1]].name]  = "switch_b_low"
    return 1.0, role_map


def detect_forward(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Forward converter — source · transformer · primary-switch ·
    secondary-rectifier-diode · freewheel-diode · output-L · Cout · Rload.

    Heuristic: source + 1 switch + 2 diodes + 2 inductors (one is the
    transformer primary, the other is the output choke — we can't
    easily distinguish without coupling info, so accept the shape).
    """
    if (_source_count(view) != 1 or _switch_count(view) != 1
            or _diode_count(view) != 2 or view.count("inductor") < 2
            or view.count("capacitor") != 1):
        return 0.0, {}

    src = _first_of_any(view, _SOURCE_KINDS)
    sw  = _first_of_any(view, _SWITCH_KINDS)
    diodes = _all_of_any(view, _DIODE_KINDS)
    inductors = view.of_kind("inductor")
    C = view.of_kind("capacitor")[0]
    R = view.of_kind("resistor")[0] if view.count("resistor") >= 1 else None
    if src is None or sw is None or len(diodes) != 2 or len(inductors) < 2:
        return 0.0, {}
    if not view.touches_ground(src):
        return 0.0, {}

    role_map = {
        src.name:       "source",
        sw.name:        "switch_primary",
        diodes[0].name: "rectifier_diode",
        diodes[1].name: "freewheel_diode",
        inductors[0].name: "transformer_primary",
        inductors[1].name: "output_inductor",
        C.name:         "output_capacitor",
    }
    if R is not None:
        role_map[R.name] = "load_resistor"
    return 0.75, role_map  # forward is structurally similar to flyback +
                            # buck — leave it lower so flyback can win.


def detect_rc_filter(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """1 source + 1 resistor + 1 capacitor in series-to-ground."""
    if (_source_count(view) != 1 or view.count("resistor") != 1
            or view.count("capacitor") != 1
            or view.count("inductor") != 0
            or _switch_count(view) != 0 or _diode_count(view) != 0):
        return 0.0, {}
    src = _first_of_any(view, _SOURCE_KINDS)
    R = view.of_kind("resistor")[0]
    C = view.of_kind("capacitor")[0]
    if not view.touches_ground(src) or not view.touches_ground(C):
        return 0.0, {}
    return 1.0, {src.name: "source", R.name: "filter_r", C.name: "filter_c"}


def detect_rl_filter(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """1 source + 1 resistor + 1 inductor (no capacitor)."""
    if (_source_count(view) != 1 or view.count("resistor") != 1
            or view.count("inductor") != 1
            or view.count("capacitor") != 0
            or _switch_count(view) != 0 or _diode_count(view) != 0):
        return 0.0, {}
    src = _first_of_any(view, _SOURCE_KINDS)
    R = view.of_kind("resistor")[0]
    L = view.of_kind("inductor")[0]
    if not view.touches_ground(src):
        return 0.0, {}
    return 1.0, {src.name: "source", R.name: "filter_r", L.name: "filter_l"}


def detect_rlc_filter(view: _CircuitView) -> tuple[float, dict[str, str]]:
    """1 source + 1 R + 1 L + 1 C, no switches/diodes."""
    if (_source_count(view) != 1 or view.count("resistor") != 1
            or view.count("inductor") != 1
            or view.count("capacitor") != 1
            or _switch_count(view) != 0 or _diode_count(view) != 0):
        return 0.0, {}
    src = _first_of_any(view, _SOURCE_KINDS)
    R = view.of_kind("resistor")[0]
    L = view.of_kind("inductor")[0]
    C = view.of_kind("capacitor")[0]
    if not view.touches_ground(src):
        return 0.0, {}
    return 1.0, {
        src.name: "source", R.name: "filter_r",
        L.name: "filter_l", C.name: "filter_c",
    }


def detect_half_wave_rectifier(
        view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Sine source + 1 diode + 1 capacitor + 1 resistor (load)."""
    if (view.count("sine_voltage_source") != 1
            or _diode_count(view) != 1
            or view.count("capacitor") != 1
            or view.count("resistor") != 1
            or view.count("inductor") != 0
            or _switch_count(view) != 0):
        return 0.0, {}
    src = view.of_kind("sine_voltage_source")[0]
    d = _first_of_any(view, _DIODE_KINDS)
    C = view.of_kind("capacitor")[0]
    R = view.of_kind("resistor")[0]
    if not view.touches_ground(src):
        return 0.0, {}
    # Diode anode is the source's hot terminal; cathode is vout.
    vin_hot = view.other_terminal(src, view.ground)
    if vin_hot is None or len(d.nodes) != 2:
        return 0.0, {}
    d_anode, d_cathode = int(d.nodes[0]), int(d.nodes[1])
    if d_anode != vin_hot:
        return 0.0, {}
    vout = d_cathode
    if not _connects(C, vout, view.ground) or not _connects(R, vout, view.ground):
        return 0.0, {}
    return 1.0, {
        src.name: "source",
        d.name: "rectifier_diode",
        C.name: "filter_capacitor",
        R.name: "load_resistor",
    }


def detect_full_wave_bridge_rectifier(
        view: _CircuitView) -> tuple[float, dict[str, str]]:
    """Sine source + 4 diodes in bridge + 1 capacitor + 1 resistor."""
    if (view.count("sine_voltage_source") != 1
            or _diode_count(view) != 4
            or view.count("capacitor") != 1
            or view.count("resistor") != 1
            or view.count("inductor") != 0
            or _switch_count(view) != 0):
        return 0.0, {}
    diodes = _all_of_any(view, _DIODE_KINDS)
    if len(diodes) != 4:
        return 0.0, {}

    src = view.of_kind("sine_voltage_source")[0]
    C = view.of_kind("capacitor")[0]
    R = view.of_kind("resistor")[0]

    # Bridge structure: every diode touches both an AC rail (touched
    # by the source) AND a DC rail (touched by the capacitor). Verify
    # all four diodes share this shape.
    src_nodes = {int(n) for n in src.nodes}
    cap_nodes = {int(n) for n in C.nodes}
    r_nodes   = {int(n) for n in R.nodes}
    if cap_nodes != r_nodes:
        # Capacitor and load resistor MUST connect across the same
        # output rails — otherwise this isn't the canonical bridge.
        return 0.0, {}
    for d in diodes:
        d_nodes = {int(n) for n in d.nodes}
        if not (d_nodes & src_nodes) or not (d_nodes & cap_nodes):
            return 0.0, {}
    role_map = {
        src.name: "source",
        diodes[0].name: "rectifier_diode_a",
        diodes[1].name: "rectifier_diode_b",
        diodes[2].name: "rectifier_diode_c",
        diodes[3].name: "rectifier_diode_d",
        C.name: "filter_capacitor",
        R.name: "load_resistor",
    }
    return 1.0, role_map


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

#: Detection order matters when two detectors return equal confidence —
#: the FIRST one wins. We list the more-specific topologies first so a
#: buck doesn't get mis-classified as a "buck inside a forward" etc.
_DETECTORS: list[tuple[str, DetectorFn]] = [
    # Filters first — their constraints are most specific (no switches).
    ("rc_filter",                 detect_rc_filter),
    ("rl_filter",                 detect_rl_filter),
    ("rlc_filter",                detect_rlc_filter),
    # Rectifiers (need sine source — narrow trigger).
    ("full_wave_bridge_rectifier", detect_full_wave_bridge_rectifier),
    ("half_wave_rectifier",        detect_half_wave_rectifier),
    # Multi-switch SMPS.
    ("half_bridge",               detect_half_bridge),
    ("full_bridge",               detect_full_bridge),
    # Isolated.
    ("flyback",                   detect_flyback),
    ("forward",                   detect_forward),
    # Non-isolated SMPS — buck/boost share most of the fingerprint, so
    # order matters only between these three.
    ("buck",                      detect_buck),
    ("boost",                     detect_boost),
    ("buck_boost",                detect_buck_boost),
]


def recognize(circuit: Any) -> Optional[RecognizedTopology]:
    """Run every detector against ``circuit``; return the highest-
    confidence match above ``MIN_CONFIDENCE``, or ``None`` when no
    detector matches.

    ``circuit`` can be:
      * a flat-namespace ``pulsim.CircuitBuilder``;
      * any object exposing ``components()`` (dict or SimpleNamespace
        entries) and ``ground``/``ground()``.
    """
    view = _CircuitView(circuit)
    best: Optional[RecognizedTopology] = None
    for name, detector in _DETECTORS:
        try:
            conf, role_map = detector(view)
        except Exception:
            # A detector bug must never break the dispatcher — log if we
            # had a logger; for now, swallow and continue.
            continue
        if conf < MIN_CONFIDENCE:
            continue
        if best is None or conf > best.confidence:
            best = RecognizedTopology(
                name=name, confidence=float(conf),
                role_map=role_map, source="heuristic")
    return best
