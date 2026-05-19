"""netlistsvg-based backend for ``pulsim.schematic``.

Replaces the in-tree ELK + schemdraw pipeline with a subprocess call to
``netlistsvg`` (https://github.com/nturley/netlistsvg) — the production
schematic renderer used by the Yosys / openlane ecosystem. ``netlistsvg``
internally uses ELK ``layered`` but with carefully-tuned options and a
purpose-built analog SVG skin (resistors, capacitors, inductors, diodes,
ground symbol, voltage-source circle, transformer two-coil glyph) that
produces textbook-quality schematics out of the box.

The transport between Python and ``netlistsvg`` is a subprocess: we write
a Yosys-format JSON netlist to a temp file, invoke ``netlistsvg`` with
the analog skin, and read back the SVG. Each Pulsim component becomes a
``cell`` with a netlistsvg cell type; each electrical net becomes a
unique integer ``bit``; each ground connection on each component is
materialised as its own dedicated ``gnd`` cell so the schematic shows
multiple ground symbols (one per device) rather than a single shared
rail — this is the standard schematic convention netlistsvg expects.

Component-kind mapping (Pulsim → netlistsvg analog skin):

    resistor                        → resistor_v
    capacitor                       → capacitor_v
    inductor                        → inductor_h
    voltage_source                  → voltage_source
    pwm_voltage_source              → voltage_source
    sine_voltage_source             → voltage_source
    pulse_voltage_source            → voltage_source
    current_source                  → current_source
    diode                           → diode_h
    switch / vcswitch / mosfet /
        igbt                        → generic (3-pin labeled box)
    transformer                     → transformer_1p_1s
    anything else                   → generic
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _find_node_modules() -> Path | None:
    """Walk up from this file's location looking for a ``node_modules/``
    directory that contains ``netlistsvg/``.

    This file may live under either the source tree
    (``python/pulsim/schematic/``) or the build tree
    (``build/python/pulsim/schematic/``). Both are nested below a repo
    root that holds the canonical ``node_modules/`` after ``npm install``.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        nm = candidate / "node_modules" / "netlistsvg"
        if nm.exists():
            return candidate / "node_modules"
    return None


def _resolve_skin() -> Path | None:
    """Prefer the Pulsim-extended analog skin (with MOSFET / IGBT / vcswitch
    symbols) over the upstream netlistsvg analog skin (which lacks them).

    The Pulsim skin sits at ``python/pulsim/schematic/skin/pulsim_analog.svg``
    in the source tree and is copied into the build tree by CMake. If it
    isn't present (slim install / wheel without the skin), fall back to
    the upstream ``analog.svg`` and accept that switching devices will
    render as ``generic`` boxes.
    """
    here = Path(__file__).resolve().parent
    pulsim_skin = here / "skin" / "pulsim_analog.svg"
    if pulsim_skin.exists():
        return pulsim_skin
    if _NODE_MODULES is not None:
        upstream = _NODE_MODULES / "netlistsvg" / "lib" / "analog.svg"
        if upstream.exists():
            return upstream
    return None


_NODE_MODULES = _find_node_modules()
_NETLISTSVG_BIN = (_NODE_MODULES / ".bin" / "netlistsvg") if _NODE_MODULES else None
_ANALOG_SKIN = _resolve_skin()

# Pulsim canonical kind → netlistsvg analog-skin cell **alias**.
# netlistsvg matches cell types against the `<s:alias val="...">` field
# in the skin SVG, NOT against the full `s:type` element name. Using the
# short alias (e.g. `r_v` instead of `resistor_v`) is what triggers the
# proper analog-symbol rendering — otherwise netlistsvg falls back to a
# generic labeled-rectangle box, which is what we got in the first PoC.
#
# Three-plus-terminal devices that don't have a dedicated analog-skin
# symbol use cell type `generic` (no alias) — netlistsvg draws a labeled
# rectangle with explicit pin labels. Extending the SVG skin with a
# proper MOSFET/IGBT glyph is a follow-up (`add-pulsim-analog-skin`).
_CELL_TYPE = {
    "resistor":            "r_v",
    "capacitor":           "c_v",
    "inductor":            "l_h",
    "voltage_source":      "v",
    "pwm_voltage_source":  "v",
    "sine_voltage_source": "v",
    "pulse_voltage_source": "v",
    "current_source":      "i",
    "diode":               "d_h",
    "transformer":         "transformer_1p_1s",
    # Switching devices use the Pulsim-extended analog skin's new
    # symbols (vertical body, control pin left, current path top↔bottom).
    # The upstream netlistsvg analog.svg has no MOSFET/IGBT/vcswitch
    # primitives — the Pulsim skin is preferred when available, and the
    # upstream skin falls back to `generic` boxes if it's missing.
    "mosfet":              "mosfet_n",
    "igbt":                "igbt",
    "vcswitch":            "vcswitch",
}

# Per cell-type port names for our supported kinds. Each entry is a list
# of (port_name, direction) per terminal index. The analog skin uses
# `+`/`-` for sources and diodes (with `+` as anode), and `A`/`B` for
# the symmetric passives. Switching-device port names match the
# `s:pid="..."` attributes in pulsim_analog.svg.
_CELL_PORTS = {
    "r_v":              [("A", "input"),  ("B", "output")],
    "c_v":              [("A", "input"),  ("B", "output")],
    "l_h":              [("A", "input"),  ("B", "output")],
    "v":                [("+", "output"), ("-", "input")],
    "i":                [("+", "output"), ("-", "input")],
    "d_h":              [("+", "input"),  ("-", "output")],
    "transformer_1p_1s": [
        ("L1.1", "input"), ("L1.2", "output"),
        ("L2.1", "input"), ("L2.2", "output"),
    ],
    # MOSFET pin order in Pulsim is [gate, drain, source]; the skin
    # symbol has the corresponding s:pid="G"/"D"/"S".
    "mosfet_n":         [("G", "input"), ("D", "input"), ("S", "output")],
    "mosfet_p":         [("G", "input"), ("D", "input"), ("S", "output")],
    # IGBT pin order is [gate, collector, emitter] — the skin uses
    # s:pid="G"/"C"/"E".
    "igbt":             [("G", "input"), ("C", "input"), ("E", "output")],
    # vcswitch in Pulsim has pin order [ctrl, t1, t2]; skin matches.
    "vcswitch":         [("ctrl", "input"), ("t1", "input"), ("t2", "output")],
}

# Generic cell layout for 3-terminal switches (vcswitch / mosfet / igbt).
# Ports: in0 = control/gate, out0 = drain/t1, out1 = source/t2.
_GENERIC_3PIN_PORTS = [("in0", "input"), ("out0", "output"), ("out1", "output")]


def _require_netlistsvg() -> str:
    """Locate the ``netlistsvg`` CLI; raise with an install hint if absent."""
    if _NETLISTSVG_BIN is not None and _NETLISTSVG_BIN.exists():
        return str(_NETLISTSVG_BIN)
    sys_bin = shutil.which("netlistsvg")
    if sys_bin:
        return sys_bin
    raise ImportError(
        "pulsim.schematic netlistsvg backend requires the `netlistsvg` CLI. "
        "Install via `npm install netlistsvg` at the repo root."
    )


def _build_yosys_json(components: list, ground_id: int) -> dict[str, Any]:
    """Convert a Pulsim component list to a Yosys-format netlist JSON.

    Each non-ground electrical node gets a unique positive integer bit;
    each ground connection on each device gets a *separate* fresh bit
    paired with a private ``gnd`` cell — the standard schematic
    convention of "each device has its own ground symbol, conceptually
    all wired together off-canvas".
    """
    cells: dict[str, dict[str, Any]] = {}
    # Bit allocator: assign each non-ground node a bit; ground bits are
    # generated on demand (one per device).
    node_to_bit: dict[int, int] = {}
    next_bit = 2  # bits 0 and 1 are special-reserved in Yosys (constants).

    def _bit_for(node_id: int) -> int:
        nonlocal next_bit
        if node_id == ground_id:
            bit = next_bit
            next_bit += 1
            return bit
        if node_id not in node_to_bit:
            node_to_bit[node_id] = next_bit
            next_bit += 1
        return node_to_bit[node_id]

    gnd_counter = 0
    for comp in components:
        kind = comp.kind
        cell_type = _CELL_TYPE.get(kind, "generic")
        n_terms = len(comp.nodes)

        # Determine port list for this cell.
        if cell_type in _CELL_PORTS:
            ports = _CELL_PORTS[cell_type]
        elif n_terms == 3:
            ports = _GENERIC_3PIN_PORTS
        else:
            # generic 2-pin fallback
            ports = [("A", "input"), ("B", "output")]

        # If pin count mismatches the skin's expected ports, use generic.
        if len(ports) != n_terms:
            cell_type = "generic"
            if n_terms == 3:
                ports = _GENERIC_3PIN_PORTS
            else:
                ports = [(f"p{i}", "input" if i < n_terms / 2 else "output")
                         for i in range(n_terms)]

        port_directions: dict[str, str] = {}
        connections: dict[str, list[int]] = {}
        for i, node_id in enumerate(comp.nodes):
            port_name, direction = ports[i]
            port_directions[port_name] = direction
            if int(node_id) == ground_id:
                # Allocate a private bit for this gnd terminal + emit a gnd cell.
                gnd_bit = _bit_for(ground_id)
                connections[port_name] = [gnd_bit]
                gnd_counter += 1
                cells[f"{comp.name}__gnd{i}"] = {
                    "type": "gnd",
                    "port_directions": {"A": "input"},
                    "connections": {"A": [gnd_bit]},
                }
            else:
                connections[port_name] = [_bit_for(int(node_id))]

        cells[comp.name] = {
            "type": cell_type,
            "port_directions": port_directions,
            "connections": connections,
        }

    return {
        "modules": {
            "circuit": {
                "ports": {},
                "cells": cells,
            }
        }
    }


# Note: helpers for position-hint support (``_export_default_layout``,
# ``_apply_position_hints``) were prototyped here and removed in
# Phase 4F.1 — netlistsvg 1.0.2's ``--layout`` is too broken to drive an
# MVP (Promise resolves ``undefined``; edge sections reference cached
# positions). The tracking proposal `add-schematic-position-hints`
# documents the upstream bugs and the realistic paths forward (fork
# netlistsvg, or replicate the render layer in Python).


def render_netlistsvg(
    circuit: Any,
    svg_path: Any,
    position_hints: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Render a Pulsim Circuit to SVG via netlistsvg + analog skin.

    ``position_hints`` is **reserved but not yet implemented** — see
    [`openspec/changes/add-schematic-position-hints`](../../../openspec/changes/add-schematic-position-hints)
    for the tracking proposal. The empirical finding: netlistsvg 1.0.2's
    ``--layout`` flag has two real upstream bugs (a) Promise-path renders
    the SVG into ``undefined`` when ``elkData`` is supplied, (b) overriding
    ``(x, y)`` on cells without recomputing the cached edge ``sections``
    produces tangled wires through empty space. Both require either a
    fork of netlistsvg or replicating its render layer in Python — neither
    feasible as an MVP. Until a path is implemented, passing
    ``position_hints`` raises :class:`NotImplementedError`.

    Args:
        circuit: Pulsim Circuit (uses ``components()`` and ``ground()``).
        svg_path: Output SVG path (must end in ``.svg``).
        position_hints: Reserved for future use; currently must be ``None``
            or empty.

    Returns:
        The :class:`pathlib.Path` of the written SVG.

    Raises:
        ImportError: If the netlistsvg CLI isn't available locally.
        RuntimeError: If the netlistsvg subprocess fails.
        NotImplementedError: If ``position_hints`` is non-empty (see above).
    """
    if position_hints:
        raise NotImplementedError(
            "pulsim.schematic position hints are tracked but not yet "
            "implemented — see openspec/changes/add-schematic-position-hints "
            "for the upstream-bug analysis and the path to a working "
            "implementation. The auto-layout (no hints) renders correctly."
        )

    binary = _require_netlistsvg()
    components = list(circuit.components())
    ground_id = int(circuit.ground())
    payload = _build_yosys_json(components, ground_id)

    out_path = Path(svg_path)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        input_json = Path(f.name)
    try:
        result = subprocess.run(
            [binary, str(input_json), "--skin", str(_ANALOG_SKIN), "-o", str(out_path)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            stdout = result.stdout.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"netlistsvg failed (exit {result.returncode}): "
                f"stderr={stderr!r} stdout={stdout!r}"
            )
    finally:
        try:
            input_json.unlink()
        except OSError:
            pass
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"netlistsvg produced no output at {out_path}")
    return out_path
