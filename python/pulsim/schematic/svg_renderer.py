"""Direct-SVG schematic renderer — Phase 8.

Drops the schemdraw dependency and renders every symbol natively as
SVG paths. The killer feature is that each symbol's body is drawn so
its electrical terminals land EXACTLY at the ``terminal_anchors``
coordinates that ``template_instantiator`` already computes — no more
"wire ending in midair near a symbol" because the geometry is under
our direct control.

Symbol library (V0):
    resistor, capacitor, inductor, diode, switch, mosfet, igbt,
    voltage_source, sine_voltage_source, pwm_voltage_source,
    pulse_voltage_source, current_source, transformer, generic-box.

Each drawer takes (placement, anchor_xs, anchor_ys) and returns an
SVG <g> string. Anchor coordinates are the contract: pin 0 is anchor
index 0, pin 1 is anchor index 1, etc. The symbol's body lies between
them (for 2-terminal devices) or is centered around them (for 3+).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .types import (
    BoundingBox,
    ComponentPlacement,
    SchematicLayout,
    TerminalAnchor,
    Wire,
)


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

_STROKE: str = "#111"
_WIRE_STROKE: str = "#111"
_GROUND_STROKE: str = "#222"
_FILL: str = "none"
_LABEL_FILL: str = "#222"
# Reference: schemdraw / KiCad-style — symbol strokes are noticeably
# thinner than the body geometry suggests; wires are thinner still so
# the eye reads "symbol > wire" hierarchy.
_STROKE_WIDTH: float = 0.9
_WIRE_WIDTH:   float = 0.65
_JUNCTION_R:   float = 1.1
_FILL_BG:      str   = "white"   # mask wires/junctions hidden behind a symbol body
_LABEL_SIZE: float = 3.8
_FONT_FAMILY: str = "Helvetica, Arial, sans-serif"


def _fmt(v: float) -> str:
    """Compact numeric formatting for SVG attributes."""
    if abs(v) < 1e-6:
        return "0"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _polyline(points: list[tuple[float, float]], stroke: str = _STROKE,
              width: float = _STROKE_WIDTH) -> str:
    pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="none" '
        f'stroke="{stroke}" stroke-width="{_fmt(width)}" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _line(x1: float, y1: float, x2: float, y2: float,
          stroke: str = _STROKE, width: float = _STROKE_WIDTH) -> str:
    return (
        f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" '
        f'x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
        f'stroke="{stroke}" stroke-width="{_fmt(width)}" '
        f'stroke-linecap="round"/>'
    )


def _circle(cx: float, cy: float, r: float, *, stroke: str = _STROKE,
            fill: str = _FILL, width: float = _STROKE_WIDTH) -> str:
    return (
        f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}" '
        f'stroke="{stroke}" stroke-width="{_fmt(width)}" '
        f'fill="{fill}"/>'
    )


def _body_mask(cx: float, cy: float, length: float, width: float,
               ux: float, uy: float) -> str:
    """White-fill quadrilateral that masks any wire crossing the body
    region of a 2-terminal symbol.

    Drawn BEFORE the symbol's body marks so the marks land on top of a
    clean white background — wires that would otherwise be visible
    behind the zigzag/coil/triangle are hidden inside the rectangle.

    The quadrilateral is centered on (cx, cy), extends ``length`` along
    the (ux, uy) lead-axis, and ``width`` perpendicular to it.
    """
    px, py = -uy, ux
    half_l = length / 2.0
    half_w = width / 2.0
    p1 = (cx + ux * half_l + px * half_w, cy + uy * half_l + py * half_w)
    p2 = (cx + ux * half_l - px * half_w, cy + uy * half_l - py * half_w)
    p3 = (cx - ux * half_l - px * half_w, cy - uy * half_l - py * half_w)
    p4 = (cx - ux * half_l + px * half_w, cy - uy * half_l + py * half_w)
    pts = " ".join(f"{_fmt(p[0])},{_fmt(p[1])}" for p in (p1, p2, p3, p4))
    return (f'<polygon points="{pts}" fill="{_FILL_BG}" '
            f'stroke="none"/>')


def _text(x: float, y: float, content: str, *,
          size: float = _LABEL_SIZE, anchor: str = "middle",
          fill: str = _LABEL_FILL) -> str:
    safe = (content.replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;"))
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" '
        f'font-family="{_FONT_FAMILY}" font-size="{_fmt(size)}" '
        f'fill="{fill}" text-anchor="{anchor}" '
        f'dominant-baseline="middle">{safe}</text>'
    )


# ---------------------------------------------------------------------------
# Symbol drawers — each draws so that its terminals land at the anchor coords
# ---------------------------------------------------------------------------
#
# The orientation convention: terminal-0 is the FIRST listed anchor and
# terminal-1 is the SECOND. For a vertical 2-terminal device (anchors at
# (cx, cy - h/2) and (cx, cy + h/2)), the body fills the middle and the
# leads are straight lines to the terminal anchors. Rotation is fully
# encoded in the relative positions of pin 0 vs pin 1 — we don't need a
# separate rotation parameter because we derive the axis from the
# anchor coords themselves.

def _terminal_axis(p0: tuple[float, float], p1: tuple[float, float]
                   ) -> str:
    """Return "h" or "v" based on whether the two terminals form a
    horizontal or vertical body."""
    dx = abs(p0[0] - p1[0])
    dy = abs(p0[1] - p1[1])
    return "h" if dx >= dy else "v"


def _draw_resistor(p: ComponentPlacement) -> str:
    """ANSI-style zigzag resistor — 6 sharp teeth on a clean axis.

    Layout along the lead axis:  lead → entry-stub → 6 teeth → exit-stub → lead.
    The body region is masked white BEFORE the zigzag, so any wire
    routed behind the symbol disappears (the user reported wires
    "passing through" components).
    """
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return ""
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    body_len = min(10.0, length * 0.62)
    amp = 1.5
    n_teeth = 6
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    s = (mx - ux * body_len / 2.0, my - uy * body_len / 2.0)
    e = (s[0] + ux * body_len, s[1] + uy * body_len)
    step = body_len / (n_teeth + 1)
    pts: list[tuple[float, float]] = [s]
    for i in range(1, n_teeth + 1):
        side = 1.0 if (i % 2 == 1) else -1.0
        x = s[0] + ux * step * i + px * amp * side
        y = s[1] + uy * step * i + py * amp * side
        pts.append((x, y))
    pts.append(e)
    mask = _body_mask(mx, my, body_len + 1.0, amp * 2 + 1.0, ux, uy)
    leads = []
    if _is_connected(p, 0):
        leads.append(_line(ax, ay, s[0], s[1]))
    if _is_connected(p, 1):
        leads.append(_line(e[0], e[1], bx, by))
    body = _polyline(pts)
    label = _label_outside(p, amp + 4.0)
    return "<g>" + mask + "".join(leads + [body, label]) + "</g>"


def _draw_capacitor(p: ComponentPlacement) -> str:
    """Two parallel plates perpendicular to the lead axis. Plates are
    thicker than the leads (the schemdraw convention) so the cap reads
    even at small sizes."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return ""
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    gap = 1.8
    plate_w = 6.0
    plate_stroke = _STROKE_WIDTH * 1.4
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    mid_a = (mx - ux * gap / 2.0, my - uy * gap / 2.0)
    mid_b = (mx + ux * gap / 2.0, my + uy * gap / 2.0)
    p1a = (mid_a[0] + px * plate_w / 2.0, mid_a[1] + py * plate_w / 2.0)
    p1b = (mid_a[0] - px * plate_w / 2.0, mid_a[1] - py * plate_w / 2.0)
    p2a = (mid_b[0] + px * plate_w / 2.0, mid_b[1] + py * plate_w / 2.0)
    p2b = (mid_b[0] - px * plate_w / 2.0, mid_b[1] - py * plate_w / 2.0)
    mask = _body_mask(mx, my, gap + 2.0, plate_w + 1.0, ux, uy)
    leads = []
    if _is_connected(p, 0):
        leads.append(_line(ax, ay, mid_a[0], mid_a[1]))
    if _is_connected(p, 1):
        leads.append(_line(mid_b[0], mid_b[1], bx, by))
    plates = [_line(p1a[0], p1a[1], p1b[0], p1b[1], width=plate_stroke),
              _line(p2a[0], p2a[1], p2b[0], p2b[1], width=plate_stroke)]
    label = _label_outside(p, plate_w / 2.0 + 2.5)
    return "<g>" + mask + "".join(leads + plates + [label]) + "</g>"


def _draw_inductor(p: ComponentPlacement) -> str:
    """Inductor — four half-circle bumps along the lead axis, drawn as
    a single SVG path so the joins are perfectly smooth.

    Body length ≈ 12 mm; bump radius = 1.5 mm. The bumps always curve
    to the same side of the axis (schemdraw convention)."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return ""
    ux, uy = dx / length, dy / length
    body_len = min(12.0, length * 0.62)
    n_bumps = 4
    bump_r = body_len / (2 * n_bumps)
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    s = (mx - ux * body_len / 2.0, my - uy * body_len / 2.0)
    angle_deg = math.degrees(math.atan2(uy, ux))
    mask = _body_mask(mx, my, body_len + 1.0, bump_r * 2 + 1.5, ux, uy)
    leads = []
    if _is_connected(p, 0):
        leads.append(_line(ax, ay, s[0], s[1]))
    if _is_connected(p, 1):
        leads.append(_line(s[0] + ux * body_len, s[1] + uy * body_len, bx, by))
    parts: list[str] = [f"M {_fmt(s[0])} {_fmt(s[1])}"]
    for i in range(n_bumps):
        nx = s[0] + ux * bump_r * 2 * (i + 1)
        ny = s[1] + uy * bump_r * 2 * (i + 1)
        parts.append(
            f"A {_fmt(bump_r)} {_fmt(bump_r)} {_fmt(angle_deg)} "
            f"0 1 {_fmt(nx)} {_fmt(ny)}"
        )
    path_d = " ".join(parts)
    body = (
        f'<path d="{path_d}" fill="none" stroke="{_STROKE}" '
        f'stroke-width="{_fmt(_STROKE_WIDTH)}" '
        f'stroke-linejoin="round"/>'
    )
    label = _label_outside(p, bump_r + 4.0)
    return "<g>" + mask + "".join(leads + [body, label]) + "</g>"


def _draw_diode(p: ComponentPlacement) -> str:
    """Diode: filled triangle pointing from anode (pin 0) toward
    cathode (pin 1), followed by a slightly-thicker cathode bar
    perpendicular to the lead axis. Filling the triangle (the standard
    convention) makes the direction unambiguous at any zoom level."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return ""
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    body_len = min(6.5, length * 0.50)
    body_w   = 5.2
    bar_stroke = _STROKE_WIDTH * 1.4
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    apex_back = (mx - ux * body_len / 2.0, my - uy * body_len / 2.0)
    apex_tip  = (mx + ux * body_len / 2.0, my + uy * body_len / 2.0)
    tri_a = (apex_back[0] + px * body_w / 2.0,
             apex_back[1] + py * body_w / 2.0)
    tri_b = (apex_back[0] - px * body_w / 2.0,
             apex_back[1] - py * body_w / 2.0)
    bar_a = (apex_tip[0] + px * body_w / 2.0,
             apex_tip[1] + py * body_w / 2.0)
    bar_b = (apex_tip[0] - px * body_w / 2.0,
             apex_tip[1] - py * body_w / 2.0)
    mask = _body_mask(mx, my, body_len + 1.0, body_w + 1.0, ux, uy)
    leads = []
    if _is_connected(p, 0):
        leads.append(_line(ax, ay, apex_back[0], apex_back[1]))
    if _is_connected(p, 1):
        leads.append(_line(apex_tip[0], apex_tip[1], bx, by))
    tri_pts = " ".join(f"{_fmt(p[0])},{_fmt(p[1])}"
                       for p in (tri_a, tri_b, apex_tip))
    triangle = (
        f'<polygon points="{tri_pts}" fill="{_STROKE}" '
        f'stroke="{_STROKE}" stroke-width="{_fmt(_STROKE_WIDTH)}" '
        f'stroke-linejoin="round"/>'
    )
    bar = _line(bar_a[0], bar_a[1], bar_b[0], bar_b[1], width=bar_stroke)
    label = _label_outside(p, body_w / 2.0 + 2.5)
    return "<g>" + mask + "".join(leads + [triangle, bar, label]) + "</g>"


def _draw_voltage_source(p: ComponentPlacement) -> str:
    """Battery / voltage source: circle with ± labels."""
    return _draw_circle_source(p, label_top="+", label_bot="-")


_SOURCE_RADIUS: float = 4.5


def _draw_sine_source(p: ComponentPlacement) -> str:
    """AC sinusoidal source: white-fill circle containing a single sine
    cycle. No polarity markers (AC has no fixed polarity)."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    r = _SOURCE_RADIUS
    body = [_circle(cx, cy, r, fill=_FILL_BG)]
    span = r * 1.3
    n = 14
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        ang = -math.pi + 2 * math.pi * t
        sx = cx - span / 2.0 + span * t
        sy = cy + r * 0.45 * math.sin(ang)
        pts.append((sx, sy))
    wave = _polyline(pts, width=_STROKE_WIDTH * 1.05)
    leads = _source_leads(p, r)
    label = _label_outside(p, r + 2.5)
    return "<g>" + "".join(leads + body + [wave, label]) + "</g>"


def _draw_pwm_source(p: ComponentPlacement) -> str:
    """PWM source: white-fill circle with a small square-wave inside."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    r = _SOURCE_RADIUS
    body = [_circle(cx, cy, r, fill=_FILL_BG)]
    sq = _polyline([
        (cx - r * 0.7, cy + r * 0.45),
        (cx - r * 0.3, cy + r * 0.45),
        (cx - r * 0.3, cy - r * 0.45),
        (cx + r * 0.1, cy - r * 0.45),
        (cx + r * 0.1, cy + r * 0.45),
        (cx + r * 0.5, cy + r * 0.45),
        (cx + r * 0.5, cy - r * 0.45),
    ], width=_STROKE_WIDTH * 1.05)
    leads = _source_leads(p, r)
    label = _label_outside(p, r + 2.5)
    return "<g>" + "".join(leads + body + [sq, label]) + "</g>"


def _draw_pulse_source(p: ComponentPlacement) -> str:
    """Single-pulse / step source: white-fill circle with a single
    rising-edge step glyph inside."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    r = _SOURCE_RADIUS
    body = [_circle(cx, cy, r, fill=_FILL_BG)]
    pulse = _polyline([
        (cx - r * 0.7, cy + r * 0.4),
        (cx - r * 0.2, cy + r * 0.4),
        (cx - r * 0.2, cy - r * 0.4),
        (cx + r * 0.2, cy - r * 0.4),
        (cx + r * 0.2, cy + r * 0.4),
        (cx + r * 0.7, cy + r * 0.4),
    ], width=_STROKE_WIDTH * 1.05)
    leads = _source_leads(p, r)
    label = _label_outside(p, r + 2.5)
    return "<g>" + "".join(leads + body + [pulse, label]) + "</g>"


def _draw_current_source(p: ComponentPlacement) -> str:
    """Current source: white-fill circle with a single arrow pointing
    from anode (pin 0) toward cathode (pin 1)."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    ux, uy = (b.x - a.x), (b.y - a.y)
    n = math.hypot(ux, uy)
    if n > 0:
        ux, uy = ux / n, uy / n
    r = _SOURCE_RADIUS
    body = [_circle(cx, cy, r, fill=_FILL_BG)]
    tail = (cx - ux * r * 0.6, cy - uy * r * 0.6)
    head = (cx + ux * r * 0.6, cy + uy * r * 0.6)
    px, py = -uy, ux
    h1 = (head[0] - ux * 1.4 + px * 0.9, head[1] - uy * 1.4 + py * 0.9)
    h2 = (head[0] - ux * 1.4 - px * 0.9, head[1] - uy * 1.4 - py * 0.9)
    arrow = [
        _line(tail[0], tail[1], head[0], head[1]),
        _line(head[0], head[1], h1[0], h1[1]),
        _line(head[0], head[1], h2[0], h2[1]),
    ]
    leads = _source_leads(p, r)
    label = _label_outside(p, r + 2.5)
    return "<g>" + "".join(leads + body + arrow + [label]) + "</g>"


def _draw_switch(p: ComponentPlacement) -> str:
    """Open-switch symbol: two contact dots and a tilted arm hinged on
    the first contact, reaching past the second (open-position).

    The arm is constrained to STOP just before the second contact (it
    used to overshoot by ~30 %, which made it look like it stuck out
    of the bounding box). Mask under the symbol so wires routed past
    the body don't shine through."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return ""
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    body_len = min(8.0, length * 0.55)
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    s = (mx - ux * body_len / 2.0, my - uy * body_len / 2.0)
    e = (mx + ux * body_len / 2.0, my + uy * body_len / 2.0)
    arm_perp = body_len * 0.30
    arm_along = body_len * 0.80
    arm_end = (
        s[0] + ux * arm_along + px * arm_perp,
        s[1] + uy * arm_along + py * arm_perp,
    )
    mask = _body_mask(mx, my, body_len + 1.0, arm_perp * 2 + 2.0, ux, uy)
    leads = []
    if _is_connected(p, 0):
        leads.append(_line(ax, ay, s[0], s[1]))
    if _is_connected(p, 1):
        leads.append(_line(e[0], e[1], bx, by))
    contact = [
        _line(s[0], s[1], arm_end[0], arm_end[1]),
        _circle(s[0], s[1], 1.2, fill="white"),
        _circle(e[0], e[1], 1.2, fill="white"),
    ]
    label = _label_outside(p, body_len * 0.5 + 3.5)
    return "<g>" + mask + "".join(leads + contact + [label]) + "</g>"


def _draw_mosfet(p: ComponentPlacement) -> str:
    """N-channel MOSFET symbol: gate bar (perpendicular to channel),
    channel line with the 3 short tick marks indicating the enhancement
    zone, source arrow showing current direction. Pin 0 = gate,
    Pin 1 = drain, Pin 2 = source."""
    if len(p.terminal_anchors) < 3:
        return _draw_generic(p)
    g, d, s = (p.terminal_anchors[0], p.terminal_anchors[1],
               p.terminal_anchors[2])

    # Channel axis from drain to source.
    chan_dx = s.x - d.x
    chan_dy = s.y - d.y
    cn = math.hypot(chan_dx, chan_dy) or 1.0
    cux, cuy = chan_dx / cn, chan_dy / cn
    # Perpendicular = gate-side direction.
    pgx, pgy = -cuy, cux

    cx, cy = (d.x + s.x) / 2.0, (d.y + s.y) / 2.0
    # Channel line slightly inset from the drain/source pins so the
    # ticks have room.
    inset = cn * 0.18
    chan_d = (d.x + cux * inset, d.y + cuy * inset)
    chan_s = (s.x - cux * inset, s.y - cuy * inset)

    # Gate-side line offset PERPENDICULAR to the channel toward gate.
    # First figure out which side the gate is on.
    gate_dot = (g.x - cx) * pgx + (g.y - cy) * pgy
    side = 1.0 if gate_dot > 0 else -1.0
    gx = cx + pgx * side * 2.5
    gy = cy + pgy * side * 2.5

    # Gate bar — perpendicular to channel, slightly shorter than channel.
    bar_half = cn * 0.35
    bar_a = (gx + cux * bar_half, gy + cuy * bar_half)
    bar_b = (gx - cux * bar_half, gy - cuy * bar_half)

    # Three enhancement-region ticks between gate bar and channel
    # (parallel to gate bar, shorter, 3 segments).
    ticks: list[str] = []
    for frac in (-0.5, 0.0, 0.5):
        tx = gx + cux * bar_half * frac
        ty = gy + cuy * bar_half * frac
        tx_close = tx - pgx * side * 1.0
        ty_close = ty - pgy * side * 1.0
        ticks.append(_line(tx, ty, tx_close, ty_close))

    leads = [
        _line(d.x, d.y, chan_d[0], chan_d[1]),
        _line(chan_d[0], chan_d[1], chan_s[0], chan_s[1]),
        _line(chan_s[0], chan_s[1], s.x, s.y),
        _line(g.x, g.y, gx, gy),
        _line(bar_a[0], bar_a[1], bar_b[0], bar_b[1]),
    ] + ticks

    # Source arrow: small triangle near the source end pointing INTO
    # the channel (for N-MOSFET).
    arrow_len = 1.8
    arrow_perp = 1.2
    apex = (chan_s[0], chan_s[1])
    tail = (chan_s[0] - cux * arrow_len, chan_s[1] - cuy * arrow_len)
    a1 = (tail[0] + pgx * side * arrow_perp,
          tail[1] + pgy * side * arrow_perp)
    arrow = [
        _polyline([a1, apex, tail, a1])
    ]

    # Label off to the OPPOSITE side from the gate bar so it doesn't
    # collide with the gate driver wire.
    lbl_x = cx - pgx * side * (bar_half + 4.0)
    lbl_y = cy - pgy * side * (bar_half + 4.0)
    anchor = "end" if (lbl_x < cx) else "start"
    label = _text(lbl_x, lbl_y, p.name, anchor=anchor)
    return "<g>" + "".join(leads + arrow + [label]) + "</g>"


def _draw_igbt(p: ComponentPlacement) -> str:
    # IGBT visually = MOSFET + a small arrow on the source. V0: same as MOSFET.
    return _draw_mosfet(p)


def _draw_transformer(p: ComponentPlacement) -> str:
    """4-terminal transformer: two inductor-like coils side by side
    connected by a coupling bar."""
    if len(p.terminal_anchors) < 4:
        return _draw_generic(p)
    p1, p2, s1, s2 = p.terminal_anchors[:4]
    # Primary coil between p1 and p2 (rough).
    pri_mid = ((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
    sec_mid = ((s1.x + s2.x) / 2.0, (s1.y + s2.y) / 2.0)
    leads = [_line(p1.x, p1.y, pri_mid[0], pri_mid[1]),
             _line(pri_mid[0], pri_mid[1], p2.x, p2.y),
             _line(s1.x, s1.y, sec_mid[0], sec_mid[1]),
             _line(sec_mid[0], sec_mid[1], s2.x, s2.y)]
    coupling = _line(pri_mid[0], pri_mid[1], sec_mid[0], sec_mid[1],
                       stroke="#999")
    label = _text((pri_mid[0] + sec_mid[0]) / 2.0,
                  (pri_mid[1] + sec_mid[1]) / 2.0 - 6.0,
                  p.name, anchor="middle")
    return "<g>" + "".join(leads + [coupling, label]) + "</g>"


def _draw_generic(p: ComponentPlacement) -> str:
    """Fallback: labeled box between first and last terminals."""
    if not p.terminal_anchors:
        return _text(p.x, p.y, p.name, anchor="middle")
    a = p.terminal_anchors[0]
    b = p.terminal_anchors[-1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    w, h = 14.0, 8.0
    box = (
        f'<rect x="{_fmt(cx - w / 2.0)}" y="{_fmt(cy - h / 2.0)}" '
        f'width="{_fmt(w)}" height="{_fmt(h)}" '
        f'stroke="{_STROKE}" fill="white" '
        f'stroke-width="{_fmt(_STROKE_WIDTH)}"/>'
    )
    leads = []
    for t in p.terminal_anchors:
        leads.append(_line(t.x, t.y, cx, cy))
    label = _text(cx, cy, p.name)
    return "<g>" + "".join(leads + [box, label]) + "</g>"


# ---------------------------------------------------------------------------
# Source helpers (shared between voltage / sine / pulse / current)
# ---------------------------------------------------------------------------

def _draw_circle_source(p: ComponentPlacement, *,
                          label_top: str = "+",
                          label_bot: str = "-") -> str:
    """Generic circle voltage source with ± markers inside. The circle
    is filled white so any wire routed behind it is hidden."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    r = 4.5
    body = [_circle(cx, cy, r, fill=_FILL_BG)]
    if abs(a.y - b.y) > abs(a.x - b.x):
        marker_offset = r * 0.55
        labels = [
            _text(cx, cy - marker_offset, label_top, size=4.6),
            _text(cx, cy + marker_offset, label_bot, size=5.6),
        ]
    else:
        marker_offset = r * 0.55
        labels = [
            _text(cx - marker_offset, cy, label_top, size=4.6),
            _text(cx + marker_offset, cy, label_bot, size=5.6),
        ]
    leads = _source_leads(p, r)
    name_lbl = _label_outside(p, r + 2.5)
    return "<g>" + "".join(leads + body + labels + [name_lbl]) + "</g>"


def _source_leads(p: ComponentPlacement, r: float) -> list[str]:
    """Stub leads from each terminal anchor to the edge of the source
    body (a circle of radius r centered between the two anchors).
    Skips floating terminals so the source's circle doesn't grow a
    stub into empty space."""
    a, b = p.terminal_anchors[0], p.terminal_anchors[1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    out = []
    for t in (a, b):
        if not _is_connected(p, t.index):
            continue
        dx, dy = t.x - cx, t.y - cy
        n = math.hypot(dx, dy)
        if n < 1e-6 or n <= r:
            continue
        edge_x = cx + dx / n * r
        edge_y = cy + dy / n * r
        out.append(_line(t.x, t.y, edge_x, edge_y))
    return out


def _label_outside(p: ComponentPlacement, offset: float) -> str:
    """Place the device name label off to the side of the body."""
    if not p.terminal_anchors:
        return ""
    a = p.terminal_anchors[0]
    b = p.terminal_anchors[-1]
    cx, cy = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
    # Pick label position perpendicular to the axis.
    if abs(a.y - b.y) > abs(a.x - b.x):
        # Vertical body → label to the LEFT.
        return _text(cx - offset, cy, p.name, anchor="end")
    # Horizontal body → label ABOVE.
    return _text(cx, cy - offset, p.name, anchor="middle")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DRAWERS = {
    "resistor":             _draw_resistor,
    "capacitor":            _draw_capacitor,
    "inductor":             _draw_inductor,
    "diode":                _draw_diode,
    "nonlinear_diode":      _draw_diode,
    "voltage_source":       _draw_voltage_source,
    "sine_voltage_source":  _draw_sine_source,
    "pwm_voltage_source":   _draw_pwm_source,
    "pulse_voltage_source": _draw_pulse_source,
    "current_source":       _draw_current_source,
    "switch":               _draw_switch,
    "mosfet_level1":        _draw_mosfet,
    "igbt_level1":          _draw_igbt,
    "vcvs":                 _draw_generic,
    "saturable_inductor":   _draw_inductor,
    "transformer":          _draw_transformer,
}


_CONNECTED: dict[str, set[int]] = {}


def _is_connected(p: ComponentPlacement, terminal: int) -> bool:
    """Return True iff terminal ``terminal`` of component ``p`` has at
    least one wire endpoint touching it. Used by the drawers to suppress
    leads that would otherwise dangle in midair (e.g. forward's Dfwd
    anode without a secondary winding)."""
    s = _CONNECTED.get(p.name)
    return s is not None and terminal in s


def _draw_component(p: ComponentPlacement) -> str:
    drawer = _DRAWERS.get(p.kind, _draw_generic)
    return drawer(p)


def _draw_wire(w: Wire) -> str:
    if not w.path or len(w.path) < 2:
        return ""
    pts = list(w.path)
    # Dedupe consecutive identical points.
    deduped: list[tuple[float, float]] = []
    for pt in pts:
        if not deduped or pt != deduped[-1]:
            deduped.append(pt)
    if len(deduped) < 2:
        return ""
    return _polyline(deduped, stroke=_WIRE_STROKE, width=_WIRE_WIDTH)


def _ground_symbol(x: float, y: float) -> str:
    """Three-bar IEC ground symbol pointing south from (x, y).

    A short vertical stem connects the rail to the top bar so the
    glyph looks tied to (not bumping into) the rail. The 3 bars step
    down by 1.6 mm with widths 7 mm → 4.5 mm → 2 mm — large enough to
    read at the rendered ~4× scale we use."""
    return (
        f'<g stroke="{_GROUND_STROKE}" fill="none" '
        f'stroke-width="{_fmt(_STROKE_WIDTH)}" '
        f'stroke-linecap="round">'
        # Tie-down stem from rail to first bar.
        f'{_line(x, y, x, y + 1.6, stroke=_GROUND_STROKE)}'
        # Three horizontal bars, decreasing width.
        f'{_line(x - 3.5, y + 1.6, x + 3.5, y + 1.6, stroke=_GROUND_STROKE)}'
        f'{_line(x - 2.25, y + 3.4, x + 2.25, y + 3.4, stroke=_GROUND_STROKE)}'
        f'{_line(x - 1.0, y + 5.2, x + 1.0, y + 5.2, stroke=_GROUND_STROKE)}'
        '</g>'
    )


def _detect_coupled_inductors(
        layout: SchematicLayout) -> list[tuple[ComponentPlacement, ComponentPlacement]]:
    """Find pairs of inductor placements that look like a transformer:
    same y-center (±2 mm) and within 20–80 mm horizontally.

    The renderer draws two short vertical bars between such pairs to
    indicate magnetic coupling. Without this, flyback / forward show
    two unconnected coils with confusing "floating" leads."""
    inductors = [
        p for p in layout.components.values()
        if p.kind in ("inductor", "saturable_inductor")
    ]
    pairs: list[tuple[ComponentPlacement, ComponentPlacement]] = []
    seen: set[str] = set()
    for i, a in enumerate(inductors):
        if a.name in seen:
            continue
        for b in inductors[i + 1:]:
            if b.name in seen:
                continue
            if abs(a.y - b.y) > 2.0:
                continue
            dx = abs(a.x - b.x)
            if 20.0 <= dx <= 80.0:
                pairs.append((a, b))
                seen.add(a.name)
                seen.add(b.name)
                break
    return pairs


def _transformer_coupling_glyph(a: ComponentPlacement,
                                 b: ComponentPlacement) -> str:
    """Two closely-spaced vertical bars centered between a transformer's
    primary and secondary coils — the standard "magnetic core" indicator.

    Bars are 8 mm tall and 1.5 mm apart, centered on the gap midpoint.
    Also drops two small polarity dots (one per coil) on the side
    facing the other winding — the convention for showing winding
    polarity."""
    cy = (a.y + b.y) / 2.0
    midpoint = (a.x + b.x) / 2.0
    bar_dx = 1.0  # half-distance between the two bars
    bar_half_h = 4.0  # bar half-length
    bars = [
        _line(midpoint - bar_dx, cy - bar_half_h,
              midpoint - bar_dx, cy + bar_half_h),
        _line(midpoint + bar_dx, cy - bar_half_h,
              midpoint + bar_dx, cy + bar_half_h),
    ]
    # Polarity dots, on the inside top of each coil (relative to the
    # other coil). Coil bodies are roughly 6 mm wide and ~12 mm tall;
    # place the dot 2 mm in from the inner edge and 4 mm above center.
    left, right = (a, b) if a.x < b.x else (b, a)
    dot_y = cy - 5.0
    dot_left = (
        f'<circle cx="{_fmt(left.x + 3.5)}" cy="{_fmt(dot_y)}" '
        f'r="0.9" fill="{_STROKE}"/>'
    )
    dot_right = (
        f'<circle cx="{_fmt(right.x - 3.5)}" cy="{_fmt(dot_y)}" '
        f'r="0.9" fill="{_STROKE}"/>'
    )
    return "<g>" + "".join(bars + [dot_left, dot_right]) + "</g>"


def _detect_ground_rail(layout: SchematicLayout) -> tuple[float, float, float] | None:
    """Inspect ``layout.wires`` for synthetic ``<gnd-rail>`` endpoints
    and return ``(rail_y, x_min, x_max)``. None if no rail is present.

    Two wire shapes carry rail geometry:
      * vertical stubs — one endpoint references ``<gnd-rail>``,
        path[-1].y is the rail's y;
      * the optional rail-to-rail wire — both endpoints reference
        ``<gnd-rail>``, path spans ``(xs_min, rail_y) → (xs_max, rail_y)``.

    A standalone schematic with only a single grounded component will
    have only one stub, so we still return a rail at that single x."""
    rail_y: float | None = None
    xs: list[float] = []
    for w in layout.wires:
        from_gnd = w.from_.component == "<gnd-rail>"
        to_gnd = w.to.component == "<gnd-rail>"
        if not (from_gnd or to_gnd):
            continue
        if not w.path:
            continue
        # Each rail-related wire has rail_y as its lowest path-y (because
        # the rail sits BELOW the components in screen coords).
        for px, py in w.path:
            if rail_y is None or py > rail_y:
                rail_y = py
        # Collect every x that lands on rail_y for x-extent computation.
        # We re-scan AFTER rail_y is known.
    if rail_y is None:
        return None
    eps = 0.25  # mm — y-comparison tolerance
    for w in layout.wires:
        if (w.from_.component != "<gnd-rail>"
                and w.to.component != "<gnd-rail>"):
            continue
        for px, py in w.path:
            if abs(py - rail_y) < eps:
                xs.append(px)
    if not xs:
        return None
    return rail_y, min(xs), max(xs)


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

def render_svg(layout: SchematicLayout, path: Any) -> Path:
    """Render ``layout`` to an SVG file at ``path``. Returns the
    resolved path. Pure Python — no schemdraw dependency."""
    out = Path(path)
    bbox = layout.canvas
    # Build the (component, terminal) connectivity set so drawers can
    # suppress dangling leads. A terminal is "connected" iff some wire
    # endpoint references it. Synthetic <gnd-rail> endpoints are
    # ignored — they aren't real components.
    _CONNECTED.clear()
    for w in layout.wires:
        if w.from_.component != "<gnd-rail>":
            _CONNECTED.setdefault(w.from_.component, set()).add(w.from_.terminal)
        if w.to.component != "<gnd-rail>":
            _CONNECTED.setdefault(w.to.component, set()).add(w.to.terminal)
    # Add a generous margin around the canvas for labels.
    margin = 10.0
    vb_x = bbox.x - margin
    vb_y = bbox.y - margin
    vb_w = bbox.width + 2 * margin
    vb_h = bbox.height + 2 * margin

    chunks: list[str] = []
    chunks.append(
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{_fmt(vb_x)} {_fmt(vb_y)} {_fmt(vb_w)} {_fmt(vb_h)}" '
        f'width="{_fmt(vb_w * 4)}" height="{_fmt(vb_h * 4)}">\n'
    )
    # White background so PNG conversions look clean.
    chunks.append(
        f'<rect x="{_fmt(vb_x)}" y="{_fmt(vb_y)}" '
        f'width="{_fmt(vb_w)}" height="{_fmt(vb_h)}" fill="white"/>\n'
    )

    # Layer 1: wires (drawn first so symbols can overlap them at terminals).
    chunks.append('<g id="wires">\n')
    for w in layout.wires:
        chunks.append(_draw_wire(w) + "\n")
    chunks.append('</g>\n')

    # Layer 2: junctions.
    chunks.append('<g id="junctions" fill="black">\n')
    for jx, jy in layout.junctions:
        chunks.append(
            f'<circle cx="{_fmt(jx)}" cy="{_fmt(jy)}" '
            f'r="{_fmt(_JUNCTION_R)}" fill="black"/>\n'
        )
    chunks.append('</g>\n')

    # Layer 3: components.
    chunks.append('<g id="components">\n')
    for placement in layout.components.values():
        chunks.append(_draw_component(placement) + "\n")
    chunks.append('</g>\n')

    # Layer 3b: transformer coupling bars between paired inductors.
    coupled = _detect_coupled_inductors(layout)
    if coupled:
        chunks.append('<g id="transformer-couplings">\n')
        for a, b in coupled:
            chunks.append(_transformer_coupling_glyph(a, b) + "\n")
        chunks.append('</g>\n')

    # Layer 4: ground symbol(s) on the rail. Renders ONE IEC ground glyph
    # at the leftmost rail x (offset 2 mm out so it doesn't overlap a
    # stub). When the rail is a single point (only one grounded
    # component), the glyph lands directly at that x.
    rail = _detect_ground_rail(layout)
    if rail is not None:
        rail_y, x_min, x_max = rail
        if x_max - x_min > 5.0:
            # Multi-component rail — anchor the glyph just past the
            # leftmost stub so it visually labels the rail.
            gx = x_min - 4.0
        else:
            gx = x_min
        chunks.append('<g id="ground-symbols">\n')
        chunks.append(_ground_symbol(gx, rail_y) + "\n")
        # On wide rails, drop a second glyph near the right end too so
        # very long rails don't look "unlabeled".
        if x_max - x_min > 80.0:
            chunks.append(_ground_symbol(x_max + 4.0, rail_y) + "\n")
        chunks.append('</g>\n')
        # Extend the rail line to MEET the leftmost ground symbol so
        # there's no visible gap between the rail and the glyph.
        if x_max - x_min > 5.0:
            chunks.append(
                _line(gx, rail_y, x_min, rail_y, stroke=_WIRE_STROKE,
                      width=_WIRE_WIDTH) + "\n"
            )
            if x_max - x_min > 80.0:
                chunks.append(
                    _line(x_max, rail_y, x_max + 4.0, rail_y,
                          stroke=_WIRE_STROKE, width=_WIRE_WIDTH) + "\n"
                )

    chunks.append('</svg>\n')
    out.write_text("".join(chunks), encoding="utf-8")
    return out


def render_png(layout: SchematicLayout, path: Any,
                width_px: int = 1400) -> Path:
    """SVG → PNG via cairosvg if installed; falls back to schemdraw's
    PNG path otherwise. Raises ImportError when no PNG path is
    available."""
    out = Path(path)
    svg_path = out.with_suffix(".svg")
    render_svg(layout, svg_path)
    try:
        import cairosvg  # type: ignore[import-not-found]
        cairosvg.svg2png(url=str(svg_path), write_to=str(out),
                          output_width=width_px)
        # Keep the SVG side-by-side — it's the canonical artifact.
        return out
    except ImportError:
        raise ImportError(
            "PNG output requires cairosvg. Install with "
            "`pip install pulsim[schematic]` (which pulls cairosvg) "
            "or use `render_svg()` for SVG-only output."
        )
