"""Schematic rendering with pluggable backends.

Default backend is **python_native** (`add-python-schematic-renderer`
Phase 5) — pure-Python composition over the existing analog SVG skin,
with elkjs called only for layout. Honors user-supplied position hints
(`Circuit.set_position(...)`, YAML `position:`) and applies topology-
aware auto-layouts for recognized sub-circuits (bridge rectifier,
boost stage, half-bridge).

Other backends available via ``PULSIM_SCHEMATIC_BACKEND`` env var:

    python_native  (default)  — pure-Python renderer + elkjs layout
    netlistsvg     (legacy)   — netlistsvg CLI + analog skin
                                 (emits DeprecationWarning; scheduled
                                 for removal in a future release once
                                 the native path has soaked)
    elk                       — ELK layered + schemdraw symbols
    spring                    — legacy force-directed + templates

The `elk` and `spring` backends produce a :class:`SchematicLayout`
which is still useful when a GUI wants component coordinates. The
default native backend exposes the same layout via
``compute_layout(circuit)``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .layout import compute_layout
from .netlistsvg_backend import render_netlistsvg
from .symbols import get_symbol_factory
from .types import ComponentPlacement, SchematicLayout, Wire


# mm → schemdraw-unit scale. Default schemdraw element length is 3 units;
# a typical layout has ~30 mm between component centers. Scaling by 1/10
# maps that to ~3 units, which is roughly schemdraw's natural element size.
_SCALE = 0.1
_DEGENERATE_OFFSET = 0.5  # schemdraw units — applied when two terminals collapse


def _require_schemdraw():
    """Lazy-import schemdraw with a clear install hint on failure."""
    try:
        import schemdraw  # type: ignore[import-not-found]
        from schemdraw import elements as elm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pulsim.schematic.render requires schemdraw. "
            "Install via: pip install pulsim[schematic]"
        ) from exc
    return schemdraw, elm


_SUPPORTED_FORMATS = frozenset({"svg", "png", "pdf", "jpg", "jpeg"})


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in _SUPPORTED_FORMATS:
        return "jpg" if suffix == "jpeg" else suffix
    raise ValueError(
        f"Cannot infer schematic format from extension {path.suffix!r}. "
        f"Pass format='svg' or format='png' explicitly."
    )


def _to_canvas(x_mm: float, y_mm: float) -> tuple[float, float]:
    """mm (screen convention) → schemdraw units (Cartesian, y flipped)."""
    return (x_mm * _SCALE, -y_mm * _SCALE)


def _draw_two_terminal(d, factory, placement: ComponentPlacement) -> None:
    """Place a 2-terminal element between its two terminal anchors.

    schemdraw auto-sizes and rotates a two-terminal element when both
    .at() and .to() are set, so this is the cleanest path. Degenerate
    case: if both terminals collapse to the same point (can happen when
    a device shares both nodes with another), nudge .to() by a small
    offset so the element has a non-zero length.
    """
    anchors = placement.terminal_anchors
    t0 = _to_canvas(anchors[0].x, anchors[0].y)
    t1 = _to_canvas(anchors[1].x, anchors[1].y)
    if abs(t0[0] - t1[0]) < 1e-6 and abs(t0[1] - t1[1]) < 1e-6:
        t1 = (t1[0] + _DEGENERATE_OFFSET, t1[1])
    d.add(factory().at(t0).to(t1).label(placement.name, loc="top"))


def _draw_multi_terminal(d, factory, placement: ComponentPlacement) -> None:
    """Place a 3+ terminal element at its centroid with the given rotation.

    Phase 3 does not yet wire the element's native schemdraw anchors to
    the layout's terminal anchors — the straight-line wires emitted by
    :func:`compute_layout` already cover the terminal-to-terminal
    connections. As a result, three-terminal devices visually look like
    "a MOSFET symbol with wires going to/from its surrounding nodes", which
    is correct topologically. Symbol-aware pin orientation is a follow-up.
    """
    cx, cy = _to_canvas(placement.x, placement.y)
    d.add(factory().at((cx, cy)).theta(float(placement.rotation)).label(placement.name, loc="top"))


def _draw_unknown(d, elm_module, placement: ComponentPlacement) -> None:
    """Fallback for kinds without a symbol mapping: labeled rectangle."""
    anchors = placement.terminal_anchors
    if len(anchors) >= 2:
        t0 = _to_canvas(anchors[0].x, anchors[0].y)
        t1 = _to_canvas(anchors[1].x, anchors[1].y)
        if abs(t0[0] - t1[0]) < 1e-6 and abs(t0[1] - t1[1]) < 1e-6:
            t1 = (t1[0] + _DEGENERATE_OFFSET, t1[1])
        d.add(elm_module.RBox().at(t0).to(t1).label(f"{placement.name}\n[{placement.kind}]", loc="top"))
    else:
        cx, cy = _to_canvas(placement.x, placement.y)
        d.add(elm_module.RBox().at((cx, cy)).label(f"{placement.name}\n[{placement.kind}]"))


def _draw_wire(d, elm_module, wire: Wire, placements: dict[str, ComponentPlacement]) -> None:
    """Straight-line wire between the two endpoint terminal anchors.

    The wire payload carries (component_name, terminal_index) at each
    endpoint plus a polyline ``path``. Phase 3 ignores the polyline and
    just draws a straight line between the two terminal-anchor coords
    fetched from the placements dict (which is the canonical source of
    coordinates after layout).
    """
    # Prefer the wire's explicit `path` (set by the ELK backend with
    # orthogonal bend points) over the terminal-anchor straight-line
    # fallback (legacy spring-layout backend). When `path` has 3+ points
    # we draw each consecutive segment as its own Line so Manhattan
    # bends are preserved in the rendered SVG/PNG.
    if wire.path and len(wire.path) >= 2:
        pts = [_to_canvas(px, py) for px, py in wire.path]
    else:
        p_from = placements[wire.from_.component].terminal_anchors[wire.from_.terminal]
        p_to = placements[wire.to.component].terminal_anchors[wire.to.terminal]
        pts = [_to_canvas(p_from.x, p_from.y), _to_canvas(p_to.x, p_to.y)]

    # Drop consecutive duplicate points (ELK can emit them at port boundaries).
    deduped: list[tuple[float, float]] = []
    for pt in pts:
        if not deduped or abs(pt[0] - deduped[-1][0]) > 1e-9 or abs(pt[1] - deduped[-1][1]) > 1e-9:
            deduped.append(pt)
    if len(deduped) < 2:
        return
    for a, b in zip(deduped[:-1], deduped[1:]):
        d.add(elm_module.Line().at(a).to(b))


def render_layout(layout: SchematicLayout, path: Any, format: str | None = None) -> Path:
    """Render a pre-computed layout to a file.

    Phase 8 default: native SVG renderer (``svg_renderer.render_svg``)
    where each component's symbol is drawn so its terminals land
    exactly at the ``terminal_anchors`` from the layout — guarantees
    visual connectivity. Set ``PULSIM_SCHEMATIC_RENDERER=schemdraw``
    to fall back to the legacy schemdraw-based pipeline (analog skin,
    prettier symbols, but with the connectivity issues Phase 8
    addresses).

    Args:
        layout: A :class:`SchematicLayout` (typically from
            :func:`compute_layout`).
        path: Output file path; the extension drives the format unless
            ``format=`` is passed.
        format: One of ``"svg"``, ``"png"``, ``"pdf"``, ``"jpg"``.

    Returns:
        The resolved :class:`pathlib.Path` of the written file.
    """
    import os as _os
    out_path = Path(path)
    fmt = format or _infer_format(out_path)
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported schematic format {fmt!r}; "
            f"expected one of {sorted(_SUPPORTED_FORMATS)}"
        )

    renderer = _os.environ.get(
        "PULSIM_SCHEMATIC_RENDERER", "svg").strip().lower()

    # Default path — native SVG renderer with guaranteed-connected
    # symbols.
    if renderer == "svg":
        from .svg_renderer import render_svg, render_png
        if fmt == "svg":
            return render_svg(layout, out_path)
        if fmt == "png":
            return render_png(layout, out_path)
        # For pdf/jpg, fall through to schemdraw (it has those exporters).

    # Schemdraw fallback / opt-in legacy path.
    schemdraw, elm = _require_schemdraw()
    with schemdraw.Drawing(show=False) as d:
        for placement in layout.components.values():
            factory = get_symbol_factory(placement.kind)
            if factory is None:
                _draw_unknown(d, elm, placement)
                continue
            if len(placement.terminal_anchors) == 2:
                _draw_two_terminal(d, factory, placement)
            else:
                _draw_multi_terminal(d, factory, placement)
        for wire in layout.wires:
            _draw_wire(d, elm, wire, layout.components)
        for jx, jy in layout.junctions:
            d.add(elm.Dot().at(_to_canvas(jx, jy)))
        image_bytes = d.get_imagedata(fmt=fmt)

    out_path.write_bytes(image_bytes)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"schemdraw produced no output at {out_path}; format={fmt!r}"
        )
    return out_path


def _svg_to_png(svg_path: Path, png_path: Path, width_px: int = 1400) -> Path:
    """Convert an SVG file to PNG using ``rsvg-convert`` (preferred) or
    ``cairosvg`` as a fallback. Raises if neither is available.
    """
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        result = subprocess.run(
            [rsvg, "-w", str(width_px), str(svg_path), "-o", str(png_path)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and png_path.exists() and png_path.stat().st_size > 0:
            return png_path
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"rsvg-convert failed (exit {result.returncode}): {stderr}")
    # Fallback to cairosvg if installed
    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "PNG output from the netlistsvg backend requires either `rsvg-convert` "
            "(`brew install librsvg` on macOS, `apt-get install librsvg2-bin` on "
            "Debian/Ubuntu) or the `cairosvg` Python package (`pip install cairosvg`). "
            "Alternatively, save as `.svg` directly."
        ) from exc
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width_px)
    return png_path


def render(
    circuit: Any,
    path: Any,
    format: str | None = None,
) -> Path:
    """Render a Pulsim ``Circuit`` to a schematic file.

    The default backend is **python_native** — a pure-Python renderer
    that reuses the ELK Java bridge for layout. Override via the
    ``PULSIM_SCHEMATIC_BACKEND`` environment variable:

    - ``python_native`` (default): in-tree ELK layered layout + Python
      SVG emitter. SVG is native output; PNG is supported via
      ``cairosvg`` or ``rsvg-convert``.
    - ``netlistsvg`` (deprecated): subprocess to the ``netlistsvg`` CLI
      with the bundled analog skin. Emits a deprecation warning when
      explicitly selected.
    - ``elk``: in-tree ELK layered layout + schemdraw render.
    - ``spring``: legacy force-directed + templates + schemdraw.

    The output format is inferred from the path extension (``.svg``,
    ``.png``, ``.pdf``, ``.jpg``) unless ``format=`` is passed.

    .. note::

       v1.3 removed the previously-documented ``position_hints=`` kwarg
       — neither backend ever shipped a working implementation (both
       raised ``NotImplementedError`` on non-empty input). Component
       placement is governed by the auto-layout. The follow-up work is
       tracked in ``openspec/changes/add-schematic-renderer-v2`` and the
       upstream-bug analysis lives in
       ``openspec/changes/add-schematic-position-hints``.

    Args:
        circuit: A Pulsim ``Circuit`` (or anything matching the
            ``components()`` / ``ground()`` / ``node_position_hint()``
            contract).
        path: Output file path.
        format: Optional explicit format override.

    Returns:
        The :class:`pathlib.Path` of the written file.
    """
    out_path = Path(path)
    fmt = format or _infer_format(out_path)
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported schematic format {fmt!r}; "
            f"expected one of {sorted(_SUPPORTED_FORMATS)}"
        )

    raw_backend = os.environ.get("PULSIM_SCHEMATIC_BACKEND", "").strip().lower()
    # Phase 5: the default is now python_native. Unset env var → native.
    backend = raw_backend or "python_native"
    if backend not in ("netlistsvg", "python_native", "elk", "spring"):
        backend = "python_native"

    # Deprecation warning when the user explicitly selects netlistsvg.
    # Only fires on explicit opt-in — the unset / native paths stay
    # silent so existing tooling doesn't flood logs after the switch.
    if raw_backend == "netlistsvg":
        import warnings
        warnings.warn(
            "The 'netlistsvg' schematic backend is deprecated and will be "
            "removed in a future release. The default (python_native) "
            "produces equivalent output without the Node-side netlistsvg "
            "subprocess. Set PULSIM_SCHEMATIC_BACKEND=python_native (or "
            "leave it unset) to silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )

    if backend == "python_native":
        # `add-python-schematic-renderer` (Phases 1-4): pure-Python
        # renderer that reuses `elk_bridge.js` for layout. The
        # position-hint pipeline (placing user-specified cells on a
        # semantic grid) is tracked in
        # ``openspec/changes/add-schematic-renderer-v2``; until then the
        # auto-layout governs every cell position.
        from .native_backend import render_native
        if fmt == "svg":
            return render_native(circuit, out_path)
        # PNG / PDF / JPG: render SVG first then convert via rsvg-convert.
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            svg_tmp = Path(tmp.name)
        try:
            render_native(circuit, svg_tmp)
            if fmt == "png":
                _svg_to_png(svg_tmp, out_path)
            else:
                rsvg = shutil.which("rsvg-convert")
                if not rsvg:
                    raise ImportError(
                        f"PNG/PDF/JPG output from the python_native backend "
                        f"requires `rsvg-convert`. Save as .svg instead, or "
                        f"install librsvg. Requested format: {fmt!r}"
                    )
                subprocess.run(
                    [rsvg, "-f", fmt, "-w", "1400", str(svg_tmp), "-o", str(out_path)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            if not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError(
                    f"python_native + conversion produced no output at {out_path}"
                )
            return out_path
        finally:
            try:
                svg_tmp.unlink()
            except OSError:
                pass

    if backend == "netlistsvg":
        if fmt == "svg":
            return render_netlistsvg(circuit, out_path)
        # PNG / PDF / JPG: render to a temp SVG first, then convert.
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            svg_tmp = Path(tmp.name)
        try:
            render_netlistsvg(circuit, svg_tmp)
            if fmt == "png":
                _svg_to_png(svg_tmp, out_path)
            else:
                # PDF / JPG: rsvg-convert supports both via its `-f` flag.
                rsvg = shutil.which("rsvg-convert")
                if not rsvg:
                    raise ImportError(
                        f"PNG/PDF/JPG output from the netlistsvg backend requires "
                        f"`rsvg-convert`. Save as .svg instead, or install librsvg. "
                        f"Requested format: {fmt!r}"
                    )
                subprocess.run(
                    [rsvg, "-f", fmt, "-w", "1400", str(svg_tmp), "-o", str(out_path)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            if not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError(
                    f"netlistsvg + conversion produced no output at {out_path}"
                )
            return out_path
        finally:
            try:
                svg_tmp.unlink()
            except OSError:
                pass

    # ELK or spring path: go through the SchematicLayout pipeline.
    layout = compute_layout(circuit)
    return render_layout(layout, path, format=format)
