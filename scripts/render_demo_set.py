"""Render the canonical demo set (RC, buck, boost PFC) for visual review."""

from __future__ import annotations

import sys
from pathlib import Path

import pulsim as ps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "schematic_demo"
OUT.mkdir(parents=True, exist_ok=True)


def render_yaml(yaml_path: Path, out_stem: str) -> None:
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _opts = parser.load(str(yaml_path))
    png = OUT / f"{out_stem}.png"
    svg = OUT / f"{out_stem}.svg"
    ps.schematic.render(ckt, png)
    ps.schematic.render(ckt, svg)
    layout = ps.schematic.compute_layout(ckt)
    print(f"  {out_stem}: {len(layout.components)} comps, "
          f"{len(layout.wires)} wires, {len(layout.junctions)} junctions  "
          f"-> {png.name} ({png.stat().st_size} B), "
          f"{svg.name} ({svg.stat().st_size} B)")


def main() -> None:
    print("ELK backend demo renders:")
    render_yaml(ROOT / "examples" / "rc_circuit.yaml", "rc")
    render_yaml(ROOT / "examples" / "buck_converter.yaml", "buck")
    # Boost PFC via the existing build script.
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_boost_pfc import build_boost_pfc  # type: ignore
    ckt = build_boost_pfc()
    png = OUT / "boost_pfc.png"
    svg = OUT / "boost_pfc.svg"
    ps.schematic.render(ckt, png)
    ps.schematic.render(ckt, svg)
    layout = ps.schematic.compute_layout(ckt)
    print(f"  boost_pfc: {len(layout.components)} comps, "
          f"{len(layout.wires)} wires, {len(layout.junctions)} junctions  "
          f"-> {png.name}, {svg.name}")


if __name__ == "__main__":
    main()
