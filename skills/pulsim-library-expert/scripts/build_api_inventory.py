#!/usr/bin/env python3
"""Generate a Markdown inventory of the Pulsim Python API surface.

The inventory is derived from:
- python/pulsim/__init__.py (export surface via __all__)
- python/pulsim/__init__.pyi (typed classes, enums, attributes, and functions)
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable


@dataclass
class ClassInfo:
    name: str
    bases: list[str] = field(default_factory=list)
    is_enum: bool = False
    enum_values: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)


def normalize_signature(text: str) -> str:
    """Collapse multi-line function signatures for readable Markdown."""
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace(": ...", "")
    return text


def parse_pyi(pyi_path: Path) -> tuple[dict[str, ClassInfo], list[str], list[str]]:
    source = pyi_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    classes: dict[str, ClassInfo] = {}
    functions: list[str] = []
    top_level_variables: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(base) for base in node.bases]
            is_enum = any(base.endswith("Enum") for base in bases)
            info = ClassInfo(name=node.name, bases=bases, is_enum=is_enum)

            for item in node.body:
                if isinstance(item, ast.Assign):
                    if is_enum:
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                info.enum_values.append(target.id)
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        segment = ast.get_source_segment(source, item) or ""
                        segment = segment.strip()
                        if segment:
                            info.attributes.append(segment)
                elif isinstance(item, ast.FunctionDef):
                    segment = ast.get_source_segment(source, item) or ""
                    segment = normalize_signature(segment)
                    if segment:
                        info.methods.append(segment)

            classes[node.name] = info
            continue

        if isinstance(node, ast.FunctionDef):
            segment = ast.get_source_segment(source, node) or ""
            segment = normalize_signature(segment)
            if segment:
                functions.append(segment)
            continue

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_variables.append(node.target.id)
            continue

    return classes, functions, top_level_variables


def parse_exports(init_path: Path) -> list[str]:
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    exports: list[str] = []
                    for element in node.value.elts:
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            exports.append(element.value)
                    return exports
    return []


def markdown_list(items: Iterable[str], indent: str = "") -> list[str]:
    out: list[str] = []
    for item in items:
        out.append(f"{indent}- `{item}`")
    return out


def generate_markdown(
    init_path: Path,
    pyi_path: Path,
    classes: dict[str, ClassInfo],
    functions: list[str],
    top_level_variables: list[str],
    exports: list[str],
) -> str:
    documented_symbols = set(classes.keys()) | {sig.split("(", 1)[0].replace("def ", "") for sig in functions} | set(top_level_variables)
    exported_symbols = set(exports)

    missing_from_stub = sorted(exported_symbols - documented_symbols)
    not_exported = sorted(documented_symbols - exported_symbols)

    enum_names = sorted(name for name, info in classes.items() if info.is_enum)
    class_names = sorted(name for name, info in classes.items() if not info.is_enum)

    lines: list[str] = []
    lines.append("# Pulsim Python API Inventory")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("- [Refresh](#refresh)")
    lines.append("- [Summary](#summary)")
    lines.append("- [Enums](#enums)")
    lines.append("- [Classes](#classes)")
    lines.append("- [Top-Level Functions](#top-level-functions)")
    lines.append("- [Export Gaps](#export-gaps)")
    lines.append("")
    lines.append("## Refresh")
    lines.append("")
    lines.append(f"Generated on: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Export source: `{init_path}`")
    lines.append(f"- Type surface source: `{pyi_path}`")
    lines.append("")
    lines.append("Regenerate with:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 skills/pulsim-library-expert/scripts/build_api_inventory.py")
    lines.append("```")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Exported symbols (`__all__`): **{len(exports)}**")
    lines.append(f"- Enums in stub: **{len(enum_names)}**")
    lines.append(f"- Classes in stub: **{len(class_names)}**")
    lines.append(f"- Top-level functions in stub: **{len(functions)}**")
    lines.append(f"- Top-level variables in stub: **{len(top_level_variables)}**")
    lines.append("")

    lines.append("## Enums")
    lines.append("")
    if not enum_names:
        lines.append("No enums found in stub.")
        lines.append("")
    for name in enum_names:
        info = classes[name]
        lines.append(f"### {name}")
        lines.append("")
        if info.enum_values:
            lines.extend(markdown_list(info.enum_values))
        else:
            lines.append("- No explicit values parsed")
        lines.append("")

    lines.append("## Classes")
    lines.append("")
    if not class_names:
        lines.append("No classes found in stub.")
        lines.append("")
    for name in class_names:
        info = classes[name]
        base_suffix = f" (bases: {', '.join(info.bases)})" if info.bases else ""
        lines.append(f"### {name}{base_suffix}")
        lines.append("")

        if info.attributes:
            lines.append("Attributes:")
            lines.extend(markdown_list(info.attributes))
            lines.append("")

        if info.methods:
            lines.append("Methods:")
            lines.extend(markdown_list(info.methods))
            lines.append("")

        if not info.attributes and not info.methods:
            lines.append("- No attributes or methods parsed")
            lines.append("")

    lines.append("## Top-Level Functions")
    lines.append("")
    if functions:
        lines.extend(markdown_list(functions))
    else:
        lines.append("- No top-level functions parsed")
    lines.append("")

    lines.append("## Export Gaps")
    lines.append("")
    lines.append("Symbols exported in `__all__` but not typed in `__init__.pyi`:")
    if missing_from_stub:
        lines.extend(markdown_list(missing_from_stub))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("Symbols present in stub but not exported in `__all__`:")
    if not_exported:
        lines.extend(markdown_list(not_exported))
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Pulsim API inventory reference")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root (defaults to script-based autodetect)",
    )
    parser.add_argument(
        "--output",
        default="skills/pulsim-library-expert/references/python-api-inventory.md",
        help="Output Markdown path relative to repo root",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    init_path = repo_root / "python/pulsim/__init__.py"
    pyi_path = repo_root / "python/pulsim/__init__.pyi"
    output_path = repo_root / args.output

    for path in (init_path, pyi_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    classes, functions, top_level_variables = parse_pyi(pyi_path)
    exports = parse_exports(init_path)

    markdown = generate_markdown(
        init_path=init_path,
        pyi_path=pyi_path,
        classes=classes,
        functions=functions,
        top_level_variables=top_level_variables,
        exports=exports,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown + "\n", encoding="utf-8")
    print(f"[OK] Wrote API inventory: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
