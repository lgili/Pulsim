#!/usr/bin/env python3
"""Check one-way dependency boundaries for mapped Pulsim v1 core layers."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


INCLUDE_RE = re.compile(r'^\s*#include\s*"([^"]+)"')


@dataclass(frozen=True)
class Violation:
    from_file: str
    from_module: str
    to_file: str
    to_module: str
    include: str
    line: int
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "from_file": self.from_file,
            "from_module": self.from_module,
            "to_file": self.to_file,
            "to_module": self.to_module,
            "include": self.include,
            "line": self.line,
            "reason": self.reason,
        }


def _load_map(map_path: Path) -> Tuple[List[str], Dict[str, str], List[str]]:
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pulsim-v1-layer-map-v1":
        raise ValueError("invalid map schema")

    layer_order = payload.get("layer_order")
    if not isinstance(layer_order, list) or not all(isinstance(x, str) for x in layer_order):
        raise ValueError("layer_order must be a list of strings")

    modules = payload.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("modules must be an object")

    file_to_module: Dict[str, str] = {}
    errors: List[str] = []

    for module_name in layer_order:
        if module_name not in modules:
            errors.append(f"module '{module_name}' missing from modules")
            continue
        module_payload = modules[module_name]
        if not isinstance(module_payload, dict):
            errors.append(f"module '{module_name}' payload must be an object")
            continue
        files = module_payload.get("files", [])
        if not isinstance(files, list):
            errors.append(f"module '{module_name}' files must be a list")
            continue
        for raw_path in files:
            if not isinstance(raw_path, str):
                errors.append(f"module '{module_name}' has non-string file entry")
                continue
            normalized = str(Path(raw_path))
            if normalized in file_to_module:
                errors.append(
                    f"file '{normalized}' assigned to multiple modules "
                    f"({file_to_module[normalized]} and {module_name})"
                )
                continue
            file_to_module[normalized] = module_name

    return layer_order, file_to_module, errors


def _resolve_project_include(project_root: Path, src_file: Path, include_path: str) -> Optional[str]:
    if include_path.startswith("pulsim/v1/"):
        candidate = (project_root / "core" / "include" / include_path).resolve()
    else:
        candidate = (src_file.parent / include_path).resolve()

    try:
        return str(candidate.relative_to(project_root))
    except ValueError:
        return None


def check_boundaries(project_root: Path, map_path: Path) -> Dict[str, object]:
    layer_order, file_to_module, map_errors = _load_map(map_path)
    layer_index = {name: idx for idx, name in enumerate(layer_order)}

    report: Dict[str, object] = {
        "schema_version": "pulsim-v1-layer-boundary-report-v1",
        "map_path": str(map_path),
        "status": "passed",
        "map_errors": map_errors,
        "map_file_count": len(file_to_module),
        "checked_file_count": 0,
        "dependency_edges_checked": 0,
        "violations": [],
        "missing_files": [],
    }

    if map_errors:
        report["status"] = "failed"
        return report

    violations: List[Violation] = []
    missing_files: List[str] = []
    checked_file_count = 0
    dependency_edges_checked = 0

    for rel_src, src_module in sorted(file_to_module.items()):
        src_path = (project_root / rel_src).resolve()
        if not src_path.is_file():
            missing_files.append(rel_src)
            continue
        checked_file_count += 1

        src_module_idx = layer_index[src_module]
        for line_no, line in enumerate(src_path.read_text(encoding="utf-8").splitlines(), start=1):
            match = INCLUDE_RE.match(line)
            if match is None:
                continue
            include_path = match.group(1)
            rel_target = _resolve_project_include(project_root, src_path, include_path)
            if rel_target is None:
                continue
            target_module = file_to_module.get(rel_target)
            if target_module is None:
                continue

            dependency_edges_checked += 1
            target_module_idx = layer_index[target_module]
            if target_module_idx > src_module_idx:
                violations.append(
                    Violation(
                        from_file=rel_src,
                        from_module=src_module,
                        to_file=rel_target,
                        to_module=target_module,
                        include=include_path,
                        line=line_no,
                        reason="upward dependency (lower layer depends on higher layer)",
                    )
                )

    report["checked_file_count"] = checked_file_count
    report["dependency_edges_checked"] = dependency_edges_checked
    report["missing_files"] = missing_files
    report["violations"] = [item.as_dict() for item in violations]

    if missing_files or violations:
        report["status"] = "failed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Pulsim v1 layer boundaries")
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("core/v1_layer_map.json"),
        help="Path to the layer map json",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when violations or map errors are found",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable json report",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    map_path = (project_root / args.map).resolve() if not args.map.is_absolute() else args.map.resolve()
    report = check_boundaries(project_root=project_root, map_path=map_path)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {report['status']}")
        print(f"mapped files: {report['map_file_count']}")
        print(f"checked files: {report['checked_file_count']}")
        print(f"edges checked: {report['dependency_edges_checked']}")
        missing_files = report.get("missing_files", [])
        if missing_files:
            print("missing mapped files:")
            for item in missing_files:
                print(f"  - {item}")
        violations = report.get("violations", [])
        if violations:
            print("violations:")
            for item in violations:
                print(
                    "  - "
                    f"{item['from_module']}:{item['from_file']} -> "
                    f"{item['to_module']}:{item['to_file']} "
                    f"(line {item['line']}, include={item['include']})"
                )

    if args.strict and report.get("status") != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
