#!/usr/bin/env python3
"""Update project version across release-managed files.

Usage:
    python scripts/update_version.py 0.3.4
    python scripts/update_version.py v0.3.4 --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    replacement: str
    expected_count: int = 1
    description: str = ""


RULES: dict[str, list[Rule]] = {
    "pyproject.toml": [
        Rule(
            pattern=re.compile(r'(?m)^version\s*=\s*"[^"]+"$'),
            replacement='version = "{version}"',
            description="Python package version",
        ),
    ],
    "CMakeLists.txt": [
        Rule(
            pattern=re.compile(
                r"(?m)^project\(pulsim VERSION \d+\.\d+\.\d+ LANGUAGES CXX\)$"
            ),
            replacement="project(pulsim VERSION {version} LANGUAGES CXX)",
            description="CMake project version",
        ),
    ],
    "python/pulsim/__init__.py": [
        Rule(
            pattern=re.compile(r'(?m)^__version__\s*=\s*"[^"]+"$'),
            replacement='__version__ = "{version}"',
            description="Python runtime __version__",
        ),
    ],
    "docs/python/conf.py": [
        Rule(
            pattern=re.compile(r"(?m)^version\s*=\s*'[^']+'$"),
            replacement="version = '{version}'",
            description="Sphinx short version",
        ),
        Rule(
            pattern=re.compile(r"(?m)^release\s*=\s*'[^']+'$"),
            replacement="release = '{version}'",
            description="Sphinx release version",
        ),
    ],
    "docs/Doxyfile": [
        Rule(
            pattern=re.compile(r'(?m)^PROJECT_NUMBER\s*=\s*"[^"]*"$'),
            replacement='PROJECT_NUMBER         = "{version}"',
            description="Doxygen project number",
        ),
    ],
}


def normalize_version(raw: str) -> str:
    version = raw[1:] if raw.startswith("v") else raw
    if not SEMVER_RE.fullmatch(version):
        raise ValueError(
            f"Invalid version '{raw}'. Expected format X.Y.Z (optionally prefixed with 'v')."
        )
    return version


def apply_rules(path: Path, rules: list[Rule], version: str) -> tuple[bool, str]:
    content = path.read_text(encoding="utf-8")
    original = content
    for rule in rules:
        replacement = rule.replacement.format(version=version)
        content, count = rule.pattern.subn(replacement, content)
        if count != rule.expected_count:
            desc = rule.description or rule.pattern.pattern
            raise RuntimeError(
                f"{path}: expected {rule.expected_count} replacement(s) for '{desc}', got {count}."
            )
    return content != original, content


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update release-managed version strings across project files."
    )
    parser.add_argument("version", help="Target version (X.Y.Z or vX.Y.Z)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    args = parser.parse_args()

    try:
        version = normalize_version(args.version)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    changed_files: list[str] = []
    touched_rules = 0

    for rel_path, rules in RULES.items():
        path = ROOT / rel_path
        if not path.exists():
            print(f"ERROR: Missing file: {path}", file=sys.stderr)
            return 1

        try:
            changed, updated_content = apply_rules(path, rules, version)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        touched_rules += len(rules)
        if changed:
            changed_files.append(rel_path)
            if not args.dry_run:
                path.write_text(updated_content, encoding="utf-8")

    mode = "DRY-RUN" if args.dry_run else "UPDATED"
    print(f"{mode}: set version to {version}")
    print(f"Checked {len(RULES)} files / {touched_rules} rules.")
    if changed_files:
        for rel_path in changed_files:
            print(f" - {rel_path}")
    else:
        print("No file content changes were necessary.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
