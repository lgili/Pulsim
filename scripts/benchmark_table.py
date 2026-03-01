#!/usr/bin/env python3
"""Render benchmark/parity results as a terminal table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt_float(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_sci(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return "-"


def _status_tag(status: Any) -> str:
    text = str(status or "-").strip().lower()
    mapping = {
        "passed": "PASS",
        "failed": "FAIL",
        "baseline": "BASE",
        "skipped": "SKIP",
    }
    return mapping.get(text, text.upper() if text else "-")


def _count_status(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"passed": 0, "failed": 0, "baseline": 0, "skipped": 0}
    for item in items:
        status = str(item.get("status", "")).strip().lower()
        if status in counts:
            counts[status] += 1
    counts["total"] = len(items)
    return counts


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def line(sep: str = "-") -> str:
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    print(line("-"))
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(line("="))
    for row in rows:
        print("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    print(line("-"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render benchmark and parity summary table")
    parser.add_argument("--runtime", type=Path, default=None, help="Path to benchmark_runner results.json")
    parser.add_argument("--parity", type=Path, default=None, help="Path to benchmark_ngspice parity_results.json")
    parser.add_argument("--title", type=str, default="Benchmark Summary")
    args = parser.parse_args()

    if args.runtime is None and args.parity is None:
        raise SystemExit("Provide --runtime and/or --parity")

    runtime_items: List[Dict[str, Any]] = []
    parity_items: List[Dict[str, Any]] = []

    if args.runtime is not None:
        runtime_payload = _load_json(args.runtime)
        runtime_items = list(runtime_payload.get("results", []))

    if args.parity is not None:
        parity_payload = _load_json(args.parity)
        parity_items = list(parity_payload.get("results", []))

    runtime_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in runtime_items:
        key = (str(item.get("benchmark_id", "")), str(item.get("scenario", "")))
        runtime_map[key] = item

    parity_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in parity_items:
        key = (str(item.get("benchmark_id", "")), str(item.get("scenario", "")))
        parity_map[key] = item

    keys = sorted(set(runtime_map.keys()) | set(parity_map.keys()))

    print()
    print(args.title)
    print("=" * len(args.title))

    if runtime_items:
        c = _count_status(runtime_items)
        print(
            f"runtime: total={c['total']} pass={c['passed']} fail={c['failed']} "
            f"baseline={c['baseline']} skip={c['skipped']}"
        )
    if parity_items:
        c = _count_status(parity_items)
        print(
            f"ltspice: total={c['total']} pass={c['passed']} fail={c['failed']} "
            f"skip={c['skipped']}"
        )

    headers = [
        "benchmark",
        "scenario",
        "runtime",
        "t_pulsim[s]",
        "max_err_rt",
        "ltspice",
        "t_lt[s]",
        "max_err_lt",
        "speedup",
    ]

    rows: List[List[str]] = []
    for key in keys:
        bench, scenario = key
        rt = runtime_map.get(key)
        lt = parity_map.get(key)

        rt_status = _status_tag(rt.get("status")) if rt else "-"
        rt_time = _fmt_float(rt.get("runtime_s"), 4) if rt else "-"
        rt_err = _fmt_sci(rt.get("max_error")) if rt else "-"

        lt_status = _status_tag(lt.get("status")) if lt else "-"
        lt_time_value = None
        if lt is not None:
            lt_time_value = lt.get("reference_runtime_s")
            if lt_time_value is None:
                lt_time_value = lt.get("ngspice_runtime_s")
            if lt_time_value is None:
                lt_time_value = lt.get("ltspice_runtime_s")
        lt_time = _fmt_float(lt_time_value, 4)
        lt_err = _fmt_sci(lt.get("max_error")) if lt else "-"
        speedup = _fmt_float(lt.get("speedup"), 2) if lt else "-"

        rows.append([bench, scenario, rt_status, rt_time, rt_err, lt_status, lt_time, lt_err, speedup])

    _print_table(headers, rows)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
