"""Pure data layer for `pulsim-bench show` and `pulsim-bench compare`.

Handles only:
    * loading the JSON artifact emitted by any of the legacy runners
    * normalising heterogeneous schemas (results.json, parity_results.json,
      stress_results.json, etc.) into one common record shape
    * computing the diff between two runs

No terminal output, no I/O beyond reading the input file. Easy to unit
test in isolation.

The "common record" shape used by every helper here is a plain dict
with these fields (all optional except the first three):

    {
        "benchmark_id":  str,
        "scenario":      str,
        "status":        str,            # passed | failed | baseline | skipped
        "runtime_s":     float | None,
        "max_error":     float | None,
        "steps":         int   | None,
        "message":       str,
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# Files we know how to consume, in priority order when given a directory.
# `kind` is the user-visible label that subcommands print in the header.
_ARTIFACT_PRIORITY: Tuple[Tuple[str, str], ...] = (
    ("results.json",        "benchmark"),
    ("parity_results.json", "parity"),
    ("stress_results.json", "stress"),
    ("summary.json",        "summary"),
)


def resolve_artifact(path: Path) -> Tuple[Path, str]:
    """Given a file or directory, return `(file, kind_label)`.

    For a file: trust the path; kind is inferred from the filename.
    For a dir: probe `_ARTIFACT_PRIORITY` in order and return the
    first one that exists.

    Raises `FileNotFoundError` when no candidate is present.
    """
    p = path.expanduser().resolve()
    if p.is_file():
        for name, label in _ARTIFACT_PRIORITY:
            if p.name == name:
                return p, label
        return p, "custom"
    if p.is_dir():
        for name, label in _ARTIFACT_PRIORITY:
            candidate = p / name
            if candidate.is_file():
                return candidate, label
        raise FileNotFoundError(
            f"No known artifact found in {p}. "
            f"Looked for: {', '.join(name for name, _ in _ARTIFACT_PRIORITY)}"
        )
    raise FileNotFoundError(f"Path does not exist: {p}")


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalise_one(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map a record from any legacy JSON shape onto the common shape."""
    # `benchmark_runner.results.json` already uses `runtime_s`.
    # `benchmark_ngspice.parity_results.json` uses `pulsim_runtime_s`.
    # `stress_results.json` mirrors `benchmark_runner` for its records.
    runtime = (
        _coerce_optional_float(record.get("runtime_s"))
        if "runtime_s" in record
        else _coerce_optional_float(record.get("pulsim_runtime_s"))
    )
    steps = (
        _coerce_optional_int(record.get("steps"))
        if "steps" in record
        else _coerce_optional_int(record.get("pulsim_steps"))
    )
    return {
        "benchmark_id": str(record.get("benchmark_id", "?")),
        "scenario": str(record.get("scenario", "?")),
        "status": str(record.get("status", "")).strip().lower() or "unknown",
        "runtime_s": runtime,
        "max_error": _coerce_optional_float(record.get("max_error")),
        "steps": steps,
        "message": str(record.get("message", "") or ""),
    }


def load_results(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Load + normalise the results stored at `path` (file or dir).

    Returns `(records, kind_label)`. `records` is a list of dicts in
    the common shape. Empty list if the file contains nothing
    recognisable.
    """
    artifact, kind = resolve_artifact(path)
    with open(artifact, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    # Most artifacts wrap a list under "results". `summary.json` carries
    # only top-level counts; we expose it as an empty list and let the
    # caller decide what to do.
    raw_records: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        candidates = payload.get("results")
        if isinstance(candidates, list):
            raw_records = [c for c in candidates if isinstance(c, dict)]
    elif isinstance(payload, list):
        raw_records = [c for c in payload if isinstance(c, dict)]

    return [_normalise_one(r) for r in raw_records], kind


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class Transition(str, Enum):
    """How a single (benchmark_id, scenario) cell changed between runs."""

    REGRESSED = "regressed"   # passed/baseline → failed
    FIXED = "fixed"           # failed → passed/baseline
    STILL_FAILING = "still_failing"
    STILL_PASSING = "still_passing"
    SKIPPED_NOW = "skipped_now"
    NEW = "new"               # only in run B
    REMOVED = "removed"       # only in run A
    OTHER = "other"


def _classify(status_a: Optional[str], status_b: Optional[str]) -> Transition:
    if status_a is None and status_b is not None:
        return Transition.NEW
    if status_a is not None and status_b is None:
        return Transition.REMOVED
    a = (status_a or "").lower()
    b = (status_b or "").lower()
    ok = {"passed", "baseline"}
    if a in ok and b == "failed":
        return Transition.REGRESSED
    if a == "failed" and b in ok:
        return Transition.FIXED
    if a == "failed" and b == "failed":
        return Transition.STILL_FAILING
    if a in ok and b in ok:
        return Transition.STILL_PASSING
    if b == "skipped":
        return Transition.SKIPPED_NOW
    return Transition.OTHER


@dataclass
class DiffRow:
    benchmark_id: str
    scenario: str
    transition: Transition
    status_a: Optional[str]
    status_b: Optional[str]
    runtime_a: Optional[float]
    runtime_b: Optional[float]
    max_error_a: Optional[float]
    max_error_b: Optional[float]
    message_a: str = ""
    message_b: str = ""

    @property
    def runtime_delta_pct(self) -> Optional[float]:
        if self.runtime_a is None or self.runtime_b is None:
            return None
        if self.runtime_a == 0.0:
            return None
        return (self.runtime_b - self.runtime_a) / self.runtime_a * 100.0


@dataclass
class DiffSummary:
    total: int
    common: int
    regressed: int
    fixed: int
    still_failing: int
    still_passing: int
    new: int
    removed: int
    median_runtime_delta_pct: Optional[float]
    runtime_delta_pcts: List[float] = field(default_factory=list)


def compute_diff(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
) -> Tuple[List[DiffRow], DiffSummary]:
    """Compute per-(benchmark, scenario) diff plus an aggregate summary.

    Pure function. `records_*` are common-shape dicts (use `load_results`
    to obtain them). The output rows are sorted by transition severity
    then by benchmark id."""
    map_a: Dict[Tuple[str, str], Dict[str, Any]] = {
        (r["benchmark_id"], r["scenario"]): r for r in records_a
    }
    map_b: Dict[Tuple[str, str], Dict[str, Any]] = {
        (r["benchmark_id"], r["scenario"]): r for r in records_b
    }
    keys = sorted(set(map_a.keys()) | set(map_b.keys()))

    rows: List[DiffRow] = []
    pct_deltas: List[float] = []
    for key in keys:
        a = map_a.get(key)
        b = map_b.get(key)
        status_a = a["status"] if a else None
        status_b = b["status"] if b else None
        rt_a = a["runtime_s"] if a else None
        rt_b = b["runtime_s"] if b else None
        row = DiffRow(
            benchmark_id=key[0],
            scenario=key[1],
            transition=_classify(status_a, status_b),
            status_a=status_a,
            status_b=status_b,
            runtime_a=rt_a,
            runtime_b=rt_b,
            max_error_a=a["max_error"] if a else None,
            max_error_b=b["max_error"] if b else None,
            message_a=a["message"] if a else "",
            message_b=b["message"] if b else "",
        )
        if row.runtime_delta_pct is not None:
            pct_deltas.append(row.runtime_delta_pct)
        rows.append(row)

    # Sort regressions to the top, fixes second, then alphabetical.
    severity_rank = {
        Transition.REGRESSED: 0,
        Transition.FIXED: 1,
        Transition.STILL_FAILING: 2,
        Transition.NEW: 3,
        Transition.REMOVED: 4,
        Transition.SKIPPED_NOW: 5,
        Transition.OTHER: 6,
        Transition.STILL_PASSING: 7,
    }
    rows.sort(key=lambda r: (severity_rank[r.transition], r.benchmark_id, r.scenario))

    common = len(set(map_a.keys()) & set(map_b.keys()))
    counts: Dict[Transition, int] = {t: 0 for t in Transition}
    for r in rows:
        counts[r.transition] += 1

    median_pct: Optional[float] = None
    if pct_deltas:
        sorted_pcts = sorted(pct_deltas)
        mid = len(sorted_pcts) // 2
        if len(sorted_pcts) % 2 == 1:
            median_pct = sorted_pcts[mid]
        else:
            median_pct = (sorted_pcts[mid - 1] + sorted_pcts[mid]) / 2.0

    summary = DiffSummary(
        total=len(rows),
        common=common,
        regressed=counts[Transition.REGRESSED],
        fixed=counts[Transition.FIXED],
        still_failing=counts[Transition.STILL_FAILING],
        still_passing=counts[Transition.STILL_PASSING],
        new=counts[Transition.NEW],
        removed=counts[Transition.REMOVED],
        median_runtime_delta_pct=median_pct,
        runtime_delta_pcts=pct_deltas,
    )
    return rows, summary
