"""Pulsim v2 — KPI gates and baseline checks (Phase E.5).

Validation harness for simulation results. Define a `KpiGate` from a
dict of named bounds or expected values, then `gate.check(measured)`
returns a `KpiReport` with per-metric pass/fail and a summary. Used
for:

  * CI regression: "did the latest commit break the buck design KPIs?"
  * Design rounds: "does this prototype meet the V_out ≤ 12.5 spec?"
  * Sweep filtering: "which (L, C) combinations pass ALL gates?"

Spec syntax — `KpiGate` accepts either tuple bounds or comparator
strings:

    gate = p.KpiGate({
        "v_out_mean":    (11.5, 12.5),     # range  → 11.5 ≤ x ≤ 12.5
        "v_out_ripple":  ("<", 0.5),       # bound  → x < 0.5
        "i_peak":        ("<=", 10.0),     # bound  → x ≤ 10.0
        "efficiency":    (">", 0.85),      # bound  → x > 0.85
        "settling_time": (None, 10e-3),    # open lower → x ≤ 10e-3
        "v_ref":         12.0,             # exact equality (rtol)
    })
    report = gate.check({
        "v_out_mean":    12.01,
        "v_out_ripple":  0.42,
        "i_peak":         8.7,
        "efficiency":    0.91,
        "settling_time": 8.5e-3,
        "v_ref":         12.0001,
    })
    print(report.summary())     # human-readable table
    print(report.passed)        # True if all gates passed

Comparator strings: ``"<"``, ``"<="``, ``">"``, ``">="``, ``"=="``.
Equality uses a relative tolerance (default 1 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


__all__ = [
    "KpiGate",
    "KpiReport",
    "KpiCheckResult",
]


@dataclass
class KpiCheckResult:
    """Outcome of one metric check."""
    name: str
    value: float
    spec: str            # human description of the bound, e.g. "11.5 ≤ x ≤ 12.5"
    passed: bool
    detail: str = ""


@dataclass
class KpiReport:
    """Aggregate result over all metrics in a KpiGate.check() call."""
    results: list = field(default_factory=list)
    passed: bool = True

    def add(self, r: KpiCheckResult) -> None:
        self.results.append(r)
        if not r.passed:
            self.passed = False

    def summary(self, *, mark_pass: str = "PASS",
                  mark_fail: str = "FAIL") -> str:
        """Human-readable table of all checks."""
        if not self.results:
            return "(empty)"
        # Column widths.
        w_name = max(len(r.name) for r in self.results)
        w_spec = max(len(r.spec) for r in self.results)
        lines = []
        lines.append(f"  {'name':<{w_name}}  {'value':>12}  "
                       f"{'spec':<{w_spec}}  result")
        lines.append("  " + "-"*(w_name + 12 + w_spec + 18))
        for r in self.results:
            mark = mark_pass if r.passed else mark_fail
            extra = f"  ({r.detail})" if r.detail and not r.passed else ""
            lines.append(f"  {r.name:<{w_name}}  {r.value:>12.4g}  "
                          f"{r.spec:<{w_spec}}  {mark}{extra}")
        lines.append("")
        total = len(self.results)
        passes = sum(r.passed for r in self.results)
        lines.append(f"  Overall: {passes}/{total} passed "
                       f"({'ALL PASS' if self.passed else 'FAILED'})")
        return "\n".join(lines)

    def failures(self) -> list:
        """Return only the failing checks."""
        return [r for r in self.results if not r.passed]


class KpiGate:
    """Define spec gates over a set of named metrics.

    The constructor takes a dict mapping metric name to either:
      * A tuple ``(lo, hi)`` — both finite → closed range. ``None`` on
        either side → open-ended.
      * A tuple ``(comparator_str, threshold)`` where comparator is
        one of ``"<"``, ``"<="``, ``">"``, ``">="``, ``"=="``.
      * A scalar — interpreted as exact equality with `rtol`
        relative tolerance.

    Parameters
    ----------
    specs
        The dict described above.
    rtol
        Relative tolerance used for scalar "==" specs. Default 1e-2.
    """

    _CMP = {"<", "<=", ">", ">=", "=="}

    def __init__(self, specs: Dict[str, Any], *, rtol: float = 1e-2):
        self.specs = dict(specs)
        self.rtol = float(rtol)

    def check(self, measured: Dict[str, float]) -> KpiReport:
        """Evaluate every spec against the corresponding measured
        value. Returns a `KpiReport`."""
        report = KpiReport()
        for name, spec in self.specs.items():
            if name not in measured:
                report.add(KpiCheckResult(
                    name=name, value=float("nan"),
                    spec=str(spec), passed=False,
                    detail="metric not measured"))
                continue
            value = float(measured[name])
            passed, spec_str, detail = self._check_one(spec, value)
            report.add(KpiCheckResult(
                name=name, value=value,
                spec=spec_str, passed=passed,
                detail=detail))
        return report

    def _check_one(self, spec: Any,
                    value: float) -> Tuple[bool, str, str]:
        # Scalar → exact equality.
        if isinstance(spec, (int, float)):
            ref = float(spec)
            tol = self.rtol * max(abs(ref), 1e-12)
            ok = abs(value - ref) <= tol
            return (ok, f"≈ {ref:.4g} (±{self.rtol*100:.1f} %)",
                      "" if ok else f"|Δ|={abs(value-ref):.3g}")

        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError(
                f"KpiGate spec must be tuple or scalar; got {spec!r}")

        a, b = spec
        # Comparator string form.
        if isinstance(a, str) and a in self._CMP:
            threshold = float(b)
            if a == "<":
                ok = value < threshold
            elif a == "<=":
                ok = value <= threshold
            elif a == ">":
                ok = value > threshold
            elif a == ">=":
                ok = value >= threshold
            elif a == "==":
                tol = self.rtol * max(abs(threshold), 1e-12)
                ok = abs(value - threshold) <= tol
            else:
                ok = False
            return ok, f"{a} {threshold:.4g}", ""

        # Range form (lo, hi) — None means open-ended.
        lo, hi = a, b
        lo_str = f"{lo:.4g}" if lo is not None else "−∞"
        hi_str = f"{hi:.4g}" if hi is not None else "+∞"
        ok_lo = (lo is None) or (value >= float(lo))
        ok_hi = (hi is None) or (value <= float(hi))
        return (ok_lo and ok_hi,
                  f"{lo_str} ≤ x ≤ {hi_str}",
                  "" if (ok_lo and ok_hi) else
                      (f"below {lo_str}" if not ok_lo
                        else f"above {hi_str}"))


def load_baseline(path) -> Dict[str, float]:
    """Load a baseline KPI dict from a JSON file.

    Used to capture a known-good design state and check against it
    later (e.g. CI: ``current_kpis == baseline.json`` within
    tolerance).
    """
    import json
    from pathlib import Path
    with open(Path(path)) as f:
        return json.load(f)


def save_baseline(kpis: Dict[str, float], path) -> None:
    """Save a KPI dict to a JSON baseline file."""
    import json
    from pathlib import Path
    with open(Path(path), "w") as f:
        json.dump(dict(kpis), f, indent=2)
