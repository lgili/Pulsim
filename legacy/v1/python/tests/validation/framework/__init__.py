"""Validation framework core components."""

from .base import (
    ValidationLevel,
    ValidationResult,
    CircuitDefinition,
    ValidationTest,
)
from .comparator import compare_results, interpolate_to_common_times
from .reporters import (
    write_json_report,
    write_csv_report,
    write_markdown_report,
)
from .spice_runner import (
    is_pyspice_available,
    run_ngspice_transient,
    run_ngspice_dc,
)

__all__ = [
    "ValidationLevel",
    "ValidationResult",
    "CircuitDefinition",
    "ValidationTest",
    "compare_results",
    "interpolate_to_common_times",
    "write_json_report",
    "write_csv_report",
    "write_markdown_report",
    "is_pyspice_available",
    "run_ngspice_transient",
    "run_ngspice_dc",
]
