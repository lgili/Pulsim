"""Optional NgSpice/PySpice runner for reference simulations.

This module is intentionally lightweight: it only attempts to import
PySpice when used. If PySpice or a shared ngspice cannot be loaded,
callers should handle RuntimeError and skip the comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def is_pyspice_available() -> bool:
    """Check if PySpice is importable and ngspice is present."""
    try:
        import PySpice  # noqa: F401
        return True
    except Exception:
        return False


def run_ngspice_transient(netlist_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Run a transient analysis using PySpice/ngspice.

    Args:
        netlist_path: Path to a SPICE netlist file

    Returns:
        times, node_voltages (2D array: shape [N, num_nodes])
    """
    if not is_pyspice_available():
        raise RuntimeError("PySpice/ngspice not available. Install pyspice>=1.5 and ensure ngspice is on PATH.")

    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from PySpice.Spice.Parser import SpiceParser

    netlist_path = Path(netlist_path)
    if not netlist_path.exists():
        raise FileNotFoundError(f"Netlist not found: {netlist_path}")

    parser = SpiceParser(str(netlist_path))
    circuit = parser.build_circuit()
    simulator = circuit.simulator(simulator=NgSpiceShared.new_instance())

    analysis = simulator.transient()
    times = np.array(analysis.time)

    # Collect node voltages into a matrix (excluding ground)
    node_names = [n for n in analysis.nodes.keys() if n != "0"]
    volt_matrix = np.column_stack([np.array(analysis[node]) for node in node_names])

    return times, volt_matrix


def run_ngspice_dc(netlist_path: str | Path) -> np.ndarray:
    """Run a DC operating point using PySpice/ngspice.

    Args:
        netlist_path: Path to a SPICE netlist file

    Returns:
        Node voltages (excluding ground) as numpy array
    """
    if not is_pyspice_available():
        raise RuntimeError("PySpice/ngspice not available. Install pyspice>=1.5 and ensure ngspice is on PATH.")

    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from PySpice.Spice.Parser import SpiceParser

    netlist_path = Path(netlist_path)
    if not netlist_path.exists():
        raise FileNotFoundError(f"Netlist not found: {netlist_path}")

    parser = SpiceParser(str(netlist_path))
    circuit = parser.build_circuit()
    simulator = circuit.simulator(simulator=NgSpiceShared.new_instance())

    analysis = simulator.operating_point()
    node_names = [n for n in analysis.nodes.keys() if n != "0"]
    return np.array([float(analysis[node]) for node in node_names])


__all__ = [
    "is_pyspice_available",
    "run_ngspice_transient",
    "run_ngspice_dc",
]
