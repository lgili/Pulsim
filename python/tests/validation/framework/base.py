"""
Base classes for the Pulsim Validation Framework.

This module provides the core infrastructure for validating Pulsim simulation
results against analytical solutions and NgSpice reference simulations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any, Tuple
from enum import Enum
import numpy as np
import time


class ValidationLevel(Enum):
    """Validation complexity levels with associated tolerances."""
    COMPONENT = 1      # Basic isolated components
    LINEAR = 2         # Linear circuits (RC, RL, RLC)
    NONLINEAR = 3      # Nonlinear components (diode, MOSFET)
    CONVERTER = 4      # Power converters
    COMPLEX = 5        # Complex multi-stage circuits


# Default tolerances per validation level
# Note: These are RELATIVE tolerances (e.g., 0.001 = 0.1%)
# The actual threshold is computed as: tolerance * max(|reference_values|)
DEFAULT_TOLERANCES = {
    ValidationLevel.COMPONENT: {"max_error": 1e-4, "rms_error": 1e-5},   # 0.01%, 0.001%
    ValidationLevel.LINEAR: {"max_error": 5e-3, "rms_error": 5e-4},      # 0.5%, 0.05%
    ValidationLevel.NONLINEAR: {"max_error": 1e-2, "rms_error": 1e-3},   # 1%, 0.1%
    ValidationLevel.CONVERTER: {"max_error": 5e-2, "rms_error": 1e-2},   # 5%, 1%
    ValidationLevel.COMPLEX: {"max_error": 1e-1, "rms_error": 2e-2},     # 10%, 2%
}


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    pulsim_time: np.ndarray
    pulsim_values: np.ndarray
    reference_time: np.ndarray
    reference_values: np.ndarray
    reference_source: str  # "analytical" or "ngspice"

    # Error metrics
    max_error: float
    rms_error: float
    max_relative_error: float
    correlation: float

    # Thresholds used
    max_error_threshold: float
    rms_error_threshold: float

    # Execution times
    execution_time_pulsim: float
    execution_time_reference: float

    # Additional info
    notes: str = ""
    node_name: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.test_name} [{status}]\n"
            f"  Reference: {self.reference_source}\n"
            f"  Max Error: {self.max_error:.2e} (threshold: {self.max_error_threshold:.2e})\n"
            f"  RMS Error: {self.rms_error:.2e} (threshold: {self.rms_error_threshold:.2e})\n"
            f"  Max Relative Error: {self.max_relative_error:.2%}\n"
            f"  Correlation: {self.correlation:.6f}\n"
            f"  Pulsim Time: {self.execution_time_pulsim*1000:.2f} ms\n"
            f"  Reference Time: {self.execution_time_reference*1000:.2f} ms"
        )


@dataclass
class CircuitDefinition:
    """Definition of a test circuit."""
    name: str
    description: str
    level: ValidationLevel

    # Circuit builder for Pulsim (function that returns configured Circuit)
    pulsim_builder: Callable

    # SPICE netlist for NgSpice reference
    spice_netlist: str

    # Simulation parameters
    tstart: float = 0.0
    tstop: float = 1e-3
    dt: float = 1e-6

    # Nodes to compare (Pulsim node name -> SPICE node name)
    compare_nodes: Dict[str, str] = field(default_factory=dict)

    # Custom tolerances (overrides level defaults)
    max_error_tolerance: Optional[float] = None
    rms_error_tolerance: Optional[float] = None

    # Analytical solution (if available)
    # Function signature: analytical(time: np.ndarray, node_name: str) -> np.ndarray
    analytical_solution: Optional[Callable] = None

    # Circuit parameters for analytical solution
    circuit_params: Dict[str, float] = field(default_factory=dict)

    # Pulsim simulation options overrides
    pulsim_options: Dict[str, Any] = field(default_factory=dict)

    def get_tolerances(self) -> Tuple[float, float]:
        """Get tolerances for this circuit (custom or level default)."""
        defaults = DEFAULT_TOLERANCES[self.level]
        max_err = self.max_error_tolerance if self.max_error_tolerance is not None else defaults["max_error"]
        rms_err = self.rms_error_tolerance if self.rms_error_tolerance is not None else defaults["rms_error"]
        return max_err, rms_err


class ValidationTest:
    """Base class for validation tests."""

    def __init__(self, circuit_def: CircuitDefinition):
        self.circuit_def = circuit_def
        self._pulsim_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._reference_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._pulsim_time = 0.0
        self._reference_time = 0.0

    def run_pulsim(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run simulation in Pulsim.

        Returns:
            Dictionary mapping node names to (time, values) tuples.
        """
        import pulsim as sl

        # Build the circuit
        circuit = self.circuit_def.pulsim_builder()

        # Configure simulation options
        opts = sl.SimulationOptions()
        opts.tstart = self.circuit_def.tstart
        opts.tstop = self.circuit_def.tstop
        opts.dt = self.circuit_def.dt

        # Apply any custom options
        for key, value in self.circuit_def.pulsim_options.items():
            if hasattr(opts, key):
                setattr(opts, key, value)

        # Run simulation using Simulator class
        start_time = time.perf_counter()
        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        self._pulsim_time = time.perf_counter() - start_time

        if result.final_status != sl.SolverStatus.Success:
            raise RuntimeError(
                f"Pulsim simulation failed: {result.error_message}\n"
                f"Status: {result.final_status}"
            )

        # Extract results for comparison nodes
        self._pulsim_results = {}
        time_array = np.array(result.time)

        # Build signal name to data index mapping
        # data is a list of tuples: [(sig0_t0, sig1_t0, ...), (sig0_t1, sig1_t1, ...), ...]
        # We need to transpose to get each signal's values over time
        signal_names = result.signal_names
        data_matrix = np.array(result.data)  # Shape: (n_time_points, n_signals)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        for pulsim_node in self.circuit_def.compare_nodes.keys():
            if pulsim_node in signal_data:
                values = signal_data[pulsim_node]
            else:
                # Try to find partial match
                found = False
                for sig_name in signal_names:
                    if pulsim_node.lower() in sig_name.lower():
                        values = signal_data[sig_name]
                        found = True
                        break
                if not found:
                    raise KeyError(
                        f"Node/branch '{pulsim_node}' not found in Pulsim results.\n"
                        f"Available signals: {signal_names}"
                    )

            self._pulsim_results[pulsim_node] = (time_array, values)

        return self._pulsim_results

    def run_analytical(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate analytical solution.

        Returns:
            Dictionary mapping node names to (time, values) tuples.
        """
        if self.circuit_def.analytical_solution is None:
            raise ValueError(f"No analytical solution defined for {self.circuit_def.name}")

        # Generate time array
        time_array = np.arange(
            self.circuit_def.tstart,
            self.circuit_def.tstop + self.circuit_def.dt,
            self.circuit_def.dt
        )

        start_time = time.perf_counter()

        self._reference_results = {}
        for pulsim_node in self.circuit_def.compare_nodes.keys():
            values = self.circuit_def.analytical_solution(
                time_array,
                pulsim_node,
                self.circuit_def.circuit_params
            )
            self._reference_results[pulsim_node] = (time_array, values)

        self._reference_time = time.perf_counter() - start_time

        return self._reference_results

    def run_ngspice(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run simulation in NgSpice.

        Returns:
            Dictionary mapping node names to (time, values) tuples.
        """
        from .spice_runner import SpiceRunner

        runner = SpiceRunner()

        start_time = time.perf_counter()
        spice_results = runner.run_transient(
            self.circuit_def.spice_netlist,
            self.circuit_def.tstart,
            self.circuit_def.tstop,
            self.circuit_def.dt
        )
        self._reference_time = time.perf_counter() - start_time

        # Map SPICE node names to Pulsim node names
        self._reference_results = {}
        for pulsim_node, spice_node in self.circuit_def.compare_nodes.items():
            if spice_node.lower() in spice_results:
                time_arr, values = spice_results[spice_node.lower()]
                self._reference_results[pulsim_node] = (time_arr, values)
            else:
                available = list(spice_results.keys())
                raise KeyError(
                    f"SPICE node '{spice_node}' not found. Available: {available}"
                )

        return self._reference_results

    def validate(self, use_analytical: bool = True, use_ngspice: bool = False) -> List[ValidationResult]:
        """
        Execute complete validation.

        Args:
            use_analytical: Use analytical solution as reference (if available)
            use_ngspice: Use NgSpice as reference

        Returns:
            List of ValidationResult for each compared node.
        """
        from .comparator import ResultComparator

        # Run Pulsim
        pulsim_results = self.run_pulsim()

        # Get reference results
        reference_source = "none"
        if use_analytical and self.circuit_def.analytical_solution is not None:
            reference_results = self.run_analytical()
            reference_source = "analytical"
        elif use_ngspice:
            reference_results = self.run_ngspice()
            reference_source = "ngspice"
        else:
            raise ValueError("No reference source specified or available")

        # Compare results
        comparator = ResultComparator()
        max_tol_rel, rms_tol_rel = self.circuit_def.get_tolerances()

        results = []
        for node_name in self.circuit_def.compare_nodes.keys():
            pulsim_time, pulsim_values = pulsim_results[node_name]
            ref_time, ref_values = reference_results[node_name]

            # Compute metrics
            metrics = comparator.compare(
                pulsim_time, pulsim_values,
                ref_time, ref_values
            )

            # Scale tolerances by signal amplitude (relative to absolute)
            ref_amplitude = np.max(np.abs(ref_values))
            max_tol = max_tol_rel * ref_amplitude if ref_amplitude > 0 else max_tol_rel
            rms_tol = rms_tol_rel * ref_amplitude if ref_amplitude > 0 else rms_tol_rel

            passed = (metrics["max_error"] <= max_tol and
                     metrics["rms_error"] <= rms_tol)

            result = ValidationResult(
                test_name=f"{self.circuit_def.name}:{node_name}",
                passed=passed,
                pulsim_time=pulsim_time,
                pulsim_values=pulsim_values,
                reference_time=ref_time,
                reference_values=ref_values,
                reference_source=reference_source,
                max_error=metrics["max_error"],
                rms_error=metrics["rms_error"],
                max_relative_error=metrics["max_relative_error"],
                correlation=metrics["correlation"],
                max_error_threshold=max_tol,
                rms_error_threshold=rms_tol,
                execution_time_pulsim=self._pulsim_time,
                execution_time_reference=self._reference_time,
                node_name=node_name,
                notes=self.circuit_def.description,
            )
            results.append(result)

        return results


def run_validation_suite(
    circuit_defs: List[CircuitDefinition],
    use_analytical: bool = True,
    use_ngspice: bool = False,
    stop_on_failure: bool = False,
) -> List[ValidationResult]:
    """
    Run a suite of validation tests.

    Args:
        circuit_defs: List of circuit definitions to test
        use_analytical: Use analytical solutions when available
        use_ngspice: Use NgSpice as reference
        stop_on_failure: Stop on first failure

    Returns:
        List of all ValidationResults
    """
    all_results = []

    for circuit_def in circuit_defs:
        test = ValidationTest(circuit_def)

        try:
            # Prefer analytical if available, otherwise NgSpice
            if use_analytical and circuit_def.analytical_solution is not None:
                results = test.validate(use_analytical=True, use_ngspice=False)
            elif use_ngspice:
                results = test.validate(use_analytical=False, use_ngspice=True)
            else:
                print(f"Skipping {circuit_def.name}: no reference available")
                continue

            all_results.extend(results)

            for r in results:
                print(r)
                print()

            if stop_on_failure and any(not r.passed for r in results):
                print(f"Stopping due to failure in {circuit_def.name}")
                break

        except Exception as e:
            print(f"ERROR in {circuit_def.name}: {e}")
            if stop_on_failure:
                raise

    return all_results
