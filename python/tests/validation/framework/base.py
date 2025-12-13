"""Base classes for validation framework."""

from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum
import numpy as np
import time


class ValidationLevel(Enum):
    """Validation complexity levels."""
    LINEAR = 1         # Linear circuits with analytical solution
    DC_ANALYSIS = 2    # DC analysis
    NONLINEAR = 3      # Nonlinear components
    CONVERTER = 4      # Power converters


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    pulsim_times: np.ndarray
    pulsim_values: np.ndarray
    reference_times: np.ndarray
    reference_values: np.ndarray
    max_error: float
    rms_error: float
    max_relative_error: float
    tolerance: float
    execution_time_ms: float
    notes: str = ""

    def summary(self) -> str:
        """Return a summary string of the validation result."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.test_name}: {status}\n"
            f"  Max Error: {self.max_error:.6e}\n"
            f"  RMS Error: {self.rms_error:.6e}\n"
            f"  Max Rel Error: {self.max_relative_error*100:.4f}%\n"
            f"  Tolerance: {self.tolerance*100:.2f}%\n"
            f"  Exec Time: {self.execution_time_ms:.2f} ms"
        )


@dataclass
class CircuitDefinition:
    """Definition of a test circuit."""
    name: str
    description: str
    level: ValidationLevel
    build_circuit: Callable[[], Any]  # Returns pulsim.Circuit
    analytical_solution: Optional[Callable[[np.ndarray], np.ndarray]] = None
    t_start: float = 0.0
    t_stop: float = 1e-3
    dt: float = 1e-6
    node_index: int = 1  # Node index to compare (0 is usually internal)
    tolerance: float = 0.01  # 1% default
    dc_tolerance: float = 0.0001  # 0.01% for DC
    spice_netlist: Optional[str] = None
    use_zero_ic: bool = False  # If True, use IC=0 instead of DC operating point
    custom_ic: Optional[Callable[[Any], np.ndarray]] = None  # Custom IC function(circuit) -> x0


class ValidationTest:
    """Base class for validation tests."""

    def __init__(self, circuit_def: CircuitDefinition):
        self.circuit_def = circuit_def
        self._import_pulsim()

    def _import_pulsim(self):
        """Import pulsim module."""
        import pulsim as ps
        self.ps = ps

    def run_pulsim_dc(self):
        """Run DC analysis in Pulsim."""
        circuit = self.circuit_def.build_circuit()
        return self.ps.dc_operating_point(circuit)

    def run_pulsim_transient(self) -> tuple:
        """Run transient simulation in Pulsim.

        Returns:
            tuple: (times, states, success, message)
        """
        circuit = self.circuit_def.build_circuit()

        # Determine initial conditions
        if self.circuit_def.custom_ic is not None:
            # Use custom IC function
            x0 = self.circuit_def.custom_ic(circuit)
        elif self.circuit_def.use_zero_ic:
            # Use zero initial conditions (for step response tests)
            x0 = np.zeros(circuit.system_size())
            # Set voltage source nodes to their DC values
            dc_result = self.ps.dc_operating_point(circuit)
            if dc_result.success:
                # Copy only the voltage source node values (usually node 0)
                # Keep capacitor/inductor nodes at 0
                x0[0] = dc_result.newton_result.solution[0]
        else:
            # DC initial condition (steady state)
            dc_result = self.ps.dc_operating_point(circuit)
            if not dc_result.success:
                raise RuntimeError(f"DC failed: {dc_result.message}")
            x0 = dc_result.newton_result.solution

        # Transient
        times, states, success, msg = self.ps.run_transient(
            circuit,
            self.circuit_def.t_start,
            self.circuit_def.t_stop,
            self.circuit_def.dt,
            x0
        )

        if not success:
            raise RuntimeError(f"Transient failed: {msg}")

        return np.array(times), states, success, msg

    def run_analytical(self, times: np.ndarray) -> Optional[np.ndarray]:
        """Calculate analytical solution."""
        if self.circuit_def.analytical_solution:
            return self.circuit_def.analytical_solution(times)
        return None

    def validate_transient(self) -> ValidationResult:
        """Validate transient simulation against analytical solution."""
        start = time.perf_counter()
        times, states, success, msg = self.run_pulsim_transient()
        exec_time = (time.perf_counter() - start) * 1000

        # Extract node of interest
        node_idx = self.circuit_def.node_index
        pulsim_values = np.array([s[node_idx] for s in states])

        # Analytical solution
        ref_values = self.run_analytical(times)
        if ref_values is None:
            raise ValueError("No analytical solution defined for this circuit")

        result = self._compare(times, pulsim_values, times, ref_values)
        result.execution_time_ms = exec_time
        return result

    def validate_dc(self, expected_values: dict) -> ValidationResult:
        """Validate DC analysis against expected values.

        Args:
            expected_values: Dict mapping node index to expected voltage
        """
        start = time.perf_counter()
        dc_result = self.run_pulsim_dc()
        exec_time = (time.perf_counter() - start) * 1000

        if not dc_result.success:
            return ValidationResult(
                test_name=self.circuit_def.name,
                passed=False,
                pulsim_times=np.array([0]),
                pulsim_values=np.array([0]),
                reference_times=np.array([0]),
                reference_values=np.array([0]),
                max_error=float('inf'),
                rms_error=float('inf'),
                max_relative_error=float('inf'),
                tolerance=self.circuit_def.dc_tolerance,
                execution_time_ms=exec_time,
                notes=f"DC analysis failed: {dc_result.message}"
            )

        solution = dc_result.newton_result.solution

        errors = []
        for node_idx, expected in expected_values.items():
            if node_idx < len(solution):
                actual = solution[node_idx]
                error = abs(actual - expected)
                rel_error = error / abs(expected) if expected != 0 else error
                errors.append((error, rel_error))

        max_error = max(e[0] for e in errors) if errors else 0
        max_rel_error = max(e[1] for e in errors) if errors else 0
        rms_error = np.sqrt(np.mean([e[0]**2 for e in errors])) if errors else 0

        passed = max_rel_error <= self.circuit_def.dc_tolerance

        return ValidationResult(
            test_name=self.circuit_def.name,
            passed=passed,
            pulsim_times=np.array([0]),
            pulsim_values=np.array(list(solution)),
            reference_times=np.array([0]),
            reference_values=np.array([expected_values.get(i, 0) for i in range(len(solution))]),
            max_error=max_error,
            rms_error=rms_error,
            max_relative_error=max_rel_error,
            tolerance=self.circuit_def.dc_tolerance,
            execution_time_ms=exec_time
        )

    def _compare(self, pulsim_times, pulsim_values, ref_times, ref_values) -> ValidationResult:
        """Compare Pulsim results vs reference."""
        from .comparator import compare_results

        return compare_results(
            test_name=self.circuit_def.name,
            pulsim_times=pulsim_times,
            pulsim_values=pulsim_values,
            ref_times=ref_times,
            ref_values=ref_values,
            tolerance=self.circuit_def.tolerance
        )
