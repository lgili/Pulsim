"""
SPICE simulation runner with PySpice and subprocess fallback.

This module provides an interface to run NgSpice simulations, using PySpice
when available and falling back to direct subprocess calls otherwise.
"""

import subprocess
import tempfile
import os
import re
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class SpiceRunner:
    """
    Interface for running NgSpice simulations.

    Attempts to use PySpice first, falls back to subprocess if unavailable.
    """

    def __init__(self, ngspice_path: Optional[str] = None):
        """
        Initialize the SPICE runner.

        Args:
            ngspice_path: Path to ngspice executable (auto-detected if None)
        """
        self._ngspice_path = ngspice_path
        self._pyspice_available = self._check_pyspice()
        self._ngspice_available = self._check_ngspice()

        if not self._pyspice_available and not self._ngspice_available:
            raise RuntimeError(
                "Neither PySpice nor ngspice executable found. "
                "Install PySpice (pip install pyspice) or ngspice "
                "(brew install ngspice / apt install ngspice)"
            )

    def _check_pyspice(self) -> bool:
        """Check if PySpice is available and working."""
        try:
            import PySpice
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared
            # Try to instantiate NgSpiceShared to verify it works
            return True
        except Exception:
            return False

    def _check_ngspice(self) -> bool:
        """Check if ngspice executable is available."""
        if self._ngspice_path:
            return os.path.isfile(self._ngspice_path)

        # Try common locations
        for path in ["/usr/bin/ngspice", "/usr/local/bin/ngspice",
                     "/opt/homebrew/bin/ngspice"]:
            if os.path.isfile(path):
                self._ngspice_path = path
                return True

        # Try to find in PATH
        try:
            result = subprocess.run(
                ["which", "ngspice"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self._ngspice_path = result.stdout.strip()
                return True
        except Exception:
            pass

        return False

    @property
    def backend(self) -> str:
        """Return which backend is being used."""
        if self._pyspice_available:
            return "pyspice"
        elif self._ngspice_available:
            return "subprocess"
        return "none"

    def run_transient(
        self,
        netlist: str,
        tstart: float,
        tstop: float,
        dt: float,
        nodes: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run a transient simulation.

        Args:
            netlist: SPICE netlist string
            tstart: Start time (seconds)
            tstop: Stop time (seconds)
            dt: Time step (seconds)
            nodes: List of nodes to extract (None = all)

        Returns:
            Dictionary mapping node names to (time, values) tuples.
            Node names are lowercase.
        """
        if self._pyspice_available:
            return self._run_pyspice(netlist, tstart, tstop, dt, nodes)
        else:
            return self._run_subprocess(netlist, tstart, tstop, dt, nodes)

    def _run_pyspice(
        self,
        netlist: str,
        tstart: float,
        tstop: float,
        dt: float,
        nodes: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run simulation using PySpice."""
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        from PySpice.Spice.Parser import SpiceParser

        # Add transient analysis if not present
        if ".tran" not in netlist.lower():
            netlist = netlist.strip() + f"\n.tran {dt} {tstop}\n"

        if ".end" not in netlist.lower():
            netlist = netlist.strip() + "\n.end\n"

        # Parse the netlist
        parser = SpiceParser(source=netlist)
        circuit = parser.build_circuit()

        # Create simulator
        simulator = circuit.simulator(
            temperature=25,
            nominal_temperature=25
        )

        # Run transient analysis
        analysis = simulator.transient(
            step_time=dt,
            end_time=tstop,
            start_time=tstart,
            use_initial_conditions=True
        )

        # Extract results
        results = {}
        time_array = np.array(analysis.time)

        # Get all node voltages
        for node_name in analysis.nodes.keys():
            if nodes is None or node_name.lower() in [n.lower() for n in nodes]:
                values = np.array(analysis[node_name])
                results[node_name.lower()] = (time_array, values)

        # Get branch currents
        for branch_name in analysis.branches.keys():
            if nodes is None or branch_name.lower() in [n.lower() for n in nodes]:
                values = np.array(analysis[branch_name])
                results[branch_name.lower()] = (time_array, values)

        return results

    def _run_subprocess(
        self,
        netlist: str,
        tstart: float,
        tstop: float,
        dt: float,
        nodes: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run simulation using ngspice subprocess."""

        # Add transient analysis and save commands
        if ".tran" not in netlist.lower():
            netlist = netlist.strip() + f"\n.tran {dt} {tstop}\n"

        # Add .control section for batch mode
        control_section = """
.control
run
wrdata {output_file} all
quit
.endc
"""
        # Remove any existing .control/.endc blocks
        netlist = re.sub(r'\.control.*?\.endc', '', netlist, flags=re.DOTALL | re.IGNORECASE)

        if ".end" in netlist.lower():
            netlist = re.sub(r'\.end', '', netlist, flags=re.IGNORECASE)

        with tempfile.TemporaryDirectory() as tmpdir:
            netlist_path = os.path.join(tmpdir, "circuit.cir")
            output_path = os.path.join(tmpdir, "output.txt")

            # Add control section with output file
            full_netlist = netlist.strip() + "\n" + control_section.format(
                output_file=output_path
            ) + "\n.end\n"

            # Write netlist
            with open(netlist_path, "w") as f:
                f.write(full_netlist)

            # Run ngspice
            result = subprocess.run(
                [self._ngspice_path, "-b", netlist_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"NgSpice failed with code {result.returncode}:\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            # Parse output
            return self._parse_wrdata_output(output_path, nodes)

    def _parse_wrdata_output(
        self,
        output_path: str,
        nodes: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Parse ngspice wrdata output file."""

        if not os.path.exists(output_path):
            raise RuntimeError(f"NgSpice output file not found: {output_path}")

        with open(output_path, "r") as f:
            lines = f.readlines()

        if not lines:
            raise RuntimeError("NgSpice output file is empty")

        # wrdata format: each line is "time value1 value2 ..."
        # First line might be headers or data
        # Try to detect format

        data = []
        headers = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # Check if this is a header line
            if not parts[0].replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
                headers = parts
                continue

            try:
                values = [float(x) for x in parts]
                data.append(values)
            except ValueError:
                continue

        if not data:
            raise RuntimeError("No data found in NgSpice output")

        data = np.array(data)
        time_array = data[:, 0]

        results = {}

        # Generate headers if not provided
        if headers is None:
            headers = ["time"] + [f"v({i})" for i in range(data.shape[1] - 1)]

        for i, header in enumerate(headers[1:], start=1):
            if i < data.shape[1]:
                node_name = header.lower()
                if nodes is None or node_name in [n.lower() for n in nodes]:
                    results[node_name] = (time_array, data[:, i])

        return results

    def run_dc(
        self,
        netlist: str,
        source_name: str,
        start_value: float,
        stop_value: float,
        step: float,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run a DC sweep simulation.

        Args:
            netlist: SPICE netlist string
            source_name: Name of source to sweep
            start_value: Start value
            stop_value: Stop value
            step: Step size

        Returns:
            Dictionary mapping node names to (sweep_values, node_values) tuples.
        """
        if self._pyspice_available:
            return self._run_dc_pyspice(netlist, source_name, start_value, stop_value, step)
        else:
            return self._run_dc_subprocess(netlist, source_name, start_value, stop_value, step)

    def _run_dc_pyspice(
        self,
        netlist: str,
        source_name: str,
        start_value: float,
        stop_value: float,
        step: float,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run DC sweep using PySpice."""
        from PySpice.Spice.Parser import SpiceParser

        if ".end" not in netlist.lower():
            netlist = netlist.strip() + "\n.end\n"

        parser = SpiceParser(source=netlist)
        circuit = parser.build_circuit()

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)

        analysis = simulator.dc(**{source_name: slice(start_value, stop_value, step)})

        results = {}
        sweep_values = np.array(analysis[source_name])

        for node_name in analysis.nodes.keys():
            values = np.array(analysis[node_name])
            results[node_name.lower()] = (sweep_values, values)

        return results

    def _run_dc_subprocess(
        self,
        netlist: str,
        source_name: str,
        start_value: float,
        stop_value: float,
        step: float,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run DC sweep using subprocess."""

        dc_line = f".dc {source_name} {start_value} {stop_value} {step}\n"

        control_section = """
.control
run
wrdata {output_file} all
quit
.endc
"""
        netlist = re.sub(r'\.control.*?\.endc', '', netlist, flags=re.DOTALL | re.IGNORECASE)

        if ".end" in netlist.lower():
            netlist = re.sub(r'\.end', '', netlist, flags=re.IGNORECASE)

        with tempfile.TemporaryDirectory() as tmpdir:
            netlist_path = os.path.join(tmpdir, "circuit.cir")
            output_path = os.path.join(tmpdir, "output.txt")

            full_netlist = netlist.strip() + "\n" + dc_line + control_section.format(
                output_file=output_path
            ) + "\n.end\n"

            with open(netlist_path, "w") as f:
                f.write(full_netlist)

            result = subprocess.run(
                [self._ngspice_path, "-b", netlist_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"NgSpice DC sweep failed:\n{result.stdout}\n{result.stderr}"
                )

            return self._parse_wrdata_output(output_path)


def check_spice_available() -> Tuple[bool, str]:
    """
    Check if SPICE simulation is available.

    Returns:
        Tuple of (available, backend_name)
    """
    try:
        runner = SpiceRunner()
        return True, runner.backend
    except RuntimeError:
        return False, "none"
