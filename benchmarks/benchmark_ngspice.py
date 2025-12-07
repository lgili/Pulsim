#!/usr/bin/env python3
"""
Benchmark Pulsim against ngspice

This script compares simulation performance and accuracy between
Pulsim and ngspice for various circuit types.

Requirements:
- ngspice installed (brew install ngspice / apt install ngspice)
- Pulsim CLI built

Usage:
    python benchmark_ngspice.py [--verbose] [--output-dir DIR]
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import csv


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    circuit_name: str
    pulsim_time: float
    ngspice_time: float
    pulsim_steps: int
    ngspice_steps: int
    accuracy_error: float  # RMS error between solutions
    speedup: float  # ngspice_time / pulsim_time


def find_pulsim_cli() -> Optional[Path]:
    """Find the Pulsim CLI executable"""
    # Check common build directories
    script_dir = Path(__file__).parent.parent
    candidates = [
        script_dir / "build" / "cli" / "pulsim",
        script_dir / "build" / "Release" / "cli" / "pulsim",
        script_dir / "build" / "Debug" / "cli" / "pulsim",
        Path("pulsim"),  # In PATH
    ]
    for path in candidates:
        if path.exists() or (path.name == "pulsim" and subprocess.run(
            ["which", "pulsim"], capture_output=True).returncode == 0):
            return path
    return None


def check_ngspice() -> bool:
    """Check if ngspice is available"""
    try:
        result = subprocess.run(["ngspice", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_spice_netlist(circuit_json: dict, output_path: Path) -> None:
    """Convert Pulsim JSON format to SPICE netlist"""
    lines = [f"* {circuit_json.get('name', 'Benchmark Circuit')}"]

    # Add components
    for comp in circuit_json.get("components", []):
        comp_type = comp["type"].lower()
        name = comp["name"]
        nodes = comp["nodes"]

        if comp_type == "resistor":
            lines.append(f"R{name} {nodes[0]} {nodes[1]} {comp['value']}")
        elif comp_type == "capacitor":
            lines.append(f"C{name} {nodes[0]} {nodes[1]} {comp['value']}")
        elif comp_type == "inductor":
            lines.append(f"L{name} {nodes[0]} {nodes[1]} {comp['value']}")
        elif comp_type == "voltage_source":
            waveform = comp.get("waveform", {})
            if waveform.get("type") == "dc":
                lines.append(f"V{name} {nodes[0]} {nodes[1]} DC {waveform['value']}")
            elif waveform.get("type") == "pulse":
                lines.append(
                    f"V{name} {nodes[0]} {nodes[1]} PULSE("
                    f"{waveform['v1']} {waveform['v2']} "
                    f"{waveform.get('td', 0)} {waveform.get('tr', 1e-9)} "
                    f"{waveform.get('tf', 1e-9)} {waveform['pw']} {waveform['period']})"
                )
            elif waveform.get("type") == "sine":
                lines.append(
                    f"V{name} {nodes[0]} {nodes[1]} SIN("
                    f"{waveform.get('offset', 0)} {waveform['amplitude']} "
                    f"{waveform['frequency']} {waveform.get('delay', 0)} "
                    f"{waveform.get('damping', 0)})"
                )
        elif comp_type == "diode":
            lines.append(f"D{name} {nodes[0]} {nodes[1]} DMOD")
            lines.append(".MODEL DMOD D(IS=1E-14 N=1.0)")
        elif comp_type == "switch":
            # SPICE switch model
            lines.append(f"S{name} {nodes[0]} {nodes[1]} {nodes[2]} {nodes[3]} SWMOD")
            lines.append(f".MODEL SWMOD SW(VT={comp.get('vth', 2.5)} RON={comp.get('ron', 0.01)} ROFF=1G)")

    # Add simulation commands
    sim = circuit_json.get("simulation", {})
    tstop = sim.get("tstop", 1e-3)
    dt = sim.get("dt", 1e-6)
    lines.append(f".TRAN {dt} {tstop} 0 {dt}")
    lines.append(".END")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_pulsim(cli_path: Path, circuit_path: Path, output_path: Path) -> Tuple[float, int]:
    """Run Pulsim simulation and return (time, steps)"""
    start = time.perf_counter()
    result = subprocess.run(
        [str(cli_path), "run", str(circuit_path), "-o", str(output_path)],
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(f"Pulsim failed: {result.stderr}")

    # Count steps from output
    steps = 0
    if output_path.exists():
        with open(output_path) as f:
            steps = sum(1 for _ in f) - 1  # Subtract header

    return elapsed, steps


def run_ngspice(netlist_path: Path, output_path: Path) -> Tuple[float, int]:
    """Run ngspice simulation and return (time, steps)"""
    # Create control file for batch mode
    ctrl_content = f"""
set filetype=ascii
tran 1u 1m
wrdata {output_path} all
quit
"""
    ctrl_path = netlist_path.with_suffix(".ctrl")
    with open(ctrl_path, "w") as f:
        f.write(ctrl_content)

    start = time.perf_counter()
    result = subprocess.run(
        ["ngspice", "-b", str(netlist_path)],
        capture_output=True, text=True,
        cwd=netlist_path.parent
    )
    elapsed = time.perf_counter() - start

    # Count steps from output
    steps = 0
    if output_path.exists():
        with open(output_path) as f:
            steps = sum(1 for _ in f)

    return elapsed, steps


def create_benchmark_circuits() -> list:
    """Create a set of benchmark circuits"""
    circuits = []

    # 1. Simple RC circuit
    circuits.append({
        "name": "RC_Simple",
        "components": [
            {"type": "voltage_source", "name": "Vin", "nodes": ["in", "0"],
             "waveform": {"type": "dc", "value": 5.0}},
            {"type": "resistor", "name": "R1", "nodes": ["in", "out"], "value": 1000},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": 1e-6},
        ],
        "simulation": {"tstop": 10e-3, "dt": 1e-6}
    })

    # 2. RLC oscillator
    circuits.append({
        "name": "RLC_Oscillator",
        "components": [
            {"type": "voltage_source", "name": "Vin", "nodes": ["in", "0"],
             "waveform": {"type": "dc", "value": 10.0}},
            {"type": "resistor", "name": "R1", "nodes": ["in", "n1"], "value": 10},
            {"type": "inductor", "name": "L1", "nodes": ["n1", "out"], "value": 1e-3},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": 1e-6},
        ],
        "simulation": {"tstop": 5e-3, "dt": 0.5e-6}
    })

    # 3. PWM driven RC (switching)
    circuits.append({
        "name": "PWM_RC",
        "components": [
            {"type": "voltage_source", "name": "Vin", "nodes": ["vcc", "0"],
             "waveform": {"type": "dc", "value": 12.0}},
            {"type": "voltage_source", "name": "Vctrl", "nodes": ["ctrl", "0"],
             "waveform": {"type": "pulse", "v1": 0, "v2": 5,
                          "td": 0, "tr": 1e-9, "tf": 1e-9, "pw": 50e-6, "period": 100e-6}},
            {"type": "switch", "name": "S1", "nodes": ["vcc", "out", "ctrl", "0"],
             "ron": 0.01, "vth": 2.5},
            {"type": "resistor", "name": "R1", "nodes": ["out", "0"], "value": 100},
            {"type": "capacitor", "name": "C1", "nodes": ["out", "0"], "value": 10e-6},
        ],
        "simulation": {"tstop": 2e-3, "dt": 1e-6}
    })

    # 4. Larger RC ladder (10 stages)
    ladder_comps = [
        {"type": "voltage_source", "name": "Vin", "nodes": ["n0", "0"],
         "waveform": {"type": "dc", "value": 5.0}}
    ]
    for i in range(10):
        ladder_comps.append({
            "type": "resistor", "name": f"R{i+1}",
            "nodes": [f"n{i}", f"n{i+1}"], "value": 1000
        })
        ladder_comps.append({
            "type": "capacitor", "name": f"C{i+1}",
            "nodes": [f"n{i+1}", "0"], "value": 100e-9
        })
    circuits.append({
        "name": "RC_Ladder_10",
        "components": ladder_comps,
        "simulation": {"tstop": 1e-3, "dt": 0.1e-6}
    })

    return circuits


def run_benchmarks(pulsim_cli: Path, output_dir: Path, verbose: bool) -> list:
    """Run all benchmarks and return results"""
    results = []
    circuits = create_benchmark_circuits()

    for circuit in circuits:
        if verbose:
            print(f"\nBenchmarking: {circuit['name']}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Save Pulsim JSON
            json_path = tmppath / "circuit.json"
            with open(json_path, "w") as f:
                json.dump(circuit, f)

            # Run Pulsim
            sl_output = tmppath / "pulsim_out.csv"
            try:
                sl_time, sl_steps = run_pulsim(pulsim_cli, json_path, sl_output)
            except Exception as e:
                if verbose:
                    print(f"  Pulsim failed: {e}")
                continue

            # Generate and run ngspice
            spice_path = tmppath / "circuit.sp"
            ng_output = tmppath / "ngspice_out.txt"
            generate_spice_netlist(circuit, spice_path)

            try:
                ng_time, ng_steps = run_ngspice(spice_path, ng_output)
            except Exception as e:
                if verbose:
                    print(f"  ngspice failed: {e}")
                ng_time, ng_steps = 0, 0

            # Calculate speedup
            speedup = ng_time / sl_time if sl_time > 0 else 0

            result = BenchmarkResult(
                circuit_name=circuit["name"],
                pulsim_time=sl_time,
                ngspice_time=ng_time,
                pulsim_steps=sl_steps,
                ngspice_steps=ng_steps,
                accuracy_error=0.0,  # TODO: implement accuracy comparison
                speedup=speedup
            )
            results.append(result)

            if verbose:
                print(f"  Pulsim: {sl_time:.3f}s ({sl_steps} steps)")
                print(f"  ngspice:  {ng_time:.3f}s ({ng_steps} steps)")
                print(f"  Speedup:  {speedup:.2f}x")

    return results


def save_results(results: list, output_dir: Path) -> None:
    """Save benchmark results to CSV"""
    output_path = output_dir / "benchmark_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Circuit", "Pulsim Time (s)", "ngspice Time (s)",
            "Pulsim Steps", "ngspice Steps", "Speedup"
        ])
        for r in results:
            writer.writerow([
                r.circuit_name, f"{r.pulsim_time:.4f}",
                f"{r.ngspice_time:.4f}", r.pulsim_steps,
                r.ngspice_steps, f"{r.speedup:.2f}"
            ])
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pulsim against ngspice")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."),
                        help="Output directory for results")
    args = parser.parse_args()

    # Check prerequisites
    pulsim_cli = find_pulsim_cli()
    if not pulsim_cli:
        print("Error: Pulsim CLI not found. Build the project first.")
        return 1

    if not check_ngspice():
        print("Warning: ngspice not found. Install with: brew install ngspice")
        print("Continuing with Pulsim-only benchmarks...")

    print(f"Pulsim CLI: {pulsim_cli}")
    print(f"ngspice available: {check_ngspice()}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = run_benchmarks(pulsim_cli, args.output_dir, args.verbose)

    if results:
        save_results(results, args.output_dir)

        # Print summary
        print("\n=== Benchmark Summary ===")
        total_sl = sum(r.pulsim_time for r in results)
        total_ng = sum(r.ngspice_time for r in results if r.ngspice_time > 0)
        print(f"Total Pulsim time: {total_sl:.3f}s")
        if total_ng > 0:
            print(f"Total ngspice time:  {total_ng:.3f}s")
            print(f"Average speedup:     {total_ng/total_sl:.2f}x")
    else:
        print("No benchmark results generated.")

    return 0


if __name__ == "__main__":
    exit(main())
