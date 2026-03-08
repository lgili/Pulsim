#!/usr/bin/env python3
"""Run averaged buck MVP example and print key channels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pulsim as ps


def main() -> int:
    parser = argparse.ArgumentParser(description="Run averaged buck MVP example")
    parser.add_argument(
        "--netlist",
        type=Path,
        default=Path("examples/10_buck_averaged_mvp_backend.yaml"),
        help="Path to averaged buck YAML netlist",
    )
    args = parser.parse_args()

    yaml_parser = ps.YamlParser(ps.YamlParserOptions())
    circuit, options = yaml_parser.load(str(args.netlist))
    if yaml_parser.errors:
        raise SystemExit("; ".join(str(item) for item in yaml_parser.errors))

    options.newton_options.num_nodes = int(circuit.num_nodes())
    options.newton_options.num_branches = int(circuit.num_branches())

    result = ps.Simulator(circuit, options).run_transient(circuit.initial_state())

    print("success:", result.success)
    print("diagnostic:", result.diagnostic.name)
    print("message:", result.message)
    print("steps:", result.total_steps)

    for channel in ("Iavg(L1)", "Vavg(out)", "Davg"):
        series = list(result.virtual_channels.get(channel, []))
        if not series:
            print(channel, "missing")
            continue
        print(channel, "samples:", len(series), "final:", float(series[-1]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
