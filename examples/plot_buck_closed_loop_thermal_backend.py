#!/usr/bin/env python3
"""Run closed-loop buck electrothermal example and plot switch thermal trace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def import_pulsim():
    repo_root = Path(__file__).resolve().parents[1]
    build_python = repo_root / "build" / "python"
    if build_python.exists():
        build_python_str = str(build_python)
        if build_python_str not in sys.path:
            sys.path.insert(0, build_python_str)

    import pulsim as ps  # type: ignore

    return ps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot closed-loop buck channels including M1 thermal trace"
    )
    parser.add_argument(
        "--netlist",
        type=Path,
        default=Path("examples/09_buck_closed_loop_loss_thermal_validation_backend.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/out_buck_closed_loop_thermal_plot.png"),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show matplotlib window in addition to saving figure",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"matplotlib is required to plot this example: {exc}") from exc

    ps = import_pulsim()

    parser_opts = ps.YamlParserOptions()
    parser_opts.strict = False
    yaml_parser = ps.YamlParser(parser_opts)
    circuit, options = yaml_parser.load(str(args.netlist.resolve()))
    if yaml_parser.errors:
        raise SystemExit("YAML parser errors: " + "; ".join(yaml_parser.errors))

    options.newton_options.num_nodes = int(circuit.num_nodes())
    options.newton_options.num_branches = int(circuit.num_branches())

    result = ps.Simulator(circuit, options).run_transient(circuit.initial_state())
    if not result.success:
        raise SystemExit(f"Simulation failed: {result.message}")

    required = ("Xout", "Xsw", "PI1", "PWM1.duty", "T(M1)")
    missing = [name for name in required if name not in result.virtual_channels]
    if missing:
        channels = ", ".join(sorted(result.virtual_channels.keys()))
        raise SystemExit(
            f"Missing required channels: {missing}. Available channels: [{channels}]"
        )

    t_ms = [float(t) * 1e3 for t in result.time]
    vout = [float(v) for v in result.virtual_channels["Xout"]]
    xsw = [float(v) for v in result.virtual_channels["Xsw"]]
    pi = [float(v) for v in result.virtual_channels["PI1"]]
    duty = [float(v) for v in result.virtual_channels["PWM1.duty"]]
    m1_temp = [float(v) for v in result.virtual_channels["T(M1)"]]

    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    ax[0, 0].plot(t_ms, vout, linewidth=1.4, label="Vout")
    ax[0, 0].axhline(6.0, color="tab:gray", linestyle="--", linewidth=1.0, label="Vref")
    ax[0, 0].set_title("Closed-loop output")
    ax[0, 0].set_ylabel("V")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(loc="best")

    ax[0, 1].plot(t_ms, pi, linewidth=1.2, label="PI output")
    ax[0, 1].plot(t_ms, duty, linewidth=1.2, linestyle="--", label="PWM duty")
    ax[0, 1].set_title("Control path")
    ax[0, 1].set_ylabel("pu")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(loc="best")

    ax[1, 0].plot(t_ms, xsw, linewidth=1.2, color="tab:orange", label="Vsw")
    ax[1, 0].set_title("Switch node voltage")
    ax[1, 0].set_ylabel("V")
    ax[1, 0].set_xlabel("time [ms]")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].legend(loc="best")

    ax[1, 1].plot(t_ms, m1_temp, linewidth=1.6, color="tab:red", label="T(M1)")
    ax[1, 1].axhline(
        float(result.thermal_summary.ambient),
        color="tab:gray",
        linestyle="--",
        linewidth=1.0,
        label="ambient",
    )
    ax[1, 1].set_title("Switch thermal signal")
    ax[1, 1].set_ylabel("degC")
    ax[1, 1].set_xlabel("time [ms]")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(loc="best")

    fig.suptitle(
        "Buck closed-loop electrothermal validation "
        f"(M1 final={m1_temp[-1]:.3f} degC, max={max(m1_temp):.3f} degC)"
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved plot: {args.output.resolve()}")
    print(
        "summary: "
        f"vout_final={vout[-1]:.4f} V, "
        f"duty_final={duty[-1]:.4f}, "
        f"m1_temp_final={m1_temp[-1]:.4f} degC, "
        f"m1_temp_peak={max(m1_temp):.4f} degC"
    )

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
