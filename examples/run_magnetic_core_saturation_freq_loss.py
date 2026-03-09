"""Run magnetic-core MVP example and inspect core-loss waveform."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import pulsim as ps


def main() -> None:
    netlist_path = Path(__file__).with_name("magnetic_core_saturation_freq_loss.yaml")
    parser = ps.YamlParser()
    circuit, opts = parser.load_string(netlist_path.read_text(encoding="utf-8"))
    if parser.errors:
        raise RuntimeError(f"YAML errors: {parser.errors}")

    opts.newton_options.num_nodes = int(circuit.num_nodes())
    opts.newton_options.num_branches = int(circuit.num_branches())
    result = ps.Simulator(circuit, opts).run_transient(circuit.initial_state())
    if not result.success:
        raise RuntimeError(f"Simulation failed: {result.message}")

    channel = "Lsat.core_loss"
    if channel not in result.virtual_channels:
        raise RuntimeError(f"Missing virtual channel: {channel}")

    t = [float(v) for v in result.time]
    p_core = [float(v) for v in result.virtual_channels[channel]]
    avg_core = sum(p_core) / len(p_core)
    peak_core = max(p_core)
    print(f"{channel}: avg={avg_core:.6e} W peak={peak_core:.6e} W")
    loss_rows = {row.device_name: row for row in result.loss_summary.device_losses}
    if "Lsat.core" in loss_rows:
        row = loss_rows["Lsat.core"]
        print(
            "Lsat.core summary: "
            f"avg={float(row.average_power):.6e} W "
            f"energy={float(row.total_energy):.6e} J"
        )

    plt.figure(figsize=(10, 4))
    plt.plot(t, p_core, linewidth=1.2, label=channel)
    plt.xlabel("Time [s]")
    plt.ylabel("Core Loss [W]")
    plt.title("Magnetic Core Loss Telemetry (MVP)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
