# Pulsim Benchmark Suite

This directory holds the simulation benchmarks Pulsim uses for
validation, regression, and SPICE-parity comparison.

## Layout

```text
benchmarks/
├── README.md                    # this file
├── REORG_PLAN.md                # historical: phases of the May-2026 cleanup
├── circuits/                    # YAML netlists, grouped by topology family
│   ├── linear/                  #   pure R/L/C, dividers, bridges, IC discharge
│   ├── switching/               #   buck/boost/flyback/forward/vcswitch/mosfet
│   ├── diodes/                  #   rectifiers, clamps, clippers, voltage doubler
│   ├── magnetics/               #   transformer, coupled/saturating inductor
│   ├── motors/                  #   dc, bldc, pmsm, induction
│   ├── thermal/                 #   electrothermal coupling tests
│   ├── stress/                  #   stiff RC/RLC, periodic, high-Q ringdown
│   ├── three_phase/             #   3φ sources, VSI, grid, PLL, vector control
│   ├── resonance/               #   LC tank, driven RLC, Bode-style sweeps
│   ├── closed_loop/             #   cl_buck_*, cl_boost_pi (controlled converters)
│   └── NOTES_known_issues.md    #   solver bugs surfaced by validation
├── spice/                       # SPICE backend netlists (paired with circuits/)
│   ├── ngspice/                 #   .cir files for ngspice
│   └── ltspice/                 #   .cir files for LTspice (subset)
├── baselines/                   # reference CSV waveforms (flat)
├── manifests/                   # bench-suite configuration
│   ├── benchmarks.yaml          #   primary benchmark + scenario list
│   ├── electrothermal.yaml      #   focused electrothermal subset
│   ├── stress_catalog.yaml      #   tier definitions (A / B / C)
│   ├── electrothermal_stress_catalog.yaml
│   ├── kpi_thresholds.yaml      #   regression thresholds
│   └── kpi_thresholds_electrothermal.yaml
├── tools/                       # bench-suite Python runners
│   ├── benchmark_runner.py      #   primary YAML runner
│   ├── benchmark_ngspice.py     #   Pulsim vs SPICE parity runner
│   ├── stress_suite.py          #   tiered stress validation
│   ├── local_limit_suite.py     #   PC-local fixed+variable limits
│   ├── kpi_gate.py              #   regression gate vs frozen baseline
│   ├── freeze_kpi_baseline.py   #   snapshot a KPI baseline
│   └── _console.py              #   shared rich UI helpers
├── kpi_baselines/               # frozen KPI snapshots + artifact manifests
└── local_limit/                 # local-limit sub-suite (own internal manifest)
```

## Path conventions

All relative paths in `manifests/*.yaml` and inside circuit YAMLs
(`path:`, `ngspice_netlist:`, `ltspice_netlist:`, `baseline:`) are
resolved relative to **`benchmarks/`** (the suite root), not to the
manifest's own parent directory. The runners enforce this via the
`benchmark_runner.suite_root()` helper.

## Running

The bench runners under `tools/` are the canonical entry points. They
assume Pulsim is importable as `import pulsim` (either an installed
wheel or with `PYTHONPATH=build/python` pointing at a local build).

```bash
# Use local build bindings
export PYTHONPATH=build/python

# Primary YAML suite
python3 benchmarks/tools/benchmark_runner.py --output-dir benchmarks/out

# Pulsim vs ngspice parity
python3 benchmarks/tools/benchmark_ngspice.py --backend ngspice \
    --output-dir benchmarks/parity_out

# Tiered stress suite
python3 benchmarks/tools/stress_suite.py --output-dir benchmarks/stress_out

# Local fixed+variable limit suite (10 progressive circuits)
python3 benchmarks/tools/local_limit_suite.py \
    --manifest benchmarks/local_limit/benchmarks_local_limit.yaml \
    --output-dir benchmarks/out_local_limit --mode both

# KPI regression gate (after a run + a frozen baseline)
python3 benchmarks/tools/kpi_gate.py \
    --bench-results benchmarks/out/results.json \
    --stress-summary benchmarks/stress_out/stress_summary.json \
    --report-out benchmarks/out/kpi_gate_report.json \
    --print-report

# Freeze a new baseline snapshot
python3 benchmarks/tools/freeze_kpi_baseline.py \
    --baseline-id $(date +%Y-%m-%d) \
    --bench-results benchmarks/out/results.json \
    --stress-summary benchmarks/stress_out/stress_summary.json \
    --source-artifacts-root benchmarks/out
```

Or via the Makefile (paths preconfigured):

```bash
make benchmark-converters
make benchmark-ltspice              # needs LTSPICE_EXE=/path/to/LTspice
make benchmark-converters-compare   # both + combined table
make benchmark-local-limit
```

## `pulsim-bench` CLI (developer tool)

`tools/bench/` at the repo root holds an optional [`typer`]-based
dev CLI that wraps the runners with a unified interface, rich
progress bars, and the `compare` / `show` subcommands for diffing
runs. Install with:

```bash
pip install -e tools/bench
```

Then:

```bash
pulsim-bench run --only rc_step
pulsim-bench parity --backend ngspice
pulsim-bench show benchmarks/out/        # re-render a previous run
pulsim-bench compare run_a/ run_b/       # diff two runs
```

See [`tools/bench/README.md`](../tools/bench/README.md) for details.

## Adding a new benchmark

1. Choose the right `circuits/<category>/` subdirectory. If your
   circuit doesn't fit, propose a new category in a PR rather than
   adding it to the top of `circuits/`.
2. Author the YAML netlist with at least a `benchmark:` block
   (id, category, observables, expectations). See any existing file
   for the shape.
3. If you want SPICE parity: add a paired `.cir` netlist under
   `spice/ngspice/` (and optionally `spice/ltspice/`).
4. Register the entry in `manifests/benchmarks.yaml`:
   ```yaml
   - path: circuits/<category>/<name>.yaml
     ngspice_netlist: spice/ngspice/<name>.cir
     ngspice_observables:
       - column: V(out)
         spice_vector: v(out)
     scenarios: [direct_trap]   # or gmres_trbdf2, trbdf2, …
   ```
5. If validating against a reference waveform: drop the CSV in
   `baselines/<name>.csv` and set `validation.type: reference` in
   the YAML.
6. Run the suite locally; if PASS, commit. Otherwise tune thresholds
   in the YAML with a comment explaining why.

## Known issues

Real solver bugs surfaced by validation are catalogued in
[`circuits/NOTES_known_issues.md`](circuits/NOTES_known_issues.md).
Each entry includes the YAML+CIR fixture, the symptom, and links to
the OpenSpec change expected to fix it. Don't add these to the
active suite — they are regression-test fixtures pending solver work.

## Output artifacts

Each run produces under `--output-dir`:

- `results.csv` — per-scenario metrics (one row per benchmark × scenario)
- `results.json` — full structured payload + telemetry
- `summary.json` — pass/fail counts
- `outputs/<benchmark>/<scenario>/pulsim.csv` — per-run captured waveforms

Parity runs also emit `parity_results.{csv,json}` and
`parity_summary.json`. Stress runs emit `stress_results.{csv,json}` +
`stress_summary.json`.
