# Pulsim Benchmark Suite

This directory holds the simulation benchmarks Pulsim uses for
validation, regression, and SPICE-parity comparison.

## Layout

```text
benchmarks/
├── README.md                    # this file
├── circuits/                    # YAML netlists, grouped by topology family
│   ├── linear/                  #   pure R/L/C, dividers, bridges, IC discharge
│   ├── switching/               #   DC-DC + inverters, sub-grouped:
│   │   ├── buck/                #     step-down (sync, interleaved, DCM, igbt/gan/…)
│   │   ├── boost/               #     step-up (pfc, sync, interleaved, …)
│   │   ├── buck_boost/          #     buck-boost, Ćuk, SEPIC
│   │   ├── isolated/            #     flyback, forward, active-clamp, DAB, full-bridge
│   │   ├── multilevel/          #     flying-cap, NPC, T-type, cascaded-H, MMC
│   │   ├── inverter/            #     half-bridge inverters, LCC resonant
│   │   └── basic/               #     vcswitch/mosfet/snubber/pwm primitives
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
├── baselines/                   # reference waveforms, gzip-compressed (<name>.csv.gz, flat)
├── manifests/                   # bench-suite configuration
│   ├── benchmarks.yaml          #   primary benchmark + scenario list
│   ├── electrothermal.yaml      #   focused electrothermal subset
│   ├── stress_catalog.yaml      #   tier definitions (A / B / C)
│   ├── electrothermal_stress_catalog.yaml
│   ├── kpi_thresholds.yaml      #   regression thresholds
│   ├── kpi_thresholds_electrothermal.yaml
│   └── local_limit.yaml         #   PC-local fixed+variable limit sub-suite
├── tools/                       # bench-suite Python runners
│   ├── benchmark_runner.py      #   primary YAML runner
│   ├── benchmark_ngspice.py     #   Pulsim vs SPICE parity runner
│   ├── stress_suite.py          #   tiered stress validation
│   ├── local_limit_suite.py     #   PC-local fixed+variable limits
│   ├── kpi_gate.py              #   regression gate vs frozen baseline
│   ├── freeze_kpi_baseline.py   #   snapshot a KPI baseline
│   └── _console.py              #   shared rich UI helpers
├── kpi/                         # KPI-metrics library (compute_thd/efficiency/…) + self-test
└── kpi_baselines/               # frozen KPI snapshots + artifact manifests
```

## Reference harness (2.0)

The 126 circuits are `schema: pulsim-v1` files, which the 2.0 kernel
loader refuses on purpose. They run through a reader that belongs to the
harness, never to `pulsim`:

| Piece | Where |
|---|---|
| v1 reader (`load(path).run(engine)`) | `tools/v1_reader.py` — mapping table in its docstring |
| Triage of all 126 files (`runs` / `skipped` / `error`, `known` cells) | `manifests/v2_triage.yaml` |
| Time-weighted metrics (`err_rms`, `err_max`, `time_mean`) | `tools/metrics.py` |
| Differential runner, circuit x engine | `tools/differential_runner.py` |
| 20 analytic circuits and their ceilings | `analytic/__init__.py`, `manifests/analytic.yaml` |
| Golden checkpoints of `engine='pwl'` (45 circuits) | `goldens/pwl/*.json`, `tools/goldens.py` |
| Counter gate (12 circuits x engines) | `manifests/perf_counters.yaml`, `tools/counters.py` |
| Tests | `python/tests/test_benchmark_harness.py` (marker `reference`) |

```bash
make reference-smoke     # analytic set + goldens + counters (the PR job, ~1 min)
make reference-nightly   # all `runs` circuits x 3 engines + ngspice parity
```

Rules the harness keeps:

- **Every average, RMS and error is time-weighted** (trapezoid on the
  samples); a plain `.mean()` over a variable-step trace once invented an
  engine defect. `err_rms = sqrt(∫(y − ref)² dt / window) / span`.
- **The differential runner compares what an engine computes.** Its
  default observables leave out the nodes an independent voltage source
  ties to ground (an ideal PWM node scored 0.06–0.11 between two tight
  references although nothing reads it), and its default measure
  (`--metric edge`) leaves out the steps of either trace across which the
  recorded switch state changes: interpolating a jump between two sparse
  samples compares a ramp neither engine computed. The window shrinks by
  those steps only; every other interval counts. `--metric interp` is the
  previous measure (every interval interpolated, every node), kept for
  comparison; `reference_status.py --metric` takes the same switch.
- **The first stored sample is excluded** where it is the initial state
  rather than the `t = 0⁺` solution (on `pwl` / `trbdf2` an RL step reads
  0 V across the inductor at `t = 0`; the answer is 10 V).
- **Goldens compare integers exactly** (`n_steps`, `n_commutations`,
  sha256 of the switch-state record) and **values with a tolerance**
  (`rtol` ≤ 1e-9 linear, ≤ 1e-6 with a Newton device; `atol` is `rtol`
  times the state's own scale). A bit-exact hash of floats is never the
  key: platforms differ in the last bits.
- **No wall-clock assertion.** The counter gate compares deterministic
  counters (`pwl` ≤ frozen, `trbdf2` / `dsed` ≤ 1.02 × frozen); wall time
  is printed only.
- **Nothing is skipped silently.** A `skipped` triage entry must still
  fail to load for its reason; an `error` entry must still fail; a
  `known` analytic defect must still disagree. A fix therefore shows up as
  a failing test that asks to promote the entry.

### Adding an analytic case

Add a `Case` to `analytic/__init__.py` (`build(k)` scaling one component
by `k`, `observe`, `exact`, `t_end`, `dt`, engines), then
`python benchmarks/tools/freeze_analytic.py --write` and review the diff:
each ceiling is 2 × the measured error (floors 1e-9, dsed 1e-6). The test
also checks that a 1 % component error fails every engine.

### Re-freezing goldens and counters

A change that moves `engine='pwl'` on purpose (roadmap rows marked **B**)
re-freezes in its own pull request and says so:

```bash
python benchmarks/tools/goldens.py --update-goldens
python benchmarks/tools/counters.py --freeze
python benchmarks/tools/freeze_analytic.py --write
git diff benchmarks/goldens benchmarks/manifests     # review every change
```

A ceiling may be tightened freely; loosening one needs a `reason` on the
entry. Disagreements found by the differential runner are recorded as
`known:` cells in the triage manifest (to be filed as issues), never as
looser tolerances.

### Legacy files

`baselines/*.csv.gz` (70) were captured from the retired v1 engine and
are an informational "drift from v1" column of `benchmark_runner.py`,
not a reference. `manifests/benchmarks.yaml` keeps its v1 scenario names
(`direct_trap`, `gmres_trbdf2`, …) as history; the reference harness
uses the engine list instead. The KPI tools and `kpi_baselines/` are not
ported.

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
    --manifest benchmarks/manifests/local_limit.yaml \
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
5. If validating against a reference waveform: set `validation.type:
   reference` + `baseline: baselines/<name>.csv` in the YAML, then run
   once with `--generate-baselines` to capture it. Baselines are stored
   **gzip-compressed** (`baselines/<name>.csv.gz`) to keep the repo
   small; the loader resolves the plain `.csv` reference to the `.gz`
   file automatically, so YAML refs stay as `.csv`.
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
