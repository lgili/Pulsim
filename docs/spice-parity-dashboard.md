# SPICE Parity Dashboard

> Status: Fase 1 shipped — pretty terminal UI on top of the existing
> `benchmarks/benchmark_ngspice.py`. Fase 2 (filling in missing
> ngspice netlists for switching circuits) and Fase 3 (using the
> dashboard to root-cause real divergences) are the natural follow-ups.

`scripts/parity_dashboard.py` is a thin wrapper around the existing
parity runner. It enumerates `benchmarks/benchmarks.yaml`, runs each
circuit one at a time, and renders a live progress view + a final
summary table. The underlying JSON / CSV artifacts that `benchmark_
ngspice.py` writes are still produced (one set per circuit, in
`<output-dir>/<benchmark_id>/`), so any existing CI integration keeps
working.

## TL;DR

```bash
# Run every benchmark with the live UI:
python scripts/parity_dashboard.py

# Just a few:
python scripts/parity_dashboard.py --only rc_step rlc_step

# Quiet mode — one summary line + exit code (CI-friendly):
python scripts/parity_dashboard.py --quiet
echo $?     # 0 = all passed; 2 = at least one failed
```

Sample output (passing run on the four passive circuits):

```
[ 1/ 4] running rc_step  ... ✓ passed   max_err=2.496e-03    3/3 scenarios — direct_trap
[ 2/ 4] running rl_step  ... ✓ passed   max_err=4.970e-02    2/2 scenarios — direct_trap
[ 3/ 4] running rlc_step ... ✓ passed   max_err=2.729e-02    2/2 scenarios — direct_trap
[ 4/ 4] running rc_dc    ... ✓ passed   max_err=1.072e-07

Pulsim vs ngspice — parity summary
  benchmark   status     max_err     rms_err  threshold  Pulsim ms ngspice ms  speedup
✓ rc_step     passed   2.496e-03   7.896e-04   3.00e-03       15.2       43.3   2.84x
✓ rl_step     passed   4.970e-02   1.116e-02   6.00e-02        3.2       27.1   8.34x
✓ rlc_step    passed   2.729e-02   4.995e-03   3.00e-02       16.7       42.2   2.53x
✓ rc_dc       passed   1.072e-07   3.402e-08   3.00e-03       14.3       37.5   2.62x

Result: 4/4 passed   failed=0   skipped=0   errors=0   pass_rate=100.0%   wall=2.4s
```

## How the UI is rendered

| Mode | When | What you see |
|---|---|---|
| `--rich` (auto on a TTY) | `rich` installed and stdout is a terminal | Live spinner + bordered tables + colored max_err vs threshold. |
| `--ascii` (auto when not a TTY) | Pipe / log file / non-rich env | Plain text, status icon (`✓ ✗ ○ !`), space-separated columns. |
| `--quiet` | CI / shell scripts | Single line `Result: P/T passed …` plus the process exit code. |

`--verbose` echoes the underlying runner's stdout/stderr per benchmark
when something looks weird (compile errors, ngspice barfing on a
netlist, etc.).

The dashboard auto-sets `PYTHONPATH=build_py/python` if that directory
exists, so you can run against a local build without exporting it
manually.

## Status icons

| Icon | Meaning |
|---|---|
| `✓` | passed — Pulsim agrees with SPICE within the per-circuit `expectations.metrics.max_error` threshold. |
| `✗` | failed — Pulsim ran but max_error blew past the threshold. The summary row is colored red and the JSON `failure_reason` is shown in the `note` column. |
| `○` | skipped — usually means "no SPICE netlist mapped for this backend yet". The dashboard counts these separately so they don't pollute the pass rate. |
| `!` | error — the runner subprocess didn't even produce a results JSON. Last line of stderr is shown. |
| `…` | running — only visible in the rich progress strip. |

## Color coding the `max_error` column

The rich table picks one of three colors for each `max_error` cell:

- **green** when `max_error / threshold < 0.5` (lots of headroom)
- **yellow** when `0.5 ≤ ratio < 1.0` (you're using more than half the
  budget — the next time numerics drift you'll fail)
- **red bold** when `ratio ≥ 1.0` (over the threshold; the row also shows
  status `failed`)

This is pure cosmetics — the pass/fail decision is the runner's, not
the dashboard's.

## Persisted artifacts

```
<output-dir>/
├── dashboard_summary.json          # top-level: aggregate + per-row metrics
├── rc_step/
│   ├── ngspice_summary.json        # what benchmark_ngspice writes
│   ├── ngspice_results.json
│   └── ngspice_results.csv
├── rlc_step/
│   └── ...
└── ...
```

`dashboard_summary.json` is the easy-to-grep top-level rollup with
`aggregate.{total, passed, failed, skipped, pass_rate, p50_max_error,
p99_max_error}` plus per-circuit metrics.

## Current state of the manifest

| Group | Status |
|---|---|
| Linear passives (`rc_step`, `rl_step`, `rlc_step`, `rc_dc`) | ✓ all 4 pass with multiple scenarios. Speedups: 2.5×–8.3× vs ngspice. |
| Nonlinear / switching (`diode_rectifier`, `buck_switching`, `boost_switching_complex`, `interleaved_buck_3ph`, `buck_mosfet_nonlinear`, `stiff_rlc`, `periodic_rc_pwm`) | ○ skipped — `entry.ngspice_netlist` not yet wired in `benchmarks.yaml` (LTspice variants exist for some). Filling these in is Fase 2 of the parity work. |

The skips are not silent failures: the dashboard prints the exact
reason in the `note` column ("Missing ngspice netlist mapping ...",
"0/N scenarios — direct_trap"), and a CI gate that wants to fail on
"any non-passed" can do so via the exit code (which today is 0 —
skipped doesn't count as a failure — but the `aggregate.skipped` field
is the easy lever to add a stricter gate).

## Roadmap

| Phase | Scope |
|---|---|
| **1 — UI shipped today** | `scripts/parity_dashboard.py` with rich + ASCII fallback, exit code, JSON summary. |
| **2 — fill the gaps** | Add ngspice netlists for the seven currently-skipped circuits. Run `python scripts/parity_dashboard.py` and confirm 11/11 pass. |
| **3 — track real bugs** | Use the dashboard as the diagnostic surface for known issues (e.g. the buck open-loop transient that produces `V_out ≈ -0.6 V` instead of settling near the design point — see [`converter-templates.md`](converter-templates.md) "open-loop bound" footnote). When the dashboard turns the `buck_switching` row red, the per-circuit JSON + CSV gives the waveform diff to root-cause it. |
| **4 — CI gate** | Wire `python scripts/parity_dashboard.py --quiet` into the PR workflow so a regression that breaks any passing circuit blocks the merge. |

## See also

- [`benchmarks-and-parity.md`](benchmarks-and-parity.md) — the broader
  benchmark architecture and KPI gating.
- `benchmarks/README.md` — manifest format reference.
- [`converter-templates.md`](converter-templates.md) — the source of
  some of the open-loop transient quirks the parity dashboard will help
  triage.
