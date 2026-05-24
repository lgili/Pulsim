# `benchmarks/` Structural Reorg — Execution Plan

**Status:** plan only — execution pending.  
**Owner:** TBD (open PR against this branch when picking it up).  
**Drafted:** 2026-05-23.

This document captures the agreed execution plan for cleaning up the
`benchmarks/` directory. The plan is split into 8 atomic phases so
each can land independently if reviewers prefer.

---

## Why a reorg

Audit of current `benchmarks/` (post-Pulsim 1.0 dead-code sweep
`64837c4`) found:

- **322 files / 106 MB** at the top level + 10 subdirs.
- `__pycache__/` committed by accident (10 `.pyc` files, ~416 KB).
- 3 dirs of historical run artifacts (`phase8_artifacts/`,
  `phase11_artifacts/`, `phase12_artifacts/`) — 29 files / 384 KB —
  preserved in git history, no reason to keep in tree.
- `compare_results.py` (240 lines, RC/RL/RLC-only) is legacy dangling
  — references retired API patterns.
- 6 YAML configs jumbled with code at top level (manifests + stress
  catalogs + KPI threshold files).
- `circuits/` is flat with 116 entries — no grouping by category.
- `ngspice/` (66 .cir) and `ltspice/` (8 .cir) are sibling top-level
  dirs; should be under a common `spice/` parent.
- `README.md` documents scripts that were deleted in `64837c4`
  (`benchmark_runner.py`, `benchmark_ngspice.py`, `stress_suite.py`,
  `local_limit_suite.py`, `validation_matrix.py`).

Net result: hard to find anything, hard to know what's authoritative,
hard to onboard a new contributor.

---

## Target structure

```text
benchmarks/
├── README.md                       # rewritten — reflects new layout
├── circuits/                       # YAML netlists, categorised
│   ├── linear/                     # ~25  (rc, rl, rlc, dividers, bridges)
│   ├── switching/                  # ~35  (buck, boost, flyback, vcswitch, …)
│   ├── diodes/                     # ~10  (rectifiers, clamps, clippers)
│   ├── magnetics/                  # ~6   (transformer, saturating, coupled)
│   ├── motors/                     # ~5   (dc, bldc, pmsm, induction)
│   ├── thermal/                    # ~5   (electrothermal, long-run)
│   ├── stress/                     # ~10  (stiff, high-Q, multi-rate)
│   ├── three_phase/                # ~10  (vsi, grid, pll, dq, inverter)
│   ├── closed_loop/                # ~10  (cl_buck_*, cl_boost_*)
│   └── NOTES_known_issues.md
├── spice/                          # SPICE backend nets grouped together
│   ├── ngspice/
│   └── ltspice/
├── baselines/                      # reference CSV waveforms (kept flat for now)
├── manifests/                      # config YAMLs
│   ├── benchmarks.yaml
│   ├── electrothermal.yaml
│   ├── stress_catalog.yaml
│   ├── electrothermal_stress_catalog.yaml
│   ├── kpi_thresholds.yaml
│   └── kpi_thresholds_electrothermal.yaml
├── tools/                          # scripts that survived the sweep
│   ├── freeze_kpi_baseline.py
│   └── kpi_gate.py
└── kpi_baselines/                  # frozen KPI snapshots
```

Net delta:
- Deletes: 39 files (`__pycache__/` + 3 `phase*_artifacts/` dirs).
- Maybe deletes: `compare_results.py` if confirmed legacy (decision in B1).
- Renames/moves: ~150 files.
- Path edits: ~150 occurrences in manifests + Makefile + any consumer.
- Rewrite: `README.md`.

---

## Phase A — Safe cleanups (no functional change)

| # | Action | Files affected |
|---|--------|----------------|
| A1 | `git rm -r benchmarks/__pycache__/` | -10 files, -416 KB |
| A2 | `git rm -r benchmarks/phase8_artifacts/` | -10 files, -84 KB |
| A3 | `git rm -r benchmarks/phase11_artifacts/` | -10 files, -152 KB |
| A4 | `git rm -r benchmarks/phase12_artifacts/` | -9 files, -148 KB |
| A5 | Append to `.gitignore`: `**/__pycache__/`, `benchmarks/out*/`, `benchmarks/stress_out*/`, `benchmarks/parity_out*/`, `benchmarks/*_out/` | prevents recurrence |

**Verification:** `git status` clean, no test/script references those
artifact dirs.

## Phase B — Move surviving scripts to `tools/`

| # | From → To | Notes |
|---|-----------|-------|
| B1 | `benchmarks/compare_results.py` → **decide: delete or move to tools/** | Pre-1.0 RC/RL/RLC-only comparator. Likely retire. |
| B2 | `benchmarks/freeze_kpi_baseline.py` → `benchmarks/tools/freeze_kpi_baseline.py` | |
| B3 | `benchmarks/kpi_gate.py` → `benchmarks/tools/kpi_gate.py` | |
| B4 | Update Makefile + any caller's `--script` path | |

## Phase C — Consolidate configs under `manifests/`

| # | From → To |
|---|-----------|
| C1 | `benchmarks/benchmarks.yaml` → `benchmarks/manifests/benchmarks.yaml` |
| C2 | `benchmarks/electrothermal_benchmarks.yaml` → `benchmarks/manifests/electrothermal.yaml` (renamed: drop redundant `_benchmarks`) |
| C3 | `benchmarks/stress_catalog.yaml` → `benchmarks/manifests/stress_catalog.yaml` |
| C4 | `benchmarks/electrothermal_stress_catalog.yaml` → `benchmarks/manifests/electrothermal_stress_catalog.yaml` |
| C5 | `benchmarks/kpi_thresholds.yaml` → `benchmarks/manifests/kpi_thresholds.yaml` |
| C6 | `benchmarks/kpi_thresholds_electrothermal.yaml` → `benchmarks/manifests/kpi_thresholds_electrothermal.yaml` |
| C7 | Update Makefile, CI workflows, any reference |

## Phase D — Consolidate SPICE backends under `spice/`

| # | From → To | File count |
|---|-----------|------------|
| D1 | `benchmarks/ngspice/` → `benchmarks/spice/ngspice/` | 66 .cir |
| D2 | `benchmarks/ltspice/` → `benchmarks/spice/ltspice/` | 8 .cir |
| D3 | Update `ngspice_netlist:` / `ltspice_netlist:` paths in `benchmarks/manifests/*.yaml` | ~150 path edits |

## Phase E — Sub-categorise `circuits/`

The largest phase. 135 YAMLs (116 current + 19 added by PR #30) into
9 subdirs. Use the `benchmark.category:` field in each YAML as the
primary hint; for ambiguous ones, the maintainer decides.

Category assignment heuristics (refine as you go):
- `linear/` — observable is V/I across an R, L, or C, no diodes / switches.
- `switching/` — has vcswitch, mosfet, igbt, OR pwm-driven topology.
- `diodes/` — diode is the dominant nonlinearity, no controlled switches.
- `magnetics/` — uses `transformer`, `coupled_inductor`, or
  `saturable_inductor`.
- `motors/` — uses `dc_motor`, `bldc_motor`, `pmsm`, `induction_motor`,
  `pmsm_foc`, `mechanical`.
- `thermal/` — has electrothermal feedback / `R_th` parameters.
- `stress/` — explicitly stress-style: stiff_*, high_q_*,
  long_run_drift_*, periodic_*.
- `three_phase/` — uses three-phase sources, VSI, PLL, dq frame.
- `closed_loop/` — starts with `cl_` prefix.

Per file:
1. Read `benchmark.category:` field, fall back to filename heuristic.
2. `git mv circuits/<name>.yaml circuits/<category>/<name>.yaml`.
3. Update `path: circuits/<name>.yaml` → `path: circuits/<category>/<name>.yaml`
   in `manifests/benchmarks.yaml` (and `electrothermal.yaml`,
   `stress_catalog.yaml`).

## Phase F — Baselines: keep flat (default) OR mirror circuits/

**Default (recommended):** keep `baselines/` flat. CSV files have
unique names matching their YAML, and the structure adds no value if
queried by name.

**Alternative:** mirror `circuits/<category>/` under `baselines/`.
Update `baseline:` field in each YAML.

Decide in PR review.

## Phase G — Rewrite docs

| # | Action |
|---|--------|
| G1 | Rewrite `benchmarks/README.md` — accurate command examples for the v1.x runner (whatever that is when this lands), no dead-script references |
| G2 | `benchmarks/BENCHMARK_REPORT.md` — archive to `benchmarks/docs/legacy_report.md` OR delete (history preserved in git) |

## Phase H — Validation

| # | Action |
|---|--------|
| H1 | `grep -r "circuits/" benchmarks/manifests/` — confirm every path resolves |
| H2 | `grep -r "benchmarks/" Makefile .github/ scripts/ docs/` — catch stale paths |
| H3 | CI is green (parity + KPI gate + whatever else lives now) |
| H4 | `git status` clean, no orphan files |

---

## Order of execution

A → B → C → D → E → F → G → H

Each phase can be its own commit (or its own PR) — they're
independent enough. Phase E is the largest; consider splitting per
category (`linear/` first, `switching/` next, etc.) if reviewer
prefers smaller diffs.

## Out of scope (for this reorg)

- Writing the new v1.x benchmark runner (the team's separate work).
- Migrating circuit content (parameters, scenarios) — pure file
  organisation only.
- Touching `kpi_baselines/` or `local_limit/` (they're sub-suites with
  their own internal structure; leave for a follow-up).
