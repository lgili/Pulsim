# Benchmark suite — TPEL methods paper §VI

Head-to-head simulation benchmarks between Pulsim and **ngspice**
(open-source SPICE) on the 10 reference converters in
`projects/converters/` and `projects/inverters/`.

The TPEL methods paper uses these data points in §VI ("Benchmark
Results") and §VII ("Discussion: when PWL pays off, when it
doesn't"). ngspice is chosen as the reference because (a) it's
open-source so every reader can reproduce the runs, and (b) it's
a SPICE-family solver, which makes the algorithmic contrast with
Pulsim's PWL cache architecturally clean.

## Layout

```
benchmarks/
├── README.md                      (you are here)
├── buck/                          ← FIRST data point — see below
│   ├── buck.cir                   ← ngspice netlist (matched to Pulsim)
│   ├── run_buck_benchmark.py      ← orchestrator (runs both, compares)
│   └── buck_ngspice_out.txt       ← ngspice ASCII trace (regenerated)
└── results/
    ├── buck_summary.csv           ← one-row table per topology
    └── buck_waveform_overlay.png  ← visual sanity check
```

Each future topology (boost, flyback, half-bridge, …, MMC) gets
its own subfolder of the same shape.

## How to run

```bash
# From the repo root:
python artigos/02_tpel_methods/benchmarks/buck/run_buck_benchmark.py
```

Requirements:
- Pulsim installed in the current Python env (already true if you
  built the repo)
- `ngspice` on `$PATH` — install via `brew install ngspice` (macOS)
  or `apt install ngspice` (Debian/Ubuntu)
- `matplotlib` (optional, for the overlay plot)

## First data point — buck converter

Setup:
- 24 V → 12 V buck, 100 kHz, D = 0.5
- L = 100 µH, C = 100 µF, R_load = 2.4 Ω
- 5 ms simulation window
- Pulsim: fixed dt = 100 ns → 50,001 samples
- ngspice: max dt = 100 ns adaptive → ~69,000 samples
- Compared over the **second half** of the window (2.5–5 ms) so
  the inrush transient is excluded

Result captured on the development machine (macOS 26.5, M-series,
Python 3.13):

| metric              | Pulsim   | ngspice  |
|---------------------|---------:|---------:|
| wall-time           | 2.18 s   | 0.31 s   |
| samples             | 50,001   | 69,006   |
| Δresident memory    | ~76 MiB  | ~5 MiB   |
| mean V_out (2.5–5 ms) | 11.99 V  | 11.71 V  |
| ripple amplitude    | ~25 mVpp | ~25 mVpp |

Cross-simulator comparison:
- **RMSE (V_out, steady state):** 286 mV
- **max |error|:** 348 mV

### Interpretation (for paper §VII)

Two honest, important findings:

1. **ngspice is faster on the trivial 2-switch buck.** The PWL
   cache build cost (LU factorisation of each switch-state's
   reduced system) and the cmake JIT-build of Pulsim's runtime
   together dominate for very small circuits. The win for PWL
   shows up only when switch count rises — see boost (next),
   bridge rectifier, NPC, MMC. This is a structural fact about
   when state-space caching pays off vs per-step Newton, and the
   paper presents it as the central trade-off of the approach.

2. **~2.4 % DC offset (290 mV / 12 V) between simulators.** Root
   cause: how each tool models conduction losses.
   - Pulsim's `add_diode("D_FW", ..., g_on=1e3, V_th=0.7)` ⇒
     ideal-rectifier behaviour with a fixed 0.7 V drop only when
     `i > 0`; no exponential I-V.
   - ngspice's `D` model uses Shockley I-V (`IS`, `N`, `VJ`, `RS`)
     ⇒ continuous I-V curve, slightly higher effective V_F at
     the buck's load current.
   For the TPEL paper the comparison is *device-model-matched*,
   not *device-physics-matched* — the goal is to show the
   *algorithmic* speed-up of PWL caching, not to claim Pulsim
   models semiconductors better than SPICE.

## Status

- [x] buck — Pulsim & ngspice runs reproducible, RMSE captured
- [ ] boost — TBD (next; expect Pulsim to start catching up at 4+ switches)
- [ ] buck-boost
- [ ] forward
- [ ] flyback
- [ ] half-bridge LLC
- [ ] boost PFC
- [ ] 3-phase VSI
- [ ] NPC 3-level
- [ ] MMC (N = 3)
