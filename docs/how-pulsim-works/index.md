# How Pulsim Works

A guided tour of Pulsim's simulation engine, from the underlying
circuit equations all the way up to the per-converter benchmark
numbers. Written for two audiences in parallel:

- **Researchers** who want to know exactly which algorithms
  Pulsim implements, with the math + literature references that
  back each design choice.
- **Engineers** building on the kernel who need a mental model
  precise enough to extend it without reading every header.

Every chapter ends with **takeaways** (the practical implications)
and a **further reading** pointer (papers + the `internals/` docs
that go even deeper).

---

## Reading order

The chapters build strictly bottom-up — chapter $N$ assumes
chapters $1..N-1$. Skip ahead at your own risk; the headline
results in **Chapter 8 (Benchmarks)** only make sense once you've
seen what's being measured.

### Part I — PWL engine (chapters 1–10)

The fixed-step trapezoidal companion + PWL state-space cache —
the default `pulsim.simulate(engine='pwl', ...)` path.

| # | Chapter | What you'll learn | Key figures |
|---|---------|-------------------|-------------|
| 1 | [Introduction](01-introduction.md) | Why switched-mode power electronics breaks general-purpose circuit simulators, and Pulsim's three structural bets | Topology census plot, simulator landscape, dt vs accuracy trade-off |
| 2 | [Modified Nodal Analysis](02-mna-foundations.md) | How a circuit becomes a linear system of equations: KCL stamps, voltage-source augmentation, the $J\mathbf{x} = \mathbf{b}$ form | Buck stamping animation, MNA sparsity pattern |
| 3 | [Trapezoidal Companion Discretisation](03-trapezoidal-companion.md) | From continuous-time $E\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$ to discrete $(E - \tfrac{\Delta t}{2}A)\mathbf{x}_{n+1} = \cdots$, why we use trapezoidal | Companion-model figure, dt = 1 µs vs 1 ns waveform comparison |
| 4 | [PWL State-Space Cache](04-pwl-state-space-cache.md) | The architectural pivot — pre-build one matrix per switch mask, never re-assemble at runtime | Cache build-vs-solve lifecycle, mask transition diagram, 50× speedup decomposition |
| 5 | [Sparse Direct LU Foundations](05-sparse-lu-foundations.md) | Why direct (not iterative), why sparse (not dense), how RCM ordering + elimination trees + Gilbert-Peierls left-looking work | Fill comparison (natural vs RCM vs COLAMD), elimination tree, sparsity-pattern evolution |
| 6 | [PulsimSparseLuSolver](06-pulsim-sparse-lu.md) | Our in-house C++23 sparse LU: `analyze` → `factorize` → `solve` lifecycle, partial-pivoting threshold check, why we ditched SuiteSparse KLU | Lifecycle state diagram, dynamic-pattern discovery, pivot-row swap visualization |
| 7 | [Path-Based Partial Refactorisation](07-rank1-partial-refactor.md) | **The algorithmic contribution of the v1.4 methods paper.** Etree-walk path computation, threshold pivot fault, lazy-union varying-set caching | Path walk diagram, single-bit-flip update example, fault-and-recover flow |
| 8 | [Benchmarks](08-benchmarks.md) | What the numbers actually mean: the 3-backend decomposition, why the speedup is real, and where the algorithm doesn't help | Speedup vs $n$ plot, per-call cost flat vs linear scaling, fallback-rate heatmap |
| 9 | [Architecture Walkthrough](09-architecture-walkthrough.md) | The 10-layer codebase, from `Layer 0` (numeric types) to `Layer 9` (the Python `simulate()` ergonomic facade) | Layer-stack figure, layered dependency graph |
| 10 | [Paper Figures Index](10-paper-figures-index.md) | Every figure regenerated for the methods paper, with provenance + the script that built it | Reference grid of all paper-bound figures |

### Part II — Path-Based Event-Driven (DSED) engine (chapters 11–16, v1.6.0)

The opt-in `pulsim.simulate(engine='dsed', ...)` path. Predicts
events analytically, integrates between them with adaptive RK
(DOPRI5) or implicit BDF2 dispatched per mode. **24× wall-clock
faster than PWL on buck CCM**; geo-mean 14.5× across 6
converter topologies.

| # | Chapter | What you'll learn | Key concepts |
|---|---------|-------------------|--------------|
| 11 | [DSED Engine Overview](11-dsed-engine-overview.md) | Why fixed-step trap struggles with PE switching, the PED scheduler loop in one figure, where DSED wins vs where it doesn't | Event-driven vs fixed-step, "path-based" insight, mode-segment graph |
| 12 | [MNA → Continuous-Time State-Space](12-dsed-state-space-extraction.md) | The full derivation of `compute_lti_state_space`: finite-difference recovery of $M_{\text{dyn}}/G_{\text{static}}$, Schur complement, $T^\top M T$ congruence for floating capacitors, projection matrix $B$ for time-varying sources | Two-$h$ recovery, Schur reduction, NPC/MMC floating-cap fix, sine source overlay |
| 13 | [DOPRI5 + Adaptive PI + Event Scan](13-dsed-dopri5-adaptive.md) | DOPRI5 Butcher tableau + FSAL, Söderlind PI step controller, cubic Hermite interpolation for sub-step events, Illinois + Brent root finder | RK5(4), PI control, dense output, predicate event localisation |
| 14 | [BDF2 + Stiffness Dispatch](14-dsed-bdf2-and-dispatch.md) | BDF2 + Crank-Nicolson bootstrap, eigenvalue-based stiffness detector, `PEDSimulatorAuto`'s per-mode-segment dispatch, when BDF2 actually fires | A-stability, $\|\lambda_{\max}\|\cdot h$ threshold, mode-id integrator cache |
| 15 | [Native Bindings — The 24× Story](15-dsed-native-bindings.md) | How the per-step hot loop moved from Python to C++ across 4 bridges. Bridge.10 (C++ scheduler) → Bridge.11 (native adapter) → Bridge.12 (`NativePwm2Switch` fast-path) → Bridge.13 (PWL also benefits) | pybind11 patterns, GIL release, `py::cast<T*>` detection, fallback hierarchy |
| 16 | [DSED Benchmarks](16-dsed-benchmarks.md) | The headline 24× on buck CCM, 14.5× geo-mean across 6 topologies, bridge-by-bridge per-step decomposition, honest scope limits | Per-step decomposition, topology sweep, when DSED doesn't win |

---

## How the figures work

Every plot or diagram you see in these chapters is **regenerable**.
The scripts live under `docs/how-pulsim-works/_figures/` and write
both PNG (for the web) and PDF (for the paper) into
`docs/how-pulsim-works/_figures/output/`:

```bash
cd docs/how-pulsim-works/_figures
python generate_all.py
```

Conceptual diagrams (elimination tree, path walks, layer stacks)
use [Mermaid](https://mermaid.js.org/) embedded inline — they
render natively in the docs site and can be exported as SVG/PDF
for paper inclusion via the Mermaid CLI:

```bash
npx @mermaid-js/mermaid-cli -i diagram.mmd -o diagram.pdf -t neutral
```

This makes the whole chapter set the **canonical source** for the
methods paper's figure inventory. When a figure changes here, the
paper version regenerates from the same script — no manual sync.

---

## How the math works

Display equations use LaTeX rendered by [KaTeX](https://katex.org/):

$$
\bigl(\,E - \tfrac{\Delta t}{2}A\bigr)\,\mathbf{x}_{n+1}
\;=\;
\bigl(\,E + \tfrac{\Delta t}{2}A\bigr)\,\mathbf{x}_{n}
\;+\;
\tfrac{\Delta t}{2}\bigl(B\mathbf{u}_n + B\mathbf{u}_{n+1}\bigr)
$$

Inline math like $E\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$ is
the same syntax. Notation is normalised across chapters; symbols
introduced in one chapter mean the same thing in every later
chapter unless otherwise flagged.

---

## Where this came from

This documentation set was authored alongside the v1.3.0 release
(2026-05-24) — the release that completed the in-house sparse-LU
rewrite and made path-based partial refactorisation the default
fast-path for PWL switching transients. Chapters 6 and 7 cover
the v1.3.0 algorithmic contributions directly; chapters 1–5 + 8
provide the foundations + measurement context needed to evaluate
those contributions.

The corresponding OpenSpec proposals
([`replace-klu-with-pulsim-sparse-lu`](https://github.com/lgili/Pulsim/tree/main/openspec/changes/archive)
and [`add-pwl-rank1-partial-refactor`](https://github.com/lgili/Pulsim/tree/main/openspec/changes/archive))
contain the requirements + design decisions; this section
explains the *why* behind those requirements at the level of an
extended technical narrative.

Pre-existing per-layer docs under `internals/` remain the
authoritative API + code-walkthrough references; this section is
the conceptual layer on top.

---

## Cross-references

- [Mental Model](../mental-model.md) — the 60-second elevator
  pitch (read this *before* chapter 1 if you've never used Pulsim)
- [Layer-by-Layer Internals](../internals/README.md) — the
  per-file walkthrough (read this *after* chapter 9 for the
  code-level detail)
- [Performance Tuning](../performance-tuning.md) — practical
  advice on `dt`, cache lazy/eager modes, and when to override
  the auto solver backend
- A methods-oriented manuscript characterising the in-house
  sparse-LU kernel + path-based partial-refactorisation on
  reference SMPS topologies is in preparation. Until publication
  the draft lives outside this repository to keep the public
  surface focused on the released software.
