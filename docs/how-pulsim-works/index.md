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

| # | Chapter | What you'll learn | Key figures |
|---|---------|-------------------|-------------|
| 1 | [Introduction](01-introduction.md) | Why switched-mode power electronics breaks general-purpose circuit simulators, and Pulsim's three structural bets | Topology census plot, simulator landscape, dt vs accuracy trade-off |
| 2 | [Modified Nodal Analysis](02-mna-foundations.md) | How a circuit becomes a linear system of equations: KCL stamps, voltage-source augmentation, the $J\mathbf{x} = \mathbf{b}$ form | Buck stamping animation, MNA sparsity pattern |
| 3 | [Trapezoidal Companion Discretisation](03-trapezoidal-companion.md) | From continuous-time $E\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$ to discrete $(E - \tfrac{\Delta t}{2}A)\mathbf{x}_{n+1} = \cdots$, why we use trapezoidal | Companion-model figure, dt = 1 µs vs 1 ns waveform comparison |
| 4 | [PWL State-Space Cache](04-pwl-state-space-cache.md) | The architectural pivot — pre-build one matrix per switch mask, never re-assemble at runtime | Cache build-vs-solve lifecycle, mask transition diagram, 50× speedup decomposition |
| 5 | [Sparse Direct LU Foundations](05-sparse-lu-foundations.md) | Why direct (not iterative), why sparse (not dense), how RCM ordering + elimination trees + Gilbert-Peierls left-looking work | Fill comparison (natural vs RCM vs COLAMD), elimination tree, sparsity-pattern evolution |
| 6 | [PulsimSparseLuSolver](06-pulsim-sparse-lu.md) | Our in-house C++23 sparse LU: `analyze` → `factorize` → `solve` lifecycle, partial-pivoting threshold check, why we ditched SuiteSparse KLU | Lifecycle state diagram, dynamic-pattern discovery, pivot-row swap visualization |
| 7 | [Path-Based Partial Refactorisation](07-rank1-partial-refactor.md) | **The algorithmic contribution of the forthcoming methods paper.** Etree-walk path computation, threshold pivot fault, lazy-union varying-set caching | Path walk diagram, single-bit-flip update example, fault-and-recover flow |
| 8 | [Benchmarks](08-benchmarks.md) | What the numbers actually mean: the 3-backend decomposition, why the speedup is real, and where the algorithm doesn't help | Speedup vs $n$ plot, per-call cost flat vs linear scaling, fallback-rate heatmap |
| 9 | [Architecture Walkthrough](09-architecture-walkthrough.md) | The 10-layer codebase, from `Layer 0` (numeric types) to `Layer 9` (the Python `simulate()` ergonomic facade) | Layer-stack figure, layered dependency graph |
| 10 | [Paper Figures Index](10-paper-figures-index.md) | Every figure regenerated for the forthcoming methods paper, with provenance + the script that built it | Reference grid of all paper-bound figures |

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
