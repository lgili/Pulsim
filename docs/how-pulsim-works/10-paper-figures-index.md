# 10. Paper Figures Index

The single page that maps every figure in the "How Pulsim
Works" doc set to the paper section it feeds. Reviewers and
co-authors land here; everything below is a stable link target.

**Status (as of iteration 3 of the doc set):** 12 of 12
matplotlib figures rendered + 4 mermaid diagrams embedded
in-line in chapters. Both PNG (web) and PDF (paper-grade,
300 dpi) outputs land in `_figures/output/`.

---

## Index

| Fig # | Chapter | Title | Paper section | Status |
|---|---|---|---|---|
| 1.1 | [§1.2](01-introduction.md#12-what-breaks-in-spice) | SPICE per-step CPU cost share | Intro § (motivation) | ✅ rendered |
| 1.2 | [§1.3](01-introduction.md#13-the-structural-insight-smps-has-finite-topology) | Topology census across 10 reference converters | Intro § (sparsity in topology space) | ✅ rendered |
| 2.1 | [§2.4](02-mna-foundations.md#24-a-worked-example-assembling-a-buck) | Buck schematic | §II MNA review | mermaid (no PDF) |
| 2.2 | [§2.5](02-mna-foundations.md#25-what-sparse-looks-like-for-real-smps) | Sparsity patterns: buck / NPC / MMC | §II.B sparsity + bandedness | ✅ rendered |
| 3.1 | [§3.3](03-trapezoidal-companion.md#33-the-companion-model-view) | Capacitor companion model | §III companion model | mermaid (no PDF) |
| 3.2 | [§3.5](03-trapezoidal-companion.md) | Buck $i_L$ at $\Delta t = 1\text{µs} / 100\text{ns} / 10\text{ns} / 1\text{ns}$ | §III.A discretisation convergence | ✅ rendered |
| 4.1 | [§4.2](04-pwl-state-space-cache.md#42-the-lifecycle) | Cache lifecycle flowchart | §IV cache mechanics | mermaid (no PDF) |
| 4.2 | [§4.3](04-pwl-state-space-cache.md#43-cost-analysis-amortisation-in-the-limit) | Build-cost amortisation curve (cache vs SPICE-style) | §IV.C cost analysis | ✅ rendered |
| 5.1 | [§5.3](05-sparse-lu-foundations.md#53-what-fill-is-and-why-it-ruins-naive-sparse-lu) | Fill comparison: original / natural-LU / RCM-LU | §V.A ordering | ✅ rendered |
| 5.2 | [§5.5](05-sparse-lu-foundations.md#55-the-elimination-tree) | Elimination tree of 8×8 RCM-ordered | §V.B etree | ✅ rendered |
| 5.3 | [§5.8](05-sparse-lu-foundations.md#58-putting-it-together-cost-vs-n) | Asymptotic cost vs $n$ (4 algorithms) | §V.C complexity | ✅ rendered |
| 6.1 | [§6.2](06-pulsim-sparse-lu.md) | `PulsimSparseLuSolver` lifecycle state diagram | §VI.A in-house impl | mermaid (no PDF) |
| 6.2 | [§6.3](06-pulsim-sparse-lu.md) | Dynamic pattern discovery vs symbolic prediction | §VI.B implementation detail | ✅ rendered |
| 6.3 | [§6.4](06-pulsim-sparse-lu.md) | Pivot-row swap visualisation | §VI.C partial pivoting | ✅ rendered |
| 7.1 | [§7.3](07-rank1-partial-refactor.md#73-the-path-walk-visualised) | Etree path walk for a changed column | **§VII (TPEL contribution)** | ✅ rendered |
| 7.2 | [§7.4](07-rank1-partial-refactor.md#74-the-pivot-fault-fallback) | Pivot fault recovery flow | **§VII.B fault recovery** | mermaid (no PDF) |
| 8.1 | [§8.3](08-benchmarks.md#83-speedup-vs-n_mathrmstate) | Captured speedups B/A, C/B, C/A vs $n_{\mathrm{state}}$ | **§VIII.A captured speedup** | ✅ rendered |
| 8.2 | [§8.4](08-benchmarks.md#84-the-decomposition-in-bar-form) | Multiplicative decomposition stacked bars | **§VIII.A decomposition** | ✅ rendered |
| 8.3 | [§8.5](08-benchmarks.md#85-per-call-cost-flat-vs-linear) | Per-call cost vs $n_{\mathrm{state}}$ (flat vs linear) | **§VIII.B asymptotic scaling** | ✅ rendered |
| 9.1 | [§9.1](09-architecture-walkthrough.md#91-the-layer-model) | Pulsim 10-layer stack | §I.B kernel architecture | ✅ rendered |
| 9.2 | [§9.3](09-architecture-walkthrough.md#93-cross-layer-dependency-graph) | Cross-layer dependency DAG | §I.B (appendix) | mermaid (no PDF) |

**Inventory at iteration 3:**

- **12 PDF/PNG matplotlib figures** in `_figures/output/`
- **6 mermaid diagrams** embedded in markdown (rendered live
  in the docs site; export to PDF via `npx @mermaid-js/mermaid-cli`
  for paper inclusion)

---

## Regenerating the figures

Single command:

```bash
python docs/how-pulsim-works/_figures/generate_all.py
```

Outputs land in `docs/how-pulsim-works/_figures/output/` as
**both PNG and PDF**.

- **PNG** is for the docs site (mkdocs serves them; KaTeX
  renders inline)
- **PDF** is for paper inclusion via `\includegraphics{...}` in
  the TPEL paper LaTeX

Mermaid diagrams render natively in the docs site. For paper
inclusion, export to PDF via the Mermaid CLI:

```bash
npx @mermaid-js/mermaid-cli -i diagram.mmd -o diagram.pdf -t neutral
```

(Pulsim's paper figures workflow doesn't bundle this yet — the
two paper-bound mermaid diagrams in this set, figs 4.1 and 7.2
flowcharts, would need a one-time CLI export when the paper
draft consumes them.)

---

## Style contract

Every PDF figure conforms to a uniform style enforced by
`generate_all.py:apply_paper_style()`:

- **Width**: 7.0 in (IEEE double-column) for most plots,
  5-6 in for single-column / standalone diagrams
- **DPI**: 300
- **Font**: Computer Modern Roman serif (matches IEEE paper
  body text)
- **Body size**: 10 pt; titles 11 pt; legends 9 pt; tick
  labels 9 pt
- **Axis line width**: 0.6 pt
- **Grid alpha**: 0.3 (visible but unobtrusive)
- **Legend**: borderless (`frameon=False`)
- **Spines**: top + right removed for plot-style figures
  (matplotlib's default frame is too cluttered for IEEE
  typography)

If you add a figure that violates this style, the visual
coherence breaks. Use the same `apply_paper_style()` call as
the first line of every `render(output_dir)` function — it's
applied automatically by `generate_all.py` if you go through
the standard entry point.

---

## Data provenance

Two figures pull from real captured data:

| Figure | Data source |
|---|---|
| 8.1, 8.2, 8.3 | `artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv` |

Every other figure is computed from closed-form math, synthetic
fixtures (the buck-like 8×8 from `core/tests/layer0/`), or
representative-data tables embedded in the script. The split
is intentional: chapters 1-7 illustrate concepts that don't
need a kernel build to demonstrate; chapter 8 measures the
captured numbers that the TPEL paper's headline claim rests on.

When the microbench is recaptured (e.g. on different hardware,
a new compiler, or under future-Pulsim improvements), the CSV
updates and figures 8.1-8.3 regenerate automatically. Hardware-
identifier metadata in the CSV header captures the provenance.

---

## Cross-paper reuse

The eventual TPEL paper draft (under
`artigos/02_tpel_methods/`) will pull figures directly from
this directory:

```latex
% In the paper's source.tex:
\includegraphics[width=\columnwidth]{
    ../../docs/how-pulsim-works/_figures/output/fig81_speedup_vs_n.pdf
}
\caption{Captured speedup ratios on the 3-backend microbench.
         Source: \texttt{docs/how-pulsim-works/\_figures/fig81\_speedup\_vs\_n.py}}
```

This way the doc set + the paper share one source of truth.
Changing a figure caption or restyling a plot updates both the
docs and the paper in a single edit. The figure-script comments
include the `Source: ...` provenance line so reviewers can
trace each figure back to its generator.

---

## What's NOT here (yet)

Some paper-bound figures are not yet in this index — they'll
land as the TPEL draft progresses:

- **Per-converter benchmark** on the 10 reference projects
  (deferred to `add-pwl-rank1-runtime-integration` proposal).
  Currently only the synthetic N-switch-chain is captured.
- **Pivot-fallback rate heatmap** sweeping `PIVOT_THRESH ∈
  {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}` × the 8 $N$ values. Would
  back chapter 8's "1e-3 is the sweet spot" claim with a
  full sensitivity sweep instead of just the captured "zero
  fallbacks at 1e-3" data point.
- **AC sweep complex-solver benchmark** — pending
  implementation of `add-pulsim-complex-sparse-lu` (the next
  major architectural change after v1.3.0).

These show up as "TBD" in the TPEL paper draft outline and
will be added to this index once captured.

---

## Cross-references

- The **TPEL paper draft** under
  `artigos/02_tpel_methods/`
- The **OpenSpec proposals** under `openspec/changes/archive/`
  for `add-pwl-rank1-update`, `add-pwl-rank1-partial-refactor`,
  `replace-klu-with-pulsim-sparse-lu`
- **`RANK1_RESULTS.md`** at the canonical writeup of the
  microbench
- **[Chapter 9 — Architecture Walkthrough](09-architecture-walkthrough.md)**
  for where each algorithm lives in the layer stack
