# 10. Paper Figures Index

The single page that maps every figure in the "How Pulsim Works"
doc set to the paper section it feeds. Reviewers and co-authors
land here; everything below is a stable link target.

!!! info "Status: index updated as chapters land"
    Figures fully rendered for chapters 1-3; chapters 4-9 figures
    in progress. The script for each figure is reproducible —
    see `docs/how-pulsim-works/_figures/generate_all.py`.

## Index

| Fig # | Chapter | Title | Paper section | Status |
|---|---|---|---|---|
| 1.1 | [§1.2](01-introduction.md#12-what-breaks-in-spice) | SPICE per-step cost share | Intro § (motivation) | ✅ rendered |
| 1.2 | [§1.3](01-introduction.md#13-the-structural-insight-smps-has-finite-topology) | Topology census across 10 reference converters | Intro § (sparsity in topology space) | ✅ rendered |
| 2.1 | [§2.4](02-mna-foundations.md#24-a-worked-example-assembling-a-buck) | Buck schematic | §II MNA review | mermaid (no PDF) |
| 2.2 | [§2.5](02-mna-foundations.md#25-what-sparse-looks-like-for-real-smps) | Sparsity patterns: buck / NPC / MMC | §II.B sparsity + bandedness | ✅ rendered |
| 3.1 | [§3.3](03-trapezoidal-companion.md#33-the-companion-model-view) | Capacitor companion model | §III companion model | mermaid (no PDF) |
| 3.2 | [§3.5](03-trapezoidal-companion.md) | Buck $i_L$ at $\Delta t = 1\text{µs} / 100\text{ns} / 10\text{ns} / 1\text{ns}$ | §III.A discretisation convergence | ✅ rendered |
| 4.1 | [§4](04-pwl-state-space-cache.md) | Cache state diagram (empty vs filled) | §IV cache mechanics | 🔲 pending |
| 4.2 | [§4](04-pwl-state-space-cache.md) | Cache lifecycle flowchart | §IV cache mechanics | 🔲 pending mermaid |
| 4.3 | [§4](04-pwl-state-space-cache.md) | Build-cost amortisation curve | §IV.C cost analysis | 🔲 pending |
| 5.1 | [§5](05-sparse-lu-foundations.md) | Natural / RCM / COLAMD fill comparison | §V.A ordering | 🔲 pending |
| 5.2 | [§5](05-sparse-lu-foundations.md) | Elimination tree of buck-like 8×8 | §V.B etree | 🔲 pending |
| 5.3 | [§5](05-sparse-lu-foundations.md) | Gilbert-Peierls column trajectory | §V.C left-looking | 🔲 pending |
| 5.4 | [§5](05-sparse-lu-foundations.md) | Asymptotic cost vs $n$ (dense / sparse / sparse+RCM) | §V.C complexity | 🔲 pending |
| 6.1 | [§6](06-pulsim-sparse-lu.md) | `PulsimSparseLuSolver` lifecycle state diagram | §VI.A in-house impl | 🔲 pending mermaid |
| 6.2 | [§6](06-pulsim-sparse-lu.md) | Dynamic pattern discovery vs symbolic prediction | §VI.B implementation detail | 🔲 pending |
| 6.3 | [§6](06-pulsim-sparse-lu.md) | Pivot-row swap visualisation | §VI.C partial pivoting | 🔲 pending |
| 7.1 | [§7](07-rank1-partial-refactor.md) | Etree path walk for a changed column | **§VII (TPEL contribution)** | 🔲 pending |
| 7.2 | [§7](07-rank1-partial-refactor.md) | Single-bit flip + path computed | **§VII.A example** | 🔲 pending |
| 7.3 | [§7](07-rank1-partial-refactor.md) | Pivot fault recovery flow | **§VII.B fault recovery** | 🔲 pending mermaid |
| 7.4 | [§7](07-rank1-partial-refactor.md) | Per-call cost breakdown vs $n$ | **§VII.C scaling** | 🔲 pending |
| 8.1 | [§8](08-benchmarks.md) | Speedup vs $n_{\mathrm{state}}$ (3-backend) | **§VIII.A captured speedup** | 🔲 pending |
| 8.2 | [§8](08-benchmarks.md) | Decomposition stacked bar | **§VIII.A decomposition** | 🔲 pending |
| 8.3 | [§8](08-benchmarks.md) | Per-call cost vs $n_{\mathrm{state}}$ | **§VIII.B asymptotic scaling** | 🔲 pending |
| 8.4 | [§8](08-benchmarks.md) | Pivot-fallback rate heatmap | §VIII.C tuning sensitivity | 🔲 pending |
| 9.1 | [§9](09-architecture-walkthrough.md) | Layer stack diagram | §I.B kernel architecture | 🔲 pending |
| 9.2 | [§9](09-architecture-walkthrough.md) | Cross-layer dependency DAG | §I.B (appendix) | 🔲 pending mermaid |
| 9.3 | [§9](09-architecture-walkthrough.md) | Test-binary timing heatmap | §I.C validation methodology | 🔲 pending |

## Regenerating the figures

```bash
python docs/how-pulsim-works/_figures/generate_all.py
```

Outputs land in `docs/how-pulsim-works/_figures/output/` as
**both PNG and PDF**. PDF is for paper inclusion via
`\includegraphics{...}`; PNG is for the docs site.

Mermaid diagrams render natively in the docs site. For paper
inclusion, export to PDF via the Mermaid CLI:

```bash
npx @mermaid-js/mermaid-cli -i diagram.mmd -o diagram.pdf -t neutral
```

## Style contract

Every PDF figure conforms to a uniform style enforced by
`generate_all.py:apply_paper_style()`:

- **Width**: 7.0 in (IEEE double-column) or 3.5 in
  (single-column) — picked per figure based on density
- **DPI**: 300
- **Font**: Computer Modern Roman serif (matches paper body
  text)
- **Body size**: 10 pt
- **Axis line width**: 0.6 pt
- **Grid alpha**: 0.3

If you add a figure that violates this style, the visual
coherence breaks. Use the same `apply_paper_style()` call as
the first line of every `render(output_dir)` function.
