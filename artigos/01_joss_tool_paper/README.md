# JOSS Tool Paper — Pulsim

**Venue:** [Journal of Open Source Software (JOSS)](https://joss.theoj.org/)
**Status:** 📝 Drafting (paper.md + paper.bib)
**Target submit:** June 2026
**DOI (after acceptance):** _TBD_

## What is JOSS

JOSS is a **diamond open-access**, peer-reviewed journal that publishes
research software. The journal is unusual:

* **The software is the artefact**, not the paper. Reviewers audit the
  GitHub repo (tests, CI, docs, licence, statement of need); the
  paper itself is 250–1000 words explaining *what the software does
  and why it exists*.
* **Open peer review on GitHub.** When you submit, JOSS opens an
  issue in their repo where 2 reviewers go through a structured
  checklist (does it install? do tests pass? is there a statement of
  need? etc.) and file improvements as PRs against your repo. This
  hardens the software during review.
* **$0 to authors, $0 to readers.** Diamond OA. Funded by NumFOCUS
  and partners.
* **Citable DOI** issued on acceptance (via Crossref).
* **Indexed** in DOAJ, Scopus, Web of Science, ADS. CiteScore ~3.2.

JOSS has published 2000+ papers since 2016 (many top-tier scientific
software: AstroPy, NetworkX, Yt, PyMC, qutip, hundreds more).

**Format expected by JOSS:**
- `paper.md` — Markdown source, ≈600 words, with sections:
  `Summary`, `Statement of need`, `Functionality` (optional),
  `Acknowledgements`.
- `paper.bib` — BibTeX file with all cited prior work.
- Both files live in this folder; they get bundled with the submission.

## Pre-submission checklist

Mark each as the prep work is done.

### Repository requirements

- [ ] Public on GitHub for at least 6 months ✅ (already true)
- [ ] OSI-approved licence ✅ (MIT/Apache?)
- [ ] `README.md` with install + minimal "hello world" example
- [ ] Functional tests with reasonable coverage
- [ ] **CI passing on Linux + macOS** (Windows nice-to-have)
- [ ] Documentation (install, API, examples) — Sphinx, MkDocs, or
  the existing `docs/` folder if MkDocs already builds
- [ ] [`CITATION.cff`](https://citation-file-format.github.io/) at
  the repo root

### Paper contents

- [ ] `paper.md` — summary, statement of need, functionality
  highlights, acknowledgements (≈600 words)
- [ ] `paper.bib` — all referenced prior work
- [ ] At most **1 figure** (JOSS prefers visual examples sparingly)
- [ ] Author affiliations + ORCID IDs

### Tag a release

- [ ] Bump version (e.g. `v1.1.0`) tagged at the exact commit JOSS
  will review
- [ ] Zenodo or Software Heritage archive of that tag (Zenodo
  auto-archives if you connect the GitHub repo to Zenodo)
- [ ] The tagged release matches the paper's claims (no
  "coming soon" features)

### Submit

- [ ] Submit at https://joss.theoj.org/papers/new
- [ ] Pre-review issue opened in
  https://github.com/openjournals/joss-reviews (usually within 24 h)

## Estimated effort

| Task | Time |
|---|---|
| CI hardening (if needed) | 1 weekend |
| `CITATION.cff` + version tag | 1 hour |
| Draft `paper.md` + `paper.bib` | 1–2 weekends |
| README install + example polish | 1 evening |
| Review iteration (reviewer comments → code/docs changes) | 1–2 weekends spread across 4–6 weeks |

**Realistic submit-by-target:** 4–6 weeks of part-time work after
green-light. JOSS median time from submission → publication is
~12 weeks.

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | This tracker (you are here) |
| `paper.md` | The submitted paper |
| `paper.bib` | BibTeX references for `paper.md` |
| `figure_pulsim_overview.png` | (optional) one figure for the paper |

## Reviewer feedback log

_(populated after JOSS opens the review issue)_

| Date | Reviewer | Comment | Status |
|---|---|---|---|
| — | — | — | — |
