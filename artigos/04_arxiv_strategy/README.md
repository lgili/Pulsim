# arXiv Companion-Preprint Strategy

**Not a paper** — this folder is the **policy + workflow** for
pairing every IEEE Transactions submission with an arXiv preprint.

## Why this exists as a first-class artefact

Our pipeline targets IEEE TPEL and JESTPE through the
**subscription route** (no APC). The trade-off is that the
final-typeset version sits behind the IEEE Xplore paywall.

This is fully mitigated by depositing the **accepted manuscript**
on arXiv:
- IEEE's Author Posting Policy (2024 revision) explicitly permits
  this for the *accepted* version (not the IEEE-typeset version).
- arXiv preprints accrue Google Scholar citations within days of
  posting — far ahead of IEEE Xplore indexing.
- Readers without IEEE access still find the paper through
  Google Scholar / Semantic Scholar / Connected Papers.
- A clearly-stated DOI cross-reference back to the IEEE version
  funnels all citations onto the canonical record.

Without this workflow, a paywalled IEEE paper from an independent
researcher is effectively invisible outside subscribing
institutions — exactly the failure mode we need to avoid.

## Contents

| File | Purpose |
|---|---|
| `README.md` | this file — high-level rationale |
| `WORKFLOW.md` | step-by-step procedure from "Accept" letter → arXiv post |
| `IEEE_POSTING_POLICY.md` | verbatim quotes + URL of relevant IEEE policy clauses, dated, so we don't lose them to a website refresh |
| `arxiv_metadata_template.txt` | pre-filled arXiv metadata reused across submissions |

## Quick reference

**What you can post on arXiv (per IEEE policy):**
- ✅ Submitted manuscript (pre-print) — *before* IEEE peer review
- ✅ Accepted manuscript (post-print) — *after* peer review,
     *before* IEEE typesetting (i.e. your own LaTeX-compiled
     version of the final accepted text)
- ❌ Final IEEE-typeset PDF (the one in IEEE Xplore) — never;
     copyright transferred to IEEE upon acceptance

**Required acknowledgment on the arXiv preprint:**
> © 20XX IEEE. Personal use of this material is permitted.
> Permission from IEEE must be obtained for all other uses, in
> any current or future media, including reprinting/republishing
> this material for advertising or promotional purposes, creating
> new collective works, for resale or redistribution to servers
> or lists, or reuse of any copyrighted component of this work in
> other works. DOI: 10.1109/TPEL.20XX.XXXXXXX

This text goes on the arXiv abstract page (the "Comments" field)
and at the top of the PDF first page.

## Workflow at a glance

```
   IEEE submission  ─►  peer review  ─►  ACCEPT letter
                                              │
                                              ▼
                              prepare "accepted manuscript"
                              (revised LaTeX, no IEEE typesetting)
                                              │
                                              ▼
                              submit to arXiv (eess.SY)
                              with IEEE attribution + DOI
                                              │
                                              ▼
                              update Pulsim README + CITATION.cff
                              with both arXiv ID + DOI
```

See `WORKFLOW.md` for the step-by-step.

## Pulsim-specific arXiv defaults

- **Author:** Luiz Carlos Gili
- **ORCID:** 0000-0002-5749-7199
- **Affiliation:** Independent Researcher, Brazil
- **Primary category:** `eess.SY` (Systems and Control)
- **Cross-listings:** `cs.MS` (Mathematical Software) for the
  methods paper; `physics.app-ph` (Applied Physics) for the
  application paper.
- **License:** [arXiv non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)
  (the default — compatible with IEEE copyright).
- **Email for arXiv account:** luizcarlosgili@gmail.com
