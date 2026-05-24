# arXiv Companion-Preprint Workflow

Step-by-step procedure to deposit an accepted IEEE Transactions
manuscript on arXiv, in compliance with the IEEE Author Posting
Policy. Run this exactly once per accepted paper.

## Phase 0 — One-time setup (do this once, before the first paper)

1. **Create arXiv account.** Use luizcarlosgili@gmail.com. arXiv
   requires endorsement for first-time submitters in some
   categories. Check whether `eess.SY` requires endorsement when
   you create the account; if so, request endorsement from any
   already-published author in that category (most are happy to
   endorse a serious submission).
2. **Verify ORCID.** Link the arXiv account to ORCID
   0000-0002-5749-7199 so all arXiv submissions automatically
   attach to the researcher record.
3. **Bookmark these URLs:**
   - arXiv submission portal: https://arxiv.org/submit
   - arXiv author identifier: https://arxiv.org/a/gili_l_1
     (this URL is reserved when you submit your first paper)
   - IEEE Author Center: https://ieeeauthorcenter.ieee.org/

## Phase 1 — After receiving the IEEE "Accept" letter

The Accept letter arrives via Manuscript Central. It will contain:
- Final manuscript ID
- Copyright transfer form (eCF) — sign it.
- Author bio + photo requests.
- DOI assignment (sometimes immediate, sometimes after typesetting).

**Action items:**
1. Sign the IEEE Copyright Form (eCF) via the link in the email.
   Keep a copy. The form confirms the rights you retain — in
   particular, your right to post the accepted manuscript on a
   personal/institutional repository.
2. Note the DOI (`10.1109/TPEL.XXXX.XXXXXXX`) — you will need it
   for the arXiv attribution text.

## Phase 2 — Prepare the "accepted manuscript" version

The arXiv version must be the **accepted manuscript**, *not* the
IEEE-typeset PDF. The arXiv version is your own LaTeX-compiled
PDF of the post-review, pre-typesetting text.

1. **Clone the submission folder.** Make a copy of the LaTeX
   source as submitted to IEEE after the final revision round
   (typically called `paper_revision_2.tex` or similar). Rename
   the copy to `paper_arxiv.tex`.
2. **Remove IEEE-specific typesetting directives.** Strip any
   `\IEEEpubid{}` macros and any IEEE copyright notices from the
   first page (those are IEEE's job).
3. **Add the arXiv attribution header** at the top of the first
   page, immediately after `\maketitle`:
   ```latex
   \begin{flushleft}\footnotesize
   \textcopyright{} 20XX IEEE. Personal use of this material is
   permitted. Permission from IEEE must be obtained for all other
   uses, in any current or future media, including
   reprinting/republishing this material for advertising or
   promotional purposes, creating new collective works, for
   resale or redistribution to servers or lists, or reuse of any
   copyrighted component of this work in other works.\\[0.3em]
   This is the accepted manuscript of a paper to appear in
   \emph{IEEE Transactions on Power Electronics}.
   DOI: \texttt{10.1109/TPEL.XXXX.XXXXXXX}
   \end{flushleft}
   ```
4. **Compile** with `pdflatex paper_arxiv.tex` until the bibliography
   resolves cleanly.
5. **Sanity check:** verify the figure quality is 300 dpi minimum
   and all references resolve.

## Phase 3 — Submit to arXiv

1. Go to https://arxiv.org/submit and click "Start new submission."
2. **Step 1 — License:** select "arXiv.org perpetual,
   non-exclusive license to distribute." This is compatible with
   IEEE copyright (IEEE owns the copyright; you grant arXiv a
   distribution licence, not an exclusive one).
3. **Step 2 — Primary category:** `eess.SY` (Systems and Control).
   **Cross-listing:** `cs.MS` (Mathematical Software) for methods
   papers; `physics.app-ph` for application papers.
4. **Step 3 — Metadata:** use the field values from
   [`arxiv_metadata_template.txt`](arxiv_metadata_template.txt).
   In particular:
   - **Comments field** must include: *"Accepted for publication
     in IEEE Transactions on Power Electronics. © 2026 IEEE.
     DOI: 10.1109/TPEL.XXXX.XXXXXXX. NN pages, NN figures."*
   - **MSC classification:** 78A55 (Technical applications) +
     93B70 (Networked control)
5. **Step 4 — Upload files:** upload `paper_arxiv.tex`, `paper.bib`,
   all `.eps`/`.pdf` figures, any class files, and `IEEEtran.cls`
   if used (arXiv accepts it).
6. **Step 5 — Preview:** verify the compiled PDF on arXiv's
   preview page matches the local compile. The attribution
   header on page 1 should be visible.
7. **Step 6 — Submit.** arXiv assigns an ID (`arXiv:YYMM.NNNNN`)
   after a short moderator review (~1 business day for known
   categories).

## Phase 4 — After arXiv accepts the submission

1. **Verify the arXiv ID resolves:** open
   `https://arxiv.org/abs/YYMM.NNNNN` and confirm metadata is
   correct.
2. **Update Pulsim repo:**
   - `CITATION.cff`: add the arXiv `identifier` and `references`
     entry pointing to both the arXiv ID and the IEEE DOI.
   - `README.md`: add a citation block under "How to cite":
     ```markdown
     If you use Pulsim in academic work, please cite:

     - **Paper (preferred):** Gili, L. C. "..." *IEEE TPEL*,
       2026. DOI: 10.1109/TPEL.XXXX.XXXXXXX
       [arXiv:YYMM.NNNNN](https://arxiv.org/abs/YYMM.NNNNN)
     - **Software:** Gili, L. C. "Pulsim: An Open-Source
       Power-Electronics Simulator." *JOSS*, 2026.
       DOI: 10.21105/joss.XXXXX
     ```
3. **Submit the arXiv ID to Google Scholar** — usually
   auto-indexed within 48 hours, but you can manually flag it
   via Scholar Inbox.
4. **Update your Google Scholar profile** with the arXiv version
   first; merge with the IEEE Xplore record once that goes live.
5. **Tweet/post a link** — but link the arXiv URL, never the
   Xplore paywall. (arXiv URLs have higher click-through.)

## Timing rules of thumb

- Time from IEEE Accept → arXiv post: aim for **≤ 14 days**.
  Faster is better; Google Scholar starts indexing arXiv preprints
  immediately, so an early arXiv post means an earlier citation
  window.
- Time from arXiv post → IEEE Xplore publication: typically 6–10
  weeks (typesetting + queue). During this gap your arXiv preprint
  is the only public version — that's the whole point.

## Don'ts

- ❌ Do **not** upload the IEEE-typeset PDF (the one IEEE returns
     to you for proofreading). That violates the copyright
     transfer.
- ❌ Do **not** submit to arXiv *before* IEEE acceptance (well,
     you *can* submit the pre-print, but it muddies the citation
     story — choose one strategy and stick to it; for Pulsim we
     post the **post-print only**).
- ❌ Do **not** forget the IEEE attribution header — this is what
     keeps the arXiv post in compliance with IEEE policy.
- ❌ Do **not** apply a CC-BY license to the arXiv version —
     IEEE owns the copyright; only the non-exclusive distribution
     license is permitted.
