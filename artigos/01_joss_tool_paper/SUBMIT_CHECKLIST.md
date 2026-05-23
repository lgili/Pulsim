# JOSS Submission — Final User Checklist

**PR #28 already merged to main.** Pulsim v1.1.0 release notes,
`LICENSE`, `CITATION.cff`, and the paper draft are all live on
`origin/main`. Everything below is what only **you** can do.

Estimated total time: **30-45 minutes** if you're already logged in
to GitHub + Zenodo + JOSS.

---

## 1. Register an ORCID (5 min) — REQUIRED

> ORCID is a persistent author identifier. JOSS won't submit without
> one, and pretty much every other journal you target later
> (TPEL, JESTPE, OJ-PEL) requires it too. One-time cost forever.

1. Go to **https://orcid.org/register**
2. Sign up with your email (you can connect Google/ORCID with your
   GitHub email so the IDs link easily)
3. Copy the ORCID iD that gets issued — it looks like
   `0000-0002-1234-5678`
4. Update **two files**:
   - `CITATION.cff` at the repo root — uncomment the `orcid:` line
     and paste your ID
   - `artigos/01_joss_tool_paper/paper.md` — replace the
     `0000-0000-0000-0000` placeholder

Commit these 2 changes as a follow-up: e.g.

```bash
git checkout main && git pull
sed -i '' 's|0000-0000-0000-0000|0000-0002-XXXX-XXXX|' \
    CITATION.cff artigos/01_joss_tool_paper/paper.md
# (remove the leading `#` from the orcid: line in CITATION.cff manually)
git add CITATION.cff artigos/01_joss_tool_paper/paper.md
git commit -m "chore(joss): add author ORCID"
git push origin main
```

---

## 2. Tag the v1.1.0 release (2 min) — REQUIRED

JOSS reviewers check out **the exact commit** matching the submitted
version. The tag is the contract.

```bash
git checkout main && git pull       # make sure local main is current
git tag -a v1.1.0 -m "Pulsim v1.1.0 — JOSS submission release"
git push origin v1.1.0
```

Verify on GitHub at **https://github.com/lgili/Pulsim/releases/new**
— select the `v1.1.0` tag, click "Generate release notes", review
(GitHub auto-pulls the CHANGELOG entry), publish.

---

## 3. Connect to Zenodo (5 min) — RECOMMENDED but optional

Zenodo auto-archives every GitHub release with a DOI. JOSS doesn't
require this for submission, but having a Zenodo DOI makes your
software citable independently of GitHub (insurance against repo
disappearance).

1. Sign in at **https://zenodo.org/login** (use your GitHub
   credentials — it's the same account)
2. Go to **https://zenodo.org/account/settings/github/**
3. Flip the `lgili/Pulsim` toggle to **ON**
4. Back in GitHub, edit the v1.1.0 release you just published →
   click "Update release" (no changes needed; this triggers
   Zenodo's webhook)
5. Within ~2 minutes Zenodo will have an archive page like
   `https://zenodo.org/records/XXXXXXX` with a DOI like
   `10.5281/zenodo.XXXXXXX`

> If you do this, also add the Zenodo DOI badge to the repo `README.md`
> top — it makes the academic credibility immediately visible.

---

## 4. Submit to JOSS (15-20 min) — THE BIG ONE

Go to **https://joss.theoj.org/papers/new** and fill in the form:

| Field | Value |
|---|---|
| Repository URL | `https://github.com/lgili/Pulsim` |
| Software version | `v1.1.0` |
| Branch with paper | `main` |
| **Path to `paper.md`** | `artigos/01_joss_tool_paper/paper.md` ⚠️ |
| Suggested subject area | `Engineering` (Power Electronics) |
| Submission DOI | (leave blank — Zenodo DOI optional, fill if you connected it) |
| Are you the author? | Yes |
| Do all listed authors agree? | Yes (you're the only author) |

After submit:

1. Within ~24h you'll get a pre-review issue at
   **https://github.com/openjournals/joss-reviews/** —
   a JOSS editor checks the submission meets basic requirements
   (paper compiles, license OK, scope OK).
2. Within ~1-2 weeks the editor finds 2 reviewers and opens
   the **review issue**. Reviewers fill a public checklist and
   ask questions/file PRs against your repo. You respond on the
   thread.
3. When the review checklist is 100% green, paper is accepted →
   DOI minted by Crossref → paper page goes live at
   `joss.theoj.org/papers/10.21105/joss.XXXXX`.

**Median timeline: 12 weeks submit → publish.** (Some take 6 weeks,
some take 6 months — depends on reviewer availability and how many
iterations you need.)

---

## 5. Post-acceptance (later)

Once the JOSS paper is accepted:

1. **Update `CITATION.cff`** — uncomment the `preferred-citation:`
   block and fill in the JOSS DOI
2. **Update `README.md`** with a JOSS badge:
   ```markdown
   [![DOI](https://joss.theoj.org/papers/10.21105/joss.XXXXX/status.svg)](https://doi.org/10.21105/joss.XXXXX)
   ```
3. **Tag `v1.1.1`** as a "post-JOSS-acceptance" patch with the
   updated citation file (optional but tidy)
4. Start drafting paper #2 from `artigos/README.md`'s plan
   (EPE-ECCE Europe 2026 — full-paper deadline 8 June 2026)

---

## Quick verification — everything in place?

Run this checklist before clicking submit:

```bash
cd /path/to/Pulsim    # main worktree
git fetch origin && git log --oneline main -5
# Should show "Merge pull request #28" near the top

# Files JOSS will check:
ls LICENSE CITATION.cff README.md \
   artigos/01_joss_tool_paper/paper.md \
   artigos/01_joss_tool_paper/paper.bib

# Tag exists?
git tag -l v1.1.0

# Versions consistent?
grep version pyproject.toml
grep __version__ python/pulsim/__init__.py
grep '^version:' CITATION.cff
```

All 4 file outputs should exist; both versions should be `1.1.0`;
the tag should appear; the CITATION.cff version should match.

---

## What this gives you

Once accepted (~12 weeks from submit), you'll have:

- A **persistent DOI** every paper using Pulsim can cite
- **JOSS paper page** indexed in Scopus, Web of Science, DOAJ, ADS
  → Pulsim shows up in your Google Scholar profile
- A **strengthened repo** (the reviewers will file 5-10 PRs improving
  docs, CI, install ergonomics — they always do)
- **Permission to start paper #2** (EPE-ECCE Europe) with the JOSS
  DOI as the canonical Pulsim citation in its references
