# Docs Versioning and Release

PulsimCore documentation is published to GitHub Pages using **MkDocs Material + mike**.

## Publish Model

- Pull requests: strict docs build only (no deploy).
- Push to `main`: deploy documentation snapshot as `dev`.
- Push of release tags `vX.Y.Z`: deploy versioned docs (`X.Y.Z`) and update `latest`.

This gives fast iteration on `main` and stable, versioned release docs.

## Manual Release Procedure

1. Update project version metadata.
2. Commit code + docs changes.
3. Create and push a semantic tag.

```bash
python scripts/update_version.py 0.5.2
git tag v0.5.2
git push origin main --tags
```

## Workflow Guarantees

- `mkdocs build --strict` runs before deploy.
- invalid tag formats are rejected.
- version selector remains available in UI through `mike`.

## GitHub Pages Setup

In repository settings:

1. Open **Settings -> Pages**.
2. Set **Source** to **GitHub Actions**.
3. Keep `gh-pages` branch managed only by docs workflow/mike.

## Rollback Strategy

If documentation quality regresses:

1. fix docs in a patch commit;
2. publish a new patch tag (`vX.Y.(Z+1)`).

Avoid force-reusing an existing published tag.
