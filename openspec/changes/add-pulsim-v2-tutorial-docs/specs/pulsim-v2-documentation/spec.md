## ADDED Requirements

### Requirement: Tutorial-Style User Documentation

The repository SHALL contain a `docs/v2/` directory with narrative tutorials covering installation, mental model, and at least 6 walk-throughs of the existing YAML showcases.

#### Scenario: Getting-started page exists
- **WHEN** a new user opens `docs/v2/getting-started.md`
- **THEN** the page SHALL guide them through building Pulsim, installing the Python wheel, and running the first transient
- **AND** SHALL include a complete copy-paste working code block

#### Scenario: All 12 showcases have a matching tutorial or reference
- **WHEN** a user looks for documentation of a specific showcase YAML (e.g. `boost_realistic_igbt.yaml`)
- **THEN** they SHALL find either a tutorial chapter or an api-reference entry that explains how the YAML was constructed and what behaviour to expect

### Requirement: CI Docs Build

The CI pipeline SHALL include a job that builds the v2 documentation on every push to `main`.

#### Scenario: Broken docs fails CI
- **WHEN** a contributor pushes a commit that breaks the docs build (e.g. malformed Markdown, broken cross-link)
- **THEN** CI SHALL fail and block the merge
