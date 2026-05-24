# Contributing to Pulsim

First — thanks for your interest. Pulsim is a single-maintainer
open-source power-electronics simulator, and contributions of any
size (bug reports, documentation fixes, new device models, full
features) are welcome.

## Quick links

- 🐛 **Found a bug?** → [Open an issue](https://github.com/lgili/Pulsim/issues/new)
- 💡 **Have a feature idea?** → [Open a discussion](https://github.com/lgili/Pulsim/discussions/new) first; we'll convert to an issue if it makes sense
- 🛠 **Want to send a PR?** → See [Code contributions](#code-contributions) below
- 📚 **Documentation issues?** → PR directly against `docs/`; the [Docs CI](https://github.com/lgili/Pulsim/actions/workflows/docs.yml) will validate
- ❓ **Just have a question?** → [Discussions](https://github.com/lgili/Pulsim/discussions) is the right venue

## Code of conduct

Be kind and constructive. Disagreements about technical decisions are
welcome and expected; personal attacks, harassment, or any form of
discrimination are not. Reports go to **luizcarlosgili@gmail.com**.

## Reporting bugs

Please include:

1. **Pulsim version**: `python -c "import pulsim; print(pulsim.__version__)"`
2. **OS + Python version**
3. **Minimal reproducer**: the *smallest* `CircuitBuilder` script that
   reproduces the bug (ideally < 30 lines)
4. **Expected vs actual behaviour**
5. **Full traceback** if there's an exception
6. **Pulsim output** if simulation produces wrong numbers, including
   the values that look wrong vs what you expected

A good bug report saves hours of back-and-forth. If you can isolate
the bug to a specific module (e.g. "the PWL cache enumeration
crashes when N > 18"), say so — but if you can't, a clean reproducer
is enough.

## Proposing a feature

For anything bigger than a one-line fix, please open a
**Discussion** first to align on the design. This avoids the
"I spent 3 weekends on this PR and it's the wrong approach"
problem.

Useful structure for a feature proposal:

* **Problem**: what use case is currently impossible / awkward?
* **Proposed API**: 5-10 line code sketch of what it would look like
* **Alternatives considered**: at least 2, with their trade-offs
* **Implementation outline**: which Pulsim layers (Layer 0..8 — see
  [`docs/internals/README.md`](docs/internals/README.md)) you'd
  touch

## Code contributions

### 1. Fork + branch

```bash
git clone https://github.com/<your-username>/Pulsim.git
cd Pulsim
git checkout -b feat/short-description-of-change
```

Branch naming convention: `feat/...`, `fix/...`, `docs/...`,
`refactor/...`, `chore/...`.

### 2. Build the kernel + Python extension

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"
```

Verify everything still works:
```bash
ctest --test-dir build --output-on-failure
python -m pytest python/tests
```

### 3. Code style

- **C++** (kernel): formatted by `clang-format` using the rules in
  [`.clang-format`](.clang-format). Run `clang-format -i` on any
  files you touched.
- **C++** (lint): `clang-tidy` configured in
  [`.clang-tidy`](.clang-tidy). The CI runs `clang-tidy` on the
  diff; fix any warnings it raises.
- **Python**: standard PEP 8. We use 4-space indentation, type
  hints on public API, and we keep modules organised by topic
  (`pulsim.topology`, `pulsim.plot`, etc.). No formatter is
  enforced yet; just match the style of the surrounding code.
- **Documentation**: Markdown for narrative docs. We use
  [MkDocs](https://www.mkdocs.org/) with the strict mode
  (`mkdocs build --strict`) — all internal links must resolve.

### 4. Tests

- **New device model**: add a Catch2 test under
  `core/tests/test_<your_device>.cpp`. At minimum: build the
  device, run a 1 ms transient, check the steady-state output
  against an analytical formula.
- **New analysis driver**: add a Python test under
  `python/tests/test_<your_analysis>.py`. Use the existing
  buck/boost validation as templates.
- **Bug fix**: add a regression test that fails on `main` and
  passes on your branch.

CI will run the full matrix (Linux + macOS, Clang + GCC, Python
3.10–3.13). PRs cannot merge until all CI jobs are green.

### 5. Commit message

[Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>(<scope>): <one-line summary, imperative mood, lowercase>

<longer explanation — what + WHY, not how>

<footer with refs to issues or other commits if applicable>
```

Examples that already shipped:

- `feat(projects): 3-phase NPC 3-level inverter`
- `fix(docs): repair 2 broken cross-refs in internals/README.md`
- `chore(joss): prepare for JOSS submission`

### 6. Open the PR

PR template autopopulates with a summary + test plan checklist.
Fill both in. If the PR is large (>500 lines), split it into
multiple smaller PRs that each stand on their own.

Reviewers will respond within 7 days; we ask the same of authors
during review iteration.

## Documentation contributions

The `docs/` folder ships with the codebase. Layout:

* `docs/getting-started.md`, `docs/mental-model.md`,
  `docs/api-reference.md` — user-facing
* `docs/internals/` — kernel architecture (Layers 0–8)
* `docs/tutorials/` — narrative walk-throughs

A PR fixing a typo or clarifying a paragraph is *always* a good
contribution. The high bar is for the kernel; the docs welcome
incremental polish.

## Releasing a new version

(Maintainer-only.) Versioning follows [SemVer](https://semver.org).

```bash
# Bump version in pyproject.toml + python/pulsim/__init__.py
# Add CHANGELOG.md entry under the new version
git commit -am "chore(release): vX.Y.Z"
git tag -a vX.Y.Z -m "Pulsim vX.Y.Z"
git push origin main vX.Y.Z
# GitHub Actions handles PyPI publish + gh-pages deploy automatically
```

## Where to learn more

- Architecture: [`docs/internals/README.md`](docs/internals/README.md)
- Mental model (for new contributors): [`docs/mental-model.md`](docs/mental-model.md)
- API reference: [`docs/api-reference.md`](docs/api-reference.md)
- Roadmap: [`ROADMAP.md`](ROADMAP.md)

## Thank you

Every PR, issue, and discussion improves Pulsim. If you want to
contribute but don't know where to start, look for issues labelled
[`good first issue`](https://github.com/lgili/Pulsim/labels/good%20first%20issue).
