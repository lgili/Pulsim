## Context

AI assistants writing Pulsim code learn the API from four places: their
training data (stale), the docs they are pointed at, the signatures their
tooling can introspect, and the docstrings they read. This change fixes
the last three. The decisions below were measured on branch
`feat/v2-ai-guide` (based on `fix/v2-polish-bucket-a`) on 2026-11-06, with
Python 3.13, pybind11-stubgen 3.0.0 and mypy 2.4.1.

## Goals / Non-Goals

- Goals
  - A signature-accurate, type-checkable `.pyi` for `pulsim._pulsim`
    that ships in the wheel and cannot drift from the binary.
  - One short guide, matched to the installed version, that teaches the
    2.x idioms and the traps. It is reachable from the installed package
    with no network access.
  - Every `python` block in the user-facing docs executes in CI, so a
    broken call fails a test instead of reaching a user.
- Non-Goals
  - Typing the pure-Python layer to mypy-strict. `py.typed` exposes its
    existing annotations. Today mypy reports about 220 errors inside the
    package itself. Users do not see them because mypy reports only the
    user's own code, but cleaning them is separate work.
  - Executing `docs/api-reference.md` and `docs/gotchas.md`. Their
    `python` fences are signature listings and fragments (`SyntaxError`
    at api-reference.md:7, `NameError: b` at gotchas.md:17). The
    mechanism supports opting them in later.
  - Doctest of Python docstrings.
  - Editor-specific rule files (`.cursorrules`, Copilot instructions).
    The guide gives a one-line snippet users paste into their own
    project's assistant config.

## Decisions

### D1. Accessor and CLI naming

- `pulsim.ai_guide() -> str` returns the guide text. `ai_guide`,
  `guide`, `usage_guide` and `help` are all absent from `dir(pulsim)`
  today (339 names checked). `ai_guide` is chosen because an agent
  listing the namespace can find it.
- The function lives in the private module `pulsim/_ai_guide.py`. The
  CLI module is `pulsim/guide.py` and holds only the
  `if __name__ == "__main__"` entry. `pulsim/__init__.py` imports
  `ai_guide` from `_ai_guide` and does NOT import `guide`. Reasons:
  - Importing `pulsim.guide` eagerly would make `python -m pulsim.guide`
    emit runpy's "found in sys.modules … prior to execution"
    `RuntimeWarning`.
  - Binding a submodule and a function under one name is the bug that
    `pulsim.scope` already had to work around
    (`python/pulsim/__init__.py:449-460`).
- The text is read with
  `importlib.resources.files("pulsim").joinpath("AI_GUIDE.md")`.
  `ai_guide()` prepends one header line,
  `pulsim <__version__> — <path>`, so the file itself never carries a
  version number that has to be bumped by hand.
- `python -m pulsim.guide --path` prints the file path only, so an agent
  can open the file directly instead of reading stdout.

### D2. What the guide contains

The guide is prescriptive: what to call and what to check. It is not a
tour. Order:

1. **Ask the installed package first.** `python -m pulsim.guide`; the
   first line of `help(p.CircuitBuilder.add_x)` is the real signature;
   the stubs; `p.catalog()` / `p.example()`; `NameNotFoundError`
   suggestions; the v1-name hints raised by `pulsim.__getattr__`.
2. **The minimal correct script.** Builder, `p.simulate`,
   `res.v(name)` / `res.i(name)`, `res.engine_used`. Run as a snippet.
3. **Signatures that are commonly guessed wrong.** Argument order for the
   sine, pulse and PWM sources, `add_diode`, `add_mosfet`, and the
   keyword-only topology helpers (`add_bridge_rectifier(b, name, *,
   ac_a=…)`). Each one is an executed call, not prose.
4. **Semantic traps.** One section each, with the wrong idiom, the
   right idiom, and why:
   - PWL diode `V_th`, `add_bridge_rectifier(V_th=…)` and
     `add_three_phase_vsi(V_f=…)` are commutation thresholds. The
     on-state is a pure `g_on` with no offset. Re-measured here: at a
     forced 10 A with `g_on = 20 S`, `V(a) = 0.500 V` for `V_th = 0` and
     for `V_th = 0.7` alike.
   - `engine='auto'` is variable-step. Use `pulsim.stats`
     (`time_average`, `time_rms`, `time_window`, `steady_state_stats`)
     or `np.trapezoid`, never `.mean()`.
   - `SwitchStateMask(n)` is a SIZE, with every bit off. Set bits with
     `.set(i, True)`. Take indices from `switch_index_of(name)` and the
     size from `graph.num_switches`. `add_mosfet` adds `<name>_body`, so
     indices interleave. Re-measured: Q1, Q1_body, D1, Q2, Q2_body map
     to 0, 1, 2, 3, 4.
   - `tol_newton_dx` / `tol_newton_res` are absolute. Do not loosen them
     to get past a Newton abort. A scaled step test was measured to
     return 3.53 V where the fixed engine gives 15.31 V, with no error.
     Fix the circuit (gate edges, body diode, consistent start) instead.
   - `p.scope(builder, result, ...)` is the plotting function. Get
     `LiveScope` with `from pulsim.scope import LiveScope`.
   - `auto_regularize` ties floating subnets with 1 GΩ. Reference an
     isolated secondary with a LARGE resistor, never a micro-ohm bond.
   - A floating DC side with every diode off can make `engine='auto'`
     abort as singular at its first tiny step. Re-measured: a bridge
     rectifier with a floating DC side raised
     `factorize failed (numerically singular) … dt=4e-14 … node dcn`,
     and the same circuit with `dcn` tied to ground through 1 MΩ ran on
     `trbdf2`. Ground or reference one rail.
   - The fixed engine starts first order from an inconsistent state.
     Give ICs (`c0=`, `i0=`) or discard the first samples.
5. **Verify a result physically.** Run as snippets:
   - Charge or energy balance from independent branch currents. For
     example, the capacitor charge `∫ i_C dt` against `C·ΔV`, and the
     source current against the sum of the device currents.
   - Two engines (`dt=` fixed versus `auto`), compared with
     time-weighted statistics.
   - The rule that 0 A, ±1e7 V or a perfectly flat trace is a setup
     error until proven otherwise.
6. **When the result is refused.** Pulsim 2 refuses rather than guesses.
   Do not wrap `simulate` in `except Exception`, and do not retry with
   loosened tolerances. Read the refusal text.
7. **Pasting into your project's assistant config.** A three-line
   snippet for `CLAUDE.md` / `AGENTS.md` / Copilot instructions: "Before
   writing pulsim code, run `python -m pulsim.guide` and follow it."

Size: at most 32 KiB, enforced by a test. That is about 8k tokens, small
enough for an agent to load whole on every task.

### D3. What the guide must NOT contain

- Calls that are not executed. Every `python` fence runs (D5). Inline
  references written as `p.<name>`, `pulsim.<name>` or
  `CircuitBuilder.<method>` are resolved against the installed package
  by a test.
- Performance figures, benchmarks, roadmap, release history, kernel
  internals. These go stale and do not change how to call the API.
  Link to the docs instead.
- Any 1.x API, even as a "do not use" example in code. A model copies
  the code, not the warning. Wrong idioms appear only as inline text
  that the D3 resolver does not match (for example, `.mean()` over
  samples).
- Workarounds for open defects. In particular, nothing compensates for
  the PWL-diode current under-report. The guide teaches the charge
  balance that exposes it and states the rule: if the balance does not
  close, the result is not to be trusted, and the installed pulsim
  should be updated or the defect reported. That sentence stays true
  after the fix ships.
- Numbers that the executed snippets do not reproduce.

### D4. How the stubs stay fresh

Two layers, because pybind11-stubgen output depends on the stubgen and
pybind11 versions:

1. **`python/tests/test_stubs.py`.** It is dependency-free and runs on
   every CI leg and platform. It checks:
   - `ast.parse` of the shipped stub succeeds.
   - The stub contains no `pybind11_stubgen` import, no `0x[0-9a-f]+`
     address and no `::`.
   - Every public class and function in `pulsim._pulsim`, and every
     public method or property on those classes, is declared in the
     stub. Python-attached members are included: `SimulationResult.v`,
     `.i`, `.power`, `.engine_used` and `.engine_route_reason`.
   - For each pybind callable, the parameter names, their order and
     whether each has a default all match between the first line of
     `__doc__` and the stub `def`. Overloads are compared as a set.

   A failure names the drifted symbol and the command
   `python scripts/generate_stubs.py`.
2. **`scripts/generate_stubs.py --check`.** It runs in one CI job (Linux,
   Python 3.12) with pinned `pybind11-stubgen`. It regenerates the stub
   and fails on any diff. The script:
   - runs stubgen with `--numpy-array-use-type-var` and
     `--enum-class-locations` for `DCStrategy` and
     `ParametricRefactorMode`;
   - rewrites `pybind11_stubgen.typing_ext.FixedSize(3)` to plain
     `list[int]`;
   - merges `scripts/stubs_overlay.pyi`, which declares the members the
     Python layer attaches;
   - fails if stubgen reports any error, using `--exit-code`.

   Measured: stubgen takes 0.8 s and writes a 2048-line stub with 437
   `def`s. With default flags it reports 36 errors. With the flags above
   it reports 8:
   - 5 raw C++ type names, from properties whose class was registered
     later (`CircuitBuilder` at `bindings.cpp:320` before `Graph` at
     `:1208` and `DevicePool` at `:1282`);
   - 3 `0x…` default reprs (`run_preflight`, `add_sum`,
     `add_inverse_clarke`).

   Five `CxxBlockChain` signatures place a required argument after a
   defaulted one, which is a `SyntaxError` in a stub. After a trial
   post-process, mypy still reports about 25 errors in the stub. 14 of
   them are implicit-`Optional` defaults (`= None` on a `Callable`).
   All of these are fixed at the source, in the bindings, not by text
   substitution: `py::class_` forward declarations, `py::kw_only()`,
   `py::arg_v(..., "PreflightOptions()")`, and `std::optional`
   parameters.

   Until those binding fixes (tasks 1.1-1.4) land, the script rewrites
   exactly these constructs deterministically, one documented rule
   each: a C++ name becomes the bound class of the same name (and fails
   if there is none), an `<... object at 0x...>` default becomes `...`,
   `*` is inserted before the first required-after-optional parameter
   (the contract `py::kw_only()` will give), and a `= None` default
   gains `| None`. stubgen tolerates only the two known invalid-
   expression patterns, so any NEW raw C++ type still fails
   `--exit-code`. Once the bindings are fixed the rules match nothing
   and are deleted. Three further rules are permanent: `FixedSize`
   annotations become `list[int]` (no generator import in the shipped
   stub), the `__eq__`/`__ne__` operand is typed `object` (pybind
   returns `NotImplemented` for a foreign operand; mypy otherwise
   reports a Liskov violation per class), and an array-shape TypeVar in
   a class attribute (`SolverSnapshot.x`) becomes `int`, since it is
   unbound there.
3. A third check confirms that the stub helps rather than only parses.
   `mypy --follow-imports=silent` runs over the concatenated snippets
   of the guide and of getting-started, and reports no error that
   points to a pulsim symbol. In the trial it reported
   `"SimulationResult" has no attribute "engine_used"`, which is the
   overlay's job. A negative control must also fail:
   `add_sine_voltage_source("V", "a", "gnd", amplitude=10.0,
   frequency=50.0)` must be rejected. The trial did reject it, with
   "did you mean `v_amplitude`?". The check is skipped when mypy is not
   installed, and it runs in the stub-check CI job.

### D5. How snippets are executed

- **Collection.** Fences whose info string is exactly `python`, in
  `python/pulsim/AI_GUIDE.md`, `README.md`, `docs/getting-started.md`
  and `docs/tutorials/*.md`. `bash`, `text`, `pycon`, `yaml` and bare
  fences never execute. There is one pytest item per file.
- **Execution model.**
  - The blocks of one file run in document order in one shared
    namespace, which is how a reader runs a tutorial.
  - The working directory is `tmp_path`, so `save=` writes nothing into
    the repo.
  - `MPLBACKEND=Agg` is set before anything imports matplotlib.
  - A failure reports `file:line` of the failing fence and the
    traceback.
- **Output check.** The block's captured stdout is checked when the
  python fence is followed by a bare or `text` fence. Between the two,
  only blank lines or a single prose line ending in `:` (for example
  "Expected output:") are allowed. Comparison rules:
  - Whitespace-normalised.
  - Non-numeric tokens must match exactly.
  - A numeric token passes within one unit of its last printed digit,
    or within a relative 1e-6 if that is larger.

  The current docs already satisfy this exactly on macOS: 9 of 9 output
  blocks matched byte for byte.
- **Annotations.** An HTML comment directly above a fence. It is
  invisible on GitHub and in MkDocs.
  - `<!-- doctest: requires=matplotlib -->` skips the block, with a
    reason, when the module cannot be imported.
  - `<!-- doctest: skip reason="…" -->` requires a non-empty reason.
  - `<!-- doctest: no-output-check -->` keeps execution and drops the
    comparison.
- **CI.** The module is marked `doc_snippets`; the marker is registered
  in `pyproject.toml`. CI's `pytest python/tests` already collects it.
  The stub-check job installs matplotlib, so the plotting blocks run
  somewhere and are not skipped everywhere.
- **Runtime budget.** Measured 6.2 s for README, getting-started and
  the six tutorials, of which Tutorial 06 is 5.0 s. Budget: 30 s per CI
  leg for all executed docs, guide included, asserted by a session-level
  timing check that warns at 30 s and fails at 60 s.

### D6. llms.txt and the docs site

- `docs/llms.txt` follows the llms.txt convention: an H1, a blockquote
  summary, then `##` sections of markdown link lists ("Start here",
  "Tutorials", "Reference", "Optional").
- Its first link is the guide. Its summary tells agents to prefer
  `python -m pulsim.guide` for the installed version, because a docs
  URL may describe a different release.
- Links point at the raw markdown on GitHub at the tag for the docs
  version. The rendered HTML is worse input for a model.
- MkDocs copies non-markdown files from `docs/`, so the file is served
  at `https://lgili.github.io/Pulsim/<version>/llms.txt` under mike. A
  copy at the site root is left open (see Open Questions).
- `docs/ai-guide.md` includes `python/pulsim/AI_GUIDE.md` with
  `pymdownx.snippets` (already enabled; MkDocs runs from the repo root).
  The site page and the in-package file are therefore one source. It
  goes into `nav` under User Guide.
- `mkdocs build --strict` stays green. The docs workflow already
  triggers on `python/pulsim/**`.

### D7. Packaging

- `wheel.packages = ["python/pulsim"]` is how non-Python files such as
  `lib/data/*.yaml` reach the wheel today, since `lib/` has no
  `install()` rule. A task verifies on a built wheel that `AI_GUIDE.md`,
  `py.typed` and `_pulsim.pyi` are present.
- The build-tree copy in `python/CMakeLists.txt` uses an explicit
  extension list without `*.md` or `py.typed`. Without a change,
  `PYTHONPATH=build/python` would import a package whose `ai_guide()`
  raises `FileNotFoundError`. The glob gains both.
- The cibuildwheel smoke test (`CIBW_TEST_COMMAND`) gains
  `python -m pulsim.guide --path` and an assertion that `py.typed` and
  `_pulsim.pyi` sit next to `__init__.py`.

## Risks / Trade-offs

- `py.typed` makes type checkers trust the Python layer's annotations.
  Wrong annotations become false positives in user code.
  → Mitigation: the D4.3 mypy check over real snippets; fix the
  annotations the snippets hit.
- Exact-diff stub checks are version-sensitive.
  → Mitigation: the exact diff runs in one pinned job. The
  cross-platform guard is the structural test.
- Numeric output checks may differ in the last digit across platforms.
  → Mitigation: tolerance of one unit in the last place, plus a
  per-block opt-out that must carry a reason.
- `py::kw_only()` changes the call contract of five internal
  `CxxBlockChain` methods.
  → Mitigation: every repo caller already uses keywords. The CHANGELOG
  records it.
- The guide can drift from the docs.
  → Mitigation: the guide states rules and links to docs for depth. Its
  code is executed and its names are resolved.

## Migration Plan

Additive. Users gain stubs, the guide and the CLI. Release notes ask
users to add the D2.7 snippet to their own assistant config. No rollback
concerns beyond reverting the commit.

## Open Questions

- Should the docs deploy job also copy `llms.txt` to the gh-pages root
  (`/Pulsim/llms.txt`)? mike manages only versioned directories and the
  root redirect.
- Should the doc-snippet module also execute `examples/*.py`? They are
  scripts, not docs, and some take minutes.
