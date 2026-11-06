## 0. Preconditions

- [ ] 0.1 Rebase the implementation branch onto the merge of
      `fix/v2-diode-record` (the per-sample switch-state record). The
      guide's charge-balance example (4.4) asserts closure for
      `V_th > 0` diodes and must fail without it. Do not weaken the
      example to pass before the merge.
- [ ] 0.2 Re-run the measurements quoted in `design.md` on the rebased
      tree and update any number that moved:
      - diode `V(a)` at 10 A;
      - the mask indices Q1 / Q1_body / D1 / Q2 / Q2_body;
      - the floating-DC-side singular abort;
      - the snippet runtimes.

## 1. Binding signature hygiene (`python/bindings.cpp`, bindings only)

- [ ] 1.1 Forward-declare `py::class_<topology::Graph>`,
      `py::class_<pwl::DevicePool>`, `py::enum_<DevicePool::StoredKind>`,
      `py::class_<SimulationResult>` and `py::class_<SolverSnapshot>`
      before the members that return them. The members are
      `CircuitBuilder.graph`, `CircuitBuilder.pool`,
      `DevicePool.kind_of`, `find_implausible_voltage` and
      `SimulationResult.final_snapshot`. Define the methods afterwards.
- [ ] 1.2 Replace `py::arg("options") = pwl::PreflightOptions{}`
      (`run_preflight`) and the `InputRef::from_const(0.0)` defaults
      (`CxxBlockChain.add_sum`, `add_inverse_clarke`) with `py::arg_v`
      carrying a readable description.
- [ ] 1.3 Add `py::kw_only()` before the first required argument that
      follows a defaulted one in `CxxBlockChain.add_sum`,
      `add_inverse_clarke`, `add_pwm_generator`,
      `add_flux_mras_observer` and `add_sliding_mode_observer`. Grep the
      repo for positional callers (expected: none) and add a CHANGELOG
      line.
- [ ] 1.4 Type every `= None` default on a callable or object parameter
      as `std::optional<…>` or `py::object`, so the rendered signature
      is `Optional[...]`. Mypy's implicit-Optional errors on
      `run_transient`, `run_transient_bdf1` and their siblings must be
      gone.
- [ ] 1.5 Reword the `add_mosfet` docstring. `V_F` is the body diode's
      commutation threshold; the on-state is `1/g_on` with no offset.
- [ ] 1.6 Rebuild. Run ctest in `.scratch/build-ctest` and the full
      Python suite.

## 2. Stub generation and shipping

- [ ] 2.1 Add `scripts/generate_stubs.py`. It:
      - runs the pinned `pybind11-stubgen` on `pulsim._pulsim` with
        `--exit-code --numpy-array-use-type-var` and
        `--enum-class-locations` for `DCStrategy` and
        `ParametricRefactorMode`;
      - rewrites `pybind11_stubgen.typing_ext.FixedSize(N)` annotations
        to plain `list[int]` and drops that import;
      - merges `scripts/stubs_overlay.pyi`;
      - writes `python/pulsim/_pulsim.pyi`.

      `--check` regenerates the stub into a temporary directory and
      exits 1 with a unified diff.
- [ ] 2.2 Write `scripts/stubs_overlay.pyi`. It declares the members the
      Python layer attaches: `SimulationResult.v`, `.i`, `.power`,
      `.engine_used`, `.engine_route_reason`, and any other attribute
      that `python/pulsim/__init__.py` assigns onto a native class. Find
      them with
      `grep -n "^[A-Z][A-Za-z]*\.[a-z_]* = " python/pulsim/__init__.py`
      and the `res.<attr> =` sites.
- [ ] 2.3 Generate and commit `python/pulsim/_pulsim.pyi`, and add an
      empty `python/pulsim/py.typed`.
- [ ] 2.4 Add `*.md` and `py.typed` to the build-tree glob in
      `python/CMakeLists.txt`. Then confirm that
      `PYTHONPATH=build/python python -c "import pulsim;
      pulsim.ai_guide()"` works from a plain CMake build.
- [ ] 2.5 Build one wheel (`pip wheel . -w .scratch/wheel`) and confirm
      that `unzip -l` lists `pulsim/py.typed`, `pulsim/_pulsim.pyi` and
      `pulsim/AI_GUIDE.md`. If any is missing, add an `install(FILES …)`
      rule. Delete the wheel tree afterwards.
- [ ] 2.6 Add pinned `pybind11-stubgen` and `mypy` to the `dev` extra in
      `pyproject.toml`.

## 3. Stub tests (`python/tests/test_stubs.py`)

- [ ] 3.1 Structural test: the stub passes `ast.parse`, and contains no
      `pybind11_stubgen`, no `0x[0-9a-f]+` and no `::`.
- [ ] 3.2 Coverage test: every public name of `pulsim._pulsim`, and every
      public member of its classes, including the overlay members, is
      declared in the stub.
- [ ] 3.3 Signature test: for each native callable, parse the first line
      of `__doc__` (handling the "Overloaded function." form) and
      compare it with the stub `def`. Compare parameter names, their
      order, and whether each has a default. The failure message names
      the symbol and `python scripts/generate_stubs.py`.
- [ ] 3.4 Prove 3.2 and 3.3 non-vacuous. Feed each test an in-memory
      copy of the stub with one method deleted, and one with two
      parameters swapped. Each must fail.
- [ ] 3.5 mypy test, skipped without mypy:
      - the stub alone checks clean with `--follow-imports=silent`;
      - the concatenated guide and getting-started snippets report no
        error on a pulsim symbol;
      - the negative control
        `add_sine_voltage_source(..., amplitude=10.0, frequency=50.0)`
        is rejected.

## 4. Usage guide

- [ ] 4.1 Add `python/pulsim/_ai_guide.py` with `ai_guide() -> str`, which
      reads `AI_GUIDE.md` via `importlib.resources` and prepends
      `pulsim <version> — <path>`. Add `python/pulsim/guide.py`
      containing only the `__main__` entry, with a `--path` flag.
      Export `ai_guide` from `pulsim/__init__.py` and list it in
      `__all__`. Do not import `pulsim.guide` there.
- [ ] 4.2 Extend both messages of the module-level `__getattr__` in
      `python/pulsim/__init__.py` to name `python -m pulsim.guide`.
- [ ] 4.3 Write `python/pulsim/AI_GUIDE.md` per design D2 and D3. Verify
      every API claim against the code before writing it. In
      particular, check the 1 GΩ tie in `simulate()`, the `c0=` / `i0=`
      ICs, `pulsim.stats.__all__`, and
      `add_bridge_rectifier(b, name, *, ac_a, ac_b, dc_pos, dc_neg)`.
- [ ] 4.4 Guide charge-balance example: a single-phase bridge rectifier
      with `V_th > 0` diodes and the DC rail referenced. It asserts that
      the time-integrated source charge equals the summed diode branch
      charge within a stated tolerance. A second example compares
      `dt=`-fixed and `auto` runs with `pulsim.stats.time_average`.
- [ ] 4.5 Guide diode-threshold example: forced 10 A with `g_on = 20 S`,
      printing `V(a)` for `V_th = 0` and `0.7`, with an expected-output
      fence.
- [ ] 4.6 Add the "paste into your project's assistant config" snippet
      (D2.7) as a `text` fence.
- [ ] 4.7 `python/tests/test_ai_guide.py` covers:
      - the accessor header and body;
      - `python -W error -m pulsim.guide`, run via `subprocess` and
        `sys.executable`, exits 0 and its stdout equals `ai_guide()`;
      - `--path` points to an existing file;
      - the size is at most 32 KiB;
      - the names in `p.<name>`, `pulsim.<name>` and
        `CircuitBuilder.<method>` all resolve;
      - no v1 symbol appears in a fence;
      - the `__getattr__` messages contain `python -m pulsim.guide`.

## 5. In-package text consistency

- [ ] 5.1 `python/pulsim/topology.py`: reword `V_f` in
      `add_three_phase_vsi` (line 273) and `V_th` in
      `add_bridge_rectifier` (line 167) as commutation thresholds. Apply
      the same wording to the `V_th` parameter of `add_buck`,
      `add_boost` and `add_flyback` (defaults at lines 502, 611 and
      680).
- [ ] 5.2 `python/pulsim/discovery.py:55`: replace the "small (1µΩ)
      values to tie floating subnets" advice with a large-resistor
      reference.
- [ ] 5.3 Add a test asserting that no shipped docstring or catalogue
      entry contains "forward drop" next to a PWL-diode threshold
      parameter, or "µΩ" next to "floating".

## 6. Executable documentation (`python/tests/test_doc_snippets.py`)

- [ ] 6.1 Implement fence collection, the per-file shared namespace,
      `tmp_path` as working directory, `MPLBACKEND=Agg`, and `file:line`
      failure reports.
- [ ] 6.2 Implement the expected-output comparison with numeric
      tolerance, per design D5.
- [ ] 6.3 Implement the `doctest:` annotations (`requires=`, `skip
      reason=`, `no-output-check`). An unknown annotation fails, and so
      does a skip without a reason.
- [ ] 6.4 Register the `doc_snippets` marker in `pyproject.toml`
      (`[tool.pytest.ini_options] markers`).
- [ ] 6.5 Annotate the matplotlib-dependent blocks: README `p.scope`,
      and the getting-started `p.scope` and `plt.plot` blocks.
- [ ] 6.6 Self-tests of the runner on inline markdown fixtures:
      - a broken call fails with the right line;
      - a stale expected value fails;
      - a last-digit difference passes;
      - a skip without a reason fails.
- [ ] 6.7 Measure the total runtime of the module, record it in the PR,
      and add the 30 s warning and 60 s failure budget check.

## 7. llms.txt and docs site

- [ ] 7.1 Write `docs/llms.txt`: an H1, a blockquote summary that
      directs agents to `python -m pulsim.guide`, and the sections
      "Start here" (guide, getting-started, mental model), "Tutorials",
      "Reference" (api-reference, gotchas, helpers) and "Optional"
      (how-pulsim-works, internals).
- [ ] 7.2 Add `docs/ai-guide.md`, which includes `python/pulsim/AI_GUIDE.md`
      via `--8<--`, and add it to `mkdocs.yml` `nav` under User Guide.
- [ ] 7.3 Add a test for the `llms.txt` structure and that its link
      targets exist.
- [ ] 7.4 Run `mkdocs build --strict` locally in a scratch venv with
      `docs/requirements.txt`. Confirm that `site/llms.txt` exists and
      that the guide page renders.

## 8. CI and release

- [ ] 8.1 `.github/workflows/ci.yml`: add a job, or extend the Linux 3.12
      leg, that installs `.[dev]` plus matplotlib. It runs
      `python scripts/generate_stubs.py --check` and the full pytest, so
      the `requires=matplotlib` blocks execute.
- [ ] 8.2 `.github/workflows/publish.yml`: extend `CIBW_TEST_COMMAND` to
      run `python -m pulsim.guide --path` and to assert that `py.typed`
      and `_pulsim.pyi` exist in the installed package.
- [ ] 8.3 Add a CHANGELOG `[Unreleased]` entry covering the stubs, the
      guide and CLI, `llms.txt`, executed docs and the `kw_only` note.
      Add a README pointer to `python -m pulsim.guide`.

## 9. Validation

- [ ] 9.1 Run `openspec validate add-ai-usage-guide --strict`.
- [ ] 9.2 Run the full Python suite: 0 failed (`--timeout=600`).
- [ ] 9.3 Run `ruff check python/ scripts/generate_stubs.py`: clean.
- [ ] 9.4 Run ctest in `.scratch/build-ctest` after the binding changes:
      green.
- [ ] 9.5 Delete the scratch venvs and wheel trees.
