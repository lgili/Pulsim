## 0. Preconditions

- [x] 0.1 (Done 2026-11-06 as a merge of `fix/v2-polish-bucket-a` @
      93020311, which contains `fix/v2-diode-record`: conflicts in
      CHANGELOG.md (both sections kept) and topology.py (the base's fuller
      V_th / V_f text kept). Both xfail blocks then passed, as designed;
      the annotations are gone and the checks assert. See 11.1.) Rebase the implementation branch onto the merge of
      `fix/v2-diode-record` (the per-sample switch-state record). The
      guide's charge-balance example (4.4) asserts closure for
      `V_th > 0` diodes and must fail without it. Do not weaken the
      example to pass before the merge.
- [x] 0.2 (Re-run 2026-11-06 on this branch, not rebased: diode V(a) 0.500 V for both V_th; mask indices 0..4 as stated; the floating-DC abort now reports dt=6e-14 and is NOT cured by a 1 MOhm or 1 GOhm rail tie on a fresh builder (design.md updated); snippets 4.9 s for nine files.) Re-run the measurements quoted in `design.md` on the rebased
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

- [x] 2.1 Add `scripts/generate_stubs.py`. It:
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
- [x] 2.2 Write `scripts/stubs_overlay.pyi`. It declares the members the
      Python layer attaches: `SimulationResult.v`, `.i`, `.power`,
      `.engine_used`, `.engine_route_reason`, and any other attribute
      that `python/pulsim/__init__.py` assigns onto a native class. Find
      them with
      `grep -n "^[A-Z][A-Za-z]*\.[a-z_]* = " python/pulsim/__init__.py`
      and the `res.<attr> =` sites.
- [x] 2.3 Generate and commit `python/pulsim/_pulsim.pyi`, and add an
      empty `python/pulsim/py.typed`.
- [x] 2.4 (Glob verified with `cmake -P` over the same patterns: the build-tree copy lists pulsim/AI_GUIDE.md, _pulsim.pyi, py.typed. This worktree's build/python/pulsim is a symlink to python/pulsim, through which `ai_guide()` works; no full CMake build was run.) (Globs added, build-tree and install rule; the `ai_guide()`
      confirmation waits for 4.1.) Add `*.md` and `py.typed` to the build-tree glob in
      `python/CMakeLists.txt`. Then confirm that
      `PYTHONPATH=build/python python -c "import pulsim;
      pulsim.ai_guide()"` works from a plain CMake build.
- [ ] 2.5 (Re-checked 2026-11-06 with scikit-build-core 0.11.6's own
      `packages_to_file_mapping` for `python/pulsim`: AI_GUIDE.md, guide.py,
      _ai_guide.py, py.typed and _pulsim.pyi are all mapped, none git-ignored.
      A real `pip wheel` in a scratch venv stalled in the yaml-cpp compile
      (compiler processes stopped, state T) and was killed; still to do.
      Earlier: pre-checked without a wheel build: scikit-build-core's
      `iter_package_files` for `wheel.packages = ["python/pulsim"]` lists
      `pulsim/py.typed` and `pulsim/_pulsim.pyi`; the wheel build itself
      waits for `AI_GUIDE.md`.) Build one wheel (`pip wheel . -w .scratch/wheel`) and confirm
      that `unzip -l` lists `pulsim/py.typed`, `pulsim/_pulsim.pyi` and
      `pulsim/AI_GUIDE.md`. If any is missing, add an `install(FILES …)`
      rule. Delete the wheel tree afterwards.
- [x] 2.6 Add pinned `pybind11-stubgen` and `mypy` to the `dev` extra in
      `pyproject.toml`.

## 3. Stub tests (`python/tests/test_stubs.py`)

- [x] 3.1 Structural test: the stub passes `ast.parse`, and contains no
      `pybind11_stubgen`, no `0x[0-9a-f]+` and no `::`.
- [x] 3.2 Coverage test: every public name of `pulsim._pulsim`, and every
      public member of its classes, including the overlay members, is
      declared in the stub.
- [x] 3.3 Signature test: for each native callable, parse the first line
      of `__doc__` (handling the "Overloaded function." form) and
      compare it with the stub `def`. Compare parameter names, their
      order, and whether each has a default. The failure message names
      the symbol and `python scripts/generate_stubs.py`.
- [x] 3.4 Prove 3.2 and 3.3 non-vacuous. Feed each test an in-memory
      copy of the stub with one method deleted, and one with two
      parameters swapped. Each must fail.
- [x] 3.5 mypy test, skipped without mypy:
      - the stub alone checks clean with `--follow-imports=silent`;
      - the concatenated guide and getting-started snippets report no
        error on a pulsim symbol;
      - the negative control
        `add_sine_voltage_source(..., amplitude=10.0, frequency=50.0)`
        is rejected.

## 4. Usage guide

- [x] 4.1 Add `python/pulsim/_ai_guide.py` with `ai_guide() -> str`, which
      reads `AI_GUIDE.md` via `importlib.resources` and prepends
      `pulsim <version> — <path>`. Add `python/pulsim/guide.py`
      containing only the `__main__` entry, with a `--path` flag.
      Export `ai_guide` from `pulsim/__init__.py` and list it in
      `__all__`. Do not import `pulsim.guide` there.
- [x] 4.2 Extend both messages of the module-level `__getattr__` in
      `python/pulsim/__init__.py` to name `python -m pulsim.guide`.
- [x] 4.3 Write `python/pulsim/AI_GUIDE.md` per design D2 and D3. Verify
      every API claim against the code before writing it. In
      particular, check the 1 GΩ tie in `simulate()`, the `c0=` / `i0=`
      ICs, `pulsim.stats.__all__`, and
      `add_bridge_rectifier(b, name, *, ac_a, ac_b, dc_pos, dc_neg)`.
- [x] 4.4 (The diode half of the balance is annotated `doctest: xfail` (strict, AssertionError only) until 0.1 lands; it fails today with diode charge 0.0000 C against 0.1994 C. The source-versus-load half closes and is asserted.) Guide charge-balance example: a single-phase bridge rectifier
      with `V_th > 0` diodes and the DC rail referenced. It asserts that
      the time-integrated source charge equals the summed diode branch
      charge within a stated tolerance. A second example compares
      `dt=`-fixed and `auto` runs with `pulsim.stats.time_average`.
- [x] 4.5 Guide diode-threshold example: forced 10 A with `g_on = 20 S`,
      printing `V(a)` for `V_th = 0` and `0.7`, with an expected-output
      fence.
- [x] 4.6 Add the "paste into your project's assistant config" snippet
      (D2.7) as a `text` fence.
- [x] 4.7 (Also: the quick-reference signatures are checked against the stub / inspect.signature, with mutation tests; the guide is ASCII.) `python/tests/test_ai_guide.py` covers:
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

- [x] 5.1 `python/pulsim/topology.py`: reword `V_f` in
      `add_three_phase_vsi` (line 273) and `V_th` in
      `add_bridge_rectifier` (line 167) as commutation thresholds. Apply
      the same wording to the `V_th` parameter of `add_buck`,
      `add_boost` and `add_flyback` (defaults at lines 502, 611 and
      680).
- [x] 5.2 `python/pulsim/discovery.py:55`: replace the "small (1µΩ)
      values to tie floating subnets" advice with a large-resistor
      reference.
- [x] 5.3 Add a test asserting that no shipped docstring or catalogue
      entry contains "forward drop" next to a PWL-diode threshold
      parameter, or "µΩ" next to "floating".

## 6. Executable documentation (`python/tests/test_doc_snippets.py`)

- [x] 6.1 Implement fence collection, the per-file shared namespace,
      `tmp_path` as working directory, `MPLBACKEND=Agg`, and `file:line`
      failure reports.
- [x] 6.2 Implement the expected-output comparison with numeric
      tolerance, per design D5.
- [x] 6.3 (Plus `xfail reason=` for a block that documents an open defect; see design D5.) Implement the `doctest:` annotations (`requires=`, `skip
      reason=`, `no-output-check`). An unknown annotation fails, and so
      does a skip without a reason.
- [x] 6.4 Register the `doc_snippets` marker in `pyproject.toml`
      (`[tool.pytest.ini_options] markers`).
- [x] 6.5 Annotate the matplotlib-dependent blocks: README `p.scope`,
      and the getting-started `p.scope` and `plt.plot` blocks.
- [x] 6.6 Self-tests of the runner on inline markdown fixtures:
      - a broken call fails with the right line;
      - a stale expected value fails;
      - a last-digit difference passes;
      - a skip without a reason fails.
- [x] 6.7 (4.9 s for all nine files on macOS / Apple silicon (Tutorial 06 3.6 s, the guide 0.6 s).) Measure the total runtime of the module, record it in the PR,
      and add the 30 s warning and 60 s failure budget check.

## 7. llms.txt and docs site

- [x] 7.1 (Links are raw GitHub URLs on `main`; the test maps each to a repository file.) Write `docs/llms.txt`: an H1, a blockquote summary that
      directs agents to `python -m pulsim.guide`, and the sections
      "Start here" (guide, getting-started, mental model), "Tutorials",
      "Reference" (api-reference, gotchas, helpers) and "Optional"
      (how-pulsim-works, internals).
- [x] 7.2 Add `docs/ai-guide.md`, which includes `python/pulsim/AI_GUIDE.md`
      via `--8<--`, and add it to `mkdocs.yml` `nav` under User Guide.
- [x] 7.3 Add a test for the `llms.txt` structure and that its link
      targets exist.
- [x] 7.4 (Built with the mkdocs / material / mike already installed in the framework Python (versions inside docs/requirements.txt ranges), into .scratch/site: strict build clean, site/llms.txt present, the guide page renders its H1.) Run `mkdocs build --strict` locally in a scratch venv with
      `docs/requirements.txt`. Confirm that `site/llms.txt` exists and
      that the guide page renders.

## 8. CI and release

- [ ] 8.1 (Partly done: the Linux / 3.12 leg of `python-tests` installs
      matplotlib, so the `requires=matplotlib` blocks run there. The
      pybind11 version is now pinned (11.2), so `--check` has one
      expected output; the step is still not added, because stubgen's
      output on Linux has not been compared with macOS here.)
      `.github/workflows/ci.yml`: add a job, or extend the Linux 3.12
      leg, that installs `.[dev]` plus matplotlib. It runs
      `python scripts/generate_stubs.py --check` and the full pytest, so
      the `requires=matplotlib` blocks execute.
- [x] 8.2 `.github/workflows/publish.yml`: extend `CIBW_TEST_COMMAND` to
      run `python -m pulsim.guide --path` and to assert that `py.typed`
      and `_pulsim.pyi` exist in the installed package.
- [x] 8.3 (No kw_only note: section 1 is not implemented in this change.) Add a CHANGELOG `[Unreleased]` entry covering the stubs, the
      guide and CLI, `llms.txt`, executed docs and the `kw_only` note.
      Add a README pointer to `python -m pulsim.guide`.

## 9. Validation

- [x] 9.1 Run `openspec validate add-ai-usage-guide --strict`.
- [x] 9.2 Run the full Python suite: 0 failed (`--timeout=600`).
- [x] 9.3 Run `ruff check python/ scripts/generate_stubs.py`: clean.
- [ ] 9.4 Run ctest in `.scratch/build-ctest` after the binding changes:
      green.
- [x] 9.5 Delete the scratch venvs and wheel trees.

## 10. Naive-agent evaluation, round 1

Three agents with no prior Pulsim context solved a buck, a bridge
rectifier and a three-phase VSI from the guide alone; a reviewer
reproduced every friction item.

- [x] 10.1 Section 3.2 printed a ripple of 67 mV pp where
      `dI / (8 f C)` gives 16 mV: `make_pwm_switch_fn` puts some edges
      one step off on a `dt` grid. The guide now builds the PWM in whole
      steps (`grid_pwm`), shows the on-step count of the helper (49/50/51)
      as an executed block, and 6.2 compares ripple, not only means.
      Tutorial 02's ripple prose no longer quotes the jitter as ringing.
- [x] 10.2 Trap 13: without `dt`, a `V_th = 0` bridge behind a series
      resistance runs half-wave (318.8 V, 12.9 V pp against 322.1 V,
      6.4 V pp). Section 2 and the short version route PWL diodes without
      `dt` to `V_th > 0`.
- [x] 10.3 Trap 14: `res.v("gnd")` returns a branch current.
- [x] 10.4 Section 3.4: both summaries average the whole record and
      `T_j` starts cold; the example adds the windowed conduction loss and
      the steady-state temperature. Reverse-recovery loss
      (`diode_specs`) is documented as unreliable with an xfail check.
- [x] 10.5 Section 3.7: three-phase VSI with SPWM and dead time
      (293.9 V against the ideal 293.9 V), the reverse-conduction and
      loss-attribution caveat; the `add_three_phase_vsi` docstring names
      the same path.
- [x] 10.6 Section 4: voltage-source current sign, PWM `phase` units,
      and the `make_three_phase_spwm_fn`, `ThreePhaseLegIndices` and
      `device_thermal_summary` signatures. Trap 1: read a Shockley
      diode's current through a series resistor. 6.1: a diode run whose
      currents close the balance, and windowed end values via
      `np.interp`. Trap 11 uses a reproducible example.
- [x] 10.7 `.scratch/` is git-ignored, so an sdist built from a working
      tree does not ship scratch files; the unknown-attribute message no
      longer names pulsim 1.0.0.
- [x] 10.8 Library defects found. Fixed since: `res.v` of a ground name
      (11.3); `diode_specs` sampling `V_R` at the commutation instant (the
      record branch reads it one `t_rr` after the turn-off: 0.988 W
      against the expected ~1 W on the 3.4 buck). The rest are follow-ups
      in 12.

## 11. Review round 2 (after the naive agents)

- [x] 11.1 Merged base (0.1). Re-measured every number the guide prints
      on the merged code: only 3.4's diode run loss moved (2.456 ->
      2.453 W). Re-checked traps: trap 13's TR-BDF2 `V_th = 0` bridge is
      still half-wave (318.79 V, 12.91 V pp), now with `BR__D1`/`BR__D3`
      at 0 A and `BR__D2`/`BR__D4` at 0.318 A instead of one diode at
      299 A; `V_th = 0.7` reads 322.05 V on both engines and every bridge
      diode 0.161 A; the VSI node-a KCL closes to 2.4e-10 A. The 3.4
      reverse-recovery block and the 6.1 diode balance (199.4 mC) now
      assert; the second 6.1 example (the workaround) is gone. Stub
      regenerated: `switch_states`, `switch_states_bytes`,
      `switch_states_unsolved`, `SolverSnapshot.stamped_mask`; the overlay
      types `switch_states` / `switch_states_unsolved` as bool arrays.
- [x] 11.2 pybind11 3.x. Reproduced in a scratch venv (pybind11 3.1.0,
      stubgen 3.0.0, build at -j2): 4 stub-test failures and 812 changed
      lines from `--check`, most of them richer types (design D4.4).
      Pinned `pybind11>=3.1,<3.2` (`[build-system]`, the three
      `CIBW_BEFORE_BUILD_*`), FetchContent `v3.1.0`; the stub is
      generated from 3.1 and records it; `--check` on another generation
      fails naming both; numeric spellings normalised to `float`/`int`;
      duplicate enum `__eq__`/`__ne__` overloads merged (mypy rejected
      them); `_pybind11_conduit_v1_` ignored by the coverage test; the
      snippet runner compares pybind11's three spellings of `float` as
      equal. Stub, snippet and guide tests pass on a 2.12 and a 3.1
      build; `--check` passes on 3.1 and refuses 2.12 with the message.
      This worktree's extension is now the 3.1 build.
- [x] 11.3 `res.v(<ground>)` returns zeros of the record's length on the
      fixed, TR-BDF2 and DSED results (`_result_v`; `AdaptiveResult` has
      no `v`); `average_power_at_node` at ground is 0.
      `test_result_v_ground.py`. Trap 14 replaced.
- [x] 11.4 `switch_specs` soft edges. `losses.py` charges each edge
      `E_ref * |V_blk|/V_ref * |I|/I_ref` with `V_blk` measured across the
      OPEN switch (before a turn-on, after a turn-off); the docstrings
      said "the sample just before the edge". Measured on the VSI of 3.7
      (4 mJ / 3.5 mJ at 600 V, 50 A, 0-40 ms): with 1 us dead time
      `HSa` P_sw 11.29 W, soft edges 0.6 mJ of 0.45 J (V_blk median
      0.82 V); with none 22.64 W, soft edges charged at 599.98 V (the
      complementary switch takes the current at the same instant). Guide
      3.4 / 3.7 and both docstrings corrected.
- [x] 11.5 VSI diodes. Measured (merged code, 1 us dead time, 20-40 ms):
      `INV__Dha` conducts on 12.9 % of samples, 11.9 % with `HSa` gated
      on (dead time 2.0 %), carrying 1.96 % of `|i_a|` =
      R_on/(R_on+R_d); with no dead time 0 %. Guide 3.7 and the
      `add_three_phase_vsi` docstring corrected.
- [x] 11.6 RMS of pulsed currents. Buck 48 -> 12 V, D = 0.25, 100 kHz,
      MOSFET over 4-6 ms: `time_rms` 1.4942 / 1.4993 / 1.5019 A at dt =
      1e-7 / 5e-8 / 2.5e-8; `sqrt(time_average(i**2))` 1.5044 A at all
      three; analytic 1.5043 A. No public exact-in-step integral exists
      (`_result_views.pl_integral_product` is the same linear form, with
      right limits only on DSED). New trap 14 and a short-version line.
- [x] 11.7 Minors: `res.i` reference directions, current-source sign and
      `t` as a sample index (section 4, and the `res.v` docstring);
      `ripple_pp` on sparse variable grids (6.2); the 17x whole-run
      factor with its spec (3.4); `TimeStats` fields (3.2);
      `docs/losses-and-thermal.md` in section 9 and `llms.txt`;
      `build-tests/` git-ignored; the sliding-mode warning no longer
      says "in DCM" as the only cause and no longer claims the average
      is always right (it names the half-wave bridge); the
      `add_bridge_rectifier` Returns text and garbled diagram replaced.
      Guide 32505 bytes (budget 32768).
- [x] 11.8 Validation on the merged tree with the pybind11 3.1 build:
      full Python suite 1809 passed, 4 skipped, 8 xfailed, 0 failed
      (329 s); `ruff check python/ scripts/generate_stubs.py` clean;
      `openspec validate --all --strict` 37 passed; `--check` up to
      date. The scratch venv and both scratch builds were deleted.

## 12. Library follow-ups (not fixed here)

- [ ] 12.1 `make_pwm_switch_fn` (`core/include/pulsim/sources/pwm_switch_fn.hpp`)
      compares `fmod(t/T, 1) < duty` in floating point: on a `dt` grid
      edges jitter by one step (49/50/51 on-steps of 50 at 100 kHz,
      dt = 1e-7). Snap the phase to the grid (a relative epsilon). Audit
      `make_dead_time_pwm_pair_fn` too (48/49 at D = 0.5 with 2e-7 dead
      time). The guide's 3.2 on-step block then prints `[50]` and must be
      updated.
- [ ] 12.2 TR-BDF2 with `V_th = 0` PWL diodes: a bridge behind 0.1 ohm
      runs half-wave (318.79 V, 12.91 V pp against 322.06 V, 6.38 V pp
      with `dt`). The sliding-mode warning names `engine='auto'` even
      when `engine='trbdf2'` was requested. `V_th > 0` is not sufficient
      either: the same bridge with 1 mH of line and the default
      `V_th = 0.7` held `BR__D2`/`BR__D4` OFF while forward-biased by
      45.1 V at t = 75.92 ms (one sample), accepted 22 steps above
      tolerance at the step floor and warned of 5 boundary rides; V_dc
      321.807 V against 321.844 V with `dt = 1e-6`.
- [ ] 12.3 `res.i()` on a Shockley diode raises `NotImplementedError`
      pointing to `device_loss_summary`, which omits Shockley diodes.
- [ ] 12.4 `device_loss_summary` / `device_thermal_summary` have no
      `t_from` / `t_to` window (all three naive agents hand-rolled one).
- [ ] 12.5 `device_loss_summary`'s `i_rms` reads pulsed switch currents
      low on a fixed grid, like `time_rms` (1.6019 / 1.6074 / 1.6101 A at
      dt = 1e-7 / 5e-8 / 2.5e-8 against `sqrt(P_avg/R_on)` = 1.6129 A on
      the 11.6 buck); its `P_avg` is right.
- [ ] 12.6 `ThreePhaseVsiResult` / `BridgeRectifierResult` are not
      exported at the top level; the DSED refusal for a floating neutral
      lists "floating capacitors" and "nonlinear devices" and points to
      `notes/DSED_BRIDGE_DESIGN.md`, which users cannot see.
- [ ] 12.7 `simulate(engine='dsed')` with no `switch_fn` warns that it
      will poll "this plain Python switch_fn".
- [ ] 12.8 With no dead time, `switch_specs` charges soft edges of a
      bidirectional PWL switch at the full bus (22.6 W against 11.3 W on
      the 3.7 VSI). Physically the complementary turn-on is the hard
      one; the soft turn-off should cost about 0.
- [ ] 12.9 The missed-commutation warning of `res.i` repeats once per
      call site (Python's default filter keys on the caller's line), so
      a script that reads `res.i` of each diode in several places prints
      it many times. Emit it once per result and diode.

## 13. Review round 3

- [x] 13.1 `test_every_inline_reference_resolves` failed without
      pyqtgraph (CI installs `.[dev]`, which lacks it): trap 6 named
      `p.LiveScope`, which `pulsim/__init__.py` exports only when
      pyqtgraph imports. Trap 6 now says `from pulsim.scope import
      LiveScope` (needs pyqtgraph). New
      `test_inline_references_resolve_without_optional_extras` runs the
      resolver in a child with pyqtgraph and matplotlib blocked; it fails
      on the old wording and passes on the new.
- [x] 13.2 Trap 13 names both warnings (`simulate`'s boundary ride and
      `res.i`'s "OFF while forward-biased" missed commutation) and says
      `V_th > 0` is no guarantee (the 12.2 case, reproduced); the short
      version says to heed `res.i` warnings. The sliding-mode warning and
      the CHANGELOG no longer say the half-wave run gives "only this
      warning".
- [x] 13.3 Trap 10 says its 5.84 / 5.99 V come from
      `make_pwm_switch_fn`.

