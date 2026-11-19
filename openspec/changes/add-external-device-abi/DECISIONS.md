# Decisions taken while implementing

## Slice 1 — the ABI, the checks, the kind, the twins (2026-11-19)

1. **The ABI is a plain C header, version 1** (`core/include/pulsim/xdev/pulsim_xdev.h`,
   C99 and C++). One exported symbol `pulsim_xdev_get(uint32_t index)` returns the
   index-th model's descriptor (NULL past the last): a struct with `abi_version`,
   `struct_size`, counts, the fixed df / dq patterns, the field table (name, unit,
   default, bounds), the probe box, claims, and function pointers (`eval` required;
   `validate`, `create`/`destroy`, `init`, `accept` optional; 4 reserved slots).
   Fixed-width integers, doubles, `const char*`, pointers only; every function
   returns an `int32_t` status (no exceptions cross); the host owns every array it
   passes, the library owns what `create` returns (freed by its own `destroy`);
   `PULSIM_XDEV_CALL` = `__cdecl` on Windows on every function pointer type.
   **Stable**: that header, and only it. **Not stable**: the model-side C++ helper
   (`xdev/adapter.hpp`, compiled INTO the library — a library built with it is
   bound only by the header) and the kernel side (`xdev/host.hpp`, `checks.hpp`).
   Growth: fields are appended only; the host refuses `struct_size` smaller than
   the version-1 struct and will read an appended field only when `struct_size`
   covers it; a version mismatch is refused by name ("built for external device ABI
   version 99, this Pulsim speaks version 1").
   Alternatives (design.md): a C++ plugin interface (templates, AD across the
   boundary) has no stable ABI across compilers / standard libraries (MSVC vs
   libstdc++ vs libc++ layouts of std::array/std::function, name mangling,
   exceptions) — rejected; a Python callable — one interpreter call per device per
   Newton iteration, no threads — rejected for the kernel (could be a prototype
   tool later). The C table costs one indirect call per device per evaluation and
   no AD across the boundary (the helper runs AD inside the library).
2. **Contract enforced at registration** (`xdev::check_contract`, called by
   `CircuitBuilder::add_external_device` unless `check=false`): canary-guarded
   buffers (writes past f/q/df/dq, entries left unwritten), finiteness, purity (two
   calls bit-identical), KCL (pin f and pin q sum to 0), declared Jacobian against
   Richardson central differences at 17 probe points (1e-5 of the row scale),
   the Jacobian between probes (secant vs mean tangent along a 2 001-point line per
   unknown plus a geometric run to ±1e-15 of the box around 0), and the claims:
   SMOOTH (no Jacobian jump > 5 % of the line scale between neighbours), PASSIVE
   (Σ f_p·v_p ≥ 0), RECIPROCAL (dq/dx symmetric in the pin block and in the
   internal block), monotone pairs (f[row] non-decreasing in x[col] to 4 ulp).
   State: n_state > 0 requires `accept`; init/accept run on a copy with guard reals,
   at registration and at EVERY commit of a run (write past n_state / non-finite
   state refused; nothing reaches the flat state unless it passed).
   Run time: every evaluation sums all outputs and tests the sum (one comparison);
   only a failure scans to name the entry. `debug_jacobian=True` re-checks the
   Jacobian at every evaluation of the run (the operating points the run visits).
   Not certifiable: the ~1e-12 smoothness Newton needs, anything outside the box.
3. **The kernel kind**: `StoredKind::ExternalDevice` (28), kind-table row
   NewtonStateful on pwl / trbdf2 only (dsed, bdf1, ac_sweep refuse through the
   capability table, `UnsupportedError(..., kind='ExternalDevice')`).
   `pwl::ExternalDeviceHistory` is one more entry of `StatefulKinds`, so both time
   loops reach it through the existing folds — NO loop edit. Its storage IS the flat
   state block (`external_device`, 1 real per "instance", count = reals: per device
   q_n, q̇_n, q_γ per row, then n_state) — #32's offset view, for this kind. The
   stamp is `stamp_fq` at run-time arity with the same `formula_a` /
   `history_term` and row rules (pin row: q̇ from the formula; internal: −f).
   Branches: `name` (pin 0 → pin 1) carries internal unknown 0; `name#k` the k-th;
   pins beyond the first two are not graph-connected (as a MOSFET gate).
4. **`device_path='legacy'` with an external device: auto-routed.** It has no hand
   stamp; its law is its only stamp, on either path (`stamp_kind` calls the same
   function). The rest of the circuit follows the path asked for.
5. **DC operating point: refused by name** (`compute_dc_operating_point`,
   `UnsupportedError(feature='dc_operating_point', kind='ExternalDevice')`): the DC
   solve stamps only the static refresh; a stateful kind would be silently missing.
   Request below.
6. **YAML: not wired.** No `external_device` YAML type exists, so a netlist cannot
   load a library at all (safe by absence); task 1.7 stays open with its opt-in
   keyword. Loading happens only in `add_external_device(..., lib=path)`, the
   caller naming the path.

7. **The built-in's G_min is part of the twin's equations.** The linear assembly
   puts 1e-12 on the branch row of a saturable inductor (so the cache factorises
   before Newton stamps); the external kind gets the same G_min on every branch
   carrying an internal unknown (`pwl/assemble.hpp`). Without it the satind twin
   differed by 4.6e-6 of scale and took twice the Newton iterations, and a lone
   external internal row made the state-space cache singular. The branch-structure
   pass (D4 dependent states) refuses an external device like the other Newton
   devices with state.
8. **Twin results (macOS, Apple clang 21, this tip).** pwl, `device_path='interface'`,
   2 ms at 1 µs: saturable inductor twin vs built-in 6.1e-16 of scale, identical
   Newton iteration counts (7 506); Shockley half-wave rectifier twin 0.0 (bit-
   identical, 4 692 iterations). Tests assert 1e-9 (CI margin), trbdf2 1e-3 of scale
   after interpolation onto the built-in's grid (rtol 1e-6). Snapshot / resume of a
   circuit with the satind twin and the stateful `energy_meter`: final `dev_state`
   bit-identical to the uninterrupted run.
9. **Cost, measured honestly** (`.scratch/xdev_instr.sh`, instructions retired per
   accepted step, K = 10, one run): satind 16 763 built-in → 35 840 external
   (**2.14×**); Shockley rectifier 15 390 → 22 979 (**1.49×**). Where it goes: the
   external stamp is generic (heap vectors per instance, NaN-poisoned outputs, the
   one-comparison sum, a run-time pattern loop with `coeffRef` per entry) where the
   built-in scatter is unrolled at compile time; the satind twin runs AD at full
   width 3 inside the library where the built-in uses a hand Jacobian with a
   one-tangent λ(i); one indirect call per evaluation. Not yet recovered: a batch
   `eval_n` over all instances of a model (one indirect call, vectorisable),
   cached matrix slot pointers instead of `coeffRef`, and a hand-Jacobian path in
   the adapter.
10. **Reviewer (fresh context) findings fixed**: the version-1 descriptor size is
    FROZEN in the header (`PULSIM_XDEV_DESCRIPTOR_V1_SIZE`, static_assert that the
    host struct has not grown unguarded) — `sizeof` would have refused every old
    library at the first appended field; the descriptor and all its arrays and
    strings are deep-copied at load (a writable descriptor changing after
    validation cannot steer indices); outputs are NaN-poisoned before every eval (a
    model returning OK without writing is refused by the finiteness test); reserved
    slots must be NULL; `ctx.reserved0` became `host_features` (0 in v1: `t` not
    filled); a failed `create` is destroyed; libraries load only from an existing
    regular file by absolute path (`LoadLibraryExW` with DLL-load-dir + default
    dirs on Windows: no CWD / PATH planting); the Jacobian is also checked with a
    state the model wrote (`accept`), not only zeros; an opt-in prototype of the
    entry point (`PULSIM_XDEV_DECLARE_ENTRY`). Open from the review: writes through
    `const` pointers (p, state, x) are not detected; registration and run differ in
    state-buffer aliasing (documented: buffers never alias); probe lines are on-axis
    only (no diagonals); a library rebuilt at the same path within one process
    returns the old code (dlopen refcount); the debug check's 1e-12 absolute floor
    misses sub-pF capacitances.

## Slice 2 — portability, CI wiring, cost, reviewer items (2026-11-19)

11. **Layout asserted in the stable header.** `pulsim_xdev.h` now asserts
    (C11 `_Static_assert`, C++ `static_assert`, C99 / MSVC-default-C negative
    array) the offsets of every field group and both frozen V1 sizes, for
    8-byte and 4-byte pointers: descriptor V1 = 200 / 120 bytes, ctx V1 =
    40 / 28, field 40 / 32. No `long`, `size_t` or enum field and no padding
    anywhere (a `#pragma pack(1)` changes nothing), so LP64 and LLP64 share
    one layout. Only asserts were added: layout and version 1 unchanged.
    Seen compiling with `-Wall -Wextra -Wpedantic -Werror`: header and
    `contract_models.c` as C99 / C11 / C17 on Apple clang 21, GCC 15, LLVM
    clang (Homebrew), and as C++17 / 20 / 23 on the same three; the header
    `-fsyntax-only -ffreestanding` cross-targeted (LLVM clang) to
    x86_64 / i686 / aarch64 Windows MSVC, x86_64 MinGW, x86_64 / i686 /
    aarch64 / armv7 Linux, C99 / C11 / C++17. Negative control: a header with
    one extra field fails all three language modes.
12. **Windows, by reading (not run).** Export: `PULSIM_XDEV_EXPORT` =
    `__declspec(dllexport)` on `pulsim_xdev_get` in the C models and the
    adapter macro; `__cdecl` on every function pointer type and every
    function assigned to one (C models via `CALL`, adapter statics). C models
    use no VLA / complex (`-Wvla` clean), only C99 math MSVC has. Fixed:
    `LoadLibraryExW` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` needs a fully
    qualified BACKSLASH path — the absolute path is now `make_preferred()`;
    `C_STANDARD_REQUIRED` dropped for the C model (MSVC has no C99 switch);
    MODULE output fixed at `<build>/xdev_test_models` with `$<0:>` so the
    Visual Studio generator adds no `Release/`; the Python finder ignores
    `.lib/.exp/.pdb`. No pointer-to-member template argument in the xdev code.
13. **CI runs the twin tests instead of skipping them (not yet seen on CI).**
    The two test model libraries are defined OUTSIDE the test block
    (`PULSIM_BUILD_XDEV_TEST_MODELS`, default ON), so the wheel build CI
    already runs (`pip install ".[dev]"`, scikit-build-core, `build-dir =
    build/{wheel_tag}`) builds them into `build/<tag>/xdev_test_models`; they
    are never installed into the wheel. `test_external_device.py` takes
    `PULSIM_XDEV_TEST_LIBDIR`, else the newest `build*/[*/]xdev_test_models`,
    else compiles with `cc`, and FAILS (not skips) when `CI` is set and none
    is found. The wheel ships the stable header only, at
    `pulsim/include/pulsim/xdev/pulsim_xdev.h`; `pulsim.get_include()`.
14. **Cost: the 1.49–2.14× was mostly the probe.** The old probe rebuilt the
    circuit inside the timed loop, so every repetition paid the REGISTRATION
    checks (now ~1.0e8 instructions per satind circuit, 6.7e7 per Shockley:
    a per-circuit cost, ~30 ms). Built once and simulated 1+K times
    (`.scratch/xdev_loop2.py`, `xdev_instr2.sh`, K = 40, three runs), the
    per-step ratios on the slice-1 code were satind 1.13–1.14×, Shockley
    1.08–1.09×. Table after each step in #15.

15. **Cost after each step** (instructions retired per accepted step, built
    once, K = 40, three runs each, macOS / Apple clang; external / built-in):

    | step | satind (AD) | satind (hand Jacobian) | Shockley rectifier |
    |---|---|---|---|
    | slice-1 code, old probe (registration in the loop) | 2.14 | — | 1.49 |
    | slice-1 code, probe fixed | 1.143 / 1.129 / 1.137 | — | 1.092 / 1.081 / 1.076 |
    | (a) cached, validated matrix slots | 1.136 / 1.104 / 1.169 | — | 1.093 / 1.084 / 1.055 |
    | (b) adapter hand Jacobian (`eval_jac`) | — | 1.124 / 1.119 / 1.130 | (built-in is AD too) |

    Target ≤ 1.2× met on both twins. (a) and (b) are within the run-to-run
    noise (±3 %, the built-in itself reads 18.6k–19.5k): neither is a
    measured gain on one instance; both kept (the slot cache is validated
    per use — column range and row index — so it cannot return a wrong
    entry after any insertion or rebuild; `eval_jac` lets a model skip AD).
    What remains (~2.3k instructions per step on satind, ~1.1k on Shockley)
    is structural: the external kind is a STATEFUL kind on every model, so
    it evaluates again at each commit and carries q history, where the
    built-in Shockley is stateless. (c) **No batch entry**: both twins have
    ONE instance, so a batch call cannot move their ratio, and an ABI field
    is frozen forever. Decided, not built: when a many-instance case
    measures the indirect call, `eval_batch` is APPENDED after `reserved`
    (read only when `struct_size` covers it; ABI version stays 1; the host
    falls back to `eval` for every older library). The reserved slots stay
    NULL-only.

16. **Second reviewer (fresh context, read-only) — fixed**: (1) a rebuilt
    library after its last user was released could still be the old image
    (glibc never unloads a library with STB_GNU_UNIQUE symbols) — the
    registry now keeps each path's identity after expiry and refuses a
    changed file while `dlopen(RTLD_NOLOAD)` still finds the old image;
    (2) the registry key is the `weakly_canonical` path (`.`/`..`/symlinks);
    (3) registration passed `ctx->state` aliased to the output state, so a
    model updating state through the const `ctx->state` passed and froze in
    a run — now a separate committed buffer, compared after init/accept
    (`liar_state_via_ctx` refused); (4) off-axis lines no longer judge SMOOTH
    (uniform grid; axis lines refine around 0) — a jump skips the interval;
    (5) off-axis resolution capped (2000·min(‖dir‖₁, 4) points); (7) the
    element structs `pulsim_xdev_field` / `pulsim_xdev_monotone` are declared
    FROZEN in the header (no size field: growth = a new appended array
    pointer); the descriptor copy reads min(struct_size, sizeof) bytes;
    (9) an unresolvable path throws instead of falling back to the relative
    name. **Open (noted, not fixed)**: debug_jacobian refuses an honest
    NON-smooth model when a central difference straddles its kink (accept a
    one-sided difference when SMOOTH is not claimed); during a run the const
    inputs are guarded only under debug_jacobian (a model writing through
    them conditionally, outside the box, is not seen); a model memoising in
    its `instance` passes purity yet is thread-unsafe; same-size same-mtime
    rebuilds; Windows paths beyond MAX_PATH (no `\\?\` prefix); the
    one-sum finiteness test refuses a finite sum that overflows.

## Slice 3 — time, DC, AC, the reviewer's open list (2026-11-20)

17. **The solve's time reaches the model.** `ctx->t` was always 0. Now
    `pwl::StepCtx` (the driver-owned step context every stateful stamp reads)
    carries `t`, set at the four places the context is written — pwl's single
    solve choke point (`run_event_iteration`: `at_t`, the sub-step's end), the
    trbdf2 probe/landing `trap_solve` (`t0 + dt`), stage 1 (`t + h_pt`, the γ
    point) and stage 2 (`t + h`). The external kind holds a pointer to it
    (`bind_time`, in `build_device_refresh`, after the `!hs.any()` return);
    evaluations before the binding (init at the start, the DC solve, the AC
    linearisation) use `set_start_time` = `t_start` / `t_eval`. No built-in law
    reads `t`: built-ins pay one store per solve. Header: `PULSIM_XDEV_HOST_TIME`
    (bit 0 of `host_features`, which the v1 header already defined as "a set
    bit = an optional ctx field is filled") — purely additive, version stays 1,
    sizes unchanged (`abi_v1_c90fc07f/` holds the header as of c90fc07f; the
    `pulsim_xdev_contract_v1` library is built against it and gets the time
    too). Test model `ramp_conductance` (g = G(1 + t/1 ms)): v_a at every
    output time equals 1/(1 + R g(t)) to 5.6e-17 on pwl and trbdf2, from both
    libraries. The registration checks evaluate at t = 0.
18. **DC operating point.** `compute_dc_operating_point` wraps the refresh
    when (and only when) the pool has an external device: after the static
    chain, `ExternalDeviceHistory::stamp_dc` — the same stamp with
    `Formula::Dc` (a = 0, no history term: f(x) and ∂f/∂x, q̇ = 0) — through
    the same Newton and cascade. A branch carrying an internal unknown gets
    the DC inductor's −1e-12 diagonal (`dc_assemble.hpp`), as #7 did for the
    transient. Model state: init(x = 0) (the zero start); pwl's
    `start_from_dc_op` re-seeds the external history at the DC point. Nothing
    is refused by name: a model with no DC path fails through the DC solver's
    own diagnosis. Twins (macOS, Apple clang): Shockley rectifier DC point 0.0
    (bit-identical), saturable inductor 0.0 after the diagonal (2.0e-12 before
    it: the built-in's 1e-12 Ω series term); test gate 1e-9 (CI lesson).
19. **Found, not this track's: the built-in saturable inductor after
    `start_from_dc_op` on pwl.** pwl seeds only the PMSM history from the DC
    point; the saturable inductor's flux history keeps λ(0) = 0 while its
    current is the DC current, so a 1 V / 0.5 Ω / L circuit that must stay at
    i = 2 A drops to ~0 A at step 1. The external twin (seeded) stays at the DC
    point to the bit. Fixing the built-in changes a default number: left to the
    device-interface track (the test pins the external against the truth and
    compares the two DC points only).
20. **AC sweep.** `run_mna_sweep` (kernel) linearises an external device at
    the DC operating point of the given mask: G += ∂f/∂x, E += ∂q/∂x (E is the
    physical mass matrix, C/2 of the transient split; the Newton Jacobian is
    df + a·dq with a = 2/h, so the signs agree). The capability row admits
    `ac_sweep` for ExternalDevice only. Checks: external Shockley at 0.39 A,
    H against r_d = n·V_T/(I + I_S) with R‖C load, 1.8e-15 relative; external
    saturable inductor at 2 A against jωL(I)/(R + jωL(I)), L(I) = dλ/di, and
    against a built-in linear inductor of that L, 5.7e-11 (the branch G_min).
21. **The reviewer's open list (#16), closed:**
    (a) `debug_jacobian` / registration at a kink: a model that does NOT
    claim SMOOTH may declare either one-sided derivative (forward and backward
    Richardson quotients accepted besides the central one); a SMOOTH claim is
    held to the central quotient (`honest_kink` passes at the probe centre
    and through a run with debug_jacobian; `liar_kink` still refused).
    (b) const inputs during a run, always on: a stateful model's committed
    state compared after every evaluation (n_state reals; zero cost for
    n_state = 0), `ctx->p` against a frozen copy once per accepted step; `x`
    is the host's per-call copy and is gathered again before `accept`.
    (c) memoising models: registration adds an interleaved check (x_a, x_b,
    x_a must repeat x_a) and a 4-thread probe (each thread evaluates the
    probes in its own order 40 times; every result equal to the serial one to
    the bit; exceptions carried out of the threads). Decided: NO THREAD_SAFE
    claim and no serial mode — the v1 header already requires reentrancy, and
    a new claim bit would silently serialise every existing library. A probe
    is not a proof (a race that does not fire is not excluded; a thread
    started with a different FP environment, e.g. FTZ set by a fast-math
    library in the main thread, could make an honest model producing
    subnormals look non-reentrant — documented).
    (d) same size and mtime: the file identity includes an FNV-1a hash of its
    bytes (read at each load request).
    (e) Windows MAX_PATH: `\\?\` (or `\\?\UNC\`) prefix for paths ≥ 260
    characters before LoadLibraryExW (not run: no Windows here; the
    std::filesystem checks before it rely on the STL's long-path support).
    (f) a finite output set whose sum overflows is no longer refused (a
    NaN/Inf entry still is).
    The Windows-skipped test (`a library rebuilt while loaded`) stays skipped
    with its reason: Windows locks a loaded DLL, so the file cannot change.
22. **Example (a real use, not a twin):** `examples/external_devices/
    self_heating_resistor.c` (plain C, hand Jacobian, one internal unknown =
    temperature rise; ILLUSTRATION parameters) built by
    `examples/scripts/run_external_self_heating_resistor.py` with the system C
    compiler against `pulsim.get_include()`; claim: the settled rise equals
    the closed-form root of θ(1 + αθ) = R_th V²/R_0.
23. **The exact engine (`dsed`) keeps refusing external devices by name**: it
    propagates each topology mode in closed form and admits no Newton kind; a
    law known only through f, q and their Jacobians at sampled points has no
    closed-form propagator. Documented as a limitation.

## Slice 4 — state, outputs, YAML allow list, native cache (2026-11-20)

24. **Merge of feat/core-integration b28f1e57** (real merge, 552c3086).
    `pwl::StepCtx::t` is still written at every place the skeleton writes
    the step context (pwl `run_event_iteration`; trbdf2 probe `trap_solve`,
    stage 1, stage 2) — no new writer appeared. The time test
    (`test_a_time_dependent_law_reads_the_solve_time`, ramp conductance, both
    libraries, pwl and trbdf2) re-run: worst |v_a − closed form| = 1.1e-16 V on all four (pwl / trbdf2 × `contract` / `contract_v1`).
25. **3.1 Lauritzen twin with an internal charge** (`lauritzen_twin`,
    `twin_models.cpp`): f = {i, −i, A·q_M − q_E/T_M}, q = {0, 0, q_M} — the
    device law's local row as an internal unknown of the circuit. Double
    pulse (400 V, 20 A, 50 nH, S1 on at 1 µs), pwl dt = 0.2 ns: worst relative gap 1.5e-11 (legacy device path) / 1.3e-11 (interface), same grid, the stray current recovers to −105.8 A in both; trbdf2 (rtol 1e-6, both 7528 samples): 2.6e-7 relative on i(Lstray), interpolated.
    The built-in condenses q_M (K0/K1, or the interface path's local row);
    the external solves it as a circuit unknown with the 1e-12 branch
    conductance (#7) — 1e-12·q_M ≈ 1e-19 A against 10 A terms.
26. **2.2 junction capacitor** (`junction_cap_twin`, hand Jacobian, the
    built-in's operation order and floor clamp; probe box above the floor,
    where the clamp is not C1): gap 2.4e-15 (legacy) / 2.8e-15 (interface) relative on pwl dt = 1 ns once the test adds the built-in's structural G_min (assemble.hpp stamps 1e-12 S on each node of a nonlinear capacitor; without it the gap was 1.03e-11 — the built-in sits 4e-9 V under 400 V, the twin does not). Charge delivered at 400 V: ∫ i(pos) dt over the recorded generic reading = 312.3106 nC. The resistor-current trapezoid gives 332.3 nC: it counts the t = 0 sample (40 A at v = 0) that no step integrated.
27. **3.2 state in the flat vector** — already true since slice 1 (cells =
    q_n, q̇_n, q_γ per row, then n_state). Verified now: on trbdf2 the
    rejected steps leave no trace (the meter's energy equals the rectangle
    sum over the stored accepted steps, rectifier + meter, trbdf2 rtol 1e-6: 496 accepted, 39 rejected, |meter − sum| / sum = 1.6e-15), a trbdf2 snapshot /
    resume carries the model state (resumed run vs straight run: 1.2e-4 relative at most (another step grid after the controller restart); on its own grid the resumed meter equals the snapshot energy plus the rectangle sum over the second run's steps (rel 1e-12 asserted)). Steady state: `steady_state` refuses by name (NotImplementedError naming the block `external_device`: its `_snapshot_like` copies only listed blocks) — the period map does not carry the model state yet; `periodic_steady_state` refuses stateful devices (UnsupportedError). Both tested. Copying the block in the period map is NOT done (the message counts the block's cells as "device(s)": 7 for one meter — another track's wording, left).
28. **3.4 named outputs — appended descriptor block 1** (ABI version stays
    1): `n_outputs`, `n_events` (reserved, must be 0), `outputs`
    ({name, unit}, frozen element), `outputs_fn(ctx, x, out)`; size 224 / 136
    (`PULSIM_XDEV_DESCRIPTOR_OUTPUTS_SIZE`, also the feature-test macro a
    library compiled against either header can use), asserted in the header.
    The host reads the block only when `struct_size` covers it (an older
    library — `contract_v1` — loads unchanged and has no outputs); names are
    unique, non-empty and not a generic key (`power`, `i(<pin>)`); outputs
    are checked finite and pure at the probes. Readings: per device,
    i(<pin>) = f + committed q̇ (the device interface's step-9 convention,
    `pwl/device_readings.hpp`), power = Σ v·i, then the named outputs with
    the committed state. NOT through `DeviceReading` (fixed 4 pins / 2 rows:
    an external device has up to 64 rows): `ExternalDeviceHistory::reading`
    computes the same quantities. Trace: `xdev::OutputTrace` on
    `StepHooks::device_state` (after the commit; on the recorders' hook set
    when there is one), only for devices added with `record=True` — a run
    without one attaches nothing (no cost). The start row comes from a copy
    of the history taken at the first accepted step (its start cells = init /
    DC seed / restored snapshot) and the stored initial state.
    `res.device_output(name, key)` maps the stored samples to traced rows by
    exact time; a stored sample that is not a committed step (trbdf2's
    post-event record of a probe solve) is refused by name, not
    interpolated. Two trbdf2 commit sites stored a sample without the
    `device_state` hook (the post-event probe commit and the diode landing):
    the hook is now fired there (shared phases of trbdf2_transient.hpp, no
    effect without hooks; nothing else consumes `device_state`).
29. **3.3 events — left open; what is missing.** The engines locate only
    their own kinds' crossings: pwl splits a step at gate edges (switch_fn
    oracle bisection) and at switched-diode crossings found from the step's
    first solve (`run_event_iteration` / `split_step`, `ev_t[]`); trbdf2
    clamps to gate edges and localises a crossing by Illinois on trial
    trapezoidal solves of ONE signal, `detail_trbdf2::crossing_signal(
    diodes, idx, x)` (DiodeEventState only). `StepHooks::on_event` is a
    REPORT after the settle, not a locator. Carrying an external g(x) needs,
    in the shared phases: (a) a generic event-signal source on
    `StatefulHistories` (per kind: `n_signals()`, `signal(k, x, t)`),
    sampled at the step's end (pwl) / stage and end points (trbdf2); (b) the
    pwl split to accept a crossing time from such a signal (secant on the
    first solve, as the diode split) and commit the sub-interval; (c) the
    trbdf2 Illinois loop to take `crossing_signal` from that source instead
    of only `DiodeEventState`; (d) after the landing, the kind's `accept`
    (where a latch flips its state). None exists without a loop edit, so the
    header reserves `n_events` (refused when non-zero) and 3.3 stays open.
30. **1.7 YAML** — owner decision: an ALLOW LIST. `load_yaml_string /
    load_yaml_file(..., allow_native_code=[paths or directories])`; the
    netlist's `lib:` (relative to the file's folder; the working directory
    for a string) and each entry are `std::filesystem::canonical`
    (symlinks followed) before the comparison; a directory entry admits
    what lies under it; an entry that does not exist admits nothing; a
    library outside the list is refused naming the device and the resolved
    path; `allow_native_code=True` is refused (TypeError for a bare string).
    The options reach `load_device` through a thread-local scope set by
    `load_text` (no signature change through the subsystem loaders). Keys
    strict (`type, name, nodes, lib, model, params`); parameters through
    the model's field table (`resolve_params`: unknown name, repeated name,
    wrong unit refused). Inline `code:` refused as an unknown key. The IR
    emitter still refuses external devices (Circuit pickling unchanged).
31. **1.6 components()** — `kind: external_device`, `params` by field name,
    `model`, `library` (canonical path), `library_hash` =
    `fnv1a64:<16 hex>` of the bytes read at load, `pins`, `internal_branch`.
32. **1.5 cache** — the per-user cache already existed (merged from the
    control-graph track, refusing a foreign or writable cached file). Now
    factored into `pulsim/_native_cache.py` (`cache_dir`, `check_owned`,
    `is_private`, `compiled`); `c_block` keeps its names as aliases; a
    cached LIBRARY that is not private (foreign owner, group/world-writable,
    symlink) is deleted and rebuilt; a cache DIRECTORY that is not private
    is still refused (Pulsim cannot repair what another user controls).
33. **Cost and byte comparisons on this tip.** Golden / stateful / engines
    byte comparisons against the baseline recorded before slice 4: all
    IDENTICAL. Newton buck, instructions per accepted step, K = 10, three
    runs: pwl 28 944 / 31 204 / 31 056 (baseline 29 113 / 30 764 / 30 572),
    trbdf2 59 297 / 59 848 / 59 610 (baseline 59 326 / 59 231 / 59 457) —
    within the run-to-run spread. Twins, external / built-in, K = 40, three
    runs: satind 1.111 / 1.185 / 1.121, satind hand Jacobian 1.138 / 1.155 /
    1.161, Shockley rectifier 1.098 / 1.141 / 1.102 (≤ 1.2×; K = 10 runs
    scatter 1.12–1.27 and are not used). A run without a `record=True`
    device attaches no trace.
34. **Section 4 (`pulsim.xdev` SymPy) not started** — goals 1–5 took the slice; nothing of section 4 exists.
35. **Review of cad0720a (fresh-context, read-only) and what changed.**
    (1) trbdf2 fired `device_state` once per ITERATION (`tw_on_step`), but a
    step commit followed by a gate-edge or breakpoint restart in the same
    `tb_commit` stores two samples: the first had no trace row and
    `device_output` refused every PWM run on trbdf2. Fixed in the shared
    phase: `device_state_once(t, x)` fires at each of the three commit
    sites (step, post-event probe, diode landing) and at the iteration end
    only if that time was not just fired (no double rows). Test: a PWM buck
    with a diode on pwl and trbdf2. (2) `BlockNames::device` did not scope
    the `nodes` list: an `external_device` inside a subsystem attached to
    top-level names. Fixed, tested. (3) A `struct_size` inside block 1 copied
    an unvalidated `n_outputs` (out-of-bounds read): the host now copies a
    block only when the library's struct holds it whole (test model
    `partial_block`). (4) `reading` called `eval` without the const-state
    guard: routed through `eval_`. (5) an empty trace's message, (8) one
    `Path` as `allow_native_code` (TypeError like one string), (9) the cache
    returns only a regular file and removes its temp library after a failed
    compile — fixed. (6)/(7) documented (relative entries are from the
    working directory; a directory admits its subfolders — never list a
    writable shared one). LEFT: (10) the older `wire_c_blocks_from_yaml`
    still compiles / loads YAML `code`/`file`/`lib` without an allow list
    (an explicit separate call; documented, not changed). trbdf2's
    `record_post_event` display samples (not commits) are still refused by
    `device_output` by name. After the fixes: byte comparisons all
    IDENTICAL again; buck pwl 28 966 / 30 874 / 30 681, trbdf2 59 893 /
    59 505 / 59 474 instructions per step (noise); full pytest 5690 passed.

## Requests to other tracks

- **Stepper track** (DONE in slice 3, #17/#18 — kept for the record): (a) the DC operating point needs the stateful fold (an
  external device's f with Formula::Dc, or a "DC stamp" member of `StatefulKind`);
  until then #5 refuses. (b) `t` in `pulsim_xdev_ctx` is always 0: the refresh
  signature carries no time; a time-dependent external law needs the solve's t
  passed to `NonlinearRefreshFn` (or `pwl::StepCtx`).

## Hand-off (2026-11-20, slice 4)

- Open, in order: **3.3 events** (#29 lists the four loop pieces: a generic
  event-signal source on `StatefulHistories`, the pwl split, the trbdf2
  Illinois signal, accept after the landing) — needs the stepper owners;
  **3.2's steady-state copy** (`steady_state._snapshot_like` must copy the
  `external_device` block; `periodic_steady_state` needs read/set of it);
  1.6's `code=` free function; `wire_c_blocks_from_yaml` behind the same
  allow list (#35 (10)); section 4 (SymPy, not started); section 5.
- `device_output` refuses trbdf2's post-event display samples; a caller who
  wants a dense reading should drop them (they are not committed states).
- Verified on this tip (macOS arm64 only): xdev / yaml / layer5_v3 / step /
  record pass on Eigen 5 and 3.4.0; full pytest, `-m reference`,
  reference_status (REFERENCE_STATUS.md unchanged), openspec --strict,
  `ruff check python/`, stubs --check. Full ctest, Linux, Windows not run.

## Hand-off (2026-11-20, slice 3)

- Open, in order: **1.7 YAML** — the design (design.md:130, spec
  device-interface:86) names `allow_native_code=True`; the brief offered
  `allow_external_devices=[paths]` (a per-path allow list is stricter: a
  netlist cannot name a library the caller did not). Pieces: `yaml/loader.hpp`
  type `external_device` (`nodes`, `lib` relative to the YAML file's folder,
  `model`, `params` as numbers or `[value, unit]`, strict keys; refused
  without the opt-in, naming the device and the keyword), `load_text`
  options threaded to `load_device`, the two bindings and their stubs; the
  IR emitter (`yaml/emitter.hpp:305`) still refuses — when it writes, it
  writes `lib` = the canonical path the device was loaded from
  (`Model::library_path()`), never one it did not get, and `Circuit`
  pickling goes through `to_yaml` (ir_bindings.cpp:294), so pickling a
  circuit with an external device needs the same opt-in on the reader.
  Then 2.2 (junction capacitor twin), 3.1 (Lauritzen twin with an internal
  charge), 3.2–3.4, 1.5, stages 4–6.
- #19 (built-in saturable inductor not seeded by pwl `start_from_dc_op`) is
  for the device-interface track.
- Verified on 518e70c4 (macOS arm64 only): every C++ test target builds on
  Eigen 5; xdev / yaml / layer5_v3 / step pass on Eigen 5 and Eigen 3.4.0;
  full pytest 0 failed. The race model `liar_racy` needed `volatile` scratch:
  at -O2 the store was forwarded and the 4-thread probe saw no race (the
  probe was right, the test model was not). Full ctest, Linux, Windows not
  run on this tip.
