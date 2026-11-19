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

## Requests to other tracks

- **Stepper track**: (a) the DC operating point needs the stateful fold (an
  external device's f with Formula::Dc, or a "DC stamp" member of `StatefulKind`);
  until then #5 refuses. (b) `t` in `pulsim_xdev_ctx` is always 0: the refresh
  signature carries no time; a time-dependent external law needs the solve's t
  passed to `NonlinearRefreshFn` (or `pwl::StepCtx`).
