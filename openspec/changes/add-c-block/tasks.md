## 1. Core (Python step fn, PWL, wires, ZOH)
- [ ] 1.1 `python/pulsim/c_block.py`: `CBlock` dataclass + `add_c_block(builder, inputs, outputs, *, dt, fn=…, name=…)`
- [ ] 1.2 Resolve input wires (`("v",node)` / `("i",branch)`) to state indices at add time
- [ ] 1.3 Insert one controlled source per output wire (`("v",n+,n-)` / `("i",n+,n-)`); record RHS rows
- [ ] 1.4 Build the throttled `step_observer` (read inputs → call fn → store outputs; ZOH) + `b_extra_fn` (inject held outputs)
- [ ] 1.5 Compose into `simulate()` alongside user `step_observer`/`b_extra_fn`/`closed_loops`
- [ ] 1.6 Validate `dt_block ≥ dt` (warn + clamp); first firing at t=0

## 2. C / C++ via shared library (ctypes)
- [ ] 2.1 Define the C ABI (`pulsim_cblock_step` + optional `init`/`term`) in a public header `include/pulsim_cblock.h`
- [ ] 2.2 `lib=` loader: `ctypes.CDLL`, resolve symbols, marshal numpy↔`double*`, manage opaque `state`
- [ ] 2.3 Lifetime: call `init` on add, `term` on result/GC; one-call-with-`**state` fallback

## 3. C / C++ inline source (auto-compile)
- [ ] 3.1 Source template wrapping user `code` with the ABI for `lang="c"|"cpp"`
- [ ] 3.2 Compile via `cc`/`c++` (`-shared -fPIC -O2`); content-hash cache; honour `include_dirs`/`extra_*_args`
- [ ] 3.3 Clear error when no compiler is found (point to `lib=`)

## 4. YAML surface
- [ ] 4.1 `c_block` node in the loader: `inputs`, `outputs`, `dt`, `lang`, inline `code` or `lib`/`file`
- [ ] 4.2 Round-trip example YAML + loader test

## 5. Tests
- [ ] 5.1 Python fn: gain block (out = k·in) drives a controlled source; verify circuit response
- [ ] 5.2 C shared-lib block: compile a tiny `.so` in the test, load, verify identical result to the Python fn
- [ ] 5.3 Inline-source block (skip if no compiler): verify same result
- [ ] 5.4 Sample time / ZOH: `dt_block = 10·dt` → output is piecewise-constant at the block rate
- [ ] 5.5 Multi-IO: 2-in / 2-in block; state persistence across steps (integrator)
- [ ] 5.6 End-to-end: a discrete PI in a c-block regulates a buck to setpoint (cross-check vs `bind_pi_to_switch`)

## 6. Docs
- [ ] 6.1 User-guide page "Custom Code Blocks (C / C++ / Python)": model, wires, sample time, the 3 delivery modes, ABI, security note, runnable examples
- [ ] 6.2 Wire into `mkdocs.yml`; `mkdocs build --strict`

## 7. GUI (PulsimGUI — separate repo, tracked here)
- [ ] 7.1 C-block node with N/M pins that serialises to the `c_block` YAML/Python representation
- [ ] 7.2 GUI ↔ kernel round-trip smoke test
