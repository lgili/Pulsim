Closed forms used below.

- Discrete P loop on an RC (τ = 1 ms) through a zero-order-hold
  controlled source, `T_s = 100 µs`, `K_p = 2`, set-point 1 V, zero
  state: `a = e^{−0.1} = 0.904837418036`,
  `v[k] = (2/3)·(1 − p^k)`, `p = a − 2(1 − a) = 0.714512254108`;
  `v[10] = 0.643545722992`.
- Same loop with `compute_delay = one_sample`:
  `v[k+1] = a·v[k] + (1 − a)·K_p·(1 − v[k−1])`, roots 0.5722416 and
  0.3325958; `v[10] = 0.659695654069`.
- Continuous PI (`K_p = 1`, `K_i = 2000 s⁻¹`) on the same RC, unit
  step: `v(t) = 1 − e^{−1000t}·cos(1000t)`; `v(0.5 ms) = 0.467719269784`,
  `v(1 ms) = 0.801233889654`.
- Hysteresis current control of an RL (10 V, 1 Ω, 1 mH, ideal freewheel
  path, band 5 A ± 0.5 A): rise and fall times `−τ·ln(4.5/5.5) =
  200.6707 µs` each, period 401.3414 µs.

## 1. Characterisation (no behaviour change)
- [x] 1.1 Tests of today's behaviour: a three-block chain inserted in reverse order differs from the same chain inserted in order (records the size of the difference); a channel read before it is written reads 0
- [x] 1.2 Tick-count tests on every engine for one loop (10 kHz, 20 ms: 200 ticks due) and two rates (2 kHz PI + 20 kHz monitor, 20 ms: 40 and 400 due); expected-failure where `fix-control-tick-semantics` has not landed (198 on pwl; 400 PI ticks on trbdf2)
- [x] 1.3 Count Python calls per run for a C++ chain, a `ClosedLoop` and a C block (baseline for the gate in 9.2)

## 2. Index slots and typed inputs
- [x] 2.1 Signals as slots in a flat array; `Input = Signal | Const | Probe | Time`; names kept in the builder only
- [x] 2.2 State arrays `xd`, `xc`; blocks hold slices (state stays in each block's POD, owned by the graph; flat `xd`/`xc` arrays wait for the snapshot work of 6.1 — DECISIONS G5)
- [x] 2.3 Test: results bit-identical to the string-keyed chain on the existing blockchain tests

## 3. Schedule
- [x] 3.1 Direct-feedthrough table per block kind and input
- [x] 3.2 Kahn order with deterministic ties; two-phase execution (outputs, then updates)
- [x] 3.3 Tarjan SCC; build error naming the loop's blocks
- [x] 3.4 Tests: any insertion order of the task 1.1 chain gives bit-identical results; `Gain → Sum → Gain → (back)` raises naming the three blocks, and runs once a `DelayBlock` is inserted; the build lists pairs whose order differs from insertion

## 4. Rates
- [x] 4.1 `Rate` on groups; inheritance for stateless blocks; error for a stateful block with no rate
- [x] 4.2 Integer tick counter; ticks delivered by the Phase 0 scheduling; remove the throttles at `control.py:1350-1355`, `c_block.py:387-390` and in the chain observer
- [x] 4.3 `compute_delay`: zero, a time, one sample
- [x] 4.4 Tests: tick counts of task 1.2 exact on every engine; discrete P loop `v[10] = 0.643545722992` and `v[k] → 2/3` within 1e-9 on the exact strategy and within the engine tolerance elsewhere; with `one_sample`, `v[10] = 0.659695654069`

## 5. Block library
- [ ] 5.1 Port `Comparator`, `RateLimiter`, `SampleHold`, `TransferFunction`, `StateMachine`, `OpAmp`, `DelayBlock`, `SignalMux`, `SignalDemux`, `LookupTable1D`
- [ ] 5.2 Remove the Python evaluator and the fallback at `blockchain.py:268-281`; unknown block kind is a build error
- [ ] 5.3 Tests: each ported block against its Python class on 10⁴ random inputs, bit-identical; `use_kernel=False` raises a deprecation error naming the replacement

## 6. Tick events and outputs (needs `add-unified-event-queue`)
- [ ] 6.1 Discrete groups on queue tick events; graph state in the solver snapshot
- [ ] 6.2 `Modulator` sink writing the duty register; `ControlledSource` sink; discrete `Gate`
- [ ] 6.3 Tests: realised duty within 1 ppm of the commanded one on the variable-step engines (0.4 ppm measured with scheduled ticks); `n_gate_events` equals the analytic edge count; snapshot and resume in the middle of a period reproduces the uninterrupted run bit for bit

## 7. Continuous rate (needs `add-unified-device-interface`)
- [ ] 7.1 Continuous subgraph compiled into one residual contribution (`f_c`, `h`) on the AD scalar
- [ ] 7.2 Comparator `Gate` as a state event with hysteresis
- [ ] 7.3 Piecewise-affine continuous blocks as affine regions
- [ ] 7.4 Tests: continuous PI on the RC — `v(0.5 ms) = 0.467719269784`, `v(1 ms) = 0.801233889654` within 1e-9 on the exact strategy and within 1e-6 on TR-BDF2 at rtol 1e-8; hysteresis control — rise and fall times 200.6707 µs within 1e-9 s and the ripple exactly 1.000 A within 1e-9; a saturating continuous PI crosses its limit as an event

## 8. C export
- [ ] 8.1 One expression per block kind rendered to the kernel and to C
- [ ] 8.2 `export_c`: state struct, init, per-rate step functions, rate tables, C-block shim for single-rate graphs
- [ ] 8.3 Refusals: `PythonBlock`, source-less `CBlock`, continuous blocks
- [ ] 8.4 Tests: for every block kind and for a PI + limiter + PWM graph, the export compiled with `-ffp-contract=off` and loaded through `add_c_block` reproduces the in-kernel graph bit for bit over 10⁴ ticks; the export compiles with `-std=c99 -Wall -Werror -pedantic` and contains no `malloc`

## 9. Consolidation
- [ ] 9.1 `MixedDomainBlockChain`, `bind_pi_to_switch`, `bind_pi_to_duty_callable`, `add_c_block`, YAML chains build a `ControlGraph`; closures through the `PythonBlock` adapter with a deprecation warning; remove the two composition blocks in `simulate()`
- [ ] 9.2 Measurement gate (counters): `n_block_evals = n_ticks × blocks in the group`; zero Python calls during a run of a graph with no `PythonBlock` (baseline from 1.3); for a 10 kHz controller on a dt = 1e-7 fixed run, block evaluations fall from one per step to one per tick (1000× fewer, by count)
- [ ] 9.3 C blocks on `trbdf2` and on the exact strategy: same output as on the fixed engine within the engine tolerance (refused / open-loop today)

## 10. Docs and verification
- [ ] 10.1 Control guide (rates, order, delay, continuous blocks, export), migration note for insertion-order chains, AI_GUIDE, CHANGELOG
- [ ] 10.2 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check
