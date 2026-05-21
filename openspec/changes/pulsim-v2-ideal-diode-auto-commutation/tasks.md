## Phase 1 — IdealDiode device model (~0.5 days)

### 1.1 `models/ideal_diode.hpp`
- [ ] 1.1.1 `struct IdealDiode { struct Params { Real g_on,
      g_off, V_th; } }`.
- [ ] 1.1.2 `kind = BranchKind::Switch` (topology treats it
      as a switch). `num_terminals = 2`. `is_dynamic = false`
      (no trap-companion needed; stamping is identical to
      `IdealSwitch`).
- [ ] 1.1.3 Static helper `decide_next_state(bool currently_on,
      Real v_diode, Real i_diode, const Params& p) → bool`:
      - If currently_on && i_diode ≤ 0 → OFF
      - If !currently_on && v_diode ≥ V_th → ON
      - Otherwise unchanged.
- [ ] 1.1.4 `template<FloatingPoint S> static S current(...)`
      contract → returns 0 (not used; switches stamp via
      stamp_switch_fixed).

### 1.2 Tests — `tests/v2/layer5_v2/test_ideal_diode.cpp`
- [ ] 1.2.1 OFF + forward bias → ON.
- [ ] 1.2.2 ON + reverse current → OFF.
- [ ] 1.2.3 OFF + reverse bias → OFF (no change).
- [ ] 1.2.4 ON + forward current → ON (no change).
- [ ] 1.2.5 V_th = 0.7 V case: OFF + 0.5 V bias → still OFF.
- [ ] 1.2.6 V_th = 0.7 V case: OFF + 0.8 V bias → ON.

## Phase 2 — DevicePool extension (~0.25 days)

### 2.1 `pwl/device_pool.hpp` — `add_diode` and queries
- [ ] 2.1.1 New `StoredKind::Diode` enum value (extend the
      union accordingly).
- [ ] 2.1.2 `add_diode(Index branch_id, Real g_on, Real g_off,
      Real V_th = 0.0)` method.
- [ ] 2.1.3 `diode_params(Index branch_id)` returns
      `IdealDiode::Params`.
- [ ] 2.1.4 `diode_branches() const → std::span<const Index>`
      returns the list of diode branch ids in branch order.
- [ ] 2.1.5 `num_diodes() const → Size`.

### 2.2 `pwl/assemble.hpp` — dispatch for diode (treat as switch)
- [ ] 2.2.1 Add a case under `BranchKind::Switch` (the diode's
      kind):
      - If `pool.kind_of(branch.id) == StoredKind::Diode`, look
        up the diode params and stamp with `closed = mask.get(
        switch_idx)` and the diode's g_on / g_off (NOT the
        switch's g_on / g_off).
      - Else (the existing switch path): stamp with the switch's
        params.

### 2.3 Tests — `tests/v2/layer5_v2/test_device_pool_diode.cpp`
- [ ] 2.3.1 Pool with 1 diode → num_diodes() == 1,
      diode_branches() returns that branch id.
- [ ] 2.3.2 Pool with 1 switch + 1 diode → num_switches() (in
      graph) == 2 (both contribute), but pool tracks them
      distinctly.
- [ ] 2.3.3 diode_params(branch_id) returns the right
      g_on/g_off/V_th.
- [ ] 2.3.4 Wrong-kind lookup throws.

## Phase 3 — DiodeEventState (~0.5 days)

### 3.1 `pwl/diode_event_state.hpp`
- [ ] 3.1.1 `class DiodeEventState`:
      - Constructor: `(const Graph&, const DevicePool&)` — builds
        per-diode entries with switch_idx (the bit position) and
        terminal indices.
      - `current_diode_mask() const → SwitchStateMask` — mask
        with diode bits set per current state, non-diode bits 0.
      - `diode_owned_bits() const → SwitchStateMask` — mask with
        1s at diode-owned bit positions, 0s elsewhere. (Used by
        Layer 5 to mask out user-supplied diode bits.)
      - `update_from_state(const Vector& x) → bool` — recompute
        next state per diode based on x; returns true if any
        bit flipped.
      - `reset()` — all diodes OFF.
- [ ] 3.1.2 Internal entry stores anode/cathode node indices,
      switch_idx (position in mask), g_on/g_off/V_th cached
      from pool.
- [ ] 3.1.3 v_diode = v_anode - v_cathode; i_diode = g * v_diode
      where g = is_on ? g_on : g_off.

### 3.2 Tests — `tests/v2/layer5_v2/test_diode_event_state.cpp`
- [ ] 3.2.1 Empty graph → empty state, current_diode_mask is
      empty.
- [ ] 3.2.2 One diode at (n0, GND), reset state OFF → mask bit
      cleared.
- [ ] 3.2.3 After update_from_state with v_n0 = +5 V (forward
      bias above 0), bit becomes ON.
- [ ] 3.2.4 After update with v_n0 = -5 V (reverse bias) → bit
      goes back OFF (well, stays OFF — was already off).
- [ ] 3.2.5 ON diode, then update with v_n0 = -0.01 V (reverse
      bias making i_diode go negative) → flips to OFF.

## Phase 4 — Layer 5 V2 run_transient (~0.5 days)

### 4.1 `solver/run_transient.hpp` — diode-aware loop
- [ ] 4.1.1 Inside the V1 6-arg run_transient, construct a
      `DiodeEventState` from (graph, pool). If pool.num_diodes()
      == 0, the state is empty and behaviour matches V1.
- [ ] 4.1.2 Compute `diode_owned = diodes.diode_owned_bits()`
      once before the loop.
- [ ] 4.1.3 Inside the dynamic-path loop, before each
      cache.solve:
      - `user_mask = switch_fn(t)`
      - `diode_mask = diodes.current_diode_mask()`
      - `combined = combine_masks(user_mask, diode_mask, diode_owned)`
        (sets the diode bits in user_mask to the diode_mask
        values, leaving non-diode bits intact).
- [ ] 4.1.4 After cache.solve, call
      `diodes.update_from_state(x)` (in addition to
      history.update_from_state).
- [ ] 4.1.5 For the static-path (cache.dt() == 0): if diodes
      are present, they still need per-step updates. So the
      static path also constructs DiodeEventState and applies
      the same per-step logic.

### 4.2 `combine_masks` helper (in run_transient.hpp anonymous
namespace OR a new switch_state utility)
- [ ] 4.2.1 `combine_masks(user, diode, diode_owned)`:
      - Returns a new mask where bit i = (diode_owned bit i ?
        diode bit i : user bit i).
- [ ] 4.2.2 Test in test_diode_event_state.cpp or a dedicated
      small test.

### 4.3 Tests — `tests/v2/layer5_v2/test_run_transient_diode.cpp`
- [ ] 4.3.1 Circuit with 1 diode, no controlled switches; run
      transient where the diode auto-commutates twice; verify
      the recorded states match the expected on/off pattern.
- [ ] 4.3.2 Empty switch_fn throws (unchanged from V1).
- [ ] 4.3.3 Circuit with 0 diodes + 1 switch — V2 path = V1
      path (regression).

## Phase 5 — Integration: half-wave rectifier (~0.5 days)

### 5.1 `tests/v2/layer5_v2/test_integration_half_wave_rectifier.cpp`
- [ ] 5.1.1 Topology: V_source(n0, GND) → Diode(n0, n1) →
      R(n1, GND).
      V_source is a SINUSOIDAL source — implemented via
      VoltageSource with V=0 in pool + b_extra_fn that
      modulates the source's constraint row.

      Actually a cleaner approach: register the source with V=0
      and have b_extra_fn provide `-V_amp · sin(2πf·t)` on the
      source's branch-current row. This makes
      `cache.b_constant + b_extra(t) = -V_amp·sin(...)`, so the
      effective source voltage is V_amp·sin(...).
- [ ] 5.1.2 Simulate 2 cycles (33 ms at 60 Hz) at dt = 100 µs
      (= 330 samples).
- [ ] 5.1.3 Verify positive-half samples: |V_out - V_sine| <
      0.5 V for samples where the diode is ON (the half-wave
      region t mod T < T/2).
- [ ] 5.1.4 Verify negative-half samples: |V_out| < 1.0 V.
- [ ] 5.1.5 Verify mean output power within 5 % of
      V_amp² / (4·R) = 2.5 W.

## Phase 6 — Integration: boost converter (~0.75 days)

### 6.1 `tests/v2/layer5_v2/test_integration_boost.cpp`
- [ ] 6.1.1 Build the boost: V_in(12V) → L(100µH) → v_sw →
      [Q to GND, Diode to V_out] → C(100µF) || R(20Ω) → GND.
- [ ] 6.1.2 Two switches in branch order: Q first (user-
      controlled), Diode second (auto-controlled).
- [ ] 6.1.3 PWM at 100 kHz with D = 0.5. Schedule controls Q;
      diode bit is overwritten by DiodeEventState.
- [ ] 6.1.4 Simulate 5 ms (= 500 PWM periods) at dt = 100 ns.
- [ ] 6.1.5 Measure mean V_out over the last 100 PWM periods.
      Expected = V_in / (1 - D) = 24 V within 10 %.
- [ ] 6.1.6 Energy balance: mean(I_L) · V_in ≈ mean(V_out)² / R
      within 10 %.
- [ ] 6.1.7 Sanity: count diode commutations per PWM cycle.
      Should be 1-2 per cycle (turn on at switch-off, turn off
      at switch-on for CCM; OR turn off when i_L → 0 for DCM).
      Should NOT be 100s of flips per cycle (that would indicate
      chatter).

## Phase 7 — CMake + Documentation (~0.25 days)

### 7.1 CMake target `pulsim_v2_layer5_v2_tests`
- [ ] 7.1.1 New executable mirroring pulsim_v2_layer5_v1_tests
      structure.

### 7.2 `docs/pulsim-v2/layer5-v2-ideal-diode.md`
- [ ] 7.2.1 Section "The diode state machine" with the FSM
      diagram.
- [ ] 7.2.2 Section "Per-step decision: when it's good enough"
      — trap-rule accuracy + < dt latency argument.
- [ ] 7.2.3 Section "Worked example: half-wave rectifier"
      with the test code.
- [ ] 7.2.4 Section "Worked example: boost converter" with
      results.
- [ ] 7.2.5 Section "V0 limitations and follow-ups".

## Phase 8 — Validation gates

- [ ] 8.1 `pulsim_v2_layer5_v2_tests` MUST pass with ≥ 25
      assertions / 8 cases.
- [ ] 8.2 All previous layer tests stay green.
- [ ] 8.3 v1 `pulsim_tests` stays green.
- [ ] 8.4 `openspec validate
      pulsim-v2-ideal-diode-auto-commutation --strict` passes.
