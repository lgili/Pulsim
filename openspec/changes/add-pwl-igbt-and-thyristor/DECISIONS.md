# Decisions — add-pwl-igbt-and-thyristor

1. **No new device kind.** A gated device is a `Diode` (`SwitchedDiode`)
   whose `Params` carry `gate` (`None` / `Held` / `Latching`), `I_H` and the
   runtime gate bit `g`. Every engine already decides PWL diodes through
   `SwitchedDiode::decide` / `crossing_signal` on `DiodeEntry::params()`, so
   the gate reaches the pwl, trbdf2 and DC decision sites without
   touching them. Copies of the diode parameters that do not carry the
   gate: the dsed scheduler and `period_integrals` (behind the dsed
   refusal) and `make_seeded_diodes` (no engine caller). The pwl
   commutation records of a gated device (both pwl paths) are the change
   of its stamped state, at the step start it takes effect from — the
   `v − V_on` crossing an ungated diode records says nothing about a
   gate; `steady_state` reads these records for its affineness check. The gate is
   read from the user mask at the one seam where the user mask meets the
   diode bits: `DiodeEventState::combine(user)` (replaces
   `combine_masks(user, diodes.current_diode_mask(), owned)` at every pwl,
   trbdf2 and DC call site). `combine` also opens a gate-held device whose
   gate is off (hard turn-off at the edge). Ungated circuits: `apply_gates`
   returns at once; the overlay is the same word-wise operation as before.
2. **Engine column `kGatedCaps`** (a form, like the branch form): pwl,
   trbdf2 and the AC sweep run gated devices; dsed and bdf1 refuse them
   by name, `UnsupportedError(engine, 'gated_diode', device, 'Diode')`.
   dsed has its own diode list (`SchedulerDiode`), LCP shadow, zero-band
   settle and predicates; carrying the gate through all of them is a
   follow-up. bdf1 already refuses every PWL diode.
3. **trbdf2 gate edges.** A gated device's bit is a gate for the edge
   learner (`gate_bits`), and the "gate changed at a step boundary" test
   compares the user's bit with the gate the device was last decided
   under (`DiodeEventState::gate_of`), not with its stamped conduction bit.
4. **`add_igbt` stage 1** (proposal Migration Plan): `reverse_blocking`
   left unset is `False` (the old bidirectional switch, bit-identical) with
   a `FutureWarning`; it is `True` when `V_ce0 > 0` is given (a knee
   implies reverse blocking; `V_ce0 > 0` with an explicit `False` is
   refused by name). Task 8 (the flip) is the next release.
5. **TRIAC** = two latching branches `<name>_fwd` (t1 → t2) and
   `<name>_rev` (t2 → t1); `switch_indices_of(name)` returns both bits.
6. **IR hash**: a gated diode hashes its mode as structure and `I_H` as a
   value; an ungated diode hashes exactly as before.
7. **Loss / `res.i`**: unchanged code — the recorded stamped mask is the
   conduction state `c`, and the diode branch power `v·i` with
   `i = g_on (v − V_f)` is `V_f·i + i²/g_on`. The missed-commutation check
   in `_diode_state` skips "OFF under forward bias" on EVERY sample of a
   gated device: the result records the stamped state `c` but not the
   gate `g` (task 2.2 is not done), so a missed turn-on of a gated device
   is not reported. The voltage-replay fallback refuses a gated device
   (it knows no gate). `switch_specs` on a gated device is refused (its
   switching energies are not modelled; they would be dropped).
8. **Not done**: DC `initial_on` for a latch (a thyristor starts OFF);
   dsed support (refused, so task 5.7's dsed RL comparison is not run);
   the gate record (2.2); a `FutureWarning` for the YAML `igbt` default
   (the loader goes straight to C++).
9. **Reference harness**: two STRICT analytic cases, `igbt_knee` (10 A,
   V = V_ce0 + r_ce·I = 1.1 V) and `scr_halfwave_r` (α = 60°, gate edge
   off the grid, latched until i = 0). Their ceilings are hand-inserted
   into `manifests/analytic.yaml`: `freeze_analytic.py --write` rewrites
   every case (drops comments, re-measures the dsed cells), so its output
   was used for the two new entries only. dsed is recorded `refused`.
   The closed form keeps the sample stored exactly at the gate edge OFF
   (the engines store the left limit there).
   `pwl+restart` disagrees on it (2.7e-2, REFERENCE_STATUS): the restart
   rule samples every user gate at the step start (design D4), so the
   off-grid gate edge fires one step late. Not a gated-device defect: a
   plain `add_switch` on the same schedule turns on at 1.68e-4 s under
   the restart rule vs 1.67e-4 s without it (probe). An on-grid α moves
   the disagree to plain `pwl` (it samples the gate at the step end), so
   the off-grid case is kept.
10. **Records do not decide.** `combine` sets `g` and opens a gate-held
   device, so a call made only to record a mask could move the gate
   off the decision time. trbdf2's `record_post_event` now records a pure
   overlay. Its other `mask_at` calls are decision sites. `gate_of` is a
   linear scan (O(gated × diodes) per step), left as is.
11. **Merge seams with integration batch 11** (2797411):
   - The pwl complementarity mode selection and the default mode guard
     (`judge_mode`, the LCP) know no gate: they would judge a gate-off
     IGBT under forward bias as violating and turn it on. A circuit with
     a gated device skips the guard and refuses `mode_selection`
     (`feature='mode_selection'`, naming the device).
   - The circuit YAML emitter (and so `pickle` of a `Circuit`) wrote a
     gated device as a plain diode. It now writes `gate` (1 Held,
     2 Latching) and `I_H`, and reads them back through
     `add_gated_diode`; an ungated diode's YAML is unchanged.
12. **dsed support for gated devices: specification for the exact-engine
   owner** (tasks 4.3, 5.7; written here because `dsed/*` belongs to the
   exact-engine track, which is rewriting the scheduler). Until then dsed
   refuses `gated_diode` by name (decision 2).
   - **State per gated device**: the conduction bit `c` (the mode bit dsed
     already carries for a diode) and the gate bit `g` (the device's bit of
     the user mask, piecewise constant between the switch function's
     declared edges). `SchedulerDiode` must carry `gate` (`Held` /
     `Latching`), `I_H` and `g`, and `params()` must return them so every
     predicate goes through `SwitchedDiode::decide` / `crossing_signal`
     (the law the other engines use). A snapshot carries `g` beside `c`
     (decision 13).
   - **Between events the device is LTI** (`g_on` with the drop `V_f` when
     `c = 1`, `g_off` when `c = 0`): within the exact propagator's model,
     like a diode. A gate-held IGBT with `g = 1` is a switch with a knee.
   - **Scheduled events (gate edges, from `next_edge_after`)**: at an edge
     set `g` from the mask at the edge, then
     `g: 0 → 1`: `c` unchanged; the next settle may turn it on (`c = 0`,
     `v ≥ V_on`). `g: 1 → 0`: `Held` → `c = 0` at once (a forced
     interruption: the inductive-cut checks of a switch opening apply);
     `Latching` → `c` unchanged (the latch).
   - **State events (root finding on the dense output)**, with the
     crossing signal of `SwitchedDiode::crossing_signal(c, v, params)`:
     `c = 1`, `Held` or `Latching` with `g = 1`: `i = g_on (v − V_f)`
     falls through 0 → `c = 0`. `c = 1`, `Latching`, `g = 0`: `i` falls
     through `I_H` (signal `v − V_f − I_H/g_on`) → `c = 0`. `c = 0`, `g = 1`:
     `v` rises through `V_on` → `c = 1`. `c = 0`, `g = 0`: no event (the
     signal is the constant −1; such a device cannot turn on).
   - **Settle / LCP / zero band**: the mode iteration after an event decides
     with `decide(c, g, v)`; a `Held` device with `g = 0` is OFF before the
     iteration starts and is not a complementarity variable; a latched
     device with `g = 0` stays ON while `i > I_H` (its complementarity pair
     is `(i − I_H, …)`, not `(i, …)`). The settle must never turn a `g = 0`
     device on.
   - **Analytic acceptance circuits** (all STRICT in
     `benchmarks/analytic`, refused on dsed today):
     `igbt_knee`: 10 A forced through a gate-held IGBT, `V_ce0 = 1 V`,
     `r_ce = 10 mΩ`: `v = 1.1 V`, `P = 11 W` (exact at every sample).
     `scr_halfwave_r`: `v_in = 10 sin(2π·1 kHz·t)`, thyristor `V_f = 0.7 V`,
     `R_on = 10 mΩ`, `R = 10 Ω`, gate pulse of 100 µs at α = 60°
     (`make_phase_control_fn`): `v_k = (v_in − 0.7)·10/10.01` for
     `θ ∈ (α, π − asin(0.07)]` (latched after the pulse until `i = 0`),
     `v_k = v_in·10/(10 + 10⁹)` otherwise; pwl 1.46e-16, trbdf2 1.3e-11.
     Six-pulse bridge (`test_six_pulse_bridge_firing_angle_and_overlap`):
     `V_LL = 400 V`, 60 Hz, `I_d = 20 A`; `V_d = (3√2/π)·V_LL·cos α
     − (3/π)·ω·L_s·I_d` = 540.2, 467.8, 270.1 V at α = 0, 30°, 60°
     (`L_s = 0`) and 460.6 V at α = 30°, `L_s = 1 mH` (overlap
     `cos α − cos(α + μ) = 2ωL_s·I_d/(√2·V_LL)`); dsed rejects a current
     source, so task 5.7 compares with an RL load (`L` large) instead.

13. **The snapshot carries the gate bit `g`** (task 4.1 seam, batch 13).
    - Why a latched thyristor survived a resume before: its latch IS the
      conduction bit `c`, which `SolverSnapshot::mask` already carries;
      `g` is re-read from `switch_fn` by `combine` before every solve.
      The only state lost was the gate the LAST decision was taken
      under: trbdf2 compares the step's gate with `gate_of(i)` to find
      an edge, and a resume started from the default `g = 1`.
    - `mask` is now the conduction bits followed by one `g` per gated
      device, in registration order (`snapshot_mode_bits`);
      `restore_on_bits` takes either length, so an older snapshot still
      resumes. Ungated circuits: unchanged length, unchanged bits.
      The two save sites are `run_transient` and `trbdf2_transient`
      (a one-word change in the file the TR-BDF2 track owns).
    - `make_seeded_diodes` decides a gated device under the caller's gate
      and under gate OFF when none is given (it seeded a forward-biased
      thyristor ON from its voltage alone; no solver calls it today).
    - Tests (`test_gated_device_seams.py`): a resume equals the
      uninterrupted run with the gate off over a blocking thyristor, with
      the gate edge on the seam, and with the gate on over a reverse-biased
      unlatched thyristor (pwl 1.4e-12, trbdf2 1.1e-5 of the peak; these
      passed before the change too: a regression guard, not a fix of a
      visible defect); the mask carries `[c, g]`.
    - `gate_of` reads a sorted (switch index → entry) table instead of a
      scan of every diode.
    - The internal rollback copies (`diodes_snap`, `last_solved_bits`)
      stay conduction-only: the retried step's `combine` re-applies the
      gate of its own start before any decision. On trbdf2 a stale
      post-edge `g` can only make the step-start compare (`gates_differ`)
      report an edge and run one extra settle. By reading, not measured;
      left as is.

14. **Smaller items of batch 13.**
    - The pwl "mask cycle" at t = 1 ms in `test_holding_current` (gate
      held on): the ramping source reaches exactly 0 A at t_end with
      `V_th = 0`, where ON and OFF are both self-inconsistent. A plain
      diode in the same circuit warns at the same sample; with
      `V_th = 1 mV`, or t_end = 0.999 ms, neither warns. Not a gated-
      device defect: the known zero-current chatter of a `V_th = 0` diode.
    - `I_H` is settable with `with_value("<name>.I_H", v)` on a gated
      device (finite, >= 0); a plain diode still refuses it by name. It
      is not a row of the diode field table: every plain diode would
      then emit and hash it.
    - The trbdf2 noise floor of an ON conductance-form device gains
      `I_H/g_on` beside `V_th` and `V_f`: a latch held off its gate rides
      a threshold `I_H/g_on` above `V_f`. Zero for an ungated device
      (bit-identical).
    - Declared gate edges under the restart rule: the fixed engine reads
      the schedule's oracle (`next_edge_after`) only with the split;
      `discontinuity_restart` alone samples the gate at the step start,
      so an off-grid edge fires one step late (0.80 of the span at that
      sample on the `scr_halfwave_r` circuit; the split: 8e-16).
      Reproducer: `test_the_restart_rule_fires_a_declared_gate_edge_on_
      time`, strict xfail, owner the event-queue track. The reference
      circuit keeps its off-grid edge (its pwl+restart cell stays a
      documented disagree).

15. **The commanded gate is recorded** (task 2.2). `SimulationResult`
    keeps `gate_state_bits` beside `switch_state_bits`, same layout (a
    gated device's `g` at its switch index), filled by
    `record_switch_state` through `gate_mask_fn`, which each engine sets
    for its run and clears before returning (it reads the engine's diode
    state). trbdf2 trims both records when it drops a superseded event
    sample. Python: `result.gate_states` (None without a gated device).
    `_diode_state` judges OFF-under-forward-bias of a gated device again,
    where its recorded gate is on (a missed turn-on); a positive control
    forces the recorded state OFF under gate on and sees violations,
    and none with the gate recorded off.
