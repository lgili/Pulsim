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
