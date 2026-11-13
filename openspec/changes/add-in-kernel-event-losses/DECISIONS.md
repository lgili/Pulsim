# Decisions — add-in-kernel-event-losses

Format: question → option → evidence → how to reverse.

## 1. Group 1 alone (slice 3 of the optimisation track)

→ `thermal/loss_table.hpp` (`LossTable3D`, `ConductionTable`), the
bindings, `LossTable.to_kernel()`; no kernel caller yet (group 2 hooks
the reading rule into `solver/run_transient.hpp`, a file another change
is rewriting). Evidence: 10 000 random points inside and outside the
grid equal `LossTable.__call__` with a maximum relative difference of
0 (the operations are the same, in the same order); the temperature
table returns 1.5 mJ at 75 °C and 2.5 mJ at 175 °C to 1e-12
(`python/tests/test_loss_table_kernel.py`, `core/tests/thermal/`).
Same result against Eigen 3.4.0 (the header uses no Eigen).
→ Reverse: delete the header and the two bindings.

## 2. A non-finite query is refused, not NaN

`LossTable.__call__` returns NaN for a NaN coordinate (numpy
propagates it, `np.maximum(NaN, 0)` is NaN). In the kernel a NaN energy
would be deposited into a thermal node and spread through the network
before anything reads it, so the kernel table raises
`std::invalid_argument` naming the axis ("i query is not finite"). An
infinite axis value is refused at construction for the same reason
(Python accepts it and returns NaN weights).
→ Reverse: drop the `isfinite` check in `detail::axis_weight`.

## 3. `ConductionTable` has no Python twin

`V_on(i, T_j)` is read with the same rule; its oracle is a `LossTable`
with a single-point `v` axis (the dimension collapses), 2 000 random
points within 1e-12.
