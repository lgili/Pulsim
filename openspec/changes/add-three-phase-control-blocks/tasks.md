## 1. Parser + arity registration
- [x] 1.1 Add aliases for `clarke_transform`, `inverse_clarke_transform`, `park_transform`, `inverse_park_transform`, `pll`, `svm` in `component_aliases()`.
- [x] 1.2 Register node arity in `component_node_arity()` for each.
- [x] 1.3 Add the new types to the `control_types` set in `execute_mixed_domain_step`.

## 2. Block implementations
- [x] 2.1 `clarke_transform` (nodes [a, b, c]): amplitude-invariant Clarke matrix → channels alpha/beta/gamma.
- [x] 2.2 `inverse_clarke_transform` (nodes [alpha, beta, gamma]): channels a/b/c.
- [x] 2.3 `park_transform` (nodes [alpha, beta] + theta_from_channel): channels d/q/zero.
  - α/β can also come from `alpha_from_channel` / `beta_from_channel`, so a Clarke block's output chains in cleanly.
- [x] 2.4 `inverse_park_transform` (nodes [d, q] + theta_from_channel): channels alpha/beta.
  - d/q can also come from `d_from_channel` / `q_from_channel`.
- [x] 2.5 `pll`: PI loop on the q-component of a single-phase Park projection; persistent state for phase + omega integral.
- [x] 2.6 `svm`: min-max SVPWM (zero-sequence injection) → 3 phase duties.

## 3. Example benchmarks
- [x] 3.1 `three_phase_dq_decoupling.yaml` — 3-phase sine sources @ 60 Hz → PLL → Clarke → Park; channels validated against locked baseline.
- [x] 3.2 `pll_grid_sync.yaml` — PLL locks to a 60 Hz sine; emits θ, ω, lock_error as channels.
- [x] 3.3 `vector_control_open_loop.yaml` — Full chain: sine → Clarke → PLL → Park → (identity) → Inverse Park → SVM → 3 gate duties.

## 4. Trace + validation plumbing
- [x] 4.1 Extend `pulsim_python_backend._write_state_csv` to append `chan:<name>` columns for every virtual channel produced by the simulation. Enables CSV-based reference validation of channel outputs without electrical-domain probes.
- [x] 4.2 Generate baselines for the three new benchmarks.
- [x] 4.3 Run the full closed-loop dashboard; confirm no regressions on the existing 47 benches (50/50 passing after the new ones).
