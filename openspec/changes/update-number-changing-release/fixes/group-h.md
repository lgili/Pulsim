# Group H — the fixed cost of one `simulate()` call (F-4-6, F-X-2)

Branch `release/h-call-overhead` (from `release/core-next-defaults` abbee805). Python only.

## 1. Profile (measured first)

Counter: the gate's own (`python/tests/test_simulate_overhead.py`): a fresh
interpreter, the tiny 2-switch buck, `sys.settrace` opcode events on the warm
(second) call with garbage collection disabled, here attributed to the code
object executing each opcode (`co_filename:co_qualname`). Python 3.13
framework build; extension of the release tip (`so_release_967c8bfe`); the
pre-release column is `git archive origin/feat/core-integration python`
(b9e46778) with `so_b9e46778`. Three calls:

- fixed step: `simulate(b, 2e-6, dt=1e-7, switch_fn=sw)` — gate `pwl`, ceiling 2890;
- no dt: `simulate(b, 4e-5, switch_fn=sw)` — the release routes it to the exact
  engine (pre-release: trbdf2); target 3760;
- no dt, `engine='trbdf2'` — gate `trbdf2`, ceiling 3760.

The findings' figures reproduce within a few bytecodes (3056 -> 3057, 3859 -> 3860);
the no-dt exact call measures 5609 here (5456 in FINDINGS: measured on an
earlier tree). `pulsim.exact_route` (the report-only classifier in
`_exact_route.py`) is NOT on the call's path: the route decision in `_route`
is one `_require_supported(..., 'exact')` binding call plus `_dsed_refused()`
(~150 bytecodes). The 4745 of F-X-2 is the exact ENGINE's per-call Python
(plan record, dispatcher, result build, segment report), table below.

### Fixed step: pre-release 2879, release 3057 (+178)

| pre-release (b9e46778) | bytecodes | release (abbee805) | bytecodes |
|---|---:|---|---:|
| `__init__.py:_plan_run` | 620 | `__init__.py:_plan_run` | 653 |
| `__init__.py:_finalize` | 254 | `__init__.py:_finalize` | 259 |
| `__init__.py:_route` | 198 | `__init__.py:_route` | 236 |
| `__init__.py:_run_pwl` | 170 | `__init__.py:_run_pwl` | 170 |
| `<string>:__create_fn__.<locals>.__init__` | 168 | `<string>:__create_fn__.<locals>.__init__` | 168 |
| `_control_plan.py:ControlPlan.for_fixed_step` | 135 | `_control_plan.py:ControlPlan.for_fixed_step` | 135 |
| `_circuit_ir.py:control_decls` | 120 | `_circuit_ir.py:control_decls` | 120 |
| `_circuit_ir.py:_controls` | 108 | `_circuit_ir.py:_controls` | 108 |
| `_run_options.py:refused` | 96 | `_run_options.py:refused` | 96 |
| `__init__.py:_step_loop_python_callbacks` | 75 | `__init__.py:_step_loop_python_callbacks` | 75 |
| `_control_plan.py:compose_control` | 73 | `_control_plan.py:compose_control` | 73 |
| `__init__.py:simulate` | 69 | `__init__.py:simulate` | 70 |
| `__init__.py:_warn_unlocated_edges` | 50 | `__init__.py:_warn_unlocated_edges` | 50 |
| `__init__.py:_run_on_simulation` | 46 | `__init__.py:_run_on_simulation` | 46 |
| `_control_plan.py:ControlPlan.merge_b_extra` | 45 | `_control_plan.py:ControlPlan.merge_b_extra` | 45 |

The difference (release - pre-release; |delta| >= 10):

| function | pre | release | delta |
|---|---:|---:|---:|
| `__init__.py:_route` | 198 | 236 | +38 |
| `__init__.py:_resolve_mode_auto` | 0 | 33 | +33 |
| `__init__.py:_plan_run` | 620 | 653 | +33 |
| `__init__.py:_uniform_checks` | 0 | 23 | +23 |
| `<frozen importlib._bootstrap>:_handle_fromlist` | 0 | 23 | +23 |
| `__init__.py:_mode_auto_record` | 0 | 16 | +16 |

### No dt (release: the exact engine; pre-release: trbdf2): pre-release 3705, release 5609 (+1904)

| pre-release (b9e46778) | bytecodes | release (abbee805) | bytecodes |
|---|---:|---|---:|
| `__init__.py:_plan_run` | 568 | `_plan.py:ResolvedFrom._dict` | 693 |
| `_diode_state.py:_build` | 431 | `__init__.py:_plan_run` | 566 |
| `<string>:__create_fn__.<locals>.__init__` | 216 | `_dsed_dispatch.py:run_dsed_from_builder` | 537 |
| `__init__.py:_route` | 182 | `_dsed_dispatch.py:_exact_result` | 339 |
| `__init__.py:_set_run_info` | 173 | `_exact_route.py:segments_from_exact` | 263 |
| `__init__.py:_finalize` | 162 | `_dsed_dispatch.py:_result_from_native` | 252 |
| `__init__.py:_run_trbdf2` | 155 | `_index_tricks_impl.py:AxisConcatenator.__getitem__` | 196 |
| `_result_views.py:switch_bits_by_branch` | 138 | `<string>:__create_fn__.<locals>.__init__` | 193 |
| `_circuit_ir.py:control_decls` | 120 | `__init__.py:_run_dsed` | 190 |
| `__init__.py:_trbdf2_warn_stats` | 113 | `<frozen importlib._bootstrap>:_handle_fromlist` | 161 |
| `_circuit_ir.py:_controls` | 108 | `_circuit_ir.py:control_decls` | 150 |
| `_run_options.py:refused` | 96 | `_circuit_ir.py:_controls` | 144 |
| `__init__.py:_step_loop_python_callbacks` | 75 | `__init__.py:_finalize` | 125 |
| `_control_plan.py:compose_control` | 73 | `_exact_route.py:segments_from_exact.<locals>.<genexpr>` | 120 |
| `_diode_state.py:_event_sample_mask` | 73 | `_function_base_impl.py:diff` | 106 |

The difference (release - pre-release; |delta| >= 10):

| function | pre | release | delta |
|---|---:|---:|---:|
| `_plan.py:ResolvedFrom._dict` | 0 | 693 | +693 |
| `_dsed_dispatch.py:run_dsed_from_builder` | 0 | 537 | +537 |
| `_dsed_dispatch.py:_exact_result` | 0 | 339 | +339 |
| `_exact_route.py:segments_from_exact` | 0 | 263 | +263 |
| `_dsed_dispatch.py:_result_from_native` | 0 | 252 | +252 |
| `_index_tricks_impl.py:AxisConcatenator.__getitem__` | 0 | 196 | +196 |
| `__init__.py:_run_dsed` | 0 | 190 | +190 |
| `<frozen importlib._bootstrap>:_handle_fromlist` | 23 | 161 | +138 |
| `_exact_route.py:segments_from_exact.<locals>.<genexpr>` | 0 | 120 | +120 |
| `_function_base_impl.py:diff` | 0 | 106 | +106 |
| `fromnumeric.py:_wrapreduction_any_all` | 0 | 102 | +102 |
| `<frozen importlib._bootstrap>:ModuleSpec.parent` | 6 | 105 | +99 |
| `_builder_bridge.py:CircuitBuilderAdapter._resolve_current` | 0 | 91 | +91 |
| `_dsed_dispatch.py:_shadow_h_e` | 0 | 72 | +72 |
| `__init__.py:_route.<locals>._dsed_refused` | 0 | 54 | +54 |
| `_dsed_dispatch.py:_resolve_dsed_options` | 0 | 50 | +50 |
| `_builder_bridge.py:CircuitBuilderAdapter.__init__` | 0 | 36 | +36 |
| `_circuit_ir.py:_controls` | 108 | 144 | +36 |
| `__init__.py:_resolve_mode_auto` | 0 | 30 | +30 |
| `_circuit_ir.py:control_decls` | 120 | 150 | +30 |
| `__init__.py:_uniform_checks` | 0 | 21 | +21 |
| `_dsed_dispatch.py:_has_edge_oracle` | 0 | 19 | +19 |
| `_dsed_dispatch.py:_result_from_native.<locals>.<genexpr>` | 0 | 18 | +18 |
| `__init__.py:_mode_auto_record` | 0 | 16 | +16 |
| `fromnumeric.py:_wrapfunc` | 0 | 16 | +16 |
| `fromnumeric.py:ravel` | 0 | 15 | +15 |
| `_circuit_ir.py:c_blocks_of` | 42 | 56 | +14 |
| `numeric.py:flatnonzero` | 0 | 12 | +12 |
| `_builder_bridge.py:CircuitBuilderAdapter.n_state` | 0 | 12 | +12 |
| `fromnumeric.py:all` | 0 | 10 | +10 |
| `fromnumeric.py:any` | 0 | 10 | +10 |
| `_diode_state.py:DiodeStateTrace.g` | 11 | 0 | -11 |
| `_control_plan.py:monitor_flags` | 13 | 0 | -13 |
| `_diode_state.py:DiodeStateTrace.current` | 13 | 0 | -13 |
| `_diode_state.py:DiodeStateTrace.__post_init__` | 15 | 0 | -15 |
| `_result_views.py:states_as_array` | 17 | 0 | -17 |
| `_result_views.py:_reinit_step_starts` | 17 | 0 | -17 |
| `__init__.py:_step_loop_python_callbacks` | 75 | 57 | -18 |
| `__init__.py:_warn_newton_roundoff` | 18 | 0 | -18 |
| `__init__.py:_trbdf2_warn_stats.<locals>._st` | 18 | 0 | -18 |
| `<string>:__create_fn__.<locals>.__init__` | 216 | 193 | -23 |
| `_result_views.py:step_start_states` | 23 | 0 | -23 |
| `_result_views.py:recorded_unsolved` | 24 | 0 | -24 |
| `_diode_state.py:_cache_key` | 25 | 0 | -25 |
| `_result_views.py:node_voltage_trace` | 27 | 0 | -27 |
| `__init__.py:_trbdf2_blockers` | 32 | 0 | -32 |
| `__init__.py:_trbdf2_warn_stats.<locals>.<genexpr>` | 33 | 0 | -33 |
| `_diode_state.py:diode_state_traces` | 37 | 0 | -37 |
| `__init__.py:_finalize` | 162 | 125 | -37 |
| `_methods.py:_any` | 56 | 14 | -42 |
| `_result_views.py:switch_bits_by_branch.<locals>.<lambda>` | 48 | 0 | -48 |
| `_result_views.py:recorded_switch_states` | 61 | 0 | -61 |
| `_diode_state.py:_event_sample_mask` | 73 | 0 | -73 |
| `__init__.py:_route` | 182 | 87 | -95 |
| `__init__.py:_trbdf2_warn_stats` | 113 | 0 | -113 |
| `_result_views.py:switch_bits_by_branch` | 138 | 0 | -138 |
| `__init__.py:_run_trbdf2` | 155 | 0 | -155 |
| `__init__.py:_set_run_info` | 173 | 0 | -173 |
| `_diode_state.py:_build` | 431 | 0 | -431 |

### No dt, engine='trbdf2': pre-release 3699, release 3860 (+161)

| pre-release (b9e46778) | bytecodes | release (abbee805) | bytecodes |
|---|---:|---|---:|
| `__init__.py:_plan_run` | 568 | `__init__.py:_plan_run` | 589 |
| `_diode_state.py:_build` | 431 | `_diode_state.py:_build` | 438 |
| `<string>:__create_fn__.<locals>.__init__` | 216 | `<string>:__create_fn__.<locals>.__init__` | 216 |
| `__init__.py:_route` | 175 | `__init__.py:_route` | 207 |
| `__init__.py:_set_run_info` | 173 | `__init__.py:_finalize` | 180 |
| `__init__.py:_finalize` | 162 | `__init__.py:_set_run_info` | 173 |
| `__init__.py:_run_trbdf2` | 155 | `__init__.py:_run_trbdf2` | 155 |
| `_result_views.py:switch_bits_by_branch` | 138 | `_result_views.py:switch_bits_by_branch` | 138 |
| `_circuit_ir.py:control_decls` | 120 | `_circuit_ir.py:control_decls` | 120 |
| `__init__.py:_trbdf2_warn_stats` | 113 | `__init__.py:_trbdf2_warn_stats` | 113 |
| `_circuit_ir.py:_controls` | 108 | `_circuit_ir.py:_controls` | 108 |
| `_run_options.py:refused` | 96 | `_run_options.py:refused` | 96 |
| `__init__.py:_step_loop_python_callbacks` | 75 | `__init__.py:_step_loop_python_callbacks` | 75 |
| `_control_plan.py:compose_control` | 73 | `_control_plan.py:compose_control` | 73 |
| `_diode_state.py:_event_sample_mask` | 73 | `_diode_state.py:_event_sample_mask` | 73 |

The difference (release - pre-release; |delta| >= 10):

| function | pre | release | delta |
|---|---:|---:|---:|
| `__init__.py:_route` | 175 | 207 | +32 |
| `<frozen importlib._bootstrap>:_handle_fromlist` | 23 | 46 | +23 |
| `__init__.py:_plan_run` | 568 | 589 | +21 |
| `__init__.py:_uniform_checks` | 0 | 19 | +19 |
| `__init__.py:_resolve_mode_auto` | 0 | 19 | +19 |
| `__init__.py:_finalize` | 162 | 180 | +18 |
| `__init__.py:_mode_auto_record` | 0 | 15 | +15 |

### The work list read from the difference

- fixed step (+178): `_route`'s `_dsed_refused` closure built on every call
  (+17 and its cells) and the `auto`-without-dt test; `_resolve_mode_auto` with
  a function-local `from . import _pulsim` (`_handle_fromlist` +23,
  `ModuleSpec.parent` +6) and `_mode_auto_record`; `_uniform_checks` as a call
  on every run; the tri-state defaults' branches in `_plan_run`.
- no dt (exact): `ResolvedFrom._dict` (693: `_run_dsed` reads ONE key,
  `resolved_from["output_rtol"]`, and the mapping builds all 53); eleven
  function-local relative imports (`_handle_fromlist` + `ModuleSpec.parent`
  ~260); the report-only segment record `segments_from_exact` (`np.r_`,
  per-row generator tuples, `np.flatnonzero`: ~600 with numpy's Python
  wrappers); `_exact_result`'s right-limit check (`np.diff`, `np.all`); a
  `CircuitBuilderAdapter` built twice; `_dsed_refused` evaluated twice by the
  `auto` route (the second time on the same answer); `c_blocks_of` /
  `control_decls` walking empty side lists (shared by every route).
