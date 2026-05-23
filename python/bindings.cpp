// =============================================================================
// PulsimCore v2 kernel — Python bindings via pybind11
// =============================================================================
//
// Exposes Layer 6's `CircuitBuilder`, Layer 4's
// `PwlStateSpaceCache`, Layer 5's `run_transient`, plus the
// option/result aggregates and `IdealDiode::Params`.
//
// Bound as a Python submodule `pulsim._pulsim`,
// re-exported nicely via `pulsim` (see
// `python/pulsim/v2.py`).
//
// The user code looks like:
//
//   import pulsim as p
//   b = p.CircuitBuilder()
//   b.add_voltage_source("Vin", "n0", "gnd", 5.0)
//   b.add_resistor("R1", "n0", "n1", 100.0)
//   cache = p.PwlStateSpaceCache(b.graph, b.pool)
//   cache.build()
//   opts = p.SimulationOptions(t_start=0, t_end=1e-3, dt=1e-5)
//   res  = p.run_transient(cache, b.graph, b.pool, opts,
//                            switch_fn=lambda t: p.SwitchStateMask(0))
//   print(res.states[-1])

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsim/analysis/mna_sweep.hpp"
#include "pulsim/blockchain/blocks.hpp"
#include "pulsim/blockchain/block_adapters.hpp"
#include "pulsim/blockchain/chain.hpp"
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/motors/mechanical.hpp"
#include "pulsim/motors/motor_adapters.hpp"
#include "pulsim/solver/bdf1.hpp"
#include "pulsim/switchgear/switchgear_adapters.hpp"
#include "pulsim/thermal/thermal_adapters.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/dc_strategy.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/combined_switch_fn.hpp"
#include "pulsim/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/sources/phase_shift_full_bridge_fn.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"
#include "pulsim/sources/spwm_pair_fn.hpp"
#include "pulsim/sources/three_phase_spwm_fn.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "pulsim/yaml/loader.hpp"

namespace py = pybind11;

namespace pulsim_kernel_bindings {

void init_module(py::module_& m) {
    m.doc() = "Pulsim kernel — PWL state-space cache, "
              "trapezoidal companion, Newton, and the "
              "high-level CircuitBuilder API.";
    m.attr("__version__") = "0.1.0";

    using namespace pulsim;

    // ---- SwitchStateMask -------------------------------------------------
    py::class_<topology::SwitchStateMask>(m, "SwitchStateMask",
        "Bit-vector identifying the on/off state of every "
        "switching branch. Construct with the total number "
        "of switches the cache enumerates (use "
        "`graph.num_switches`); then call `.set(i, True)` "
        "to flip individual bits.")
        .def(py::init<Size>(), py::arg("num_switches") = 0)
        .def("set",
              [](topology::SwitchStateMask& self,
                 Size i, bool v) { self.set(i, v); },
              py::arg("i"), py::arg("v"),
              "Set switch `i` to state `v` (True=ON).")
        .def("get",
              &topology::SwitchStateMask::get,
              py::arg("i"),
              "Return True if switch `i` is ON.")
        .def_property_readonly("size",
              &topology::SwitchStateMask::size,
              "Number of switches in this mask.")
        .def("to_string",
              &topology::SwitchStateMask::to_string)
        .def("__repr__",
              [](const topology::SwitchStateMask& m) {
                  return std::string("SwitchStateMask(") +
                         m.to_string() + ")";
              });

    // ---- IdealDiode::Params (smooth-blend nonlinear) ---------------------
    py::class_<models::IdealDiode::Params>(m, "IdealDiodeParams",
        "Parameters for the smooth-blend nonlinear "
        "IdealDiode model (Layer 4 V3). Used by "
        "CircuitBuilder.add_nonlinear_diode.")
        .def(py::init<>())
        .def(py::init([](Real V_F0, Real R_d,
                           Real G_off, Real kappa) {
            return models::IdealDiode::Params{
                .V_F0 = V_F0, .R_d = R_d,
                .G_off = G_off, .kappa = kappa};
        }), py::arg("V_F0") = 0.7,
            py::arg("R_d") = 0.01,
            py::arg("G_off") = 1e-9,
            py::arg("kappa") = 20.0)
        .def_readwrite("V_F0",
                        &models::IdealDiode::Params::V_F0)
        .def_readwrite("R_d",
                        &models::IdealDiode::Params::R_d)
        .def_readwrite("G_off",
                        &models::IdealDiode::Params::G_off)
        .def_readwrite("kappa",
                        &models::IdealDiode::Params::kappa);

    // ---- CircuitBuilder ---------------------------------------------------
    py::class_<builder::CircuitBuilder>(m, "CircuitBuilder",
        "High-level v2 circuit constructor. Hides the "
        "two-object Graph + DevicePool setup; users pass "
        "string node names and SI-unit parameter values.")
        .def(py::init<>())
        .def("node", &builder::CircuitBuilder::node,
              py::arg("name"),
              "Look up or create a node by name. "
              "\"gnd\" / \"GND\" / \"0\" alias to ground.")
        .def("add_voltage_source",
              &builder::CircuitBuilder::add_voltage_source,
              py::arg("name"), py::arg("from"),
              py::arg("to"), py::arg("V"),
              py::return_value_policy::reference,
              "Add a DC voltage source. V in volts.")
        .def("add_current_source",
              &builder::CircuitBuilder::add_current_source,
              py::arg("name"), py::arg("from"),
              py::arg("to"), py::arg("I"),
              py::return_value_policy::reference,
              "Add a DC current source. I in amperes. "
              "EE convention: I flows OUT of `from` (the "
              "+ terminal) into the external circuit, "
              "back to `to`. Does NOT add a branch-current "
              "unknown — the current is fixed at I.")
        .def("add_pwm_voltage_source",
              &builder::CircuitBuilder::add_pwm_voltage_source,
              py::arg("name"), py::arg("from"),
              py::arg("to"),
              py::arg("v_high"), py::arg("v_low"),
              py::arg("frequency"), py::arg("duty"),
              py::arg("phase") = 0.0,
              py::return_value_policy::reference,
              "Add a PWM voltage source. Square wave "
              "between v_high and v_low at `frequency` "
              "with `duty` ∈ [0, 1] cycle. The time-"
              "varying value is automatically overlaid by "
              "run_transient — no b_extra_fn lambda needed.")
        .def("add_sine_voltage_source",
              &builder::CircuitBuilder::add_sine_voltage_source,
              py::arg("name"), py::arg("from"),
              py::arg("to"),
              py::arg("v_dc"), py::arg("v_amplitude"),
              py::arg("frequency"), py::arg("phase") = 0.0,
              py::return_value_policy::reference,
              "Add a sinusoidal AC voltage source. Output "
              "is v_dc + v_amplitude·sin(2π·f·t + φ). "
              "`phase` is in RADIANS. The time-varying "
              "value is automatically overlaid by "
              "run_transient — no b_extra_fn lambda needed. "
              "Used for AC mains analysis, rectifier "
              "studies, audio-amp testing.")
        .def("add_pulse_voltage_source",
              &builder::CircuitBuilder::add_pulse_voltage_source,
              py::arg("name"), py::arg("from"),
              py::arg("to"),
              py::arg("v_initial"), py::arg("v_pulsed"),
              py::arg("t_start"), py::arg("pulse_width"),
              py::arg("period") = 0.0,
              py::arg("rise_time") = 0.0,
              py::arg("fall_time") = 0.0,
              py::return_value_policy::reference,
              "Add a pulse / step voltage source. Output "
              "is v_initial for t < t_start, v_pulsed for "
              "t ∈ [t_start + rise_time, t_start + rise + "
              "pulse_width), with optional linear ramps "
              "rise_time / fall_time (SPICE-style PULSE). "
              "If period > 0, repeats. Default rise/fall = 0 "
              "→ instantaneous transition (V12 backward "
              "compat).")
        .def("add_vcvs",
              &builder::CircuitBuilder::add_vcvs,
              py::arg("name"),
              py::arg("in_pos"),  py::arg("in_neg"),
              py::arg("out_pos"), py::arg("out_neg"),
              py::arg("gain"),
              py::return_value_policy::reference,
              "Add a voltage-controlled voltage source: "
              "V(out_pos) - V(out_neg) = gain · (V(in_pos) "
              "- V(in_neg)). Linear device — no Newton "
              "refresh needed.")
        .def("add_op_amp_ideal",
              &builder::CircuitBuilder::add_op_amp_ideal,
              py::arg("name"),
              py::arg("in_pos"), py::arg("in_neg"),
              py::arg("out"),
              py::arg("gain") = 1e5,
              py::return_value_policy::reference,
              "Add an ideal op-amp (single-ended output to "
              "gnd, high gain). Combine with negative "
              "feedback to enforce V_in_pos ≈ V_in_neg.")
        .def("add_igbt_level1",
              &builder::CircuitBuilder::add_igbt_level1,
              py::arg("name"),
              py::arg("collector"), py::arg("emitter"),
              py::arg("gate"),
              py::arg("V_CE_sat") = 1.5,
              py::arg("R_CE_sat") = 0.05,
              py::arg("V_T")      = 5.0,
              py::arg("kappa")    = 10.0,
              py::return_value_policy::reference,
              "Add a 3-terminal IGBT Level 1 (linear-"
              "conduction model with cutoff sigmoid). "
              "Collector-emitter is a Nonlinear branch; "
              "gate is an ideal voltage reference (no gate "
              "current). On-state I_C = (V_CE-V_CE_sat) / "
              "R_CE_sat. Use `run_transient(..., enable_"
              "nonlinear_refresh=True)` to stamp the "
              "device per Newton iteration.")
        .def("add_mosfet_level1",
              &builder::CircuitBuilder::add_mosfet_level1,
              py::arg("name"),
              py::arg("drain"), py::arg("source"),
              py::arg("gate"),
              py::arg("K"), py::arg("V_T"),
              py::arg("lambda_") = 0.02,
              py::arg("kappa")   = 15.0,
              py::arg("with_body_diode") = false,
              py::return_value_policy::reference,
              "Add a 3-terminal SH1 MOSFET (Shichman-Hodges "
              "Level 1). Cutoff/triode/saturation regions "
              "blended via sigmoid for C¹-smooth Newton "
              "convergence. K [A/V²], V_T [V], lambda_ [1/V] "
              "channel-length modulation, kappa [1/V] sigmoid "
              "sharpness. Drain → source is a Nonlinear "
              "branch; gate is a node reference (no gate "
              "current — ideal gate). Set "
              "`with_body_diode=True` (proposal #3.1) to "
              "also add an anti-parallel SwitchedDiode "
              "(source→drain) — needed for inductive-load "
              "switching to keep V_DS bounded. Call "
              "`run_transient` with "
              "`enable_nonlinear_refresh=True` so the Newton "
              "loop stamps the MOSFET each iteration.")
        .def("add_resistor",
              &builder::CircuitBuilder::add_resistor,
              py::arg("name"), py::arg("from"),
              py::arg("to"), py::arg("R_ohms"),
              py::return_value_policy::reference,
              "Add a linear resistor. R in ohms.")
        .def("add_capacitor",
              &builder::CircuitBuilder::add_capacitor,
              py::arg("name"), py::arg("from"),
              py::arg("to"), py::arg("C_farads"),
              py::return_value_policy::reference,
              "Add a linear capacitor. C in farads.")
        .def("add_inductor",
              &builder::CircuitBuilder::add_inductor,
              py::arg("name"), py::arg("from"),
              py::arg("to"), py::arg("L_henries"),
              py::return_value_policy::reference,
              "Add a linear inductor. L in henries.")
        .def("add_saturable_inductor",
              &builder::CircuitBuilder::add_saturable_inductor,
              py::arg("name"), py::arg("from"),
              py::arg("to"),
              py::arg("L_0"), py::arg("I_sat"),
              py::arg("L_residual") = 0.0,
              py::return_value_policy::reference,
              "Add a V17 nonlinear saturable inductor: the "
              "small-signal inductance is L_0 at i=0 and "
              "drops smoothly toward L_residual as |i| → I_sat. "
              "Requires Newton refresh — `simulate()` enables "
              "it automatically.")
        .def("add_diode",
              &builder::CircuitBuilder::add_diode,
              py::arg("name"), py::arg("anode"),
              py::arg("cathode"), py::arg("g_on"),
              py::arg("g_off"), py::arg("V_th") = 0.0,
              py::return_value_policy::reference,
              "Add a binary switched diode (V2's "
              "SwitchedDiode). g_on/g_off in siemens, "
              "V_th in volts.")
        .def("add_nonlinear_diode",
              &builder::CircuitBuilder::add_nonlinear_diode,
              py::arg("name"), py::arg("anode"),
              py::arg("cathode"), py::arg("params"),
              py::return_value_policy::reference,
              "Add a smooth-blend nonlinear diode "
              "(V3's IdealDiode, AD-driven).")
        .def("add_switch",
              &builder::CircuitBuilder::add_switch,
              py::arg("name"), py::arg("from"),
              py::arg("to"), py::arg("g_on"),
              py::arg("g_off"),
              py::return_value_policy::reference,
              "Add a controlled switch (driven by switch_fn "
              "at simulation time).")
        // ---- Layer 2 V1: power-device helpers ----
        .def("add_mosfet",
              &builder::CircuitBuilder::add_mosfet,
              py::arg("name"), py::arg("drain"),
              py::arg("source"),
              py::arg("R_on") = 1e-3,
              py::arg("R_off") = 1e9,
              py::return_value_policy::reference,
              "Add an n-channel power MOSFET as a "
              "controlled switch (drain → source). "
              "Defaults: R_on=1mΩ, R_off=1GΩ.")
        .def("add_mosfet_with_body_diode",
              &builder::CircuitBuilder::add_mosfet_with_body_diode,
              py::arg("name"), py::arg("drain"),
              py::arg("source"),
              py::arg("R_on") = 1e-3,
              py::arg("R_off") = 1e9,
              py::arg("V_F") = 0.7,
              py::arg("g_on_diode") = 1e3,
              py::arg("g_off_diode") = 1e-9,
              py::return_value_policy::reference,
              "Add an n-channel power MOSFET WITH "
              "intrinsic anti-parallel body diode. Adds "
              "TWO branches: switch (drain→source) + "
              "diode (source→drain).")
        .def("add_igbt",
              &builder::CircuitBuilder::add_igbt,
              py::arg("name"), py::arg("collector"),
              py::arg("emitter"),
              py::arg("R_on") = 10e-3,
              py::arg("R_off") = 1e9,
              py::return_value_policy::reference,
              "Add an IGBT as a controlled switch "
              "(collector → emitter). Defaults: "
              "R_on=10mΩ, R_off=1GΩ. No anti-parallel "
              "diode by default.")
        .def("add_transformer",
              &builder::CircuitBuilder::add_transformer,
              py::arg("name"),
              py::arg("p_from"), py::arg("p_to"),
              py::arg("s_from"), py::arg("s_to"),
              py::arg("L_p"), py::arg("L_s"),
              py::arg("k") = 1.0,
              py::return_value_policy::reference,
              "Add a two-winding linear transformer. "
              "Creates two coupled inductors with "
              "mutual inductance M = k·√(L_p·L_s). "
              "k=1 (default) for ideal coupling; "
              "k∈[0.9, 0.99] for realistic leakage.")
        .def_property_readonly("graph",
              &builder::CircuitBuilder::graph,
              py::return_value_policy::reference_internal,
              "Const ref to the internal Graph.")
        .def_property_readonly("pool",
              &builder::CircuitBuilder::pool,
              py::return_value_policy::reference_internal,
              "Const ref to the internal DevicePool.")
        .def_property_readonly("num_branches",
              &builder::CircuitBuilder::num_branches,
              "Number of branches added so far.")
        .def("node_id_of",
              &builder::CircuitBuilder::node_id_of,
              py::arg("name"),
              "Return the node index for `name`. Throws if "
              "the name was never registered.");

    // ---- Graph / DevicePool (opaque handles) -----------------------------
    //
    // We only need lvalue refs to feed into PwlStateSpaceCache /
    // run_transient. Bind as opaque classes — users don't
    // construct them directly (they go through CircuitBuilder).
    py::class_<topology::Graph>(m, "Graph",
        "Pulsim topology graph. Build via "
        "CircuitBuilder; access via builder.graph.")
        .def_property_readonly("num_nodes",
              &topology::Graph::num_nodes)
        .def_property_readonly("num_branches",
              &topology::Graph::num_branches)
        .def_property_readonly("num_switches",
              &topology::Graph::num_switches,
              "Count of branches with kind = Switch — "
              "this is the bit-width the PWL cache "
              "enumerates over. Use it as the argument "
              "to SwitchStateMask(num_switches).")
        // ground() is a static method on the C++ side
        // (returns a compile-time sentinel). Expose it as
        // a CALLABLE method on the Python instance for
        // ergonomic access — pybind11 needs `static`
        // because the C++ free function takes no `this`.
        .def_static("ground", &topology::Graph::ground);

    py::class_<pwl::DevicePool>(m, "DevicePool",
        "Pulsim device parameter pool. Build via "
        "CircuitBuilder; access via builder.pool.")
        .def("has_nonlinear_devices",
              &pwl::DevicePool::has_nonlinear_devices,
              "Return True if any registered branch is a "
              "smooth-blend IdealDiode, SH1 MOSFET, Level 1 "
              "IGBT, or saturable inductor (i.e. requires "
              "Newton iteration). Used by the Python "
              "`simulate()` wrapper to auto-enable the "
              "nonlinear-refresh pass.")
        .def("branch_var_id_for_inductor",
              &pwl::DevicePool::branch_var_id_for_inductor,
              py::arg("branch_id"), py::arg("graph"),
              "Resolve the state-vector index for an inductor "
              "(or saturable inductor) branch's i_L unknown. "
              "Throws if `branch_id` is not registered as an "
              "inductor.")
        .def("branch_var_id_for_source",
              &pwl::DevicePool::branch_var_id_for_source,
              py::arg("branch_id"), py::arg("graph"),
              "Resolve the state-vector index for a voltage-"
              "source branch's i_source unknown (also works "
              "for PWM/Sine/Pulse/VCVS sources, since they "
              "share the same branch-var numbering).")
        .def("state_size",
              &pwl::DevicePool::state_size,
              py::arg("graph"),
              "Total state-vector size for this pool + graph "
              "(= num_active_nodes + num_sources + num_inductors).");

    // ---- PwlStateSpaceCache ----------------------------------------------
    py::class_<pwl::PwlStateSpaceCache>(m, "PwlStateSpaceCache",
        "PWL state-space cache. Pre-factors the MNA matrix "
        "for every reachable switch combination + dt.")
        .def(py::init<const topology::Graph&,
                       const pwl::DevicePool&>(),
              py::arg("graph"), py::arg("pool"),
              py::keep_alive<1, 2>(),
              py::keep_alive<1, 3>())
        .def("build",
              [](pwl::PwlStateSpaceCache& self, Real dt) {
                  self.build(dt);
              }, py::arg("dt") = 0.0,
              "Build the PWL cache eagerly. dt=0 means "
              "static-only (no trap companion).")
        .def("dt", &pwl::PwlStateSpaceCache::dt);

    // ---- SimulationOptions / SimulationResult ----------------------------
    using namespace pulsim::solver;

    py::class_<SimulationOptions>(m, "SimulationOptions",
        "Inputs to run_transient: time window + dt + "
        "Newton / event-iteration knobs.")
        .def(py::init<>())
        .def(py::init([](Real t_start, Real t_end, Real dt) {
            SimulationOptions opts;
            opts.t_start = t_start;
            opts.t_end   = t_end;
            opts.dt      = dt;
            return opts;
        }), py::arg("t_start"), py::arg("t_end"),
            py::arg("dt"))
        .def_readwrite("t_start",
                        &SimulationOptions::t_start)
        .def_readwrite("t_end",
                        &SimulationOptions::t_end)
        .def_readwrite("dt", &SimulationOptions::dt)
        .def_readwrite("max_event_iterations",
                        &SimulationOptions::max_event_iterations)
        .def_readwrite("max_newton_iterations",
                        &SimulationOptions::max_newton_iterations)
        .def_readwrite("tol_newton_dx",
                        &SimulationOptions::tol_newton_dx)
        .def_readwrite("tol_newton_res",
                        &SimulationOptions::tol_newton_res)
        .def_readwrite("enable_newton_line_search",
                        &SimulationOptions::enable_newton_line_search)
        .def_readwrite("enable_newton_lm",
                        &SimulationOptions::enable_newton_lm)
        .def_readwrite("enable_substep_state_correction",
                        &SimulationOptions::enable_substep_state_correction)
        .def("valid", &SimulationOptions::valid)
        .def("expected_step_count",
              &SimulationOptions::expected_step_count);

    py::class_<CommutationEvent>(m, "CommutationEvent",
        "Sub-step commutation timing event (V2.2 + V3).")
        .def_readonly("t_estimated",
                       &CommutationEvent::t_estimated)
        .def_readonly("branch_id",
                       &CommutationEvent::branch_id)
        .def_readonly("new_state",
                       &CommutationEvent::new_state);

    py::class_<SimulationResult>(m, "SimulationResult",
        "Output of run_transient: parallel `times` and "
        "`states` arrays, plus event diagnostics.")
        .def_readonly("times", &SimulationResult::times)
        .def_readonly("states", &SimulationResult::states)
        .def_readonly("event_iteration_count",
                       &SimulationResult::event_iteration_count)
        .def_readonly("commutation_events",
                       &SimulationResult::commutation_events)
        .def("num_steps", &SimulationResult::num_steps);

    // ---- run_transient ---------------------------------------------------
    //
    // V0 binding: takes the cache + graph + pool + opts +
    // switch_fn (required) and optional b_extra_fn /
    // start_from_dc_op. We omit the NonlinearRefreshFn
    // overload for V0 — Python users wanting nonlinear
    // refresh can use the AD-stack via direct C++ at this
    // stage (V1 add-on).
    // ---- YAML loader (Layer 8) -------------------------------------------
    py::class_<yaml::LoadedCircuit>(m, "LoadedCircuit",
        "Result of `load_yaml_*`: a populated "
        "`CircuitBuilder` + `SimulationOptions`.")
        // `builder` is move-only (Graph has deleted copy)
        // so we expose by-reference. The LoadedCircuit
        // OWNS the builder; this view stays valid as
        // long as the LoadedCircuit lives.
        .def_property_readonly("builder",
            [](yaml::LoadedCircuit& self)
                -> builder::CircuitBuilder& {
                return self.builder;
            },
            py::return_value_policy::reference_internal)
        .def_readwrite("options",
                        &yaml::LoadedCircuit::options);

    m.def("load_yaml_string",
        &yaml::load_string, py::arg("yaml_text"),
        "Parse a YAML circuit description from an in-memory "
        "string. Returns LoadedCircuit(builder, options).");
    m.def("load_yaml_file",
        &yaml::load_file, py::arg("path"),
        "Parse a YAML circuit description from disk. "
        "Returns LoadedCircuit(builder, options).");

    // ---- PWM switch_fn helper (Layer 2 V5) -------------------------------
    //
    // SMPS users no longer need to write `lambda t: ...`
    // boilerplate for the SWITCH side of PWM. The helper
    // returns a Python callable produced by std::function ⇒
    // py::function conversion; it can be passed directly as
    // `switch_fn=` to `run_transient`.
    m.def("make_pwm_switch_fn",
        &sources::make_pwm_switch_fn,
        py::arg("frequency"), py::arg("duty"),
        py::arg("switch_idx"), py::arg("num_switches"),
        py::arg("phase") = 0.0,
        "Build a SwitchScheduleFn that toggles `switch_idx` "
        "ON during the first `duty · T` fraction of each "
        "period (T = 1 / frequency) and OFF for the rest. "
        "All other switch bits stay OFF. Returns a Python "
        "callable usable as `switch_fn=` in run_transient.");

    m.def("make_dead_time_pwm_pair_fn",
        &sources::make_dead_time_pwm_pair_fn,
        py::arg("frequency"), py::arg("duty"),
        py::arg("hs_switch_idx"), py::arg("ls_switch_idx"),
        py::arg("num_switches"), py::arg("dead_time"),
        py::arg("phase") = 0.0,
        "Build a SwitchScheduleFn driving a complementary "
        "HS/LS half-bridge pair at `frequency` with `duty` "
        "and symmetric `dead_time` inserted at each "
        "commutation. Shoot-through is prevented by "
        "construction (HS and LS are never both ON). Other "
        "switch bits stay OFF.");

    m.def("make_spwm_pair_fn",
        &sources::make_spwm_pair_fn,
        py::arg("carrier_frequency"),
        py::arg("modulation_frequency"),
        py::arg("modulation_index"),
        py::arg("hs_switch_idx"), py::arg("ls_switch_idx"),
        py::arg("num_switches"), py::arg("dead_time"),
        py::arg("carrier_phase") = 0.0,
        py::arg("modulation_phase") = 0.0,
        "Build a SwitchScheduleFn driving a HS/LS half-"
        "bridge with naturally-sampled SPWM: instantaneous "
        "duty = 0.5 + 0.5·M·sin(2π·f_mod·t + φ_mod). "
        "Symmetric `dead_time` is inserted at each "
        "commutation; shoot-through is impossible by "
        "construction.");

    // ---- 3-phase VSI helper (Layer 2 V8) ---------------------------------
    py::class_<sources::ThreePhaseLegIndices>(m,
        "ThreePhaseLegIndices",
        "Switch-index assignment for the 6 power devices "
        "of a 3-phase 2-level VSI. Caller must add the "
        "switches in the same order to the CircuitBuilder.")
        .def(py::init([](Size hs_a, Size ls_a,
                           Size hs_b, Size ls_b,
                           Size hs_c, Size ls_c) {
            return sources::ThreePhaseLegIndices{
                hs_a, ls_a, hs_b, ls_b, hs_c, ls_c};
        }), py::arg("hs_a"), py::arg("ls_a"),
            py::arg("hs_b"), py::arg("ls_b"),
            py::arg("hs_c"), py::arg("ls_c"))
        .def_readwrite("hs_a",
            &sources::ThreePhaseLegIndices::hs_a)
        .def_readwrite("ls_a",
            &sources::ThreePhaseLegIndices::ls_a)
        .def_readwrite("hs_b",
            &sources::ThreePhaseLegIndices::hs_b)
        .def_readwrite("ls_b",
            &sources::ThreePhaseLegIndices::ls_b)
        .def_readwrite("hs_c",
            &sources::ThreePhaseLegIndices::hs_c)
        .def_readwrite("ls_c",
            &sources::ThreePhaseLegIndices::ls_c);

    m.def("make_combined_switch_fn",
        &sources::make_combined_switch_fn,
        py::arg("num_switches"), py::arg("fns"),
        "Compose multiple SwitchScheduleFn callbacks into a "
        "single one whose mask is the bitwise OR of all "
        "sub-masks. Useful for combining e.g. a primary-"
        "side PWM helper with a sync-rectifier helper "
        "where each drives different switch indices.");

    m.def("make_phase_shift_full_bridge_fn",
        &sources::make_phase_shift_full_bridge_fn,
        py::arg("switching_frequency"),
        py::arg("phase_shift"),
        py::arg("leg_a_hs_idx"), py::arg("leg_a_ls_idx"),
        py::arg("leg_b_hs_idx"), py::arg("leg_b_ls_idx"),
        py::arg("num_switches"), py::arg("dead_time"),
        py::arg("carrier_phase") = 0.0,
        "Build a SwitchScheduleFn driving a phase-shift "
        "full-bridge: two HS/LS half-bridge legs at 50 %% "
        "duty each, with leg B lagging leg A by "
        "`phase_shift` radians. φ=0 → synchronous (no power); "
        "φ=π → anti-phase (full ±V_bus square wave on "
        "v_AB). Used by ZVS full-bridge, DAB, and resonant "
        "LLC converters.");

    m.def("make_three_phase_spwm_fn",
        &sources::make_three_phase_spwm_fn,
        py::arg("carrier_frequency"),
        py::arg("modulation_frequency"),
        py::arg("modulation_index"),
        py::arg("legs"),
        py::arg("num_switches"),
        py::arg("dead_time"),
        py::arg("modulation_phase") = 0.0,
        py::arg("carrier_phase") = 0.0,
        "Build a SwitchScheduleFn driving a 3-phase 2-level "
        "VSI with SPWM: common carrier, sine modulation "
        "references at 0°/-120°/-240° on legs A/B/C, "
        "symmetric dead-time. Shoot-through prevented on "
        "every leg by construction.");

    // Note on `nl_refresh`: pybind11's std::function adapter
    // routes the callback through Python, which serialises
    // the sparse-matrix argument BY VALUE. The C++ refresh
    // function mutates J_nl/f_nl in place, but those
    // mutations are lost across the Python boundary.
    // Workaround: expose a boolean flag that tells the
    // Python binding to construct + use the C++ refresh
    // directly (no Python roundtrip). For custom refresh
    // logic, drop down to the C++ API.
    m.def("run_transient",
        [](const pwl::PwlStateSpaceCache& cache,
           const topology::Graph& graph,
           const pwl::DevicePool& pool,
           const SimulationOptions& opts,
           SwitchScheduleFn switch_fn,
           BExtraFn b_extra_fn,
           bool start_from_dc_op,
           bool enable_nonlinear_refresh,
           StepObserverFn step_observer,
           py::object initial_state,
           ShouldContinueFn should_continue) {
            pwl::NonlinearRefreshFn nl_refresh{};
            if (enable_nonlinear_refresh) {
                nl_refresh =
                    pwl::make_combined_diode_mosfet_refresh();
            }
            // Convert optional initial_state to a Vector pointer.
            Vector x_init;
            const Vector* x_init_ptr = nullptr;
            if (!initial_state.is_none()) {
                x_init = initial_state.cast<Vector>();
                x_init_ptr = &x_init;
            }
            // RELEASE the GIL during the heavy kernel loop so the
            // GUI / main thread can run. The std::function callbacks
            // (step_observer, switch_fn, should_continue) carry
            // Python objects in their closures; pybind11 will
            // re-acquire the GIL automatically for each invocation.
            // We re-acquire the GIL HERE (end of inner scope) BEFORE
            // the std::function args are destroyed so dec_ref runs
            // with the GIL held.
            SimulationResult result;
            {
                py::gil_scoped_release rel;
                result = run_transient(cache, graph, pool, opts,
                                          switch_fn, b_extra_fn,
                                          start_from_dc_op,
                                          nl_refresh,
                                          step_observer,
                                          x_init_ptr,
                                          should_continue);
            }
            return result;
        },
        py::arg("cache"), py::arg("graph"), py::arg("pool"),
        py::arg("opts"), py::arg("switch_fn"),
        py::arg("b_extra_fn") = BExtraFn{},
        py::arg("start_from_dc_op") = false,
        py::arg("enable_nonlinear_refresh") = false,
        py::arg("step_observer") = StepObserverFn{},
        py::arg("initial_state") = py::none(),
        py::arg("should_continue") = ShouldContinueFn{},
        "Run a fixed-dt transient simulation. switch_fn(t) "
        "→ SwitchStateMask; b_extra_fn(t) → Vector adds "
        "to b_constant at each step. `step_observer(t, x)` "
        "(optional) is a void callback fired at the start "
        "of every step BEFORE switch_fn/b_extra_fn evaluate "
        "— use it to update Python-side controller state "
        "(PI, comparator, etc.) so the next switch_fn call "
        "reads the new duty. Set "
        "`enable_nonlinear_refresh=True` for circuits "
        "with smooth-blend IdealDiode / SH1 MOSFET "
        "branches (constructs the refresh inside the "
        "binding to avoid Python-roundtrip aliasing of "
        "sparse-matrix references).");

    // ---- DC operating-point strategies (Phase A.2) -----------------------
    //
    // The kernel-side compute_dc_op_with_strategy() dispatches between
    // naive, pseudo-transient, and source-stepping DC solves. The
    // Python wrapper in pulsim.compute_dc_op() chooses the strategy
    // by string; below we expose the enum + a thin function call
    // surface for direct kernel access.
    py::enum_<pwl::DCStrategy>(m, "DCStrategy",
        "DC operating-point strategy selector.")
        .value("Naive",            pwl::DCStrategy::Naive,
                "Single-shot compute_dc_op — fastest, fails on stiff problems.")
        .value("PseudoTransient",  pwl::DCStrategy::PseudoTransient,
                "Modified Newton with dt regularisation — globally convergent.")
        .value("SourceStepping",   pwl::DCStrategy::SourceStepping,
                "Source-amplitude homotopy from α=0 to α=1 in n_steps.")
        .value("Auto",             pwl::DCStrategy::Auto,
                "Try naive → pseudo-trans → source-stepping in order.");

    m.def("compute_dc_op_with_strategy",
        [](const topology::Graph& graph,
            const pwl::DevicePool& pool,
            const topology::SwitchStateMask& mask,
            pwl::DCStrategy strategy,
            Real t_eval,
            Real pt_dt_init,
            Real pt_dt_max,
            Size pt_max_iters,
            Real pt_tol_res,
            Size ss_n_steps) {
            pwl::PseudoTransientConfig pt;
            pt.dt_init = pt_dt_init;
            pt.dt_max  = pt_dt_max;
            pt.max_iters = pt_max_iters;
            pt.tol_res = pt_tol_res;
            pwl::SourceSteppingConfig ss;
            ss.n_steps = ss_n_steps;
            return pwl::compute_dc_op_with_strategy(
                graph, pool, mask, strategy, t_eval, pt, ss);
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("strategy") = pwl::DCStrategy::Auto,
        py::arg("t_eval") = Real{0},
        py::arg("pt_dt_init") = Real{1.0},
        py::arg("pt_dt_max") = Real{1e10},
        py::arg("pt_max_iters") = Size{500},
        py::arg("pt_tol_res") = Real{1e-7},
        py::arg("ss_n_steps") = Size{10},
        "Compute the DC operating-point state vector with strategy "
        "selection. Returns a numpy array of length "
        "pool.state_size(graph).");

    // ---- Naive DC entry point (matches the Python wrapper's `naive`) ----
    m.def("compute_dc_op",
        [](const topology::Graph& graph,
            const pwl::DevicePool& pool,
            const topology::SwitchStateMask& mask,
            Real t_eval) {
            return pwl::compute_dc_op(graph, pool, mask, t_eval);
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("t_eval") = Real{0},
        "Naive single-shot DC operating-point solve. Returns a numpy "
        "array of length pool.state_size(graph). Use "
        "compute_dc_op_with_strategy(...) for stiff circuits.");

    // ---- MNA-linearised AC sweep (Phase A.3) ----------------------------
    //
    // Kernel-side complex AC sweep: linearise the MNA matrix at the
    // operating point, solve (jωI − A) X = B per frequency via
    // Eigen::SparseLU<complex<Real>>. ~200× faster than swept-sine.
    py::class_<analysis::MnaSweepResult>(m, "MnaSweepKernelResult",
        "Kernel-side MNA AC sweep result. `freqs` and `H` are "
        "parallel arrays; `H` is complex.")
        .def_readonly("freqs", &analysis::MnaSweepResult::freqs)
        .def_readonly("H",     &analysis::MnaSweepResult::H);

    m.def("run_mna_sweep_kernel",
        [](const topology::Graph& graph,
            const pwl::DevicePool& pool,
            const topology::SwitchStateMask& mask,
            const std::vector<Real>& freqs,
            Size input_state_idx,
            Size output_node_idx) {
            return analysis::run_mna_sweep(
                graph, pool, mask, freqs,
                input_state_idx, output_node_idx);
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("freqs"), py::arg("input_state_idx"),
        py::arg("output_node_idx"),
        "Direct kernel AC sweep via complex sparse LU at each "
        "frequency. Returns a MnaSweepKernelResult with parallel "
        "freqs/H arrays.");

    // ---- Control blocks (Phase A.1 in kernel) ----------------------------
    //
    // Header-only C++ implementations of the mixed-domain control blocks
    // exposed by the Python `pulsim.control` module. The Python
    // wrappers in `python/pulsim/control.py` remain the user-facing
    // API; these bindings let the BlockChain executor run the blocks
    // via direct C++ calls when the kernel is available.
    using namespace pulsim::blockchain;
    auto cls_gain = py::class_<Gain>(m, "CxxGain", "C++ Gain block.")
        .def(py::init<>())
        .def_readwrite("k", &Gain::k)
        .def("reset", &Gain::reset)
        .def("update", &Gain::update, py::arg("x"));
    (void)cls_gain;

    py::class_<Subtract>(m, "CxxSubtract", "C++ Subtract block.")
        .def(py::init<>())
        .def("reset", &Subtract::reset)
        .def("update", &Subtract::update,
              py::arg("a"), py::arg("b"));

    py::class_<FirstOrderLowPass>(m, "CxxFirstOrderLowPass",
        "C++ first-order IIR low-pass filter.")
        .def(py::init<>())
        .def_readwrite("tau", &FirstOrderLowPass::tau)
        .def_readwrite("y",   &FirstOrderLowPass::y)
        .def("reset", &FirstOrderLowPass::reset)
        .def("update", &FirstOrderLowPass::update,
              py::arg("input_value"), py::arg("dt"));

    py::class_<PIController>(m, "CxxPIController",
        "C++ PI controller with anti-windup (trapezoidal integration).")
        .def(py::init<>())
        .def_readwrite("Kp", &PIController::Kp)
        .def_readwrite("Ki", &PIController::Ki)
        .def_readwrite("output_min", &PIController::output_min)
        .def_readwrite("output_max", &PIController::output_max)
        .def_readwrite("integral",   &PIController::integral)
        .def_readwrite("prev_error", &PIController::prev_error)
        .def("reset", &PIController::reset)
        .def("update", &PIController::update,
              py::arg("setpoint"), py::arg("measured"), py::arg("dt"));

    py::class_<PIDController>(m, "CxxPIDController",
        "C++ PID controller with anti-windup + derivative filter.")
        .def(py::init<>())
        .def_readwrite("Kp", &PIDController::Kp)
        .def_readwrite("Ki", &PIDController::Ki)
        .def_readwrite("Kd", &PIDController::Kd)
        .def_readwrite("tau_d", &PIDController::tau_d)
        .def_readwrite("output_min", &PIDController::output_min)
        .def_readwrite("output_max", &PIDController::output_max)
        .def("reset", &PIDController::reset)
        .def("update", &PIDController::update,
              py::arg("setpoint"), py::arg("measured"), py::arg("dt"));

    py::class_<PwmGenerator>(m, "CxxPwmGenerator",
        "C++ PWM generator (sawtooth comparator, 0/1 output).")
        .def(py::init<>())
        .def_readwrite("frequency", &PwmGenerator::frequency)
        .def_readwrite("phase",     &PwmGenerator::phase)
        .def("reset", &PwmGenerator::reset)
        .def("update", &PwmGenerator::update,
              py::arg("duty"), py::arg("t"));

    py::class_<Limiter>(m, "CxxLimiter", "C++ hard-clamp limiter.")
        .def(py::init<>())
        .def_readwrite("min_v", &Limiter::min_v)
        .def_readwrite("max_v", &Limiter::max_v)
        .def("reset", &Limiter::reset)
        .def("update", &Limiter::update, py::arg("x"));

    py::class_<Integrator>(m, "CxxIntegrator", "C++ integrator with clamp.")
        .def(py::init<>())
        .def_readwrite("gain", &Integrator::gain)
        .def_readwrite("output_min", &Integrator::output_min)
        .def_readwrite("output_max", &Integrator::output_max)
        .def_readwrite("y", &Integrator::y)
        .def("reset", &Integrator::reset)
        .def("update", &Integrator::update,
              py::arg("x"), py::arg("dt"));

    py::class_<ClarkeTransform>(m, "CxxClarkeTransform",
        "C++ Clarke transform (abc → αβ0).")
        .def(py::init<>())
        .def("reset", &ClarkeTransform::reset)
        .def("update", [](const ClarkeTransform& self,
                           Real a, Real b, Real c) {
            auto o = self.update(a, b, c);
            return py::make_tuple(o.alpha, o.beta, o.zero);
        }, py::arg("a"), py::arg("b"), py::arg("c"));

    py::class_<ParkTransform>(m, "CxxParkTransform",
        "C++ Park transform (αβ → dq).")
        .def(py::init<>())
        .def("reset", &ParkTransform::reset)
        .def("update", [](const ParkTransform& self,
                           Real alpha, Real beta, Real theta) {
            auto o = self.update(alpha, beta, theta);
            return py::make_tuple(o.d, o.q);
        }, py::arg("alpha"), py::arg("beta"), py::arg("theta"));

    py::class_<InverseParkTransform>(m, "CxxInverseParkTransform",
        "C++ inverse Park transform (dq → αβ).")
        .def(py::init<>())
        .def("reset", &InverseParkTransform::reset)
        .def("update", [](const InverseParkTransform& self,
                           Real d, Real q, Real theta) {
            auto o = self.update(d, q, theta);
            return py::make_tuple(o.alpha, o.beta);
        }, py::arg("d"), py::arg("q"), py::arg("theta"));

    py::class_<SpaceVectorModulator>(m, "CxxSpaceVectorModulator",
        "C++ centered space-vector modulator (αβ → 3 duty cycles).")
        .def(py::init<>())
        .def_readwrite("v_dc", &SpaceVectorModulator::v_dc)
        .def("reset", &SpaceVectorModulator::reset)
        .def("update", [](const SpaceVectorModulator& self,
                           Real v_alpha, Real v_beta) {
            auto o = self.update(v_alpha, v_beta);
            return py::make_tuple(o.da, o.db, o.dc);
        }, py::arg("v_alpha"), py::arg("v_beta"));

    py::class_<PLL>(m, "CxxPLL",
        "C++ phase-locked loop on αβ (cross-product PD + PI on q).")
        .def(py::init<>())
        .def_readwrite("f_nominal", &PLL::f_nominal)
        .def_readwrite("Kp", &PLL::Kp)
        .def_readwrite("Ki", &PLL::Ki)
        .def_readwrite("theta",    &PLL::theta)
        .def_readwrite("omega",    &PLL::omega)
        .def_readwrite("integral", &PLL::integral)
        .def("reset", &PLL::reset)
        .def("update", [](PLL& self, Real va, Real vb, Real dt) {
            auto o = self.update(va, vb, dt);
            return py::make_tuple(o.theta, o.omega, o.freq);
        }, py::arg("v_alpha"), py::arg("v_beta"), py::arg("dt"));

    // ====================================================================
    // BlockChain executor — Phase A.1 stage 2 / C++ port
    // ====================================================================
    //
    // The Python `MixedDomainBlockChain` becomes a builder + thin
    // wrapper around the C++ `BlockChain`. All per-step work happens
    // in C++ — no Python roundtrip cost per simulation step.

    py::class_<InputRef>(m, "CxxInputRef",
        "Where to read a block input each step.")
        .def_static("const_value", &InputRef::from_const, py::arg("v"))
        .def_static("channel",     &InputRef::from_channel, py::arg("name"))
        .def_static("node",        &InputRef::from_node, py::arg("name"))
        .def_static("node_idx",
            [](Index idx) {
                InputRef r;
                r.kind = InputRef::Kind::Node;
                r.name = "";
                r.node_idx = idx;
                return r;
            }, py::arg("idx"),
            "Build a pre-resolved Node InputRef — skips name lookup.")
        .def_static("time",        &InputRef::from_time)
        .def_static("dt",          &InputRef::from_dt);

    py::class_<BlockChain>(m, "CxxBlockChain",
        "C++ block chain executor. Build via add_<block>(...) "
        "methods, bind nodes, then pass make_step_observer(dt) to "
        "simulate().")
        .def(py::init<>())
        .def("size", &BlockChain::size)
        .def("reset", &BlockChain::reset)
        .def("get_channel", &BlockChain::get_channel, py::arg("name"))
        .def("set_channel", [](BlockChain& self, const std::string& name,
                                   Real value) {
            self.channels()[name] = value;
        }, py::arg("name"), py::arg("value"))
        .def("bind_nodes", [](BlockChain& self,
                                const std::function<Index(const std::string&)>&
                                    node_id_of) {
            self.bind_nodes(node_id_of);
        }, py::arg("node_id_of"),
            "Resolve every InputRef::Node by calling node_id_of(name).")
        .def("make_step_observer", [](BlockChain& self, Real dt) {
            return self.make_step_observer(dt);
        }, py::arg("dt"),
            "Return a step_observer(t, x) callable that the kernel "
            "uses to invoke the chain each step.")
        .def("make_b_extra_fn",
            [](BlockChain& self, Size state_size) {
                return self.make_b_extra_fn(state_size);
            }, py::arg("state_size"),
            "Return a b_extra_fn(t) callable for run_transient. "
            "Use together with make_step_observer when the chain "
            "has motor blocks that inject back-EMF / current "
            "values per step.")
        .def("record_channel", &BlockChain::record_channel,
            py::arg("name"), py::arg("reserve_n") = 0,
            "Register a channel name for per-step logging. After "
            "simulate() returns, fetch the trace via "
            "get_channel_history(name).")
        .def("get_channel_history",
            [](const BlockChain& self, const std::string& name) {
                const auto& vec = self.get_channel_history(name);
                // Return as a 1-D numpy array (copy).
                py::array_t<Real> arr(vec.size());
                if (!vec.empty()) {
                    std::memcpy(arr.mutable_data(), vec.data(),
                                  vec.size() * sizeof(Real));
                }
                return arr;
            }, py::arg("name"),
            "Return the recorded history of `name` as a 1-D numpy "
            "array. Returns an empty array if the channel wasn't "
            "registered via record_channel().")
        .def("get_recording_times",
            [](const BlockChain& self) {
                const auto& vec = self.get_recording_times();
                py::array_t<Real> arr(vec.size());
                if (!vec.empty()) {
                    std::memcpy(arr.mutable_data(), vec.data(),
                                  vec.size() * sizeof(Real));
                }
                return arr;
            },
            "Return the parallel time vector — one entry per "
            "recorded step. Same length as each channel history.")
        .def("clear_recordings", &BlockChain::clear_recordings,
            "Clear all recorded histories (keeps the registered "
            "names). Call between simulations to get a fresh trace.")
        .def("set_b_extra",
            [](BlockChain& self, Index idx, Real value) {
                // No public accessor — go through ctx; the user
                // typically sets via motor adapters, not directly.
                self.channels()["__noop"] = value;  // placeholder
                (void)idx;
            }, py::arg("idx"), py::arg("value"),
            "Diagnostic — directly set a b_extra entry. Most "
            "users should add motor blocks instead.")
        // ---- block factories — one per BlockType in blocks.hpp ----
        .def("add_gain",
            [](BlockChain& self, Real k, InputRef x, std::string out) {
                auto blk = std::make_shared<Gain>();
                blk->k = k;
                auto p = make_gain_step(blk, x, out);
                if (x.kind == InputRef::Kind::Node) {
                    self.register_node_ref(&x);
                }
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("k"), py::arg("x"), py::arg("output"))
        .def("add_subtract",
            [](BlockChain& self, InputRef a, InputRef b, std::string out) {
                auto blk = std::make_shared<Subtract>();
                auto p = make_subtract_step(blk, a, b, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("a"), py::arg("b"), py::arg("output"))
        .def("add_sum",
            [](BlockChain& self, Real w0, Real w1, Real w2,
                InputRef a, InputRef b, InputRef c,
                std::string out) {
                auto blk = std::make_shared<Sum>();
                blk->w0 = w0; blk->w1 = w1; blk->w2 = w2;
                auto p = make_sum_step(blk, a, b, c, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("w0") = 1.0, py::arg("w1") = 1.0,
                py::arg("w2") = 0.0,
                py::arg("a"), py::arg("b"),
                py::arg("c") = InputRef::from_const(0.0),
                py::arg("output"))
        .def("add_math_block",
            [](BlockChain& self, const std::string& op,
                InputRef a, InputRef b, std::string out) {
                auto blk = std::make_shared<MathBlock>();
                if      (op == "add") blk->op = MathBlock::Op::Add;
                else if (op == "sub") blk->op = MathBlock::Op::Sub;
                else if (op == "mul") blk->op = MathBlock::Op::Mul;
                else if (op == "div") blk->op = MathBlock::Op::Div;
                else if (op == "abs") blk->op = MathBlock::Op::Abs;
                else if (op == "neg") blk->op = MathBlock::Op::Neg;
                else if (op == "sqrt") blk->op = MathBlock::Op::Sqrt;
                else if (op == "pow2") blk->op = MathBlock::Op::Pow2;
                else throw std::runtime_error("unknown MathBlock op: " + op);
                auto p = make_math_block_step(blk, a, b, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("op"), py::arg("a"), py::arg("b"),
                py::arg("output"))
        .def("add_pi_controller",
            [](BlockChain& self, Real Kp, Real Ki,
                Real output_min, Real output_max,
                InputRef setpoint, InputRef measured, InputRef dt_ref,
                std::string out) {
                auto blk = std::make_shared<PIController>();
                blk->Kp = Kp; blk->Ki = Ki;
                blk->output_min = output_min; blk->output_max = output_max;
                auto p = make_pi_controller_step(blk, setpoint, measured,
                                                       dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("Kp"), py::arg("Ki"),
                py::arg("output_min"), py::arg("output_max"),
                py::arg("setpoint"), py::arg("measured"),
                py::arg("dt"), py::arg("output"))
        .def("add_pid_controller",
            [](BlockChain& self, Real Kp, Real Ki, Real Kd, Real tau_d,
                Real output_min, Real output_max,
                InputRef setpoint, InputRef measured, InputRef dt_ref,
                std::string out) {
                auto blk = std::make_shared<PIDController>();
                blk->Kp = Kp; blk->Ki = Ki; blk->Kd = Kd;
                blk->tau_d = tau_d;
                blk->output_min = output_min; blk->output_max = output_max;
                auto p = make_pid_controller_step(blk, setpoint, measured,
                                                        dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("Kp"), py::arg("Ki"), py::arg("Kd"),
                py::arg("tau_d"), py::arg("output_min"),
                py::arg("output_max"), py::arg("setpoint"),
                py::arg("measured"), py::arg("dt"), py::arg("output"))
        .def("add_first_order_lpf",
            [](BlockChain& self, Real tau,
                InputRef input_value, InputRef dt_ref,
                std::string out) {
                auto blk = std::make_shared<FirstOrderLowPass>();
                blk->tau = tau;
                auto p = make_first_order_lpf_step(blk, input_value,
                                                         dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("tau"), py::arg("input_value"), py::arg("dt"),
                py::arg("output"))
        .def("add_integrator",
            [](BlockChain& self, Real gain, Real output_min,
                Real output_max, InputRef x, InputRef dt_ref,
                std::string out) {
                auto blk = std::make_shared<Integrator>();
                blk->gain = gain;
                blk->output_min = output_min; blk->output_max = output_max;
                auto p = make_integrator_step(blk, x, dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("gain"), py::arg("output_min"),
                py::arg("output_max"), py::arg("x"), py::arg("dt"),
                py::arg("output"))
        .def("add_differentiator",
            [](BlockChain& self, Real filter_alpha,
                InputRef x, InputRef dt_ref, std::string out) {
                auto blk = std::make_shared<Differentiator>();
                blk->filter_alpha = filter_alpha;
                auto p = make_differentiator_step(blk, x, dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("filter_alpha"), py::arg("x"), py::arg("dt"),
                py::arg("output"))
        .def("add_limiter",
            [](BlockChain& self, Real min_v, Real max_v,
                InputRef x, std::string out) {
                auto blk = std::make_shared<Limiter>();
                blk->min_v = min_v; blk->max_v = max_v;
                auto p = make_limiter_step(blk, x, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("min_v"), py::arg("max_v"),
                py::arg("x"), py::arg("output"))
        .def("add_moving_average",
            [](BlockChain& self, Size window, InputRef x,
                std::string out) {
                auto blk = std::make_shared<MovingAverageFilter>();
                blk->window = window;
                auto p = make_moving_average_step(blk, x, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("window"), py::arg("x"), py::arg("output"))
        .def("add_pwm_generator",
            [](BlockChain& self, Real frequency, Real phase,
                InputRef duty, InputRef t_ref, std::string out) {
                auto blk = std::make_shared<PwmGenerator>();
                blk->frequency = frequency; blk->phase = phase;
                auto p = make_pwm_generator_step(blk, duty, t_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("frequency"), py::arg("phase") = 0.0,
                py::arg("duty"), py::arg("t"), py::arg("output"))
        .def("add_svm",
            [](BlockChain& self, Real v_dc,
                InputRef v_alpha, InputRef v_beta,
                std::string da, std::string db, std::string dc) {
                auto blk = std::make_shared<SpaceVectorModulator>();
                blk->v_dc = v_dc;
                auto p = make_svm_step(blk, v_alpha, v_beta, da, db, dc);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("v_dc"), py::arg("v_alpha"), py::arg("v_beta"),
                py::arg("output_a"), py::arg("output_b"),
                py::arg("output_c"))
        .def("add_clarke",
            [](BlockChain& self, InputRef a, InputRef b, InputRef c,
                std::string alpha, std::string beta, std::string zero) {
                auto blk = std::make_shared<ClarkeTransform>();
                auto p = make_clarke_step(blk, a, b, c, alpha, beta, zero);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("a"), py::arg("b"), py::arg("c"),
                py::arg("output_alpha"), py::arg("output_beta"),
                py::arg("output_zero"))
        .def("add_inverse_clarke",
            [](BlockChain& self, InputRef alpha, InputRef beta,
                InputRef zero, std::string a, std::string b, std::string c) {
                auto blk = std::make_shared<InverseClarkeTransform>();
                auto p = make_inverse_clarke_step(blk, alpha, beta, zero,
                                                        a, b, c);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("alpha"), py::arg("beta"),
                py::arg("zero") = InputRef::from_const(0.0),
                py::arg("output_a"), py::arg("output_b"),
                py::arg("output_c"))
        .def("add_park",
            [](BlockChain& self, InputRef alpha, InputRef beta,
                InputRef theta, std::string d, std::string q) {
                auto blk = std::make_shared<ParkTransform>();
                auto p = make_park_step(blk, alpha, beta, theta, d, q);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("alpha"), py::arg("beta"), py::arg("theta"),
                py::arg("output_d"), py::arg("output_q"))
        .def("add_inverse_park",
            [](BlockChain& self, InputRef d, InputRef q, InputRef theta,
                std::string alpha, std::string beta) {
                auto blk = std::make_shared<InverseParkTransform>();
                auto p = make_inverse_park_step(blk, d, q, theta,
                                                      alpha, beta);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("d"), py::arg("q"), py::arg("theta"),
                py::arg("output_alpha"), py::arg("output_beta"))
        .def("add_pll",
            [](BlockChain& self, Real f_nominal, Real Kp, Real Ki,
                InputRef v_alpha, InputRef v_beta, InputRef dt_ref,
                std::string theta, std::string omega, std::string freq) {
                auto blk = std::make_shared<PLL>();
                blk->f_nominal = f_nominal; blk->Kp = Kp; blk->Ki = Ki;
                auto p = make_pll_step(blk, v_alpha, v_beta, dt_ref,
                                            theta, omega, freq);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("f_nominal"), py::arg("Kp"), py::arg("Ki"),
                py::arg("v_alpha"), py::arg("v_beta"), py::arg("dt"),
                py::arg("output_theta"), py::arg("output_omega"),
                py::arg("output_freq"))

        // ---- Motor blocks (Phase D / C++ port) ----
        .def("add_dc_motor",
            [](BlockChain& self,
                Real J, Real B, Real T_load,
                Real R_a, Real L_a, Real Ke, Real Kt,
                Index armature_branch_var_idx,
                Index bemf_source_idx,
                std::string omega_channel,
                std::string theta_channel) {
                pulsim::motors::Mechanical mech;
                mech.J_kgm2 = J;
                mech.B_Nms_per_rad = B;
                mech.T_load_Nm = T_load;
                pulsim::motors::add_dc_motor_to_chain(
                    self, mech, R_a, L_a, Ke, Kt,
                    armature_branch_var_idx, bemf_source_idx,
                    std::move(omega_channel),
                    std::move(theta_channel));
            },
            py::arg("J"), py::arg("B"), py::arg("T_load"),
            py::arg("R_a"), py::arg("L_a"),
            py::arg("Ke"), py::arg("Kt"),
            py::arg("armature_branch_var_idx"),
            py::arg("bemf_source_idx"),
            py::arg("omega_channel"), py::arg("theta_channel"))

        .def("add_pmsm",
            [](BlockChain& self,
                Real J, Real B, Real T_load,
                Real R_s, Real L_s, Real psi_pm, int pole_pairs,
                std::array<Index, 3> phase_inductor_idx,
                std::array<Index, 3> bemf_source_idx,
                std::string omega_channel,
                std::string theta_channel) {
                pulsim::motors::Mechanical mech;
                mech.J_kgm2 = J;
                mech.B_Nms_per_rad = B;
                mech.T_load_Nm = T_load;
                pulsim::motors::add_three_phase_motor_to_chain(
                    self, mech, R_s, L_s, psi_pm, pole_pairs,
                    pulsim::motors::ThreePhaseBemfShape::Sinusoidal,
                    phase_inductor_idx, bemf_source_idx,
                    std::move(omega_channel),
                    std::move(theta_channel));
            },
            py::arg("J"), py::arg("B"), py::arg("T_load"),
            py::arg("R_s"), py::arg("L_s"),
            py::arg("psi_pm"), py::arg("pole_pairs"),
            py::arg("phase_inductor_idx"),
            py::arg("bemf_source_idx"),
            py::arg("omega_channel"), py::arg("theta_channel"))

        .def("add_bldc",
            [](BlockChain& self,
                Real J, Real B, Real T_load,
                Real R_s, Real L_s, Real psi_pm, int pole_pairs,
                std::array<Index, 3> phase_inductor_idx,
                std::array<Index, 3> bemf_source_idx,
                std::string omega_channel,
                std::string theta_channel) {
                pulsim::motors::Mechanical mech;
                mech.J_kgm2 = J;
                mech.B_Nms_per_rad = B;
                mech.T_load_Nm = T_load;
                pulsim::motors::add_three_phase_motor_to_chain(
                    self, mech, R_s, L_s, psi_pm, pole_pairs,
                    pulsim::motors::ThreePhaseBemfShape::Trapezoidal,
                    phase_inductor_idx, bemf_source_idx,
                    std::move(omega_channel),
                    std::move(theta_channel));
            },
            py::arg("J"), py::arg("B"), py::arg("T_load"),
            py::arg("R_s"), py::arg("L_s"),
            py::arg("psi_pm"), py::arg("pole_pairs"),
            py::arg("phase_inductor_idx"),
            py::arg("bemf_source_idx"),
            py::arg("omega_channel"), py::arg("theta_channel"))

        // ---- Thermal blocks (Phase C.1 / C++ port) ----
        .def("add_thermal_power_injection",
            [](BlockChain& self, InputRef P_ref,
                Index junction_node_idx, std::string power_channel) {
                pulsim::thermal::add_thermal_power_injection_to_chain(
                    self, P_ref, junction_node_idx,
                    std::move(power_channel));
            },
            py::arg("P_ref"), py::arg("junction_node_idx"),
            py::arg("power_channel") = std::string{},
            "Inject the dissipated power into the junction node's "
            "b_extra row. P_ref can be a Const, Channel, or Node "
            "InputRef. Optionally writes the value to "
            "`power_channel` for downstream logging / averaging.")

        .def("add_resistive_power_injection",
            [](BlockChain& self, Real R_ohm, InputRef i_ref,
                Index junction_node_idx, std::string power_channel) {
                pulsim::thermal::add_resistive_power_injection_to_chain(
                    self, R_ohm, i_ref, junction_node_idx,
                    std::move(power_channel));
            },
            py::arg("R_ohm"), py::arg("i_ref"),
            py::arg("junction_node_idx"),
            py::arg("power_channel") = std::string{},
            "Convenience: P = R·I² with single current input. "
            "Typical use is a MOSFET conduction loss "
            "(R = R_DS_ON, i = inductor branch current).")

        // ---- Switchgear blocks (Phase C.4 / C++ port) ----
        .def("add_thyristor",
            [](BlockChain& self, InputRef gate_ref, InputRef current_ref,
                std::string output_channel, Real i_holding) {
                pulsim::switchgear::add_thyristor_to_chain(
                    self, gate_ref, current_ref,
                    std::move(output_channel), i_holding);
            },
            py::arg("gate"), py::arg("current"),
            py::arg("output_channel"), py::arg("i_holding") = 0.0,
            "Gate-triggered latching switch. Sets output_channel to "
            "1.0 when latched ON, 0.0 otherwise. Latches ON when "
            "gate > 0.5; latches OFF when |current| < i_holding.")

        .def("add_fuse",
            [](BlockChain& self, InputRef current_ref,
                Real i2t_threshold, std::string output_channel,
                bool initial_intact) {
                pulsim::switchgear::add_fuse_to_chain(
                    self, current_ref, i2t_threshold,
                    std::move(output_channel), initial_intact);
            },
            py::arg("current"), py::arg("i2t_threshold"),
            py::arg("output_channel"), py::arg("initial_intact") = true,
            "Thermal I²t fuse. Integrates i²·dt; opens irreversibly "
            "when the integral exceeds i2t_threshold (A²·s). "
            "output_channel: 1.0 intact, 0.0 blown.");

    // Helper for make_chain_switch_fn — wires chain channels →
    // SwitchStateMask bits.
    m.def("make_chain_switch_fn",
        [](BlockChain& chain, Size num_switches,
            const std::vector<std::pair<std::string, Index>>& mapping) {
            return make_chain_switch_fn(chain, num_switches, mapping);
        },
        py::arg("chain"), py::arg("num_switches"), py::arg("mapping"),
        "Build a switch_fn(t) that sets bit `idx` ON when "
        "`chain.channels[channel_name] > 0.5`. `mapping` is a list "
        "of (channel_name, switch_idx) pairs.");

    // ----- BDF1 (implicit Euler) — Phase B.2 ------------------------------
    m.def("run_transient_bdf1",
        [](const builder::CircuitBuilder& builder,
            const SimulationOptions& opts,
            SwitchScheduleFn switch_fn,
            BExtraFn b_extra_fn,
            bool start_from_dc_op,
            StepObserverFn step_observer) {
            SimulationResult result;
            {
                py::gil_scoped_release rel;
                result = run_transient_bdf1(
                    builder, opts, switch_fn, b_extra_fn,
                    start_from_dc_op, step_observer);
            }
            return result;
        },
        py::arg("builder"), py::arg("opts"), py::arg("switch_fn"),
        py::arg("b_extra_fn") = BExtraFn{},
        py::arg("start_from_dc_op") = false,
        py::arg("step_observer") = StepObserverFn{},
        "BDF1 (implicit Euler) transient simulation. Use when the "
        "trapezoidal path rings on stiff problems — BDF1 is L-stable "
        "(adds artificial damping that kills numerical oscillation). "
        "Slower than trap because it assembles + factors per step; "
        "intended for ROBUSTNESS over speed.");

    // ----- Fast-path: run_transient that takes a BlockChain DIRECTLY -----
    //
    // Skips the pybind11 std::function wrap on the step_observer.
    // For small chains the saving is modest; for large chains (FOC
    // with 13+ blocks) it's substantial.
    m.def("run_transient_with_chain",
        [](const pwl::PwlStateSpaceCache& cache,
           const topology::Graph& graph,
           const pwl::DevicePool& pool,
           const SimulationOptions& opts,
           SwitchScheduleFn switch_fn,
           BlockChain& chain,
           Real chain_dt,
           BExtraFn b_extra_fn,
           bool start_from_dc_op,
           bool enable_nonlinear_refresh,
           py::object initial_state,
           ShouldContinueFn should_continue) {
            pwl::NonlinearRefreshFn nl_refresh{};
            if (enable_nonlinear_refresh) {
                nl_refresh =
                    pwl::make_combined_diode_mosfet_refresh();
            }
            // Build a step_observer that calls the chain DIRECTLY in
            // C++ — no Python roundtrip per step.
            auto step_observer = chain.make_step_observer(chain_dt);
            Vector x_init;
            const Vector* x_init_ptr = nullptr;
            if (!initial_state.is_none()) {
                x_init = initial_state.cast<Vector>();
                x_init_ptr = &x_init;
            }
            SimulationResult result;
            {
                py::gil_scoped_release rel;
                result = run_transient(cache, graph, pool, opts,
                                          switch_fn, b_extra_fn,
                                          start_from_dc_op,
                                          nl_refresh,
                                          step_observer,
                                          x_init_ptr,
                                          should_continue);
            }
            return result;
        },
        py::arg("cache"), py::arg("graph"), py::arg("pool"),
        py::arg("opts"), py::arg("switch_fn"),
        py::arg("chain"), py::arg("chain_dt"),
        py::arg("b_extra_fn") = BExtraFn{},
        py::arg("start_from_dc_op") = false,
        py::arg("enable_nonlinear_refresh") = false,
        py::arg("initial_state") = py::none(),
        py::arg("should_continue") = ShouldContinueFn{},
        "Run transient with a BlockChain as the per-step observer. "
        "The chain's step is invoked directly in C++ each step — "
        "no Python interpreter cost per step. Equivalent to "
        "`run_transient(..., step_observer=chain.make_step_observer(dt))` "
        "but ~10x faster on chains with > 5 blocks.");
}

}  // namespace pulsim_kernel_bindings

// =============================================================================
// Python module entry point
// =============================================================================
// The extension is named ``_pulsim`` (matches
// ``pybind11_add_module(_pulsim …)`` in ``python/CMakeLists.txt``); all
// kernel symbols are bound directly on the module (no submodule), which
// is what ``python/pulsim/__init__.py`` imports via ``from ._pulsim import …``.
PYBIND11_MODULE(_pulsim, m) {
    m.doc() = "Pulsim — power-electronics simulator. Header-only C++23 "
              "kernel: PWL state-space cache, trapezoidal companion, "
              "Newton refinement, and the high-level CircuitBuilder API.";
    pulsim_kernel_bindings::init_module(m);
}
