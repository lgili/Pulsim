// =============================================================================
// PulsimCore v2 kernel — Python bindings via pybind11
// =============================================================================
//
// Exposes Layer 6's `CircuitBuilder`, Layer 4's
// `PwlStateSpaceCache`, Layer 5's `run_transient`, plus the
// option/result aggregates and `IdealDiode::Params`.
//
// Bound as a Python submodule `pulsim._pulsim.v2_kernel`,
// re-exported nicely via `pulsim.v2` (see
// `python/pulsim/v2.py`).
//
// The user code looks like:
//
//   import pulsim.v2 as p
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

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/models/ideal_diode.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/v2/solver/options.hpp"
#include "pulsim/v2/solver/result.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/sources/combined_switch_fn.hpp"
#include "pulsim/v2/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/v2/sources/phase_shift_full_bridge_fn.hpp"
#include "pulsim/v2/sources/pwm_switch_fn.hpp"
#include "pulsim/v2/sources/spwm_pair_fn.hpp"
#include "pulsim/v2/sources/three_phase_spwm_fn.hpp"
#include "pulsim/v2/topology/graph.hpp"
#include "pulsim/v2/topology/switch_state.hpp"
#include "pulsim/v2/yaml/loader.hpp"

namespace py = pybind11;

namespace pulsim_v2_kernel_bindings {

void init_module(py::module_& m) {
    m.doc() = "Pulsim v2 kernel — PWL state-space cache, "
              "trapezoidal companion, Newton, and the "
              "high-level CircuitBuilder API.";
    m.attr("__version__") = "0.1.0";

    using namespace pulsim::v2;

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
              py::return_value_policy::reference,
              "Add a 3-terminal SH1 MOSFET (Shichman-Hodges "
              "Level 1). Cutoff/triode/saturation regions "
              "blended via sigmoid for C¹-smooth Newton "
              "convergence. K [A/V²], V_T [V], lambda_ [1/V] "
              "channel-length modulation, kappa [1/V] sigmoid "
              "sharpness. Drain → source is a Nonlinear "
              "branch; gate is a node reference (no gate "
              "current — ideal gate). Call `run_transient` "
              "with `nl_refresh=make_combined_diode_mosfet_"
              "refresh()` (or pass `refresh_mosfets_level1` "
              "directly) so the Newton loop stamps the "
              "MOSFET each iteration.")
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
        "Pulsim v2 topology graph. Build via "
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
        "Pulsim v2 device parameter pool. Build via "
        "CircuitBuilder; access via builder.pool.");

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
    using namespace pulsim::v2::solver;

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
           bool enable_nonlinear_refresh) {
            pwl::NonlinearRefreshFn nl_refresh{};
            if (enable_nonlinear_refresh) {
                nl_refresh =
                    pwl::make_combined_diode_mosfet_refresh();
            }
            return run_transient(cache, graph, pool, opts,
                                  switch_fn, b_extra_fn,
                                  start_from_dc_op,
                                  nl_refresh);
        },
        py::arg("cache"), py::arg("graph"), py::arg("pool"),
        py::arg("opts"), py::arg("switch_fn"),
        py::arg("b_extra_fn") = BExtraFn{},
        py::arg("start_from_dc_op") = false,
        py::arg("enable_nonlinear_refresh") = false,
        "Run a fixed-dt transient simulation. switch_fn(t) "
        "→ SwitchStateMask; b_extra_fn(t) → Vector adds "
        "to b_constant at each step. Set "
        "`enable_nonlinear_refresh=True` for circuits "
        "with smooth-blend IdealDiode / SH1 MOSFET "
        "branches (constructs the refresh inside the "
        "binding to avoid Python-roundtrip aliasing of "
        "sparse-matrix references).");
}

}  // namespace pulsim_v2_kernel_bindings
