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
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/analysis/mna_sweep.hpp"
#include "pulsim/blockchain/blocks.hpp"
#include "pulsim/blockchain/block_adapters.hpp"
#include "pulsim/blockchain/chain.hpp"
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/mmc/arm.hpp"
#include "pulsim/motors/mechanical.hpp"
#include "pulsim/integrators/dormand_prince5.hpp"
#include "pulsim/integrators/radau_iia3.hpp"
#include "pulsim/magnetics/hysteretic_inductor.hpp"
#include "pulsim/motors/motor_adapters.hpp"
#include "pulsim/observers/sensorless.hpp"
#include "pulsim/solver/bdf1.hpp"
#include "pulsim/switchgear/switchgear_adapters.hpp"
#include "pulsim/thermal/thermal_adapters.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/dc_operating_point.hpp"
#include "pulsim/pwl/dc_strategy.hpp"
#include "pulsim/pwl/gmin.hpp"
#include "pulsim/solver/voltage_sanity.hpp"
#include "pulsim/mmc/thevenin_arm.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"
#include "pulsim/sources/combined_switch_fn.hpp"
#include "pulsim/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/sources/phase_shift_full_bridge_fn.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/sources/spwm_pair_fn.hpp"
#include "pulsim/sources/three_phase_spwm_fn.hpp"
#include "pulsim/streaming/live_ring.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "pulsim/yaml/loader.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // for "name"_a kwarg shorthand

namespace pulsim_kernel_bindings {

void init_module(py::module_& m) {
    m.doc() = "Pulsim kernel — PWL state-space cache, "
              "trapezoidal companion, Newton, and the "
              "high-level CircuitBuilder API.";
    m.attr("__version__") = "1.6.4";

    using namespace pulsim;

    // add-python-builder-ergonomics (v1.5): register the C++
    // `Cancelled` exception with pybind11 so calls into the kernel
    // that throw `analysis::Cancelled` surface as a typed
    // `pulsim._CxxCancelled` Python exception. The user-facing
    // `pulsim.Cancelled` (defined in
    // `python/pulsim/_builder_ergonomics.py`) catches this and
    // re-raises with its own attribute layout.
    static py::object cancelled_exc = py::register_exception<
        analysis::Cancelled>(m, "_CxxCancelled", PyExc_RuntimeError);

    // v2.0 Phase 2: a transient that dies part-way brings the run
    // with it. `py::register_exception` can only carry a message, so
    // the payload needs a translator that builds the Python
    // exception object and attaches the partial result to it.
    //
    // The type is created here rather than in Python because the
    // translator has to name it at throw time, and the result object
    // it carries is a pybind-bound C++ type.
    static py::object aborted_exc = py::exception<void>(
        m, "SimulationAborted", PyExc_RuntimeError);
    py::register_exception_translator(
        [](std::exception_ptr pe) {
            if (!pe) {
                return;
            }
            try {
                std::rethrow_exception(pe);
            } catch (const solver::SimulationAborted& e) {
                py::object inst = aborted_exc(e.what());
                // `partial` is a COPY: the C++ exception is destroyed
                // as the translator returns, so a reference would
                // dangle the moment the caller touched it.
                inst.attr("partial") = py::cast(e.partial());
                inst.attr("t_failed") = py::cast(e.t_failed());
                PyErr_SetObject(aborted_exc.ptr(), inst.ptr());
            }
        });

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
        // v2.0 — value equality. Without it Python falls back to
        // IDENTITY, so two masks with the same bits compared
        // False: a silent wrong answer for anyone diffing
        // schedules (which is how it was found — a guard that
        // compared two gate schedules never fired).
        .def("__eq__",
              [](const topology::SwitchStateMask& a,
                  const topology::SwitchStateMask& b) {
                  return a == b;
              }, py::is_operator())
        .def("__ne__",
              [](const topology::SwitchStateMask& a,
                  const topology::SwitchStateMask& b) {
                  return !(a == b);
              }, py::is_operator())
        .def("__hash__",
              [](const topology::SwitchStateMask& m) {
                  std::size_t h = std::hash<std::size_t>{}(
                      static_cast<std::size_t>(m.size()));
                  for (Size i = 0; i < m.size(); ++i) {
                      h = h * 131 + (m.get(i) ? 1u : 0u);
                  }
                  return h;
              })
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

    // ---- CircuitBuilder::DeviceInfo --------------------------------------
    py::class_<builder::CircuitBuilder::DeviceInfo>(m, "DeviceInfo",
        "Lightweight POD describing one registered device — name +\n"
        "topological kind + terminal node names. Returned by\n"
        "`CircuitBuilder.devices()`. Useful for GUI introspection\n"
        "and the 'enumerate every component' pattern that PulsimGUI\n"
        "hits often.")
        .def_readonly("name",
              &builder::CircuitBuilder::DeviceInfo::name)
        .def_readonly("kind",
              &builder::CircuitBuilder::DeviceInfo::kind,
              "Topological kind: 'passive', 'source', 'switch',\n"
              "or 'nonlinear'. Finer device-family detail (e.g.\n"
              "MOSFET vs IGBT) is not exposed at this level — query\n"
              "the DevicePool for that.")
        .def_readonly("terminals",
              &builder::CircuitBuilder::DeviceInfo::terminals,
              "Ordered list of node names the device is wired to,\n"
              "in the same order the original `add_*` call passed\n"
              "them (e.g. `['vin', 'sw']` for a MOSFET).")
        .def("__repr__",
              [](const builder::CircuitBuilder::DeviceInfo& d) {
                  std::string ts;
                  for (std::size_t i = 0; i < d.terminals.size(); ++i) {
                      if (i) ts += ", ";
                      ts += "\"" + d.terminals[i] + "\"";
                  }
                  return std::format(
                      "DeviceInfo(name=\"{}\", kind=\"{}\", "
                      "terminals=[{}])",
                      d.name, d.kind, ts);
              });

    // ---- CircuitBuilder ---------------------------------------------------
    // ---- v2.0 Phase 2 (B.1) — preflight + auto-regularization -----------
    //
    // Bound (not C++-only) deliberately: an unbound diagnostic is an
    // untestable one, and users need to see exactly what Pulsim
    // changed about their circuit.
    py::enum_<pwl::PreflightIssue>(m, "PreflightIssue")
        .value("IsolatedSubnet", pwl::PreflightIssue::IsolatedSubnet,
                "No connection to ground through any device.")
        .value("NoDcPathToGround", pwl::PreflightIssue::NoDcPathToGround,
                "Connected, but only through capacitors / current "
                "sources, which are open at DC.");

    py::class_<pwl::PreflightFinding>(m, "PreflightFinding",
        "One topology issue the preflight pass found, and what it "
        "did about it.")
        .def_readonly("issue", &pwl::PreflightFinding::issue)
        .def_readonly("anchor_node", &pwl::PreflightFinding::anchor_node,
                       "Node id that received the reference tie.")
        .def_readonly("component", &pwl::PreflightFinding::component,
                       "Every node id in the affected subnet.")
        .def_readonly("inserted_resistance",
                       &pwl::PreflightFinding::inserted_resistance,
                       "Ohms of the inserted tie; 0 when the pass only "
                       "reported the issue.")
        .def_readonly("detail", &pwl::PreflightFinding::detail,
                       "Human-facing explanation, naming the node.")
        .def("was_fixed", &pwl::PreflightFinding::was_fixed)
        .def("__repr__", [](const pwl::PreflightFinding& f) {
            return std::string("PreflightFinding(node=") +
                   std::to_string(f.anchor_node) +
                   ", fixed=" + (f.was_fixed() ? "True" : "False") + ")";
        });

    py::class_<pwl::PreflightReport>(m, "PreflightReport",
        "What the topology preflight found, and which reference ties "
        "it inserted. Empty when the circuit was already well-posed.")
        .def_readonly("findings", &pwl::PreflightReport::findings)
        .def("empty", &pwl::PreflightReport::empty)
        .def("num_fixed", &pwl::PreflightReport::num_fixed)
        .def("summary", &pwl::PreflightReport::summary,
              "One paragraph naming every issue and fix; empty string "
              "when there is nothing to report.")
        .def("__len__", [](const pwl::PreflightReport& r) {
            return r.findings.size();
        })
        .def("__repr__", [](const pwl::PreflightReport& r) {
            return std::string("PreflightReport(findings=") +
                   std::to_string(r.findings.size()) + ", fixed=" +
                   std::to_string(r.num_fixed()) + ")";
        });

    py::class_<pwl::PreflightOptions>(m, "PreflightOptions",
        "Knobs for the preflight pass.")
        .def(py::init<>())
        .def(py::init([](bool auto_regularize, Real tie_resistance,
                          std::string name_prefix) {
                  pwl::PreflightOptions o;
                  o.auto_regularize = auto_regularize;
                  o.tie_resistance  = tie_resistance;
                  o.name_prefix     = std::move(name_prefix);
                  return o;
              }),
              py::arg("auto_regularize") = true,
              py::arg("tie_resistance")  = Real{1e9},
              py::arg("name_prefix")     = std::string{"R_auto_iso_"})
        .def_readwrite("auto_regularize",
                        &pwl::PreflightOptions::auto_regularize,
                        "Insert the ties (default) rather than only "
                        "reporting them.")
        .def_readwrite("tie_resistance",
                        &pwl::PreflightOptions::tie_resistance,
                        "Ohms of an inserted reference tie. The "
                        "default 1e9 draws nanoamps, so it supplies "
                        "the MNA reference without loading the node — "
                        "a SMALL value here would be a galvanic bond "
                        "that silently rewires the circuit.")
        .def_readwrite("name_prefix",
                        &pwl::PreflightOptions::name_prefix);

    py::class_<builder::CircuitBuilder>(m, "CircuitBuilder",
        "High-level v2 circuit constructor. Hides the "
        "two-object Graph + DevicePool setup; users pass "
        "string node names and SI-unit parameter values.",
        py::dynamic_attr())  // allow Python-side metadata
                              // (initial conditions, aliases)
        .def(py::init<>())
        .def("node", &builder::CircuitBuilder::node,
              py::arg("name"),
              "Look up or create a node by name. "
              "\"gnd\" / \"GND\" / \"0\" alias to ground.")
        .def("add_voltage_source",
              &builder::CircuitBuilder::add_voltage_source,
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("V"),
              py::return_value_policy::reference,
              "Add a DC voltage source. V in volts.")
        .def("add_current_source",
              &builder::CircuitBuilder::add_current_source,
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("I"),
              py::return_value_policy::reference,
              "Add a DC current source. I in amperes. "
              "EE convention: I flows OUT of `n_pos` (the "
              "+ terminal) into the external circuit, "
              "back to `n_neg`. Does NOT add a "
              "branch-current unknown — the current is "
              "fixed at I.")
        .def("add_pwm_voltage_source",
              &builder::CircuitBuilder::add_pwm_voltage_source,
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"),
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
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"),
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
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"),
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
        .def("run_preflight",
              [](builder::CircuitBuilder& self,
                  const pwl::PreflightOptions& opts) {
                  return self.run_preflight(opts);
              },
              py::arg("options") = pwl::PreflightOptions{},
              "Check the topology for nodes with no voltage "
              "reference and (by default) tie each offending subnet "
              "to ground through a high-value resistor. Returns a "
              "PreflightReport naming everything it found and "
              "inserted. `simulate()` runs this automatically; call "
              "it directly to inspect a circuit, or with "
              "PreflightOptions(auto_regularize=False) to look "
              "without touching.")
        .def("add_resistor",
              &builder::CircuitBuilder::add_resistor,
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("R_ohms"),
              py::return_value_policy::reference,
              "Add a linear resistor. R in ohms.")
        .def("add_capacitor",
              &builder::CircuitBuilder::add_capacitor,
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("C_farads"),
              py::arg("c0") = std::nullopt,
              py::return_value_policy::reference,
              "Add a linear capacitor. C in farads.\n"
              "Optional `c0` records an initial voltage (V) on the\n"
              "capacitor — consumed by `simulate(initial_state=None)`\n"
              "to seed the cap's positive-terminal node.")
        .def("add_inductor",
              &builder::CircuitBuilder::add_inductor,
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("L_henries"),
              py::arg("i0") = std::nullopt,
              py::return_value_policy::reference,
              "Add a linear inductor. L in henries.\n"
              "Optional `i0` records an initial current (A) flowing\n"
              "from the `n_pos` terminal toward `n_neg` — consumed by\n"
              "`simulate(initial_state=None)`.")
        .def("set_initial",
              &builder::CircuitBuilder::set_initial,
              py::arg("name"), py::arg("value"),
              "Record or override an initial condition for a named\n"
              "capacitor (volts) or inductor (amps). Raises\n"
              "out_of_range if the name doesn't match a registered\n"
              "branch.")
        .def("set_alias",
              [](builder::CircuitBuilder& self,
                  std::string_view human_name,
                  std::optional<std::string_view> node,
                  std::optional<std::string_view> branch) {
                  self.set_alias(human_name, node, branch);
              },
              py::arg("human_name"),
              py::arg("node") = std::nullopt,
              py::arg("branch") = std::nullopt,
              "Register an alias `human_name` for an existing node or\n"
              "branch — exactly one of `node` / `branch` must be set.\n"
              "Raises ValueError on empty input, alias collision\n"
              "with a canonical name, or both/neither kwargs.")
        .def("aliases",
              [](const builder::CircuitBuilder& self) {
                  // Return as Python dict {human: (kind_str, target)}.
                  py::dict out;
                  for (const auto& [name, target] : self.aliases()) {
                      const char* kind_str =
                          target.kind == builder::CircuitBuilder::AliasKind::Node
                              ? "node" : "branch";
                      out[py::cast(name)] = py::make_tuple(
                          kind_str, target.target);
                  }
                  return out;
              },
              "Return the registered `{human_name: (kind, target)}` map\n"
              "for round-tripping through GUI file formats. `kind` is\n"
              "the string \"node\" or \"branch\".")
        .def("initial_state",
              &builder::CircuitBuilder::initial_state,
              "Synthesise the flat initial-state vector from recorded\n"
              "ICs. Returns a numpy array of size num_nodes +\n"
              "num_state_branches with each IC slot populated and zero\n"
              "elsewhere. Consumed by `pulsim.simulate` when\n"
              "`initial_state=None`.")
        .def("state_var_names",
              &builder::CircuitBuilder::state_var_names,
              "Human-readable name for every entry of the state vector\n"
              "the kernel solves. Returns ``list[str]`` of size\n"
              "``pool.state_size(graph)`` — same layout the kernel uses.\n"
              "Entries are ``V(<node>)`` for node voltages,\n"
              "``Is(<branch>)`` for voltage-source currents, and\n"
              "``I(<branch>)`` for inductor currents.\n"
              "\n"
              "Used by ``pulsim.NativeLiveStream`` (the live scope) to\n"
              "label channels without the caller having to recompute\n"
              "the layout themselves. ``simulate(live_stream=…)``\n"
              "auto-fills the names into the stream when attaching.")
        .def("add_nonlinear_capacitor",
              &builder::CircuitBuilder::add_nonlinear_capacitor,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("C0"), py::arg("V0"),
              py::arg("m") = 0.5, py::arg("v_floor") = -0.9,
              py::return_value_policy::reference,
              "Charge-based nonlinear capacitor — a MOSFET's Coss: "
              "C(v) = C0 / (1 + v/V0)^m.\n\n"
              "What decides ZVS is the CHARGE Q(V) = int C dv, not "
              "the small-signal C a datasheet quotes at the "
              "operating point: for C0 = 2 nF, V0 = 25 V, m = 0.5 "
              "at 400 V those differ by 1.61x, which is the "
              "difference between a dead time that reads as clean "
              "ZVS and one that leaves the node at 209 V when the "
              "switch turns on (17 uJ per edge). m = 0 gives an "
              "ordinary linear capacitor, exactly — useful for the "
              "A/B.")
        .def("add_saturable_inductor",
              &builder::CircuitBuilder::add_saturable_inductor,
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"),
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
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("g_on"),
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
        // The non-const + const `pool()` overloads (v1.4.0) make
        // taking the member-function pointer ambiguous; cast to the
        // const overload explicitly for the read-only property.
        // Python users that need to mutate the pool go through the
        // `update_*` helpers below.
        .def_property_readonly("pool",
              static_cast<const pwl::DevicePool& (builder::CircuitBuilder::*)() const noexcept>(
                  &builder::CircuitBuilder::pool),
              py::return_value_policy::reference_internal,
              "Const ref to the internal DevicePool.")
        .def_property_readonly("num_branches",
              &builder::CircuitBuilder::num_branches,
              "Number of branches added so far.")
        .def("compute_time_varying_b_extra",
              [](const builder::CircuitBuilder& self, Real t) {
                  // Sum the contributions of all time-varying voltage
                  // sources (sine + PWM + pulse) into a single
                  // full-MNA-sized b_extra vector. Bridge.6 — the
                  // DSED adapter calls this each step and projects
                  // through b_projection · b_extra(t) to overlay
                  // time-varying sources without re-stamping.
                  const auto& g = self.graph();
                  const auto& p = self.pool();
                  Vector b = sources::compute_pwm_b_extra(p, g, t);
                  b += sources::compute_sine_b_extra(p, g, t);
                  b += sources::compute_pulse_b_extra(p, g, t);
                  return b;
              },
              py::arg("t"),
              "Sum the b_extra contributions from all time-varying "
              "voltage sources (sine + PWM + pulse) at time `t`. "
              "Returns a full-MNA-sized vector. The pulsim.simulate "
              "engine='pwl' applies this automatically; engine='dsed' "
              "calls it via the CircuitBuilderAdapter to overlay "
              "AC / PWM source values per PED step.")
        .def("node_id_of",
              &builder::CircuitBuilder::node_id_of,
              py::arg("name"),
              "Return the node index for `name`. Throws if "
              "the name was never registered.")
        .def("name_of",
              [](const builder::CircuitBuilder& self, Index branch_id) {
                  return std::string(self.name_of(branch_id));
              },
              py::arg("branch_id"),
              "User-supplied component name for `branch_id`, set by "
              "the `add_*` call. Returns an empty string if the "
              "branch was never registered or had no name.")
        .def("branch_id_of",
              &builder::CircuitBuilder::branch_id_of,
              py::arg("name"),
              "Inverse of name_of: lookup the branch_id for a "
              "user-supplied component name. v1.4.0+ — used by "
              "the parametric refactor pipeline to translate "
              "user-facing strings like 'L_out' into the pool's "
              "branch_id for update_* calls. Raises ValueError "
              "if the name was never registered.")
        .def("branch_index_of",
              &builder::CircuitBuilder::branch_id_of,
              py::arg("name"),
              "Alias for branch_id_of, named to match the\n"
              "`SimulationResult.i(name)` accessor's promise that\n"
              "the returned int is the post-node state-vector\n"
              "offset (`state_idx == num_nodes + branch_index_of(name)`).")
        .def("switch_index_of",
              &builder::CircuitBuilder::switch_index_of,
              py::arg("name"),
              "Bit position of `name` in the SwitchStateMask\n"
              "produced by `switch_fn(t)`. Counts only switching\n"
              "branches (MOSFET / IGBT / binary diode / explicit\n"
              "add_switch) in builder-call order. Raises\n"
              "out_of_range if `name` is a passive or source.")
        .def("devices",
              &builder::CircuitBuilder::devices,
              "Ordered list of DeviceInfo records for every named\n"
              "device the builder has accepted. Anonymous branches\n"
              "(snubber RC expansion, transformer parallels) are\n"
              "skipped. List order matches `add_*` call order.")
        .def("update_resistor_R",
              [](builder::CircuitBuilder& self,
                  std::string_view name, Real new_R_ohms) {
                  self.pool().update_resistor_R(
                      self.branch_id_of(name), new_R_ohms);
              },
              py::arg("name"), py::arg("new_R_ohms"),
              "Update a resistor's R value (ohms) by component name. "
              "Topology unchanged. Designed for the parametric "
              "refactor pipeline: pair with PwlStateSpaceCache."
              "refactor_parametric(branch_id_of(name), new_R_ohms) "
              "to push the change through to every active mask.")
        .def("update_inductor_L",
              [](builder::CircuitBuilder& self,
                  std::string_view name, Real new_L_henries) {
                  self.pool().update_inductor_L(
                      self.branch_id_of(name), new_L_henries);
              },
              py::arg("name"), py::arg("new_L_henries"),
              "Update an inductor's L value (henries) by component "
              "name. See update_resistor_R for the parametric "
              "refactor usage.")
        .def("update_capacitor_C",
              [](builder::CircuitBuilder& self,
                  std::string_view name, Real new_C_farads) {
                  self.pool().update_capacitor_C(
                      self.branch_id_of(name), new_C_farads);
              },
              py::arg("name"), py::arg("new_C_farads"),
              "Update a capacitor's C value (farads) by component "
              "name. See update_resistor_R for the parametric "
              "refactor usage.")
        .def("update_voltage_source_V",
              [](builder::CircuitBuilder& self,
                  std::string_view name, Real new_V) {
                  self.pool().update_voltage_source_V(
                      self.branch_id_of(name), new_V);
              },
              py::arg("name"), py::arg("new_V"),
              "Update a DC voltage source's V value by component "
              "name. Pure-RHS update — no LU refactor needed; the "
              "next solve() picks up the change via b_constant.")
        .def("components",
              [](const builder::CircuitBuilder& self) {
                  // Adapter consumed by `pulsim.schematic` — yields one
                  // `ComponentDescriptor` per branch added through the
                  // builder, in insertion order. The descriptor exposes
                  // the four fields the schematic layout engine needs:
                  //   .kind   — canonical string (see proposal §2.3)
                  //   .name   — user-supplied name (or "" if unnamed)
                  //   .nodes  — list of node IDs in pin order
                  //   .params — dict[str, float|int] of stored params
                  py::list out;
                  const auto& graph = self.graph();
                  const auto& pool  = self.pool();
                  const auto n_branches = graph.num_branches();
                  for (Index bid = 0; bid < n_branches; ++bid) {
                      const auto& br = graph.branch(bid);
                      py::dict params;
                      const char* kind_str = "unknown";
                      try {
                          const auto kind = pool.kind_of(bid);
                          using K = pwl::DevicePool::StoredKind;
                          switch (kind) {
                          case K::Resistor: {
                              kind_str = "resistor";
                              const auto& p =
                                  pool.resistor_params(bid);
                              params["R_ohms"] = Real{1} / p.G;
                              break;
                          }
                          case K::VoltageSource: {
                              kind_str = "voltage_source";
                              const auto& p =
                                  pool.voltage_source_params(bid);
                              params["V"] = p.V;
                              break;
                          }
                          case K::Switch: {
                              kind_str = "switch";
                              params["g_on"]  = pool.switch_g_on(bid);
                              params["g_off"] = pool.switch_g_off(bid);
                              break;
                          }
                          case K::Capacitor: {
                              kind_str = "capacitor";
                              const auto& p =
                                  pool.capacitor_params(bid);
                              params["C_farads"] = p.C;
                              break;
                          }
                          case K::Inductor: {
                              kind_str = "inductor";
                              const auto& p =
                                  pool.inductor_params(bid);
                              params["L_henries"] = p.L;
                              break;
                          }
                          case K::Diode: {
                              kind_str = "diode";
                              const auto& p = pool.diode_params(bid);
                              params["g_on"]  = p.g_on;
                              params["g_off"] = p.g_off;
                              params["V_th"]  = p.V_th;
                              break;
                          }
                          case K::NonlinearDiode: {
                              kind_str = "nonlinear_diode";
                              // IdealDiode::Params struct varies; just
                              // expose what schematic needs (none yet).
                              break;
                          }
                          case K::CurrentSource: {
                              kind_str = "current_source";
                              const auto& p =
                                  pool.current_source_params(bid);
                              params["I"] = p.I;
                              break;
                          }
                          case K::PWMVoltageSource:
                              kind_str = "pwm_voltage_source"; break;
                          case K::SineVoltageSource:
                              kind_str = "sine_voltage_source"; break;
                          case K::PulseVoltageSource:
                              kind_str = "pulse_voltage_source"; break;
                          case K::MosfetLevel1:
                              kind_str = "mosfet_level1"; break;
                          case K::IgbtLevel1:
                              kind_str = "igbt_level1"; break;
                          case K::VCVS:
                              kind_str = "vcvs"; break;
                          case K::SaturableInductor:
                              kind_str = "saturable_inductor"; break;
                          }
                      } catch (const std::exception&) {
                          // Branch present in graph but not registered
                          // in pool — unusual, keep "unknown" kind.
                      }
                      py::dict desc;
                      desc["kind"]   = kind_str;
                      desc["name"]   = std::string(self.name_of(bid));
                      desc["nodes"]  = py::make_tuple(
                          static_cast<int>(br.from),
                          static_cast<int>(br.to));
                      desc["params"] = params;
                      desc["branch_id"] = static_cast<int>(bid);
                      out.append(desc);
                  }
                  return out;
              },
              "Enumerate every device in the circuit. Returns a list "
              "of dicts (kind, name, nodes, params, branch_id) in "
              "insertion order. Consumed by `pulsim.schematic`.")
        .def("ground",
              [](const builder::CircuitBuilder& self) {
                  return self.graph().ground();
              },
              "Ground node ID (always -1 in Pulsim's convention). "
              "Exposed on the builder for the schematic API "
              "compatibility.")
        .def("node_position_hint",
              [](const builder::CircuitBuilder& /*self*/,
                 Index node_id) {
                  // V0 stub: return "ground" for ground, "internal"
                  // for everything else. Phase 4 of the schematic V2
                  // proposal will refine this.
                  if (node_id == topology::Graph::ground()) {
                      return std::string("ground");
                  }
                  return std::string("internal");
              },
              py::arg("node_id"),
              "Role hint for `node_id`. V0 returns 'ground' for the "
              "ground node and 'internal' for all others.")
        .def("position_hints",
              [](const builder::CircuitBuilder& self) {
                  py::dict out;
                  const auto n = self.graph().num_nodes();
                  for (Index i = 0; i < n; ++i) {
                      out[py::int_(i)] = std::string("internal");
                  }
                  out[py::int_(topology::Graph::ground())] =
                      std::string("ground");
                  return out;
              },
              "Dict mapping every node_id used in the circuit to its "
              "current role hint.");

    // ---- BranchKind enum + Branch struct (for schematic backends
    //      that want graph-level structural info) -------------------
    py::enum_<topology::BranchKind>(m, "BranchKind",
        "Topological category of a branch. Layer 4 enumerates "
        "over `Switch`-kind branches; the rest are unconditional.")
        .value("PassiveLinear", topology::BranchKind::PassiveLinear)
        .value("Source",        topology::BranchKind::Source)
        .value("Switch",        topology::BranchKind::Switch)
        .value("Nonlinear",     topology::BranchKind::Nonlinear);


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
        // v2.0 Phase 1 gave branches names so kernel diagnostics
        // could speak them; expose them here too, so tooling and
        // tests can see what a pass (e.g. preflight) inserted.
        .def("branch_name",
              [](const topology::Graph& g, Index b) {
                  return std::string{g.branch_name(b)};
              }, py::arg("branch_id"),
              "The branch's user-facing name, or '' when it was "
              "created without one.")
        .def("node_name",
              [](const topology::Graph& g, Index n) {
                  return (n < 0 || n >= g.num_nodes())
                      ? std::string{} : g.node(n).name;
              }, py::arg("node_id"),
              "The node's name, or '' when out of range/unnamed.")
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
        .def_static("ground", &topology::Graph::ground)
        .def_property_readonly("branches",
              [](const topology::Graph& self) {
                  // Lightweight branch enumeration for the schematic
                  // module. Each entry is a dict (id, from_, to, kind)
                  // — no need for a custom py::class_ since callers
                  // just iterate.
                  py::list out;
                  const auto n = self.num_branches();
                  for (Index i = 0; i < n; ++i) {
                      const auto& br = self.branch(i);
                      py::dict d;
                      d["id"]    = static_cast<int>(br.id);
                      d["from_"] = static_cast<int>(br.from);
                      d["to"]    = static_cast<int>(br.to);
                      d["kind"]  = br.kind;
                      out.append(d);
                  }
                  return out;
              },
              "List of all branches in insertion order. Each entry "
              "is a dict {id, from_, to, kind} where `kind` is a "
              "`BranchKind` enum value.")
        .def_property_readonly("nodes",
              [](const topology::Graph& self) {
                  // Node names + IDs — also useful for schematic
                  // labelling.
                  py::list out;
                  const auto n = self.num_nodes();
                  for (Index i = 0; i < n; ++i) {
                      const auto& nd = self.node(i);
                      py::dict d;
                      d["id"]   = static_cast<int>(nd.id);
                      d["name"] = nd.name;
                      out.append(d);
                  }
                  return out;
              },
              "List of all nodes in insertion order. Each entry "
              "is a dict {id, name}.");

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
              "(= num_active_nodes + num_sources + num_inductors).")
        // v1.4.0 — kind_of helps Python clients (sweep_path_aware
        // etc.) dispatch the right update_* mutator from a
        // branch_id alone. The returned StoredKind enum is bound
        // below as a sibling type so `.name` gives the canonical
        // string ("Resistor", "Inductor", …).
        .def("kind_of",
              &pwl::DevicePool::kind_of,
              py::arg("branch_id"),
              "Return the StoredKind enum for `branch_id`. Used by "
              "the v1.4.0 parametric refactor pipeline to pick the "
              "right update_* mutator from a string-based component "
              "name lookup.");

    // The StoredKind enum (sibling of DevicePool) — v1.4.0 exposes
    // it for sweep_path_aware's auto-dispatch over device kinds.
    // The kernel maintains the full taxonomy in DevicePool::Entry's
    // std::variant; we mirror just the cases relevant to parametric
    // refactor here (the others are not user-facing). New variants
    // added to the kernel must be appended below to keep the Python
    // enum exhaustive.
    py::enum_<pwl::DevicePool::StoredKind>(m, "StoredKind",
        "Device kind stored in DevicePool. v1.4.0+ — exposed so "
        "Python helpers (sweep_path_aware etc.) can auto-dispatch "
        "update_* mutators per device type.")
        .value("Resistor",            pwl::DevicePool::StoredKind::Resistor)
        .value("VoltageSource",       pwl::DevicePool::StoredKind::VoltageSource)
        .value("Switch",              pwl::DevicePool::StoredKind::Switch)
        .value("Capacitor",           pwl::DevicePool::StoredKind::Capacitor)
        .value("Inductor",            pwl::DevicePool::StoredKind::Inductor)
        .value("Diode",               pwl::DevicePool::StoredKind::Diode)
        .value("NonlinearDiode",      pwl::DevicePool::StoredKind::NonlinearDiode)
        .value("CurrentSource",       pwl::DevicePool::StoredKind::CurrentSource)
        .value("PWMVoltageSource",    pwl::DevicePool::StoredKind::PWMVoltageSource)
        .value("SineVoltageSource",   pwl::DevicePool::StoredKind::SineVoltageSource)
        .value("PulseVoltageSource",  pwl::DevicePool::StoredKind::PulseVoltageSource)
        .value("MosfetLevel1",        pwl::DevicePool::StoredKind::MosfetLevel1)
        .value("IgbtLevel1",          pwl::DevicePool::StoredKind::IgbtLevel1)
        .value("VCVS",                pwl::DevicePool::StoredKind::VCVS)
        .value("SaturableInductor",   pwl::DevicePool::StoredKind::SaturableInductor);

    // ---- ParametricRefactorResult (v1.4.0) -------------------------------
    py::class_<pwl::ParametricRefactorResult>(m,
        "ParametricRefactorResult",
        "Return value of PwlStateSpaceCache.refactor_parametric. "
        "Invariant: path_refactor_hits + fallback_hits == masks_processed.")
        .def_readonly("masks_processed",
                       &pwl::ParametricRefactorResult::masks_processed)
        .def_readonly("path_refactor_hits",
                       &pwl::ParametricRefactorResult::path_refactor_hits)
        .def_readonly("fallback_hits",
                       &pwl::ParametricRefactorResult::fallback_hits)
        .def_readonly("wall_time_us",
                       &pwl::ParametricRefactorResult::wall_time_us);

    py::enum_<pwl::ParametricRefactorMode>(m, "ParametricRefactorMode")
        .value("AllActive",   pwl::ParametricRefactorMode::AllActive)
        .value("CurrentOnly", pwl::ParametricRefactorMode::CurrentOnly);

    // ---- PwlStateSpaceCache ----------------------------------------------
    // Phase-0 fix #4 helper: exact controlled-vs-diode switch census,
    // computed with the SAME walk the solver uses (DiodeEventState),
    // so Python-side policy (warn on driverless controlled switches)
    // can never drift from kernel semantics.
    m.def("_switch_census",
          [](const topology::Graph& graph, const pwl::DevicePool& pool) {
              pwl::DiodeEventState diodes(graph, pool);
              const auto owned = diodes.diode_owned_bits();
              const auto n_sw = graph.num_switches();
              std::vector<Size> controlled;
              for (Size i = 0; i < n_sw; ++i) {
                  if (!owned.get(i)) controlled.push_back(i);
              }
              return py::make_tuple(n_sw, diodes.num_diodes(),
                                     controlled);
          },
          py::arg("graph"), py::arg("pool"),
          "Return (num_switches, num_diode_owned, controlled_switch_"
          "indices). Diode bits are solver-owned (overlaid over the "
          "user mask by combine_masks); only the controlled indices "
          "need a switch_fn/driver.");

    // v2.0 Phase 1 — cache telemetry. Was C++-only, which made the
    // eviction and event-solver behaviour unobservable (and so
    // untestable) from Python (adversarial-review finding F10).
    py::class_<pwl::CacheMetrics>(m, "CacheMetrics",
        "Snapshot of PwlStateSpaceCache counters. Monotonic across "
        "the cache's lifetime.")
        .def_readonly("rank1_hits", &pwl::CacheMetrics::rank1_hits)
        .def_readonly("multi_bit_rank1_hits",
                       &pwl::CacheMetrics::multi_bit_rank1_hits)
        .def_readonly("full_refactor_hits",
                       &pwl::CacheMetrics::full_refactor_hits)
        .def_readonly("fallbacks", &pwl::CacheMetrics::fallbacks)
        .def_readonly("segment_evictions",
                       &pwl::CacheMetrics::segment_evictions,
                       "Lazy-mode segments dropped by the byte-budget "
                       "LRU.")
        .def_readonly("event_hits", &pwl::CacheMetrics::event_hits,
                       "solve_at calls served by a resident factor at "
                       "the same dt.")
        .def_readonly("event_refactors",
                       &pwl::CacheMetrics::event_refactors,
                       "solve_at calls that re-factored a resident "
                       "entry at a new dt (no re-analysis).")
        .def_readonly("event_builds", &pwl::CacheMetrics::event_builds,
                       "solve_at calls that had to build a fresh "
                       "event solver.")
        .def("__repr__", [](const pwl::CacheMetrics& x) {
            return std::string("CacheMetrics(evictions=") +
                   std::to_string(x.segment_evictions) +
                   ", event_hits=" + std::to_string(x.event_hits) +
                   ", event_refactors=" +
                   std::to_string(x.event_refactors) +
                   ", event_builds=" +
                   std::to_string(x.event_builds) + ")";
        });

    py::class_<pwl::PwlStateSpaceCache>(m, "PwlStateSpaceCache",
        "PWL state-space cache. Pre-factors the MNA matrix "
        "for every reachable switch combination + dt.")
        .def(py::init<const topology::Graph&,
                       pwl::DevicePool&>(),
              py::arg("graph"), py::arg("pool"),
              py::keep_alive<1, 2>(),
              py::keep_alive<1, 3>())
        .def("build",
              [](pwl::PwlStateSpaceCache& self, Real dt) {
                  self.build(dt);
              }, py::arg("dt") = 0.0,
              "Build the PWL cache eagerly. dt=0 means "
              "static-only (no trap companion).")
        .def("build_lazy",
              [](pwl::PwlStateSpaceCache& self, Real dt) {
                  self.build_lazy(dt);
              }, py::arg("dt"),
              "Lazy build: store dt but defer per-mask "
              "factorisation to first access. A PWM converter "
              "visits only a handful of the 2^N switch states, "
              "so this avoids the eager 2^N enumeration that "
              "makes many-switch circuits (multilevel, MMC) "
              "unbuildable.")
        .def("num_built_segments",
              [](const pwl::PwlStateSpaceCache& self) {
                  return self.num_segments();
              },
              "Number of per-mask segments actually factorised "
              "so far (useful with build_lazy to see how many "
              "of the 2^N states the run really visited).")
        .def("dt", &pwl::PwlStateSpaceCache::dt)
        // v2.0 Phase 1 — LRU byte-budget on the lazy segment cache
        // (audit finding no-mode-cache-eviction).
        .def("set_segment_budget_bytes",
              &pwl::PwlStateSpaceCache::set_segment_budget_bytes,
              py::arg("bytes"),
              "Set the lazy-mode segment cache byte budget. Once "
              "the estimated resident bytes would exceed it, the "
              "least-recently-solved mask's factor is evicted "
              "(transparently rebuilt on re-visit). 0 disables "
              "eviction. Default: 1 GiB. Eager build() never "
              "evicts.")
        .def("segment_budget_bytes",
              &pwl::PwlStateSpaceCache::segment_budget_bytes,
              "Current lazy segment cache byte budget (0 = "
              "unbounded).")
        .def("segment_cache_bytes",
              &pwl::PwlStateSpaceCache::segment_cache_bytes,
              "Estimated resident bytes of the per-mask segment "
              "cache (assembled matrices + LU factors). Sample it "
              "BETWEEN runs: it walks the segment map, which lazy "
              "builds and evictions mutate, and run_transient "
              "releases the GIL — `metrics()` is the thread-safe "
              "counter set.")
        .def("metrics", &pwl::PwlStateSpaceCache::metrics,
              "Snapshot the cache counters (evictions, event-solver "
              "hits/refactors/builds, rank-1 telemetry).")
        .def("num_event_entries",
              &pwl::PwlStateSpaceCache::num_event_entries,
              "Number of resident event-dt solver entries used by "
              "sub-step event correction (LRU-bounded at 8; each "
              "holds ONE factor refreshed in place per dt).")
        // v1.4.0 — Part B parametric refactor API.
        .def("refactor_parametric",
              [](pwl::PwlStateSpaceCache& self,
                  Index branch_id, Real new_value,
                  pwl::ParametricRefactorMode mode) {
                  return self.refactor_parametric(branch_id, new_value, mode);
              },
              py::arg("branch_id"), py::arg("new_value"),
              py::arg("mode") = pwl::ParametricRefactorMode::AllActive,
              "Path-based refactor of every active mask's cached "
              "LU factor for a single parameter change. Returns a "
              "ParametricRefactorResult. The branch_id is looked up "
              "via CircuitBuilder.branch_id_of(name).")
        .def("refactor_parametric_batch",
              [](pwl::PwlStateSpaceCache& self,
                  const std::vector<std::pair<Index, Real>>& updates,
                  pwl::ParametricRefactorMode mode) {
                  std::vector<pwl::ParametricUpdate> us;
                  us.reserve(updates.size());
                  for (const auto& [bid, val] : updates) {
                      us.push_back({bid, val});
                  }
                  return self.refactor_parametric(
                      std::span<const pwl::ParametricUpdate>(
                          us.data(), us.size()), mode);
              },
              py::arg("updates"),
              py::arg("mode") = pwl::ParametricRefactorMode::AllActive,
              "Batch parametric refactor — takes a list of "
              "(branch_id, new_value) tuples for simultaneous updates.")
        // -------------------------------------------------------------
        // DSED bridge — continuous-time LTI state-space extraction
        // (currently a stub; see notes/DSED_BRIDGE_DESIGN.md §5.1).
        // -------------------------------------------------------------
        .def("compute_lti_state_space",
              [](pwl::PwlStateSpaceCache& self,
                  const topology::SwitchStateMask& mask,
                  Real h_a, Real h_b) {
                  auto r = self.compute_lti_state_space(mask, h_a, h_b);
                  // Convert to a Python tuple
                  // (A, b_constant, state_row_indices, state_is_cap,
                  //  b_projection)
                  // for downstream consumption by the Python
                  // CircuitBuilderAdapter. The projection matrix
                  // ``b_projection`` is the linear map from full-MNA
                  // b → state-reduced b, used to overlay time-varying
                  // source contributions (sine/PWM/pulse) without
                  // re-stamping per step (Bridge.6).
                  return py::make_tuple(
                      r.A,
                      r.b_constant,
                      r.state_row_indices,
                      r.state_is_cap,
                      r.b_projection);
              },
              py::arg("mask"),
              py::arg("h_a") = Real{1e-6},
              py::arg("h_b") = Real{5e-7},
              "Extract continuous-time LTI state-space (A, b) for "
              "the given switch mask from the assembler's exact "
              "(G, C, b) split (v2.0 — this replaced the earlier "
              "finite-difference recovery; `h_a`/`h_b` are accepted "
              "for backwards compatibility and ignored). "
              "Raises RuntimeError for MAGNETICALLY COUPLED "
              "inductors (a transformer with k != 0): the "
              "extraction applies a diagonal inverse mass matrix "
              "and cannot represent the mutual terms — use "
              "engine='pwl' there. A k=0 transformer models two "
              "independent windings and extracts normally. "
              "Returns a tuple "
              "(A_ndarray, b_ndarray, state_row_indices, "
              "state_is_cap, b_projection). "
              "Used by pulsim.dsed.PEDSimulatorAuto through the "
              "Python CircuitBuilderAdapter. "
              "Supports grounded + floating capacitors via T^T·M·T "
              "congruence (Phase 5.1b). Time-varying source overlay "
              "via b_projection · b_extra_mna(t) at each PED step "
              "(Bridge.6).");

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
        .def_readwrite("store_every",
                        &SimulationOptions::store_every,
                        "Output decimation: record every m-th step "
                        "(default 1 = every step). The solver still "
                        "integrates at `dt`; only what is STORED "
                        "changes, and the recorded grid stays "
                        "uniform at m*dt so FFT/harmonic analysis "
                        "stays valid. Cuts result memory by m on "
                        "long high-fidelity runs.")
        .def("expected_sample_count",
              &SimulationOptions::expected_sample_count,
              "Number of samples the run will RECORD "
              "(expected_step_count() decimated by store_every).")
        .def_readwrite("strict_event_iterations",
                        &SimulationOptions::strict_event_iterations,
                        "Phase-0 fix #9: true restores the pre-v1.8 "
                        "hard throw on diode event-iteration cycles/"
                        "budget exhaustion; false (default) records "
                        "an EventIterationBreach and continues.")
        .def_readwrite("max_dt_halvings",
                        &SimulationOptions::max_dt_halvings,
                        "v2.0 Phase 2: when a step's inner solve "
                        "fails, re-take it as 2^n sub-steps of "
                        "dt/2^n, up to this many halvings, instead "
                        "of ending the run. The output grid is "
                        "unchanged — sub-steps are internal. 0 "
                        "restores the pre-v2.0 hard failure.")
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
        .def_readwrite("inductor_freeze_di_max",
                        &SimulationOptions::inductor_freeze_di_max)
        .def_readwrite("inductor_abs_clamp",
                        &SimulationOptions::inductor_abs_clamp)
        // Phase 2.4 adaptive RK selector (schema v1.5, wiring v1.6).
        .def_readwrite("integrator",
                        &SimulationOptions::integrator)
        .def_readwrite("rtol", &SimulationOptions::rtol)
        .def_readwrite("atol", &SimulationOptions::atol)
        .def_readwrite("dt_init", &SimulationOptions::dt_init)
        .def("valid", &SimulationOptions::valid)
        .def("expected_step_count",
              &SimulationOptions::expected_step_count);

    // ---- MMC Thevenin arm (v2.0 Phase 3, "obra No.2") ---------------
    py::class_<mmc::ThevArmParams>(m, "ThevArmParams",
        "GGJ Thevenin-aggregated MMC arm parameters.")
        .def(py::init([](Size n_sm, Real c_sm, Real r_on, Real dt,
                          Real v_c_init) {
                 mmc::ThevArmParams p;
                 p.n_sm = n_sm;
                 p.c_sm = c_sm;
                 p.r_on = r_on;
                 p.dt = dt;
                 p.v_c_init = v_c_init;
                 return p;
             }),
             py::arg("n_sm"), py::arg("c_sm"),
             py::arg("r_on") = Real{1e-3}, py::arg("dt") = Real{0},
             py::arg("v_c_init") = Real{0})
        .def_readwrite("n_sm", &mmc::ThevArmParams::n_sm)
        .def_readwrite("c_sm", &mmc::ThevArmParams::c_sm)
        .def_readwrite("r_on", &mmc::ThevArmParams::r_on)
        .def_readwrite("dt", &mmc::ThevArmParams::dt)
        .def_readwrite("v_c_init", &mmc::ThevArmParams::v_c_init);

    py::class_<mmc::ThevArm>(m, "ThevArm",
        "One half-bridge MMC arm under GGJ (PSCAD-style) Thevenin "
        "aggregation: the network sees a single (R_eq, V_eq) branch "
        "that costs the mode cache ZERO bits; capacitor voltages are "
        "back-solved after each network solve with the same "
        "trapezoidal rule the rest of the circuit uses.")
        .def(py::init<const mmc::ThevArmParams&>(), py::arg("params"))
        .def("pre_step",
             [](mmc::ThevArm& a, Real i_arm_prev, Size n_on) {
                 const auto st = a.pre_step(i_arm_prev, n_on);
                 return py::make_tuple(st.r_eq, st.v_eq,
                                        st.r_changed);
             },
             py::arg("i_arm_prev"), py::arg("n_on"),
             "Finalise the previous step's capacitor voltages and "
             "return (r_eq, v_eq, r_changed) for the coming step.")
        .def("finalize_step", &mmc::ThevArm::finalize_step,
             py::arg("i_arm_prev"),
             "Fold the LAST solved step into v_c without "
             "re-selecting — end-of-run bookkeeping (pre_step's "
             "re-selection would hand phantom trailing half-steps "
             "to capacitors 'leaving' a step that never runs).")
        .def("set_v_c", &mmc::ThevArm::set_v_c,
             py::arg("i"), py::arg("v"))
        .def_property_readonly("v_c",
             [](const mmc::ThevArm& a) {
                 return std::vector<Real>(a.v_c());
             })
        .def_property_readonly("inserted",
             [](const mmc::ThevArm& a) {
                 std::vector<int> out(a.inserted().begin(),
                                       a.inserted().end());
                 return out;
             })
        .def_property_readonly("r_c", &mmc::ThevArm::r_c)
        .def_property_readonly("total_stored_voltage",
             &mmc::ThevArm::total_stored_voltage);

    py::class_<solver::ImplausibleVoltage>(m, "ImplausibleVoltage",
        "A node whose voltage left the range the circuit's own "
        "sources can account for. `node == -1` means the trace is "
        "plausible.")
        .def_readonly("node", &solver::ImplausibleVoltage::node)
        .def_readonly("peak", &solver::ImplausibleVoltage::peak)
        .def_readonly("source_scale",
                       &solver::ImplausibleVoltage::source_scale)
        .def_readonly("t_peak", &solver::ImplausibleVoltage::t_peak);

    m.def("find_implausible_voltage",
        [](const topology::Graph& graph, const pwl::DevicePool& pool,
            const solver::SimulationResult& result, Real factor) {
            return solver::find_implausible_voltage(graph, pool,
                                                      result, factor);
        },
        py::arg("graph"), py::arg("pool"), py::arg("result"),
        py::arg("factor") = Real{100},
        "Scan a finished run for a node voltage the circuit's "
        "sources cannot account for. Reads only; changes nothing.");

    m.def("describe_implausible_voltage",
        [](const topology::Graph& graph, const pwl::DevicePool& pool,
            const solver::ImplausibleVoltage& v) {
            return solver::describe(graph, pool, v);
        },
        py::arg("graph"), py::arg("pool"), py::arg("finding"));

    py::class_<solver::InductorGuardAction>(m, "InductorGuardAction",
        "A post-solve guard OVERWROTE the solver's answer for an "
        "inductor current. `inductor_freeze_di_max` and "
        "`inductor_abs_clamp` do not solve anything — they replace a "
        "number the solver produced with one you configured — so a "
        "guard that fires means the plotted current is the LIMIT, "
        "not the circuit, and the model is missing something "
        "(usually a snubber across a path that opens).")
        .def_readonly("branch_id",
                       &solver::InductorGuardAction::branch_id)
        .def_readonly("freeze_count",
                       &solver::InductorGuardAction::freeze_count,
                       "Steps snapped back to the previous current.")
        .def_readonly("clamp_count",
                       &solver::InductorGuardAction::clamp_count,
                       "Steps hard-limited to the absolute bound.")
        .def_readonly("t_first",
                       &solver::InductorGuardAction::t_first)
        .def_readonly("worst_solved",
                       &solver::InductorGuardAction::worst_solved,
                       "The largest |i| the solver actually produced.")
        .def_readonly("reported_limit",
                       &solver::InductorGuardAction::reported_limit,
                       "The |i| you see in the trace instead.")
        .def("total", &solver::InductorGuardAction::total)
        .def("__repr__", [](const solver::InductorGuardAction& g) {
            return "<InductorGuardAction branch=" +
                    std::to_string(g.branch_id) + " fired=" +
                    std::to_string(g.total()) + ">";
        });

    py::class_<solver::DtRetry>(m, "DtRetry",
        "A step the solver could not take at the nominal dt and "
        "re-took as 2^halvings sub-steps. The sampling grid is "
        "unaffected; what changed is that the interval was "
        "integrated more finely than requested.")
        .def_readonly("t", &solver::DtRetry::t)
        .def_readonly("halvings", &solver::DtRetry::halvings)
        .def_readonly("reason", &solver::DtRetry::reason,
                       "What the nominal-dt attempt reported.")
        .def("__repr__", [](const solver::DtRetry& r) {
            return "<DtRetry t=" + std::to_string(r.t) + " dt/" +
                    std::to_string(std::size_t{1} << r.halvings) + ">";
        });

    py::class_<solver::EventIterationBreach>(m, "EventIterationBreach",
        "One time step where the diode event-iteration loop hit a "
        "mask cycle or its budget (Phase-0 fix #9). The solver kept "
        "the last consistent solve and continued.")
        .def_readonly("t", &solver::EventIterationBreach::t)
        .def_readonly("iterations",
                       &solver::EventIterationBreach::iterations)
        .def_readonly("cycle_detected",
                       &solver::EventIterationBreach::cycle_detected);

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
        "`states` arrays, plus event diagnostics.",
        py::dynamic_attr())  // allow `result._builder = b` from Python
        .def_readonly("times", &SimulationResult::times)
        .def_readonly("final_snapshot",
                       &SimulationResult::final_snapshot,
                       "Complete state at the end of the run. Pass "
                       "it to simulate(resume_from=...) to continue "
                       "EXACTLY where this left off.")
        // v2.0 Phase 1 (audit finding
        // `waveform-storage-vector-of-vectors`): `states` is ONE
        // contiguous (n_samples x n_state) buffer in C++, handed
        // out here as a ZERO-COPY 2-D numpy view whose base is the
        // result object itself (so the buffer outlives the view).
        //
        // v1.x returned a fresh Python LIST of n_samples 1-D
        // ndarrays — a full copy of the entire run on EVERY
        // attribute access. Indexing, slicing, iteration, len(),
        // and np.asarray() all behave the same on the 2-D array;
        // the view is READ-ONLY because it aliases kernel memory
        // (call `.copy()` to get a mutable array).
        .def_property_readonly("states",
            [](py::object self) -> py::object {
                const auto& r = self.cast<const SimulationResult&>();
                const auto& tr = r.states;
                const auto rows = static_cast<py::ssize_t>(tr.rows());
                const auto cols = static_cast<py::ssize_t>(tr.cols());
                if (rows == 0 || cols == 0 ||
                    tr.data() == nullptr) {
                    // Nothing recorded: an owned empty array, so
                    // shape stays well-defined without pointing at
                    // a null buffer.
                    return py::array_t<Real>(
                        std::vector<py::ssize_t>{rows, cols});
                }
                py::array_t<Real> arr(
                    std::vector<py::ssize_t>{rows, cols},
                    std::vector<py::ssize_t>{
                        static_cast<py::ssize_t>(sizeof(Real)) * cols,
                        static_cast<py::ssize_t>(sizeof(Real))},
                    tr.data(),
                    self);   // base → keeps the result alive
                arr.attr("setflags")(py::arg("write") = false);
                return std::move(arr);
            },
            "Recorded state samples as a read-only "
            "(num_steps, state_size) numpy view — zero-copy over "
            "the kernel's contiguous buffer. Use `.copy()` for a "
            "mutable array.")
        .def_property_readonly("total_bytes",
            [](const SimulationResult& r) { return r.approx_bytes(); },
            "Estimated bytes held by the whole result: the "
            "contiguous sample buffer plus the parallel times and "
            "diagnostic vectors.")
        .def_property_readonly("states_bytes",
            [](const SimulationResult& r) { return r.states.bytes(); },
            "Bytes held by the contiguous sample buffer.")
        .def_readonly("event_iteration_count",
                       &SimulationResult::event_iteration_count)
        .def_readonly("commutation_events",
                       &SimulationResult::commutation_events)
        .def_readonly("inductor_guard_actions",
                       &SimulationResult::inductor_guard_actions,
                       "Inductors whose current a post-solve guard "
                       "overwrote. Empty unless a guard was enabled "
                       "AND fired.")
        .def_readonly("dt_retries", &SimulationResult::dt_retries,
                       "Steps that had to be re-taken at a reduced "
                       "dt. Empty on a run that never needed one.")
        .def_readonly("event_iteration_breaches",
                       &SimulationResult::event_iteration_breaches)
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
    // -------------------------------------------------------------------
    // Bridge.12 — native PWM switch_fn classes. The DSED scheduler
    // bindings (dsed_bindings.cpp::PySwitchFnSSM) detect these at
    // construction and call operator() / next_edge_after() in pure C++,
    // bypassing pybind11 entirely. ~25× cheaper per-call than the
    // equivalent Python `class PWM` pattern.
    // -------------------------------------------------------------------

    py::class_<sources::NativePwm2Switch>(m, "NativePwm2Switch",
        "Native 2-switch complementary PWM for DSED. Pass directly "
        "as `switch_fn=` to `pulsim.simulate(engine='dsed', ...)` — "
        "the dispatch detects the type and calls the C++ implementation "
        "without GIL roundtrips per scheduler step. ~25× faster per "
        "switch_fn call than the equivalent Python class.")
        .def(py::init<Real, Real, int, bool>(),
              py::arg("T_sw"), py::arg("D"), py::arg("n_switches"),
              py::arg("hs_first") = true,
              "Build PWM with period T_sw (s), duty D in [0, 1], "
              "n_switches in the mask. `hs_first=True` (default): "
              "switch 0 closed during D·T, switch 1 during (1-D)·T. "
              "Set to False for boost-style (LS closed first).")
        .def("__call__",
              [](const sources::NativePwm2Switch& self, Real t) {
                  return self(t);
              },
              py::arg("t"),
              "Return the active mask at time `t`.")
        .def("next_edge_after",
              &sources::NativePwm2Switch::next_edge_after,
              py::arg("t"),
              "Return the next gate-edge time strictly greater than `t`.")
        .def_readonly("T_sw", &sources::NativePwm2Switch::T_sw)
        .def_readonly("D", &sources::NativePwm2Switch::D);

    py::class_<sources::NativeMultiMaskPwm>(m, "NativeMultiMaskPwm",
        "Native multi-mask PWM (K-mask schedule cycling through "
        "phase boundaries). For 3-level inverters (P→O→N pattern), "
        "multi-leg interleaved converters, etc. Same fast-path "
        "detection as NativePwm2Switch.")
        .def(py::init<Real, std::vector<Real>,
                       std::vector<topology::SwitchStateMask>>(),
              py::arg("T_sw"), py::arg("phase_boundaries"),
              py::arg("masks"),
              "Build a K-mask PWM. `phase_boundaries[k]` is the END "
              "of mask k's active window as a fraction of T_sw "
              "(strictly increasing in (0, 1]). `masks[k]` is the "
              "SwitchStateMask active in window k.")
        .def("__call__",
              [](const sources::NativeMultiMaskPwm& self, Real t) {
                  return self(t);
              },
              py::arg("t"))
        .def("next_edge_after",
              &sources::NativeMultiMaskPwm::next_edge_after,
              py::arg("t"))
        .def_readonly("T_sw", &sources::NativeMultiMaskPwm::T_sw);

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
    // ---- LiveRing — backs ``pulsim.NativeLiveStream`` --------------------
    //
    // C++ ring buffer holding the (t, x) stream during a transient.
    // ``simulate(live_stream=…)`` attaches an instance via
    // ``NativeLiveStream.attach(state_size)`` which constructs
    // ``LiveRingHandle(...)`` here. The kernel pushes samples into
    // this buffer under ``py::gil_scoped_release``; a separate
    // (GUI) thread reads ``t_buf``/``x_buf`` as numpy views.
    py::class_<streaming::LiveRing, std::shared_ptr<streaming::LiveRing>>(
        m, "LiveRingHandle",
        "Fixed-capacity SPSC ring buffer for live ``(t, x)`` "
        "streaming. Constructed by ``NativeLiveStream.attach()``; "
        "read via the zero-copy ``t_buf`` / ``x_buf`` numpy views.")
        .def(py::init([](std::size_t capacity,
                          std::size_t state_size,
                          std::size_t decimate) {
                  return std::make_shared<streaming::LiveRing>(
                      capacity, state_size, decimate);
              }),
              py::arg("capacity"),
              py::arg("state_size"),
              py::arg("decimate") = 1,
              "Allocate the (t, x) ring. ``capacity`` must be > 0; "
              "``decimate`` defaults to 1 (no decimation).")
        .def_property_readonly("capacity",
            &streaming::LiveRing::capacity)
        .def_property_readonly("state_size",
            &streaming::LiveRing::state_size)
        .def_property_readonly("decimate",
            &streaming::LiveRing::decimate)
        .def("get_head", &streaming::LiveRing::head,
            "Monotonic count of samples ever pushed. Wrap into "
            "slot index via ``% capacity``.")
        .def("request_stop", &streaming::LiveRing::request_stop,
            "Ask the simulation to halt at the next step boundary.")
        .def("stopped", &streaming::LiveRing::stopped,
            "True iff ``request_stop`` was called.")
        .def_property_readonly("t_buf",
            [](streaming::LiveRing& self) {
                // Zero-copy numpy view. The capsule keeps the
                // LiveRing alive while the array references it.
                return py::array_t<Real>(
                    { self.capacity() },        // shape
                    { sizeof(Real) },           // strides
                    self.t_data(),              // data
                    py::cast(&self)             // owner — keeps alive
                );
            },
            "Numpy view of the ring's time buffer, shape "
            "``(capacity,)``. Zero-copy; aliases the C++ memory.")
        .def_property_readonly("x_buf",
            [](streaming::LiveRing& self) {
                const std::size_t c = self.capacity();
                const std::size_t n = self.state_size();
                return py::array_t<Real>(
                    { c, n },                                    // shape
                    { sizeof(Real) * n, sizeof(Real) },          // strides
                    self.x_data(),                               // data
                    py::cast(&self)                              // owner
                );
            },
            "Numpy view of the ring's state buffer, shape "
            "``(capacity, state_size)``. Zero-copy; aliases the "
            "C++ memory.");

    py::class_<SolverSnapshot>(m, "SolverSnapshot",
        "The COMPLETE state of a run at one instant.\n\n"
        "`initial_state=` restores only the MNA vector and then "
        "INVENTS the companion history from it (a capacitor gets "
        "i_prev = 0, an inductor v_prev = 0), so resuming from it "
        "does NOT reproduce the run — a continuous 2T RLC and a "
        "T-then-resume differ by 2.3e-4 where a true resume is "
        "~1e-15. A snapshot carries what was missing: the "
        "trapezoidal companion history and the solver-owned diode "
        "bits. Pass it as simulate(resume_from=...).")
        .def(py::init<>())
        .def_readwrite("t", &SolverSnapshot::t)
        .def_readwrite("x", &SolverSnapshot::x)
        .def_readwrite("history", &SolverSnapshot::history)
        .def_readwrite("diode_on", &SolverSnapshot::diode_on)
        .def_readwrite("valid", &SolverSnapshot::valid)
        .def("__repr__", [](const SolverSnapshot& s) {
            return "<SolverSnapshot t=" + std::to_string(s.t)
                   + " n=" + std::to_string(s.x.size())
                   + " history=" + std::to_string(s.history.size())
                   + (s.valid ? " valid>" : " EMPTY>");
        });

    m.def("run_transient",
        [](const pwl::PwlStateSpaceCache& cache,
           const topology::Graph& graph,
           const pwl::DevicePool& pool,
           const SimulationOptions& opts,
           py::object switch_fn_obj,         // Bridge.13: was SwitchScheduleFn
           BExtraFn b_extra_fn,
           bool start_from_dc_op,
           bool enable_nonlinear_refresh,
           StepObserverFn step_observer,
           py::object initial_state,
           ShouldContinueFn should_continue,
           std::shared_ptr<streaming::LiveRing> live_ring,
           py::object resume_from_obj) {
            // Bridge.13 — detect native PWM switch_fn (NativePwm2Switch
            // / NativeMultiMaskPwm) and build a std::function lambda
            // that calls the C++ method directly, no Python __call__.
            // For generic Python callables, fall back to pybind11's
            // auto-conversion path.
            SwitchScheduleFn switch_fn;
            {
                sources::NativePwm2Switch* native_pwm2 = nullptr;
                sources::NativeMultiMaskPwm* native_multi = nullptr;
                try {
                    native_pwm2 = switch_fn_obj.cast<
                        sources::NativePwm2Switch*>();
                } catch (const py::cast_error&) {}
                if (!native_pwm2) {
                    try {
                        native_multi = switch_fn_obj.cast<
                            sources::NativeMultiMaskPwm*>();
                    } catch (const py::cast_error&) {}
                }
                if (native_pwm2) {
                    // Capture py::object to keep the C++ instance alive
                    // for the lambda's lifetime (pybind11 ref-count).
                    auto keepalive = switch_fn_obj;
                    switch_fn = [native_pwm2, keepalive](Real t) {
                        return (*native_pwm2)(t);
                    };
                } else if (native_multi) {
                    auto keepalive = switch_fn_obj;
                    switch_fn = [native_multi, keepalive](Real t) {
                        return (*native_multi)(t);
                    };
                } else {
                    // Generic Python callable — pybind11 wraps as
                    // std::function via its automatic converter.
                    switch_fn = switch_fn_obj.cast<SwitchScheduleFn>();
                }
            }
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
            // Live-streaming hook: if the caller supplied a
            // ``LiveRing`` we wrap the user-supplied
            // step_observer + should_continue with composites that
            // also push samples and honour the ring's stop flag.
            // The composites are GIL-free in their LiveRing parts
            // (atomics only); the user-supplied callbacks still
            // re-acquire the GIL via pybind11 trampolines.
            StepObserverFn observer_eff = step_observer;
            ShouldContinueFn continue_eff = should_continue;
            if (live_ring) {
                auto ring = live_ring;  // shared_ptr copy into closure
                auto user_obs = step_observer;
                observer_eff = [ring, user_obs](Real t, const Vector& x) {
                    if (user_obs) {
                        user_obs(t, x);
                    }
                    if (ring->should_decimate()) {
                        ring->push(t, x.data(),
                                   static_cast<std::size_t>(x.size()));
                    }
                };
                auto user_cont = should_continue;
                continue_eff = [ring, user_cont]() {
                    if (ring->stopped()) {
                        return false;
                    }
                    return user_cont ? user_cont() : true;
                };
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
            SolverSnapshot resume_snap;
            const SolverSnapshot* resume_ptr = nullptr;
            if (!resume_from_obj.is_none()) {
                resume_snap = resume_from_obj.cast<SolverSnapshot>();
                resume_ptr = &resume_snap;
            }
            {
                py::gil_scoped_release rel;
                result = run_transient(cache, graph, pool, opts,
                                          switch_fn, b_extra_fn,
                                          start_from_dc_op,
                                          nl_refresh,
                                          observer_eff,
                                          x_init_ptr,
                                          continue_eff,
                                          resume_ptr);
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
        py::arg("live_ring") = std::shared_ptr<streaming::LiveRing>{},
        py::arg("resume_from") = py::none(),
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
                "Single-shot solve — fastest; the common case never "
                "leaves this rung.")
        .value("GminStepping",     pwl::DCStrategy::GminStepping,
                "Conductance homotopy: clamp every node to ground, "
                "then relax by decades. Fixes a badly-pivoted matrix "
                "AND a Newton that cannot find the basin.")
        .value("SourceStepping",   pwl::DCStrategy::SourceStepping,
                "Source-amplitude homotopy from α=0 to α=1. Fixes a "
                "Newton basin problem; cannot fix a singular matrix.")
        .value("PseudoTransient",  pwl::DCStrategy::PseudoTransient,
                "Integrate dx/dt = -F to equilibrium. Last resort: "
                "unstable on MNA voltage-source constraint rows.")
        .value("Auto",             pwl::DCStrategy::Auto,
                "naive → gmin stepping → source stepping → "
                "pseudo-transient, first one that answers wins.");

    py::class_<pwl::DCSolveReport>(m, "DCSolveReport",
        "Which rung of the DC cascade produced the answer.")
        .def_property_readonly("strategy",
            [](const pwl::DCSolveReport& r) -> std::string {
                // Return the same vocabulary `compute_dc_op(
                // strategy=...)` accepts. One name per concept beats
                // a str argument and an enum result that look alike
                // and do not compare equal.
                switch (r.strategy) {
                case pwl::DCStrategy::Naive:           return "naive";
                case pwl::DCStrategy::GminStepping:    return "gmin_step";
                case pwl::DCStrategy::SourceStepping:  return "source_step";
                case pwl::DCStrategy::PseudoTransient: return "pseudo_trans";
                case pwl::DCStrategy::Auto:            return "auto";
                }
                return "unknown";
            },
            "The rung that succeeded, as one of the strategy names "
            "`compute_dc_op()` accepts.")
        .def_readonly("rungs_attempted",
                       &pwl::DCSolveReport::rungs_attempted,
                       "Homotopy rungs solved, including retries "
                       "inserted when a decade was too wide.")
        .def_readonly("final_gmin", &pwl::DCSolveReport::final_gmin,
                       "Conductance-to-ground (S) still present in "
                       "the returned answer.")
        .def_readonly("residual", &pwl::DCSolveReport::residual,
                       "||f(x)||_inf of the ORIGINAL circuit — not "
                       "the regularized one that was solved.")
        .def("summary", &pwl::DCSolveReport::summary)
        .def("__repr__", [](const pwl::DCSolveReport& r) {
            return "<DCSolveReport: " + r.summary() + ">";
        });

    py::class_<pwl::DCOperatingPoint>(m, "DCOperatingPoint",
        "A resolved DC operating point: the state vector, the switch "
        "and diode configuration it was solved at, and how it was "
        "obtained.")
        .def_property_readonly("x",
            [](const pwl::DCOperatingPoint& op) {
                // An owned, writable copy. `def_readonly` on an Eigen
                // member hands back a READ-ONLY view that also pins
                // the C++ object alive — the same leak Phase 1's
                // review caught in res.v()/res.i(). An operating
                // point is a plain vector users subtract, scale and
                // index into.
                return Vector(op.x);
            },
            "The DC state vector (an owned, writable copy).")
        .def_readonly("mask", &pwl::DCOperatingPoint::mask,
                       "Switch mask with the resolved PWL diode bits "
                       "overlaid.")
        .def_readonly("report", &pwl::DCOperatingPoint::report)
        .def_readonly("event_iterations",
                       &pwl::DCOperatingPoint::event_iterations,
                       "Diode-consistency rounds it took to settle.");

    // ---- Adaptive RK (Phase 2.4) ----
    m.def("dopri5_solve",
        [](py::function f_py,
            Real t0, Real t_end,
            const Eigen::VectorXd& x0,
            Real rtol, Real atol,
            Real dt_min, Real dt_max,
            Real safety,
            Real growth_max, Real shrink_min,
            Real dt_init) {
            // Wrap the Python callable in a typed std::function so the
            // C++ template can call it.
            auto f_cpp = [f_py](Real t, const Eigen::VectorXd& x)
                -> Eigen::VectorXd {
                py::gil_scoped_acquire gil;
                py::object res = f_py(t, x);
                return py::cast<Eigen::VectorXd>(res);
            };
            pulsim::integrators::DormandPrince5<decltype(f_cpp)>
                solver(f_cpp);
            solver.rtol = rtol;
            solver.atol = atol;
            solver.dt_min = dt_min;
            solver.dt_max = (dt_max > 0)
                ? dt_max
                : std::numeric_limits<Real>::infinity();
            solver.safety = safety;
            solver.growth_max = growth_max;
            solver.shrink_min = shrink_min;

            py::gil_scoped_release release;
            auto sol = solver.solve(t0, t_end, x0, dt_init);
            py::gil_scoped_acquire acquire;

            // Convert the trajectory to NumPy arrays.
            const Index n_pts = static_cast<Index>(sol.t.size());
            const Index n_dim = x0.size();
            py::array_t<Real> t_arr(n_pts);
            py::array_t<Real> x_arr(
                std::vector<py::ssize_t>{n_pts, n_dim});
            auto t_ptr = static_cast<Real*>(t_arr.request().ptr);
            auto x_ptr = static_cast<Real*>(x_arr.request().ptr);
            for (Index i = 0; i < n_pts; ++i) {
                t_ptr[i] = sol.t[i];
                for (Index j = 0; j < n_dim; ++j) {
                    x_ptr[i * n_dim + j] = sol.x[i][j];
                }
            }
            return py::dict(
                "t"_a = std::move(t_arr),
                "x"_a = std::move(x_arr),
                "n_accepted"_a = sol.n_accepted,
                "n_rejected"_a = sol.n_rejected,
                "n_f_evals"_a = sol.n_f_evals);
        },
        py::arg("f"), py::arg("t0"), py::arg("t_end"),
        py::arg("x0"),
        py::arg("rtol") = Real{1e-5},
        py::arg("atol") = Real{1e-8},
        py::arg("dt_min") = Real{1e-12},
        py::arg("dt_max") = Real{0.0},   // 0 → infinity
        py::arg("safety") = Real{0.9},
        py::arg("growth_max") = Real{5.0},
        py::arg("shrink_min") = Real{0.2},
        py::arg("dt_init") = Real{-1.0},  // negative → use heuristic
        "Dormand-Prince 5(4) embedded adaptive Runge-Kutta solver "
        "(Phase 2.4 C++ port). Solves dx/dt = f(t, x) from t0 to t_end. "
        "Returns a dict with t, x, n_accepted, n_rejected, n_f_evals. "
        "The Python f(t, x) callable is invoked from C++ with the GIL "
        "re-acquired around each evaluation.");

    m.def("radau_iia3_solve",
        [](py::function f_py,
            Real t0, Real t_end,
            const Eigen::VectorXd& x0,
            Real rtol, Real atol,
            Real dt_min, Real dt_max,
            Real safety,
            Real growth_max, Real shrink_min,
            int newton_max_iter, Real newton_tol,
            Real dt_init) {
            auto f_cpp = [f_py](Real t, const Eigen::VectorXd& x)
                -> Eigen::VectorXd {
                py::gil_scoped_acquire gil;
                py::object res = f_py(t, x);
                return py::cast<Eigen::VectorXd>(res);
            };
            pulsim::integrators::RadauIIA3<decltype(f_cpp)>
                solver(f_cpp);
            solver.rtol = rtol;
            solver.atol = atol;
            solver.dt_min = dt_min;
            solver.dt_max = (dt_max > 0)
                ? dt_max
                : std::numeric_limits<Real>::infinity();
            solver.safety = safety;
            solver.growth_max = growth_max;
            solver.shrink_min = shrink_min;
            solver.newton_max_iter = newton_max_iter;
            solver.newton_tol = newton_tol;

            py::gil_scoped_release release;
            auto sol = solver.solve(t0, t_end, x0, dt_init);
            py::gil_scoped_acquire acquire;

            const Index n_pts = static_cast<Index>(sol.t.size());
            const Index n_dim = x0.size();
            py::array_t<Real> t_arr(n_pts);
            py::array_t<Real> x_arr(
                std::vector<py::ssize_t>{n_pts, n_dim});
            auto t_ptr = static_cast<Real*>(t_arr.request().ptr);
            auto x_ptr = static_cast<Real*>(x_arr.request().ptr);
            for (Index i = 0; i < n_pts; ++i) {
                t_ptr[i] = sol.t[i];
                for (Index j = 0; j < n_dim; ++j) {
                    x_ptr[i * n_dim + j] = sol.x[i][j];
                }
            }
            return py::dict(
                "t"_a = std::move(t_arr),
                "x"_a = std::move(x_arr),
                "n_accepted"_a = sol.n_accepted,
                "n_rejected"_a = sol.n_rejected,
                "n_f_evals"_a = sol.n_f_evals);
        },
        py::arg("f"), py::arg("t0"), py::arg("t_end"),
        py::arg("x0"),
        py::arg("rtol") = Real{1e-5},
        py::arg("atol") = Real{1e-8},
        py::arg("dt_min") = Real{1e-12},
        py::arg("dt_max") = Real{0.0},
        py::arg("safety") = Real{0.9},
        py::arg("growth_max") = Real{5.0},
        py::arg("shrink_min") = Real{0.2},
        py::arg("newton_max_iter") = 8,
        py::arg("newton_tol") = Real{1e-8},
        py::arg("dt_init") = Real{-1.0},
        "Radau IIA(3) implicit Runge-Kutta solver (Phase 2.4 C++ port). "
        "L-stable, order 3 — designed for stiff ODEs. 2-stage Newton "
        "block per step + step-doubling error estimate.");

    // Build the standard static-device nonlinear chain when the
    // caller wants one. `std::nullopt` means "decide from the pool" —
    // a bool default here could not distinguish "the user asked for
    // no refresh" from "the user said nothing", and getting that
    // wrong is the difference between 0.7 V and 5.0 V at a diode.
    auto resolve_refresh =
        [](const pwl::DevicePool& pool,
            std::optional<bool> enable) -> pwl::NonlinearRefreshFn {
            const bool want =
                enable.has_value() ? *enable
                                    : pool.has_nonlinear_devices();
            if (!want) {
                return {};
            }
            return pwl::make_combined_diode_mosfet_refresh();
        };

    m.def("compute_dc_op_with_strategy",
        [resolve_refresh](const topology::Graph& graph,
            const pwl::DevicePool& pool,
            const topology::SwitchStateMask& mask,
            pwl::DCStrategy strategy,
            Real t_eval,
            Real pt_dt_init,
            Real pt_dt_max,
            Size pt_max_iters,
            Real pt_tol_res,
            Size ss_n_steps,
            analysis::ShouldContinueFn should_continue,
            std::optional<bool> enable_nonlinear_refresh,
            Real gmin,
            Real gmin_start,
            Size gmin_steps) {
            pwl::PseudoTransientConfig pt;
            pt.dt_init = pt_dt_init;
            pt.dt_max  = pt_dt_max;
            pt.max_iters = pt_max_iters;
            pt.tol_res = pt_tol_res;
            pwl::SourceSteppingConfig ss;
            ss.n_steps = ss_n_steps;
            pwl::GminConfig g;
            g.floor = gmin;
            g.start = gmin_start;
            g.steps = gmin_steps;
            pwl::DCSolveReport report;
            auto refresh =
                resolve_refresh(pool, enable_nonlinear_refresh);
            Vector x = pwl::compute_dc_op_with_strategy(
                graph, pool, mask, strategy, t_eval, pt, ss,
                std::move(should_continue), refresh, g, &report);
            return py::make_tuple(std::move(x), report);
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("strategy") = pwl::DCStrategy::Auto,
        py::arg("t_eval") = Real{0},
        py::arg("pt_dt_init") = Real{1.0},
        py::arg("pt_dt_max") = Real{1e10},
        py::arg("pt_max_iters") = Size{500},
        py::arg("pt_tol_res") = Real{1e-7},
        py::arg("ss_n_steps") = Size{10},
        py::arg("should_continue") = analysis::ShouldContinueFn{},
        py::arg("enable_nonlinear_refresh") = py::none(),
        py::arg("gmin") = pwl::kDefaultGmin,
        py::arg("gmin_start") = Real{1e-2},
        py::arg("gmin_steps") = Size{10},
        "Compute the DC operating point with strategy selection. "
        "Returns `(x, DCSolveReport)`.\n"
        "`enable_nonlinear_refresh=None` (the default) stamps the "
        "smooth diode / MOSFET / IGBT chain when the pool contains "
        "any; pass False to deliberately solve them as open "
        "circuits.\n"
        "Optional `should_continue` is invoked between fallback\n"
        "strategies in Auto mode; raises pulsim.Cancelled when it\n"
        "returns False.");

    m.def("compute_dc_operating_point",
        [resolve_refresh](const topology::Graph& graph,
            const pwl::DevicePool& pool,
            const topology::SwitchStateMask& mask,
            pwl::DCStrategy strategy,
            Real t_eval,
            std::optional<bool> enable_nonlinear_refresh,
            Size max_event_iterations,
            Size max_newton_iters,
            Real tol_dx,
            Real tol_res,
            bool enable_cascade,
            Real gmin,
            Real gmin_start,
            Size gmin_steps,
            Size ss_n_steps,
            Real pt_tol_res,
            Size pt_max_iters,
            analysis::ShouldContinueFn should_continue) {
            pwl::DCOperatingPointOptions o;
            o.should_continue = std::move(should_continue);
            o.strategy = strategy;
            o.t_eval = t_eval;
            o.max_event_iterations = max_event_iterations;
            o.max_newton_iters = max_newton_iters;
            o.tol_dx = tol_dx;
            o.tol_res = tol_res;
            o.enable_cascade = enable_cascade;
            o.gmin.floor = gmin;
            o.gmin.start = gmin_start;
            o.gmin.steps = gmin_steps;
            o.source_stepping.n_steps = ss_n_steps;
            o.pseudo_transient.tol_res = pt_tol_res;
            o.pseudo_transient.max_iters = pt_max_iters;
            auto refresh =
                resolve_refresh(pool, enable_nonlinear_refresh);
            pwl::DiodeEventState diodes{graph, pool};
            diodes.reset();
            return pwl::compute_dc_operating_point(
                graph, pool, mask, refresh, o,
                diodes.num_diodes() > 0 ? &diodes : nullptr,
                "compute_dc_operating_point");
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("strategy") = pwl::DCStrategy::Auto,
        py::arg("t_eval") = Real{0},
        py::arg("enable_nonlinear_refresh") = py::none(),
        py::arg("max_event_iterations") = Size{16},
        py::arg("max_newton_iters") = Size{50},
        py::arg("tol_dx") = Real{1e-9},
        py::arg("tol_res") = Real{1e-9},
        py::arg("enable_cascade") = true,
        py::arg("gmin") = pwl::kDefaultGmin,
        py::arg("gmin_start") = Real{1e-2},
        py::arg("gmin_steps") = Size{10},
        py::arg("ss_n_steps") = Size{10},
        py::arg("pt_tol_res") = Real{1e-7},
        py::arg("pt_max_iters") = Size{500},
        py::arg("should_continue") = analysis::ShouldContinueFn{},
        "THE DC operating point: nonlinear devices stamped, PWL "
        "diode states iterated to consistency, and the DC cascade "
        "walked if the direct solve fails. Returns a "
        "`DCOperatingPoint`.\n"
        "This is what `simulate(start_from_dc_op=True)` uses "
        "internally; prefer it over `compute_dc_op`, which is the "
        "raw linear primitive and treats every nonlinear device as "
        "an open circuit.");

    // ---- Raw linear DC primitive ---------------------------------------
    m.def("compute_dc_op",
        [](const topology::Graph& graph,
            const pwl::DevicePool& pool,
            const topology::SwitchStateMask& mask,
            Real t_eval,
            Real gmin) {
            return pwl::compute_dc_op(graph, pool, mask, t_eval,
                                        gmin);
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("t_eval") = Real{0},
        py::arg("gmin") = pwl::kDefaultGmin,
        "Raw single-shot LINEAR DC solve. Nonlinear devices (smooth "
        "diode / MOSFET / IGBT) are stamped as OPEN CIRCUITS and PWL "
        "diode states are taken from `mask` as given — so on a "
        "circuit with any of those this answers a different "
        "question. Use compute_dc_operating_point(...) unless you "
        "specifically want the linear system.\n"
        "`gmin` is the conductance floor to ground (S); 0 disables "
        "it.");

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
            Size output_node_idx,
            analysis::ShouldContinueFn should_continue) {
            return analysis::run_mna_sweep(
                graph, pool, mask, freqs,
                input_state_idx, output_node_idx,
                std::move(should_continue));
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("freqs"), py::arg("input_state_idx"),
        py::arg("output_node_idx"),
        py::arg("should_continue") = analysis::ShouldContinueFn{},
        "Direct kernel AC sweep via complex sparse LU at each "
        "frequency. Returns a MnaSweepKernelResult with parallel "
        "freqs/H arrays.\n"
        "Optional `should_continue` is invoked between frequency\n"
        "points; raises pulsim.Cancelled when it returns False.");

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

        .def("add_induction_motor",
            [](BlockChain& self,
                Real J, Real B, Real T_load,
                Real R_s, Real L_s,
                Real R_r, Real L_r, Real L_m,
                int pole_pairs,
                std::array<Index, 3> phase_inductor_idx,
                std::array<Index, 3> bemf_source_idx,
                std::string omega_channel,
                std::string theta_channel,
                std::string psi_alpha_channel,
                std::string psi_beta_channel,
                std::string torque_channel,
                std::string slip_channel) {
                pulsim::motors::Mechanical mech;
                mech.J_kgm2 = J;
                mech.B_Nms_per_rad = B;
                mech.T_load_Nm = T_load;
                pulsim::motors::add_induction_motor_to_chain(
                    self, mech,
                    R_s, L_s, R_r, L_r, L_m, pole_pairs,
                    phase_inductor_idx, bemf_source_idx,
                    std::move(omega_channel),
                    std::move(theta_channel),
                    std::move(psi_alpha_channel),
                    std::move(psi_beta_channel),
                    std::move(torque_channel),
                    std::move(slip_channel));
            },
            py::arg("J"), py::arg("B"), py::arg("T_load"),
            py::arg("R_s"), py::arg("L_s"),
            py::arg("R_r"), py::arg("L_r"), py::arg("L_m"),
            py::arg("pole_pairs"),
            py::arg("phase_inductor_idx"),
            py::arg("bemf_source_idx"),
            py::arg("omega_channel"), py::arg("theta_channel"),
            py::arg("psi_alpha_channel") = std::string{},
            py::arg("psi_beta_channel")  = std::string{},
            py::arg("torque_channel")    = std::string{},
            py::arg("slip_channel")      = std::string{},
            "Squirrel-cage induction motor block (5-state Krause αβ "
            "model). The user must have already added per-phase Rs + "
            "σ·Ls + back-EMF dummy source to the builder and resolved "
            "the per-phase inductor branch-var indices and dummy "
            "source-row indices. Optional psi/torque/slip channels "
            "are written when the channel name is non-empty.")

        // ---- Magnetics (Phase 2.2 / C++ port) ----
        .def("add_hysteretic_inductor",
            [](BlockChain& self,
                Real Ms, Real a, Real alpha, Real c, Real k,
                int N_turns, Real l_m, Real A_core,
                Index inductor_branch_var_idx,
                Index bemf_source_idx,
                std::string M_channel,
                std::string B_channel,
                std::string H_channel,
                std::string vm_channel) {
                pulsim::magnetics::JilesAthertonParams params;
                params.Ms = Ms;
                params.a = a;
                params.alpha = alpha;
                params.c = c;
                params.k = k;
                pulsim::magnetics::add_hysteretic_inductor_to_chain(
                    self, params, N_turns, l_m, A_core,
                    inductor_branch_var_idx, bemf_source_idx,
                    std::move(M_channel),
                    std::move(B_channel),
                    std::move(H_channel),
                    std::move(vm_channel));
            },
            py::arg("Ms"), py::arg("a"), py::arg("alpha"),
            py::arg("c"), py::arg("k"),
            py::arg("N_turns"), py::arg("l_m"), py::arg("A_core"),
            py::arg("inductor_branch_var_idx"),
            py::arg("bemf_source_idx"),
            py::arg("M_channel")  = std::string{},
            py::arg("B_channel")  = std::string{},
            py::arg("H_channel")  = std::string{},
            py::arg("vm_channel") = std::string{},
            "Jiles-Atherton hysteretic-inductor block (Phase 2.2). "
            "The user must have added L_0 air-core + dummy V source "
            "via `pulsim.add_hysteretic_inductor` and resolved the "
            "branch-var indices. Writes v_M = +N·A·μ₀·dM/dt into the "
            "dummy source's residual row. Sign matches the Python "
            "observer's v1.5+ convention.")

        // ---- Sensorless observers (Phase 2.3 / C++ port) ----
        .def("add_sliding_mode_observer",
            [](BlockChain& self,
                Real Rs, Real Ls, Real K_sl, Real omega_lpf,
                Real Kp_pll, Real Ki_pll, Real f_init_hz,
                Real epsilon, Real low_speed_threshold,
                std::string v_alpha_channel, std::string v_beta_channel,
                std::string i_alpha_channel, std::string i_beta_channel,
                std::string theta_hat_channel,
                std::string omega_hat_channel,
                std::string e_alpha_hat_channel,
                std::string e_beta_hat_channel,
                std::string low_speed_flag_channel) {
                pulsim::observers::SlidingModeObserverParams p;
                p.Rs = Rs; p.Ls = Ls; p.K_sl = K_sl;
                p.omega_lpf = omega_lpf;
                p.Kp_pll = Kp_pll; p.Ki_pll = Ki_pll;
                p.f_init_hz = f_init_hz;
                p.epsilon = epsilon;
                p.low_speed_threshold = low_speed_threshold;
                pulsim::observers::add_sliding_mode_observer_to_chain(
                    self, p,
                    std::move(v_alpha_channel),
                    std::move(v_beta_channel),
                    std::move(i_alpha_channel),
                    std::move(i_beta_channel),
                    std::move(theta_hat_channel),
                    std::move(omega_hat_channel),
                    std::move(e_alpha_hat_channel),
                    std::move(e_beta_hat_channel),
                    std::move(low_speed_flag_channel));
            },
            py::arg("Rs"), py::arg("Ls"),
            py::arg("K_sl") = Real{50.0},
            py::arg("omega_lpf") =
                Real{2.0} * Real{3.14159265358979323846} * Real{500.0},
            py::arg("Kp_pll") = Real{200.0},
            py::arg("Ki_pll") = Real{1e4},
            py::arg("f_init_hz") = Real{50.0},
            py::arg("epsilon") = Real{0.5},
            py::arg("low_speed_threshold") = Real{0.05},
            py::arg("v_alpha_channel"), py::arg("v_beta_channel"),
            py::arg("i_alpha_channel"), py::arg("i_beta_channel"),
            py::arg("theta_hat_channel"),
            py::arg("omega_hat_channel"),
            py::arg("e_alpha_hat_channel") = std::string{},
            py::arg("e_beta_hat_channel")  = std::string{},
            py::arg("low_speed_flag_channel") = std::string{},
            "PMSM Sliding-Mode Observer (Phase 2.3 C++ port). Reads "
            "(v_alpha, v_beta, i_alpha, i_beta) channels and writes "
            "(theta_hat, omega_hat, optional e_alpha/beta_hat + "
            "low_speed_flag).")

        .def("add_flux_mras_observer",
            [](BlockChain& self,
                Real Rs, Real Ls, Real Lr, Real Lm, Real Rr,
                Real voltage_model_hpf_omega,
                Real Kp_mras, Real Ki_mras,
                bool normalise_eps, Real eps_norm_floor,
                std::string v_alpha_channel, std::string v_beta_channel,
                std::string i_alpha_channel, std::string i_beta_channel,
                std::string omega_hat_channel,
                std::string psi_adj_alpha_channel,
                std::string psi_adj_beta_channel,
                std::string psi_ref_alpha_channel,
                std::string psi_ref_beta_channel) {
                pulsim::observers::FluxMRASObserverParams p;
                p.Rs = Rs; p.Ls = Ls; p.Lr = Lr; p.Lm = Lm; p.Rr = Rr;
                p.voltage_model_hpf_omega = voltage_model_hpf_omega;
                p.Kp_mras = Kp_mras; p.Ki_mras = Ki_mras;
                p.normalise_eps = normalise_eps;
                p.eps_norm_floor = eps_norm_floor;
                pulsim::observers::add_flux_mras_observer_to_chain(
                    self, p,
                    std::move(v_alpha_channel),
                    std::move(v_beta_channel),
                    std::move(i_alpha_channel),
                    std::move(i_beta_channel),
                    std::move(omega_hat_channel),
                    std::move(psi_adj_alpha_channel),
                    std::move(psi_adj_beta_channel),
                    std::move(psi_ref_alpha_channel),
                    std::move(psi_ref_beta_channel));
            },
            py::arg("Rs"), py::arg("Ls"),
            py::arg("Lr"), py::arg("Lm"), py::arg("Rr") = Real{0.0},
            py::arg("voltage_model_hpf_omega") = Real{5.0},
            py::arg("Kp_mras") = Real{50.0},
            py::arg("Ki_mras") = Real{1e3},
            py::arg("normalise_eps") = true,
            py::arg("eps_norm_floor") = Real{1e-6},
            py::arg("v_alpha_channel"), py::arg("v_beta_channel"),
            py::arg("i_alpha_channel"), py::arg("i_beta_channel"),
            py::arg("omega_hat_channel"),
            py::arg("psi_adj_alpha_channel") = std::string{},
            py::arg("psi_adj_beta_channel")  = std::string{},
            py::arg("psi_ref_alpha_channel") = std::string{},
            py::arg("psi_ref_beta_channel")  = std::string{},
            "IM Flux-MRAS Observer (Phase 2.3 C++ port). Reads "
            "(v_alpha, v_beta, i_alpha, i_beta) channels and writes "
            "omega_hat. Default normalise_eps=True activates the "
            "bootstrap-fixed cross-product.")

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
    // ------------------------------------------------------------------
    // v2.0 Phase 3 — variable-step TR-BDF2 on the sparse MNA kernel
    // (engine='auto'). Returns (SimulationResult, stats_dict); the
    // result's time grid is the ACCEPTED (irregular) one.
    // ------------------------------------------------------------------
    m.def("run_transient_trbdf2",
        [](pwl::PwlStateSpaceCache& cache,
           const topology::Graph& graph,
           const pwl::DevicePool& pool,
           Real t_start, Real t_end, Real rtol, Real atol,
           Real h_init, Real h_max,
           py::object switch_fn_obj,
           solver::BExtraFn b_extra_fn,
           py::object initial_state,
           Size max_event_iterations,
           solver::ShouldContinueFn should_continue,
           Real observer_period,
           py::object observer_obj,
           bool enable_nonlinear_refresh,
           Size max_newton_iterations,
           Real tol_newton_dx, Real tol_newton_res,
           bool enable_newton_line_search, bool enable_newton_lm) {
            solver::TrBdf2Options o;
            o.t_start = t_start;
            o.t_end   = t_end;
            o.rtol    = rtol;
            o.atol    = atol;
            o.h_init  = h_init;
            o.h_max   = h_max;
            if (max_event_iterations > 0) {
                o.max_event_iterations = max_event_iterations;
            }
            o.observer_period = observer_period;
            if (max_newton_iterations > 0) {
                o.max_newton_iterations = max_newton_iterations;
            }
            if (tol_newton_dx > Real{0}) {
                o.tol_newton_dx = tol_newton_dx;
            }
            if (tol_newton_res > Real{0}) {
                o.tol_newton_res = tol_newton_res;
            }
            o.enable_newton_line_search = enable_newton_line_search;
            o.enable_newton_lm = enable_newton_lm;
            pwl::NonlinearRefreshFn nl_refresh{};
            if (enable_nonlinear_refresh) {
                nl_refresh =
                    pwl::make_combined_diode_mosfet_refresh();
            }
            std::optional<Vector> x0;
            if (!initial_state.is_none()) {
                x0 = initial_state.cast<Vector>();
            }
            // Bridge.13 pattern: a native PWM schedule runs the
            // mask function in pure C++ AND hands the stepper its
            // analytic next_edge_after — edge landing then costs
            // zero Python round-trips and no bisection.
            solver::SwitchScheduleFn switch_fn;
            std::function<Real(Real)> next_edge_fn;
            {
                sources::NativePwm2Switch* native = nullptr;
                try {
                    native = switch_fn_obj.cast<
                        sources::NativePwm2Switch*>();
                } catch (const py::cast_error&) {}
                if (native) {
                    switch_fn = [native](Real t) {
                        return (*native)(t);
                    };
                    next_edge_fn = [native](Real t) {
                        return native->next_edge_after(t);
                    };
                } else {
                    switch_fn = switch_fn_obj.cast<
                        solver::SwitchScheduleFn>();
                }
            }
            std::function<void(Real, const Vector&)> observer_fn;
            if (!observer_obj.is_none()) {
                // Re-acquires the GIL per call through pybind's
                // functional caster, like every other Python
                // callback the released-GIL engines invoke.
                observer_fn = observer_obj.cast<
                    std::function<void(Real, const Vector&)>>();
            }
            solver::TrBdf2Stats st;
            SimulationResult res;
            {
                // Release the GIL for the run like every sibling
                // engine binding does: with a native PWM schedule
                // the hot loop makes zero Python calls, and a
                // Python switch_fn/b_extra_fn re-acquires per call
                // through pybind's functional caster. Without this
                // an engine='auto' run froze every other Python
                // thread — including Ctrl-C — for its full
                // wall-clock (measured: a background thread ran at
                // 0.6% of baseline).
                py::gil_scoped_release release;
                res = solver::run_transient_trbdf2(
                    cache, graph, pool, o, switch_fn, b_extra_fn,
                    x0, &st, next_edge_fn, should_continue,
                    observer_fn, nl_refresh);
            }
            py::dict d;
            d["n_accept"]       = st.n_accept;
            d["n_reject"]       = st.n_reject;
            d["n_gate_events"]  = st.n_gate_events;
            d["n_diode_events"] = st.n_diode_events;
            d["n_solves"]       = st.n_solves;
            d["n_chatter_breaks"] = st.n_chatter_breaks;
            d["n_forced_accepts"] = st.n_forced_accepts;
            d["n_ctrl_ticks"]     = st.n_ctrl_ticks;
            d["n_newton_retries"] = st.n_newton_retries;
            return py::make_tuple(std::move(res), std::move(d));
        },
        py::arg("cache"), py::arg("graph"), py::arg("pool"),
        py::arg("t_start"), py::arg("t_end"),
        py::arg("rtol"), py::arg("atol"),
        py::arg("h_init"), py::arg("h_max"),
        py::arg("switch_fn"),
        py::arg("b_extra_fn") = solver::BExtraFn{},
        py::arg("initial_state") = py::none(),
        py::arg("max_event_iterations") = Size{0},
        py::arg("should_continue") = solver::ShouldContinueFn{},
        py::arg("observer_period") = Real{0},
        py::arg("step_observer") = py::none(),
        py::arg("enable_nonlinear_refresh") = false,
        py::arg("max_newton_iterations") = Size{0},
        py::arg("tol_newton_dx") = Real{0},
        py::arg("tol_newton_res") = Real{0},
        py::arg("enable_newton_line_search") = false,
        py::arg("enable_newton_lm") = false,
        "Variable-step TR-BDF2 transient (engine='auto'): "
        "L-stable, 2nd order, LTE-controlled step, gate edges "
        "landed by bisection, diode crossings localized by "
        "Illinois. Returns (SimulationResult, stats dict).");

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
           ShouldContinueFn should_continue,
           std::shared_ptr<streaming::LiveRing> live_ring) {
            pwl::NonlinearRefreshFn nl_refresh{};
            if (enable_nonlinear_refresh) {
                nl_refresh =
                    pwl::make_combined_diode_mosfet_refresh();
            }
            // Build a step_observer that calls the chain DIRECTLY in
            // C++ — no Python roundtrip per step.
            auto step_observer = chain.make_step_observer(chain_dt);
            // Wrap with a LiveRing push when streaming is requested.
            // Both layers are GIL-free in the hot path: chain step is
            // C++ and ``LiveRing::push`` uses only atomics + memcpy.
            StepObserverFn observer_eff = step_observer;
            ShouldContinueFn continue_eff = should_continue;
            if (live_ring) {
                auto ring = live_ring;
                auto chain_obs = step_observer;
                observer_eff = [ring, chain_obs](Real t, const Vector& x) {
                    if (chain_obs) {
                        chain_obs(t, x);
                    }
                    if (ring->should_decimate()) {
                        ring->push(t, x.data(),
                                   static_cast<std::size_t>(x.size()));
                    }
                };
                auto user_cont = should_continue;
                continue_eff = [ring, user_cont]() {
                    if (ring->stopped()) {
                        return false;
                    }
                    return user_cont ? user_cont() : true;
                };
            }
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
                                          observer_eff,
                                          x_init_ptr,
                                          continue_eff);
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
        py::arg("live_ring") = std::shared_ptr<streaming::LiveRing>{},
        "Run transient with a BlockChain as the per-step observer. "
        "The chain's step is invoked directly in C++ each step — "
        "no Python interpreter cost per step. Equivalent to "
        "`run_transient(..., step_observer=chain.make_step_observer(dt))` "
        "but ~10x faster on chains with > 5 blocks.");

    // =========================================================================
    // MMC L0/L1 hotpath — Phase 20.11
    // =========================================================================
    //
    // C++ versions of the inner-loop helpers exposed in
    // ``pulsim/mmc.py``. The Python module dispatches to these when
    // available (always — they are part of the kernel build) and
    // falls back to the pure-Python implementations only if the
    // import fails (e.g., during partial source-tree builds).

    py::enum_<mmc::SubmoduleType>(m, "_MmcSubmoduleType")
        .value("half_bridge", mmc::SubmoduleType::HalfBridge)
        .value("full_bridge", mmc::SubmoduleType::FullBridge)
        .export_values();

    m.def("_cpp_ps_pwm_switching_function",
          [](Real m_ref, Real t, Index n_sm, Real f_carrier,
             const std::string& sm_type) {
              const mmc::SubmoduleType type =
                  (sm_type == "full_bridge")
                      ? mmc::SubmoduleType::FullBridge
                      : mmc::SubmoduleType::HalfBridge;
              return mmc::ps_pwm_switching_function(
                  m_ref, t, n_sm, f_carrier, type);
          },
          py::arg("m_ref"), py::arg("t"), py::arg("n_sm"),
          py::arg("f_carrier"),
          py::arg("sm_type") = "half_bridge",
          "C++ hotpath: PS-PWM switching function — returns the "
          "integer count of SMs to insert at time `t` given the "
          "modulation reference `m_ref`. Equivalent to "
          "`ps_pwm_switching_function` in pulsim/mmc.py.");

    m.def("_cpp_ipd_switching_function",
          [](Real m_ref, Real t, Index n_sm, Real f_carrier,
             const std::string& sm_type) {
              const mmc::SubmoduleType type =
                  (sm_type == "full_bridge")
                      ? mmc::SubmoduleType::FullBridge
                      : mmc::SubmoduleType::HalfBridge;
              return mmc::ipd_switching_function(
                  m_ref, t, n_sm, f_carrier, type);
          },
          py::arg("m_ref"), py::arg("t"), py::arg("n_sm"),
          py::arg("f_carrier"),
          py::arg("sm_type") = "half_bridge",
          "C++ hotpath: IPD (In-Phase Disposition) switching "
          "function. All N carriers share the same phase but sit "
          "at stacked DC offsets, so the output only switches "
          "between two adjacent levels per step. Cleaner spectrum "
          "than PS-PWM at small N.");

    m.def("_cpp_mmc_arm_average_step",
          [](Real v_C, Real m_b, Real i_b, Real dt,
             Real c_arm, Real r_p_inv) {
              const auto res = mmc::mmc_arm_average_step(
                  v_C, m_b, i_b, dt, c_arm, r_p_inv);
              return py::make_tuple(res.v_C_next, res.v_b);
          },
          py::arg("v_C"), py::arg("m_b"), py::arg("i_b"),
          py::arg("dt"), py::arg("c_arm"),
          py::arg("r_p_inv") = Real{0.0},
          "C++ hotpath: L0 forward-Euler step. Returns `(v_C_next, "
          "v_b)`. Pass `r_p_inv = 1.0 / r_p` for the lossy variant; "
          "default 0.0 disables the leak term.");

    m.def("_cpp_mmc_arm_multilevel_step",
          [](Real v_C, Real m_ref, Real i_b, Real dt, Real t,
             Index n_sm, Real c_arm, Real f_carrier,
             const std::string& sm_type, Real r_p_inv,
             const std::string& modulation_scheme) {
              const mmc::SubmoduleType type =
                  (sm_type == "full_bridge")
                      ? mmc::SubmoduleType::FullBridge
                      : mmc::SubmoduleType::HalfBridge;
              const mmc::ModulationScheme sch =
                  (modulation_scheme == "ipd")
                      ? mmc::ModulationScheme::Ipd
                      : mmc::ModulationScheme::PsPwm;
              const auto res = mmc::mmc_arm_multilevel_step(
                  v_C, m_ref, i_b, dt, t, n_sm, c_arm,
                  f_carrier, type, r_p_inv, sch);
              return py::make_tuple(
                  res.v_C_next, res.v_b,
                  static_cast<py::int_>(res.s_b));
          },
          py::arg("v_C"), py::arg("m_ref"), py::arg("i_b"),
          py::arg("dt"), py::arg("t"), py::arg("n_sm"),
          py::arg("c_arm"), py::arg("f_carrier"),
          py::arg("sm_type") = "half_bridge",
          py::arg("r_p_inv") = Real{0.0},
          py::arg("modulation_scheme") = "ps_pwm",
          "C++ hotpath: L1 multilevel forward-Euler step. Returns "
          "`(v_C_next, v_b, s_b)` where `s_b` is the integer "
          "switching count for this step (PS-PWM or IPD).");

    m.def("_cpp_mmc_arm_equivalent_step",
          [](Real v_C,
             py::array_t<std::int8_t, py::array::c_style |
                                          py::array::forcecast> bit_s1,
             py::array_t<std::int8_t, py::array::c_style |
                                          py::array::forcecast> bit_s2,
             py::array_t<Real, py::array::c_style |
                                  py::array::forcecast> in_dead_time_until,
             py::array_t<Real, py::array::c_style |
                                  py::array::forcecast> last_toggle_time,
             Real m_ref, Real i_b, Real dt, Real t,
             Index n_sm, Real c_arm, Real f_carrier,
             Real t_dead, Real t_min, Real r_p_inv,
             const std::string& modulation_scheme,
             const std::string& sm_type) {
              if (bit_s1.size() != n_sm || bit_s2.size() != n_sm ||
                  in_dead_time_until.size() != n_sm ||
                  last_toggle_time.size() != n_sm) {
                  throw std::invalid_argument(
                      "L2 state arrays must all have length n_sm");
              }
              const mmc::ModulationScheme sch =
                  (modulation_scheme == "ipd")
                      ? mmc::ModulationScheme::Ipd
                      : mmc::ModulationScheme::PsPwm;
              const mmc::SubmoduleType type =
                  (sm_type == "full_bridge")
                      ? mmc::SubmoduleType::FullBridge
                      : mmc::SubmoduleType::HalfBridge;
              const auto res = mmc::mmc_arm_equivalent_step(
                  v_C,
                  bit_s1.mutable_data(),
                  bit_s2.mutable_data(),
                  in_dead_time_until.mutable_data(),
                  last_toggle_time.mutable_data(),
                  m_ref, i_b, dt, t, n_sm, c_arm, f_carrier,
                  t_dead, t_min, r_p_inv, sch, type);
              return py::make_tuple(
                  v_C, res.v_b,
                  static_cast<py::int_>(res.s_w),
                  static_cast<py::int_>(res.s_u),
                  static_cast<py::int_>(res.s_n));
          },
          py::arg("v_C"), py::arg("bit_s1"), py::arg("bit_s2"),
          py::arg("in_dead_time_until"), py::arg("last_toggle_time"),
          py::arg("m_ref"), py::arg("i_b"), py::arg("dt"),
          py::arg("t"), py::arg("n_sm"), py::arg("c_arm"),
          py::arg("f_carrier"), py::arg("t_dead"), py::arg("t_min"),
          py::arg("r_p_inv") = Real{0.0},
          py::arg("modulation_scheme") = "ps_pwm",
          py::arg("sm_type") = "half_bridge",
          "C++ hotpath: L2 SM-equivalent forward-Euler step "
          "(dead-time + min-pulse-width). State arrays are mutated "
          "in place. Returns `(v_C_next, v_b, s_w, s_u, s_n)` where "
          "``s_n`` counts negatively-inserted SMs (full-bridge only).");

    m.def("_cpp_mmc_arm_detailed_step",
          [](py::array_t<Real, py::array::c_style |
                                  py::array::forcecast> v_C_per_sm,
             py::array_t<std::int8_t, py::array::c_style |
                                          py::array::forcecast> insertion_mask,
             Real m_ref, Real i_b, Real dt, Real t,
             Index n_sm, Real c_sm, Real f_carrier,
             const std::string& sm_type,
             const std::string& balancing, Real r_p_inv_per_sm,
             const std::string& modulation_scheme) {
              if (v_C_per_sm.size() != n_sm ||
                  insertion_mask.size() != n_sm) {
                  throw std::invalid_argument(
                      "L3 state arrays must have length n_sm");
              }
              const mmc::SubmoduleType type =
                  (sm_type == "full_bridge")
                      ? mmc::SubmoduleType::FullBridge
                      : mmc::SubmoduleType::HalfBridge;
              const mmc::BalancingScheme scheme =
                  (balancing == "none")
                      ? mmc::BalancingScheme::None
                      : mmc::BalancingScheme::SortAndSelect;
              const mmc::ModulationScheme msch =
                  (modulation_scheme == "ipd")
                      ? mmc::ModulationScheme::Ipd
                      : mmc::ModulationScheme::PsPwm;
              const auto res = mmc::mmc_arm_detailed_step(
                  v_C_per_sm.mutable_data(),
                  insertion_mask.mutable_data(),
                  m_ref, i_b, dt, t, n_sm, c_sm, f_carrier,
                  type, scheme, r_p_inv_per_sm, msch);
              return py::make_tuple(
                  res.v_b, static_cast<py::int_>(res.s_b));
          },
          py::arg("v_C_per_sm"), py::arg("insertion_mask"),
          py::arg("m_ref"), py::arg("i_b"), py::arg("dt"),
          py::arg("t"), py::arg("n_sm"), py::arg("c_sm"),
          py::arg("f_carrier"),
          py::arg("sm_type") = "half_bridge",
          py::arg("balancing") = "sort_and_select",
          py::arg("r_p_inv_per_sm") = Real{0.0},
          py::arg("modulation_scheme") = "ps_pwm",
          "C++ hotpath: L3 detailed per-SM forward-Euler step. "
          "`v_C_per_sm` and `insertion_mask` are mutated in place. "
          "Returns `(v_b, s_b)`.");
}

}  // namespace pulsim_kernel_bindings

// =============================================================================
// Python module entry point
// =============================================================================
// The extension is named ``_pulsim`` (matches
// ``pybind11_add_module(_pulsim …)`` in ``python/CMakeLists.txt``); all
// kernel symbols are bound directly on the module (no submodule), which
// is what ``python/pulsim/__init__.py`` imports via ``from ._pulsim import …``.
// Forward decl: DSED scheduler bindings (Bridge.10) — separate TU.
namespace pulsim_dsed_bindings {
    void init_module(pybind11::module_& m);
}

PYBIND11_MODULE(_pulsim, m) {
    m.doc() = "Pulsim — power-electronics simulator. Header-only C++23 "
              "kernel: PWL state-space cache, trapezoidal companion, "
              "Newton refinement, and the high-level CircuitBuilder API.";
    pulsim_kernel_bindings::init_module(m);
    pulsim_dsed_bindings::init_module(m);
}
