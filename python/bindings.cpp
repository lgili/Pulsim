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
#include <pybind11/typing.h>

#include <optional>

#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/analysis/mna_sweep.hpp"
#include "pulsim/pwl/device_readings.hpp"
#include "pulsim/ir/fields.hpp"
#include "pulsim/blockchain/blocks.hpp"
#include "pulsim/blockchain/block_adapters.hpp"
#include "pulsim/blockchain/chain.hpp"
#include "pulsim/blockchain/ported_blocks.hpp"
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
#include "pulsim/thermal/loss_table.hpp"
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
#include "pulsim/pwl/mode_ports.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sim/simulation.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/errors/convergence.hpp"
#include "pulsim/sources/combined_switch_fn.hpp"
#include "pulsim/sources/current_waveform_b_extra.hpp"
#include "pulsim/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/sources/phase_shift_full_bridge_fn.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"
#include "pulsim/sources/phase_control_fn.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/sources/spwm_pair_fn.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/events/gate_source.hpp"   // DutyRegister, CarrierPwm, EdgeList
#include "pulsim/sources/three_phase_spwm_fn.hpp"
#include "pulsim/streaming/live_ring.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "pulsim/yaml/loader.hpp"
#include "switch_fn_adapter.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // for "name"_a kwarg shorthand

namespace pulsim_kernel_bindings {

using pulsim_bindings_common::resolve_b_extra;

/// The static Newton chain for a transient driver. The kernel always
/// builds it when the pool needs one, so `None` and `True` are the same
/// (deprecated); `False` on a circuit with static Newton devices is
/// refused, because it means "simulate them as open circuits".
inline pulsim::pwl::NonlinearRefreshFn transient_refresh(
    const pulsim::topology::Graph& graph,
    const pulsim::pwl::DevicePool& pool, std::optional<bool> enable,
    const char* engine) {
    // One rule for every entry point (add-kernel-simulation-object).
    return pulsim::sim::transient_refresh(graph, pool, enable, engine);
}

void init_module(py::module_& m) {
    m.doc() = "Pulsim kernel — PWL state-space cache, "
              "trapezoidal companion, Newton, and the "
              "high-level CircuitBuilder API.";
    m.attr("__version__") = "2.1.0";

    using namespace pulsim;

    // add-ai-usage-guide 1.1: registered before the members that return
    // them (builder.graph / .pool, kind_of, find_implausible_voltage,
    // final_snapshot), so their signatures name the Python type; the
    // methods are defined further down.
    py::class_<topology::Graph> graph_cls(m, "Graph",
        "Pulsim topology graph. Build via "
        "CircuitBuilder; access via builder.graph.");
    py::class_<pwl::DevicePool> pool_cls(m, "DevicePool",
        "Pulsim device parameter pool. Build via "
        "CircuitBuilder; access via builder.pool.");
    py::enum_<pwl::DevicePool::StoredKind> stored_kind_cls(m, "StoredKind",
        "Device kind stored in DevicePool. v1.4.0+ — exposed so "
        "Python helpers (sweep_path_aware etc.) can auto-dispatch "
        "update_* mutators per device type.");
    py::class_<solver::SimulationResult> result_cls(m, "SimulationResult",
        "Output of run_transient: parallel `times` and "
        "`states` arrays, plus event diagnostics.",
        py::dynamic_attr());  // allow `result._builder = b` from Python
    py::class_<solver::SolverSnapshot> snapshot_cls(m, "SolverSnapshot",
        "The COMPLETE state of a run at one instant.\n\n"
        "`initial_state=` restores only the MNA vector and then "
        "INVENTS the companion history from it (a capacitor gets "
        "i_prev = 0, an inductor v_prev = 0), so resuming from it "
        "does NOT reproduce the run — a continuous 2T RLC and a "
        "T-then-resume differ by 2.3e-4 where a true resume is "
        "~1e-15. A snapshot carries what was missing: the "
        "trapezoidal companion history, the state of every "
        "stateful device (saturable / gapped / hysteretic "
        "inductors, charge-based Coss, Lauritzen diodes, IGBT "
        "tails, MNA PMSMs) and the solver-owned diode bits. Pass "
        "it as simulate(resume_from=...): exact on engine='pwl', "
        "to tolerance on engine='trbdf2'. Pickles.");

    // add-python-builder-ergonomics (v1.5): register the C++
    // `Cancelled` exception with pybind11 so calls into the kernel
    // that throw `analysis::Cancelled` surface as a typed
    // `pulsim._CxxCancelled` Python exception. The user-facing
    // `pulsim.Cancelled` (defined in
    // `python/pulsim/_builder_ergonomics.py`) catches this and
    // re-raises with its own attribute layout.

    // v2.0 Phase 2: a transient that dies part-way brings the run
    // with it. `py::register_exception` can only carry a message, so
    // the payload needs a translator that builds the Python
    // exception object and attaches the partial result to it.
    //
    // The type is created here rather than in Python because the
    // translator has to name it at throw time, and the result object
    // it carries is a pybind-bound C++ type.
    //
    // `ConvergenceError` (add-parameter-sensitivities, task 5a) is the
    // common base of every non-convergence: a RuntimeError, so the
    // `except RuntimeError` of callers keeps working.
    //
    // add-kernel-simulation-object (8.1): every pulsim error is also a
    // `PulsimError`, so one `except pulsim.PulsimError` catches them all
    // while each keeps its standard base (ValueError, RuntimeError, ...).
    static py::object pulsim_exc = py::exception<void>(
        m, "PulsimError", PyExc_Exception);
    const auto pulsim_error_type = [&m](const char* name, PyObject* base) {
        py::object t = py::reinterpret_borrow<py::object>(
            reinterpret_cast<PyObject*>(&PyType_Type))(
            name, py::make_tuple(py::handle(base), pulsim_exc),
            py::dict(py::arg("__module__") = m.attr("__name__")));
        m.attr(name) = t;
        return t;
    };
    static py::object convergence_exc =
        pulsim_error_type("ConvergenceError", PyExc_RuntimeError);
    // The kernel's Cancelled (`pulsim.Cancelled` catches and re-raises it
    // with its own attributes); a PulsimError too.
    static py::object cancelled_exc =
        pulsim_error_type("_CxxCancelled", PyExc_RuntimeError);
    static py::object aborted_exc = py::exception<void>(
        m, "SimulationAborted", convergence_exc);
    // The run's Newton roundoff-acceptance budget ran out: a
    // SimulationAborted (same payload), typed so callers need not parse
    // the message.
    static py::object budget_exc = py::exception<void>(
        m, "NewtonRoundoffBudgetExceeded", aborted_exc.ptr());
    py::register_exception_translator(
        [](std::exception_ptr pe) {
            if (!pe) {
                return;
            }
            try {
                std::rethrow_exception(pe);
            } catch (const solver::SimulationAborted& e) {
                py::object& type =
                    e.roundoff_budget() ? budget_exc : aborted_exc;
                py::object inst = type(e.what());
                // `partial` is a COPY: the C++ exception is destroyed
                // as the translator returns, so a reference would
                // dangle the moment the caller touched it.
                inst.attr("partial") = py::cast(e.partial());
                inst.attr("t_failed") = py::cast(e.t_failed());
                PyErr_SetObject(type.ptr(), inst.ptr());
            } catch (const ConvergenceFailure& e) {
                // The exact engine's give-up sites (settle | chatter |
                // no_progress | max_steps | nan): typed, so no caller
                // has to match message text.
                py::object inst = convergence_exc(e.what());
                inst.attr("kind") = e.kind_name();
                inst.attr("t_failed") = e.time();
                inst.attr("device") = e.device();
                PyErr_SetObject(convergence_exc.ptr(), inst.ptr());
            }
        });

    // The typed refusal (add-engine-conformance-matrix). A ValueError,
    // so `except ValueError` keeps working; the four fields say which
    // engine refused which device or feature.
    static py::object unsupported_exc =
        pulsim_error_type("UnsupportedError", PyExc_ValueError);
    // A RunOptions value no engine can take (sim::OptionsError): a
    // ValueError naming the option.
    static py::object options_exc =
        pulsim_error_type("OptionsError", PyExc_ValueError);
    options_exc.attr("option") = "";
    // An engine broke its own invariant (pulsim::InternalError): a pulsim bug.
    static py::object internal_exc =
        pulsim_error_type("InternalError", PyExc_RuntimeError);
    for (const char* a : {"engine", "feature", "device", "kind"}) {
        unsupported_exc.attr(a) = "";
    }
    py::register_exception_translator(
        [](std::exception_ptr pe) {
            if (!pe) {
                return;
            }
            try {
                std::rethrow_exception(pe);
            } catch (const UnsupportedError& e) {
                py::object inst = unsupported_exc(e.what());
                inst.attr("engine") = e.engine();
                inst.attr("feature") = e.feature();
                inst.attr("device") = e.device();
                inst.attr("kind") = e.kind();
                PyErr_SetObject(unsupported_exc.ptr(), inst.ptr());
            } catch (const analysis::Cancelled& e) {
                PyErr_SetString(cancelled_exc.ptr(), e.what());
            } catch (const sim::OptionsError& e) {
                py::object inst = options_exc(e.what());
                inst.attr("option") = e.option();
                PyErr_SetObject(options_exc.ptr(), inst.ptr());
            } catch (const InternalError& e) {
                PyErr_SetString(internal_exc.ptr(), e.what());
            }
        });

    m.def("engine_capabilities",
        []() {
            py::dict out;
            for (const auto& row : pwl::kKindCaps) {
                py::dict d;
                d["class"] = std::string(pwl::kSolveClassNames[
                    static_cast<std::size_t>(row.cls)]);
                for (std::size_t e = 0; e < pwl::kEngineNames.size(); ++e) {
                    d[py::str(std::string(pwl::kEngineNames[e]))] =
                        row.supports(static_cast<pwl::Engine>(e));
                }
                out[py::str(std::string(row.name))] = d;
            }
            return out;
        },
        "The kernel's device-kind capability table: {kind: {'class': "
        "solve class, 'pwl'|'trbdf2'|'dsed'|'bdf1'|'ac_sweep': bool}}. "
        "An engine marked False refuses the kind with UnsupportedError.");
    m.def("_branch_form_capabilities",
        []() {
            py::dict out;
            for (const auto& row : pwl::kBranchFormCaps) {
                py::dict d;
                for (std::size_t e = 0; e < pwl::kEngineNames.size(); ++e) {
                    d[py::str(std::string(pwl::kEngineNames[e]))] =
                        row.supports(static_cast<pwl::Engine>(e));
                }
                out[py::str(std::string(row.name))] = d;
            }
            return out;
        },
        "The branch form's engine column (add-branch-form-ideal-switch): "
        "{'Switch'|'Diode': {engine: bool}}; False refuses with "
        "UnsupportedError(feature='branch_form').");
    m.def("_stateful_kinds",
        []() {
            py::dict out;
            const pwl::StatefulHistories kinds;
            kinds.for_each([&out](const auto& k) {
                out[py::str(std::string(device::kStateBlocks[
                    static_cast<std::size_t>(k.kBlock)].name))] =
                    std::string(pwl::stored_kind_name(k.kStoredKind));
            });
            return out;
        },
        "The kernel's stateful Newton kinds (pwl::StatefulKinds), in fold "
        "order: {state block name: StoredKind name of the kind-table row "
        "it runs under}.");
    m.def("_device_path_ported",
        []() {
            // add-unified-device-interface DECISIONS #23: which Newton
            // kinds run on device_path='interface', from the registry.
            py::dict out;
            for (std::size_t i = 0; i < pwl::kKindCaps.size(); ++i) {
                const auto sk = static_cast<pwl::DevicePool::StoredKind>(i);
                const auto& c = pwl::kKindCaps[i];
                if (c.cls == pwl::SolveClass::NewtonStatic) {
                    out[py::str(std::string(c.name))] =
                        pwl::iface::static_kind_ported(sk);
                } else if (c.cls == pwl::SolveClass::NewtonStateful) {
                    bool ported = false;
                    const pwl::StatefulHistories kinds;
                    kinds.for_each([&](const auto& k) {
                        using K = std::remove_cvref_t<decltype(k)>;
                        if (pwl::iface::history_covers<K>(sk)) ported = pwl::iface::kPorted<K>;
                    });
                    out[py::str(std::string(c.name))] = ported;
                }
            }
            py::dict blocks;
            const pwl::StatefulHistories kinds;
            kinds.for_each([&](const auto& k) {
                using K = std::remove_cvref_t<decltype(k)>;
                blocks[py::str(std::string(device::kStateBlocks[
                    static_cast<std::size_t>(K::kBlock)].name))] =
                    pwl::iface::kPorted<K>;
            });
            return py::make_tuple(out, blocks);
        },
        "({Newton kind: ported}, {stateful block: ported}): the kinds "
        "simulate(device_path='interface') runs; the others refuse it by "
        "name.");
    m.def("_gated_capabilities",
        []() {
            py::dict out;
            for (const auto& row : pwl::kGatedCaps) {
                py::dict d;
                for (std::size_t e = 0; e < pwl::kEngineNames.size(); ++e) {
                    d[py::str(std::string(pwl::kEngineNames[e]))] =
                        row.supports(static_cast<pwl::Engine>(e));
                }
                out[py::str(std::string(row.name))] = d;
            }
            return out;
        },
        "The gated PWL devices' engine column (add-pwl-igbt-and-"
        "thyristor): {'GateHeld'|'Latching': {engine: bool}}; False "
        "refuses with UnsupportedError(feature='gated_diode').");
    m.def("_lcp_circuit_refusal",
        [](const topology::Graph& graph, const pwl::DevicePool& pool) {
            const auto r = pwl::lcp_circuit_refusal(graph, pool);
            return py::make_tuple(r.reason, r.device);
        },
        "(reason, device): why mode_selection='lcp' refuses this circuit on "
        "pwl — '' (it applies), 'no_diodes', 'gated', 'newton'. Read by "
        "mode_selection='auto'.");
    m.def("_require_supported",
        [](const topology::Graph& graph, const pwl::DevicePool& pool,
           const std::string& engine_in, bool loss_port) {
            // 'exact' is the canonical name of the dsed column (add-exact-propagator-strategy 13.2)
            const std::string engine = engine_in == "exact" ? std::string("dsed") : engine_in;
            for (std::size_t e = 0; e < pwl::kEngineNames.size(); ++e) {
                if (pwl::kEngineNames[e] == engine) {
                    require_supported(graph, pool,
                                      static_cast<pwl::Engine>(e), loss_port);
                    return;
                }
            }
            throw std::invalid_argument("unknown engine '" + engine + "'");
        },
        py::arg("graph"), py::arg("pool"), py::arg("engine"),
        py::arg("loss_port") = false,
        "Raise UnsupportedError for the first branch `engine` cannot "
        "simulate (the capability table's column). loss_port: the caller "
        "is the dsed event-driven run, which honours a loss port.");

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
                 Size i, bool v) {
                  // An index past the width wrote a bit nothing reads.
                  if (i >= self.size()) {
                      throw py::index_error(
                          "SwitchStateMask.set: bit " + std::to_string(i) +
                          " is out of range for a mask of width " +
                          std::to_string(self.size()));
                  }
                  self.set(i, v);
              },
              py::arg("i"), py::arg("v"),
              "Set switch `i` to state `v` (True=ON).")
        .def("get",
              [](const topology::SwitchStateMask& self, Size i) {
                  if (i >= self.size()) {
                      throw py::index_error(
                          "SwitchStateMask.get: bit " + std::to_string(i) +
                          " is out of range for a mask of width " +
                          std::to_string(self.size()));
                  }
                  return self.get(i);
              },
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
        .def_readonly("domain", &pwl::PreflightFinding::domain,
                       "Domain of the subnet (electrical, thermal, "
                       "mechanical); set by Circuit.with_preflight.")
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

    // ---- add-in-kernel-event-losses group 1: datasheet tables ---------
    py::class_<thermal::LossTable3D>(m, "LossTable3D",
        "Switching energy E(v, i, Tj) [J] on a rectilinear grid, read like "
        "pulsim.LossTable: multilinear, linear extrapolation past the "
        "edges, clamped at 0. Build it with LossTable.to_kernel().")
        .def(py::init<std::vector<Real>, std::vector<Real>,
                      std::vector<Real>, std::vector<Real>>(),
              py::arg("v_axis"), py::arg("i_axis"), py::arg("tj_axis"),
              py::arg("energy"),
              "Strictly ascending axes; energy row-major (v, i, Tj).")
        .def("__call__", &thermal::LossTable3D::operator(), py::arg("v"),
              py::arg("i"), py::arg("tj"), "E [J] at one point.")
        .def_property_readonly("v_axis", &thermal::LossTable3D::v_axis)
        .def_property_readonly("i_axis", &thermal::LossTable3D::i_axis)
        .def_property_readonly("tj_axis", &thermal::LossTable3D::tj_axis)
        .def_property_readonly("energy", &thermal::LossTable3D::energy);
    py::class_<thermal::ConductionTable>(m, "ConductionTable",
        "On-state voltage V_on(i, Tj) [V] on a rectilinear grid, read like "
        "pulsim.LossTable (multilinear, extrapolated, clamped at 0).")
        .def(py::init<std::vector<Real>, std::vector<Real>,
                      std::vector<Real>>(),
              py::arg("i_axis"), py::arg("tj_axis"), py::arg("v_on"),
              "Strictly ascending axes; v_on row-major (i, Tj).")
        .def("__call__", &thermal::ConductionTable::operator(), py::arg("i"),
              py::arg("tj"), "V_on [V] at one point.")
        .def_property_readonly("i_axis", &thermal::ConductionTable::i_axis)
        .def_property_readonly("tj_axis", &thermal::ConductionTable::tj_axis)
        .def_property_readonly("v_on", &thermal::ConductionTable::v_on);
    // ---- add-in-kernel-event-losses group 2: records, reading rule ------
    py::class_<thermal::LossEvent>(m, "LossEvent",
        "One commutation of a device with a loss port: t (start of the step "
        "solved under the new state), turn_on, |v| and |i| read by the "
        "reading rule, the junction temperature tj and the table energy e.")
        .def_readonly("t", &thermal::LossEvent::t)
        .def_readonly("turn_on", &thermal::LossEvent::turn_on)
        .def_readonly("v", &thermal::LossEvent::v)
        .def_readonly("i", &thermal::LossEvent::i)
        .def_readonly("tj", &thermal::LossEvent::tj)
        .def_readonly("e", &thermal::LossEvent::e);
    py::class_<thermal::DeviceLossRecord>(m, "DeviceLossRecord",
        "A device's loss record in a result (res.loss_events / "
        "res.device_energy read it).")
        .def_readonly("device", &thermal::DeviceLossRecord::device)
        .def_readonly("node", &thermal::DeviceLossRecord::node)
        .def_readonly("events", &thermal::DeviceLossRecord::events)
        .def_readonly("E_on", &thermal::DeviceLossRecord::E_on)
        .def_readonly("E_off", &thermal::DeviceLossRecord::E_off)
        .def_readonly("E_cond_electrical",
                      &thermal::DeviceLossRecord::E_cond_electrical)
        .def_readonly("E_cond_table", &thermal::DeviceLossRecord::E_cond_table)
        .def_readonly("E_impulse", &thermal::DeviceLossRecord::E_impulse)
        .def_readonly("n_on", &thermal::DeviceLossRecord::n_on)
        .def_readonly("n_off", &thermal::DeviceLossRecord::n_off);
    m.def("_commutation_points",
        [](const std::vector<std::tuple<bool, Real, Real>>& seq, bool gated,
           Real g_on, Real g_off) {
            const thermal::EdgeLaw law{gated, g_on, g_off};
            // One reading rule for the transient tracker and the periodic
            // orbit (_period_kpis.commutations): thermal::commutation_point.
            std::vector<std::tuple<Real, Real, bool>> on, off;
            for (std::size_t k = 1; k < seq.size(); ++k) {
                const auto& [c0, v0, i0] = seq[k - 1];
                const auto& [c1, v1, i1] = seq[k];
                if (c0 == c1) continue;
                const auto pt = thermal::commutation_point(c1, v0, i0, v1, i1, law, false);
                (c1 ? on : off).emplace_back(pt.v, pt.i, pt.soft);
            }
            return py::make_tuple(on, off);
        },
        py::arg("seq"), py::arg("gated"), py::arg("g_on"), py::arg("g_off"),
        "(on, off): the (|v|, |i|, soft) of every turn-on and turn-off of an "
        "edge sequence [(closed, v, i), ...] by the kernel's reading rule "
        "(turn-on: v before, i after; turn-off: the reverse). With `gated` (a "
        "switch, not a diode) an edge whose open-side v and closed-side i "
        "have opposite signs is soft: v = 0, no energy -- unless that current is "
        "within (g_off + eps g_on)|v|, the open device's own leakage and the "
        "closed law's rounding (thermal::commutation_point).");

    // ---- add-power-mosfet-datasheet-device: charge laws ----------------
    py::class_<models::FluxMapDQ, std::shared_ptr<models::FluxMapDQ>>(m, "FluxMapDQ",
        "Saturating, cross-coupled PMSM dq flux map as the gradient of ONE "
        "co-energy W'(i_d, i_q), a C2 bicubic spline: lambda_d = dW'/di_d, "
        "lambda_q = dW'/di_q, incremental inductances = the Hessian "
        "(symmetric: reciprocal by construction). Build it from flux tables "
        "with `FluxMapDQ.from_tables` (Python, least-squares fit, refuses a "
        "non-reciprocal or non-positive-definite table by name); this "
        "constructor takes the fitted cells: knots_d, knots_q and 16 "
        "power-basis coefficients per cell, W' = sum c[4m+n] u^m v^n with "
        "u = i_d - knots_d[a], v = i_q - knots_q[b], cell (a, b) at "
        "((a*(len(knots_q)-1)) + b)*16. Past the grid W' continues as "
        "its second-order Taylor extension (smooth); the law is defined on "
        "the grid plus a band of 0.1 x each axis span, positive definite "
        "there (checked), and an accepted state outside it is refused by "
        "name. Pass it as `flux_map=` to "
        "`CircuitBuilder.add_pmsm_mna`.")
        .def(py::init([](std::vector<Real> kd, std::vector<Real> kq,
                          std::vector<Real> coef, std::string source,
                          Real fit_residual) {
                  auto f = std::make_shared<models::FluxMapDQ>();
                  f->kd = std::move(kd);
                  f->kq = std::move(kq);
                  f->coef = std::move(coef);
                  f->source = std::move(source);
                  f->fit_residual = fit_residual;
                  f->validate();
                  return f;
              }),
              py::arg("knots_d"), py::arg("knots_q"), py::arg("coef"),
              py::arg("source"), py::arg("fit_residual") = 0.0)
        .def_readonly("knots_d", &models::FluxMapDQ::kd)
        .def_readonly("knots_q", &models::FluxMapDQ::kq)
        .def_readonly("coef", &models::FluxMapDQ::coef)
        .def_readonly("source", &models::FluxMapDQ::source)
        .def_readonly("fit_residual", &models::FluxMapDQ::fit_residual)
        .def("evaluate",
             [](const models::FluxMapDQ& f, Real i_d, Real i_q) {
                 const auto e = f.eval(i_d, i_q);
                 return py::make_tuple(e.W, e.gd, e.gq, e.hdd, e.hdq, e.hqq);
             },
             py::arg("i_d"), py::arg("i_q"),
             "(W', lambda_d, lambda_q, L_dd, L_dq, L_qq) at (i_d, i_q): the "
             "kernel's own evaluation.")
        .def("increment",
             [](const models::FluxMapDQ& f, Real i_d, Real i_q, Real d_d,
                Real d_q) {
                 const auto g = f.grad_increment(i_d, i_q, d_d, d_q);
                 return py::make_tuple(g[0], g[1]);
             },
             py::arg("i_d"), py::arg("i_q"), py::arg("d_d"), py::arg("d_q"),
             "lambda(i + d) - lambda(i), in the increment form the flux row "
             "uses (round-off proportional to |d|).");

    py::class_<models::ChargeTable>(m, "ChargeTable",
        "Monotone charge law Q(v) for a nonlinear capacitor: knots "
        "(v, Q, C), cubic Hermite with C as exact tangents, so C(v) is the "
        "exact derivative of Q(v). Not odd; linear tails at the end-knot "
        "slopes; one knot must sit at v = 0 with Q = 0. Fritsch-Carlson "
        "monotonicity is checked and a violating tangent clamped.")
        .def(py::init([](std::vector<Real> v, std::vector<Real> q,
                          std::vector<Real> c) {
                  return models::ChargeTable(std::move(v), std::move(q),
                                             std::move(c));
              }),
              py::arg("v"), py::arg("Q"), py::arg("C"),
              "From triples: voltages [V], charges [C], capacitances [F].")
        .def_static("from_capacitance",
              [](const std::vector<Real>& v, const std::vector<Real>& c) {
                  return models::ChargeTable::from_capacitance(v, c);
              },
              py::arg("v"), py::arg("C"),
              "From (v, C) pairs, e.g. a digitised datasheet curve: Q is the "
              "exact integral of the monotone cubic interpolant of C. Use "
              "pulsim.datasheet.fit_charge_table to get a table with a "
              "checked residual.")
        .def("charge", &models::ChargeTable::charge, py::arg("v"),
              "Q(v) [C].")
        .def("capacitance", &models::ChargeTable::capacitance, py::arg("v"),
              "C(v) = dQ/dv [F], exact derivative of charge().")
        .def_property_readonly("v_knots", &models::ChargeTable::v_knots)
        .def_property_readonly("q_knots", &models::ChargeTable::q_knots)
        .def_property_readonly("c_knots", &models::ChargeTable::c_knots,
              "Knot tangents after Fritsch-Carlson clamping.")
        .def_property_readonly("clamped_tangents",
              &models::ChargeTable::clamped_tangents);

    py::class_<models::NonlinearCapacitor::Params>(m, "ChargeLaw",
        "A nonlinear capacitor's charge law: junction C0/(1+v/V0)^m, a "
        "ChargeTable, or the SPICE VDMOS gate-drain law. Build one with "
        "ChargeLaw.junction / .table / .vdmos_cgd and pass it to "
        "add_nonlinear_capacitor or as add_power_mosfet's C_gd / C_ds.")
        .def_static("junction",
              &models::NonlinearCapacitor::junction,
              py::arg("C0"), py::arg("V0"), py::arg("m") = 0.5,
              py::arg("v_floor") = -0.9,
              "C(v) = C0 / (1 + v/V0)^m (C0 [F], V0 [V]).")
        .def_static("table",
              [](const models::ChargeTable& t) {
                  return models::NonlinearCapacitor::from_table(
                      std::make_shared<const models::ChargeTable>(t));
              },
              py::arg("table"), "Charge law from a ChargeTable.")
        .def_static("vdmos_cgd",
              &models::NonlinearCapacitor::vdmos_cgd,
              py::arg("cgd_max"), py::arg("cgd_min"), py::arg("a"),
              "ngspice VDMOS C_gd(v_gd): s = (CGDMAX-CGDMIN)/(1+pi/2), "
              "C = CGDMAX - s + s*tanh(a*u) for u > 0, "
              "CGDMAX - s + s*atan(a*u) for u <= 0.")
        .def_static("tanh",
              [](Real c_max, Real c_min, Real v0) {
                  auto p = models::NonlinearCapacitor::tanh_law(c_max, c_min, v0);
                  models::NonlinearCapacitor::validate(p);
                  return p;
              },
              py::arg("c_max"), py::arg("c_min"), py::arg("v0"),
              "Odd saturation law Q = c_min*v + (c_max - c_min)*v0*tanh(v/v0); "
              "C(0) = c_max, C -> c_min. The law of add_saturable_permeance.")
        .def_property_readonly("kind",
              [](const models::NonlinearCapacitor::Params& p) {
                  return std::string{models::NonlinearCapacitor::law_name(p)};
              }, "'junction', 'table', 'vdmos_cgd' or 'tanh'.")
        .def("charge",
              [](const models::NonlinearCapacitor::Params& p, Real v) {
                  return models::NonlinearCapacitor::charge(p, v);
              }, py::arg("v"), "Q(v) [C].")
        .def("capacitance",
              [](const models::NonlinearCapacitor::Params& p, Real v) {
                  return models::NonlinearCapacitor::capacitance(p, v);
              }, py::arg("v"), "C(v) = dQ/dv [F].");

    py::class_<builder::CircuitBuilder>(m, "CircuitBuilder",
        "High-level v2 circuit constructor. Hides the "
        "two-object Graph + DevicePool setup; users pass "
        "string node names and SI-unit parameter values.",
        py::dynamic_attr())  // allow Python-side metadata
                              // (initial conditions, aliases)
        .def(py::init<>())
        .def("node", py::overload_cast<std::string_view>(&builder::CircuitBuilder::node),
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
        .def("add_inductor_coupling",
              &builder::CircuitBuilder::add_inductor_coupling,
              py::arg("name_a"), py::arg("name_b"), py::arg("k"),
              py::return_value_policy::reference,
              "Magnetically couple two existing linear inductors by "
              "name with coefficient k in [0, 1], as add_transformer "
              "does for the pair it creates.")
        .def("add_hysteretic_core_inductor",
              [](builder::CircuitBuilder& b, const std::string& name,
                 const std::string& from, const std::string& to,
                 double N, double Ae, double le, double lg,
                 double Ms, double a, double alpha, double c, double k,
                 int substeps_min, double M0) -> builder::CircuitBuilder& {
                  models::JilesAthertonParams ja;
                  ja.Ms = Ms; ja.a = a; ja.alpha = alpha; ja.c = c; ja.k = k;
                  return b.add_hysteretic_core_inductor(
                      name, from, to, N, Ae, le, lg, ja, substeps_min, M0);
              },
              py::arg("name"), py::arg("from_node"), py::arg("to_node"),
              py::arg("N"), py::arg("Ae"), py::arg("le"), py::arg("lg"),
              py::arg("Ms"), py::arg("a"), py::arg("alpha"), py::arg("c"),
              py::arg("k"), py::arg("substeps_min") = 8, py::arg("M0") = 0.0,
              py::return_value_policy::reference,
              "Jiles-Atherton hysteretic core INSIDE the Newton loop: N "
              "turns on a core (Ae [m^2], le [m], TOTAL gap lg [m]) of a "
              "material (Ms, a, alpha, c, k). The magnetisation is solved "
              "at the same time level as the current with its exact "
              "tangent in the Jacobian -- no dummy source, no observer, no "
              "one-step lag. i(name) is the winding current; replay the "
              "B-H trajectory from it with pulsim.compute_bh_loop.")
        .def("add_hysteretic_permeance",
              [](builder::CircuitBuilder& b, const std::string& name,
                 const std::string& m_a, const std::string& m_b,
                 double area, double length,
                 double Ms, double a, double alpha, double c, double k,
                 int substeps_min, double M0) -> builder::CircuitBuilder& {
                  models::JilesAthertonParams ja;
                  ja.Ms = Ms; ja.a = a; ja.alpha = alpha; ja.c = c; ja.k = k;
                  return b.add_hysteretic_permeance(
                      name, m_a, m_b, area, length, ja, substeps_min, M0);
              },
              py::arg("name"), py::arg("m_a"), py::arg("m_b"),
              py::arg("area"), py::arg("length"),
              py::arg("Ms"), py::arg("a"), py::arg("alpha"), py::arg("c"),
              py::arg("k"), py::arg("substeps_min") = 8, py::arg("M0") = 0.0,
              py::return_value_policy::reference,
              "Hysteretic permeance: a core segment (area [m^2], length [m]) "
              "with the Jiles-Atherton law (Ms, a, alpha, c, k) between "
              "magnetic nodes m_a, m_b -- the dual of a one-turn "
              "add_hysteretic_core_inductor: a unit winding '<name>.w' on "
              "the internal node '<name>.e' and the JA inductor '<name>' "
              "(N = 1, no gap). The MMF F(m_a) - F(m_b) is i(name) (H = "
              "F/length), the flux rate is v('<name>.e'). Newton engines "
              "(pwl, trbdf2) only.")
        .def("add_multi_winding_transformer",
              [](builder::CircuitBuilder& b, const std::string& name,
                 const std::vector<std::tuple<std::string, std::string, double>>& windings,
                 const std::vector<std::vector<double>>& k_matrix)
                  -> builder::CircuitBuilder& {
                  std::vector<builder::CircuitBuilder::WindingSpec> ws;
                  for (const auto& [f, t, L] : windings) {
                      ws.push_back({f, t, L});
                  }
                  std::vector<std::vector<Real>> km;
                  for (const auto& row : k_matrix) km.emplace_back(row.begin(), row.end());
                  return b.add_multi_winding_transformer(name, ws, km);
              },
              py::arg("name"), py::arg("windings"),
              py::arg("k_matrix") = std::vector<std::vector<double>>{},
              py::return_value_policy::reference,
              "N-winding LINEAR coupled-inductor transformer, no ceiling: "
              "windings = [(from, to, L), ...], k_matrix = N x N couplings "
              "in [0, 1] (upper triangle read; empty = 1 everywhere). "
              "Couplings that are not realisable (inductance matrix not "
              "positive definite) are refused by name. Branch k is named "
              "name + '.w' + k.")
        .def("add_saturable_transformer_n",
              [](builder::CircuitBuilder& b, const std::string& name,
                 const std::string& p_from, const std::string& p_to,
                 double N_p, double L_leak_p,
                 const std::vector<std::tuple<std::string, std::string, double, double>>& secondaries,
                 double Ae, double le, double lg, double mu_r0, double B_sat)
                  -> builder::CircuitBuilder& {
                  std::vector<builder::CircuitBuilder::SecondaryWinding> sw;
                  for (const auto& [f, t, N, Ll] : secondaries) {
                      sw.push_back({f, t, N, Ll});
                  }
                  return b.add_saturable_transformer_n(
                      name, p_from, p_to, N_p, L_leak_p, sw, Ae, le, lg, mu_r0, B_sat);
              },
              py::arg("name"), py::arg("p_from"), py::arg("p_to"),
              py::arg("N_p"), py::arg("L_leak_p"), py::arg("secondaries"),
              py::arg("Ae"), py::arg("le"), py::arg("lg"),
              py::arg("mu_r0") = 2000.0, py::arg("B_sat") = 0.35,
              py::return_value_policy::reference,
              "Saturable transformer with ANY number of secondaries: "
              "secondaries = [(s_from, s_to, N_s, L_leak_s), ...]. One "
              "gapped-core magnetising branch (name + '.m') on the primary "
              "and one ideal transformer per secondary (name + '.s' + k, "
              "n_k = N_k/N_p). No winding ceiling.")
        .def("add_saturable_transformer",
              &builder::CircuitBuilder::add_saturable_transformer,
              py::arg("name"),
              py::arg("p_from"), py::arg("p_to"),
              py::arg("s_from"), py::arg("s_to"),
              py::arg("N_p"), py::arg("N_s"),
              py::arg("Ae"), py::arg("le"), py::arg("lg"),
              py::arg("mu_r0") = 2000.0, py::arg("B_sat") = 0.35,
              py::arg("L_leak_p") = 0.0, py::arg("L_leak_s") = 0.0,
              py::return_value_policy::reference,
              "Saturable transformer on a gapped core: per-winding "
              "leakage + ideal n = N_s/N_p + ONE magnetising flux "
              "branch lambda(i) from geometry (N_p turns, Ae, le, TOTAL "
              "gap lg, mu_r0, B_sat = mu0*Ms), referred to the primary. "
              "i(name) is the secondary current, i(name + '.m') the "
              "magnetising current. Below saturation it equals "
              "add_transformer with L_p = L_leak_p + L_m, M = n*L_m, "
              "L_s = n^2*L_m + L_leak_s.")
        .def("add_ideal_transformer",
              &builder::CircuitBuilder::add_ideal_transformer,
              py::arg("name"),
              py::arg("p_from"), py::arg("p_to"),
              py::arg("s_from"), py::arg("s_to"),
              py::arg("n"),
              py::return_value_policy::reference,
              "Add an IDEAL transformer: v_s = n*v_p, i_p = -n*i_s, "
              "n = N_s/N_p. The secondary is the branch (i(name) is "
              "its current); the primary is a node pair receiving the "
              "reflected current. It transforms DC -- add a "
              "magnetising inductance across the primary if the real "
              "device has one.")
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
        // add-controlled-source-set
        .def("add_vccs", &builder::CircuitBuilder::add_vccs,
              py::arg("name"), py::arg("in_pos"), py::arg("in_neg"),
              py::arg("n_pos"), py::arg("n_neg"), py::arg("gm"),
              py::return_value_policy::reference,
              "Voltage-controlled current source: gm*(V(in_pos) - "
              "V(in_neg)) [S] flows OUT of `n_pos` into the circuit "
              "(the add_current_source convention).")
        .def("add_ccvs", &builder::CircuitBuilder::add_ccvs,
              py::arg("name"), py::arg("control"), py::arg("out_pos"),
              py::arg("out_neg"), py::arg("rm"),
              py::return_value_policy::reference,
              "Current-controlled voltage source: V(out_pos) - V(out_neg) "
              "= rm * i(control) [ohm]. `control` names a branch added "
              "earlier that carries a current unknown (a voltage source "
              "of any kind -- a 0 V one is the sense element --, an "
              "inductor, a VCVS, a CCVS, an ideal-transformer secondary).")
        .def("add_cccs", &builder::CircuitBuilder::add_cccs,
              py::arg("name"), py::arg("control"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("gain"),
              py::return_value_policy::reference,
              "Current-controlled current source: gain * i(control) flows "
              "OUT of `n_pos`. `control` as for add_ccvs.")
        .def("add_gyrator", &builder::CircuitBuilder::add_gyrator,
              py::arg("name"), py::arg("p1_pos"), py::arg("p1_neg"),
              py::arg("p2_pos"), py::arg("p2_neg"), py::arg("g"),
              py::return_value_policy::reference,
              "Gyrator: i1 = g*v2, i2 = -g*v1 [S], currents INTO p1_pos "
              "and p2_pos. Lossless; an impedance Z on one port appears "
              "as 1/(g^2 Z) on the other (a capacitor C as an inductor "
              "C/g^2). res.i(name, port=1|2).")
        // ---- add-permeance-magnetic-domain ---------------------------
        .def("add_winding", &builder::CircuitBuilder::add_winding,
              py::arg("name"), py::arg("e_pos"), py::arg("e_neg"),
              py::arg("m_pos"), py::arg("m_neg"), py::arg("turns"),
              py::return_value_policy::reference,
              "Winding of `turns` turns between an electrical port (e_pos, "
              "e_neg) and a magnetic port (m_pos, m_neg): v = N*dPhi/dt, "
              "F(m_pos) - F(m_neg) = N*i, with i INTO e_pos and the flux "
              "rate leaving m_pos. A gyrator with g = 1/N (lossless). "
              "res.i(name, port=1) is the winding current.")
        .def("add_permeance", &builder::CircuitBuilder::add_permeance,
              py::arg("name"), py::arg("m_a"), py::arg("m_b"), py::arg("P"),
              py::return_value_policy::reference,
              "Linear permeance P [H] between magnetic nodes (MMF [A] is "
              "the node potential, flux rate [V] the through variable): "
              "Phi = P*F. Every magnetic island needs one node tied to gnd.")
        .def("add_air_gap", &builder::CircuitBuilder::add_air_gap,
              py::arg("name"), py::arg("m_a"), py::arg("m_b"),
              py::arg("area"), py::arg("length"),
              py::return_value_policy::reference,
              "Air gap: permeance mu0*area/length (area [m^2], length [m]; "
              "no fringing).")
        .def("add_transmission_line",
              &builder::CircuitBuilder::add_transmission_line,
              py::arg("name"), py::arg("a"), py::arg("a_ref"),
              py::arg("b"), py::arg("b_ref"), py::arg("Z0"),
              py::arg("delay"), py::arg("R") = 0.0,
              py::return_value_policy::reference,
              "Bergeron transmission line (openspec add-realistic-passives "
              "D18): surge impedance Z0 [ohm], one-way delay [s], between "
              "port a (a, a_ref) and port b (b, b_ref). Branches 'name' "
              "(port a) and 'name.b' (port b) carry the current INTO the "
              "line. R > 0 [ohm, total, R <= Z0]: lumped R/4 - R/2 - R/4 "
              "around two half lines. pwl (dt <= delay) and trbdf2 (step "
              "<= delay); the AC sweep stamps the exact two-port; a wire "
              "at DC. Refused by name on the exact engine, bdf1 and "
              "integrator='esdirk43'.")
        .def("add_excess_loss", &builder::CircuitBuilder::add_excess_loss,
              py::arg("name"), py::arg("m_a"), py::arg("m_b"), py::arg("c"),
              py::arg("eps"),
              py::return_value_policy::reference,
              "Excess (anomalous) core loss between magnetic nodes m_a, m_b: "
              "F = c*u/(u^2 + eps^2)^0.25 for the flux rate u [V] through it "
              "(m_a -> m_b), Bertotti's square-root law c*sign(u)*|u|^0.5 "
              "(c = k_x*l/sqrt(A) [A/V^0.5]) made linear below |u| ~ eps [V]. "
              "Sine loss within (3/4)*(eps/U)^2 of c*U^1.5*0.55641 at flux-rate "
              "amplitude U: eps <= 1.15e-3*U for 1e-6. A unit winding "
              "'<name>.w' on the internal node '<name>.e' (its voltage is u) "
              "and the ExcessLoss conductance '<name>' (i(name) is F). Newton "
              "engines (pwl, trbdf2) and the AC sweep; the exact engine "
              "refuses it by name.")
        .def("add_magnetic_resistance",
              &builder::CircuitBuilder::add_magnetic_resistance,
              py::arg("name"), py::arg("m_a"), py::arg("m_b"), py::arg("R_m"),
              py::return_value_policy::reference,
              "Magnetic resistance R_m [S = A/V]: F = R_m*dPhi/dt, loss "
              "R_m*(dPhi/dt)^2 inside the solve (classical eddy loss; a "
              "lamination: R_m = sigma*d^2*l/(12*A)). Seen from a winding of "
              "N turns it is a resistance N^2/R_m across the winding.")
        .def("add_saturable_permeance",
              [](builder::CircuitBuilder& self, std::string_view name,
                 std::string_view m_a, std::string_view m_b,
                 std::optional<Real> area, std::optional<Real> length,
                 std::optional<Real> mu_r, std::optional<Real> B_sat,
                 const std::optional<models::ChargeTable>& table)
                  -> builder::CircuitBuilder& {
                  if (table) {
                      if (mu_r || B_sat) {
                          throw std::invalid_argument(std::format(
                              "add_saturable_permeance(\"{}\"): pass either "
                              "table= or mu_r and B_sat, not both.", name));
                      }
                      return self.add_permeance_law(
                          name, m_a, m_b,
                          models::NonlinearCapacitor::from_table(
                              std::make_shared<const models::ChargeTable>(*table)),
                          area.value_or(Real{0}), length.value_or(Real{0}));
                  }
                  if (!area || !length || !mu_r || !B_sat) {
                      throw std::invalid_argument(std::format(
                          "add_saturable_permeance(\"{}\"): needs area, length, "
                          "mu_r and B_sat (or table=).", name));
                  }
                  return self.add_saturable_permeance(name, m_a, m_b, *area,
                                                      *length, *mu_r, *B_sat);
              },
              py::arg("name"), py::arg("m_a"), py::arg("m_b"),
              py::arg("area") = py::none(), py::arg("length") = py::none(),
              py::kw_only(), py::arg("mu_r") = py::none(),
              py::arg("B_sat") = py::none(), py::arg("table") = py::none(),
              py::return_value_policy::reference,
              "Saturable core segment (area [m^2], length [m]) with the "
              "GappedCore material law B = mu0*(H + Ms*tanh(H/H0)), "
              "Ms = B_sat/mu0, H0 = Ms/(mu_r - 1): Phi(F) = mu0*A*[F/l + "
              "Ms*tanh(F/(l*H0))], stamped in flux form. Or table= a "
              "ChargeTable with knots (F, Phi, P) covering both signs. "
              "Newton engines (pwl, trbdf2) only.")
        .def("magnetic_element",
              [](const builder::CircuitBuilder& self, std::string_view name) {
                  using K = pwl::DevicePool::StoredKind;
                  const auto& g = self.graph();
                  const auto& pool = self.pool();
                  const Index bid = self.branch_id_of(name);
                  const auto& br = g.branch(bid);
                  auto label = [&](Index n) {
                      return n == topology::Graph::ground()
                                 ? std::string{"gnd"} : g.node(n).name;
                  };
                  const bool mag = pool.is_flux_node(br.from) || pool.is_flux_node(br.to);
                  py::dict d;
                  const auto k = pool.kind_of(bid);
                  if (mag && k == K::Capacitor) {
                      d["kind"] = "permeance";
                      d["P"] = pool.capacitor_params(bid).C;
                  } else if (mag && k == K::NonlinearCapacitor) {
                      d["kind"] = "saturable_permeance";
                      d["law"] = pool.nonlinear_capacitor_params(bid);
                  } else if (mag && k == K::Resistor) {
                      d["kind"] = "magnetic_resistance";
                      d["R_m"] = Real{1} / pool.resistor_params(bid).G;
                  } else if (mag && k == K::Gyrator) {
                      d["kind"] = "winding";
                      d["turns"] = Real{1} / pool.gyrator_params(bid).g;
                  } else if (k == K::SaturableInductor &&
                             br.from != topology::Graph::ground() &&
                             g.node(br.from).name == std::string{name} + ".e") {
                      // add_hysteretic_permeance: Φ depends on the JA state,
                      // not on F alone
                      throw py::key_error(std::format(
                          "'{}' is a hysteretic permeance: its flux depends on "
                          "the Jiles-Atherton state, so flux / b_field / "
                          "h_field are not derived from F. Read the MMF as "
                          "res.i('{}') and the flux rate as res.v('{}.e')",
                          name, name, name));
                  } else {
                      throw py::key_error(std::format(
                          "'{}' is not a magnetic element (winding, permeance, "
                          "air gap, saturable permeance or magnetic resistance)",
                          name));
                  }
                  d["m_a"] = label(br.from);
                  d["m_b"] = label(br.to);
                  d["area"] = py::none();
                  d["length"] = py::none();
                  for (const auto& mg : pool.magnetic_geometry()) {
                      if (mg.branch == bid) { d["area"] = mg.area; d["length"] = mg.length; }
                  }
                  return d;
              },
              py::arg("name"),
              "Description of a magnetic element: kind ('winding', "
              "'permeance', 'saturable_permeance', 'magnetic_resistance'), "
              "its magnetic nodes m_a, m_b, its value (P, law, R_m or turns) "
              "and area / length (None without geometry). KeyError otherwise.")
        .def("record_composite",
              [](builder::CircuitBuilder& b, const std::string& kind,
                 const std::string& name, const std::string& anchor,
                 std::vector<std::string> members) -> builder::CircuitBuilder& {
                  return b.record_composite(kind, name, anchor, std::move(members));
              },
              py::arg("kind"), py::arg("name"), py::arg("anchor"),
              py::arg("members"), py::return_value_policy::reference,
              "Record a composite built from existing kinds by an authoring "
              "helper (add_dc_machine does): components() then gives its "
              "anchor member's entry the keys composite, composite_name, "
              "sub_elements (and composite_params for a dc_machine). "
              "Metadata only; to_yaml / from_yaml keep it.")
        .def("composites",
              [](const builder::CircuitBuilder& b) {
                  py::list out;
                  for (const auto& c : b.composites()) {
                      py::dict d;
                      d["kind"] = c.kind; d["name"] = c.name;
                      d["anchor"] = c.anchor; d["members"] = c.members;
                      out.append(d);
                  }
                  return out;
              },
              "The recorded composites: [{kind, name, anchor, members}].")
        .def("magnetic_nodes",
              [](const builder::CircuitBuilder& self) {
                  std::vector<std::string> out;
                  for (const Index n : self.pool().flux_nodes()) {
                      out.push_back(self.graph().node(n).name);
                  }
                  return out;
              },
              "Names of the magnetic nodes (touched by a magnetic element).")
        .def("add_sine_current_source",
              &builder::CircuitBuilder::add_sine_current_source,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("i_dc"), py::arg("i_amplitude"), py::arg("frequency"),
              py::arg("phase") = 0.0,
              py::return_value_policy::reference,
              "Sine current source: i_dc + i_amplitude*sin(2*pi*f*t + "
              "phase) OUT of `n_pos`; `phase` in radians. Evaluated in "
              "the kernel (no b_extra_fn).")
        .def("add_pulse_current_source",
              &builder::CircuitBuilder::add_pulse_current_source,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("i_initial"), py::arg("i_pulsed"), py::arg("t_start"),
              py::arg("pulse_width"), py::arg("period") = 0.0,
              py::arg("rise_time") = 0.0, py::arg("fall_time") = 0.0,
              py::return_value_policy::reference,
              "Pulse current source OUT of `n_pos`: the trapezoid of "
              "add_pulse_voltage_source, same parameters and edge rules.")
        .def("add_pwl_current_source",
              &builder::CircuitBuilder::add_pwl_current_source,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("t"), py::arg("i"), py::arg("periodic") = false,
              py::return_value_policy::reference,
              "Piecewise-linear table current OUT of `n_pos`: linear "
              "between the knots (t[k], i[k]), i[0] before t[0], then "
              "i[-1] -- or the table repeated with period t[-1] - t[0] "
              "when `periodic` (which needs i[0] == i[-1]). Refuses "
              "fewer than two knots, non-increasing t and non-finite "
              "values.")
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
        .def("add_shockley_diode",
              &builder::CircuitBuilder::add_shockley_diode,
              py::arg("name"),
              py::arg("anode"), py::arg("cathode"),
              py::arg("I_S")   = 1e-12,
              py::arg("n")     = 1.0,
              py::arg("V_T")   = 0.025852,
              py::arg("G_min") = 1e-12,
              py::arg("BV")    = 0.0,
              py::return_value_policy::reference,
              "Add an EXPONENTIAL (Shockley) diode: "
              "i = I_S*(exp(v/(n*V_T)) - 1) + v*G_min. Unlike "
              "`add_diode` (binary PWL) and the smooth-blend "
              "nonlinear diode, the forward drop is not fixed: a "
              "real junction's V_F rises ~60 mV per decade of "
              "current, so the same device drops 0.53 V at 1 mA "
              "and 0.77 V at 10 A. Use "
              "`thermal_voltage(T_kelvin)` for a junction "
              "temperature other than ~300 K. `BV` is a POSITIVE "
              "breakdown magnitude (0 disables it) — give 5.1 "
              "for a 5.1 V Zener. No series-resistance "
              "parameter on purpose: put a resistor in series, "
              "which is exact.")
        .def("add_lauritzen_diode",
              &builder::CircuitBuilder::add_lauritzen_diode,
              py::arg("name"),
              py::arg("anode"), py::arg("cathode"),
              py::arg("I_S")   = 1e-12,
              py::arg("n")     = 1.0,
              py::arg("V_T")   = 0.025852,
              py::arg("tau")   = 1e-7,
              py::arg("T_M")   = 1e-8,
              py::arg("G_min") = 1e-12,
              py::return_value_policy::reference,
              "Add a Lauritzen-Mattsson diode: an exponential "
              "junction that also STORES CHARGE, so it RECOVERS. "
              "Every other diode here is static I-V, and a static "
              "law cannot recover, because recovery is stored "
              "charge leaving the device — on a double-pulse test "
              "(400 V, 20 A, 50 nH loop) `add_diode` and "
              "`add_shockley_diode` both commutate 20 A straight "
              "to zero with a reverse peak of exactly 0 A, while "
              "a real fast-recovery Si part peaks 15-30 A "
              "NEGATIVE through the turning-on switch, where it "
              "usually dominates turn-on loss. `tau` (carrier "
              "lifetime) sets HOW LONG recovery lasts — 10-100 ns "
              "for fast-recovery Si, microseconds for a standard "
              "rectifier; `T_M` (transit time) sets HOW HARD the "
              "reverse peak is and must stay well below `tau`. "
              "The DC curve is unchanged: in steady state this is "
              "an ordinary Shockley junction with I_S scaled by "
              "tau/(tau + T_M). A Schottky stores no charge at "
              "all — use `add_shockley_diode` for one rather than "
              "driving `tau` toward zero here.")
        .def("add_pmsm_mna",
              [](builder::CircuitBuilder& b, std::string_view name,
                 std::string_view pa, std::string_view pb, std::string_view pc,
                 std::string_view n, std::string_view om, std::string_view th,
                 Real R_s, Real L_d, Real L_q, Real psi_pm, Real pp, Real J,
                 Real B, Real T_load, Real L_0, Real omega0, Real theta0,
                 const std::vector<std::array<Real, 3>>& harmonics,
                 std::optional<std::shared_ptr<models::FluxMapDQ>> flux_map)
                  -> builder::CircuitBuilder& {
                  return b.add_pmsm_mna(name, pa, pb, pc, n, om, th, R_s, L_d,
                                        L_q, psi_pm, pp, J, B, T_load, L_0,
                                        omega0, theta0, harmonics,
                                        flux_map.value_or(nullptr));
              },
              py::arg("name"),
              py::arg("phase_a"), py::arg("phase_b"), py::arg("phase_c"),
              py::arg("neutral"),
              py::arg("omega_node"), py::arg("theta_node"),
              py::arg("R_s"), py::arg("L_d"), py::arg("L_q"),
              py::arg("psi_pm"), py::arg("pole_pairs"),
              py::arg("J"),
              py::arg("B")      = 0.0,
              py::arg("T_load") = 0.0,
              py::arg("L_0")    = 0.0,
              py::arg("omega0") = 0.0,
              py::arg("theta0") = 0.0,
              py::arg("harmonics") = std::vector<std::array<Real, 3>>{},
              py::arg("flux_map") = py::none(),
              py::return_value_policy::reference,
              "Add an MNA-NATIVE PMSM (audit C.3): three stator "
              "branches `<name>_a/_b/_c` (phase -> neutral) stamped "
              "per Newton iteration with the FULL theta-dependent "
              "inductance matrix L(theta) = T^-1 diag(L_d, L_q, L_0) T "
              "and the PM flux, so an IPM's d/q dynamics are exact and "
              "there is no one-step lag; mechanics are NODES "
              "(`omega_node` carries a capacitor J and a resistor 1/B, "
              "`theta_node` a 1 F capacitor fed by omega), so they get "
              "the same trapezoidal companion in the same solve. T_em "
              "comes from the co-energy of the same L(theta) — the "
              "reluctance torque appears because L(theta) is in the "
              "matrix. A speed-dependent load is just another element "
              "on `omega_node`. Read speed with res.v(omega_node), angle "
              "with res.v(theta_node), phase currents with "
              "res.i('<name>_a'). Compare with the observer-based "
              "`add_pmsm` + `make_pmsm_observer`, which uses one "
              "average L (tau_d +100 %, tau_q -33 % on a 1/3 mH IPM) "
              "and a first-order lag (-0.37 % of speed at dt=5e-5). "
              "`harmonics=[(h, psi_h, phi_h), ...]` replaces the PM flux "
              "psi_pm*cos(theta_e) by sum psi_h*cos(h*theta_e + phi_h) "
              "(phases b, c at theta_e -/+ 120 deg): a non-sinusoidal "
              "back-EMF with the torque from the same series; see "
              "`CircuitBuilder.add_bldc_mna`. The `<name>.i_d/i_q` "
              "traces stay on the theta_e axis, so they are off the "
              "PM-fundamental axis by phi_1 when phi_1 != 0. "
              "`flux_map=FluxMapDQ` replaces the constant L_d, L_q, psi_pm "
              "by a saturating, cross-coupled co-energy map (L_d, L_q, "
              "psi_pm must then equal the map's values at i = 0; see "
              "`CircuitBuilder.add_pmsm_flux_map`, which passes them).")
        .def("add_induction_machine_mna",
              &builder::CircuitBuilder::add_induction_machine_mna,
              py::arg("name"),
              py::arg("phase_a"), py::arg("phase_b"), py::arg("phase_c"),
              py::arg("neutral"), py::arg("omega_node"),
              py::arg("R_s"), py::arg("L_ls"), py::arg("R_r"),
              py::arg("L_lr"), py::arg("L_m"), py::arg("pole_pairs"),
              py::arg("J"),
              py::arg("B")      = 0.0,
              py::arg("T_load") = 0.0,
              py::arg("omega0") = 0.0,
              py::return_value_policy::reference,
              "Add an MNA-NATIVE squirrel-cage induction machine: three "
              "stator branches `<name>_a/_b/_c` (phase -> neutral) with a "
              "constant stationary-frame inductance, the rotor flux "
              "linkage as the voltages of two internal nodes "
              "`<name>_psi_ra`, `<name>_psi_rb` [Wb] (1 F capacitors), "
              "and the torque 1.5*pp*(L_m/L_r)*(psi_ra*i_sb - psi_rb*i_sa) "
              "into `omega_node` (mechanical speed [rad/s], capacitor J, "
              "resistor 1/B). Rotor quantities referred to the stator, "
              "amplitude-invariant Clarke. Solved by Newton on pwl and "
              "trbdf2; dsed and the AC sweep refuse it by name. Read speed "
              "with res.v(omega_node), phase currents with "
              "res.i('<name>_a').")
        .def("add_igbt_level1",
              &builder::CircuitBuilder::add_igbt_level1,
              py::arg("name"),
              py::arg("collector"), py::arg("emitter"),
              py::arg("gate"),
              py::arg("V_CE_sat") = 1.5,
              py::arg("R_CE_sat") = 0.05,
              py::arg("V_T")      = 5.0,
              py::arg("kappa")    = 10.0,
              py::arg("v_knee")   = 0.01,
              py::arg("with_fwd") = false,
              py::arg("tau_tail") = 0.0,
              py::arg("k_tail")   = 0.0,
              py::arg("V_F")      = 0.7,
              py::kw_only(), py::arg("form") = "conductance",
              py::return_value_policy::reference,
              "Add a 3-terminal IGBT Level 1 (linear-"
              "conduction model with cutoff sigmoid). "
              "Collector-emitter is a Nonlinear branch; "
              "gate is an ideal voltage reference (no gate "
              "current). On-state I_C = (V_CE-V_CE_sat) / "
              "R_CE_sat, clamped to I_C >= 0 with a knee of "
              "width `v_knee` — an IGBT is a minority-carrier "
              "device and cannot conduct in reverse. Pass "
              "`with_fwd=True` for the co-packaged anti-"
              "parallel freewheeling diode, which an inductive "
              "load needs: without it there is no path for the "
              "freewheeling current and the solve fails. `V_F` is "
              "that diode's CONDUCTION DROP (v = V_F + i/1e3 S "
              "while on; it turns on at V_F). The transient "
              "engines stamp the device per Newton iteration. "
              "`tau_tail`/`k_tail` add the TURN-OFF TAIL: when the "
              "gate falls below threshold the MOS channel cuts off "
              "in nanoseconds, but the carriers stored in the n- "
              "drift region can only recombine, and that tail is "
              "40-70% of an IGBT's turn-off energy. `k_tail` is "
              "the fraction of on-state current still flowing just "
              "after cutoff and `tau_tail` its decay constant "
              "(0.1-10 us) — both read straight off a datasheet "
              "turn-off waveform, and they must be given together. "
              "They default to OFF, unlike the MOSFET's body "
              "diode, because NPT/PT/trench/field-stop parts "
              "differ by orders of magnitude, so any default would "
              "put a fabricated number in your loss report. The DC "
              "curve is unchanged either way.")
        .def("add_mosfet_level1",
              &builder::CircuitBuilder::add_mosfet_level1,
              py::arg("name"),
              py::arg("drain"), py::arg("source"),
              py::arg("gate"),
              py::arg("K"), py::arg("V_T"),
              py::arg("lambda_") = 0.02,
              py::arg("kappa")   = 15.0,
              py::arg("with_body_diode") = true,
              py::arg("V_F") = 0.7,
              py::kw_only(), py::arg("form") = "conductance",
              py::return_value_policy::reference,
              "Add a 3-terminal SH1 MOSFET (Shichman-Hodges "
              "Level 1). Cutoff/triode/saturation regions "
              "blended via sigmoid for C¹-smooth Newton "
              "convergence. K [A/V²], V_T [V], lambda_ [1/V] "
              "channel-length modulation, kappa [1/V] sigmoid "
              "sharpness. Drain → source is a Nonlinear "
              "branch; gate is a node reference (no gate "
              "current — ideal gate). The intrinsic "
              "anti-parallel body diode (source→drain, named "
              "`<name>_body`) is added BY DEFAULT — it is what "
              "carries the current when the gate is off and an "
              "inductive load would otherwise have no path. "
              "Set `with_body_diode=False` for an eGaN HEMT, "
              "which has no p-n body diode. `V_F` is the body "
              "diode's CONDUCTION DROP (v_SD = V_F + i/1e3 S while "
              "on; it turns on at V_F). The transient engines "
              "stamp the MOSFET each Newton iteration.")
        .def("run_preflight",
              [](builder::CircuitBuilder& self,
                  const pwl::PreflightOptions& opts) {
                  return self.run_preflight(opts);
              },
              py::arg_v("options", pwl::PreflightOptions{}, "PreflightOptions()"),
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
              "as the voltage across it, v(n_pos) - v(n_neg); a loop of\n"
              "c0 values that breaks KVL raises ValueError.")
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
              py::call_guard<py::gil_scoped_release>(),
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
        .def("add_external_device",
              [](builder::CircuitBuilder& b, const std::string& name,
                 const std::vector<std::string>& nodes, const std::string& lib,
                 const std::string& model, const py::object& params,
                 bool check_contract, bool debug_jacobian,
                 bool record) -> builder::CircuitBuilder& {
                  // add-external-device-abi: loading `lib` runs its native
                  // code; this call (the user naming the path) is the opt-in.
                  auto m = xdev::load_model(lib, model);
                  std::vector<xdev::NamedParam> given;
                  if (!params.is_none()) {
                      for (const auto& [k, v] : params.cast<py::dict>()) {
                          xdev::NamedParam np;
                          np.name = py::cast<std::string>(k);
                          if (py::isinstance<py::tuple>(v)) {
                              const auto t = v.cast<py::tuple>();
                              if (t.size() != 2) {
                                  throw py::value_error("add_external_device(\"" + name
                                      + "\"): parameter '" + np.name
                                      + "' must be a number or (value, unit)");
                              }
                              np.value = t[0].cast<double>();
                              np.unit = t[1].cast<std::string>();
                              np.has_unit = true;
                          } else {
                              np.value = v.cast<double>();
                          }
                          given.push_back(std::move(np));
                      }
                  }
                  auto p = xdev::resolve_params(*m, name, given);
                  return b.add_external_device(name, nodes, std::move(m), std::move(p),
                                               check_contract, debug_jacobian, record);
              },
              py::arg("name"), py::arg("nodes"), py::arg("lib"), py::arg("model"),
              py::arg("params") = py::none(), py::kw_only(),
              py::arg("check_contract") = true, py::arg("debug_jacobian") = false,
              py::arg("record") = false,
              py::return_value_policy::reference,
              "Add an instance of an EXTERNAL device model (add-external-device-abi): "
              "`lib` is a shared library built against pulsim_xdev.h; loading it runs "
              "arbitrary native code with your privileges (no sandbox). `nodes` are "
              "its pins in the model's order; `params` maps field names to a value "
              "or (value, unit) (the unit must equal the declared one). The model's "
              "contract (Jacobian, finiteness, KCL, declared claims) is checked here "
              "and refused by name; `debug_jacobian=True` re-checks the Jacobian at "
              "every evaluation of the run. `record=True` traces the device's "
              "readings at every accepted step of a pwl / trbdf2 run: "
              "res.device_output(name, key) with key 'i(<pin>)', 'power' or one of "
              "the model's named outputs.")
        .def("add_saturable_inductor_table",
              &builder::CircuitBuilder::add_saturable_inductor_table,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("i_knots"), py::arg("lambda_knots"),
              py::return_value_policy::reference,
              "Saturable inductor from a (i, lambda) table for i >= 0 "
              "starting at the origin, strictly increasing in both; "
              "odd-extended, monotone-cubic between knots, linear "
              "beyond the last. Same device as add_saturable_inductor "
              "with a tabulated law.")
        .def("add_gapped_core_inductor",
              &builder::CircuitBuilder::add_gapped_core_inductor,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("N"), py::arg("Ae"), py::arg("le"), py::arg("lg"),
              py::arg("mu_r0") = 2000.0, py::arg("B_sat") = 0.35,
              py::arg("knots") = 128,
              py::return_value_policy::reference,
              "Saturable inductor on a gapped core from geometry: N "
              "turns, Ae [m^2], le [m], TOTAL gap lg [m], initial mu_r, "
              "B_sat [T] (= mu0*Ms, the datasheet value). lambda(i) is "
              "generated from Ampere's law by sweeping the core field "
              "through M = Ms*tanh(H/H0), with exact slopes.")
        .def("add_diode",
              [](builder::CircuitBuilder& self, std::string_view name,
                 std::string_view anode, std::string_view cathode,
                 std::optional<double> g_on, std::optional<double> g_off,
                 double V_th, double V_f, std::string_view form,
                 std::optional<double> R_on) -> builder::CircuitBuilder& {
                  // add-branch-form-ideal-switch: form='branch' (R_on,
                  // g_off default 0; a given g_on means R_on = 1/g_on).
                  if (form == "branch") {
                      if (R_on && g_on) {   // one resistance, two values: refuse, never drop one
                          throw py::value_error(std::format(
                              "add_diode(\"{}\"): form='branch' takes R_on or g_on "
                              "(R_on = 1/g_on), not both.", name));
                      }
                      const double r = R_on ? *R_on
                                     : g_on ? 1.0 / *g_on : 0.0;
                      return self.add_diode_branch(name, anode, cathode, r,
                                                   g_off.value_or(0.0), V_th,
                                                   V_f);
                  }
                  if (form != "conductance") {
                      throw py::value_error(std::format(
                          "add_diode(\"{}\"): form='{}'; expected "
                          "'conductance' or 'branch'.", name, form));
                  }
                  if (!g_on || !g_off || R_on) {
                      throw py::value_error(std::format(
                          "add_diode(\"{}\"): the conductance form takes g_on "
                          "and g_off (R_on is the branch form's: "
                          "form='branch').", name));
                  }
                  return self.add_diode(name, anode, cathode, *g_on, *g_off,
                                        V_th, V_f);
              },
              py::arg("name"), py::arg("anode"),
              py::arg("cathode"), py::arg("g_on") = py::none(),
              py::arg("g_off") = py::none(), py::arg("V_th") = 0.0,
              py::arg("V_f") = 0.0, py::kw_only(),
              py::arg("form") = "conductance",
              py::arg("R_on") = py::none(),
              py::return_value_policy::reference,
              "Add a binary switched diode (V2's "
              "SwitchedDiode). g_on/g_off in siemens. "
              "V_f: forward drop while ON (PSIM's diode voltage "
              "drop), in volts: v = V_f + i/g_on; must be finite "
              "and >= 0. V_th: turn-on threshold in volts, "
              "effectively max(V_th, V_f) when V_f > 0; V_th > V_f "
              "is a hysteresis band. OFF is plain g_off.")
        .def("add_nonlinear_diode",
              &builder::CircuitBuilder::add_nonlinear_diode,
              py::arg("name"), py::arg("anode"),
              py::arg("cathode"), py::arg("params"),
              py::return_value_policy::reference,
              "Add a smooth-blend nonlinear diode "
              "(V3's IdealDiode, AD-driven).")
        .def("add_ideal_switch",
              &builder::CircuitBuilder::add_ideal_switch,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("R_on") = 0.0, py::arg("g_off") = 0.0,
              py::return_value_policy::reference,
              "Add a controlled switch in BRANCH form (its own current "
              "unknown): R_on >= 0 in series when closed (0 = ideal), "
              "g_off >= 0 in parallel when open (0 = ideal). Engines and "
              "refusals: the capability table (add-branch-form-ideal-switch).")
        .def("add_switch",
              [](builder::CircuitBuilder& self, std::string_view name,
                 std::string_view n_pos, std::string_view n_neg,
                 std::optional<double> g_on, std::optional<double> g_off,
                 std::string_view form,
                 std::optional<double> R_on) -> builder::CircuitBuilder& {
                  // add-branch-form-ideal-switch D6: as add_diode — the
                  // branch form takes R_on (else 1/g_on, else 0) and g_off
                  // (else 0); the conductance form needs g_on and g_off.
                  if (form == "branch") {
                      if (R_on && g_on) {   // one resistance, two values: refuse, never drop one
                          throw py::value_error(std::format(
                              "add_switch(\"{}\"): form='branch' takes R_on or g_on "
                              "(R_on = 1/g_on), not both.", name));
                      }
                      const double r = R_on ? *R_on
                                     : g_on ? 1.0 / *g_on : 0.0;
                      return self.add_ideal_switch(name, n_pos, n_neg, r,
                                                   g_off.value_or(0.0));
                  }
                  if (form != "conductance") {
                      throw py::value_error(std::format(
                          "add_switch(\"{}\"): form='{}'; expected "
                          "'conductance' or 'branch'.", name, form));
                  }
                  if (!g_on || !g_off || R_on) {
                      throw py::value_error(std::format(
                          "add_switch(\"{}\"): the conductance form takes g_on "
                          "and g_off (R_on is the branch form's: "
                          "form='branch').", name));
                  }
                  return self.add_switch(name, n_pos, n_neg, *g_on, *g_off);
              },
              py::arg("name"), py::arg("n_pos"),
              py::arg("n_neg"), py::arg("g_on") = py::none(),
              py::arg("g_off") = py::none(), py::kw_only(),
              py::arg("form") = "conductance",
              py::arg("R_on") = py::none(),
              py::return_value_policy::reference,
              "Add a controlled switch (driven by switch_fn at simulation "
              "time). form='conductance' (default): g_on/g_off in siemens, "
              "both required. form='branch': its own current unknown, "
              "R_on >= 0 in series when closed (default 0 = ideal; a given "
              "g_on means R_on = 1/g_on), g_off >= 0 when open (default 0) "
              "— the same element as add_ideal_switch.")
        // ---- Layer 2 V1: power-device helpers ----
        .def("add_mosfet",
              &builder::CircuitBuilder::add_mosfet,
              py::arg("name"), py::arg("drain"),
              py::arg("source"),
              py::arg("R_on") = 1e-3,
              py::arg("R_off") = 1e9,
              py::arg("body_diode") = true,
              py::arg("V_F") = 0.7,
              py::kw_only(), py::arg("form") = "conductance",
              py::return_value_policy::reference,
              "Add an n-channel power MOSFET: a controlled "
              "switch (drain → source) PLUS its intrinsic "
              "anti-parallel body diode (source → drain), "
              "named `<name>_body`, which is on by DEFAULT. "
              "No vertical power MOSFET exists without one, "
              "and a gate-off device in an inductive path has "
              "no freewheeling route without it. Pass "
              "`body_diode=False` for an eGaN HEMT (which "
              "conducts in reverse through the channel, not a "
              "p-n junction) or for a bare switch. Defaults: "
              "R_on=1mΩ, R_off=1GΩ, body V_F=0.7 V. V_F is the "
              "body diode's CONDUCTION DROP (add_diode's V_f): "
              "v_SD = V_F + i/1e3 S while on, and it turns on at "
              "V_F.")
        .def("add_mosfet_with_body_diode",
              &builder::CircuitBuilder::add_mosfet_with_body_diode,
              py::arg("name"), py::arg("drain"),
              py::arg("source"),
              py::arg("R_on") = 1e-3,
              py::arg("R_off") = 1e9,
              py::arg("V_F") = 0.7,
              py::arg("g_on_diode") = 1e3,
              py::arg("g_off_diode") = 1e-9,
              py::kw_only(), py::arg("form") = "conductance",
              py::return_value_policy::reference,
              "Add an n-channel power MOSFET WITH "
              "intrinsic anti-parallel body diode. Adds "
              "TWO branches: switch (drain→source) + "
              "diode (source→drain). V_F is the body diode's "
              "CONDUCTION DROP (add_diode's V_f): v_SD = V_F + "
              "i/g_on_diode while on, and it turns on at V_F.")
        .def("_set_device_losses",
              [](builder::CircuitBuilder& b, const std::string& device,
                 const std::string& junction, double tj,
                 const std::string& heat_node,
                 const thermal::LossTable3D* e_on,
                 const thermal::LossTable3D* e_off,
                 const thermal::ConductionTable* v_on,
                 const std::string& conduction) {
                  thermal::DeviceLossSpec s;
                  s.device = device;
                  s.junction = junction;
                  s.tj = tj;
                  s.heat_node = heat_node;
                  if (e_on) s.e_on = std::make_shared<const thermal::LossTable3D>(*e_on);
                  if (e_off) s.e_off = std::make_shared<const thermal::LossTable3D>(*e_off);
                  if (v_on) s.v_on = std::make_shared<const thermal::ConductionTable>(*v_on);
                  if (conduction == "table") {
                      s.conduction = thermal::ConductionHeat::Table;
                  } else if (conduction != "electrical") {
                      throw std::invalid_argument(
                          "set_device_losses(\"" + device + "\"): conduction "
                          "must be 'electrical' or 'table', not '" +
                          conduction + "'");
                  }
                  b.set_device_losses(std::move(s));
              },
              py::arg("device"), py::arg("junction"), py::arg("tj"),
              py::arg("heat_node"), py::arg("e_on").none(true),
              py::arg("e_off").none(true), py::arg("v_on").none(true),
              py::arg("conduction"),
              "Kernel half of CircuitBuilder.set_device_losses (tj = NaN "
              "when not given, '' for an absent node name).")
        .def("_device_loss_names",
              [](const builder::CircuitBuilder& b) {
                  std::vector<std::string> out;
                  for (const auto& s : b.pool().device_losses()) out.push_back(s.device);
                  return out;
              },
              "Devices with a loss port, in set_device_losses order.")
        .def("add_igbt",
              [](builder::CircuitBuilder& self, std::string_view name,
                 std::string_view collector, std::string_view emitter,
                 double R_on, double R_off, double V_ce0,
                 std::optional<bool> reverse_blocking,
                 double V_th, std::string_view form) -> builder::CircuitBuilder& {
                  // add-pwl-igbt-and-thyristor stage 1: the default stays
                  // the bidirectional switch for one release, with a
                  // warning (a knee implies reverse blocking).
                  if (!reverse_blocking && V_ce0 == 0.0) {
                      if (PyErr_WarnEx(PyExc_FutureWarning, std::format(
                              "add_igbt(\"{}\"): reverse_blocking is not "
                              "given; it defaults to False (a switch that "
                              "conducts both ways) in this release and to "
                              "True in the next. Pass reverse_blocking=True "
                              "(the IGBT) or False (the old device).",
                              name).c_str(), 1) != 0) {
                          throw py::error_already_set();
                      }
                  }
                  return self.add_igbt(name, collector, emitter, R_on, R_off,
                                       V_ce0,
                                       reverse_blocking.value_or(V_ce0 > 0.0),
                                       V_th, form);
              },
              py::arg("name"), py::arg("collector"),
              py::arg("emitter"),
              py::arg("R_on") = 10e-3,
              py::arg("R_off") = 1e9,
              py::kw_only(),
              py::arg("V_ce0") = 0.0,
              py::arg("reverse_blocking") = py::none(),
              py::arg("V_th") = 0.0,
              py::arg("form") = "conductance",
              py::return_value_policy::reference,
              "Add a PWL IGBT (collector → emitter), its mask bit the "
              "gate. reverse_blocking=True: ON while the gate is on and "
              "i = (v_CE - V_ce0)/R_on > 0, turned on at v_CE >= "
              "max(V_th, V_ce0); conduction power V_ce0*i + R_on*i^2. "
              "reverse_blocking=False: a switch that conducts both ways "
              "(V_ce0 must be 0). Not given: False with a FutureWarning "
              "(True when V_ce0 > 0). V_ce0, R_on: a straight line "
              "through two points of the datasheet V_CE(I_C) curve. No "
              "anti-parallel diode: add_diode(..., V_f=...) separately.")
        .def("add_thyristor",
              &builder::CircuitBuilder::add_thyristor,
              py::arg("name"), py::arg("anode"), py::arg("cathode"),
              py::arg("R_on") = 1e-3, py::arg("R_off") = 1e9,
              py::kw_only(),
              py::arg("V_f") = 0.0, py::arg("I_H") = 0.0,
              py::arg("V_th") = 0.0,
              py::return_value_policy::reference,
              "Add a PWL thyristor (SCR), its mask bit the gate: fires "
              "when the gate is on and v_AK >= max(V_th, V_f), conducts "
              "i = (v - V_f)/R_on, stays on while i > 0 (gate on) or "
              "i > I_H (gate off). Turn-off is the engine's current-"
              "crossing event. V_f, R_on, I_H: datasheet V_T0, r_T, I_H.")
        .def("add_gto",
              &builder::CircuitBuilder::add_gto,
              py::arg("name"), py::arg("anode"), py::arg("cathode"),
              py::arg("R_on") = 1e-3, py::arg("R_off") = 1e9,
              py::kw_only(),
              py::arg("V_f") = 0.0, py::arg("V_th") = 0.0,
              py::return_value_policy::reference,
              "Add a PWL GTO: conducts i = (v - V_f)/R_on > 0 while its "
              "mask bit (the gate-drive state) is on; the bit off turns "
              "it off whatever the current.")
        .def("add_triac",
              &builder::CircuitBuilder::add_triac,
              py::arg("name"), py::arg("t1"), py::arg("t2"),
              py::arg("R_on") = 1e-3, py::arg("R_off") = 1e9,
              py::kw_only(),
              py::arg("V_f") = 0.0, py::arg("I_H") = 0.0,
              py::arg("V_th") = 0.0,
              py::return_value_policy::reference,
              "Add a PWL TRIAC: two anti-parallel thyristors "
              "`<name>_fwd` (t1 -> t2) and `<name>_rev` (t2 -> t1), one "
              "gate bit each: switch_indices_of(name).")
        .def("switch_indices_of",
              &builder::CircuitBuilder::switch_indices_of,
              py::arg("name"),
              "Mask bit positions of a device: [i] for a switch or "
              "diode, [i_fwd, i_rev] for a TRIAC.")
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
                  Vector iw;  // sine / pulse / table current sources
                  sources::compute_current_waveform_b_extra(p, g, t, iw);
                  b += iw;
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
                          // add-unified-device-interface 12.2: the scalar
                          // params are the kind's listed field rows
                          // (ir::kKindFields, `listed` = the scalars this
                          // listed before, so the output is unchanged),
                          // keyed by the authoring name; the cases below add
                          // only kind names and the non-scalar entries
                          // (references, tables, forms).
                          {
                              const auto& v = pool.params_of(bid);
                              for (const auto& f : ir::fields_of(kind)) {
                                  if (f.listed) {
                                      params[py::str(std::string(f.key()))] = f.get(v);
                                  }
                              }
                          }
                          switch (kind) {
                          case K::Resistor: {
                              kind_str = "resistor";
                              break;
                          }
                          case K::VoltageSource: {
                              kind_str = "voltage_source";
                              break;
                          }
                          case K::Switch: {
                              kind_str = "switch";
                              if (pool.is_branch_form(bid)) {
                                  params["R_on"] = pool.branch_R_on(bid);
                                  params["branch_form"] = 1;
                              }
                              break;
                          }
                          case K::Capacitor: {
                              kind_str = "capacitor";
                              break;
                          }
                          case K::Inductor: {
                              kind_str = "inductor";
                              break;
                          }
                          case K::Diode: {
                              kind_str = "diode";
                              const auto& p = pool.diode_params(bid);
                              if (p.gate != models::SwitchedDiode::Gate::None) {
                                  params["gate"] =
                                      p.gate == models::SwitchedDiode::Gate::Held
                                          ? 1 : 2;
                                  params["I_H"] = p.I_H;
                              }
                              if (pool.is_branch_form(bid)) {
                                  params["R_on"] = pool.branch_R_on(bid);
                                  params["branch_form"] = 1;
                              }
                              break;
                          }
                          case K::NonlinearDiode: {
                              kind_str = "nonlinear_diode";
                              break;
                          }
                          case K::CurrentSource: {
                              kind_str = "current_source";
                              break;
                          }
                          case K::PWMVoltageSource:
                              kind_str = "pwm_voltage_source"; break;
                          case K::SineVoltageSource: {
                              kind_str = "sine_voltage_source";
                              break;
                          }
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
                          case K::NonlinearCapacitor:
                              kind_str = "nonlinear_capacitor"; break;
                          case K::ShockleyDiode:
                              kind_str = "shockley_diode"; break;
                          case K::ExcessLoss:
                              kind_str = "excess_loss"; break;
                          case K::LauritzenDiode:
                              kind_str = "lauritzen_diode"; break;
                          case K::ExternalDevice: {
                              kind_str = "external_device";
                              const auto& in = *pool.external_device_params(bid).inst;
                              const auto& md = in.model().d();
                              for (std::uint32_t k = 0; k < md.n_fields; ++k) {
                                  params[md.fields[k].name] = in.params()[k];
                              }
                              break;
                          }
                          case K::PmsmMna:
                              kind_str = "pmsm_mna"; break;
                          case K::InductionMachineMna:
                              kind_str = "induction_machine_mna"; break;
                          case K::IdealTransformer: {
                              kind_str = "ideal_transformer";
                              break;
                          }
                          case K::TransmissionLine: {
                              kind_str = "transmission_line";
                              const Index pb = pool.line_partner(bid);
                              params["partner"] = std::string(self.name_of(pb));
                              params["port"] = pool.is_line_port_b(bid) ? "b" : "a";
                              break;
                          }
                          // add-controlled-source-set
                          case K::VCCS: {
                              kind_str = "vccs";
                              const auto [cp, cn] = pool.control_nodes(bid);
                              params["in_pos"] = static_cast<int>(cp);
                              params["in_neg"] = static_cast<int>(cn);
                              break;
                          }
                          case K::CCCS:
                              kind_str = "cccs";
                              params["control"] = std::string(
                                  self.name_of(pool.control_branch(bid)));
                              break;
                          case K::CCVS:
                              kind_str = "ccvs";
                              params["control"] = std::string(
                                  self.name_of(pool.control_branch(bid)));
                              break;
                          case K::Gyrator: {
                              kind_str = "gyrator";
                              const auto [p1, n1] = pool.control_nodes(bid);
                              params["p1_pos"] = static_cast<int>(p1);
                              params["p1_neg"] = static_cast<int>(n1);
                              break;
                          }
                          case K::SineCurrentSource: {
                              kind_str = "sine_current_source";
                              break;
                          }
                          case K::PulseCurrentSource: {
                              kind_str = "pulse_current_source";
                              break;
                          }
                          case K::PwlCurrentSource: {
                              kind_str = "pwl_current_source";
                              const auto& w =
                                  pool.pwl_current_source_params(bid).w;
                              params["t"] = w.t;
                              params["i"] = w.v;
                              params["periodic"] = w.periodic;
                              break;
                          }
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
                      if (std::string_view(kind_str) == "external_device") {
                          // add-external-device-abi 1.6: trace a result to its code
                          const auto& ep = pool.external_device_params(bid);
                          const auto& m = ep.inst->model();
                          char hx[17];
                          std::snprintf(hx, sizeof hx, "%016llx",
                                        static_cast<unsigned long long>(m.library_hash()));
                          desc["model"] = m.name();
                          desc["library"] = m.library_path();
                          desc["library_hash"] = std::string("fnv1a64:") + hx;
                          py::list pins;
                          for (const Index n : ep.pins) pins.append(static_cast<int>(n));
                          desc["pins"] = py::tuple(pins);
                          desc["internal_branch"] = static_cast<int>(
                              std::find(ep.branches.begin(), ep.branches.end(), bid)
                              - ep.branches.begin());
                      }
                      // add-mna-native-machines D2: a recorded composite's
                      // anchor entry carries it, its parameters read back
                      // from the members (never stale after with_value).
                      for (const auto& cr : self.composites()) {
                          if (cr.anchor != self.name_of(bid)) continue;
                          desc["composite"] = cr.kind;
                          desc["composite_name"] = cr.name;
                          desc["sub_elements"] = cr.members;
                          if (cr.kind == "dc_machine") {
                              auto member = [&](const char* sfx) -> std::optional<Index> {
                                  const std::string m = cr.name + sfx;
                                  if (std::find(cr.members.begin(), cr.members.end(), m)
                                      == cr.members.end()) return std::nullopt;
                                  return self.branch_id_of(m);
                              };
                              py::dict pd;
                              const auto ra = member(".ra");
                              pd["R_a"] = ra ? Real{1} / pool.resistor_params(*ra).G : Real{0};
                              pd["L_a"] = pool.inductor_params(member(".la").value()).L;
                              pd["K"] = pool.ideal_transformer_params(member(".em").value()).n;
                              pd["J"] = pool.capacitor_params(member(".j").value()).C;
                              const auto bb = member(".b");
                              pd["B"] = bb ? pool.resistor_params(*bb).G : Real{0};
                              const auto tl = member(".tl");
                              pd["T_load"] = tl ? pool.current_source_params(*tl).I : Real{0};
                              desc["composite_params"] = pd;
                          }
                      }
                      // add-power-mosfet-datasheet-device: the channel's
                      // entry stands for the whole composite.
                      if (const auto* pm = self.power_mosfet_record(bid)) {
                          const auto& pp = pm->params;
                          auto cap = [](const models::PowerMosfet::Capacitance& c) {
                              py::dict d;
                              if (const Real* x = std::get_if<Real>(&c)) {
                                  d["law"] = *x > Real{0} ? "linear" : "none";
                                  d["C"] = *x;
                                  return d;
                              }
                              const auto& q = std::get<models::NonlinearCapacitor::Params>(c);
                              d["law"] = models::NonlinearCapacitor::law_name(q);
                              if (q.law == models::ChargeLawKind::Junction) {
                                  d["C0"] = q.C0; d["V0"] = q.V0; d["m"] = q.m;
                              } else if (q.law == models::ChargeLawKind::VdmosCgd) {
                                  d["cgd_max"] = q.cgd_max; d["cgd_min"] = q.cgd_min;
                                  d["a"] = q.a;
                              } else if (q.law == models::ChargeLawKind::Tanh) {
                                  d["c_max"] = q.cgd_max; d["c_min"] = q.cgd_min;
                                  d["V0"] = q.V0;
                              } else {
                                  d["knots"] = q.table->size();
                              }
                              return d;
                          };
                          py::dict pd;
                          pd["K"] = pp.K; pd["V_T"] = pp.V_T;
                          pd["lambda_"] = pp.lambda; pd["kappa"] = pp.kappa;
                          pd["R_d"] = pp.R_d; pd["C_gs"] = pp.C_gs;
                          pd["C_gd"] = cap(pp.C_gd); pd["C_ds"] = cap(pp.C_ds);
                          pd["R_g"] = pp.R_g; pd["L_s"] = pp.L_s;
                          using BD = models::PowerMosfet::BodyDiode;
                          pd["body_diode"] = pp.body == BD::Lauritzen ? "lauritzen"
                              : pp.body == BD::Shockley ? "shockley" : "none";
                          pd["T_j"] = pp.T_j; pd["rds_tc1"] = pp.rds_tc1;
                          pd["rds_tc2"] = pp.rds_tc2; pd["vt_tc"] = pp.vt_tc;
                          pd["K_eff"] = pm->K_eff; pd["V_T_eff"] = pm->V_T_eff;
                          pd["R_d_eff"] = pm->R_d_eff;
                          desc["kind"] = "power_mosfet";
                          desc["params"] = pd;
                          desc["pins"] = py::make_tuple(pm->drain, pm->source, pm->gate);
                          desc["sub_elements"] = pm->sub_elements;
                      }
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
              "current role hint.")
        // ---- add-power-mosfet-datasheet-device -------------------------
        .def("add_nonlinear_capacitor",
              &builder::CircuitBuilder::add_nonlinear_capacitor_law,
              py::arg("name"), py::arg("n_pos"), py::arg("n_neg"),
              py::arg("law"),
              py::return_value_policy::reference,
              "Charge-based nonlinear capacitor with an explicit "
              "ChargeLaw (junction, table or vdmos_cgd). Stamped in "
              "charge form, so charge is conserved over any cycle.")
        .def("add_power_mosfet",
              [](builder::CircuitBuilder& self, std::string_view name,
                 std::string_view drain, std::string_view source,
                 std::string_view gate, Real K, Real V_T, Real lambda_,
                 Real kappa, Real R_d, Real C_gs, const py::object& C_gd,
                 const py::object& C_ds, Real R_g, Real L_s,
                 const py::object& body_diode, Real T_j, Real rds_tc1,
                 Real rds_tc2, Real vt_tc) -> builder::CircuitBuilder& {
                  using PM = models::PowerMosfet;
                  auto cap = [&](const py::object& o, const char* what)
                      -> PM::Capacitance {
                      if (py::isinstance<models::NonlinearCapacitor::Params>(o)) {
                          return o.cast<models::NonlinearCapacitor::Params>();
                      }
                      if (py::isinstance<models::ChargeTable>(o)) {
                          return models::NonlinearCapacitor::from_table(
                              std::make_shared<const models::ChargeTable>(
                                  o.cast<models::ChargeTable>()));
                      }
                      if (py::isinstance<py::float_>(o) || py::isinstance<py::int_>(o)) {
                          return o.cast<Real>();
                      }
                      throw py::type_error(std::format(
                          "add_power_mosfet(\"{}\"): {} must be a float "
                          "(linear, farads), a ChargeTable or a ChargeLaw; "
                          "got {}", name, what,
                          std::string(py::str(py::type::handle_of(o).attr("__name__")))));
                  };
                  PM::Params p;
                  p.K = K; p.V_T = V_T; p.lambda = lambda_; p.kappa = kappa;
                  p.R_d = R_d; p.C_gs = C_gs;
                  p.C_gd = cap(C_gd, "C_gd");
                  p.C_ds = cap(C_ds, "C_ds");
                  p.R_g = R_g; p.L_s = L_s; p.T_j = T_j;
                  p.rds_tc1 = rds_tc1; p.rds_tc2 = rds_tc2; p.vt_tc = vt_tc;
                  if (!body_diode.is_none()) {
                      if (!py::isinstance<py::dict>(body_diode)) {
                          throw py::type_error(std::format(
                              "add_power_mosfet(\"{}\"): body_diode must be "
                              "None or a dict with 'model' = 'lauritzen' or "
                              "'shockley'", name));
                      }
                      py::dict d = body_diode.cast<py::dict>();
                      if (!d.contains("model")) {
                          throw py::value_error(std::format(
                              "add_power_mosfet(\"{}\"): body_diode needs "
                              "'model' ('lauritzen' or 'shockley')", name));
                      }
                      const auto model = d["model"].cast<std::string>();
                      std::vector<std::string> allowed;
                      if (model == "lauritzen") {
                          p.body = PM::BodyDiode::Lauritzen;
                          allowed = {"model", "I_S", "n", "V_T", "tau", "T_M", "G_min"};
                      } else if (model == "shockley") {
                          p.body = PM::BodyDiode::Shockley;
                          allowed = {"model", "I_S", "n", "V_T", "G_min", "BV"};
                      } else {
                          throw py::value_error(std::format(
                              "add_power_mosfet(\"{}\"): body_diode model "
                              "'{}' — expected 'lauritzen' or 'shockley' "
                              "(pass body_diode=None for no diode)", name, model));
                      }
                      for (auto item : d) {
                          const auto k = item.first.cast<std::string>();
                          if (std::find(allowed.begin(), allowed.end(), k) == allowed.end()) {
                              throw py::value_error(std::format(
                                  "add_power_mosfet(\"{}\"): body_diode key "
                                  "'{}' is not a {} parameter", name, k, model));
                          }
                      }
                      auto get = [&](const char* k, Real dflt) {
                          return d.contains(k) ? d[k].cast<Real>() : dflt;
                      };
                      auto& L = p.lauritzen;
                      auto& S = p.shockley;
                      if (p.body == PM::BodyDiode::Lauritzen) {
                          L.I_S = get("I_S", L.I_S); L.n = get("n", L.n);
                          L.V_T = get("V_T", L.V_T); L.tau = get("tau", L.tau);
                          L.T_M = get("T_M", L.T_M); L.G_min = get("G_min", L.G_min);
                      } else {
                          S.I_S = get("I_S", S.I_S); S.n = get("n", S.n);
                          S.V_T = get("V_T", S.V_T); S.G_min = get("G_min", S.G_min);
                          S.BV = get("BV", S.BV);
                      }
                  }
                  return self.add_power_mosfet(name, drain, source, gate, p);
              },
              py::arg("name"), py::arg("drain"), py::arg("source"),
              py::arg("gate"), py::kw_only(),
              py::arg("K"), py::arg("V_T"),
              py::arg("lambda_") = 0.0, py::arg("kappa") = 15.0,
              py::arg("R_d") = 0.0, py::arg("C_gs"),
              py::arg("C_gd"), py::arg("C_ds"),
              py::arg("R_g") = 0.0, py::arg("L_s") = 0.0,
              py::arg("body_diode") = py::none(),
              py::arg("T_j") = 25.0, py::arg("rds_tc1") = 0.0,
              py::arg("rds_tc2") = 0.0, py::arg("vt_tc") = 0.0,
              py::return_value_policy::reference,
              "Datasheet power MOSFET as one call: Level-1 channel (K "
              "[A/V^2], V_T [V], third quadrant by terminal swap), drift "
              "resistance R_d [ohm], C_gs [F], C_gd and C_ds (float = linear "
              "farads, 0 = absent; or a ChargeTable / ChargeLaw), internal "
              "gate resistance R_g [ohm], common source inductance L_s [H], "
              "body_diode None or {'model': 'lauritzen'|'shockley', ...}. "
              "T_j [degC] is run-constant: R_d *= r, K /= r with r = 1 + "
              "rds_tc1*dT + rds_tc2*dT^2, V_T += vt_tc*dT (dT = T_j - 25). "
              "Sub-elements: name (channel), name.rd/.rg/.ls/.cgs/.cgd/.cds, "
              "name_body. Runs on pwl (time step must resolve the gate "
              "transient: ~0.2 ns for a 60 ns plateau) and trbdf2. "
              "components() reports one 'power_mosfet' entry.");

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
    graph_cls
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

    pool_cls
        .def("has_nonlinear_devices",
              &pwl::DevicePool::has_nonlinear_devices,
              "Alias of requires_newton(): True if any branch "
              "needs a Newton solve (the capability table's "
              "NewtonStatic / NewtonStateful kinds, see "
              "pulsim.engine_capabilities()).")
        .def("requires_newton",
              &pwl::DevicePool::requires_newton,
              "True if any branch needs a Newton solve.")
        .def("requires_static_refresh",
              &pwl::DevicePool::requires_static_refresh,
              "True if any branch needs the static Newton chain "
              "(smooth / Shockley diode, MOSFET, IGBT).")
        .def("branch_var_id_for_inductor",
              &pwl::DevicePool::branch_var_id_for_inductor,
              py::arg("branch_id"), py::arg("graph"),
              "Resolve the state-vector index for an inductor "
              "(or saturable inductor) branch's i_L unknown. "
              "Throws if `branch_id` is not registered as an "
              "inductor.")
        .def("branch_var_id_for_switch",
              &pwl::DevicePool::branch_var_id_for_switch,
              py::arg("branch_id"), py::arg("graph"),
              "State-vector index of a branch-form switch or diode's "
              "current unknown i_s, or -1 for the conductance form "
              "(add-branch-form-ideal-switch).")
        .def("branch_var_id_for_source",
              &pwl::DevicePool::branch_var_id_for_source,
              py::arg("branch_id"), py::arg("graph"),
              "Resolve the state-vector index for a voltage-"
              "source branch's i_source unknown (also works "
              "for PWM/Sine/Pulse/VCVS sources, since they "
              "share the same branch-var numbering).")
        .def("control_column",
              &pwl::DevicePool::control_column,
              py::arg("branch_id"), py::arg("graph"),
              "State-vector column of the current a CCCS / CCVS senses.")
        .def("current_waveform_value",
              &pwl::DevicePool::current_waveform_value,
              py::arg("branch_id"), py::arg("t"),
              "Value [A] of a sine / pulse / table current source at "
              "`t`, leaving at its `n_pos`.")
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
        .def("pmsm_mna_machines",
              [](const pwl::DevicePool& self) {
                  py::list out;
                  for (const auto& m : self.pmsm_mna_machines()) {
                      py::dict d;
                      d["branch_ids"] = py::make_tuple(
                          static_cast<int>(m.branch_id[0]),
                          static_cast<int>(m.branch_id[1]),
                          static_cast<int>(m.branch_id[2]));
                      d["omega_node"] = static_cast<int>(m.omega_node);
                      d["theta_node"] = static_cast<int>(m.theta_node);
                      d["omega0"] = m.omega0;   // the DC point holds these
                      d["theta0"] = m.theta0;
                      py::dict pr;
                      pr["R_s"] = m.params.R_s;
                      pr["L_d"] = m.params.L_d;
                      pr["L_q"] = m.params.L_q;
                      pr["L_0"] = models::PmsmMna::L_zero(m.params);
                      pr["psi_pm"] = m.params.psi_pm;
                      pr["pole_pairs"] = m.params.pole_pairs;
                      pr["T_load"] = m.params.T_load;
                      py::list hs;
                      if (m.params.harmonics) {
                          for (const auto& t : *m.params.harmonics) {
                              hs.append(py::make_tuple(t.h, t.psi, t.phi));
                          }
                      }
                      pr["harmonics"] = hs;
                      pr["flux_map"] = m.params.flux_map
                          ? py::cast(std::const_pointer_cast<models::FluxMapDQ>(
                                m.params.flux_map))
                          : py::none();
                      d["params"] = pr;
                      out.append(d);
                  }
                  return out;
              },
              "MNA-native PMSMs registered on this pool, one dict per "
              "machine: branch_ids (a, b, c), omega_node, theta_node "
              "(state-vector indices) and params. `simulate` reads this "
              "to attach the `<name>.omega/theta/T_em/i_d/i_q` traces.")
        .def("induction_machines",
              [](const pwl::DevicePool& self) {
                  py::list out;
                  for (const auto& m : self.induction_machines()) {
                      py::dict d;
                      d["branch_ids"] = py::make_tuple(
                          static_cast<int>(m.branch_id[0]),
                          static_cast<int>(m.branch_id[1]),
                          static_cast<int>(m.branch_id[2]));
                      d["omega_node"] = static_cast<int>(m.omega_node);
                      d["rotor_nodes"] = py::make_tuple(
                          static_cast<int>(m.rotor_node[0]),
                          static_cast<int>(m.rotor_node[1]));
                      d["omega0"] = m.omega0;   // the DC point holds it
                      py::dict pr;
                      pr["R_s"] = m.params.R_s;
                      pr["L_ls"] = m.params.L_ls;
                      pr["R_r"] = m.params.R_r;
                      pr["L_lr"] = m.params.L_lr;
                      pr["L_m"] = m.params.L_m;
                      pr["pole_pairs"] = m.params.pole_pairs;
                      pr["T_load"] = m.params.T_load;
                      d["params"] = pr;
                      out.append(d);
                  }
                  return out;
              },
              "MNA-native induction machines on this pool, one dict per "
              "machine: branch_ids (a, b, c), omega_node, rotor_nodes "
              "(state-vector indices of psi_ra, psi_rb) and params.")
        .def("igbt_has_tail",
              &pwl::DevicePool::igbt_has_tail,
              py::arg("branch_id"),
              "True when this branch is an IGBT with a "
              "turn-off tail (tau_tail and k_tail both set). "
              "The engine router uses it to decide whether "
              "TR-BDF2 has to refuse the circuit.")
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
    stored_kind_cls
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
        .value("SaturableInductor",   pwl::DevicePool::StoredKind::SaturableInductor)
        .value("NonlinearCapacitor",  pwl::DevicePool::StoredKind::NonlinearCapacitor)
        .value("ShockleyDiode",       pwl::DevicePool::StoredKind::ShockleyDiode)
        .value("LauritzenDiode",      pwl::DevicePool::StoredKind::LauritzenDiode)
        .value("PmsmMna",             pwl::DevicePool::StoredKind::PmsmMna)
        .value("IdealTransformer",    pwl::DevicePool::StoredKind::IdealTransformer)
        .value("VCCS",                pwl::DevicePool::StoredKind::VCCS)
        .value("CCCS",                pwl::DevicePool::StoredKind::CCCS)
        .value("CCVS",                pwl::DevicePool::StoredKind::CCVS)
        .value("Gyrator",             pwl::DevicePool::StoredKind::Gyrator)
        .value("SineCurrentSource",   pwl::DevicePool::StoredKind::SineCurrentSource)
        .value("PulseCurrentSource",  pwl::DevicePool::StoredKind::PulseCurrentSource)
        .value("PwlCurrentSource",    pwl::DevicePool::StoredKind::PwlCurrentSource)
        .value("InductionMachineMna", pwl::DevicePool::StoredKind::InductionMachineMna)
        .value("ExternalDevice", pwl::DevicePool::StoredKind::ExternalDevice)
        .value("ExcessLoss", pwl::DevicePool::StoredKind::ExcessLoss)
        .value("TransmissionLine", pwl::DevicePool::StoredKind::TransmissionLine);

    // ---- Phase 4 C.4: exact B–H replay for the in-loop JA device ----------
    m.def("ja_replay",
          [](double N, double Ae, double le, double lg,
             double Ms, double a, double alpha, double c, double k,
             double M0, py::array_t<double> current) {
              models::JilesAthertonCore::Params cp;
              cp.N = N; cp.Ae = Ae; cp.le = le; cp.lg = lg; cp.M0 = M0;
              cp.ja.Ms = Ms; cp.ja.a = a; cp.ja.alpha = alpha; cp.ja.c = c; cp.ja.k = k;
              models::JilesAthertonCore::validate(cp, "ja_replay");
              auto buf = current.unchecked<1>();
              const py::ssize_t n = buf.shape(0);
              py::array_t<double> H(n), M(n), B(n);
              auto h = H.mutable_unchecked<1>();
              auto mm = M.mutable_unchecked<1>();
              auto bb = B.mutable_unchecked<1>();
              models::JilesAthertonCore::State st;
              st.H = 0.0; st.M = M0; st.n_sub = cp.substeps_min;
              for (py::ssize_t i = 0; i < n; ++i) {
                  const auto ev = models::JilesAthertonCore::evaluate(cp, st, buf(i));
                  // Same per-step sub-step and direction rules as the
                  // history commit, so the replay IS the solver's path.
                  const double span = 2.0 * std::abs(ev.H - st.H);
                  const double leg = ev.H - st.H;
                  st.delta_hint = leg > 0 ? 1.0 : leg < 0 ? -1.0 : st.delta_hint;
                  st.n_sub = models::JilesAthertonCore::substeps_for(cp, span);
                  st.H = ev.H; st.M = ev.M;
                  h(i) = ev.H; mm(i) = ev.M;
                  bb(i) = models::ja_detail::MU_0 * (ev.H + ev.M);
              }
              return py::make_tuple(H, M, B);
          },
          py::arg("N"), py::arg("Ae"), py::arg("le"), py::arg("lg"),
          py::arg("Ms"), py::arg("a"), py::arg("alpha"), py::arg("c"),
          py::arg("k"), py::arg("M0"), py::arg("current"),
          "Replay the in-loop Jiles-Atherton device on a current trace with "
          "the kernel's own integrator; returns (H, M, B) arrays.");

    m.def("ja_evaluate",
          [](double N, double Ae, double le, double lg,
             double Ms, double a, double alpha, double c, double k,
             double H_n, double M_n, int n_sub, double hint, double i) {
              models::JilesAthertonCore::Params cp;
              cp.N = N; cp.Ae = Ae; cp.le = le; cp.lg = lg;
              cp.ja.Ms = Ms; cp.ja.a = a; cp.ja.alpha = alpha; cp.ja.c = c; cp.ja.k = k;
              models::JilesAthertonCore::State st;
              st.H = H_n; st.M = M_n; st.n_sub = n_sub; st.delta_hint = hint;
              const auto ev = models::JilesAthertonCore::evaluate(cp, st, i);
              return py::make_tuple(ev.H, ev.M, ev.lambda, ev.L, ev.dM_dH);
          },
          py::arg("N"), py::arg("Ae"), py::arg("le"), py::arg("lg"),
          py::arg("Ms"), py::arg("a"), py::arg("alpha"), py::arg("c"),
          py::arg("k"), py::arg("H_n"), py::arg("M_n"), py::arg("n_sub"),
          py::arg("hint"), py::arg("i"),
          "One in-loop Jiles-Atherton evaluation at a trial current from a "
          "given step-start state: (H, M, lambda, L, dM_dH). A probe for "
          "tests and diagnostics.");

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
              const auto gated = diodes.gated_bits();
              const auto n_sw = graph.num_switches();
              std::vector<Size> controlled;
              for (Size i = 0; i < n_sw; ++i) {
                  if (!owned.get(i) || gated.get(i)) controlled.push_back(i);
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
        .def_readonly("event_evictions",
                       &pwl::CacheMetrics::event_evictions,
                       "solve_at entries evicted (LRU) under the byte "
                       "budget.")
        .def_readonly("structure_builds",
                       &pwl::CacheMetrics::structure_builds,
                       "Sparse-structure (template) builds of G/C/J: one "
                       "per assembly regime.")
        .def_readonly("symbolic_analyses",
                       &pwl::CacheMetrics::symbolic_analyses,
                       "Symbolic analyses run: one per assembly regime, "
                       "shared by every switch state.")
        .def_readonly("symbolic_fallbacks",
                       &pwl::CacheMetrics::symbolic_fallbacks,
                       "Matrices whose pattern refused the shared "
                       "analysis and were analysed privately.")
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
              }, py::arg("dt"), py::call_guard<py::gil_scoped_release>(),
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

    py::enum_<device::DevicePath>(m, "DevicePath",
        "Which code stamps the Newton kinds (add-unified-device-interface): "
        "Legacy = the hand stamps (default), Interface = every kind's law "
        "on the device interface.")
        .value("Legacy", device::DevicePath::Legacy)
        .value("Interface", device::DevicePath::Interface);

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
        .def_readwrite("newton_roundoff_budget",
                        &SimulationOptions::newton_roundoff_budget,
                        "Most consecutive Newton solves accepted at roundoff "
                        "before NewtonRoundoffBudgetExceeded (default 130).")
        .def_readwrite("enable_substep_state_correction",
                        &SimulationOptions::enable_substep_state_correction)
        .def_readwrite("split_snap", &SimulationOptions::split_snap,
                        "Split events within split_snap*dt of a step "
                        "boundary are taken at it (default 1e-6).")
        .def_readwrite("resolution_diagnostics",
                        &SimulationOptions::resolution_diagnostics,
                        "Record SimulationResult.resolution (truncation "
                        "estimate + ringing detector) on the pwl engine. "
                        "Read-only: states are bit-identical either way.")
        .def_readwrite("on_impulse_error",
                        &SimulationOptions::on_impulse_error,
                        "add-branch-form-ideal-switch: refuse the run at the "
                        "first impulsive re-initialisation (simulate("
                        "on_impulse='error')) instead of recording it.")
        .def_readwrite("mode_selection",
                        &SimulationOptions::mode_selection,
                        "Fixed engine diode decision: 0 the settle (guarded "
                        "by mode_guard), 1 'shadow', 2 'lcp' "
                        "(add-complementarity-mode-selection).")
        .def_readwrite("mode_guard", &SimulationOptions::mode_guard,
                        "Fixed engine: re-decide with the LCP a settle that "
                        "ends in a mode violating a diode's law.")
        .def_readwrite("device_path", &SimulationOptions::device_path,
                        "Which code stamps the Newton kinds: "
                        "DevicePath.Legacy (the hand stamps, default) or "
                        "DevicePath.Interface (each kind's law on the "
                        "device interface; refuses an unported kind by "
                        "name).")
        .def_readwrite("discontinuity_restart",
                        &SimulationOptions::discontinuity_restart,
                        "Fixed engine: take the step after a discontinuity "
                        "(mask change, diode flip, source jump, inconsistent "
                        "start) as two backward-Euler half-steps; gate mask "
                        "sampled at the step start. Off = the engine before "
                        "the rule, bit for bit.")
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
    // The same scan on a trace that is not a native SimulationResult
    // (the dsed engine's): `states` rows are full-MNA samples at `times`.
    m.def("find_implausible_voltage",
        [](const topology::Graph& graph, const pwl::DevicePool& pool,
            const std::vector<Real>& times,
            const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>& states, Real factor) {
            solver::SimulationResult r;
            r.times = times;
            for (Eigen::Index k = 0; k < states.rows(); ++k) {
                r.states.push_back(states.row(k).transpose());
            }
            return solver::find_implausible_voltage(graph, pool, r, factor);
        },
        py::arg("graph"), py::arg("pool"), py::arg("times"),
        py::arg("states"), py::arg("factor") = Real{100});

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

    py::class_<ReinitRecord>(m, "ReinitRecord",
        "One impulsive re-initialisation (add-branch-form-ideal-switch "
        "D4): kind 'cut_set' (lambda = flux, V*s) or 'capacitor_loop' "
        "(lambda = charge, C), or on the exact engine 'start_projection' "
        "(a given initial state the start mode cannot keep, moved at t0; "
        "projection_gap the largest move); dW the energy removed [J]; sample the "
        "stored sample of the step that carried the pulse (-1: not "
        "stored).")
        .def_readonly("t", &ReinitRecord::t)
        .def_readonly("kind", &ReinitRecord::kind)
        .def_readonly("devices", &ReinitRecord::devices)
        .def_readonly("states", &ReinitRecord::states)
        .def_readonly("lambda_", &ReinitRecord::lambda)
        .def_readonly("dW", &ReinitRecord::dW)
        .def_readonly("projection_gap", &ReinitRecord::projection_gap)
        .def_readonly("sample", &ReinitRecord::sample)
        .def_readonly("cause", &ReinitRecord::cause,
                      "The switch the impulsive energy dW is booked to "
                      "(pulsim.losses.impulse_losses).");

    py::class_<CommutationEvent>(m, "CommutationEvent",
        "Sub-step commutation timing event (V2.2 + V3).")
        .def_readonly("t_estimated",
                       &CommutationEvent::t_estimated)
        .def_readonly("branch_id",
                       &CommutationEvent::branch_id)
        .def_readonly("new_state",
                       &CommutationEvent::new_state);

    py::class_<solver::NewtonCounters>(m, "NewtonCounters",
        "Newton linear-algebra counters of one run "
        "(update-newton-persistent-solver). All zero without Newton.")
        .def_readonly("solves", &solver::NewtonCounters::solves)
        .def_readonly("iterations", &solver::NewtonCounters::iterations)
        .def_readonly("analyses", &solver::NewtonCounters::analyses)
        .def_readonly("factorizations",
                      &solver::NewtonCounters::factorizations)
        .def_readonly("partial_refactors",
                      &solver::NewtonCounters::partial_refactors)
        .def_readonly("lm_attempts", &solver::NewtonCounters::lm_attempts)
        .def_property_readonly("iterations_per_solve",
            [](const solver::NewtonCounters& c) {
                return py::array_t<std::uint16_t>(
                    static_cast<py::ssize_t>(c.iterations_per_solve.size()),
                    c.iterations_per_solve.data());
            },
            "Iterations of each Newton call, in call order (uint16).");
    py::class_<SimulationResult::RingingRecord>(m, "RingingRecord",
        "A run of >= 4 sign-alternating steps of one state entry "
        "(add-fixed-step-resolution-diagnostics).")
        .def_readonly("var", &SimulationResult::RingingRecord::var)
        .def_readonly("t_start", &SimulationResult::RingingRecord::t_start)
        .def_readonly("length", &SimulationResult::RingingRecord::length)
        .def_readonly("max_delta",
                       &SimulationResult::RingingRecord::max_delta);
    py::class_<SimulationResult::Resolution>(m, "ResolutionRecord",
        "Fixed-step resolution diagnostics. `lte` is an ESTIMATE of "
        "the per-step truncation relative to each variable's running "
        "peak, not an error bound; stateful devices are not covered.")
        .def_property_readonly("lte",
            [](const SimulationResult::Resolution& r) {
                return py::array_t<Real>(
                    static_cast<py::ssize_t>(r.lte.size()), r.lte.data());
            })
        .def_readonly("lte_max", &SimulationResult::Resolution::lte_max)
        .def_readonly("unestimated_fraction",
                       &SimulationResult::Resolution::unestimated_fraction)
        .def_readonly("lte_branch",
                       &SimulationResult::Resolution::lte_branch)
        .def_readonly("lte_time", &SimulationResult::Resolution::lte_time)
        .def_readonly("n_branches_covered",
                       &SimulationResult::Resolution::n_branches_covered)
        .def_readonly("ringing", &SimulationResult::Resolution::ringing);

    py::enum_<record::GridKind>(m, "GridKind",
        "What grid a result's rows sit on (add-probes-and-recorders D3).")
        .value("Uniform", record::GridKind::Uniform)
        .value("Accepted", record::GridKind::Accepted)
        .value("Requested", record::GridKind::Requested);
    {
        auto rs = py::class_<record::RunStats>(m, "RunStats",
            "Engine-independent run counters; zero where an engine does not "
            "report one. `engine` names the engine that filled them.");
        rs.def_readonly("engine", &record::RunStats::engine);
#define PULSIM_RUNSTAT(f) rs.def_readonly(#f, &record::RunStats::f);
        PULSIM_RUNSTAT(n_steps) PULSIM_RUNSTAT(n_accept) PULSIM_RUNSTAT(n_reject)
        PULSIM_RUNSTAT(n_solves) PULSIM_RUNSTAT(n_factorizations)
        PULSIM_RUNSTAT(n_gate_events) PULSIM_RUNSTAT(n_state_events)
        PULSIM_RUNSTAT(n_mode_events) PULSIM_RUNSTAT(n_ctrl_ticks)
        PULSIM_RUNSTAT(n_newton_iterations) PULSIM_RUNSTAT(n_newton_retries)
        PULSIM_RUNSTAT(n_forced_accepts) PULSIM_RUNSTAT(n_chatter_breaks)
        PULSIM_RUNSTAT(n_gate_undersampled) PULSIM_RUNSTAT(n_rk45_steps)
        PULSIM_RUNSTAT(n_bdf2_steps) PULSIM_RUNSTAT(n_exact_steps)
        PULSIM_RUNSTAT(gate_pulse_min) PULSIM_RUNSTAT(gate_sample_spacing_max)
        PULSIM_RUNSTAT(cpu_time_seconds)
#undef PULSIM_RUNSTAT
    }
    result_cls
        // update-fixed-step-loop-bookkeeping: `times` is a read-only
        // zero-copy 1-D view like `states` (it was a list copied on
        // every access). `list(res.times)` restores the old type.
        .def_property_readonly("times",
            [](py::object self) -> py::object {
                const auto& tv = self.cast<const SimulationResult&>().times;
                if (tv.empty()) {
                    return py::array_t<Real>(0);
                }
                py::array_t<Real> arr(
                    static_cast<py::ssize_t>(tv.size()), tv.data(), self);
                arr.attr("setflags")(py::arg("write") = false);
                return std::move(arr);
            },
            "Sample times as a read-only numpy view over the result "
            "(zero-copy). `list(res.times)` gives the old list.")
        .def_property_readonly("resolution",
            [](py::object self) -> py::object {
                const auto& r = self.cast<const SimulationResult&>();
                if (!r.resolution.enabled) return py::none();
                return py::cast(&r.resolution,
                                py::return_value_policy::reference_internal,
                                self);
            },
            "Resolution diagnostics of a pwl run (None when off, and "
            "for trbdf2/dsed). See pulsim.ResolutionWarning.")
        .def_property_readonly("mode_diagnostics",
            [](const SimulationResult& r) {
                // add-complementarity-mode-selection on pwl: the dsed keys.
                auto rep = [](const solver::ModeReport& m) {
                    py::dict d;
                    d["events"] = m.lcp_events;
                    d["lcp_count"] = m.lcp_events;
                    d["sites"] = py::dict(
                        py::arg("step") = m.sites_step,
                        py::arg("restart") = m.sites_restart,
                        py::arg("substep") = m.sites_substep,
                        py::arg("static") = m.sites_static);
                    d["lcps"] = m.lcps;
                    d["lcp_pivots_total"] = m.pivots_total;
                    d["max_pivots_per_event"] = m.max_pivots_per_event;
                    d["ties"] = m.ties;
                    d["widenings"] = m.widenings;
                    d["capped"] = m.capped;
                    d["leak_pins"] = m.leak_pins;
                    d["pin_judgements"] = m.pin_judgements;
                    d["unfixed"] = m.unfixed;
                    d["enumerations"] = m.enumerations;
                    d["nonunique_modes"] = m.nonunique_modes;
                    d["hysteresis_released"] = m.released;
                    d["shorted_held"] = m.shorted_held;
                    d["shorted_released"] = m.shorted_released;
                    d["settle_rounds_total"] = m.settle_rounds_total;
                    d["mask_cycles"] = m.mask_cycles;
                    d["inconsistent"] = m.inconsistent;
                    d["port_solves"] = m.port_solves;
                    d["port_matrices"] = m.port_matrices;
                    d["quiet_sites"] = m.quiet_sites;
                    d["cached_sites"] = m.cached_sites;
                    d["port_model_problems"] = m.lcps;
                    d["errors"] = m.errors;
                    d["last_error"] = m.last_error;
                    d["passivity"] = py::dict(py::arg("class") = m.passivity,
                                              py::arg("device") = m.passivity_device,
                                              py::arg("kind") = m.passivity_kind);
                    py::list dl;
                    for (const auto& x : m.disagreements) {
                        py::dict e;
                        e["t"] = x.t;
                        e["site"] = x.site;
                        e["settle_mode"] = x.settle_mode;
                        e["lcp_mode"] = x.lcp_mode;
                        e["previous_mode"] = x.previous_mode;
                        e["settle_rounds"] = x.settle_rounds;
                        e["pivots"] = x.pivots;
                        e["status"] = x.status;
                        e["settle_consistent"] = x.settle_consistent;
                        e["lcp_consistent"] = x.lcp_consistent;
                        e["settle_violation"] = x.settle_violation;
                        e["lcp_violation"] = x.lcp_violation;
                        dl.append(e);
                    }
                    d["disagreements"] = dl;
                    return d;
                };
                py::dict out;
                if (r.mode_report.mode == 1) out["mode_shadow"] = rep(r.mode_report);
                if (r.mode_report.mode == 2) out["mode_selection"] = rep(r.mode_report);
                if (r.mode_guard.enabled) {
                    const auto& g = r.mode_guard;
                    py::dict d = rep(g.lcp);
                    d["checks"] = g.checks;
                    d["repairs"] = g.repairs;
                    d["confirmed"] = g.confirmed;
                    d["failed"] = g.failed;
                    d["settle_caps"] = g.settle_caps;
                    d["lcp"] = rep(g.lcp);   // dsed's nesting; the flat keys stay
                    d["repair_times"] = g.repair_times;
                    d["repair_diodes"] = g.repair_diodes;
                    d["repair_violations"] = g.repair_violations;
                    d["passivity"] = rep(r.mode_report)["passivity"];
                    out["mode_guard"] = d;
                }
                return out;
            },
            "add-complementarity-mode-selection on pwl: 'mode_shadow' / "
            "'mode_selection' counters and the default guard's "
            "'mode_guard' (simulate() copies them to res.diagnostics).")
        .def_property_readonly("counters",
            [](const SimulationResult& r) {
                py::dict d;
                d["switch_fn_calls"] = r.switch_fn_calls;
                d["segment_searches"] = r.segment_searches;
                d["rollback_snapshots"] = r.rollback_snapshots;
                d["stage2_retries"] = r.stage2_retries;
                d["stage2_be_kept"] = r.stage2_be_kept;
                d["stage2_retry_failures"] = r.stage2_retry_failures;
                d["unlocated_gate_steps"] = r.unlocated_gate_steps;
                d["split_event_overflows"] = r.split_event_overflows;
                d["n_gate_events"] = r.n_gate_events;
                d["n_gate_events_unresolved"] = r.n_gate_events_unresolved;
                d["n_gate_events_bisected"] = r.n_gate_events_bisected;
                d["split_locator_trials"] = r.split_locator_trials;
                d["split_locator_giveups"] = r.split_locator_giveups;
                d["mode_structure_checks"] = r.mode_structure_checks;
                d["ideal_loop_releases"] = r.ideal_loop_releases;
                return d;
            },
            "Fixed-step loop counters (pwl engine; 0 elsewhere): "
            "`switch_fn_calls` (schedule evaluations in the step loop, "
            "one per distinct solve time), `segment_searches` (segment "
            "lookups not absorbed by the last-segment memo), "
            "`rollback_snapshots` (steps whose pre-step state was "
            "copied for a possible rollback).")
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
        .def_readonly("newton", &SimulationResult::newton,
                      "Newton counters of the run (NewtonCounters).")
        // add-parameter-sensitivities group 5: the stage recurrence's trace
        // (pwl / trbdf2), None when no sensitivities were requested.
        .def_property_readonly("sensitivity_trace",
            [](const SimulationResult& r) -> py::object {
                const auto& tr = r.sensitivity;
                if (tr.params.empty()) return py::none();
                const auto nt = static_cast<py::ssize_t>(tr.times.size());
                const auto np = static_cast<py::ssize_t>(tr.params.size());
                const auto ns = static_cast<py::ssize_t>(tr.n_state);
                py::array_t<Real> st({nt, np, ns});
                std::copy(tr.states.begin(), tr.states.end(), st.mutable_data());
                py::dict d;
                d["params"] = tr.params;
                d["times"] = py::array_t<Real>(nt, tr.times.data());
                d["states"] = st;   // (sample, param, state)
                d["n_sens_solves"] = tr.n_sens_solves;
                d["n_sens_iterations"] = tr.n_sens_iterations;
                d["n_sens_discarded"] = tr.n_sens_discarded;
                return d;
            },
            "Forward parameter sensitivities of a pwl / trbdf2 run "
            "(add-parameter-sensitivities group 5): a dict with 'params', "
            "'times' (the start and every accepted step), 'states' "
            "(sample, param, state) and the counters 'n_sens_solves', "
            "'n_sens_iterations' (of accepted steps), 'n_sens_discarded' "
            "(back-substitutions of rejected attempts); None without "
            "sensitivities.")
        // Per-sample record of the STAMPED switch / diode mask
        // (result.hpp, `switch_state_bits`). Unpacked here into a
        // (num_steps, num_switches) bool array; column j is bit j of
        // the SwitchStateMask, i.e. `builder.switch_index_of(name)`.
        .def_property_readonly("switch_states",
            [](const SimulationResult& r) -> py::object {
                if (!r.switch_states_complete()) {
                    return py::none();
                }
                const auto n = static_cast<py::ssize_t>(r.times.size());
                const auto nb = static_cast<py::ssize_t>(r.num_switch_bits);
                const Size stride = r.switch_state_stride;
                py::array_t<bool> out(std::vector<py::ssize_t>{n, nb});
                auto* ptr = out.mutable_data();
                const std::uint8_t* src = r.switch_state_bits.data();
                for (py::ssize_t k = 0; k < n; ++k) {
                    const std::uint8_t* row =
                        src + static_cast<std::size_t>(k) * stride;
                    for (py::ssize_t j = 0; j < nb; ++j) {
                        ptr[k * nb + j] =
                            ((row[j / 8] >> (j % 8)) & 1U) != 0U;
                    }
                }
                return std::move(out);
            },
            "Per-sample record of the switch / diode mask the solver "
            "STAMPED for the solve that produced states[k], as a "
            "(num_steps, num_switches) bool array (column j = "
            "builder.switch_index_of(name)). A switched branch's "
            "current is exactly (g_on if bit else g_off) * v_branch, "
            "less g_on * V_f for an ON diode with a forward drop V_f. "
            "None when the result carries no record (a result not "
            "produced by a kernel engine). Raw: where "
            "switch_states_unsolved[k] is True (a sample 0 seeded from "
            "initial_state) the row is the bits the solver carried in, "
            "not a solve, and g(bit) * v does not hold there; result.i() "
            "and the loss / thermal summaries read that row with the "
            "mask of the first solve after it. With store_every > 1 a "
            "row is the mask of the LAST solve before the sample, not of "
            "the whole stored interval, so duty cycles and time "
            "integrals over a decimated grid alias. Unpacked on each "
            "access; the kernel stores ceil(num_switches/8) bytes per "
            "sample.")
        .def_property_readonly("gate_states",
            [](const SimulationResult& r) -> py::object {
                const Size stride = r.switch_state_stride;
                if (!r.switch_states_complete() || r.gate_state_bits.empty() ||
                    r.gate_state_bits.size() != r.switch_state_bits.size()) {
                    return py::none();
                }
                const auto n = static_cast<py::ssize_t>(r.times.size());
                const auto nb = static_cast<py::ssize_t>(r.num_switch_bits);
                py::array_t<bool> out(std::vector<py::ssize_t>{n, nb});
                auto* ptr = out.mutable_data();
                for (py::ssize_t k = 0; k < n; ++k) {
                    const std::uint8_t* row = r.gate_state_bits.data()
                        + static_cast<std::size_t>(k) * stride;
                    for (py::ssize_t j = 0; j < nb; ++j) {
                        ptr[k * nb + j] = ((row[j / 8] >> (j % 8)) & 1U) != 0U;
                    }
                }
                return std::move(out);
            },
            "Per-sample record of the COMMANDED gate bit each gated PWL "
            "device (IGBT, GTO, thyristor, TRIAC) was decided under, in "
            "the layout of switch_states (column j = "
            "builder.switch_index_of(name); False for every other "
            "column). switch_states holds the stamped conduction state. "
            "None in a circuit without a gated device.")
        // Release group B: the event limits kept beside the samples (the
        // fixed engine under the restart rule; the exact engine's adapter
        // assigns its own). None: no record.
        .def_property("right_limit_index",
            [](const SimulationResult& r) -> py::object {
                if (!r.has_event_limits) return py::none();
                const auto m = static_cast<py::ssize_t>(r.right_limit_index.size());
                py::array_t<std::int64_t> out(m);
                for (py::ssize_t j = 0; j < m; ++j)
                    out.mutable_data()[j] = static_cast<std::int64_t>(r.right_limit_index[j]);
                out.attr("setflags")(py::arg("write") = false);
                return std::move(out);
            },
            [](SimulationResult& r, py::object v) {
                r.right_limit_index.clear();
                if (v.is_none()) {
                    r.has_event_limits = false;
                    return;
                }
                const auto a = py::array_t<std::int64_t, py::array::forcecast>(v);
                for (py::ssize_t j = 0; j < a.size(); ++j)
                    r.right_limit_index.push_back(static_cast<Size>(a.data()[j]));
                r.has_event_limits = true;
            },
            "Sample indices (increasing) whose RIGHT limit is kept: a "
            "stored sample at an event is its left limit, the next step "
            "starts from right_limit_states[j]. pulsim.with_event_limits(res) "
            "is the trace with both. None: the result keeps no event limits.")
        .def_property("right_limit_states",
            [](const SimulationResult& r) -> py::object {
                if (!r.has_event_limits) return py::none();
                const auto m = static_cast<py::ssize_t>(r.right_limit_index.size());
                const auto w = static_cast<py::ssize_t>(r.event_limit_width);
                if (r.right_limit_states.size() != static_cast<std::size_t>(m * w))
                    throw std::invalid_argument(
                        "SimulationResult: right_limit_states does not hold "
                        "len(right_limit_index) rows of the state width");
                py::array_t<double> out(std::vector<py::ssize_t>{m, w});
                if (m * w > 0)
                    std::copy_n(r.right_limit_states.data(), m * w, out.mutable_data());
                out.attr("setflags")(py::arg("write") = false);
                return std::move(out);
            },
            [](SimulationResult& r, py::object v) {
                r.right_limit_states.clear();
                if (v.is_none()) return;
                const auto a = py::array_t<double, py::array::c_style | py::array::forcecast>(v);
                if (a.ndim() == 2) r.event_limit_width = static_cast<Size>(a.shape(1));
                r.right_limit_states.assign(a.data(), a.data() + a.size());
            },
            "(len(right_limit_index), n_states): the right limit of each "
            "of those samples: the state value one instant after the "
            "event (each capacitor / inductor state continuous, or moved "
            "by an ideal element's jump; the other unknowns solved in the "
            "event's mode). Recorded only at events; on the fixed engine "
            "under the restart rule never at a restart window's later "
            "steps nor at t = 0 unless the start is a probed jump.")
        .def_property("right_limit_switch_states",
            [](const SimulationResult& r) -> py::object {
                if (!r.has_event_limits) return py::none();
                const auto m = static_cast<py::ssize_t>(r.right_limit_index.size());
                const auto nb = static_cast<py::ssize_t>(r.event_limit_switches);
                py::array_t<bool> out(std::vector<py::ssize_t>{m, nb});
                for (py::ssize_t k = 0; k < m * nb; ++k)
                    out.mutable_data()[k] = static_cast<std::size_t>(k) < r.right_limit_switch_bits.size()
                        && r.right_limit_switch_bits[static_cast<std::size_t>(k)] != 0;
                out.attr("setflags")(py::arg("write") = false);
                return std::move(out);
            },
            [](SimulationResult& r, py::object v) {
                r.right_limit_switch_bits.clear();
                if (v.is_none()) return;
                const auto a = py::array_t<bool, py::array::c_style | py::array::forcecast>(v);
                if (a.ndim() == 2) r.event_limit_switches = static_cast<Size>(a.shape(1));
                for (py::ssize_t k = 0; k < a.size(); ++k)
                    r.right_limit_switch_bits.push_back(a.data()[k] ? 1 : 0);
            },
            "(len(right_limit_index), n_switches) bool: the mask each "
            "right limit is under.")
        .def_property_readonly("inner_events",
            [](const SimulationResult& r) -> py::object {
                if (!r.has_event_limits) return py::none();
                const auto m = static_cast<py::ssize_t>(r.inner_event_times.size());
                const auto w = static_cast<py::ssize_t>(r.event_limit_width);
                const auto nb = static_cast<py::ssize_t>(r.event_limit_switches);
                py::array_t<std::int64_t> after(m);
                py::array_t<double> times(m);
                for (py::ssize_t j = 0; j < m; ++j) {
                    after.mutable_data()[j] = static_cast<std::int64_t>(r.inner_event_after[j]);
                    times.mutable_data()[j] = r.inner_event_times[j];
                }
                auto rows = [&](const std::vector<Real>& v) {
                    if (v.size() != static_cast<std::size_t>(m * w))
                        throw std::logic_error("SimulationResult.inner_events: row size");
                    py::array_t<double> out(std::vector<py::ssize_t>{m, w});
                    if (m * w > 0) std::copy_n(v.data(), m * w, out.mutable_data());
                    return out;
                };
                auto bits = [&](const std::vector<std::uint8_t>& v) {
                    if (v.size() != static_cast<std::size_t>(m * nb))
                        throw std::logic_error("SimulationResult.inner_events: mask size");
                    py::array_t<bool> out(std::vector<py::ssize_t>{m, nb});
                    for (py::ssize_t k = 0; k < m * nb; ++k) out.mutable_data()[k] = v[k] != 0;
                    return out;
                };
                py::dict d;
                d["after"] = after;
                d["times"] = times;
                d["left_states"] = rows(r.inner_event_left);
                d["left_switch_states"] = bits(r.inner_event_left_bits);
                d["right_states"] = rows(r.inner_event_right);
                d["right_switch_states"] = bits(r.inner_event_right_bits);
                return std::move(d);
            },
            "Events INSIDE a stored step (the fixed engine's event split "
            "under the restart rule): dict of arrays, per event the sample "
            "it follows ('after'), its time, its left limit (state and "
            "mask of the sub-step that ended there) and its right limit "
            "(a NaN row: none). None: no record. "
            "pulsim.with_event_limits(res) inserts them as rows.")
        .def_property_readonly("switch_states_unsolved",
            [](const SimulationResult& r) -> py::object {
                if (!r.switch_states_complete()) {
                    return py::none();
                }
                const auto n = static_cast<py::ssize_t>(r.times.size());
                py::array_t<bool> out(n);
                auto* ptr = out.mutable_data();
                std::fill(ptr, ptr + n, false);
                for (const Size k : r.switch_state_unsolved) {
                    if (static_cast<py::ssize_t>(k) < n) {
                        ptr[k] = true;
                    }
                }
                return std::move(out);
            },
            "Per-sample bool: True where states[k] was NOT produced by "
            "a solve under switch_states[k] — today only a sample 0 "
            "seeded from initial_state, whose mask is the bits the "
            "solver carried in (result.i() and the loss / thermal "
            "summaries read that row with the first solve's mask). A "
            "resume_from sample 0 carries the bits of the solve that "
            "produced it and is not flagged. None without a record.")
        .def_property_readonly("switch_states_bytes",
            [](const SimulationResult& r) {
                return r.switch_state_bits.size() +
                       r.switch_state_unsolved.size() * sizeof(Size);
            },
            "Bytes held by the stamped-mask record.")
        .def_readonly("commutation_events",
                       &SimulationResult::commutation_events)
        .def_readonly("_device_losses", &SimulationResult::device_losses,
                       "add-in-kernel-event-losses: one DeviceLossRecord per "
                       "loss port (read with loss_events / device_energy).")
        .def("_set_device_losses",
             [](SimulationResult& r, std::vector<thermal::DeviceLossRecord> recs,
                Real injected, Real pending) {
                 r.device_losses = std::move(recs);
                 r.heat_injected = injected;
                 r.heat_pending = pending;
             },
             py::arg("records"), py::arg("heat_injected"), py::arg("heat_pending"),
             "The exact engine's loss records (dsed loss port).")
        .def("_set_final_snapshot",
             [](SimulationResult& r, SolverSnapshot s) { r.final_snapshot = std::move(s); },
             py::arg("snapshot"),
             "The exact engine's end-of-run snapshot (dsed).")
        .def_readonly("heat_injected", &SimulationResult::heat_injected,
                       "J of heat the run deposited into thermal nodes.")
        .def_readonly("heat_pending", &SimulationResult::heat_pending,
                       "J of heat recorded in the last step and not yet "
                       "deposited (the step that would carry it is past "
                       "t_end).")
        .def_readonly("reinit_records",
                       &SimulationResult::reinit_records,
                       "Impulsive re-initialisations of dependent modes "
                       "(add-branch-form-ideal-switch D4).")
        .def_readonly("inductor_guard_actions",
                       &SimulationResult::inductor_guard_actions,
                       "Inductors whose current a post-solve guard "
                       "overwrote. Empty unless a guard was enabled "
                       "AND fired.")
        .def_readonly("dt_retries", &SimulationResult::dt_retries,
                       "Steps that had to be re-taken at a reduced "
                       "dt. Empty on a run that never needed one.")
        .def_readonly("n_restart_steps", &SimulationResult::n_restart_steps,
                       "Steps taken as SDIRK(2,2) restart steps "
                       "(discontinuity_restart). 0 with the rule off.")
        .def_readonly("n_restart_factorizations",
                       &SimulationResult::n_restart_factorizations,
                       "Factorisations made for the restart stages: one "
                       "per mask at 2*gamma*dt (more only if the segment "
                       "byte budget evicts), plus every refactor for a "
                       "restart stage of another size (event-split "
                       "sub-intervals, the stage-2 re-try). 0 with the "
                       "rule off.")
        .def_readonly("kernel_tick_counts",
                       &SimulationResult::control_tick_counts,
                       "Ticks delivered per tick_fns source (run_transient "
                       "tick_periods=); simulate() copies them into "
                       "control_tick_counts.")
        .def_readonly("kernel_tick_jitter_max",
                       &SimulationResult::control_tick_jitter_max,
                       "Largest step time minus scheduled instant of a "
                       "kernel tick (0 when every period is a multiple of "
                       "dt).")
        .def_readonly("n_split_steps", &SimulationResult::n_split_steps,
                       "Steps split at a gate edge, source jump or diode "
                       "crossing inside them "
                       "(enable_substep_state_correction). 0 with the "
                       "split off.")
        .def_readonly("n_right_limit_fallbacks",
                       &SimulationResult::n_right_limit_fallbacks,
                       "Restart rule: event right limits that are not "
                       "certified: a row whose solve did not converge or "
                       "whose diode mode does not hold at t_e+ (no row is "
                       "stored), or a located turn-off solved through g_off "
                       "(the row is stored). 0 with the rule off.")
        .def_readonly("event_iteration_breaches",
                       &SimulationResult::event_iteration_breaches)
        .def_readonly("newton_roundoff_accepts",
                       &SimulationResult::newton_roundoff_accepts,
                       "Transient Newton solves accepted at roundoff "
                       "level (stagnated dx, row backward error <= "
                       "8 eps) instead of aborting.")
        .def_readonly("newton_roundoff_omega_max",
                       &SimulationResult::newton_roundoff_omega_max)
        .def_readonly("newton_roundoff_row",
                       &SimulationResult::newton_roundoff_row)
        .def("num_steps", &SimulationResult::num_steps)
        .def("empty", &SimulationResult::empty, "True when nothing was recorded.")
        // add-probes-and-recorders D3: one result type on every engine.
        .def(py::init<>(),
             "An empty result. Kernel adapters fill it (`_adapter_fill`, "
             "`_set_run_info`); every engine returns this type.")
        .def("_adapter_fill",
            [](SimulationResult& r,
               py::array_t<Real, py::array::c_style | py::array::forcecast> t,
               py::array_t<Real, py::array::c_style | py::array::forcecast> x,
               py::object sw, py::list reinit) {
                if (!r.times.empty() || r.states.rows() != 0) {
                    throw std::logic_error("_adapter_fill: the result is already filled");
                }
                if (t.ndim() != 1 || x.ndim() != 2
                    || (t.shape(0) != 0 && x.shape(0) != t.shape(0))) {
                    throw std::invalid_argument(
                        "_adapter_fill: times must be 1-D and states 2-D with one row per time");
                }
                const auto rows = static_cast<Size>(t.shape(0));
                const auto cols = static_cast<Index>(x.shape(1));
                r.times.assign(t.data(), t.data() + rows);
                r.states.set_state_size(cols);
                r.states.reserve(rows);
                for (Size k = 0; k < rows; ++k) {
                    r.states.push_back(Eigen::Map<const Vector>(
                        x.data() + k * static_cast<Size>(cols), cols));
                }
                if (!sw.is_none()) {
                    auto b = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(sw);
                    if (!b || b.ndim() != 2 || static_cast<Size>(b.shape(0)) != rows) {
                        throw std::invalid_argument(
                            "_adapter_fill: switch_states must be (len(times), n_switches) bool");
                    }
                    const auto nb = static_cast<Size>(b.shape(1));
                    r.begin_switch_state_record(nb, rows);
                    const bool* src = b.data();
                    for (Size k = 0; k < rows; ++k) {
                        for (Size byte = 0; byte < r.switch_state_stride; ++byte) {
                            std::uint8_t v = 0;
                            for (Size j = byte * 8; j < std::min(nb, byte * 8 + 8); ++j) {
                                v |= static_cast<std::uint8_t>(src[k * nb + j] ? 1U : 0U) << (j % 8);
                            }
                            r.switch_state_bits.push_back(v);
                        }
                    }
                }
                for (const auto& item : reinit) {
                    const auto d = item.cast<py::dict>();
                    ReinitRecord rr;
                    rr.t = d["t"].cast<Real>();
                    rr.kind = d["kind"].cast<std::string>();
                    rr.devices = d["devices"].cast<std::vector<std::string>>();
                    rr.states = d["states"].cast<std::vector<std::string>>();
                    rr.lambda = d["lambda"].cast<Real>();
                    rr.dW = d["dW"].cast<Real>();
                    rr.projection_gap = d["projection_gap"].cast<Real>();
                    rr.sample = d["sample"].cast<long>();
                    if (d.contains("cause")) rr.cause = d["cause"].cast<std::string>();
                    r.reinit_records.push_back(std::move(rr));
                }
            },
            "times"_a, "states"_a, "switch_states"_a = py::none(),
            "reinit_records"_a = py::list(),
            "Adapter use only: fill an empty result from arrays (times, "
            "states, the per-sample mask, re-initialisation dicts).")
        .def("_set_run_info",
            [](SimulationResult& r, const std::string& engine, record::GridKind grid,
               const py::dict& d) {
                r.grid = grid;
                auto& s = r.stats;
                s = record::RunStats{};
                s.engine = engine;
                // decltype, not `Size record::RunStats::*`: MSVC rejects that spelling in a template argument
                using SizeMember = decltype(&record::RunStats::n_steps);
                using RealMember = decltype(&record::RunStats::gate_pulse_min);
                const std::pair<const char*, SizeMember> sizes[] = {
                    {"n_steps", &record::RunStats::n_steps},
                    {"n_accept", &record::RunStats::n_accept},
                    {"n_reject", &record::RunStats::n_reject},
                    {"n_solves", &record::RunStats::n_solves},
                    {"n_factorizations", &record::RunStats::n_factorizations},
                    {"n_gate_events", &record::RunStats::n_gate_events},
                    {"n_state_events", &record::RunStats::n_state_events},
                    {"n_mode_events", &record::RunStats::n_mode_events},
                    {"n_ctrl_ticks", &record::RunStats::n_ctrl_ticks},
                    {"n_newton_iterations", &record::RunStats::n_newton_iterations},
                    {"n_newton_retries", &record::RunStats::n_newton_retries},
                    {"n_forced_accepts", &record::RunStats::n_forced_accepts},
                    {"n_chatter_breaks", &record::RunStats::n_chatter_breaks},
                    {"n_gate_undersampled", &record::RunStats::n_gate_undersampled},
                    {"n_rk45_steps", &record::RunStats::n_rk45_steps},
                    {"n_bdf2_steps", &record::RunStats::n_bdf2_steps},
                    {"n_exact_steps", &record::RunStats::n_exact_steps}};
                const std::pair<const char*, RealMember> reals[] = {
                    {"gate_pulse_min", &record::RunStats::gate_pulse_min},
                    {"gate_sample_spacing_max", &record::RunStats::gate_sample_spacing_max},
                    {"cpu_time_seconds", &record::RunStats::cpu_time_seconds}};
                std::size_t used = 0;
                for (const auto& [k, m] : sizes) {
                    if (d.contains(k)) { s.*m = d[k].cast<Size>(); ++used; }
                }
                for (const auto& [k, m] : reals) {
                    if (d.contains(k)) { s.*m = d[k].cast<Real>(); ++used; }
                }
                if (used != d.size()) {
                    throw std::invalid_argument("_set_run_info: unknown RunStats key");
                }
            },
            "engine"_a, "grid"_a, "stats"_a,
            "Adapter use only: set the grid kind and the run statistics.")
        .def_readonly("grid", &SimulationResult::grid,
            "GridKind of the rows: Uniform (pwl), Accepted (trbdf2, dsed) "
            "or Requested. A plain .mean() over an Accepted grid is not a "
            "time average: use pulsim.time_average.")
        .def_readonly("stats", &SimulationResult::stats,
            "RunStats: engine-independent counters (zero where the engine "
            "does not report one).");

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
        "Result of `load_yaml_*`: a populated `CircuitBuilder` "
        "(`.builder`) + the `simulation:` block (`.options`).\n\n"
        "Read-only conveniences: `source`, `has_simulation`, "
        "`t_start`, `t_end`, `dt`, `switch_names`, `node_names`, "
        "`warnings`, `simulation_keys`. `pulsim.simulate(loaded, ...)` "
        "takes every `simulation:` key from it; a keyword you pass wins.")
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
                        &yaml::LoadedCircuit::options)
        .def_readonly("simulation_keys",
                       &yaml::LoadedCircuit::simulation_keys,
                       "The known `simulation:` keys the YAML sets, in "
                       "file order; `simulate(loaded)` forwards these.")
        .def_property_readonly("source",
            [](const yaml::LoadedCircuit& self) { return self.source; },
            "Where the YAML came from: the path given to "
            "`load_yaml_file`, or '<string>' for `load_yaml_string`.")
        .def_property_readonly("has_simulation",
            [](const yaml::LoadedCircuit& self) {
                return self.has_simulation;
            },
            "True when the YAML has a top-level `simulation:` block. "
            "When False, `t_end` and `dt` are 0.0 (not set) and "
            "`simulate(loaded)` refuses until you pass `t_end`.")
        .def_property_readonly("t_start",
            [](const yaml::LoadedCircuit& self) {
                return self.options.t_start;
            },
            "`simulation.t_start` (s); 0.0 when not set. Same value "
            "as `options.t_start`.")
        .def_property_readonly("t_end",
            [](const yaml::LoadedCircuit& self) {
                return self.options.t_end;
            },
            "`simulation.t_end` (s); 0.0 when the YAML does not set "
            "it. Same value as `options.t_end`.")
        .def_property_readonly("dt",
            [](const yaml::LoadedCircuit& self) {
                return self.options.dt;
            },
            "`simulation.dt` (s); 0.0 when the YAML does not set it "
            "(then `simulate(loaded)` runs without a dt, i.e. on the "
            "variable-step engine where it qualifies). Same value as "
            "`options.dt`.")
        .def_property_readonly("switch_names",
            [](const yaml::LoadedCircuit& self) {
                const auto& g = self.builder.graph();
                std::vector<std::string> names;
                for (Index i = 0; i < g.num_branches(); ++i) {
                    if (g.branch(i).kind
                            == topology::BranchKind::Switch) {
                        names.emplace_back(g.branch_name(i));
                    }
                }
                return names;
            },
            "Names of the switching branches in SwitchStateMask bit "
            "order: `switch_names[k]` is the device at bit k, i.e. "
            "`builder.switch_index_of(switch_names[k]) == k`. "
            "Includes the generated ones (a MOSFET's '<name>_body' "
            "diode, binary diodes).")
        .def_property_readonly("node_names",
            [](const yaml::LoadedCircuit& self) {
                const auto& g = self.builder.graph();
                std::vector<std::string> names;
                names.reserve(static_cast<std::size_t>(g.num_nodes()));
                for (Index i = 0; i < g.num_nodes(); ++i) {
                    names.push_back(g.node(i).name);
                }
                return names;
            },
            "Non-ground node names in node-id order (the names "
            "`result.v(name)` accepts). Ground ('gnd' / '0') is not "
            "listed.")
        .def_property_readonly("warnings",
            [](const yaml::LoadedCircuit& self) {
                return self.warnings;
            },
            "Non-fatal loader findings (unknown keys), "
            "each also emitted as a UserWarning at load time.")
        .def("__repr__",
            [](const yaml::LoadedCircuit& self) {
                return std::format(
                    "LoadedCircuit(source='{}', has_simulation={}, "
                    "t_end={}, dt={}, nodes={}, switches={})",
                    self.source, self.has_simulation ? "True" : "False",
                    self.options.t_end, self.options.dt,
                    self.builder.graph().num_nodes(),
                    self.builder.graph().num_switches());
            });

    // Re-emit the loader's non-fatal findings as Python UserWarnings.
    // Under `-W error` PyErr_WarnEx returns -1 and the warning becomes
    // the exception, as Python users expect.
    const auto emit_load_warnings = [](const yaml::LoadedCircuit& lc) {
        for (const auto& w : lc.warnings) {
            const std::string msg = std::format(
                "load_yaml ({}): {}", lc.source, w);
            if (PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1) != 0) {
                throw py::error_already_set();
            }
        }
    };

    // add-external-device-abi 1.7: the libraries a netlist may load. A
    // list of paths (files or directories); True is refused on purpose: a
    // netlist from an untrusted source must not choose the library.
    auto native_allow = [](const py::object& a) {
        yaml::LoadOptions o;
        if (a.is_none() || (py::isinstance<py::bool_>(a) && !a.cast<bool>())) return o;
        if (py::isinstance<py::bool_>(a)) {
            throw py::value_error(
                "allow_native_code=True is not accepted: pass the library paths or "
                "directories the netlist may load, allow_native_code=[path, ...] "
                "(a netlist must not choose the native code it runs)");
        }
        const py::module_ os = py::module_::import("os");
        if (py::isinstance<py::str>(a) || py::isinstance(a, os.attr("PathLike"))) {
            throw py::type_error("allow_native_code must be a list of paths, not one path");
        }
        for (const auto& e : a) o.allow_native_code.push_back(os.attr("fspath")(e).cast<std::string>());
        return o;
    };
    m.def("load_yaml_string",
        [emit_load_warnings, native_allow](const std::string& yaml_text,
                                           const py::object& allow_native_code) {
            auto lc = yaml::load_string(yaml_text, native_allow(allow_native_code));
            emit_load_warnings(lc);
            return lc;
        }, py::arg("yaml_text"), py::kw_only(), py::arg("allow_native_code") = py::none(),
        "Parse a YAML circuit description from an in-memory "
        "string. Returns LoadedCircuit(builder, options). Warns on "
        "unknown top-level / `circuit:` / `simulation:` keys; refuses "
        "a `simulation:` block indented under `circuit:`. "
        "`allow_native_code`: the shared libraries (paths or directories, "
        "symlinks followed) an `external_device` entry may load; without "
        "it every such entry is refused (a relative `lib:` is taken from "
        "the working directory).");
    m.def("load_yaml_file",
        [emit_load_warnings, native_allow](const std::string& path,
                                           const py::object& allow_native_code) {
            auto lc = yaml::load_file(path, native_allow(allow_native_code));
            emit_load_warnings(lc);
            return lc;
        }, py::arg("path"), py::kw_only(), py::arg("allow_native_code") = py::none(),
        "Parse a YAML circuit description from disk. "
        "Returns LoadedCircuit(builder, options); errors and "
        "warnings name the path. Same key checks as "
        "`load_yaml_string`; a relative `lib:` of an `external_device` "
        "is taken from the file's folder.");

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

    py::class_<sources::SwitchSchedule>(m, "SwitchSchedule",
        "A switch_fn that knows its own gate edges: returned by "
        "make_pwm_switch_fn, make_dead_time_pwm_pair_fn, "
        "make_spwm_pair_fn, make_phase_shift_full_bridge_fn, "
        "make_three_phase_spwm_fn and make_combined_switch_fn. "
        "Call it with t for the SwitchStateMask. "
        "`next_edge_after(t)` is the first gate edge strictly "
        "after t (inf when unknown), so the variable-step engines "
        "(engine='trbdf2', 'dsed') land their steps ON the edges "
        "instead of sampling for them; every engine evaluates it "
        "in C++ without the GIL. Edges that fall on a dt grid "
        "point take effect at that point.")
        .def("__call__",
              [](const sources::SwitchSchedule& self, Real t) {
                  return self(t);
              },
              py::arg("t"),
              "Return the switch mask at time `t`.")
        .def("next_edge_after",
              &sources::SwitchSchedule::next_edge_after,
              py::arg("t"),
              "First gate-edge time strictly greater than `t`; inf "
              "when the schedule does not know it (a combination "
              "with a plain Python callable). An edge may be a no-op "
              "(same mask on both sides), never a missed one.")
        .def_property_readonly("has_edges",
              &sources::SwitchSchedule::has_edges,
              "False when the schedule cannot name its edges (a "
              "make_combined_switch_fn with a plain Python callable "
              "in it); the variable-step engines then search for "
              "them by sampling.")
        .def_property_readonly("follows_controller",
              [](const sources::SwitchSchedule& s) {
                  return s.follows_controller;
              },
              "True when a part is a control graph's PWM gate (a "
              "closed loop): its edges are those of the duty held "
              "now, so shooting and sensitivities refuse it.");

    m.def("make_pwm_switch_fn",
        &sources::make_pwm_switch_fn,
        py::arg("frequency"), py::arg("duty"),
        py::arg("switch_idx"), py::arg("num_switches"),
        py::arg("phase") = 0.0,
        "Build a SwitchSchedule that toggles `switch_idx` "
        "ON during the first `duty · T` fraction of each "
        "period (T = 1 / frequency) and OFF for the rest; "
        "`phase` is a time offset in seconds. All other switch "
        "bits stay OFF. Edges on a dt grid point take effect "
        "at that point, so an on-time of a whole number of "
        "steps is exactly that many steps in every period. "
        "Usable as `switch_fn=` in simulate / run_transient.");

    // add-unified-event-queue task 6.1: the modulator primitives.
    py::class_<events::DutyRegister, std::shared_ptr<events::DutyRegister>>(
        m, "DutyRegister",
        "A duty a controller writes (`write(value, t)`) and a "
        "make_carrier_pwm_fn modulator reads. latch='immediate' takes a "
        "write at once; latch='carrier_start' holds it until the next "
        "start of the carrier (period, phase), which must equal the "
        "modulator's.")
        .def(py::init([](Real value, const std::string& latch, Real period,
                         Real phase) {
                 events::DutyRegister r{value};
                 if (latch == "carrier_start") {
                     if (!(period > Real{0})) {
                         throw std::invalid_argument(
                             "DutyRegister: latch='carrier_start' needs period > 0");
                     }
                     r.latch = events::LatchPolicy::AtCarrierStart;
                     r.T = period;
                     r.phase = phase;
                 } else if (latch != "immediate") {
                     throw std::invalid_argument(
                         "DutyRegister: latch must be 'immediate' or "
                         "'carrier_start'; got '" + latch + "'");
                 }
                 return std::make_shared<events::DutyRegister>(r);
             }),
             py::arg("value") = 0.0, py::arg("latch") = "immediate",
             py::arg("period") = 0.0, py::arg("phase") = 0.0)
        .def("write", &events::DutyRegister::write, py::arg("value"), py::arg("t"),
             "Write a duty; `t` is the tick's scheduled instant.")
        .def("read", &events::DutyRegister::read, py::arg("t"),
             "The duty in force at t.")
        .def_property_readonly("value",
             [](const events::DutyRegister& r) { return r.value; },
             "The latched duty (a 'carrier_start' write not yet latched "
             "is not in it).");

    m.def("make_carrier_pwm_fn",
        [](Real period, std::shared_ptr<events::DutyRegister> duty,
           Index switch_idx, Size num_switches, Index low_side, Real dead_time,
           Real phase) {
            if (!duty) throw std::invalid_argument("make_carrier_pwm_fn: duty is None");
            events::GateSet set(num_switches);
            set.add(events::CarrierPwm{period, phase, duty.get(), switch_idx,
                                       low_side, dead_time}, "carrier_pwm");
            const auto s = set.to_schedule();
            // the schedule holds the register: it lives as long as the gate
            sources::SwitchSchedule out{
                [s, duty](Real t) { return s(t); },
                [s, duty](Real t) { return s.next_edge_after(t); }};
            out.follows_controller = true;
            return out;
        },
        py::arg("period"), py::arg("duty"), py::arg("switch_idx"),
        py::arg("num_switches"), py::arg("low_side") = -1,
        py::arg("dead_time") = 0.0, py::arg("phase") = 0.0,
        "A SwitchSchedule: a sawtooth carrier of `period` (starting at "
        "`phase` seconds) compared with the DutyRegister `duty`; "
        "`switch_idx` is ON while the carrier is below the duty. With "
        "`low_side` >= 0 a complementary leg with `dead_time`. For a "
        "constant duty it equals make_pwm_switch_fn / "
        "make_dead_time_pwm_pair_fn exactly. Its edges are those of the "
        "duty held now: write the register from a controller tick "
        "(closed_loops / a step_observer with a period) and every engine "
        "re-asks the edge after the tick.");

    m.def("make_edge_list_fn",
        [](const std::vector<std::tuple<Real, Index, bool>>& edges,
           Size num_switches) {
            std::vector<events::EdgeList::Edge> e;
            e.reserve(edges.size());
            for (const auto& [t, bit, on] : edges) e.push_back({t, bit, on});
            events::GateSet set(num_switches);
            set.add(events::EdgeList{std::move(e)}, "edge_list");
            return set.to_schedule();
        },
        py::arg("edges"), py::arg("num_switches"),
        "A SwitchSchedule from explicit gate edges (t, switch_idx, on): "
        "the bit takes `on` at t and holds it until its next edge, OFF "
        "before its first. Engines land every edge exactly.");

    m.def("make_phase_control_fn",
        &sources::make_phase_control_fn,
        py::arg("frequency"), py::arg("alpha"), py::arg("width"),
        py::arg("switch_indices"), py::arg("num_switches"),
        py::arg("phase") = 0.0,
        "Firing schedule of a p-pulse line-commutated converter "
        "(p = len(switch_indices)): gate k is ON for `width` seconds "
        "from (phase + alpha + k*2*pi/p) / (2*pi*frequency), every line "
        "period. alpha, phase in radians; phase puts gate 0's natural "
        "commutation instant (alpha = 0) on the line. A SwitchSchedule "
        "(edges known to every engine).");

    m.def("make_constant_switch_fn",
        [](const topology::SwitchStateMask& mask) {
            return sources::constant_schedule(mask);
        },
        py::arg("mask"),
        "A SwitchSchedule that returns `mask` at every time (no "
        "edges). Evaluated in C++: simulate() uses it for the pwl "
        "engine's default when no switch_fn is given.");

    m.def("make_dead_time_pwm_pair_fn",
        &sources::make_dead_time_pwm_pair_fn,
        py::arg("frequency"), py::arg("duty"),
        py::arg("hs_switch_idx"), py::arg("ls_switch_idx"),
        py::arg("num_switches"), py::arg("dead_time"),
        py::arg("phase") = 0.0,
        "Build a SwitchSchedule driving a complementary "
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
        "Build a SwitchSchedule driving a HS/LS half-"
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
        [](Size num_switches, const std::vector<py::object>& fns) {
            // A SwitchSchedule goes in by value, so the combination
            // keeps its edge oracle (and its C++ call path); any
            // other callable goes through pybind11's std::function
            // adapter, which makes the combined edges unknown.
            std::vector<SwitchScheduleFn> cpp_fns;
            cpp_fns.reserve(fns.size());
            for (const auto& fn : fns) {
                if (fn.is_none()) {
                    cpp_fns.emplace_back();
                } else if (py::isinstance<sources::SwitchSchedule>(fn)) {
                    cpp_fns.emplace_back(
                        fn.cast<sources::SwitchSchedule>());
                } else if (py::isinstance<pulsim::blockchain::ChainPwmGate>(fn)) {
                    const auto g = fn.cast<pulsim::blockchain::ChainPwmGate>();
                    sources::SwitchSchedule s{
                        g, [g](Real t) { return g.next_edge_after(t); }};
                    s.follows_controller = true;
                    cpp_fns.emplace_back(std::move(s));
                } else if (py::isinstance<pulsim::blockchain::ChainSignalGate>(fn)) {
                    const auto g = fn.cast<pulsim::blockchain::ChainSignalGate>();
                    sources::SwitchSchedule s{
                        g, [g](Real t) { return g.next_edge_after(t); }};
                    s.follows_controller = true;
                    cpp_fns.emplace_back(std::move(s));
                } else {
                    cpp_fns.emplace_back(fn.cast<SwitchScheduleFn>());
                }
            }
            py::object out = py::cast(sources::make_combined_switch_fn(
                num_switches, std::move(cpp_fns)));
            using Ret = py::typing::Union<sources::SwitchSchedule>;  // stub type
            // a ChainPwmGate (or a chain's SwitchSchedule) is copied by value
            // but reads its chain through a pointer: each part (which pins
            // its chain) must outlive the combination, not just the list
            // (clearing the list freed the chain)
            for (const auto& fn : fns)
                if (!fn.is_none()) py::detail::keep_alive_impl(out, fn);
            return py::reinterpret_steal<Ret>(out.release());
        },
        py::arg("num_switches"), py::arg("fns"),
        "Compose multiple switch_fn callbacks into a single "
        "SwitchSchedule whose mask is the bitwise OR of all "
        "sub-masks. Useful for combining e.g. a primary-"
        "side PWM helper with a sync-rectifier helper "
        "where each drives different switch indices. Its "
        "`next_edge_after` is the earliest edge of the parts "
        "when every part is a SwitchSchedule, else inf.");

    m.def("make_phase_shift_full_bridge_fn",
        &sources::make_phase_shift_full_bridge_fn,
        py::arg("switching_frequency"),
        py::arg("phase_shift"),
        py::arg("leg_a_hs_idx"), py::arg("leg_a_ls_idx"),
        py::arg("leg_b_hs_idx"), py::arg("leg_b_ls_idx"),
        py::arg("num_switches"), py::arg("dead_time"),
        py::arg("carrier_phase") = 0.0,
        "Build a SwitchSchedule driving a phase-shift "
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
        "Build a SwitchSchedule driving a 3-phase 2-level "
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

    snapshot_cls
        .def(py::init<>())
        .def_readwrite("t", &SolverSnapshot::t)
        .def_readwrite("x", &SolverSnapshot::x)
        .def_readwrite("dev_state", &SolverSnapshot::dev_state,
            "Every device state the run carries, one flat vector in the "
            "blocks named by ``dev_layout_names`` (linear companion history, "
            "then the stateful-device histories).")
        .def_readonly("dev_layout", &SolverSnapshot::dev_layout,
            "Instance count per block of ``dev_state``; empty for a 2.0-style "
            "payload.")
        .def_readonly("layout_signature", &SolverSnapshot::layout_signature,
            "Hash of the block names and counts; a resume refuses a "
            "snapshot whose layout is not the circuit's, naming the block.")
        .def_property_readonly("dev_layout_names",
            [](const SolverSnapshot&) {
                std::vector<std::string> n;
                for (const auto& b : device::kStateBlocks) n.emplace_back(b.name);
                return n;
            },
            "The block names of ``dev_state``, in layout order.")
        .def_readwrite("mask", &SolverSnapshot::mask,
            "The solver-owned mode bits the next step starts from (which "
            "switched diodes conduct).")
        .def_property("diode_on",
            [](const SolverSnapshot& s) { return s.mask; },
            [](SolverSnapshot& s, std::vector<bool> v) { s.mask = std::move(v); },
            "Deprecated name of ``mask``.")
        .def_readwrite("recorder_state", &SolverSnapshot::recorder_state,
            "The in-loop recorders' state at the snapshot (pickle field 18; "
            "simulate(recorders=..., resume_from=) continues them). Empty: no "
            "recorder fed in the loop, or an older snapshot.")
        .def_readwrite("recorder_spec", &SolverSnapshot::recorder_spec,
            "Signature of the recorders that wrote recorder_state.")
        .def_readwrite("line_state", &SolverSnapshot::line_state,
            "The transmission lines' committed past (pickle field 19, "
            "present only with a line): per line Z0, delay, n, then n x "
            "(t, w_a, w_b). Empty: no line.")
        .def_readwrite("tick_origin", &SolverSnapshot::tick_origin,
            "Per control-tick source, in registration order: the origin of "
            "its grid (ticks at origin + offset + k*period). simulate("
            "resume_from=) passes it back so the resumed ticks stay on the "
            "run's grid. Empty: no ticks, or an older snapshot.")
        .def_readwrite("tick_next_k", &SolverSnapshot::tick_next_k,
            "Per control-tick source: the index k of its next instant.")
        .def_readwrite("stamped_mask", &SolverSnapshot::stamped_mask,
            "The switch / diode mask stamped for the solve that "
            "produced ``x`` (one bool per SwitchStateMask bit): the "
            "previous run's last ``switch_states`` row. A resume "
            "records it as its row 0. Not ``diode_on`` + "
            "``switch_fn(t)``: a variable-step run that ends on a gate "
            "edge stores that sample under the pre-edge mask and only "
            "then re-decides the diodes. Empty when unknown (a "
            "hand-built or older snapshot); row 0 is then flagged in "
            "``switch_states_unsolved``.")
        .def_property("history",
            [](const SolverSnapshot& s) {
                return s.view(device::StateBlock::LinearCompanion);
            },
            [](SolverSnapshot& s, std::vector<Real> v) {
                s.set_view(device::StateBlock::LinearCompanion, std::move(v));
            },
            "Deprecated view of one block of ``dev_state`` (the 2.0 field). "
            "Writing it turns the snapshot into a 2.0-style payload.")
        .def_property("saturable_history",
            [](const SolverSnapshot& s) {
                return s.view(device::StateBlock::SaturableInductor);
            },
            [](SolverSnapshot& s, std::vector<Real> v) {
                s.set_view(device::StateBlock::SaturableInductor, std::move(v));
            },
            "Deprecated view of one block of ``dev_state`` (the 2.0 field). "
            "Writing it turns the snapshot into a 2.0-style payload.")
        .def_property("coss_history",
            [](const SolverSnapshot& s) {
                return s.view(device::StateBlock::NonlinearCapacitor);
            },
            [](SolverSnapshot& s, std::vector<Real> v) {
                s.set_view(device::StateBlock::NonlinearCapacitor, std::move(v));
            },
            "Deprecated view of one block of ``dev_state`` (the 2.0 field). "
            "Writing it turns the snapshot into a 2.0-style payload.")
        .def_property("lauritzen_history",
            [](const SolverSnapshot& s) {
                return s.view(device::StateBlock::LauritzenDiode);
            },
            [](SolverSnapshot& s, std::vector<Real> v) {
                s.set_view(device::StateBlock::LauritzenDiode, std::move(v));
            },
            "Deprecated view of one block of ``dev_state`` (the 2.0 field). "
            "Writing it turns the snapshot into a 2.0-style payload.")
        .def_property("igbt_tail_history",
            [](const SolverSnapshot& s) {
                return s.view(device::StateBlock::IgbtTail);
            },
            [](SolverSnapshot& s, std::vector<Real> v) {
                s.set_view(device::StateBlock::IgbtTail, std::move(v));
            },
            "Deprecated view of one block of ``dev_state`` (the 2.0 field). "
            "Writing it turns the snapshot into a 2.0-style payload.")
        .def_property("pmsm_history",
            [](const SolverSnapshot& s) {
                return s.view(device::StateBlock::PmsmMna);
            },
            [](SolverSnapshot& s, std::vector<Real> v) {
                s.set_view(device::StateBlock::PmsmMna, std::move(v));
            },
            "Deprecated view of one block of ``dev_state`` (the 2.0 field). "
            "Writing it turns the snapshot into a 2.0-style payload.")
        .def_readwrite("valid", &SolverSnapshot::valid)
        .def_readwrite("invalid_reason", &SolverSnapshot::invalid_reason,
            "Why the run that produced this snapshot left it invalid, when it "
            "knows (a route that keeps no snapshot); resume_from refuses it with "
            "this cause. Empty for a valid snapshot. Not pickled.")
        .def_readwrite("grid_origin", &SolverSnapshot::grid_origin,
            "The fixed-step grid ``t`` sits on: t == grid_origin + "
            "grid_step * grid_dt to rounding (NaN: none). A fixed-step resume "
            "with the same dt continues it, so its trace is the continuous "
            "run's bit for bit.")
        .def_readwrite("grid_step", &SolverSnapshot::grid_step,
            "Index of ``t`` on the fixed-step grid (see ``grid_origin``).")
        .def_readwrite("grid_dt", &SolverSnapshot::grid_dt,
            "Step of the fixed-step grid (see ``grid_origin``); NaN: none.")
        .def_readwrite("post_edge_derivatives",
            &SolverSnapshot::post_edge_derivatives,
            "True when a variable-step run ended exactly on a gate edge "
            "after refreshing the device derivatives for the post-edge "
            "circuit (``x`` is the pre-edge solve). engine='pwl' refuses "
            "to resume from it; engine='trbdf2' resumes.")
        .def_property_readonly("controller_state",
            [](const SolverSnapshot& s) -> py::object {
                if (s.controller.empty()) return py::none();
                return py::make_tuple(static_cast<int>(s.controller.kind),
                                      s.controller.v);
            },
            "The step-size controller's state (step::ControllerState): "
            "``(kind, values)`` with kind 1 = fixed, 2 = LTE (TR-BDF2; "
            "layout [h, h_mem_gate0, h_mem_gate1, pending_gate_slot, "
            "diode_slot, gate_parity, h_mem_diode...]), 3 = the exact "
            "engine's committed instant (reduced state, schedule position, "
            "step memory; engine='dsed' resumes from it, the others refuse "
            "it), or None: the resumed engine then restarts its controller.")
        .def_readwrite("restart_steps_left",
            &SolverSnapshot::restart_steps_left,
            "discontinuity_restart: SDIRK restart steps still owed when "
            "the run ended (the rule takes a discontinuity's step and the "
            "next two); a resumed run takes them first. 0 = none.")
        .def_property("_control_state",
            [](const SolverSnapshot& s) -> py::object {
                if (!s.control_state) return py::none();
                return *static_cast<py::object*>(s.control_state.get());
            },
            [](SolverSnapshot& s, py::object v) {
                if (v.is_none()) { s.control_state.reset(); return; }
                // freed wherever the last copy dies: take the GIL there
                s.control_state = std::shared_ptr<void>(
                    new py::object(std::move(v)), [](void* p) {
                        if (!Py_IsInitialized()) return;   // leaked at exit
                        py::gil_scoped_acquire gil;
                        delete static_cast<py::object*>(p);
                    });
            },
            "The controllers' state at t (pulsim internal; set by "
            "simulate(), read by resume_from).")
        .def_property_readonly("restart_pending",
            [](const SolverSnapshot& s) { return s.restart_steps_left > 0; },
            "True when restart_steps_left > 0.")
        .def("__repr__", [](const SolverSnapshot& s) {
            std::string r = "<SolverSnapshot t=" + std::to_string(s.t)
                   + " n=" + std::to_string(s.x.size());
            if (s.is_legacy()) {
                r += " legacy";
            } else {
                for (std::size_t b = 0; b < device::kNumStateBlocks; ++b) {
                    if (s.dev_layout[b] > 0) {
                        r += " " + std::string(device::kStateBlocks[b].name)
                             + "=" + std::to_string(s.dev_layout[b]);
                    }
                }
            }
            return r + (s.valid ? " valid>" : " EMPTY>");
        })
        .def(py::pickle(
            [](const SolverSnapshot& s) {
                // A tag first, so it never reads as the 2.0 tuples
                // (which start with t). Version 5 (batch 14a) is the one
                // layout that carries every block, always written:
                //   0 tag, 1 version, 2-10 as version 3, 11 legacy payload
                //   or None, 12 the controllers'-state marker, 13 the grid
                //   (origin, step, dt, post_edge_derivatives), 14 the step
                //   controller (kind, values) or None, 15 tick_origin,
                //   16 tick_next_k, 17 the loss tracker's state (in-kernel
                //   event losses; 17-field version-5 tuples still load),
                //   18 the recorders' state (spec, values) or None
                //   (add-probes-and-recorders 0b.7; 18-field tuples load).
                // Version 4 had four layouts on parallel branches (13, 14
                // or 15 fields; field 12 a bool or a tuple); all still load.
                const auto& L = s.legacy;
                py::tuple base = py::make_tuple(
                    std::string("SolverSnapshot"), 5, s.t, s.x, s.dev_state,
                    s.dev_layout, s.layout_signature, s.mask, s.stamped_mask,
                    s.restart_steps_left, s.valid,
                    s.is_legacy()
                        ? py::object(py::make_tuple(
                              L.history, L.saturable_history, L.coss_history,
                              L.lauritzen_history, L.igbt_tail_history,
                              L.pmsm_history))
                        : py::object(py::none()),
                    // the controllers' state as bytes where every part
                    // has them (pulsim._control_state.pickle_payload),
                    // else True: a resume from the copy refuses by name
                    // instead of continuing whatever the graphs hold
                    s.control_state
                        ? py::module_::import("pulsim._control_state")
                              .attr("pickle_payload")(
                                  *static_cast<py::object*>(s.control_state.get()))
                        : py::object(py::bool_(false)),
                    py::make_tuple(s.grid_origin, s.grid_step, s.grid_dt,
                                   s.post_edge_derivatives),
                    s.controller.empty()
                        ? py::object(py::none())
                        : py::object(py::make_tuple(
                              static_cast<int>(s.controller.kind),
                              s.controller.v)),
                    s.tick_origin, s.tick_next_k, s.loss_state,
                    s.recorder_spec.empty() && s.recorder_state.empty()
                        ? py::object(py::none())
                        : py::object(py::make_tuple(s.recorder_spec, s.recorder_state)));
                // 19 the transmission lines' past (add-realistic-passives
                // D18), appended only when present: a snapshot without a
                // line pickles exactly as before
                if (s.line_state.empty()) return base;
                py::list all(base);
                all.append(py::cast(s.line_state));
                return py::tuple(all);
            },
            [](py::tuple t) {
              try {
                SolverSnapshot s;
                if (t.size() >= 2 && py::isinstance<py::str>(t[0])) {
                    const int ver = t[1].cast<int>();
                    const std::size_t n = t.size();
                    if (t[0].cast<std::string>() != "SolverSnapshot"
                        || !((ver == 3 && (n == 12 || n == 13))
                             || (ver == 4 && n >= 13 && n <= 15)
                             || (ver == 5 && n >= 17 && n <= 20))) {
                        throw std::runtime_error(
                            "SolverSnapshot: pickle version "
                            + std::to_string(ver) + " with "
                            + std::to_string(n)
                            + " fields; this Pulsim reads version 3 "
                              "(12 or 13 fields), 4 (13 to 15) and 5 (17 to 20)");
                    }
                    // Where each block sits (npos = absent). Version 4's
                    // variants differ in field 12: a bool there is the
                    // marker (the PR's 14 fields: grid at 13; the event
                    // queue's 15: the tick phase at 13-14; the device
                    // interface's 15: grid 13, controller 14), a tuple is the
                    // grid (the device interface: marker at 13, controller
                    // at 14).
                    constexpr std::size_t npos = std::size_t(-1);
                    std::size_t marker = npos, grid = npos, ctrl = npos,
                                ticks = npos;
                    if (ver == 3) {
                        if (n == 13) marker = 12;
                    } else if (ver == 5) {
                        marker = 12; grid = 13; ctrl = 14; ticks = 15;
                    } else if (n == 13) {
                        grid = 12;
                    } else if (py::isinstance<py::bool_>(t[12])) {
                        marker = 12;
                        if (n == 14) {
                            grid = 13;
                        } else if (py::isinstance<py::tuple>(t[13])) {
                            // the device interface's last v4 (52f82c08):
                            // marker, grid, controller
                            grid = 13; ctrl = 14;
                        } else {
                            ticks = 13;
                        }
                    } else {
                        grid = 12; marker = 13;
                        if (n == 15) ctrl = 14;
                    }
                    if (grid != npos) {
                        auto G = t[grid].cast<py::tuple>();
                        if (G.size() != 4) {
                            throw std::runtime_error(
                                "SolverSnapshot: the grid field "
                                "has " + std::to_string(G.size())
                                + " values, expected 4");
                        }
                        s.grid_origin = G[0].cast<Real>();
                        s.grid_step = G[1].cast<Size>();
                        s.grid_dt = G[2].cast<Real>();
                        s.post_edge_derivatives = G[3].cast<bool>();
                    }
                    if (ctrl != npos && !t[ctrl].is_none()) {
                        auto C = t[ctrl].cast<py::tuple>();
                        const int kind = C.size() == 2 ? C[0].cast<int>() : -1;
                        if (kind < 1 || kind > 3) {
                            throw std::runtime_error(
                                "SolverSnapshot: the controller-state field "
                                "must be (kind, values) with kind 1 (fixed), "
                                "2 (LTE) or 3 (the exact engine)");
                        }
                        s.controller.kind =
                            static_cast<pulsim::step::ControllerState::Kind>(kind);
                        s.controller.v = C[1].cast<std::vector<Real>>();
                    }
                    if (ticks != npos) {
                        s.tick_origin = t[ticks].cast<std::vector<Real>>();
                        s.tick_next_k =
                            t[ticks + 1].cast<std::vector<std::uint64_t>>();
                        if (s.tick_origin.size() != s.tick_next_k.size()) {
                            throw std::runtime_error(
                                "SolverSnapshot: tick_origin and tick_next_k "
                                "differ in length");
                        }
                    }
                    if (ver == 5 && n >= 18) {
                        s.loss_state = t[17].cast<std::vector<Real>>();
                    }
                    if (ver == 5 && n == 20) {
                        s.line_state = t[19].cast<std::vector<Real>>();
                    }
                    if (ver == 5 && n >= 19 && !t[18].is_none()) {
                        const auto rs = t[18].cast<py::tuple>();
                        s.recorder_spec = rs[0].cast<std::string>();
                        s.recorder_state = rs[1].cast<std::vector<Real>>();
                    }
                    s.t = t[2].cast<Real>();
                    s.x = t[3].cast<Vector>();
                    s.dev_state = t[4].cast<std::vector<Real>>();
                    s.dev_layout = t[5].cast<std::vector<std::uint32_t>>();
                    s.layout_signature = t[6].cast<std::uint64_t>();
                    s.mask = t[7].cast<std::vector<bool>>();
                    s.stamped_mask = t[8].cast<std::vector<bool>>();
                    s.restart_steps_left = t[9].cast<Size>();
                    s.valid = t[10].cast<bool>();
                    if (!t[11].is_none()) {
                        auto L = t[11].cast<py::tuple>();
                        if (L.size() != 6) {
                            throw std::runtime_error(
                                "SolverSnapshot: the version-3 legacy "
                                "payload has "
                                + std::to_string(L.size())
                                + " fields, expected 6");
                        }
                        auto& g = s.legacy;
                        g.history = L[0].cast<std::vector<Real>>();
                        g.saturable_history = L[1].cast<std::vector<Real>>();
                        g.coss_history = L[2].cast<std::vector<Real>>();
                        g.lauritzen_history = L[3].cast<std::vector<Real>>();
                        g.igbt_tail_history = L[4].cast<std::vector<Real>>();
                        g.pmsm_history = L[5].cast<std::vector<Real>>();
                    }
                    if (marker != npos && !py::isinstance<py::bool_>(t[marker])
                        && !t[marker].is_none()) {
                        // the controllers' state (pulsim._control_state)
                        s.control_state = std::shared_ptr<void>(
                            new py::object(py::reinterpret_borrow<py::object>(t[marker])),
                            [](void* p) {
                                if (!Py_IsInitialized()) return;
                                py::gil_scoped_acquire gil;
                                delete static_cast<py::object*>(p);
                            });
                    } else if (marker != npos && t[marker].cast<bool>()) {
                        // a marker the resume refuses by name (pulsim._control_state)
                        s.control_state = std::shared_ptr<void>(
                            new py::object(py::str("dropped by pickling")),
                            [](void* p) {
                                if (!Py_IsInitialized()) return;
                                py::gil_scoped_acquire gil;
                                delete static_cast<py::object*>(p);
                            });
                    }
                    return s;
                }
                // Pulsim 2.0 (versions 1 and 2): 10 fields, 11 with
                // `stamped_mask`, 12 with the restart state (a bool
                // `restart_pending` before it became the SDIRK step
                // count). Loaded as a legacy payload: only the circuit
                // knows its layout, so a resume converts it.
                if (t.size() < 10 || t.size() > 12) {
                    throw std::runtime_error(
                        "SolverSnapshot: pickled state has "
                        + std::to_string(t.size())
                        + " fields, expected a version-3 tuple or the 10, "
                          "11 or 12 fields of Pulsim 2.0");
                }
                s.t = t[0].cast<Real>();
                s.x = t[1].cast<Vector>();
                auto& g = s.legacy;
                g.history = t[2].cast<std::vector<Real>>();
                s.mask = t[3].cast<std::vector<bool>>();
                g.saturable_history = t[4].cast<std::vector<Real>>();
                g.coss_history = t[5].cast<std::vector<Real>>();
                g.lauritzen_history = t[6].cast<std::vector<Real>>();
                g.igbt_tail_history = t[7].cast<std::vector<Real>>();
                g.pmsm_history = t[8].cast<std::vector<Real>>();
                s.valid = t[9].cast<bool>();
                if (t.size() >= 11) {
                    s.stamped_mask = t[10].cast<std::vector<bool>>();
                }
                if (t.size() >= 12) {
                    s.restart_steps_left =
                        py::isinstance<py::bool_>(t[11])
                            ? (t[11].cast<bool>() ? Size{3} : Size{0})
                            : t[11].cast<Size>();
                }
                return s;
              } catch (const py::cast_error& e) {
                // a malformed tuple: the named error, not pybind's cast message
                throw std::runtime_error(
                    std::string("SolverSnapshot: malformed pickle state (a "
                                "field has the wrong type): ") + e.what());
              } catch (py::error_already_set& e) {
                // a field that is not a sequence where one is read (tuple(bool))
                if (!e.matches(PyExc_TypeError)) throw;
                throw std::runtime_error(
                    std::string("SolverSnapshot: malformed pickle state (a "
                                "field has the wrong type): ") + e.what());
              }
            }));

    m.def("run_transient",
        [](const pwl::PwlStateSpaceCache& cache,
           const topology::Graph& graph,
           const pwl::DevicePool& pool,
           const SimulationOptions& opts,
           py::object switch_fn_obj,         // Bridge.13: was SwitchScheduleFn
           py::object b_extra_obj,
           bool start_from_dc_op,
           std::optional<bool> enable_nonlinear_refresh,
           py::object step_observer_obj,
           py::object initial_state,
           std::optional<ShouldContinueFn> should_continue_arg,
           std::optional<std::shared_ptr<streaming::LiveRing>> live_ring_arg,
           py::object resume_from_obj,
           std::vector<Real> tick_periods,
           std::vector<py::object> tick_fn_objs,
           py::object tick_before_obj,
           Real tick_origin,
           std::vector<Real> tick_origins,
           std::vector<std::uint64_t> tick_next_ks,
           std::vector<bool> tick_closed_ends) {
            // add-unified-event-queue group 5: Python tick sources, which
            // the kernel's queue schedules (source j calls tick_fns[j]).
            // a graph's queue tick (CxxChainTick) runs in C++, no GIL
            std::vector<std::function<void(Real, const Vector&)>> tick_fns;
            for (const auto& o : tick_fn_objs) {
                tick_fns.push_back(
                    py::isinstance<pulsim::blockchain::ChainTick>(o)
                        ? o.cast<pulsim::blockchain::ChainTick&>().fn
                        : o.cast<std::function<void(Real, const Vector&)>>());
            }
            if (tick_periods.size() != tick_fns.size()) {
                throw std::invalid_argument(
                    "run_transient: tick_periods and tick_fns must have the "
                    "same length");
            }
            if (!tick_origins.empty() && tick_origins.size() != tick_periods.size()) {
                throw std::invalid_argument(
                    "run_transient: tick_origins must be empty or as long as "
                    "tick_periods");
            }
            pulsim::events::TickPlan tick_plan;
            for (std::size_t j = 0; j < tick_periods.size(); ++j) {
                tick_plan.rates.push_back(pulsim::events::Rate{
                    tick_periods[j], Real{0},
                    tick_origins.empty() ? tick_origin : tick_origins[j],
                    tick_next_ks.size() == tick_periods.size() && !tick_origins.empty()
                        ? tick_next_ks[j]
                        : std::numeric_limits<std::uint64_t>::max(),
                    j < tick_closed_ends.size() && tick_closed_ends[j]});
            }
            tick_plan.fn = [&tick_fns](std::uint32_t j, const pulsim::events::TickInfo&,
                                       Real t, const Vector& xs) {
                tick_fns[j](t, xs);
            };
            if (py::isinstance<pulsim::blockchain::ChainTick>(tick_before_obj)) {
                tick_plan.before =
                    tick_before_obj.cast<pulsim::blockchain::ChainTick&>().fn;
            } else if (!tick_before_obj.is_none()) {
                tick_plan.before = tick_before_obj.cast<StepObserverFn>();
            }
            // A composed control graph (CxxChainTick) runs in C++.
            StepObserverFn step_observer;
            if (py::isinstance<pulsim::blockchain::ChainTick>(step_observer_obj)) {
                step_observer =
                    step_observer_obj.cast<pulsim::blockchain::ChainTick&>().fn;
            } else if (!step_observer_obj.is_none()) {
                step_observer = step_observer_obj.cast<StepObserverFn>();
            }
            // Bridge.13: native schedules run their mask in C++ (no
            // Python __call__); the detection is shared by every engine.
            auto adapted =
                pulsim_bindings_common::adapt_switch_fn(switch_fn_obj);
            const SwitchScheduleFn switch_fn = std::move(adapted.fn);
            // add-fixed-step-event-splitting: the schedule's edges, read
            // only when enable_substep_state_correction splits steps.
            // A Python schedule may declare its edges the same way, with a
            // `next_edge_after(t)` method (the DSED engine reads it too).
            std::function<Real(Real)> next_edge_fn =
                std::move(adapted.next_edge);
            if (!next_edge_fn && py::hasattr(switch_fn_obj, "next_edge_after")) {
                next_edge_fn = switch_fn_obj.attr("next_edge_after")
                                   .cast<std::function<Real(Real)>>();
            }
            // add-discontinuity-restart-rule: a b_extra_fn may DECLARE
            // its jumps with a `next_jump_after(t)` attribute (first
            // jump strictly after t, inf when none); the restart rule
            // then restarts on them and reads their left limit.
            BExtraFn b_extra_fn;
            std::function<Real(Real)> b_extra_next_jump;
            resolve_b_extra(b_extra_obj, b_extra_fn, b_extra_next_jump);
            const pwl::NonlinearRefreshFn nl_refresh = transient_refresh(
                graph, pool, enable_nonlinear_refresh, "pwl");
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
            const ShouldContinueFn should_continue = should_continue_arg.value_or(nullptr);
            const auto live_ring = live_ring_arg.value_or(nullptr);
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
                                          resume_ptr,
                                          b_extra_next_jump,
                                          next_edge_fn,
                                          tick_periods.empty() ? nullptr
                                                               : &tick_plan);
            }
            return result;
        },
        py::arg("cache"), py::arg("graph"), py::arg("pool"),
        py::arg("opts"), py::arg("switch_fn"),
        py::arg("b_extra_fn") = py::none(),
        py::arg("start_from_dc_op") = false,
        py::arg("enable_nonlinear_refresh") = py::none(),
        py::arg("step_observer") = py::none(),
        py::arg("initial_state") = py::none(),
        py::arg("should_continue") = py::none(),
        py::arg("live_ring") = py::none(),
        py::arg("resume_from") = py::none(),
        py::arg("tick_periods") = std::vector<Real>{},
        py::arg("tick_fns") = std::vector<py::object>{},
        py::arg("tick_before") = py::none(),
        py::arg("tick_origin") = std::numeric_limits<Real>::quiet_NaN(),
        py::arg("tick_origins") = std::vector<Real>{},
        py::arg("tick_next_ks") = std::vector<std::uint64_t>{},
        py::arg("tick_closed_ends") = std::vector<bool>{},
        "Run a fixed-dt transient simulation. switch_fn(t) "
        "→ SwitchStateMask; b_extra_fn(t) → Vector adds "
        "to b_constant at each step. `step_observer(t, x)` "
        "(optional) is a void callback fired at the start "
        "of every step BEFORE switch_fn/b_extra_fn evaluate "
        "— use it to update Python-side controller state "
        "(PI, comparator, etc.) so the next switch_fn call "
        "reads the new duty. A `b_extra_fn` may declare its jump "
        "instants with a `next_jump_after(t)` attribute (first jump "
        "strictly after t, inf when none): with "
        "discontinuity_restart they restart the step and are read "
        "at their left limit; an undeclared jump is not restarted. "
        "`enable_nonlinear_refresh`: "
        "None (default) lets the kernel stamp the Newton "
        "devices (smooth / Shockley diode, MOSFET, IGBT) "
        "when the circuit has any; True is the same "
        "(deprecated); False with such devices raises "
        "UnsupportedError.");

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
    auto resolve_refresh = &pwl::dc_device_refresh;

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
            std::optional<analysis::ShouldContinueFn> should_continue,
            std::optional<bool> enable_nonlinear_refresh,
            Real gmin,
            Real gmin_start,
            Size gmin_steps,
            device::DevicePath device_path) {
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
                resolve_refresh(pool, enable_nonlinear_refresh, device_path);
            Vector x = pwl::compute_dc_op_with_strategy(
                graph, pool, mask, strategy, t_eval, pt, ss,
                should_continue.value_or(nullptr), refresh, g, &report);
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
        py::arg("should_continue") = py::none(),
        py::arg("enable_nonlinear_refresh") = py::none(),
        py::arg("gmin") = pwl::kDefaultGmin,
        py::arg("gmin_start") = Real{1e-2},
        py::arg("gmin_steps") = Size{10},
        py::arg("device_path") = device::DevicePath::Legacy,
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
            std::optional<analysis::ShouldContinueFn> should_continue,
            device::DevicePath device_path) {
            pwl::DCOperatingPointOptions o;
            o.should_continue = should_continue.value_or(nullptr);
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
                resolve_refresh(pool, enable_nonlinear_refresh, device_path);
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
        py::arg("should_continue") = py::none(),
        py::arg("device_path") = device::DevicePath::Legacy,
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
            std::optional<analysis::ShouldContinueFn> should_continue) {
            return analysis::run_mna_sweep(
                graph, pool, mask, freqs,
                input_state_idx, output_node_idx,
                should_continue.value_or(nullptr));
        },
        py::arg("graph"), py::arg("pool"), py::arg("mask"),
        py::arg("freqs"), py::arg("input_state_idx"),
        py::arg("output_node_idx"),
        py::arg("should_continue") = py::none(),
        "Direct kernel AC sweep via complex sparse LU at each "
        "frequency. Returns a MnaSweepKernelResult with parallel "
        "freqs/H arrays.\n"
        "Optional `should_continue` is invoked between frequency\n"
        "points; raises pulsim.Cancelled when it returns False.");

    // add-unified-device-interface 9.2: a static Newton branch's current and
    // absorbed power at each stored state, from its device law (the reading
    // the in-loop recorders take; `pwl::static_reading`).
    m.def("static_device_readings",
        [](const topology::Graph& graph, const pwl::DevicePool& pool,
           py::array_t<double, py::array::c_style | py::array::forcecast> states,
           Index branch_id) -> py::object {
            // an IGBT with a tail: its current is i_ss plus the tail's delta,
            // which needs the stored charge (not in the states)
            if (pool.is_registered(branch_id)
                && pool.kind_of(branch_id) == pwl::DevicePool::StoredKind::IgbtLevel1
                && models::IgbtLevel1::has_tail(pool.igbt_level1_params(branch_id))) {
                return py::none();
            }
            const auto s = states.unchecked<2>();
            const auto n_rows = s.shape(0), n_cols = s.shape(1);
            if (n_cols < static_cast<py::ssize_t>(pool.state_size(graph))) {
                throw py::value_error(std::format(
                    "static_device_readings: states has {} columns, the circuit "
                    "needs {}", n_cols, pool.state_size(graph)));
            }
            py::array_t<double> cur(n_rows), pw(n_rows);
            auto c = cur.mutable_unchecked<1>();
            auto w = pw.mutable_unchecked<1>();
            Vector x(n_cols);
            for (py::ssize_t r = 0; r < n_rows; ++r) {
                for (py::ssize_t k = 0; k < n_cols; ++k) x[k] = s(r, k);
                pwl::DeviceReading d;
                if (!pwl::static_reading(graph, pool, x, branch_id, d)) return py::none();
                c(r) = d.i_pin[0];
                w(r) = d.power;
            }
            return py::make_tuple(std::move(cur), std::move(pw));
        },
        py::arg("graph"), py::arg("pool"), py::arg("states"), py::arg("branch_id"),
        "The current INTO the first pin and the absorbed power of a static "
        "Newton branch (smooth / Shockley diode, level-1 MOSFET / IGBT "
        "without the tail's delta) at each row of `states`, from its device "
        "law. None when the branch is not a static Newton branch, or is an "
        "IGBT with a tail (its delta needs the stored charge).");

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
        .def_readwrite("phase",     &PwmGenerator::phase,
                        "Seconds, a delay (an advance in periods before 2026-11).")
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

    // add-compiled-control-graph: an algebraic loop, a signal with two
    // writers, an unknown block kind. A ValueError subclass.
    py::register_exception<pulsim::blockchain::ControlGraphBuildError>(
        m, "ControlGraphBuildError", PyExc_ValueError);

    py::class_<ChainTick>(m, "CxxChainTick", py::dynamic_attr(),
        "A control-graph tick callback the variable-step engine calls in "
        "C++ (no Python call per tick).")
        .def("__call__", [](ChainTick& self, Real t, const Vector& x) {
            self.fn(t, x);
        });

    py::class_<ChainPwmGate>(m, "CxxChainPwmGate", py::dynamic_attr(),
        "A PWM switch_fn whose duty is a control-graph signal; every "
        "engine calls it in C++.")
        .def("__call__", [](const ChainPwmGate& self, Real t) { return self(t); },
            py::arg("t"))
        .def("next_edge_after", &ChainPwmGate::next_edge_after, py::arg("t"),
            "First gate edge strictly after t under the duties held now "
            "(period starts and OFF thresholds); stale after a tick.")
        .def("legs", [](const ChainPwmGate& self) {
                std::vector<std::tuple<Index, Index, Real, Real, Real, Real>> out;
                for (Size i = 0; i < self.legs.size(); ++i) {
                    const auto& l = self.legs[i];
                    const auto d = l.edge_duty_derivative();
                    out.emplace_back(l.hs, self.leg_slots[i], l.T, l.phase, d[0], d[1]);
                }
                return out;
            },
            "Per leg: (switch bit, duty slot, period, phase, d t_rise / d duty, "
            "d t_fall / d duty) from the modulator's geometry.");

    py::class_<ChainSignalGate>(m, "CxxChainSignalGate", py::dynamic_attr(),
        "A discrete gate: a switch is ON while its control-graph signal "
        "exceeds 0.5; its edges are the graph's publication instants. "
        "Every engine calls it in C++.")
        .def("__call__", [](const ChainSignalGate& self, Real t) { return self(t); },
            py::arg("t"))
        .def("next_edge_after", &ChainSignalGate::next_edge_after, py::arg("t"),
            "First instant strictly after t at which the gate may switch "
            "(a publication of the graph; inf for a label-time graph).");

    py::class_<ChainBExtra>(m, "CxxChainBExtra",
        "A control graph's driven-source b_extra_fn(t), with the instants "
        "its value may jump (next_jump_after), both in C++.")
        .def("__call__", [](ChainBExtra& self, Real t) { return self.fn(t); })
        .def("next_jump_after", [](ChainBExtra& self, Real t) {
            return self.chain->next_jump_after(t);
        });

    py::class_<BlockChain::State>(m, "CxxBlockChainState",
        "A copy of everything a run changes in one control graph (block "
        "states, signals, tick grid, recordings); restorable only into "
        "the graph that saved it. Opaque; does not pickle.");

    py::class_<BlockChain>(m, "CxxBlockChain",
        "C++ block chain executor. Build via add_<block>(...) "
        "methods, bind nodes, then pass make_step_observer(dt) to "
        "simulate().")
        .def(py::init<>())
        .def("size", &BlockChain::size)
        .def("reset", &BlockChain::reset)
        .def("save_state", &BlockChain::save_state,
            "A copy of the graph's run state (snapshot / resume).")
        .def("restore_state", &BlockChain::restore_state, py::arg("state"),
            "Put back a state saved from this graph.")
        .def("dump_state", [](BlockChain& self, const BlockChain::State& s) {
                return py::bytes(self.dump_state(s));
            }, py::arg("state"),
            "A state saved from this graph as bytes (what a pickled "
            "snapshot carries; read back on the same machine and build).")
        .def("load_state", [](BlockChain& self, const py::bytes& b) {
                const std::string buf = b;
                return self.load_state(buf);
            }, py::arg("data"),
            "The state dump_state wrote, restorable into this graph "
            "(ValueError: bytes of another graph). The graph is unchanged.")
        .def("next_due_per_group", &BlockChain::next_due_per_group,
            "Each rate group's next tick instant (group 1, 2, ...).")
        .def("next_due", &BlockChain::next_due,
            "Instant of the next tick on the graph's own grid (inf: none).")
        .def_property_readonly("n_calls", &BlockChain::n_calls,
            "Observer calls since the last restart of the tick grid.")
        .def("uncaptured_blocks", &BlockChain::uncaptured_blocks,
            "Blocks holding state that save_state() does not copy.")
        .def("name_last", &BlockChain::name_last, py::arg("name"),
            py::arg("kind") = std::string{},
            "Name (and kind) of the most recently added block.")
        .def("compile", &BlockChain::compile,
            "Order the blocks (direct-feedthrough topological order, "
            "ties by insertion); raises ControlGraphBuildError on an "
            "algebraic loop.")
        .def("schedule", &BlockChain::schedule,
            "Block names in evaluation order.")
        .def("reorders", &BlockChain::reorders,
            "Pairs whose evaluation order differs from insertion order.")
        .def("add_group", &BlockChain::add_group, py::arg("period"),
            py::arg("offset") = 0.0, py::arg("delay_kind") = 0,
            py::arg("delay_time") = 0.0,
            "Add a Discrete{period, offset} rate group (delay_kind 0 = "
            "zero, 1 = a time, 2 = one sample); returns its id (>= 1).")
        .def("set_block_group", &BlockChain::set_block_group,
            py::arg("block"), py::arg("group"))
        .def_property_readonly("has_rates", &BlockChain::has_rates)
        .def("group_period", &BlockChain::group_period, py::arg("group"))
        .def("slot", [](BlockChain& self, const std::string& name) { return self.slot(name); },
            py::arg("name"), "The signal slot of channel `name` (allocated on first mention).")
        .def("tick_jacobian",
            [](BlockChain& self, int group, Real t, const Vector& x,
               const std::vector<std::string>& params, bool commit) {
                const TickJacobian j = self.tick_jacobian(group, t, x, params, commit);
                const auto nr = static_cast<py::ssize_t>(j.rows.size());
                const auto nc = static_cast<py::ssize_t>(j.cols.size());
                py::array_t<Real> J({nr, nc});
                if (!j.J.empty()) std::memcpy(J.mutable_data(), j.J.data(), j.J.size() * sizeof(Real));
                py::list kinks;
                for (const auto& k : j.kinks) {
                    kinks.append(py::dict(py::arg("block") = k.block, py::arg("law") = k.law,
                                          py::arg("a") = k.a, py::arg("b") = k.b,
                                          py::arg("side") = k.side, py::arg("live") = k.live));
                }
                return py::dict(py::arg("rows") = j.rows, py::arg("cols") = j.cols,
                                py::arg("values") = j.values, py::arg("J") = J,
                                py::arg("kinks") = kinks);
            }, py::arg("group"), py::arg("t"), py::arg("x"),
            py::arg("params") = std::vector<std::string>{}, py::arg("commit") = false,
            "Exact Jacobian of one tick of rate group `group` at (t, x) from the "
            "graph's current state, by the block laws on the AD scalar: rows = "
            "the states after the tick then 'sig:<channel>' for every signal; "
            "cols = the states before it, the signals, the probed state entries "
            "'x[i]', then `params` ('<block>.<field>' parameters or constant "
            "inputs). `kinks`: every comparison of a law the tick took (operands "
            "a, b, the side, live = it depends on a column). commit=True then "
            "runs the group's own tick at (t, x) (no counters, no recording).")
        .def("group_offset", &BlockChain::group_offset, py::arg("group"))
        .def("tick_counts", &BlockChain::tick_counts,
            "Ticks per group (index 0: evaluations of the every-call group).")
        .def_property_readonly("tick_jitter_max", &BlockChain::tick_jitter_max)
        .def("add_source_sink", &BlockChain::add_source_sink,
            py::arg("name"), py::arg("row"), py::arg("gain"),
            "b_extra[row] += gain * (signal `name` after compute_delay).")
        .def("set_solver_dt", &BlockChain::set_solver_dt, py::arg("dt"))
        .def("set_label_time", &BlockChain::set_label_time, py::arg("on"),
            "Label-time mode: on the fixed engine a tick runs at the call's "
            "label t_n with the state x_{n-1} (the closed-loop helpers' "
            "protocol); see docs/control-graph.md.")
        .def_property_readonly("label_time", &BlockChain::label_time)
        .def("set_horizon", &BlockChain::set_horizon, py::arg("t_end"),
            "Label-time mode: no tick at or after t_end on the fixed engine.")
        .def("set_output", &BlockChain::set_output, py::arg("name"),
            py::arg("value"),
            "Set a signal and what its circuit sinks see before the first "
            "tick (call after compile()).")
        .def("make_pwm_gate",
            [](BlockChain& self, Size num_switches,
               const std::vector<std::tuple<std::string, Index, Real, Real>>& legs) {
                ChainPwmGate g{num_switches, {}};
                for (const auto& [ch, idx, period, phase] : legs) {
                    if (idx < 0 || static_cast<Size>(idx) >= num_switches ||
                        !(period > Real{0})) {
                        throw std::invalid_argument(
                            "make_pwm_gate: switch index out of range or "
                            "period <= 0");
                    }
                    g.add_leg(self, self.slot(ch), idx, period, phase);
                }
                return g;
            }, py::arg("num_switches"), py::arg("legs"), py::keep_alive<0, 1>(),
            "A switch_fn (C++, no GIL): for each (channel, switch_idx, "
            "period, phase) the bit is ON while pulsim.control.pwm_on(t, "
            "period, duty, phase) with duty = what the channel's sinks see.")
        .def("make_signal_gate",
            [](BlockChain& self, Size num_switches,
               const std::vector<std::pair<std::string, Index>>& mapping) {
                ChainSignalGate g{&self, num_switches, {}};
                for (const auto& [ch, idx] : mapping) {
                    if (idx < 0 || static_cast<Size>(idx) >= num_switches) {
                        throw std::invalid_argument(
                            "make_signal_gate: switch index out of range");
                    }
                    g.pairs.emplace_back(self.slot(ch), idx);
                }
                return g;
            }, py::arg("num_switches"), py::arg("mapping"), py::keep_alive<0, 1>(),
            "A discrete gate (C++, no GIL): for each (channel, switch_idx) "
            "the bit is ON while what the channel's sinks see is > 0.5.")
        .def("make_tick_observer", [](BlockChain& self) {
                return ChainTick{&self, self.make_tick_observer()};
            }, py::keep_alive<0, 1>(),
            "Tick callback for the variable-step engine (exact protocol).")
        .def("set_queue_driven", &BlockChain::set_queue_driven, py::arg("on"),
            "Fixed engine: the kernel's event queue decides when a rate "
            "group ticks (make_queue_tick); the graph's observer runs the "
            "marked groups.")
        .def_property_readonly("queue_driven", &BlockChain::queue_driven)
        .def("drop_horizon_state", &BlockChain::drop_horizon_state,
            "Forget the tick the last run's horizon left out (a run that "
            "does not resume the snapshot holding it).")
        .def_property_readonly("last_call_t", &BlockChain::last_call_t,
            "The label of the last observer call.")
        .def_property_readonly("grid_t0", &BlockChain::grid_t0,
            "The tick grid's t0 (the first observer call's time).")
        .def("group_next_k", &BlockChain::group_next_k, py::arg("group"),
            "Rate group `group`'s next tick index on the graph's grid.")
        .def("make_queue_tick", [](BlockChain& self, int group) {
                if (group < 1 || static_cast<Size>(group) >= self.n_groups()) {
                    throw std::invalid_argument("make_queue_tick: no such rate group");
                }
                BlockChain* ch = &self;
                return ChainTick{ch, [ch, group](Real, const Vector&) {
                    ch->mark_queued(group);
                }};
            }, py::arg("group"), py::keep_alive<0, 1>(),
            "The kernel tick source of rate group `group` (C++, no GIL): "
            "marks it due for the graph's next fixed-step pass.")
        .def_property_readonly("n_evals", &BlockChain::n_evals)
        .def_property_readonly("n_block_evals", &BlockChain::n_block_evals)
        .def("get_channel", &BlockChain::get_channel, py::arg("name"))
        .def("sink_value", [](BlockChain& self, const std::string& name) {
            return self.sink_value(self.find_slot(name));
        }, py::arg("name"),
        "What a circuit sink (switch mask, driven source) sees: a rate "
        "group's output after its compute_delay.")
        .def("set_channel", [](BlockChain& self, const std::string& name,
                                   Real value) {
            self.set_channel(name, value);
        }, py::arg("name"), py::arg("value"))
        .def("bind_nodes", [](BlockChain& self,
                                const std::function<Index(const std::string&)>&
                                    node_id_of) {
            self.bind_nodes(node_id_of);
        }, py::arg("node_id_of"),
            "Resolve InputRef::Node inputs given by name, by calling "
            "node_id_of(name), for the blocks added AFTER this call.")
        .def("make_step_observer", [](BlockChain& self, Real dt) {
            return self.make_step_observer(dt);
        }, py::arg("dt"), py::keep_alive<0, 1>(),
            "Return a step_observer(t, x) callable that the kernel "
            "uses to invoke the chain each step.")
        .def("make_sink_b_extra_fn",
            [](BlockChain& self, Size state_size) {
                return ChainBExtra{&self, self.make_b_extra_fn(state_size)};
            }, py::arg("state_size"), py::keep_alive<0, 1>(),
            "Return the driven-source b_extra_fn(t) with next_jump_after(t) "
            "(the instants the published values may change).")
        .def("make_b_extra_fn",
            [](BlockChain& self, Size state_size) {
                return self.make_b_extra_fn(state_size);
            }, py::arg("state_size"), py::keep_alive<0, 1>(),
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
                (void)self; (void)idx; (void)value;  // placeholder
            }, py::arg("idx"), py::arg("value"),
            "Diagnostic — directly set a b_extra entry. Most "
            "users should add motor blocks instead.")
        // ---- block factories — one per BlockType in blocks.hpp ----
        .def("add_gain",
            [](BlockChain& self, Real k, InputRef x, std::string out) {
                auto blk = std::make_shared<Gain>();
                blk->k = k;
                auto p = make_gain_step(self, blk, x, out);
                self.add(std::move(p.step), std::move(p.reset));
                self.set_last_ad(std::move(p.ad));
            }, py::arg("k"), py::arg("x"), py::arg("output"))
        .def("add_subtract",
            [](BlockChain& self, InputRef a, InputRef b, std::string out) {
                auto blk = std::make_shared<Subtract>();
                auto p = make_subtract_step(self, blk, a, b, out);
                self.add(std::move(p.step), std::move(p.reset));
                self.set_last_ad(std::move(p.ad));
            }, py::arg("a"), py::arg("b"), py::arg("output"))
        .def("add_sum",
            [](BlockChain& self, Real w0, Real w1, Real w2,
                InputRef a, InputRef b, InputRef c,
                std::string out) {
                auto blk = std::make_shared<Sum>();
                blk->w0 = w0; blk->w1 = w1; blk->w2 = w2;
                auto p = make_sum_step(self, blk, a, b, c, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("w0") = 1.0, py::arg("w1") = 1.0,
                py::arg("w2") = 0.0, py::kw_only(),
                py::arg("a"), py::arg("b"),
                py::arg_v("c", InputRef::from_const(0.0), "CxxInputRef.const_value(0.0)"),
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
                auto p = make_math_block_step(self, blk, a, b, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("op"), py::arg("a"), py::arg("b"),
                py::arg("output"))
        .def("add_pi_controller",
            [](BlockChain& self, Real Kp, Real Ki,
                Real output_min, Real output_max,
                InputRef setpoint, InputRef measured, InputRef dt_ref,
                std::string out, Real integrator_state, Real prev_error) {
                auto blk = std::make_shared<PIController>();
                blk->Kp = Kp; blk->Ki = Ki;
                blk->output_min = output_min; blk->output_max = output_max;
                blk->integral = integrator_state;
                blk->prev_error = prev_error;
                auto p = make_pi_controller_step(self, blk, setpoint, measured,
                                                       dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
                self.set_last_ad(std::move(p.ad));
            }, py::arg("Kp"), py::arg("Ki"),
                py::arg("output_min"), py::arg("output_max"),
                py::arg("setpoint"), py::arg("measured"),
                py::arg("dt"), py::arg("output"),
                py::arg("integrator_state") = Real{0},
                py::arg("prev_error") = Real{0})
        .def("add_pid_controller",
            [](BlockChain& self, Real Kp, Real Ki, Real Kd, Real tau_d,
                Real output_min, Real output_max,
                InputRef setpoint, InputRef measured, InputRef dt_ref,
                std::string out) {
                auto blk = std::make_shared<PIDController>();
                blk->Kp = Kp; blk->Ki = Ki; blk->Kd = Kd;
                blk->tau_d = tau_d;
                blk->output_min = output_min; blk->output_max = output_max;
                auto p = make_pid_controller_step(self, blk, setpoint, measured,
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
                auto p = make_first_order_lpf_step(self, blk, input_value,
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
                auto p = make_integrator_step(self, blk, x, dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("gain"), py::arg("output_min"),
                py::arg("output_max"), py::arg("x"), py::arg("dt"),
                py::arg("output"))
        .def("add_differentiator",
            [](BlockChain& self, Real filter_alpha,
                InputRef x, InputRef dt_ref, std::string out) {
                auto blk = std::make_shared<Differentiator>();
                blk->filter_alpha = filter_alpha;
                auto p = make_differentiator_step(self, blk, x, dt_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("filter_alpha"), py::arg("x"), py::arg("dt"),
                py::arg("output"))
        .def("add_limiter",
            [](BlockChain& self, Real min_v, Real max_v,
                InputRef x, std::string out) {
                auto blk = std::make_shared<Limiter>();
                blk->min_v = min_v; blk->max_v = max_v;
                auto p = make_limiter_step(self, blk, x, out);
                self.add(std::move(p.step), std::move(p.reset));
                self.set_last_ad(std::move(p.ad));
            }, py::arg("min_v"), py::arg("max_v"),
                py::arg("x"), py::arg("output"))
        .def("add_moving_average",
            [](BlockChain& self, Size window, InputRef x,
                std::string out) {
                auto blk = std::make_shared<MovingAverageFilter>();
                blk->window = window;
                auto p = make_moving_average_step(self, blk, x, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("window"), py::arg("x"), py::arg("output"))
        .def("add_pwm_generator",
            [](BlockChain& self, Real frequency, Real phase,
                std::optional<Real> phase_fraction,
                InputRef duty, InputRef t_ref, std::string out) {
                auto blk = std::make_shared<PwmGenerator>();
                blk->frequency = frequency; blk->phase = phase;
                if (phase_fraction) {
                    if (phase != Real{0}) {
                        throw std::invalid_argument(
                            "add_pwm_generator: give phase (seconds) or "
                            "phase_fraction (of the period), not both");
                    }
                    // The old `phase` reading (an advance of f periods),
                    // bit for bit: offset -(f * (1/frequency)).
                    blk->phase = -(*phase_fraction * (Real{1} / frequency));
                }
                auto p = make_pwm_generator_step(self, blk, duty, t_ref, out);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("frequency"), py::arg("phase") = 0.0,
                py::kw_only(), py::arg("phase_fraction") = py::none(),
                py::arg("duty"), py::arg("t"), py::arg("output"),
            "PWM comparator. phase is in seconds, a delay, as "
            "PwmGenerator(phase=); phase_fraction= is an advance in periods "
            "(what phase= read before 2026-11: phase=x is now "
            "phase_fraction=x, or phase=-x/frequency).")
        .def("add_svm",
            [](BlockChain& self, Real v_dc,
                InputRef v_alpha, InputRef v_beta,
                std::string da, std::string db, std::string dc) {
                auto blk = std::make_shared<SpaceVectorModulator>();
                blk->v_dc = v_dc;
                auto p = make_svm_step(self, blk, v_alpha, v_beta, da, db, dc);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("v_dc"), py::arg("v_alpha"), py::arg("v_beta"),
                py::arg("output_a"), py::arg("output_b"),
                py::arg("output_c"))
        .def("add_clarke",
            [](BlockChain& self, InputRef a, InputRef b, InputRef c,
                std::string alpha, std::string beta, std::string zero) {
                auto blk = std::make_shared<ClarkeTransform>();
                auto p = make_clarke_step(self, blk, a, b, c, alpha, beta, zero);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("a"), py::arg("b"), py::arg("c"),
                py::arg("output_alpha"), py::arg("output_beta"),
                py::arg("output_zero"))
        .def("add_inverse_clarke",
            [](BlockChain& self, InputRef alpha, InputRef beta,
                InputRef zero, std::string a, std::string b, std::string c) {
                auto blk = std::make_shared<InverseClarkeTransform>();
                auto p = make_inverse_clarke_step(self, blk, alpha, beta, zero,
                                                        a, b, c);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("alpha"), py::arg("beta"),
                py::arg_v("zero", InputRef::from_const(0.0), "CxxInputRef.const_value(0.0)"),
                py::kw_only(), py::arg("output_a"), py::arg("output_b"),
                py::arg("output_c"))
        .def("add_park",
            [](BlockChain& self, InputRef alpha, InputRef beta,
                InputRef theta, std::string d, std::string q) {
                auto blk = std::make_shared<ParkTransform>();
                auto p = make_park_step(self, blk, alpha, beta, theta, d, q);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("alpha"), py::arg("beta"), py::arg("theta"),
                py::arg("output_d"), py::arg("output_q"))
        .def("add_inverse_park",
            [](BlockChain& self, InputRef d, InputRef q, InputRef theta,
                std::string alpha, std::string beta) {
                auto blk = std::make_shared<InverseParkTransform>();
                auto p = make_inverse_park_step(self, blk, d, q, theta,
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
                auto p = make_pll_step(self, blk, v_alpha, v_beta, dt_ref,
                                            theta, omega, freq);
                self.add(std::move(p.step), std::move(p.reset));
            }, py::arg("f_nominal"), py::arg("Kp"), py::arg("Ki"),
                py::arg("v_alpha"), py::arg("v_beta"), py::arg("dt"),
                py::arg("output_theta"), py::arg("output_omega"),
                py::arg("output_freq"))

        // ---- Ported from pulsim.control (add-compiled-control-graph) ----
        .def("add_ported",
            [](BlockChain& self, const std::string& kind,
               const PortedParams& params, std::vector<InputRef> inputs,
               const std::vector<std::string>& outputs) {
                try {
                    auto p = make_ported_step(self, kind, params,
                                              std::move(inputs), outputs);
                    self.add(std::move(p.step), std::move(p.reset),
                             std::move(p.update));
                    self.set_last_ad(std::move(p.ad));
                } catch (...) {
                    self.discard_pending();
                    throw;
                }
            }, py::arg("kind"), py::arg("params"), py::arg("inputs"),
                py::arg("outputs"),
            "A block ported from pulsim.control, Python-exact (kinds: "
            "comparator, rate_limiter, sample_hold, transfer_function, "
            "state_machine, op_amp, lookup_table_1d, math, ema, "
            "differentiator, pid, sum, signal_mux, signal_demux).")
        .def("add_python_block",
            [](BlockChain& self, py::function fn,
               std::vector<InputRef> inputs,
               const std::vector<std::string>& outputs, bool with_state) {
                std::vector<InputRef> in;
                for (auto& r : inputs) in.push_back(self.in(std::move(r)));
                std::vector<Index> out;
                for (const auto& o : outputs) out.push_back(self.out(o));
                // Owned by the chain; the chain dies under the GIL.
                auto f = std::make_shared<py::function>(std::move(fn));
                auto step = [f, in, out, with_state](ChainContext& c) {
                    py::gil_scoped_acquire gil;   // the kernel runs without it
                    const Size k0 = with_state ? 1 : 0;
                    py::tuple args(in.size() + k0);
                    if (with_state) {   // the state vector, as an observer gets it
                        args[0] = c.x != nullptr ? py::cast(*c.x)
                                                 : py::cast(Vector{});
                    }
                    for (Size i = 0; i < in.size(); ++i) {
                        args[k0 + i] = py::float_(resolve(in[i], c));
                    }
                    py::object r = (*f)(*args);
                    if (out.size() == 1) {
                        c.put(out[0], r.cast<Real>());
                        return;
                    }
                    const auto v = r.cast<std::vector<Real>>();
                    if (v.size() != out.size()) {
                        // std::invalid_argument crosses the kernel's run
                        // loop as itself (ValueError), a py::value_error
                        // became SimulationAborted.
                        throw std::invalid_argument(
                            "PythonBlock returned " + std::to_string(v.size()) +
                            " values for " + std::to_string(out.size()) +
                            " outputs");
                    }
                    for (Size i = 0; i < out.size(); ++i) c.put(out[i], v[i]);
                };
                self.add(std::move(step), {});
            }, py::arg("fn"), py::arg("inputs"), py::arg("outputs"),
                py::arg("with_state") = false,
            "A block that calls fn(*input_values) (Python, under the GIL) "
            "once per evaluation; returns a float or one value per output. "
            "with_state=True passes the state vector x first.")
        .def("add_c_function_block",
            [](BlockChain& self, std::uintptr_t step_addr,
               std::uintptr_t state_addr, std::uintptr_t in_addr,
               std::uintptr_t out_addr, std::vector<InputRef> inputs,
               const std::vector<std::string>& outputs) {
                // The C-block ABI (pulsim.CBLOCK_ABI), called in C++: the
                // caller (add_c_block) keeps the library, the state pointer
                // and the two double buffers alive with the handle.
                using CStep = void (*)(const double*, int, double*, int,
                                       double, double, void**);
                auto* f = reinterpret_cast<CStep>(step_addr);
                auto* st = reinterpret_cast<void**>(state_addr);
                auto* inb = reinterpret_cast<double*>(in_addr);
                auto* outb = reinterpret_cast<double*>(out_addr);
                if (f == nullptr || st == nullptr ||
                    (!inputs.empty() && inb == nullptr) ||
                    (!outputs.empty() && outb == nullptr)) {
                    throw std::invalid_argument("add_c_function_block: null pointer");
                }
                std::vector<InputRef> in;
                for (auto& r : inputs) in.push_back(self.in(std::move(r)));
                std::vector<Index> out;
                for (const auto& o : outputs) out.push_back(self.out(o));
                auto step = [f, st, inb, outb, in, out](ChainContext& c) {
                    for (Size i = 0; i < in.size(); ++i) inb[i] = resolve(in[i], c);
                    f(inb, static_cast<int>(in.size()), outb,
                      static_cast<int>(out.size()), c.t, c.dt, st);
                    for (Size i = 0; i < out.size(); ++i) c.put(out[i], outb[i]);
                };
                self.add(std::move(step), {});
            }, py::arg("step_addr"), py::arg("state_addr"), py::arg("in_addr"),
                py::arg("out_addr"), py::arg("inputs"), py::arg("outputs"),
            "A compiled C block (pulsim.CBLOCK_ABI) evaluated in C++ (no GIL): "
            "inputs are copied to in_addr, the step writes out_addr, whose "
            "values become the outputs. Used by add_c_block.")
        .def("add_delay",
            [](BlockChain& self, long long samples, Real initial_value,
                InputRef x, std::string out) {
                if (samples < 1) {
                    throw ControlGraphBuildError(
                        "DelayBlock: samples must be >= 1 (got " +
                        std::to_string(samples) + ")");
                }
                auto blk = std::make_shared<DelayBlock>();
                blk->samples = static_cast<Size>(samples);
                blk->initial_value = initial_value;
                auto p = make_delay_step(self, blk, x, out);
                self.add(std::move(p.step), std::move(p.reset),
                         std::move(p.update));
            }, py::arg("samples"), py::arg("initial_value"),
                py::arg("x"), py::arg("output"),
            "y[k] = x[k - samples]; not direct feedthrough, so it breaks "
            "an algebraic loop.")

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
            py::kw_only(),
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
            py::kw_only(),
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
                if (PyErr_WarnEx(PyExc_DeprecationWarning,
                        "BlockChain.add_thyristor (add_thyristor_to_chain) is "
                        "deprecated: a latch updated once per step with a "
                        "sampled |i| < i_holding test. Use "
                        "CircuitBuilder.add_thyristor, whose latch the "
                        "engines decide at the current zero.", 1) != 0) {
                    throw py::error_already_set();
                }
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

    // Several graphs' fixed-step observers composed in C++, in order (each
    // chain must outlive the run): the kernel calls them without the GIL.
    m.def("block_laws_source", [] {
        return std::string(
#include "block_laws_source.inc"
        );
    }, "The C99 law table (block_laws.h) the kernel graph calls; "
       "pulsim.export_c copies it into the C it generates.");
    m.def("compose_chain_observers",
        [](const std::vector<std::pair<BlockChain*, Real>>& parts) {
            std::vector<std::function<void(Real, const Vector&)>> fns;
            for (const auto& [ch, dt] : parts) fns.push_back(ch->make_step_observer(dt));
            py::object out = py::cast(ChainTick{nullptr, [fns](Real t, const Vector& x) {
                for (const auto& f : fns) f(t, x);
            }});
            // the observers read each chain through a pointer: pin each
            // chain (its existing Python wrapper), not just the list
            for (const auto& part : parts)
                py::detail::keep_alive_impl(
                    out, py::cast(part.first, py::return_value_policy::reference));
            return py::reinterpret_steal<py::typing::Union<ChainTick>>(out.release());
        }, py::arg("parts"),
        "One CxxChainTick that runs each (chain, dt)'s step observer in "
        "order; run_transient calls it in C++.");

    // Helper for make_chain_switch_fn — wires chain channels →
    // SwitchStateMask bits.
    m.def("make_chain_switch_fn",
        [](BlockChain& chain, Size num_switches,
            const std::vector<std::pair<std::string, Index>>& mapping) {
            return make_chain_switch_fn(chain, num_switches, mapping);
        },
        py::arg("chain"), py::arg("num_switches"), py::arg("mapping"), py::keep_alive<0, 1>(),
        "Build a switch_fn(t) that sets bit `idx` ON when "
        "`chain.sink_value(channel_name) > 0.5`. `mapping` is a list "
        "of (channel_name, switch_idx) pairs.");

    // ----- BDF1 (implicit Euler) — Phase B.2 ------------------------------
    m.def("run_transient_bdf1",
        [](const builder::CircuitBuilder& builder,
            const SimulationOptions& opts,
            SwitchScheduleFn switch_fn,
            std::optional<BExtraFn> b_extra_fn,
            bool start_from_dc_op,
            std::optional<StepObserverFn> step_observer) {
            SimulationResult result;
            {
                py::gil_scoped_release rel;
                result = run_transient_bdf1(
                    builder, opts, switch_fn, b_extra_fn.value_or(nullptr),
                    start_from_dc_op, step_observer.value_or(nullptr));
            }
            return result;
        },
        py::arg("builder"), py::arg("opts"), py::arg("switch_fn"),
        py::arg("b_extra_fn") = py::none(),
        py::arg("start_from_dc_op") = false,
        py::arg("step_observer") = py::none(),
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
           py::object b_extra_obj,
           py::object initial_state,
           std::optional<Size> max_event_iterations,
           std::optional<solver::ShouldContinueFn> should_continue,
           Real observer_period,
           py::object observer_obj,
           std::optional<bool> enable_nonlinear_refresh,
           std::optional<Size> max_newton_iterations,
           Real tol_newton_dx, Real tol_newton_res,
           bool enable_newton_line_search, bool enable_newton_lm,
           py::object resume_from,
           std::vector<Real> observer_periods,
           std::vector<py::object> observers,
           std::optional<Size> newton_roundoff_budget, bool on_impulse_error,
           std::vector<Real> observer_offsets,
           Real observer_origin,
           std::vector<bool> observer_monitors,
           std::vector<Real> observer_origins,
           std::vector<std::uint64_t> observer_next_k,
           device::DevicePath device_path, int mode_selection, int integrator,
           int roundoff_floor, bool trace_forced) {
            solver::BExtraFn b_extra_fn;
            std::function<Real(Real)> unused_next_jump;  // trbdf2 lands its ticks
            resolve_b_extra(b_extra_obj, b_extra_fn, unused_next_jump);
            solver::TrBdf2Options o;
            o.device_path = device_path;
            o.t_start = t_start;
            o.t_end   = t_end;
            o.rtol    = rtol;
            o.atol    = atol;
            o.h_init  = h_init;
            o.h_max   = h_max;
            // add-kernel-simulation-object: None keeps the kernel's
            // default (sim::Defaults); every value is a value. The variable
            // step has no off switch for its event localisation, so 0 is
            // refused rather than replaced.
            if (max_event_iterations) {
                if (*max_event_iterations == 0) {
                    throw UnsupportedError("trbdf2", "max_event_iterations",
                        "", "", "run_transient_trbdf2: max_event_iterations=0 "
                        "(no event localisation) is not available on the "
                        "variable step; pass None for the default.");
                }
                o.max_event_iterations = *max_event_iterations;
            }
            o.observer_period = observer_period;
            if (observers.size() != observer_periods.size()) {
                throw std::invalid_argument(
                    "run_transient_trbdf2: observers and "
                    "observer_periods must have the same length");
            }
            o.observer_periods = std::move(observer_periods);
            o.observer_offsets = std::move(observer_offsets);
            if (!observer_monitors.empty()
                && observer_monitors.size() != o.observer_periods.size()) {
                throw std::invalid_argument(
                    "run_transient_trbdf2: observer_monitors must be empty or "
                    "as long as observer_periods");
            }
            o.observer_monitors.assign(observer_monitors.begin(), observer_monitors.end());
            o.observer_origin = observer_origin;
            o.observer_origins = std::move(observer_origins);
            o.observer_next_k = std::move(observer_next_k);
            if (max_newton_iterations) {
                if (*max_newton_iterations == 0) {
                    throw sim::OptionsError("max_newton_iterations",
                        "run_transient_trbdf2: max_newton_iterations must be "
                        ">= 1 (None = the default)");
                }
                o.max_newton_iterations = *max_newton_iterations;
            }
            // A value when >= 0 (simulate() passes the resolved one); the
            // negative default keeps the kernel's (sim::Defaults).
            if (tol_newton_dx >= Real{0}) {
                o.tol_newton_dx = tol_newton_dx;
            }
            if (tol_newton_res >= Real{0}) {
                o.tol_newton_res = tol_newton_res;
            }
            o.enable_newton_line_search = enable_newton_line_search;
            o.enable_newton_lm = enable_newton_lm;
            if (newton_roundoff_budget) {
                o.newton_roundoff_budget = *newton_roundoff_budget;
            }
            o.on_impulse_error = on_impulse_error;
            o.mode_selection = mode_selection;
            o.integrator = integrator;
            o.roundoff_floor = roundoff_floor;   // C++-level handle (DECISIONS 17)
            o.trace_forced = trace_forced;       // diagnostic (update-trbdf2-error-control)
            const pwl::NonlinearRefreshFn nl_refresh = transient_refresh(
                graph, pool, enable_nonlinear_refresh, "trbdf2");
            std::optional<Vector> x0;
            if (!initial_state.is_none()) {
                x0 = initial_state.cast<Vector>();
            }
            // Bridge.13 pattern: a native schedule runs the mask
            // function in pure C++ AND hands the stepper its analytic
            // next_edge_after — edge landing then costs zero Python
            // round-trips and no bisection.
            auto adapted = pulsim_bindings_common::adapt_switch_fn(
                switch_fn_obj, static_cast<Size>(graph.num_switches()));
            const solver::SwitchScheduleFn switch_fn = std::move(adapted.fn);
            const std::function<Real(Real)> next_edge_fn =
                std::move(adapted.next_edge);
            std::function<void(Real, const Vector&)> observer_fn;
            if (!observer_obj.is_none()) {
                // Re-acquires the GIL per call through pybind's
                // functional caster, like every other Python
                // callback the released-GIL engines invoke.
                observer_fn = observer_obj.cast<
                    std::function<void(Real, const Vector&)>>();
            }
            std::vector<std::function<void(Real, const Vector&)>> obs_fns;
            for (const auto& ob : observers) {
                if (py::isinstance<ChainTick>(ob)) {   // control graph: C++ path
                    obs_fns.push_back(ob.cast<ChainTick&>().fn);
                    continue;
                }
                obs_fns.push_back(ob.cast<
                    std::function<void(Real, const Vector&)>>());
            }
            std::function<void(Size, Real, const Vector&)> tick_fn;
            if (!obs_fns.empty()) {
                tick_fn = [&obs_fns](Size j, Real t, const Vector& x) {
                    obs_fns[j](t, x);
                };
            }
            solver::TrBdf2Stats st;
            SimulationResult res;
            SolverSnapshot resume_snap;
            const SolverSnapshot* resume_ptr = nullptr;
            if (!resume_from.is_none()) {
                resume_snap = resume_from.cast<SolverSnapshot>();
                resume_ptr = &resume_snap;
            }
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
                    x0, &st, next_edge_fn, should_continue.value_or(nullptr),
                    observer_fn, nl_refresh, resume_ptr, tick_fn);
            }
            return py::make_tuple(std::move(res),
                pulsim_bindings_common::trbdf2_stats_dict(st));
        },
        py::arg("cache"), py::arg("graph"), py::arg("pool"),
        py::arg("t_start"), py::arg("t_end"),
        py::arg("rtol"), py::arg("atol"),
        py::arg("h_init"), py::arg("h_max"),
        py::arg("switch_fn"),
        py::arg("b_extra_fn") = py::none(),
        py::arg("initial_state") = py::none(),
        py::arg("max_event_iterations") = py::none(),
        py::arg("should_continue") = py::none(),
        py::arg("observer_period") = Real{0},
        py::arg("step_observer") = py::none(),
        py::arg("enable_nonlinear_refresh") = py::none(),
        py::arg("max_newton_iterations") = py::none(),
        py::arg("tol_newton_dx") = Real{-1},
        py::arg("tol_newton_res") = Real{-1},
        py::arg("enable_newton_line_search") = false,
        py::arg("enable_newton_lm") = false,
        py::arg("resume_from") = py::none(),
        py::arg("observer_periods") = std::vector<Real>{},
        py::arg("observers") = std::vector<py::object>{},
        py::arg("newton_roundoff_budget") = py::none(),
        py::arg("on_impulse_error") = false,
        py::arg("observer_offsets") = std::vector<Real>{},
        py::arg("observer_origin") = std::numeric_limits<Real>::quiet_NaN(),
        py::arg("observer_monitors") = std::vector<bool>{},
        py::arg("observer_origins") = std::vector<Real>{},
        py::arg("observer_next_k") = std::vector<std::uint64_t>{},
        py::arg("device_path") = device::DevicePath::Legacy,
        py::arg("mode_selection") = 0,
        py::arg("integrator") = 0,
        py::arg("roundoff_floor") = 0,
        py::arg("trace_forced") = false,
        "Variable-step TR-BDF2 transient (engine='auto'): "
        "L-stable, 2nd order, LTE-controlled step, gate edges "
        "landed by bisection, diode crossings localized by "
        "Illinois. Returns (SimulationResult, stats dict). "
        "`observers[j]` ticks at t_start + observer_offsets[j] + "
        "k*observer_periods[j] (ceil((span - offset_j)/T_j) times, index "
        "order when coincident; offsets default to 0); "
        "observer_origin (NaN = t_start) moves the grid's origin, so a "
        "resumed run ticks where the uninterrupted one would; "
        "stats['control_tick_counts'] counts each.");

    m.def("run_transient_with_chain",
        [](const pwl::PwlStateSpaceCache& cache,
           const topology::Graph& graph,
           const pwl::DevicePool& pool,
           const SimulationOptions& opts,
           py::object switch_fn_obj,
           BlockChain& chain,
           Real chain_dt,
           py::object b_extra_obj,
           bool start_from_dc_op,
           std::optional<bool> enable_nonlinear_refresh,
           py::object initial_state,
           std::optional<ShouldContinueFn> should_continue_arg,
           std::optional<std::shared_ptr<streaming::LiveRing>> live_ring_arg,
           py::object resume_from) {
            const pwl::NonlinearRefreshFn nl_refresh = transient_refresh(
                graph, pool, enable_nonlinear_refresh, "pwl");
            // The declarations run_transient reads (a native schedule's
            // edges, a b_extra_fn's next_jump_after): this path dropped
            // them, so the restart rule never saw a declared jump here.
            auto adapted =
                pulsim_bindings_common::adapt_switch_fn(switch_fn_obj);
            const SwitchScheduleFn switch_fn = std::move(adapted.fn);
            std::function<Real(Real)> next_edge_fn = std::move(adapted.next_edge);
            if (!next_edge_fn && py::hasattr(switch_fn_obj, "next_edge_after")) {
                next_edge_fn = switch_fn_obj.attr("next_edge_after")
                                   .cast<std::function<Real(Real)>>();
            }
            BExtraFn b_extra_fn;
            std::function<Real(Real)> b_extra_next_jump;
            resolve_b_extra(b_extra_obj, b_extra_fn, b_extra_next_jump);
            // Build a step_observer that calls the chain DIRECTLY in
            // C++ — no Python roundtrip per step.
            chain.set_solver_dt(opts.dt);   // rate groups: state at the tick
            auto step_observer = chain.make_step_observer(chain_dt);
            // Wrap with a LiveRing push when streaming is requested.
            // Both layers are GIL-free in the hot path: chain step is
            // C++ and ``LiveRing::push`` uses only atomics + memcpy.
            StepObserverFn observer_eff = step_observer;
            const ShouldContinueFn should_continue = should_continue_arg.value_or(nullptr);
            const auto live_ring = live_ring_arg.value_or(nullptr);
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
            SolverSnapshot resume_snap;
            const SolverSnapshot* resume_ptr = nullptr;
            if (!resume_from.is_none()) {
                resume_snap = resume_from.cast<SolverSnapshot>();
                resume_ptr = &resume_snap;
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
                                          continue_eff,
                                          resume_ptr,
                                          b_extra_next_jump,
                                          next_edge_fn);
            }
            return result;
        },
        py::arg("cache"), py::arg("graph"), py::arg("pool"),
        py::arg("opts"), py::arg("switch_fn"),
        py::arg("chain"), py::arg("chain_dt"),
        py::arg("b_extra_fn") = py::none(),
        py::arg("start_from_dc_op") = false,
        py::arg("enable_nonlinear_refresh") = py::none(),
        py::arg("initial_state") = py::none(),
        py::arg("should_continue") = py::none(),
        py::arg("live_ring") = py::none(),
        py::arg("resume_from") = py::none(),
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

    m.def("thermal_voltage",
           &models::ShockleyDiode::thermal_voltage,
           py::arg("T_kelvin"),
           "Thermal voltage kT/q [V] at a junction temperature "
           "[K]. 25.85 mV at 300 K. NOTE, because it is the "
           "opposite of what everyone remembers: raising V_T "
           "ALONE RAISES the forward drop (V_F = n*V_T*ln(i/I_S) "
           "at fixed I_S, so 0.710 V at 25 C becomes 0.948 V at "
           "125 C). A real diode's negative temperature "
           "coefficient comes from I_S, which roughly doubles "
           "every 10 C. A temperature sweep must scale BOTH — "
           "pair this with `shockley_saturation_current_at`.");

    m.def("shockley_saturation_current_at",
           &models::ShockleyDiode::saturation_current_at,
           py::arg("I_S_ref"), py::arg("T_kelvin"),
           py::arg("T_ref_kelvin") = 300.15,
           py::arg("n") = 1.0,
           py::arg("E_g_eV") = 1.11,
           py::arg("XTI") = 3.0,
           "Saturation current at temperature `T_kelvin`, given "
           "its value at `T_ref_kelvin` — SPICE's law. This is "
           "the term that makes a hot diode drop LESS; see "
           "`thermal_voltage`.");

    m.def("shockley_voltage_for_current",
           [](Real i, Real I_S, Real n, Real V_T) {
               const models::ShockleyDiode::Params p{
                   .I_S = I_S, .n = n, .V_T = V_T};
               return models::ShockleyDiode::voltage_for_current(
                   p, i);
           },
           py::arg("i"), py::arg("I_S") = 1e-12,
           py::arg("n") = 1.0, py::arg("V_T") = 0.025852,
           "Closed-form inverse of the Shockley law: the forward "
           "voltage that carries `i` amps. Useful for sizing and "
           "for checking a simulation against the model.");

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
namespace pulsim_ir_bindings {  // add-immutable-circuit-ir (ir_bindings.cpp)
    void init_module(pybind11::module_& m);
}
namespace pulsim_sim_bindings {  // add-kernel-simulation-object (sim_bindings.cpp)
    void init_module(pybind11::module_& m);
}
namespace pulsim_record_bindings {  // add-probes-and-recorders (record_bindings.cpp)
    void init_module(pybind11::module_& m);
}

PYBIND11_MODULE(_pulsim, m) {
    m.doc() = "Pulsim — power-electronics simulator. Header-only C++23 "
              "kernel: PWL state-space cache, trapezoidal companion, "
              "Newton refinement, and the high-level CircuitBuilder API.";
    pulsim_kernel_bindings::init_module(m);
    pulsim_dsed_bindings::init_module(m);
    pulsim_ir_bindings::init_module(m);
    pulsim_sim_bindings::init_module(m);
    pulsim_record_bindings::init_module(m);
}
