// =============================================================================
// add-kernel-simulation-object — Python bindings of `sim::RunOptions`, the
// option x engine table and `sim::Simulation`
// =============================================================================
//
// A separate translation unit (like ir_bindings.cpp) so the hunk in
// bindings.cpp is three lines.

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsim/ir/circuit.hpp"
#include "pulsim/sim/options.hpp"
#include "pulsim/record/loop_feed.hpp"
#include "pulsim/sim/simulation.hpp"
#include "pulsim/xdev/output_trace.hpp"
#include "switch_fn_adapter.hpp"

#include <limits>
#include <optional>
#include <string_view>
#include <set>
#include <string>

namespace py = pybind11;

namespace pulsim_sim_bindings {

using namespace pulsim;
using sim::RunOptions;

namespace {

py::list engine_names(std::uint8_t bits) {
    py::list out;
    for (auto e : sim::kRunEngines) {
        if (bits & pwl::engine_bit(e)) out.append(std::string(pwl::engine_name(e)));
    }
    return out;
}

const char* unset_name(sim::Unset u) {
    switch (u) {
        case sim::Unset::Always: return "always";
        case sim::Unset::None: return "none";
        case sim::Unset::Falsy: return "falsy";
        case sim::Unset::NoneOrOne: return "none_or_one";
        case sim::Unset::NoneOrZero: return "none_or_zero";
        case sim::Unset::Legacy: return "legacy";
        case sim::Unset::Integrator: return "integrator";
    }
    return "?";
}

template <class E>
E parse_enum(const std::string& v, std::initializer_list<std::pair<const char*, E>> m,
             const char* what) {
    for (const auto& [k, e] : m) {
        if (v == k) return e;
    }
    std::string ok;
    for (const auto& [k, e] : m) ok += (ok.empty() ? "'" : ", '") + std::string(k) + "'";
    throw sim::OptionsError(what, "RunOptions." + std::string(what) + "='" + v
                                      + "': expected one of " + ok);
}

std::string engine_str(sim::EngineChoice e) {
    switch (e) {
        case sim::EngineChoice::Pwl: return "pwl";
        case sim::EngineChoice::TrBdf2: return "trbdf2";
        case sim::EngineChoice::Exact: return "exact";
        case sim::EngineChoice::Auto: break;
    }
    return "auto";
}

py::dict trbdf2_dict(const solver::TrBdf2Options& o) {
    py::dict d;
    d["t_start"] = o.t_start;
    d["t_end"] = o.t_end;
    d["rtol"] = o.rtol;
    d["atol"] = o.atol;
    d["h_init"] = o.h_init;
    d["h_max"] = o.h_max;
    d["max_event_iterations"] = o.max_event_iterations;
    d["max_newton_iterations"] = o.max_newton_iterations;
    d["tol_newton_dx"] = o.tol_newton_dx;
    d["tol_newton_res"] = o.tol_newton_res;
    d["enable_newton_line_search"] = o.enable_newton_line_search;
    d["enable_newton_lm"] = o.enable_newton_lm;
    d["newton_roundoff_budget"] = o.newton_roundoff_budget;
    d["on_impulse_error"] = o.on_impulse_error;
    d["mode_selection"] = o.mode_selection;
    d["integrator"] = o.integrator;
    return d;
}

}  // namespace

void init_module(py::module_& m) {
    m.def("run_option_table", [] {
            py::list rows;
            for (const auto& o : sim::kRunOptionTable) {
                py::dict r;
                r["name"] = std::string(o.name);
                r["engines"] = engine_names(o.engines);
                r["unset"] = unset_name(o.unset);
                r["default"] = std::string(o.default_text);
                r["why"] = std::string(o.why);
                rows.append(r);
            }
            return rows;
        },
        "The option x engine table (sim::kRunOptionTable): one row per "
        "simulate() keyword, the engines that honour it, which value means "
        "'not requested', its documented default and the mechanism the "
        "other engines lack. pulsim.option_capabilities() is the public "
        "view.");

    m.def("run_option_defaults", [] {
            using D = sim::Defaults;
            py::dict d;
            d["rtol_variable"] = D::rtol_variable;
            d["atol_variable"] = D::atol_variable;
            d["rtol_exact"] = D::rtol_exact;
            d["atol_exact"] = D::atol_exact;
            d["dt_init_exact"] = D::dt_init_exact;
            d["dt_max_exact"] = D::dt_max_exact;
            d["stiffness_threshold"] = D::stiffness_threshold;
            d["h_bdf2"] = D::h_bdf2;
            d["max_newton_iterations"] = D::max_newton_iterations;
            d["tol_newton_dx"] = D::tol_newton_dx;
            d["tol_newton_res"] = D::tol_newton_res;
            d["newton_roundoff_budget"] = D::newton_roundoff_budget;
            d["max_event_iterations"] = D::max_event_iterations;
            d["max_dt_halvings"] = D::max_dt_halvings;
            d["store_every"] = D::store_every;
            d["voltage_sanity_factor"] = D::voltage_sanity_factor;
            d["split_snap_state"] = D::split_snap_state;
            return d;
        },
        "THE default values of a run (sim::Defaults).");

    py::class_<RunOptions>(m, "RunOptions",
        "Run options of sim::Simulation: one field per simulate() keyword "
        "the kernel reads. None = not set (the default of sim::Defaults "
        "for the engine that runs); never a sentinel value.")
        .def(py::init([](Real t_end, std::optional<Real> dt,
                         const std::string& engine, Real t_start) {
                 RunOptions o;
                 o.t_end = t_end;
                 o.dt = dt;
                 o.t_start = t_start;
                 o.engine = parse_enum<sim::EngineChoice>(engine,
                     {{"auto", sim::EngineChoice::Auto},
                      {"pwl", sim::EngineChoice::Pwl},
                      {"trbdf2", sim::EngineChoice::TrBdf2},
                      {"exact", sim::EngineChoice::Exact},
                      {"dsed", sim::EngineChoice::Exact}}, "engine");
                 return o;
             }),
             py::arg("t_end") = Real{0}, py::arg("dt") = py::none(),
             py::arg("engine") = "auto", py::arg("t_start") = Real{0})
        .def_property("engine",
            [](const RunOptions& o) { return engine_str(o.engine); },
            [](RunOptions& o, const std::string& v) {
                o.engine = parse_enum<sim::EngineChoice>(v,
                    {{"auto", sim::EngineChoice::Auto},
                     {"pwl", sim::EngineChoice::Pwl},
                     {"trbdf2", sim::EngineChoice::TrBdf2},
                     {"exact", sim::EngineChoice::Exact},
                     {"dsed", sim::EngineChoice::Exact}}, "engine");
            })
        .def_readwrite("t_start", &RunOptions::t_start)
        .def_readwrite("t_end", &RunOptions::t_end)
        .def_readwrite("dt", &RunOptions::dt)
        .def_readwrite("rtol", &RunOptions::rtol)
        .def_readwrite("atol", &RunOptions::atol)
        .def_readwrite("dt_init", &RunOptions::dt_init)
        .def_readwrite("stiffness_threshold", &RunOptions::stiffness_threshold)
        .def_readwrite("h_bdf2", &RunOptions::h_bdf2)
        .def_readwrite("enable_nonlinear_refresh",
                       &RunOptions::enable_nonlinear_refresh)
        .def_readwrite("max_newton_iterations", &RunOptions::max_newton_iterations)
        .def_readwrite("tol_newton_dx", &RunOptions::tol_newton_dx)
        .def_readwrite("tol_newton_res", &RunOptions::tol_newton_res)
        .def_readwrite("enable_newton_line_search",
                       &RunOptions::enable_newton_line_search)
        .def_readwrite("enable_newton_lm", &RunOptions::enable_newton_lm)
        .def_readwrite("newton_roundoff_budget", &RunOptions::newton_roundoff_budget)
        .def_readwrite("max_event_iterations", &RunOptions::max_event_iterations)
        .def_readwrite("strict_event_iterations",
                       &RunOptions::strict_event_iterations)
        .def_property("mode_selection",
            [](const RunOptions& o) {
                return std::string(o.mode_selection == sim::ModeSelection::Shadow
                                       ? "shadow"
                                       : o.mode_selection == sim::ModeSelection::Lcp
                                             ? "lcp" : "legacy");
            },
            [](RunOptions& o, const std::string& v) {
                o.mode_selection = parse_enum<sim::ModeSelection>(v,
                    {{"legacy", sim::ModeSelection::Legacy},
                     {"shadow", sim::ModeSelection::Shadow},
                     {"lcp", sim::ModeSelection::Lcp}}, "mode_selection");
            })
        .def_property("on_impulse",
            [](const RunOptions& o) {
                return std::string(o.on_impulse == sim::OnImpulse::Warn
                                       ? "warn"
                                       : o.on_impulse == sim::OnImpulse::Error
                                             ? "error" : "record");
            },
            [](RunOptions& o, const std::string& v) {
                o.on_impulse = parse_enum<sim::OnImpulse>(v,
                    {{"record", sim::OnImpulse::Record},
                     {"warn", sim::OnImpulse::Warn},
                     {"error", sim::OnImpulse::Error}}, "on_impulse");
            })
        .def_readwrite("mode_guard", &RunOptions::mode_guard)
        .def_property("control_tick",
            [](const RunOptions& o) {
                return std::string(o.control_tick_state ? "state" : "label");
            },
            [](RunOptions& o, const std::string& v) {
                o.control_tick_state = parse_enum<bool>(v,
                    {{"label", false}, {"state", true}}, "control_tick");
            })
        .def_property("device_path",
            [](const RunOptions& o) {
                return std::string(o.device_path.value_or(sim::Defaults::device_path)
                                       == device::DevicePath::Interface
                                       ? "interface" : "legacy");
            },
            [](RunOptions& o, const std::string& v) {
                o.device_path = parse_enum<device::DevicePath>(v,
                    {{"legacy", device::DevicePath::Legacy},
                     {"interface", device::DevicePath::Interface}}, "device_path");
            })
        .def_readwrite("max_dt_halvings", &RunOptions::max_dt_halvings)
        .def_readwrite("store_every", &RunOptions::store_every)
        .def_readwrite("start_from_dc_op", &RunOptions::start_from_dc_op)
        .def_readwrite("enable_substep_state_correction",
                       &RunOptions::enable_substep_state_correction)
        .def_readwrite("discontinuity_restart", &RunOptions::discontinuity_restart)
        .def_readwrite("split_snap", &RunOptions::split_snap)
        .def_readwrite("resolution_diagnostics",
                       &RunOptions::resolution_diagnostics)
        .def_readwrite("inductor_freeze_di_max", &RunOptions::inductor_freeze_di_max)
        .def_readwrite("inductor_abs_clamp", &RunOptions::inductor_abs_clamp)
        .def_readwrite("auto_regularize", &RunOptions::auto_regularize)
        .def_readwrite("voltage_sanity_factor", &RunOptions::voltage_sanity_factor)
        .def_readwrite("sensitivities", &RunOptions::sensitivities)
        .def("requested", [](const RunOptions& o) {
                std::vector<std::string> r;
                for (auto n : sim::requested_options(o)) r.emplace_back(n);
                return r;
            },
            "The table rows these options request, in table order.")
        .def("__repr__", [](const RunOptions& o) {
            return "<RunOptions engine=" + engine_str(o.engine)
                   + " t=[" + std::to_string(o.t_start) + ", "
                   + std::to_string(o.t_end) + "]>";
        });

    // simulate()'s hot path: a RunOptions from keywords in one call (None =
    // not set), filled in C++ (no Python attribute writes), then resolved by
    // the kernel. Values are coerced as Python's int() / float() / bool()
    // would; a negative count is an OptionsError; an unknown name a
    // TypeError.
    const auto from_kwargs = [](Real t_end, std::optional<Real> dt,
                                const std::string& engine, Real t_start,
                                const py::kwargs& kw) {
        RunOptions o;
        o.t_end = t_end;
        o.dt = dt;
        o.t_start = t_start;
        o.engine = parse_enum<sim::EngineChoice>(engine,
            {{"auto", sim::EngineChoice::Auto}, {"pwl", sim::EngineChoice::Pwl},
             {"trbdf2", sim::EngineChoice::TrBdf2},
             {"exact", sim::EngineChoice::Exact},
             {"dsed", sim::EngineChoice::Exact}}, "engine");
        // As strict as the pybind setters the old path used: an integer
        // (bool and numpy integers included) for a count, a number (not a
        // string) for a real, a bool / integer for a flag.
        const auto bad = [](std::string_view key, py::handle v, const char* want) {
            return py::type_error("simulate(" + std::string(key) + "="
                + py::repr(v).cast<std::string>() + "): expected " + want);
        };
        const auto count = [&bad](std::string_view key, py::handle v) {
            PyObject* i = PyNumber_Index(v.ptr());
            if (i == nullptr) {
                PyErr_Clear();
                throw bad(key, v, "an integer");
            }
            const long long n = PyLong_AsLongLong(i);
            Py_DECREF(i);
            if (n == -1 && PyErr_Occurred()) throw py::error_already_set();
            if (n < 0) {
                throw sim::OptionsError(std::string(key), "simulate("
                    + std::string(key) + "=" + py::repr(v).cast<std::string>()
                    + "): must be >= 0 (None = the default)");
            }
            return static_cast<Size>(n);
        };
        const auto real = [&bad](std::string_view key, py::handle v) {
            if (PyUnicode_Check(v.ptr()) || PyBytes_Check(v.ptr())) {
                throw bad(key, v, "a number");
            }
            const double x = PyFloat_AsDouble(v.ptr());
            if (x == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                throw bad(key, v, "a number");
            }
            return static_cast<Real>(x);
        };
        const auto flag = [&bad](std::string_view key, py::handle v) {
            if (PyBool_Check(v.ptr())) return v.ptr() == Py_True;
            PyObject* i = PyNumber_Index(v.ptr());  // numpy.bool_, 0 / 1
            if (i == nullptr) {
                PyErr_Clear();
                const int t = PyObject_IsTrue(v.ptr());
                if (t < 0 || !py::hasattr(v, "dtype")) {
                    PyErr_Clear();
                    throw bad(key, v, "a bool");
                }
                return t == 1;
            }
            const int t = PyObject_IsTrue(i);
            Py_DECREF(i);
            return t == 1;
        };
        for (const auto& [k, v] : kw) {
            if (v.is_none()) continue;
            Py_ssize_t len = 0;
            const char* c = PyUnicode_AsUTF8AndSize(k.ptr(), &len);
            if (c == nullptr) throw py::error_already_set();
            const std::string_view key(c, static_cast<std::size_t>(len));
            if (key == "max_newton_iterations") o.max_newton_iterations = count(key, v);
            else if (key == "max_event_iterations") o.max_event_iterations = count(key, v);
            else if (key == "newton_roundoff_budget") o.newton_roundoff_budget = count(key, v);
            else if (key == "max_dt_halvings") o.max_dt_halvings = count(key, v);
            else if (key == "store_every") o.store_every = count(key, v);
            else if (key == "rtol") o.rtol = real(key, v);
            else if (key == "atol") o.atol = real(key, v);
            else if (key == "dt_init") o.dt_init = real(key, v);
            else if (key == "stiffness_threshold") o.stiffness_threshold = real(key, v);
            else if (key == "h_bdf2") o.h_bdf2 = real(key, v);
            else if (key == "tol_newton_dx") o.tol_newton_dx = real(key, v);
            else if (key == "tol_newton_res") o.tol_newton_res = real(key, v);
            else if (key == "inductor_freeze_di_max") o.inductor_freeze_di_max = real(key, v);
            else if (key == "inductor_abs_clamp") o.inductor_abs_clamp = real(key, v);
            else if (key == "split_snap") o.split_snap = real(key, v);
            else if (key == "voltage_sanity_factor") o.voltage_sanity_factor = real(key, v);
            else if (key == "enable_nonlinear_refresh") o.enable_nonlinear_refresh = flag(key, v);
            else if (key == "enable_newton_line_search") o.enable_newton_line_search = flag(key, v);
            else if (key == "enable_newton_lm") o.enable_newton_lm = flag(key, v);
            else if (key == "enable_substep_state_correction") o.enable_substep_state_correction = flag(key, v);
            else if (key == "discontinuity_restart") o.discontinuity_restart = flag(key, v);
            else if (key == "resolution_diagnostics") o.resolution_diagnostics = flag(key, v);
            else if (key == "strict_event_iterations") o.strict_event_iterations = flag(key, v);
            else if (key == "mode_guard") o.mode_guard = flag(key, v);
            else if (key == "start_from_dc_op") o.start_from_dc_op = flag(key, v);
            else if (key == "auto_regularize") o.auto_regularize = flag(key, v);
            else if (key == "mode_selection") {
                o.mode_selection = parse_enum<sim::ModeSelection>(
                    v.cast<std::string>(),
                    {{"legacy", sim::ModeSelection::Legacy},
                     {"shadow", sim::ModeSelection::Shadow},
                     {"lcp", sim::ModeSelection::Lcp}}, "mode_selection");
            } else if (key == "integrator") {
                o.integrator_esdirk43 = parse_enum<bool>(v.cast<std::string>(),
                    {{"esdirk43", true}}, "integrator");
            } else if (key == "on_impulse") {
                o.on_impulse = parse_enum<sim::OnImpulse>(v.cast<std::string>(),
                    {{"record", sim::OnImpulse::Record},
                     {"warn", sim::OnImpulse::Warn},
                     {"error", sim::OnImpulse::Error}}, "on_impulse");
            } else if (key == "control_tick") {
                o.control_tick_state = parse_enum<bool>(v.cast<std::string>(),
                    {{"label", false}, {"state", true}}, "control_tick");
            } else if (key == "device_path") {
                o.device_path = parse_enum<device::DevicePath>(
                    v.cast<std::string>(),
                    {{"legacy", device::DevicePath::Legacy},
                     {"interface", device::DevicePath::Interface}}, "device_path");
            } else {
                throw py::type_error("RunOptions has no field '"
                                     + std::string(key) + "'");
            }
        }
        return o;
    };
    m.def("_simulation_options_kw",
          [from_kwargs](Real t_end, std::optional<Real> dt,
                        const std::string& engine, Real t_start,
                        const py::kwargs& kw) {
              return sim::to_simulation_options(
                  from_kwargs(t_end, dt, engine, t_start, kw));
          },
          py::arg("t_end"), py::arg("dt"), py::arg("engine"),
          py::arg("t_start"));
    m.def("_trbdf2_options_kw",
          [from_kwargs](Real t_end, std::optional<Real> dt,
                        const std::string& engine, Real t_start,
                        const py::kwargs& kw) {
              return trbdf2_dict(sim::to_trbdf2_options(
                  from_kwargs(t_end, dt, engine, t_start, kw)));
          },
          py::arg("t_end"), py::arg("dt"), py::arg("engine"),
          py::arg("t_start"));
    m.def("_run_options_kw", from_kwargs, py::arg("t_end"), py::arg("dt"),
          py::arg("engine"), py::arg("t_start"));

    m.def("_to_simulation_options", &sim::to_simulation_options,
          py::arg("options"),
          "The fixed-step driver's SimulationOptions resolved from a "
          "RunOptions (sim::to_simulation_options).");
    // A module function (not a method) so profilers see the kernel run as
    // one C call, as they see run_transient.
    m.def("_run_simulation",
          [](sim::Simulation& s) {
              py::gil_scoped_release rel;
              return s.run();
          },
          py::arg("simulation"),
          "Simulation.run() without the result handles: simulate()'s call.");
    m.def("_to_trbdf2_options",
          [](const RunOptions& o) { return trbdf2_dict(sim::to_trbdf2_options(o)); },
          py::arg("options"),
          "The variable-step driver's options resolved from a RunOptions, as "
          "a dict (sim::to_trbdf2_options).");

    // add-external-device-abi 3.4: the readings of record=True external
    // devices at every accepted step (res.device_output).
    py::class_<xdev::OutputTrace>(m, "_XdevTrace")
        .def(py::init<>())
        .def_static("wanted",
            [](const builder::CircuitBuilder& b) {
                for (const Index br : b.pool().external_device_branches()) {
                    if (b.pool().external_device_params(br).record) return true;
                }
                return false;
            }, py::arg("builder"))
        .def_property_readonly("times", [](const xdev::OutputTrace& tr) {
            return Eigen::Map<const Vector>(tr.times().data(),
                                            static_cast<Eigen::Index>(tr.times().size()));
        })
        .def("names", [](const xdev::OutputTrace& tr) {
            std::vector<std::string> n;
            for (const auto& d : tr.devices()) n.push_back(d.name);
            return n;
        })
        .def("columns", [](const xdev::OutputTrace& tr, const std::string& name) {
            return tr.device(name).columns;
        }, py::arg("name"))
        .def("values", [](const xdev::OutputTrace& tr, const std::string& name) {
            const auto& d = tr.device(name);
            const auto w = static_cast<Eigen::Index>(d.columns.size());
            using M = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            return M(Eigen::Map<const M>(d.rows.data(), w ? static_cast<Eigen::Index>(d.rows.size()) / w : 0, w));
        }, py::arg("name"))
        .def("start_row", &xdev::OutputTrace::start_row, py::arg("name"), py::arg("x0"),
             py::arg("t0"));

    py::class_<sim::Simulation>(m, "Simulation", py::dynamic_attr(),
        "The kernel object that is a simulation: Simulation(circuit, "
        "options) plans the run (preflight on a private copy, routing, the "
        "option x engine refusals, the device capability check, the Newton "
        "stamps); attach gate sources, ticks and source terms; run() or "
        "run_until(t) + snapshot() / restore(). Engines 'pwl' and "
        "'trbdf2'; the exact engine is refused (pulsim.simulate composes "
        "it). C blocks / graph-driven sources are refused (simulate "
        "composes them).")
        .def(py::init<ir::Circuit, RunOptions, bool>(), py::arg("circuit"),
             py::arg("options"), py::arg("controls_composed") = false,
             "controls_composed=True: the caller drives the circuit's C "
             "blocks / graph-driven sources through the attachments (as "
             "pulsim.simulate() does); otherwise they are refused.")
        .def("set_switch_fn",
            [](py::object self, py::object fn) {
                // Holds the callable (replacing the previous one): the
                // kernel keeps only a C++ handle to it.
                self.attr("_held_switch_fn") = fn;
                auto& s = self.cast<sim::Simulation&>();
                // Width-checked, as simulate() and the raw drivers check it.
                auto a = pulsim_bindings_common::adapt_switch_fn(
                    fn, s.builder().graph().num_switches());
                std::function<Real(Real)> edge = std::move(a.next_edge);
                // A Python schedule's declared edges: the fixed step reads
                // them (as run_transient does); the variable step lands
                // native edges only (as run_transient_trbdf2 does).
                if (!edge && s.plan().engine == sim::Engine::FixedTrap
                    && py::hasattr(fn, "next_edge_after")) {
                    edge = fn.attr("next_edge_after")
                               .cast<std::function<Real(Real)>>();
                }
                s.set_switch_fn(std::move(a.fn), std::move(edge));
            },
            py::arg("switch_fn"),
            "The gate source: t -> SwitchStateMask (a native schedule runs "
            "in C++). Kept alive by the Simulation.")
        .def("set_source_term",
            [](py::object self, py::object fn) {
                // Holds the callable (replacing the previous one): the
                // kernel keeps only a C++ handle to it.
                self.attr("_held_source_term") = fn;
                auto& s = self.cast<sim::Simulation&>();
                solver::BExtraFn b;
                std::function<Real(Real)> jump;
                pulsim_bindings_common::resolve_b_extra(fn, b, jump);
                s.set_source_term(std::move(b), std::move(jump));
            },
            py::arg("b_extra_fn"))
        .def("set_step_observer",
            [](py::object self, py::object fn) {
                // Holds the callable (replacing the previous one): the
                // kernel keeps only a C++ handle to it.
                self.attr("_held_step_observer") = fn;
                auto& s = self.cast<sim::Simulation&>();
                s.set_step_observer(pulsim_bindings_common::step_fn(fn));
            },
            py::arg("step_observer"))
        .def("add_tick",
            [](sim::Simulation& s, Real period, py::object fn, Real offset,
               bool monitor) {
                s.add_tick(period, pulsim_bindings_common::step_fn(fn), offset,
                           monitor);
            },
            py::arg("period"), py::arg("fn"), py::arg("offset") = 0.0,
            py::arg("monitor") = false, py::keep_alive<1, 3>(),
            "A tick at t_start + offset + k*period. monitor=True: a "
            "Kind::Monitor (trbdf2 neither lands on it nor re-samples the "
            "gate after it; pwl ticks it on its grid as any tick).")
        .def("set_tick_before",
            [](py::object self, py::object fn) {
                // Holds the callable (replacing the previous one): the
                // kernel keeps only a C++ handle to it.
                self.attr("_held_tick_before") = fn;
                auto& s = self.cast<sim::Simulation&>();
                s.set_tick_before(pulsim_bindings_common::step_fn(fn));
            },
            py::arg("fn"),
            "pwl: runs at every tick instant before the ticks.")
        .def("set_loop_feed",
            [](py::object self, py::object feed) {
                // add-probes-and-recorders 0b.6: the recorders' hooks; the
                // object holds the feed so the hooks outlive every run
                auto& f = feed.cast<record::LoopFeed&>();
                auto& s = self.cast<sim::Simulation&>();
                f.set_circuit(s.builder().graph(), s.builder().pool());
                self.attr("_held_feed") = feed;
                s.set_hooks(&f.hooks());
                if (py::hasattr(self, "_held_xdev_trace")) {
                    self.attr("_held_xdev_trace").cast<xdev::OutputTrace&>().attach(f.hooks());
                }
            },
            py::arg("feed"))
        .def("clear_loop_feed",
            [](py::object self) {
                // after a run (it may have thrown): the next run on this
                // object must not feed a finished record
                self.cast<sim::Simulation&>().set_hooks(nullptr);
                self.attr("_held_feed") = py::none();
            })
        .def("set_xdev_trace",
            [](py::object self, py::object trace) {
                // on the recorders' hooks when they are set (one hook set per
                // run), else on the trace's own; the object holds the trace
                if (py::hasattr(self, "_held_xdev_trace")) {
                    throw py::value_error("set_xdev_trace: this Simulation already has one");
                }
                auto& tr = trace.cast<xdev::OutputTrace&>();
                self.attr("_held_xdev_trace") = trace;
                // clear_loop_feed leaves _held_feed = None
                if (py::hasattr(self, "_held_feed") && !self.attr("_held_feed").is_none()) {
                    tr.attach(self.attr("_held_feed").cast<record::LoopFeed&>().hooks());
                } else {
                    tr.attach(tr.hooks());
                    self.cast<sim::Simulation&>().set_hooks(&tr.hooks());
                }
            },
            py::arg("trace"))
        .def("set_live_ring", &sim::Simulation::set_live_ring,
             py::arg("ring"))
        .def_property_readonly("cache", &sim::Simulation::cache,
             py::return_value_policy::reference_internal,
             "The run's PWL cache (built on first use, kept across runs).")
        .def_property_readonly("builder", &sim::Simulation::builder,
             py::return_value_policy::reference_internal,
             "The builder of the circuit the run solves (read-only use).")
        .def("set_cancel",
            [](py::object self, py::object fn) {
                // Holds the callable (replacing the previous one): the
                // kernel keeps only a C++ handle to it.
                self.attr("_held_cancel") = fn;
                auto& s = self.cast<sim::Simulation&>();
                s.set_cancel(fn.cast<solver::ShouldContinueFn>());
            },
            py::arg("should_continue"))
        .def("set_initial_state", &sim::Simulation::set_initial_state,
             py::arg("x"))
        .def("run",
            [](sim::Simulation& s) {
                py::gil_scoped_release rel;
                return s.run();
            })
        .def("run_until",
            [](sim::Simulation& s, Real t) {
                py::gil_scoped_release rel;
                return s.run_until(t);
            },
            py::arg("t"))
        .def("snapshot", &sim::Simulation::snapshot)
        .def("restore", &sim::Simulation::restore, py::arg("snapshot"),
             py::arg("at_snapshot_time") = true)
        .def_property_readonly("circuit", &sim::Simulation::circuit,
            "The circuit the run solves: the caller's plus any preflight "
            "ties (the caller's circuit is never changed).")
        .def_property_readonly("plan", [](const sim::Simulation& s) {
                const auto& p = s.plan();
                py::dict d;
                d["engine"] = std::string(pwl::engine_name(p.engine));
                d["engine_asked"] = engine_str(p.asked);
                d["route_reasons"] = p.route_reasons;
                py::list ties;
                for (const auto& t : p.ties) {
                    ties.append(py::make_tuple(t.name, t.node, t.resistance));
                }
                d["ties"] = ties;
                d["undriven"] = p.undriven;
                return d;
            })
        .def_property_readonly("diagnostics", [](const sim::Simulation& s) {
                py::list out;
                for (const auto& d : s.diagnostics()) {
                    py::dict r;
                    r["severity"] = d.severity == sim::Severity::Error ? "error"
                                    : d.severity == sim::Severity::Info ? "info"
                                                                        : "warning";
                    r["code"] = d.code;
                    r["message"] = d.message;
                    r["device"] = d.device;
                    r["t"] = d.t;
                    out.append(r);
                }
                return out;
            })
        .def_property_readonly("trbdf2_stats", [](const sim::Simulation& s) {
                return pulsim_bindings_common::trbdf2_stats_dict(s.trbdf2_stats());
            });
}

}  // namespace pulsim_sim_bindings
