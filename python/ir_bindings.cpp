// =============================================================================
// add-immutable-circuit-ir — Python bindings of `pulsim.Circuit`
// =============================================================================
//
// A separate translation unit (like dsed_bindings.cpp) so the IR's hunk
// in bindings.cpp is three lines. `Circuit` exposes queries and
// derivations only: nothing reachable from it mutates its data. The
// lowered builder is never handed out; `to_builder()` returns a private
// clone.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/ir/circuit.hpp"
#include "pulsim/ir/expr.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/yaml/emitter.hpp"
#include "switch_fn_adapter.hpp"

#include <atomic>
#include <exception>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <thread>

namespace py = pybind11;

namespace pulsim_ir_bindings {

using pulsim::builder::CircuitBuilder;
using pulsim::ir::Circuit;

namespace {

/// `CircuitBuilder.components` as bound in C++, captured at module init:
/// a later Python patch of the class attribute never runs against a
/// circuit's shared lowered builder. The temporary handle passed to it
/// does not own the builder and does not outlive the call.
py::object* g_components = nullptr;  // leaked: outlives interpreter teardown

py::object lowered_components(const Circuit& c) {
    return (*g_components)(py::cast(&c.lowered(), py::return_value_policy::reference));
}

// ---- Group 3: control declarations carry a Python object ------------------
std::shared_ptr<const void> py_handle(const py::object& obj) {
    if (obj.is_none()) return {};
    // The last reference may drop on a thread without the GIL (a derived
    // circuit dying inside a sweep worker): take it to decref; leak at
    // interpreter shutdown rather than touch a finalised interpreter.
    return std::shared_ptr<const void>(new py::object(obj), [](const void* p) {
        if (!Py_IsInitialized()) return;
        py::gil_scoped_acquire gil;
        delete static_cast<const py::object*>(p);
    });
}

/// `with builder.scope(path, kind):`.
struct BuilderScope {
    CircuitBuilder* b;
    std::string path;
    std::string kind;
};

py::list scopes_to_py(std::span<const pulsim::ir::Scope> sc) {
    py::list out;
    for (const auto& x : sc) {
        py::dict e;
        e["path"] = x.path;
        e["name"] = x.name;
        e["kind"] = x.kind;
        e["parent"] = x.parent == 0 ? std::string{} : sc[x.parent - 1].path;
        out.append(e);
    }
    return out;
}

py::list controls_to_py(std::span<const pulsim::ir::ControlDecl> cs) {
    py::list out;
    for (const auto& c : cs) {
        out.append(py::make_tuple(
            c.name, c.kind, c.content_hash,
            c.handle ? *static_cast<const py::object*>(c.handle.get())
                     : py::object(py::none())));
    }
    return out;
}

// ---- Task 5.2: the kernel path of `sweep(circuit, ...)` -------------------
//
// One point = derive (`with_params` / `with_value`), regularise, clone,
// cache, run: what `simulate(circuit.with_*(...), **kw)` does on the pwl
// engine, here without the GIL. Nothing reachable from `base` is
// written: derivations clone, the graph's id cache was filled at
// construction, the shared `SymbolicPlan` is mutex-guarded; each point's
// builder and cache are its own.
using RowTable = Eigen::Matrix<pulsim::Real, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;

Circuit derive_point(const Circuit& base, const std::vector<std::string>& names,
                     const pulsim::Real* row) {
    std::vector<std::pair<std::string_view, pulsim::Real>> ps;
    for (std::size_t j = 0; j < names.size(); ++j) {
        if (names[j].find('.') == std::string::npos) ps.emplace_back(names[j], row[j]);
    }
    Circuit c = ps.empty() ? base : base.with_params(ps);
    for (std::size_t j = 0; j < names.size(); ++j) {
        if (names[j].find('.') != std::string::npos) c = c.with_value(names[j], row[j]);
    }
    return c;
}

struct PointRun {
    std::optional<Circuit> circuit;
    std::optional<CircuitBuilder> builder;
    std::optional<pulsim::solver::SimulationResult> result;
    std::optional<pulsim::ir::Circuit::Regularized> own_preflight;  // structure changed
};

py::tuple run_pwl_points(const Circuit& base, const std::vector<std::string>& names,
                         const RowTable& table,
                         const pulsim::solver::SimulationOptions& opts,
                         pulsim::Real dt, const py::object& switch_fn_obj,
                         bool start_from_dc_op, std::optional<bool> /*enable_nl*/,
                         std::optional<pulsim::Vector> initial_state,
                         bool auto_regularize, int n_threads) {
    if (table.cols() != static_cast<Eigen::Index>(names.size())) {
        throw std::invalid_argument(std::format(
            "_run_pwl_points: {} names but {} table columns", names.size(),
            table.cols()));
    }
    auto sw = pulsim_bindings_common::adapt_switch_fn(switch_fn_obj);
    const auto n = static_cast<std::size_t>(table.rows());
    std::vector<PointRun> runs(n);
    std::atomic<std::size_t> next{0};
    std::atomic<std::uint64_t> analyses{0}, preflights{0};
    std::optional<Circuit::Regularized> reg0;
    {
        py::gil_scoped_release rel;
        // Regularise once per structure: the preflight reads topology and
        // kinds only (pwl/preflight.hpp), so a point whose structure_hash
        // equals the base's needs the base's ties, nothing else. A point
        // derived from the regularised base equals, bit for bit, the
        // regularised point: the ties are appended after every device in
        // both, with the same names and resistance.
        const pulsim::pwl::PreflightOptions pf{.auto_regularize = true};
        if (auto_regularize) {
            reg0 = base.with_preflight(pf);
            preflights.fetch_add(1);
        }
        const Circuit& root = reg0 ? reg0->circuit : base;
        auto work = [&] {
            for (std::size_t i; (i = next.fetch_add(1)) < n;) {
                auto& r = runs[i];
                try {
                    Circuit c = derive_point(root, names, table.row(i).data());
                    if (reg0 && c.structure_hash() != root.structure_hash()) {
                        // with_value on a branch-form R_on / g_off (to or
                        // from 0) changes the structure: preflight again.
                        r.own_preflight = derive_point(base, names, table.row(i).data())
                                              .with_preflight(pf);
                        preflights.fetch_add(1);
                        c = r.own_preflight->circuit;
                    }
                    CircuitBuilder b = c.to_builder();
                    pulsim::pwl::PwlStateSpaceCache cache(b.graph(), b.pool());
                    cache.share_symbolic(c.symbolic_plan());
                    cache.build_lazy(dt);
                    pulsim::Vector x0;
                    const pulsim::Vector* xp = nullptr;
                    if (initial_state) {
                        xp = &*initial_state;
                    } else {  // simulate(): the builder's ICs when any is set
                        x0 = b.initial_state();
                        if ((x0.array() != 0.0).any()) xp = &x0;
                    }
                    // None / True: the kernel builds the Newton stamps
                    // (False is not sent here).
                    const pulsim::pwl::NonlinearRefreshFn nl{};
                    r.result = pulsim::solver::run_transient(
                        cache, b.graph(), b.pool(), opts, sw.fn, {},
                        start_from_dc_op, nl, {}, xp, {}, nullptr, {}, sw.next_edge);
                    analyses.fetch_add(cache.metrics().symbolic_analyses);
                    r.circuit = std::move(c);
                    r.builder = std::move(b);
                } catch (...) {
                    // The caller re-runs this point through simulate(),
                    // which raises or records exactly as the Python path.
                    r = PointRun{};
                }
            }
        };
        const auto nt = static_cast<std::size_t>(std::max(1, n_threads));
        std::vector<std::thread> pool;
        for (std::size_t t = 1; t < std::min(nt, n); ++t) {
            try { pool.emplace_back(work); } catch (...) { break; }  // fewer threads
        }
        work();
        for (auto& th : pool) th.join();
    }
    py::list out;
    for (auto& r : runs) {
        if (!r.result) { out.append(py::none()); continue; }
        py::object own = py::none();
        if (r.own_preflight) {
            py::list ties;
            for (const auto& t : r.own_preflight->ties) {
                ties.append(py::make_tuple(t.name, t.node, t.resistance));
            }
            own = py::make_tuple(std::move(r.own_preflight->report), ties);
        }
        out.append(py::make_tuple(std::move(*r.result), std::move(*r.circuit),
                                  std::move(*r.builder), own));
    }
    py::dict stats;
    stats["symbolic_analyses"] = analyses.load();
    stats["preflights"] = preflights.load();
    py::list ties;  // the base's regularisation, shared by its structure
    if (reg0) {
        for (const auto& t : reg0->ties) {
            ties.append(py::make_tuple(t.name, t.node, t.resistance));
        }
    }
    stats["report"] = reg0 ? py::cast(std::move(reg0->report)) : py::none();
    stats["ties"] = ties;
    return py::make_tuple(out, stats);
}

}  // namespace

void init_module(py::module_& m) {
    m.def("_eval_expr", [](const std::string& text, const std::vector<std::string>& names,
                           const std::vector<pulsim::Real>& values, const std::string& who) {
            if (names.size() != values.size()) {
                throw std::invalid_argument("_eval_expr: names and values differ in length");
            }
            const auto e = pulsim::ir::Expr::compile(text,
                [&](std::string_view n) -> std::optional<std::uint32_t> {
                    for (std::size_t i = 0; i < names.size(); ++i) {
                        if (names[i] == n) return static_cast<std::uint32_t>(i);
                    }
                    return std::nullopt;
                }, who);
            return e.eval(values);
        }, py::arg("text"), py::arg("names"), py::arg("values"),
        py::arg("who") = std::string{"expression"},
        "Evaluate a parameter expression (ir/expr.hpp) over names = values.");
    g_components = new py::object(m.attr("CircuitBuilder").attr("components"));
    using pulsim::ir::Parameter;
    py::class_<Parameter>(m, "Parameter",
        "A named parameter (`builder.param(name, value, unit)`). Pass it "
        "to any `add_*` in place of a number: the field takes its value, "
        "and `Circuit.with_params` re-values it.")
        .def_readonly("name", &Parameter::name)
        .def_readonly("value", &Parameter::value)
        .def_readonly("unit", &Parameter::unit)
        .def_readonly("lo", &Parameter::lo)
        .def_readonly("hi", &Parameter::hi)
        .def_readonly("expr", &Parameter::expr,
                      "The defining expression, or \"\" for a literal.")
        .def("__repr__", [](const Parameter& p) {
            return std::format("<pulsim.Parameter {} = {}{}{}>", p.name,
                               p.value, p.unit.empty() ? "" : " ", p.unit);
        });

    py::class_<Circuit>(m, "Circuit",
        "Immutable circuit (add-immutable-circuit-ir). Built with "
        "`builder.build()`; a handle to shared, read-only data, cheap to "
        "copy and safe to share between threads. `simulate(circuit, ...)` "
        "runs it; the builder it came from can keep being edited.")
        .def("structure_hash", &Circuit::structure_hash,
             "64-bit hash of topology, kinds and names.")
        .def("value_hash", &Circuit::value_hash,
             "64-bit hash of the structure and every value (fields, "
             "couplings, initial conditions, aliases).")
        .def("to_yaml", [](const Circuit& c) { return pulsim::ir::to_yaml(c); },
             py::call_guard<py::gil_scoped_release>(),
             "The lowered circuit as YAML; `Circuit.from_yaml` reads it back "
             "with the same `value_hash`. Refuses a C block (a Python "
             "callable).")
        .def_static("from_yaml", [](const std::string& text) { return pulsim::ir::from_yaml(text); },
                    py::arg("text"), py::call_guard<py::gil_scoped_release>(),
                    "Inverse of `to_yaml`; errors name the device and the key.")
        .def("clone", [](const Circuit& c) { return c; },
             "Another handle to the same immutable data (no copy).")
        .def(py::pickle(
            [](const Circuit& c) { return py::make_tuple(pulsim::ir::to_yaml(c)); },
            [](const py::tuple& t) {
                if (t.size() != 1) throw std::runtime_error("Circuit: bad pickle state");
                return pulsim::ir::from_yaml(t[0].cast<std::string>(), /*bounds=*/false);
            }))
        .def_property_readonly("nodes", [](const Circuit& c) {
                py::list out;
                for (const auto& n : c.nodes()) out.append(n.name);
                return out;
            }, "Node names; entry i is unknown i.")
        .def("devices", [](const Circuit& c) {
                py::list out;
                for (const auto& d : c.devices()) {
                    py::dict fields;
                    const auto row = pulsim::ir::kKindFields[d.kind];
                    py::dict params;
                    for (const auto& f : d.fields) {
                        const auto n = py::str(std::string{row[f.index].name});
                        fields[n] = c.evaluate(f.value);
                        if (const auto* id = std::get_if<pulsim::ir::ParamId>(&f.value)) {
                            params[n] = c.parameters()[id->v].name;
                        }
                    }
                    py::dict e;
                    e["name"] = d.name;
                    e["kind"] = std::string{pulsim::pwl::kKindCaps[d.kind].name};
                    e["pins"] = d.pins;
                    e["fields"] = fields;
                    e["params"] = params;
                    out.append(e);
                }
                return out;
            },
            "One dict per lowered branch, in insertion order: name, kind "
            "(capability-table name), pins (node indices, -1 = ground), "
            "fields ({name: value}, the names `with_value` takes) and "
            "params ({field: parameter name} for the fields a parameter "
            "drives).")
        .def("parameters", [](const Circuit& c) {
                return std::vector<pulsim::ir::Parameter>(
                    c.parameters().begin(), c.parameters().end());
            }, "The named parameters, in declaration order.")
        .def("scopes", [](const Circuit& c) { return scopes_to_py(c.scopes()); },
             "The scope tree (subsystem instances), parents first: one dict "
             "per scope with path, name, kind (the subsystem definition, \"\" "
             "for a plain path segment) and parent (its path, \"\" = root).")
        .def("node_domain", [](const Circuit& c, std::string_view n) {
                return std::string{pulsim::ir::domain_name(c.node_info(n).domain)};
            }, py::arg("name"),
            "Domain of node `name`: declared, implied by a typed pin (a "
            "PMSM's speed and angle are mechanical) or electrical.")
        .def("unit_of", [](const Circuit& c, std::string_view n) {
                return std::string{c.node_info(n).unit};
            }, py::arg("name"),
            "Across unit a probe on node `name` reports: V, degC (thermal), "
            "rad/s or rad (a PMSM's speed / angle node).")
        .def("scope_of", [](const Circuit& c, std::string_view n) {
                return std::string{c.lowered().scope_path(c.scope_of(n))};
            }, py::arg("name"),
            "Path of the scope a device, node or parameter named `name` "
            "belongs to (\"\" = the root).")
        .def("with_value", &Circuit::with_value,
             py::arg("device_dot_field"), py::arg("value"),
             py::call_guard<py::gil_scoped_release>(),
             "A new circuit with one field set, e.g. with_value(\"R1.R\", "
             "2.0): equal, bit for bit, to rebuilding with that literal. "
             "Every field of every kind (see devices()); the receiver is "
             "unchanged. ValueError names device and field on an unknown "
             "name or an out-of-bound value.")
        .def("with_params", [](const Circuit& c, const py::dict& values,
                               const py::kwargs& kw) {
                std::vector<std::string> names;
                std::vector<std::pair<std::string_view, pulsim::Real>> vs;
                for (const auto* d : {&values, static_cast<const py::dict*>(&kw)}) {
                    for (const auto& [k, v] : *d) {
                        names.push_back(k.cast<std::string>());
                    }
                }
                std::size_t i = 0;
                for (const auto* d : {&values, static_cast<const py::dict*>(&kw)}) {
                    for (const auto& [k, v] : *d) {
                        vs.emplace_back(names[i++], v.cast<pulsim::Real>());
                    }
                }
                py::gil_scoped_release rel;  // threads of a sweep
                return c.with_params(vs);
            }, py::arg("values") = py::dict(),
            "A new circuit with parameter values replaced: "
            "with_params({\"Lbase\": 2e-3}) or with_params(Lbase=2e-3). "
            "Every field bound to a parameter follows it; the structure "
            "(and structure_hash) is unchanged, the receiver too.")
        .def("initial_conditions", [](const Circuit& c) {
                py::dict out;
                for (const auto& ic : c.initial_conditions()) {
                    out[py::str(c.devices()[ic.device.v].name)] = ic.value;
                }
                return out;
            }, "{device name: value} of the recorded initial conditions.")
        .def("aliases", [](const Circuit& c) {
                py::dict out;
                for (const auto& a : c.aliases()) {
                    out[py::str(a.name)] =
                        py::make_tuple(a.node ? "node" : "branch", a.target);
                }
                return out;
            }, "{alias: (kind, target)}, as CircuitBuilder.aliases().")
        .def_property_readonly("num_switches", [](const Circuit& c) {
                return c.graph().num_switches();
            }, "Number of switching branches (the SwitchStateMask size).")
        .def("components", &lowered_components,
             "As CircuitBuilder.components().")
        .def("initial_state", [](const Circuit& c) {
                return c.lowered().initial_state();
            }, "As CircuitBuilder.initial_state().")
        .def("state_var_names", [](const Circuit& c) {
                return c.lowered().state_var_names();
            }, "As CircuitBuilder.state_var_names().")
        .def("switch_index_of", [](const Circuit& c, std::string_view n) {
                return c.lowered().switch_index_of(n);
            }, py::arg("name"), "As CircuitBuilder.switch_index_of().")
        .def("branch_id_of", [](const Circuit& c, std::string_view n) {
                return c.lowered().branch_id_of(n);
            }, py::arg("name"), "As CircuitBuilder.branch_id_of().")
        .def("node_id_of", [](const Circuit& c, std::string_view n) {
                return c.lowered().node_id_of(n);
            }, py::arg("name"), "As CircuitBuilder.node_id_of().")
        .def("get_value", [](const Circuit& c, std::string_view f) {
                return c.lowered().get_value(f);
            }, py::arg("device_dot_field"),
            "Value of one field, e.g. get_value(\"R1.R\").")
        .def("with_ties", [](const Circuit& c, const py::list& ties) {
                std::vector<pulsim::ir::Tie> ts;
                for (const auto& t : ties) {
                    auto tup = t.cast<py::tuple>();
                    ts.push_back(pulsim::ir::Tie{tup[0].cast<std::string>(),
                                                 tup[1].cast<pulsim::Index>(),
                                                 tup[2].cast<pulsim::Real>()});
                }
                py::gil_scoped_release rel;
                return c.with_ties(ts);
            }, py::arg("ties"),
            "The circuit with regularising ties added: a list of (name, "
            "node index, resistance) to ground, as `result.plan.ties` "
            "reports them. The receiver is unchanged.")
        .def("with_preflight", [](const Circuit& c,
                                  const pulsim::pwl::PreflightOptions& o) {
                auto r = [&] {
                    py::gil_scoped_release rel;
                    return c.with_preflight(o);
                }();
                py::list ties;
                for (const auto& t : r.ties) {
                    ties.append(py::make_tuple(t.name, t.node, t.resistance));
                }
                return py::make_tuple(std::move(r.circuit), std::move(r.report),
                                      ties);
            }, py::arg("options"),
            "Run the topology preflight on this circuit without changing "
            "it: returns (circuit, report, ties) — the circuit a run uses "
            "(this one when no tie was needed), the PreflightReport, and "
            "the inserted ties as (name, node index, resistance).")
        .def("controls", [](const Circuit& c) { return controls_to_py(c.controls()); },
             "The control declarations the circuit carries, as (name, "
             "kind, content_hash, handle) tuples: C blocks (kind "
             "\"c_block\", handle = the CBlockHandle) and graph-driven "
             "sources (\"graph_driven\"). simulate(circuit) runs them.")
        .def("with_controls", [](const Circuit& c, const py::list& decls) {
                std::vector<pulsim::ir::ControlDecl> ds;
                for (const auto& d : decls) {
                    auto t = d.cast<py::tuple>();
                    ds.push_back({t[0].cast<std::string>(), t[1].cast<std::string>(),
                                  t[2].cast<std::uint64_t>(),
                                  py_handle(py::reinterpret_borrow<py::object>(t[3]))});
                }
                return c.with_controls(ds);
            }, py::arg("decls"),
            "A new circuit with control declarations appended: a list of "
            "(name, kind, content_hash, handle). name and kind enter "
            "structure_hash, content_hash value_hash; the handle is "
            "passed through to the run. The receiver is unchanged.")
        .def("to_builder", &Circuit::to_builder,
             py::call_guard<py::gil_scoped_release>(),
             "A new, independent CircuitBuilder holding this circuit "
             "(the circuit is not affected by edits to it).")
        .def("__repr__", [](const Circuit& c) {
                return std::format(
                    "<pulsim.Circuit: {} nodes, {} devices, value_hash "
                    "{:016x}>", c.nodes().size(), c.devices().size(),
                    c.value_hash());
            });

    m.def("_run_pwl_points", &run_pwl_points, py::arg("circuit"),
          py::arg("names"), py::arg("table"), py::arg("options"),
          py::arg("dt"), py::arg("switch_fn"), py::arg("start_from_dc_op"),
          py::arg("enable_nonlinear_refresh"), py::arg("initial_state"),
          py::arg("auto_regularize"), py::arg("n_threads"),
          "Internal (sweep(circuit, ...) kernel path, task 5.2): run the "
          "points of a parameter table (rows = points, columns = names: "
          "parameters, or \"device.field\") on the pwl engine, n_threads "
          "native threads, without the GIL. Returns ([(result, circuit, "
          "builder, own_preflight) or None per point], stats); None = "
          "re-run that point through simulate().");

    py::reinterpret_borrow<py::class_<pulsim::pwl::PwlStateSpaceCache>>(
        m.attr("PwlStateSpaceCache"))
        .def("share_symbolic", [](pulsim::pwl::PwlStateSpaceCache& cache,
                                  const Circuit& c) {
                cache.share_symbolic(c.symbolic_plan());
            }, py::arg("circuit"),
            "Share symbolic analyses with every cache of a circuit of "
            "this structure (with_params / with_value derivations): one "
            "analysis per sweep. Call before the first build; results "
            "are bit-identical.");

    py::class_<BuilderScope>(m, "BuilderScope",
        "What CircuitBuilder.scope() returns: a context manager.")
        .def("__enter__", [](BuilderScope& s) {
                return std::string{s.b->scope_path(s.b->open_scope(s.path, s.kind))};
            })
        .def("__exit__", [](BuilderScope& s, const py::args&) {
                s.b->close_scope();
                return false;
            });
    auto builder = py::reinterpret_borrow<py::class_<CircuitBuilder>>(
        m.attr("CircuitBuilder"));
    builder
        .def("build", &CircuitBuilder::build,
             "Immutable snapshot of this builder (a pulsim.Circuit). The "
             "builder can keep being edited; the circuit does not change.")
        .def("set_value", &CircuitBuilder::set_value,
             py::arg("device_dot_field"), py::arg("value"),
             "Set one value field in place, e.g. set_value(\"R1.R\", 2.0); "
             "every field of every kind (see Circuit.devices()). Raises "
             "ValueError naming device and field on an unknown name or an "
             "out-of-bound value.")
        .def("get_value", &CircuitBuilder::get_value,
             py::arg("device_dot_field"),
             "Value of one field, e.g. get_value(\"R1.R\").")
        .def("param", [](CircuitBuilder& b, std::string_view name, const py::object& value,
                         std::string_view unit, std::optional<pulsim::Real> lo,
                         std::optional<pulsim::Real> hi) {
                if (py::isinstance<py::str>(value)) {
                    return b.param_expr(name, value.cast<std::string>(), unit, lo, hi);
                }
                return b.param(name, value.cast<pulsim::Real>(), unit, lo, hi);
            }, py::arg("name"),
             py::arg("value"), py::arg("unit") = std::string{},
             py::arg("lo") = std::nullopt, py::arg("hi") = std::nullopt,
             "Declare a named parameter and return it (a "
             "pulsim.Parameter); pass it to any add_* in place of a "
             "number. lo / hi bound its value. A str value is an "
             "expression over earlier parameters (\"2*Lbase\"), "
             "re-evaluated when they change.")
        .def("bind_param", &CircuitBuilder::bind_param,
             py::arg("device_dot_field"), py::arg("name"),
             "Drive one field by a parameter: bind_param(\"L1.L\", "
             "\"Lbase\"). add_* does this for a Parameter argument.")
        .def("set_param", &CircuitBuilder::set_param, py::arg("name"),
             py::arg("value"),
             "Re-value a parameter and every field bound to it, in place.")
        .def("controls", [](const CircuitBuilder& b) {
                return controls_to_py(b.controls());
            }, "Control declarations carried by this builder (a builder "
               "made by Circuit.to_builder()), as Circuit.controls().")
        .def("parameters", [](const CircuitBuilder& b) {
                return std::vector<pulsim::ir::Parameter>(
                    b.parameters().begin(), b.parameters().end());
            }, "The named parameters, in declaration order.")
        .def("node", [](CircuitBuilder& b, std::string_view name, std::string_view domain) {
                const auto d = pulsim::ir::domain_from_name(domain);
                if (!d) {
                    throw std::invalid_argument(std::format(
                        "node(\"{}\"): unknown domain \"{}\" (electrical, thermal, "
                        "mechanical)", name, domain));
                }
                return b.node(name, *d);
            }, py::arg("name"), py::arg("domain"),
            "Look up or create node `name` and declare its domain: "
            "\"electrical\", \"thermal\" or \"mechanical\". A pin of another "
            "domain connected to it is refused at build(), naming device, "
            "pin and node.")
        .def("open_scope", [](CircuitBuilder& b, std::string_view path,
                              std::string_view kind) {
                return b.scope_path(b.open_scope(path, kind));
            }, py::arg("path"), py::arg("kind") = std::string{},
            "Open scope `path` (relative to the open scope; returns its full "
            "path). Parameters declared until close_scope() belong to it and "
            "expressions look names up from it outward. Prefer "
            "`with builder.scope(path):`.")
        .def("scope", [](CircuitBuilder& b, std::string path, std::string kind) {
                return BuilderScope{&b, std::move(path), std::move(kind)};
            }, py::arg("path"), py::arg("kind") = std::string{},
            py::keep_alive<0, 1>(),
            "Context manager: `with builder.scope(\"leg_a\", kind=\"HB\"):` "
            "opens the scope (open_scope) and closes it on exit; `as` gives "
            "its full path. Parameters declared inside belong to it.")
        .def("close_scope", &CircuitBuilder::close_scope,
             "Return to the scope open before the last open_scope().")
        .def("scopes", [](const CircuitBuilder& b) { return scopes_to_py(b.scopes()); },
             "As Circuit.scopes().")
        .def_property_readonly("current_scope", [](const CircuitBuilder& b) {
                return std::string{b.scope_path(b.current_scope())};
            }, "Path of the open scope (\"\" = root).")
        .def("scoped_name", &CircuitBuilder::scoped_name, py::arg("local"),
             "`local` as named in the open scope: \"leg_a/sm17/C\".")
        .def("scope_of", [](const CircuitBuilder& b, std::string_view n) {
                return std::string{b.scope_path(b.scope_of(n))};
            }, py::arg("name"), "As Circuit.scope_of().");
}

}  // namespace pulsim_ir_bindings
