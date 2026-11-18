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
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"

#include <format>
#include <string>

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

}  // namespace

void init_module(py::module_& m) {
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
        .def("with_value", &Circuit::with_value,
             py::arg("device_dot_field"), py::arg("value"),
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
                return c.with_ties(ts);
            }, py::arg("ties"),
            "The circuit with regularising ties added: a list of (name, "
            "node index, resistance) to ground, as `result.plan.ties` "
            "reports them. The receiver is unchanged.")
        .def("with_preflight", [](const Circuit& c,
                                  const pulsim::pwl::PreflightOptions& o) {
                auto r = c.with_preflight(o);
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
        .def("to_builder", &Circuit::to_builder,
             "A new, independent CircuitBuilder holding this circuit "
             "(the circuit is not affected by edits to it).")
        .def("__repr__", [](const Circuit& c) {
                return std::format(
                    "<pulsim.Circuit: {} nodes, {} devices, value_hash "
                    "{:016x}>", c.nodes().size(), c.devices().size(),
                    c.value_hash());
            });

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
        .def("param", &CircuitBuilder::param, py::arg("name"),
             py::arg("value"), py::arg("unit") = std::string{},
             py::arg("lo") = std::nullopt, py::arg("hi") = std::nullopt,
             "Declare a named parameter and return it (a "
             "pulsim.Parameter); pass it to any add_* in place of a "
             "number. lo / hi bound its value.")
        .def("bind_param", &CircuitBuilder::bind_param,
             py::arg("device_dot_field"), py::arg("name"),
             "Drive one field by a parameter: bind_param(\"L1.L\", "
             "\"Lbase\"). add_* does this for a Parameter argument.")
        .def("set_param", &CircuitBuilder::set_param, py::arg("name"),
             py::arg("value"),
             "Re-value a parameter and every field bound to it, in place.")
        .def("parameters", [](const CircuitBuilder& b) {
                return std::vector<pulsim::ir::Parameter>(
                    b.parameters().begin(), b.parameters().end());
            }, "The named parameters, in declaration order.");
}

}  // namespace pulsim_ir_bindings
