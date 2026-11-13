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
#include "pulsim/pwl/device_pool.hpp"

#include <format>
#include <string>

namespace py = pybind11;

namespace pulsim_ir_bindings {

using pulsim::builder::CircuitBuilder;
using pulsim::ir::Circuit;

namespace {

/// Call a read-only `CircuitBuilder` method of the circuit's lowered
/// builder through its Python binding (one implementation of
/// `components()`, `initial_state()`, … for both classes). The
/// temporary Python handle does not own the builder and does not
/// outlive the call; only const methods are named here.
py::object lowered_call(const Circuit& c, const char* method, py::args args) {
    auto handle = py::cast(&c.lowered(), py::return_value_policy::reference);
    return handle.attr(method)(*args);
}

}  // namespace

void init_module(py::module_& m) {
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
                    for (const auto& f : d.fields) {
                        fields[py::str(std::string{row[f.index].name})] =
                            std::get<pulsim::Real>(f.value);
                    }
                    py::dict e;
                    e["name"] = d.name;
                    e["kind"] = std::string{pulsim::pwl::kKindCaps[d.kind].name};
                    e["pins"] = d.pins;
                    e["fields"] = fields;
                    out.append(e);
                }
                return out;
            },
            "One dict per lowered branch, in insertion order: name, kind "
            "(capability-table name), pins (node indices, -1 = ground) and "
            "fields ({name: value}, the names `with_value` takes).")
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
        .def("components", [](const Circuit& c) {
                return lowered_call(c, "components", py::args{});
            }, "As CircuitBuilder.components().")
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
        .def("to_builder", &Circuit::to_builder,
             "A new, independent CircuitBuilder holding this circuit "
             "(the circuit is not affected by edits to it).")
        .def("__repr__", [](const Circuit& c) {
                return std::format(
                    "<pulsim.Circuit: {} nodes, {} devices, value_hash "
                    "{:016x}>", c.nodes().size(), c.devices().size(),
                    c.value_hash());
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
             "Value of one field, e.g. get_value(\"R1.R\").");
}

}  // namespace pulsim_ir_bindings
