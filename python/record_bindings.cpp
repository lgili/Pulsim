// add-probes-and-recorders: Python access to the kernel probes and
// recorders (core/include/pulsim/record/). Python feeds the recorder with
// a kernel-side pass over the stored result arrays (feed_table), the
// active feeding mode until the unified stepper's per-step hook lands.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsim/record/recorder.hpp"

#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using pulsim::Real;
using pulsim::Size;
namespace rec = pulsim::record;

namespace pulsim_record_bindings {

namespace {

using F64 = py::array_t<Real, py::array::c_style | py::array::forcecast>;

py::array_t<Real> matrix(const std::vector<Real>& v, Size rows, Size cols) {
    py::array_t<Real> a({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

py::dict stats(const std::vector<rec::WindowStat>& w, Size np) {
    const Size n = np == 0 ? 0 : w.size() / np;
    const std::vector<std::pair<const char*, Real (*)(const rec::WindowStat&)>> fields{
        {"t0", [](const rec::WindowStat& s) { return s.t0; }},
        {"t1", [](const rec::WindowStat& s) { return s.t1; }},
        {"integral", [](const rec::WindowStat& s) { return s.integral; }},
        {"avg", [](const rec::WindowStat& s) { return s.avg(); }},
        {"rms", [](const rec::WindowStat& s) { return s.rms(); }},
        {"min", [](const rec::WindowStat& s) { return s.min; }},
        {"t_min", [](const rec::WindowStat& s) { return s.t_min; }},
        {"max", [](const rec::WindowStat& s) { return s.max; }},
        {"t_max", [](const rec::WindowStat& s) { return s.t_max; }},
        {"complete", [](const rec::WindowStat& s) { return s.complete() ? Real{1} : Real{0}; }},
    };
    py::dict d;
    for (const auto& [name, get] : fields) {
        py::array_t<Real> a({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(np)});
        auto* out = a.mutable_data();
        for (Size k = 0; k < w.size(); ++k) out[k] = get(w[k]);
        d[name] = a;
    }
    return d;
}

}  // namespace

void init_module(py::module_& m) {
    m.def("_probe_atoms", [](const std::string& text) {
        const auto p = rec::Probe::compile(text);
        return std::vector<std::string>(p.atoms().begin(), p.atoms().end());
    }, py::arg("text"),
    "The readings of a probe expression, normalised, in first-use order "
    "(add-probes-and-recorders). Raises ValueError naming the probe.");

    m.def("_record_pass",
          [](const std::vector<std::string>& probe_texts, const std::string& keep,
             Size every_n, const std::vector<std::pair<Real, Real>>& windows, Real period,
             Real offset, const std::vector<std::string>& reading_names, const F64& t,
             const F64& readings, const std::vector<Size>& event_rows, const F64& right) {
              std::vector<rec::Probe> probes;
              probes.reserve(probe_texts.size());
              for (const auto& s : probe_texts) probes.push_back(rec::Probe::compile(s));
              rec::RecorderSpec spec;
              if (keep == "every") spec.keep = rec::RecorderSpec::Keep::Every;
              else if (keep == "decimate") spec.keep = rec::RecorderSpec::Keep::Decimate;
              else if (keep == "none") spec.keep = rec::RecorderSpec::Keep::None;
              else throw std::invalid_argument("keep must be 'every', 'decimate' or 'none'");
              spec.every_n = every_n;
              for (const auto& [a, b] : windows) spec.windows.push_back({a, b});
              spec.period = period;
              spec.offset = offset;
              rec::Recorder r(std::move(probes), std::move(spec));
              r.bind(reading_names);
              const Size m_r = reading_names.size();
              {
                  py::gil_scoped_release nogil;
                  rec::feed_table(r, {t.data(), static_cast<Size>(t.size())},
                                  {readings.data(), static_cast<Size>(readings.size())}, m_r,
                                  event_rows, {right.data(), static_cast<Size>(right.size())});
              }
              const Size np = r.n_probes();
              py::dict out;
              out["times"] = py::array_t<Real>(static_cast<py::ssize_t>(r.times().size()),
                                               r.times().data());
              out["values"] = matrix(r.values(), r.times().size(), np);
              out["event_t"] = py::array_t<Real>(static_cast<py::ssize_t>(r.event_times().size()),
                                                 r.event_times().data());
              out["event_left"] = matrix(r.event_left(), r.event_times().size(), np);
              out["event_right"] = matrix(r.event_right(), r.event_times().size(), np);
              out["windows"] = stats(r.windows(), np);
              out["periods"] = stats(r.periods(), np);
              out["n_samples"] = r.n_samples();
              return out;
          },
          py::arg("probes"), py::arg("keep"), py::arg("every_n"), py::arg("windows"),
          py::arg("period"), py::arg("offset"), py::arg("reading_names"), py::arg("t"),
          py::arg("readings"), py::arg("event_rows"), py::arg("right"),
          "Kernel recorder pass over stored arrays (add-probes-and-recorders); "
          "use pulsim.simulate(..., recorders=[...]).");
}

}  // namespace pulsim_record_bindings
