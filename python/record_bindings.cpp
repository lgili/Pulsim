// add-probes-and-recorders: Python access to the kernel probes and
// recorders (core/include/pulsim/record/). Python feeds the recorder with
// a kernel-side pass over the stored result arrays (feed_table), the
// active feeding mode until the unified stepper's per-step hook lands.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsim/record/loop_feed.hpp"
#include "pulsim/record/recorder.hpp"

#include <memory>
#include <string>
#include <tuple>
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


rec::Recorder make_recorder(const std::vector<std::string>& probe_texts, const std::string& keep,
                            Size every_n, const std::vector<std::pair<Real, Real>>& windows,
                            Real period, Real offset, const std::vector<Real>& sample_times) {
    std::vector<rec::Probe> probes;
    probes.reserve(probe_texts.size());
    for (const auto& s : probe_texts) probes.push_back(rec::Probe::compile(s));
    rec::RecorderSpec spec;
    if (keep == "every") spec.keep = rec::RecorderSpec::Keep::Every;
    else if (keep == "decimate") spec.keep = rec::RecorderSpec::Keep::Decimate;
    else if (keep == "none") spec.keep = rec::RecorderSpec::Keep::None;
    else if (keep == "times") spec.keep = rec::RecorderSpec::Keep::Times;
    else throw std::invalid_argument("keep must be 'every', 'decimate', 'none' or 'times'");
    spec.sample_times = sample_times;
    spec.every_n = every_n;
    for (const auto& [a, b] : windows) spec.windows.push_back({a, b});
    spec.period = period;
    spec.offset = offset;
    return rec::Recorder(std::move(probes), std::move(spec));
}

py::dict recording_dict(const rec::Recorder& r) {
    const Size np = r.n_probes();
    const auto ne = static_cast<py::ssize_t>(r.event_times().size());
    py::dict out;
    out["times"] = py::array_t<Real>(static_cast<py::ssize_t>(r.times().size()), r.times().data());
    out["values"] = matrix(r.values(), r.times().size(), np);
    out["event_t"] = py::array_t<Real>(ne, r.event_times().data());
    out["event_t_right"] = py::array_t<Real>(ne, r.event_t_right().data());
    py::array_t<std::uint8_t> res(ne);
    std::copy(r.event_resolution().begin(), r.event_resolution().end(), res.mutable_data());
    out["event_resolution"] = res;
    out["event_left"] = matrix(r.event_left(), r.event_times().size(), np);
    out["event_right"] = matrix(r.event_right(), r.event_times().size(), np);
    out["windows"] = stats(r.windows(), np);
    out["periods"] = stats(r.periods(), np);
    out["n_samples"] = r.n_samples();
    py::array_t<bool> interp(static_cast<py::ssize_t>(r.interpolated().size()));
    std::copy(r.interpolated().begin(), r.interpolated().end(), interp.mutable_data());
    out["interpolated"] = interp;
    out["n_times_unreached"] = r.n_times_unreached();
    out["capacity"] = py::make_tuple(r.times().capacity(), r.values().capacity());
    return out;
}

rec::LinearReading linear(const std::vector<pulsim::Index>& cols, const std::vector<Real>& coef,
                          Real div) {
    rec::LinearReading l;
    l.cols = cols;
    l.coef = coef;
    l.div = div;
    return l;
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
             const F64& readings, const std::vector<Size>& event_rows, const F64& right,
             const std::vector<Real>& sample_times) {
              rec::Recorder r = make_recorder(probe_texts, keep, every_n, windows, period,
                                              offset, sample_times);
              r.bind(reading_names);
              const Size m_r = reading_names.size();
              {
                  py::gil_scoped_release nogil;
                  rec::feed_table(r, {t.data(), static_cast<Size>(t.size())},
                                  {readings.data(), static_cast<Size>(readings.size())}, m_r,
                                  event_rows, {right.data(), static_cast<Size>(right.size())});
              }
              return recording_dict(r);
          },
          py::arg("probes"), py::arg("keep"), py::arg("every_n"), py::arg("windows"),
          py::arg("period"), py::arg("offset"), py::arg("reading_names"), py::arg("t"),
          py::arg("readings"), py::arg("event_rows"), py::arg("right"),
          py::arg("sample_times") = std::vector<Real>{},
          "Kernel recorder pass over stored arrays (add-probes-and-recorders); "
          "use pulsim.simulate(..., recorders=[...]).");

    // (a cols, a coef, a div, product, b cols, b coef, b div, device branch
    // or -1, device power, constant source current)
    using ReadingSpec = std::tuple<std::vector<pulsim::Index>, std::vector<Real>, Real, bool,
                                   std::vector<pulsim::Index>, std::vector<Real>, Real,
                                   pulsim::Index, bool, Real>;
    using RecSpec = std::tuple<std::vector<std::string>, std::string, Size,
                               std::vector<std::pair<Real, Real>>, Real, Real, std::vector<Real>>;
    py::class_<rec::LoopFeed>(m, "_LoopFeed",
                     "In-loop recorder feed (add-probes-and-recorders 0b.6); "
                     "use pulsim.simulate(..., recorders=[...]).")
        .def(py::init([](const std::vector<std::string>& names,
                         const std::vector<ReadingSpec>& readings,
                         const std::vector<RecSpec>& specs) {
                 std::vector<rec::StateReading> rs;
                 for (const auto& [ca, wa, da, prod, cb, wb, db, dev, dev_p, dev_v] : readings) {
                     rec::StateReading r;
                     r.a = linear(ca, wa, da);
                     r.product = prod;
                     r.b = linear(cb, wb, db);
                     r.dev.branch = dev;
                     r.dev.power = dev_p;
                     r.dev.value = dev_v;
                     rs.push_back(std::move(r));
                 }
                 std::vector<rec::Recorder> recs;
                 for (const auto& [pr, keep, every, win, period, offset, times] : specs) {
                     recs.push_back(make_recorder(pr, keep, every, win, period, offset, times));
                 }
                 return std::make_unique<rec::LoopFeed>(names, std::move(rs), std::move(recs));
             }),
             py::arg("reading_names"), py::arg("readings"), py::arg("recorders"))
        .def("reserve", &rec::LoopFeed::reserve,
             py::arg("n_samples"), py::arg("t_last"), py::arg("n_events") = 0)
        .def("arm", &rec::LoopFeed::attach_own, py::arg("bracket_events"),
             "Register on the feed's own hooks; Simulation.set_loop_feed passes them.")
        .def("finish", &rec::LoopFeed::finish)
        .def("save", &rec::LoopFeed::save, "The feed's run state (snapshot field 18).")
        .def("restore", [](rec::LoopFeed& f, const std::vector<Real>& s) { f.restore(s); },
             py::arg("state"), "Continue from save()'s output.")
        .def_property_readonly("restored", &rec::LoopFeed::restored)
        .def_property_readonly("armed", &rec::LoopFeed::attached)
        .def_property_readonly("n_steps", &rec::LoopFeed::n_steps)
        .def_property_readonly("n_jumps", &rec::LoopFeed::n_jumps)
        .def_property_readonly("n_events", &rec::LoopFeed::n_events)
        .def_property_readonly("n_dropped", &rec::LoopFeed::n_dropped)
        .def("result", [](const rec::LoopFeed& f, Size k) { return recording_dict(f.recorders().at(k)); },
             py::arg("k"));
}

}  // namespace pulsim_record_bindings
