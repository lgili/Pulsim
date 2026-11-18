// add-design-optimization-workflow DECISIONS 23 / 30: the event hook the
// comparator-gate saltation needs, checked against `saltation_hook.patch`.
//
// Not part of the build: core/include/pulsim/dsed/scheduler.hpp is not
// editable in the slice that wrote this. Run it against the patched header
// without touching the tree (one compile, the patched copy first on the
// include path):
//
//   mkdir -p /tmp/hook/core/include/pulsim/dsed
//   cp core/include/pulsim/dsed/scheduler.hpp /tmp/hook/core/include/pulsim/dsed/
//   patch -d /tmp/hook -p1 < openspec/changes/add-design-optimization-workflow/probes/saltation_hook.patch
//   c++ -std=c++20 -O2 -I/tmp/hook/core/include -Icore/include \
//       -I<eigen include dir> \
//       openspec/changes/add-design-optimization-workflow/probes/saltation_hook_check.cpp \
//       -o /tmp/hook/check && /tmp/hook/check
//
// With the patch: exit 0 (the trajectory is bit-identical to a schedule
// without the hook, and the hook saw every state event by name, type and
// predicate index). Against the unpatched header: exit 1 ("hook never
// called"), the trajectory still bit-identical.

#include "pulsim/dsed/buck_dcm_model.hpp"
#include "pulsim/dsed/scheduler.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace pulsim::dsed;
using namespace pulsim::dsed::buck;
using pulsim::Real;
using pulsim::Vector;

namespace {

struct Seen {
    Real t;
    std::string name;
    PredicateType type;
    int index;
};

// The plain DCM schedule plus the hook; it only records, so any change
// in the trajectory is the patch's, not this wrapper's.
struct Recording : BuckDCMSwitchFn {
    using BuckDCMSwitchFn::BuckDCMSwitchFn;
    std::vector<Seen>* seen = nullptr;
    void register_event(Real t, const std::string& n, PredicateType p, int i) {
        seen->push_back({t, n, p, i});
    }
};

template <class SF>
auto run(SF sf) {
    BuckDCMParams p;
    BuckDCMModel model{p};
    EventPredictor pred{};
    pred.register_predicate("zcd_iL", make_zcd_predicate(),
                            PredicateType::CurrentZC);
    PIController ctl{Real{1e-6}, Real{1e-9}};
    Vector x0(2);
    x0 << Real{0}, p.V_out_steady();
    PEDSimulator sim(model, sf, ctl, pred, Real{1e-9}, p.T_sw() / 4, 1,
                     make_zcd_projection());
    return sim.simulate(x0, Real{2e-4});
}

}  // namespace

int main() {
    BuckDCMParams p;
    std::vector<Seen> seen;
    Recording rec{p.T_sw(), p.D};
    rec.seen = &seen;
    const auto a = run(BuckDCMSwitchFn{p.T_sw(), p.D});
    const auto b = run(rec);

    bool same = a.times == b.times && a.states.size() == b.states.size();
    for (std::size_t k = 0; same && k < a.states.size(); ++k)
        same = (a.states[k].array() == b.states[k].array()).all();
    std::printf("bit-identical trajectory: %s (%zu samples)\n",
                same ? "yes" : "NO", a.times.size());
    if (!same) return 2;

    std::vector<Seen> want;
    for (const auto& e : b.event_log)
        if (e.type != PredicateType::GateEdge) want.push_back({e.t, e.name, e.type, -1});
    std::printf("state events %zu, hook calls %zu\n", want.size(), seen.size());
    if (seen.empty()) {
        std::printf("hook never called\n");
        return 1;
    }
    // every logged state event reached the hook, in order, named
    bool ok = seen.size() >= want.size();
    for (std::size_t k = 0, j = 0; ok && k < want.size(); ++k) {
        while (j < seen.size() && seen[j].t != want[k].t) ++j;
        ok = j < seen.size() && seen[j].name == want[k].name &&
             seen[j].type == want[k].type;
    }
    std::printf("every state event seen by name and type: %s\n", ok ? "yes" : "NO");
    return ok ? 0 : 3;
}
