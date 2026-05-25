/// add-python-builder-ergonomics (v1.5) — cancellation primitives
/// for long-running analyses. Used uniformly by ``compute_dc_op``,
/// ``run_ac_sweep`` / ``run_mna_sweep``, and
/// ``compute_temperature`` so a GUI's "Cancel" button can preempt
/// any of them within one checkpoint interval.
///
/// Cost when disabled
/// ------------------
/// The callback is a ``std::function<bool()>``. When the caller
/// passes a default-constructed (empty) function, every analysis
/// checks ``if (cb) cb()`` once per checkpoint — a single branch
/// predict-not-taken, well under 1 ns. No measurable overhead on
/// a default-configured analysis.
///
/// Throwing model
/// --------------
/// On cancellation the analysis throws :class:`Cancelled`. The
/// exception carries a ``where`` label (e.g. ``"compute_dc_op"``)
/// and an optional ``progress_index`` so callers can report
/// "cancelled at frequency point 12 of 50" without round-tripping
/// to the GUI for the count.
///
/// This is preferred over returning a tagged result because the
/// analyses already throw on solver divergence; reusing the
/// exception path keeps the contract uniform.
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsim::analysis {

/// Predicate invoked at well-defined checkpoints inside long-
/// running analyses. Returns ``true`` to continue, ``false`` to
/// request cancellation. Empty function (default) disables the
/// check entirely with zero overhead.
using ShouldContinueFn = std::function<bool()>;

/// Thrown when the user's ``ShouldContinueFn`` returns ``false``.
/// Inherits from :class:`std::runtime_error` so callers can keep
/// using ``catch (const std::runtime_error&)`` without forcing a
/// migration.
class Cancelled : public std::runtime_error {
public:
    Cancelled(std::string where, long progress_index = -1)
        : std::runtime_error(
              progress_index >= 0
                  ? "analysis '" + where + "' cancelled at progress index "
                        + std::to_string(progress_index)
                  : "analysis '" + where + "' cancelled"),
          where_(std::move(where)),
          progress_index_(progress_index) {}

    [[nodiscard]] const std::string& where() const noexcept {
        return where_;
    }
    [[nodiscard]] long progress_index() const noexcept {
        return progress_index_;
    }

private:
    std::string where_;
    long        progress_index_;
};

/// Lightweight helper: invoke the callback if it's non-empty and
/// throw :class:`Cancelled` if it returned ``false``. Inlined so
/// the disabled path collapses to a single null check.
inline void check_cancellation(
    const ShouldContinueFn& cb,
    std::string_view where,
    long progress_index = -1) {
    if (cb && !cb()) {
        throw Cancelled(std::string{where}, progress_index);
    }
}

}  // namespace pulsim::analysis
