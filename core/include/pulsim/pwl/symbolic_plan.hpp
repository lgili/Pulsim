#pragma once

// =============================================================================
// Pulsim — symbolic plan shared by every cache of one circuit structure
// =============================================================================
//
// add-immutable-circuit-ir (group 10). A `PwlStateSpaceCache` analyses
// one sparsity pattern per assembly regime (dynamic, static). Circuits
// derived by `with_params` / `with_value` have the same topology, so a
// sweep over them needs that analysis once: every such circuit holds the
// same `SymbolicPlan`, and a cache given it (`share_symbolic`) adopts
// the stored analysis — after checking the pattern — or offers its own.
// The factor is bit-equal either way: it depends only on the column
// ordering, which depends only on the pattern.
//
// The only mutable state an immutable `ir::Circuit` reaches: written
// once per regime, under the mutex, by whichever run analyses first.

#include "pulsim/sparse/solver.hpp"

#include <memory>
#include <mutex>

namespace pulsim::pwl {

class SymbolicPlan {
public:
    using Analysis = std::shared_ptr<const sparse::SymbolicAnalysis>;

    [[nodiscard]] Analysis get(bool dynamic) const {
        std::lock_guard lock{m_};
        return dynamic ? dynamic_ : static_;
    }

    /// Store `a` for the regime unless one is stored already.
    void offer(bool dynamic, Analysis a) {
        std::lock_guard lock{m_};
        auto& slot = dynamic ? dynamic_ : static_;
        if (!slot) slot = std::move(a);
    }

private:
    mutable std::mutex m_;
    Analysis dynamic_;
    Analysis static_;
};

}  // namespace pulsim::pwl
