// Hard gate for update-newton-persistent-solver stage A1 (task 3.2):
// does Eigen::SparseLU (COLAMD, the backend make_default_solver() gives)
// produce byte-identical solutions and permutations when one
// analyzePattern is reused across factorize() calls with new values on
// the same pattern, compared with a fresh object per factorisation?
//
//   c++ -std=c++23 -O2 -ffp-contract=off -DNDEBUG \
//       -I/opt/homebrew/include/eigen3 gate_sparselu.cpp -o gate && ./gate
//
// Exit status 0 and "0 differ" = gate passed.
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using SpM = Eigen::SparseMatrix<double, 0, int>;
using LU = Eigen::SparseLU<SpM, Eigen::COLAMDOrdering<int>>;

static std::vector<Eigen::Triplet<double>> pattern(int n, std::mt19937& g) {
    std::vector<Eigen::Triplet<double>> t;
    std::uniform_int_distribution<int> col(0, n - 1);
    for (int i = 0; i < n; ++i) {
        t.emplace_back(i, i, 0.0);
        for (int k = 0; k < 4; ++k) t.emplace_back(i, col(g), 0.0);
    }
    return t;
}

static SpM fill(int n, std::vector<Eigen::Triplet<double>> t, std::mt19937& g,
                double diag) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto& e : t) {
        e = Eigen::Triplet<double>(e.row(), e.col(),
                                   e.row() == e.col() ? diag * (1.0 + u(g)) : u(g));
    }
    SpM M(n, n);
    M.setFromTriplets(t.begin(), t.end(), [](double a, double b) { return a + b; });
    M.makeCompressed();
    return M;
}

static bool same(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

int main() {
    long cases = 0, diff = 0;
    for (int n : {50, 400, 3000}) {
        for (double diag : {10.0, 1.0, 0.05}) {
            for (int seed = 1; seed <= 4; ++seed) {
                std::mt19937 g(static_cast<unsigned>(seed * 1000 + n));
                const auto t = pattern(n, g);
                std::vector<SpM> Ms;
                for (int k = 0; k < 8; ++k) Ms.push_back(fill(n, t, g, diag));
                const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, -1.0, 1.0);
                LU reused;
                reused.analyzePattern(Ms[0]);
                for (const auto& M : Ms) {
                    reused.factorize(M);
                    LU fresh;
                    fresh.analyzePattern(M);
                    fresh.factorize(M);
                    ++cases;
                    if (reused.info() != fresh.info()) { ++diff; continue; }
                    if (fresh.info() != Eigen::Success) continue;
                    const Eigen::VectorXd xr = reused.solve(b), xf = fresh.solve(b);
                    const bool ok =
                        same(xr, xf) &&
                        reused.rowsPermutation().indices() == fresh.rowsPermutation().indices() &&
                        reused.colsPermutation().indices() == fresh.colsPermutation().indices();
                    if (!ok) {
                        ++diff;
                        std::printf("DIFF n=%d diag=%g seed=%d\n", n, diag, seed);
                    }
                }
            }
        }
    }
    std::printf("GATE: %ld factorizations compared, %ld differ\n", cases, diff);
    return diff != 0 ? 1 : 0;
}
