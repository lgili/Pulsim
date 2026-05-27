# 5. Sparse Direct LU Foundations

Chapter 4's PWL cache stores one analyzed-and-factorised sparse
LU per switch mask. To understand how Pulsim's in-house solver
works (chapter 6) and what its path-based partial refactor
achieves (chapter 7), you first need to know what "sparse LU"
actually does — and *why* the three algorithms Pulsim implements
(Reverse Cuthill-McKee ordering, elimination trees,
Gilbert-Peierls left-looking factorisation) are the right
choices for the SMPS matrix shape established in chapter 2.

This chapter is the textbook chapter: math + visual intuition +
literature references. Skim it if you're already fluent in
Davis 2006; deep-read it if you've only seen dense LU before.

---

## 5.1 Why direct, not iterative

The classical choice when solving $A\mathbf{x} = \mathbf{b}$ at
production scale is between **direct** methods (compute
$L \cdot U$ once, then back-substitute) and **iterative** methods
(start with a guess, repeatedly multiply by $A$ until convergence
— GMRES, BiCGSTAB, CG).

For Pulsim's workload the choice is direct. Three reasons:

### Matrices are small

SMPS state sizes range from $n \approx 5$ (buck) to $n \approx
50$ (MMC arm), maybe up to $n \approx 200$ for whole-power-
electronic-system simulations. Iterative solvers' asymptotic
advantage kicks in at $n \gtrsim 10^4$; below that, direct beats
iterative by 1-2 orders of magnitude per call.

### We need exact answers every step

Pulsim does $10^6$-$10^9$ solves per simulation. An iterative
solver returns an *approximation*; the per-step residual would
accumulate over millions of steps and corrupt the simulated
waveform. Direct solvers return the exact (machine-precision)
answer for $A\mathbf{x} = \mathbf{b}$, modulo standard floating-
point round-off.

### The factor is reusable

Once we've computed $A = LU$, every subsequent solve with the
same $A$ but different $\mathbf{b}$ costs only the triangular
solves $L\mathbf{y} = \mathbf{b}$ + $U\mathbf{x} = \mathbf{y}$
— typically $\sim 100$ ns. The PWL cache (chapter 4) and the
sliding-solver pattern of chapter 7 both depend on this
factor-reuse property; an iterative solver has no factor to
reuse.

The trade-off: direct solvers consume more memory than iterative
ones (the $L+U$ fill-in). For SMPS this isn't a constraint —
even an MMC's $n = 200$ matrix is $\sim 100$ KB stored, which is
nothing.

---

## 5.2 Why sparse, not dense

A dense $n \times n$ matrix needs $O(n^2)$ storage and an
$O(n^3/3)$ LU factorisation. For $n = 50$ that's $\sim 40\,000$
multiply-adds — small in absolute terms ($\sim 5$ µs on modern
hardware) but $\sim 10\times$ more than the equivalent sparse LU.

A sparse $n \times n$ matrix with $\mathrm{nnz}$ nonzeros and a
good ordering needs $O(\mathrm{nnz})$ storage and an
$O(\mathrm{nnz} \cdot \log n)$ factorisation — for $n = 50$ and
$\mathrm{nnz} = 200$, that's $\sim 1\,000$ multiply-adds.

The catch: **getting from $\mathrm{nnz} = 200$ stored to
$O(\mathrm{nnz})$ factorisation cost requires fill control**.
A naïve LU on a sparse matrix can introduce far more nonzeros in
$L+U$ than were present in $A$, blowing the storage and the
operation count. The next section explains how.

---

## 5.3 What "fill" is, and why it ruins naïve sparse LU

LU decomposition is *Gaussian elimination*: zero out the
sub-diagonal entries of column $k$ by subtracting a multiple of
row $k$ from rows $k+1, k+2, \ldots, n-1$.

When we subtract row $k$ (with nonzero pattern $S_k$) from row
$i > k$, the resulting row $i$ has nonzero pattern $S_i \cup
S_k$ — entries that were zero in $A$ but are nonzero in $L$ or
$U$ are called **fill**.

For the buck-like 8×8 fixture of chapters 2 and 6, three column
orderings give wildly different fill counts:

![Fill comparison: natural vs RCM vs COLAMD ordering on a representative MNA matrix](_figures/output/fig51_fill_comparison.png){ .center loading=lazy width="700" }

*Figure 5.1 — Fill comparison on a representative MNA matrix.
Left: original $A$ with $\mathrm{nnz}(A) = 22$. Middle:
factorised $L+U$ under the **natural ordering** (just take the
columns in order). Right: factorised $L+U$ under **RCM
ordering** (chapter 6 §6.2's choice). RCM produces $\sim 30$
nonzeros in $L+U$ vs natural's $\sim 60$. The difference comes
entirely from fill control; both decompositions are
algebraically equivalent up to permutation.*

The lesson: **the order in which you eliminate columns matters
enormously**. A clever ordering can reduce fill by 2-10×; a bad
ordering can balloon a sparse matrix into a dense one.

---

## 5.4 RCM: reverse Cuthill-McKee ordering

Pulsim uses **Reverse Cuthill-McKee** (RCM) ordering, originally
from George 1971. The idea:

1. Build the symmetric adjacency graph of $|A| + |A^T|$ (treat
   $A$ as if it were symmetric for the purpose of ordering).
2. Pick a starting vertex (Pulsim picks the minimum-degree
   unvisited vertex).
3. BFS from there, ordering vertices in ascending neighbour
   degree.
4. **Reverse** the resulting sequence.

The output is a permutation $P$ such that $P \cdot A \cdot P^T$
has a **band structure** — nonzeros cluster near the diagonal.
Band matrices factorise with minimal fill because Gaussian
elimination can only introduce fill *within* the band, never
outside it.

For SMPS topologies whose physical structure is already chain-
like (MMC arms, multi-cell flying-cap converters, the
N-switch-chain benchmark fixture), RCM recovers the natural
banded structure essentially perfectly. For more irregular
topologies (NPC, three-phase VSI) RCM gives a 1.5-3× fill
improvement over natural ordering — not as dramatic as the
band cases, but solid.

The alternatives Pulsim could have used:

- **COLAMD** (column approximate minimum-degree, Davis & Hu
  2004): typically 10-30 % better fill than RCM on irregular
  matrices, but its symbolic phase is 2-5× slower.
- **AMD** (approximate minimum-degree, Amestoy/Davis/Duff 1996):
  similar trade-off to COLAMD.

The Pulsim v1.3.0 design choice was RCM for the MVP. The
proposal explicitly lists COLAMD/AMD as out-of-scope follow-ups;
they'll land in a future release once benchmark data shows RCM
is the bottleneck.

---

## 5.5 The elimination tree

Once $P \cdot A \cdot P^T$ is permuted to band form, we can ask:
**when we eliminate column $k$, which other columns are affected
by the fill it produces?**

The answer is governed by the **elimination tree** (etree),
a data structure defined as:

> $\mathrm{parent}[k]$ is the smallest column index $j > k$
> such that $L_{jk} \ne 0$ — i.e. the next column down the
> $k$-th column of $L$.

(If no such $j$ exists, $k$ is a root.)

The etree captures fill propagation in a way that makes the
chapter 6's symbolic factorisation linear-time (Davis 2006
§4.10):

- **To predict the fill in column $k$**, you only need to look
  at the etree subtrees rooted at $k$'s direct neighbours in $A$.
- **To do path-based partial refactor** (chapter 7), you walk
  the etree from a changed column up to the root and update
  only those columns — every column off the path is unaffected.

Visually, the etree for a representative MNA matrix:

![Elimination tree for an 8x8 MNA matrix](_figures/output/fig52_etree.png){ .center loading=lazy width="500" }

*Figure 5.2 — Elimination tree (etree) for an 8×8 MNA matrix
after RCM ordering. Each box is a column index in the permuted
order; arrows point parent → child. Columns 0, 1, 2 are leaves
(no $L_{jk} \ne 0$ for any $k < j$); column 7 is the unique
root. Eliminating column 2 only affects columns 4, 6, 7 (its
path to the root); column 3's subtree is independent.*

Pulsim's `PulsimSparseLuSolver::etree_parent_` is exactly this
data structure: a `std::vector<Index>` of length $n$ where
`etree_parent_[k] == j` means $j$ is the next column down
the $k$-th column of $L$, or $-1$ if $k$ is a root.

---

## 5.6 Gilbert-Peierls left-looking factorisation

There are two algorithmic families for sparse LU:

- **Right-looking** ("LU as outer products"): factor column $k$,
  then immediately propagate its effect to all columns $j > k$.
  Memory bandwidth is the bottleneck (each remaining column gets
  re-loaded once per left column).
- **Left-looking** ("LU as inner products"): for column $k$,
  gather all the corrections that *prior* columns $j < k$ owe
  to $k$, apply them in one pass. Memory bandwidth is much
  better; this is the choice for sparse direct.

Pulsim implements the **Gilbert-Peierls** left-looking algorithm
(Gilbert & Peierls, *SIAM J. Sci. Stat. Comput.* 9, 1988). The
core inner loop, in pseudo-code:

```
for k in 0..n-1:
    x = A[:, P_col[k]]            # load the column from A
    for j in 0..k-1:
        if x[j] != 0:               # there's a pending L-update
            x[j+1:n] -= L[j+1:n, j] * x[j]
    pivot = argmax(|x[k:n]|)        # partial pivoting
    swap rows k and pivot in L and P_row
    L[k+1:n, k] = x[k+1:n] / x[k]
    U[0:k+1, k] = x[0:k+1]
```

Three things make this fast in the sparse case:

1. **The inner `for j` loop only visits columns $j$ where
   $x[j] \ne 0$.** For a banded matrix with bandwidth $b$, at
   most $b$ such columns exist for each $k$. The total work is
   $O(n \cdot b) = O(\mathrm{nnz})$, not $O(n^2)$.
2. **The L-update `x[j+1:n] -= L[j+1:n, j] * x[j]` is a sparse
   SAXPY** — it touches only the nonzero rows of $L[:, j]$, not
   all $n - j - 1$ rows below the diagonal.
3. **The dense workspace `x[n]`** is the only allocation; no
   per-column malloc / free.

The cost: $O(\mathrm{nnz}(L) + \mathrm{nnz}(U) + n)$ per
factorisation, with a small constant. On the buck-like 8×8
fixture, $\sim 50$ multiply-adds total. On a $n = 200$ MMC,
$\sim 2\,000$ multiply-adds — still microseconds of work.

---

## 5.7 Partial pivoting under sparse direct

Pure Gaussian elimination assumes the diagonal pivot $x[k]$ is
nonzero. For circuit MNA this fails when a voltage-source
constraint row has zero diagonal (chapter 2 §2.3 explains why —
the constraint row reads $V_a - V_b = V_{\mathrm{src}}$, with
no self-conductance contribution).

**Partial pivoting** fixes this by swapping the pivot row $k$
with the row $i \ge k$ whose $|x[i]|$ is largest, before
dividing. The catch in sparse direct: row swaps change the row
permutation $P_{\mathrm{row}}$, which means the symbolic
factorisation pattern computed pre-pivot becomes incorrect for
the actual $L$ and $U$ stored.

Pulsim's solution (chapter 6 §6.3 in detail): **discover the
$L+U$ pattern dynamically from the runtime nonzeros of $x$**,
ignoring the pre-pivot symbolic phase. This is slightly slower
than a "symbolic-then-numeric" pipeline but is bug-free under
any pivot choice.

For path-based partial refactor (chapter 7), the threshold
pivoting rule (KLU-style, `PIVOT_THRESH = 1e-3`) allows pivots
that are "close enough" to the row maximum — specifically:

> A pivot $x[k]$ is acceptable if $|x[k]| \ge \mathrm{PIVOT\_THRESH}
> \cdot |x|_\infty$ over $i \ge k$.

When this fails, the path-based update rejects with a fallback
to full re-factorise. The captured benchmark (chapter 8) shows
zero fallbacks across all 1999 single-bit flips per N on the
microbench — the threshold is loose enough to absorb circuit
MNA's natural pivot-magnitude swings.

---

## 5.8 Putting it together: cost vs $n$

Here's how the four cost regimes compare on a single chart:

![Per-call cost vs n: dense GE / dense LU / sparse LU natural / sparse LU + RCM](_figures/output/fig53_cost_vs_n.png){ .center loading=lazy width="700" }

*Figure 5.3 — Asymptotic per-call cost of factorising an $n
\times n$ matrix under four algorithms: dense Gaussian
elimination $O(n^3)$, dense LU with pivoting $O(n^3/3)$, sparse
LU under the natural ordering (fill explodes — the curve is
between $n^2$ and $n^3$ in practice), and sparse LU with RCM
ordering on a banded matrix $O(\mathrm{nnz} \log n) \approx
O(n \log n)$. The dashed grey line shows the SMPS-relevant range
($n \in [5, 200]$). Even at the top of the range, sparse LU
with RCM is $\sim 1000\times$ faster than dense LU.*

This is the chart that justifies every architectural choice in
chapters 6 and 7. We're at the bottom of the cost hierarchy
because we use:

- Sparse storage (not dense) — collapses $n^2$ to nnz
- RCM column ordering — collapses fill from $O(n^2)$ to $O(n
  \log n)$ for banded matrices
- Gilbert-Peierls left-looking — collapses factorisation to
  $O(\mathrm{nnz}(L) + \mathrm{nnz}(U))$
- Path-based partial refactor (chapter 7) — collapses single-
  bit-flip refactor from $O(\mathrm{nnz})$ to $O(\sqrt{n})$

The cumulative speedup vs textbook dense LU is **5 orders of
magnitude** at $n = 200$. That's why Pulsim can run real-time
MMC simulations on a laptop where SPICE-style simulators need
a cluster.

---

## 5.9 Where this lives in the codebase

| Concept | Where it's implemented |
|---|---|
| `Eigen::SparseMatrix` CSC container | Used as a passive storage layout throughout `core/include/pulsim/sparse/` |
| RCM ordering | `PulsimSparseLuSolver::compute_rcm_ordering_` in `core/include/pulsim/sparse/pulsim_lu_solver.hpp` |
| Etree construction | `PulsimSparseLuSolver::compute_etree_` (Liu 1986 disjoint-set ancestor compression) |
| Gilbert-Peierls factorise | `PulsimSparseLuSolver::factorize` |
| Threshold partial pivoting | inline in `factorize` (`PIVOT_THRESH = 1e-3`) |
| Triangular solves | `PulsimSparseLuSolver::solve` |
| `Eigen::SparseLU` reference | Used by `SparseLuSolver` (`Backend::Eigen`) intentionally retained as the benchmark baseline |

The next chapter is the line-by-line walkthrough of
`PulsimSparseLuSolver`.

---

## 5.10 Takeaways

- Pulsim uses **direct** sparse LU (not iterative) because the
  per-step solve must be exact and the factor must be reusable.
- **Fill control** is the make-or-break of sparse LU; a good
  column ordering reduces fill by 2-10×, a bad one ruins
  sparsity entirely.
- Pulsim's MVP fill-reducing ordering is **RCM** (George 1971)
  — recovers banded structure perfectly on chain-like SMPS
  topologies, gives 1.5-3× improvement on irregular ones.
- The **elimination tree** captures fill propagation. Pulsim
  computes it via Davis 2006 §4.10's disjoint-set ancestor
  compression in linear time.
- **Gilbert-Peierls left-looking** is the factorisation
  algorithm; it touches $O(\mathrm{nnz})$ floating-point
  operations per factorise instead of $O(n^3)$.
- **Threshold partial pivoting** (KLU-style, `PIVOT_THRESH =
  1e-3`) handles voltage-source constraint rows whose diagonal
  is zero, and lets chapter 7's path-based partial refactor
  succeed on most circuit MNA workloads.

## 5.11 Further reading

- **Davis 2006** — *Direct Methods for Sparse Linear Systems*,
  SIAM. The bible. Chapters 4-5 cover etrees and Gilbert-Peierls
  in textbook detail; chapter 6 covers ordering algorithms.
- **George 1971** — A. George, "Computer implementation of the
  finite element method", Stanford PhD thesis. Original RCM
  paper.
- **Gilbert & Peierls 1988** — "Sparse partial pivoting in time
  proportional to arithmetic operations", *SIAM J. Sci. Stat.
  Comput.* 9(5):862-874. The left-looking algorithm.
- **Liu 1986** — J. W. H. Liu, "On the storage requirement in
  the out-of-core multifrontal method for sparse factorization",
  *ACM Trans. Math. Soft.* 12(3):249-264. Etree construction
  via disjoint sets.
- **Demmel 1997** — *Applied Numerical Linear Algebra*, SIAM.
  §3.4 covers partial pivoting under sparse direct.
- **In this doc set** — [Chapter 6](06-pulsim-sparse-lu.md) is
  the line-by-line walkthrough of how all of the above lands as
  ~900 lines of C++23.
