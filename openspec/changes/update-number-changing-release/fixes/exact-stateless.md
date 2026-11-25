# Exact engine: a loss port on a circuit with no dynamic state (SIGSEGV)

Branch `fix/core-exact-stateless-loss-port`, from the PR head 3db22cde
(`origin/feat/core-integration`; the crash reproduces there, it does not need
the commutation-edge-rule branch). Found by the 12.8 follow-up
(`fixes-12-8.md` on `fix/core-commutation-edge-rule`).

## Defect

`simulate(engine='dsed' | 'exact')` with `set_device_losses` on a circuit
with no inductor and no capacitor killed the process (exit -11). Same circuit
without the port: runs.

Boundary (PR-head extension, each run in a subprocess):

| circuit (600 V leg, HS/LS, 10 kHz pair) | dsed |
|---|---|
| no port | runs |
| port, no load / R load / antiparallel diode (0 states) | SIGSEGV |
| port + one capacitor (1 state) | runs |
| pwl / trbdf2 / auto, every variant | runs |

## Cause

Crash report: `EXC_BAD_ACCESS (SIGSEGV) at 0x0` in
`ExactLossPort<…>::energies_` (`core/include/pulsim/dsed/loss_port.hpp`),
called from the `on_accept` hook of the first accepted step. The line:

    const Real floor = eps * (x1.cwiseAbs().maxCoeff() + mag.maxCoeff()) + 1e-300;

With zero states `x1` (the reduced state) and `mag` (its modal magnitude,
`head(n)`) are empty; Eigen's `maxCoeff()` of an empty vector asserts in a
debug build and, under `NDEBUG`, reads the first coefficient through the
null data pointer. A 6-line program against the same Eigen (5.0) shows it:
`VectorXd x(0); x.lpNorm<Infinity>()` prints 0, `x.cwiseAbs().maxCoeff()`
dies with SIGSEGV.

## Fix

`x1.lpNorm<Eigen::Infinity>() + mag.lpNorm<Eigen::Infinity>()`: the same
expression (`cwiseAbs().maxCoeff()`) behind Eigen's own size-0 guard (it
returns 0; Eigen 3.4.0 has the same guard). Every non-empty case is the
same arithmetic, so nothing else moves.

Nothing else was missing: with n = 0 the augmented modal form is the single
constant mode (μ = 0), `P2 = h·φ₁(0) = h`, and the port books `v·i·h`, the
exact conduction energy of an algebraic mode. Edges read the algebraic
solution on each side (`recover_full` of the empty state).

## Truth (no simulator)

Leg: 600 V, HS dcp→a and LS a→gnd (g_on 1 kS, g_off 1 nS), 10 Ω a→gnd, pair
at 10 kHz, D = 0.5, no dead time, port on HS, t_end = 975 µs (10 turn-offs,
9 turn-ons). One-node nodal solve per mode → HS current and power constant;
every HS edge hard (forward current): E·(v_open/600)·(i_closed/50).

| engine | E_on (J) | E_off (J) | E_cond (J) | E_cond − truth |
|---|---|---|---|---|
| truth | 0.04319568086382107 | 0.04199580083982604 | 0.0017998110899814896 | |
| dsed / exact | 0.04319568086382107 | 0.041995800839826035 | 0.0017998110899814894 | −2e-19 |
| trbdf2 / auto | same | same | 0.0017997986827388039 | −1.24e-8 |
| pwl dt = 100 ns | same | same | 0.0017993911919604942 | −4.20e-7 |

`auto` without `dt` routes to trbdf2 on the PR line (it routes to the exact
engine only on the release branch).

The pwl / trbdf2 conduction misses are PRE-EXISTING start / edge readings,
not this defect: both start from the zero vector, so the t = 0 sample reads
p = 0 (sources off) and the first step's quadrature weight on it is lost —
trbdf2 1/(2√2)·h₁ (proportional to h₁ over three t_end values; gone when a
state is present), pwl 2/3·dt at every t_end (also with a capacitor, also
with `start_from_dc_op=True` half a step remains) — and pwl reads each edge's
step with the edge's one-sided sample (±p·dt/2 per edge; 10 off − 9 on → one
half step). The test bounds them by that cause (≤ p_on·h₁·(1 + edges)),
dsed is held to 1e-12.

## Audit of the same assumption (empty state vector)

`grep maxCoeff|minCoeff|size() - 1|front()|back()` over `core/include/pulsim/dsed/`:

- `loss_port.hpp:261` — the crash; fixed.
- `loss_port.hpp:244/268` `names_.front()` — only in refusals; `names_` is
  non-empty whenever the port exists.
- `builder_adapter.hpp:619, 643` — inside `reinit_to_constraint`, which
  returns early when the mode has no constraint (`d == 0`); a constraint row
  implies states.
- `builder_adapter.hpp:852` `head(nn)` — nn = number of nodes, ≥ 1 for a deposit
  node; `:858` already guarded (`r.size() > 0`).
- `builder_adapter.hpp:1490` — L of one coupled-winding group, ≥ 1 winding.
- `exact_lti.hpp:271/319` — only reached from the eigen path of a mode with
  n > 0; `:402` guarded; `:467` the (n+1)×(n+1) augmented matrix, ≥ 1×1;
  `:502` after the explicit `n == 0` static-mode return.
- `periodic_orbit.hpp:337` and `sig[n - 1]` — not reached with n = 0: an empty
  monodromy returns `Grazing` first, and `periodic_steady_state` refuses a
  circuit with no L or C by name before (ValueError).
- `scheduler.hpp:210/701/895/2158`, `period_integrals.hpp:422–700` — index the
  time / sample arrays (≥ 1 sample), not the state.
- `step_controller.hpp:71` (`sqrt(s2 / n)`) and `bdf2_integrator.hpp:169`
  (`.mean()`) — 0/0 = NaN for n = 0 (no crash; a NaN error norm would reject
  every step). LATENT, not guarded: not reached by any zero-state input the
  reviewer tried on the numeric integrators (sine / pulse sources,
  `b_extra_fn`, rk45 / bdf2 / auto integrators all ran).
- `loss_port.hpp` `step_heat_` (junction loop) — reached only with a deposit
  node; a node without capacitance is refused by name (ValueError) before the
  run, one with a capacitor is a state (n ≥ 1). Both covered by tests.

Probed by running (all subprocess, all run or refuse by name): zero states with
a recorder (with / without the port), a diode, a thermal node with and without
capacitance, a resume of a stateless loss-port run, a single switch shorting a
source, `periodic_steady_state` (refused by name), `engine='exact'`.

## What did not move

Against the PR head's extension (so_be5454f3; 3db22cde changes a test only):
goldens + counters, stateful, engines, steps, `pwl_identity` 390/390,
`trbdf2_identity` 371/371, dsed bit-identity 36/36, and a digest of every
`simulate()` result (times, states, loss records, loss events) in six loss /
exact test files (98 runs, 18 of them dsed loss-port runs): identical.
Cost (instructions, dsed leg with an R-L load and two loss ports, 5 ms,
min of six interleaved): 20 150 849 → 20 078 766 (−0.36 %, the base `.so` is a
different build of the same source; the change is a size test on a path taken
once per accepted step).

## Side findings (reviewer, not fixed here; none caused by this fix)

1. SILENT DIVERGENCE with `t_start > 0` (reproduced): the exact engine runs
   the 0-start run and relabels its times (`_dsed_dispatch.py` ~801), so the
   gate schedule is read at t − t_start; trbdf2 reads it at absolute t. The
   leg with `t_start=5e-5, t_end=1e-4` (HS scheduled off there): dsed books
   HS conducting the whole window (E_cond 1.7996e-4 J, v(a) 599.94 V), trbdf2
   HS off (1.8e-8 J, one turn-on, v(a) 6e-10 V). No warning; affects circuits
   with states too.
2. Edge counting at the run's ends differs: trbdf2 counts an edge at t = 0 or
   exactly at t_end, dsed does not (t_end = 1 ms: n_on 10 vs 9). The test uses
   t_end = 975 us to stay off an edge.
3. An ideal switch closing on a charged capacitor (a state): dsed's E_on reads
   the post-impulse current (~1 A), trbdf2 the spike (~94 A).
