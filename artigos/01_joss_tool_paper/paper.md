---
title: 'Pulsim: A piecewise-linear state-space simulator for switched-mode power electronics'
tags:
  - C++
  - Python
  - power electronics
  - circuit simulation
  - switched-mode converters
  - multilevel converters
  - modular multilevel converter
  - MMC
  - PWL state-space
  - sparse LU
authors:
  - name: Luiz Carlos Gili
    orcid: 0000-0002-5749-7199
    corresponding: true
    affiliation: 1
affiliations:
 - name: Independent Researcher, Brazil
   index: 1
date: 24 May 2026
bibliography: paper.bib
---

# Summary

`Pulsim` is an open-source power-electronics circuit simulator with a
header-only C++23 kernel and a Python-first user-facing API. It is
designed for fast transient simulation of **switched-mode** converters
— buck and boost converters, isolated topologies (flyback, forward,
half-bridge), three-phase voltage-source inverters (VSIs), and
multilevel converters including the Neutral-Point-Clamped (NPC)
inverter [@Nabae:1981] and the Modular Multilevel Converter (MMC)
[@Lesnicar:2003]. The user describes a topology with string-named
nodes (`b.add_voltage_source("Vin", "vin", "gnd", 24.0)`,
`b.add_mosfet_with_body_diode(...)`, `b.add_inductor(...)`), provides
a modulator as a switch-state function, and calls
`pulsim.simulate(b, t_end, dt, switch_fn=...)`. The same surface
supports transient analysis, AC small-signal sweeps, frequency-
response analysis, parameter sweeps, periodic-steady-state shooting,
and harmonic balance.

`Pulsim`'s central design choice is a **piecewise-linear (PWL) state-
space cache**: every reachable combination of binary switch and diode
states is enumerated once, the corresponding linear state-space
matrix is factored once, and the transient solver indexes the cached
factor at every time step. From `v1.3.0` onward the LU factorisation
is performed by an **in-house C++23 sparse LU solver** that requires
no third-party LU dependency; `v1.4.0` extends this kernel to a
templated `Scalar` type (real and `std::complex<double>`) and adds a
generalised path-based partial-refactorisation framework that
amortises factor work across switch flips, multi-bit transitions, and
parameter sweeps. The package also ships **ten validated reference
projects** spanning buck through MMC, each combining an analytical-
derivation notebook, a closed-loop controller-design notebook, and an
*executed* Pulsim cross-validation notebook whose waveforms render
inline on GitHub.

# Statement of need

Researchers and educators modelling switched-mode power-electronics
systems face a long-standing tooling dilemma. The dominant high-
throughput algorithm for these workloads — the PWL state-space cache
combined with Newton refinement for nonlinear devices — is only
available in **commercial, closed-source** packages. The two main
open-source alternatives, the SPICE family and bespoke `numpy`/`scipy`
scripts, are either too slow for systematic studies of multi-cell
multilevel converters or too limited to model anything beyond ideal-
switch toys. The consequence is that reproducible, extensible power-
electronics simulation work cannot be shared as a self-contained
code repository; either the dependency is proprietary or the
simulator must be reimplemented from scratch by each researcher.

Pulsim's purpose is to close this gap. It provides the same
algorithmic architecture used by industrial-grade simulators under an
**MIT licence**, with a header-only C++23 kernel that any team can
embed and extend, and a Python API tuned for the reproducibility
conventions of modern computational research.

# State of the field

| Tool | Licence | PWL cache | Open scriptable | Multilevel-converter support |
|---|---|:---:|:---:|:---:|
| PLECS [@Allmendinger:2002] | Commercial | ✓ | partial | first-class |
| PSIM [@Powersim:2024] | Commercial | ✓ | partial | first-class |
| Simscape Electrical [@MathWorks:2024] | Commercial (MATLAB) | ✓ | partial | first-class |
| Saber [@SiemensSaber:2024] | Commercial | partial | limited | first-class |
| ngspice [@Vogt:2020:ngspice] | GPL | ✗ | ✓ | manual |
| Xyce [@Keiter:2020:Xyce] | GPL | ✗ | ✓ | manual |
| `numpy` / `scipy` scripts | n/a | ✗ | ✓ | manual |
| **Pulsim** | **MIT** | **✓** | **✓** | **first-class** |

PLECS, PSIM, Simscape and Saber implement the PWL cache and are the
de-facto standard in industry for hard-switched power-electronics
design, but their proprietary licences exclude their use in fully
open, reproducible research workflows. The SPICE-family open-source
simulators (`ngspice`, `LTspice`, Xyce) are mature but use a fully-
iterated Newton step at every commutation event, which makes them
one to two orders of magnitude slower than PWL-cache simulators on
switching workloads [@Allmendinger:2002]. Pulsim is, to the best of
the author's knowledge, the first permissively-licensed simulator
that combines the PWL state-space cache, automatic switch + diode
event detection, optional Newton refinement for nonlinear devices,
an in-house sparse LU with path-based partial-refactorisation
[@Chan:1986; @Dinkelbach:2021], and a Python-first user API in a
single package.

# Software design

Four design decisions deserve explicit mention because they are
trade-offs that shape what Pulsim is good at — and what it is not.

**Header-only C++23 kernel.** All of the solver, the device library
and the state-space cache live in `core/include/pulsim/`. Embedding
Pulsim in another C++ project is a `target_link_libraries(...
pulsim::core)` away; no static archive to maintain. The cost is
longer template-instantiation time at first compile, accepted as a
fair price for the integration convenience.

**Eager PWL cache enumeration.** The cache pre-factors every
reachable combination of switch states upfront, giving constant-time
per-step lookup. The trade-off is that the cache size grows as
$2^{N_{\text{switches}}}$; on the desktop hardware used for this
project the practical ceiling is around 18 mask bits. Topologies
that exceed this — typically full three-phase multilevel converters
with $N \ge 4$ sub-modules per arm — must either model only one
phase, or use the smooth-blend `add_nonlinear_diode` primitive that
does not enter the switch mask. This trade-off is documented in the
MMC reference project and is honestly acknowledged in the user-
facing API.

**Modulator decoupled from topology.** The C++ `add_*` primitives
register only the power-stage components; the PWM modulator is an
external `switch_fn(t) -> SwitchStateMask` closure that the
simulator calls at each step. This lets a single
`add_three_phase_vsi` helper support open-loop SPWM, closed-loop
FOC, grid-tie current control, and any other modulation scheme
without parameter explosion on the topology side. Closed-loop
control adds an additional `step_observer(t, x)` callback that runs
at C++ kernel speed via the `MixedDomainBlockChain` (no Python-
interpreter cost per step).

**In-house sparse LU with path-based partial-refactorisation.**
From `v1.3.0` onward the LU factorisation is performed by Pulsim's
own C++23 sparse LU implementation (RCM ordering, Liu-Davis
elimination tree, Gilbert-Peierls left-looking factor with threshold
partial pivoting), avoiding any third-party LU dependency
(SuiteSparse, KLU, UMFPACK). When consecutive switch masks differ in
one bit — the common case under Gray-coded PWM — the solver re-
eliminates only the columns along the etree path of the affected
column, following Chan, Brandwajn, and Tinney [@Chan:1986] and the
more recent treatment by Dinkelbach, Reichenbach, and Pflugradt
[@Dinkelbach:2021]. `v1.4.0` templates the solver on the scalar type
so the same code path serves the real-valued transient solver and a
complex-valued AC sweep, and generalises the path-based update to
multi-bit switch transitions (via the union of column etree paths)
and to parameter sweeps on $R$, $L$, $C$, and source voltages
(via `PwlStateSpaceCache::refactor_parametric`). The trade-off is
algorithmic complexity inside the kernel; reproducibility is
preserved through a comprehensive unit and microbenchmark suite
(498 C++ tests, 6 Python tests as of the release version), and the
mathematical primitives are exposed both at the C++ and Python
levels (`sweep_path_aware`, `monte_carlo_path_aware`).

# Research impact

Pulsim is a young project; its near-term impact case rests on three
concrete artefacts that already ship in the repository:

1. The **ten-converter validation library** — buck, boost,
   buck-boost, flyback, forward, half-bridge, boost PFC, three-phase
   VSI, NPC 3-level inverter, and single-phase MMC — each with an
   analytical-derivation notebook, a controller-design notebook, and
   an *executed* Pulsim cross-validation notebook. These act as both
   tutorials and as a regression suite that exercises the solver on
   topologies of escalating complexity.

2. The **MMC project** is a working demonstration of the canonical
   sort-and-select capacitor-balancing algorithm
   [@Saeedifard:2010:MMC] driving an actual switched simulation — to
   the author's knowledge the first open-source MMC reference
   implementation paired with an executed simulator.

3. The **path-based partial-refactorisation kernel** is, to the
   author's knowledge, the first permissively-licensed implementation
   of the Chan/Brandwajn/Tinney update applied beyond Newton-Raphson
   linear-system refactoring [@Chan:1986; @Dinkelbach:2021]: the same
   etree-path machinery is applied to single-bit switch flips,
   multi-bit transitions, and physical-parameter sweeps. A
   methods-oriented manuscript characterising this kernel on
   reference SMPS topologies is in preparation.

The project welcomes external collaborations; bug reports, feature
proposals, and converter case studies are tracked through the
[GitHub issue tracker](https://github.com/lgili/Pulsim/issues) and
the [OpenSpec proposal workflow](https://github.com/lgili/Pulsim/tree/main/openspec)
in the repository.

# Acknowledgements

The author thanks the broader open-source power-electronics
community — in particular the maintainers of `ngspice` and the
authors of `scikit-build-core` who made the C++/Python build
straightforward. Development was supported by personal time.

# AI usage disclosure

The draft of this paper was prepared with the assistance of
Anthropic's *Claude* generative AI (model versions in the
*Sonnet 4.5*–*Opus 4.7* range, accessed between May 2026 and the
submission date). Claude was used for two purposes: (i) drafting
initial text from a structured outline supplied by the author, and
(ii) suggesting comparable-software citations to consider for the
*State of the field* section. The author reviewed, edited and
validated every paragraph; every fact, citation, claim of novelty,
performance number, and design-trade-off rationale was
independently checked against the Pulsim source, its commit
history, and the cited references. The same author wrote and
reviewed every line of code in the repository; AI-assisted code
edits were inspected before commit. No AI tool was used to
communicate with JOSS editors or reviewers.

# References
