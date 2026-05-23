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
authors:
  - name: Luiz Carlos Gili
    orcid: 0000-0000-0000-0000  # TODO: paste your ORCID before submitting
    affiliation: 1
affiliations:
 - name: Independent Researcher, Brazil
   index: 1
date: 23 May 2026
bibliography: paper.bib
---

# Summary

`Pulsim` is an open-source power-electronics circuit simulator written
in C++23 with a Python-first user-facing API. It is designed for fast,
accurate transient simulation of **switched-mode** converters — buck,
boost, isolated topologies, three-phase voltage-source inverters,
multilevel converters such as the Neutral-Point-Clamped (NPC) inverter
[@Nabae:1981] and the Modular Multilevel Converter (MMC)
[@Lesnicar:2003] — where the dominant computational cost in
general-purpose SPICE-like simulators is the repeated re-factorisation
of the system matrix after every switching event.

`Pulsim`'s central design choice is a **piecewise-linear (PWL)
state-space cache**: every reachable combination of binary switch and
diode states is enumerated once at startup, the corresponding linear
state-space matrices are factored once, and the time-stepper then
selects the appropriate cached factor at each step in $O(1)$ time.
This is the same algorithmic strategy that powers commercial tools
such as PLECS [@Allmendinger:2002] and Simulink/Simscape Power Systems
[@MathWorks:2024], previously unavailable as a permissively-licensed
open-source implementation.

# Statement of need

Researchers and educators working on switched-mode power-electronics
modelling face a long-standing tooling dilemma:

* **Commercial simulators** such as PLECS [@Allmendinger:2002], PSIM
  [@Powersim:2024], MATLAB/Simulink with Simscape Power Systems
  [@MathWorks:2024], and Saber [@SiemensSaber:2024] dominate the
  industry because they implement the PWL state-space cache + Newton
  refinement architecture needed for high-throughput simulation of
  hard-switched converters. They are expensive, closed-source, and
  hard to extend with novel device models or numerical methods.

* **Open-source SPICE-family simulators** (`ngspice`
  [@Vogt:2020:ngspice], LTspice, Xyce [@Keiter:2020:Xyce]) are
  general-purpose and rigorous, but their nonlinear Newton-Raphson
  step is invoked at every commutation event. This makes them
  one to two orders of magnitude slower than PWL simulators on
  switching-power-electronics workloads [@Allmendinger:2002].

* **Educational tools** built around `numpy`/`scipy` (e.g.
  `pyleecan`, custom forward-Euler scripts) are accessible but
  generally limited to ideal-switch models, lack the
  topology-aware caching, and do not scale to multi-cell
  multilevel converters.

Pulsim closes this gap by providing the same fast PWL-cache
architecture under the **MIT license**, with a header-only C++23
kernel and idiomatic Python bindings. The user describes the topology
in Python with string-named nodes
(`b.add_voltage_source("Vin", "vin", "gnd", 24.0)`,
`b.add_mosfet_with_body_diode(...)`, `b.add_inductor(...)`), supplies
a switch-state function for the modulator, and calls `pulsim.simulate(b,
t_end, dt, switch_fn=...)`. The same surface supports transient
analysis, AC small-signal sweeps, frequency-response analysis (FRA),
parameter sweeps, periodic-steady-state shooting, and harmonic
balance.

Pulsim also ships a library of **ten validated reference projects** —
buck, boost, buck-boost, flyback, forward, half-bridge, boost PFC,
three-phase voltage-source inverter, NPC 3-level inverter, and
single-phase modular multilevel converter — each with an
analytical-derivation notebook, a closed-loop controller-design
notebook, and an executed Pulsim cross-validation notebook whose
figures render directly on GitHub. The reference library acts both as
end-to-end documentation and as a regression suite that exercises the
solver on topologies of escalating complexity.

# Functionality

The Python API exposes the following primitives, all forwarded to the
C++ kernel via `pybind11`:

* `pulsim.CircuitBuilder` — accepts string node names; methods
  include `add_resistor`, `add_capacitor`, `add_inductor`,
  `add_switch`, `add_diode`, `add_nonlinear_diode`,
  `add_mosfet`, `add_mosfet_with_body_diode`,
  `add_transformer`, `add_voltage_source`, `add_pulse_voltage_source`,
  `add_sine_voltage_source`, and `add_current_source`.
* `pulsim.topology.*` — composite helpers (`add_three_phase_vsi`,
  `add_three_phase_rl_load`, `add_bridge_rectifier`) for common
  multi-device sub-circuits.
* `pulsim.simulate(builder, t_end, dt, switch_fn=...)` — the
  primary entry point; runs a fixed-step trapezoidal-rule transient
  with the PWL cache lookup at each step.
* `pulsim.make_pwm_switch_fn`, `pulsim.make_three_phase_spwm_fn`,
  `pulsim.make_dead_time_pwm_pair_fn` — switch-function factories
  for common modulation schemes.
* `pulsim.MixedDomainBlockChain` — composable control blocks
  (PI/PID, comparators, rate limiters, op-amps, FOC transforms,
  thermal observers) that execute at C++ kernel speed.
* `pulsim.run_ac_sweep`, `pulsim.run_fra`,
  `pulsim.run_periodic_shooting`, `pulsim.run_harmonic_balance`,
  `pulsim.sweep`, `pulsim.monte_carlo` — additional analysis
  drivers.

The MMC reference project in particular exercises every layer of the
stack: 12 controllable switches plus 6 floating capacitors per
phase leg, multicarrier phase-shifted PWM at the modulator level,
and a step-observer-driven sort-and-select balancing controller
[@Saeedifard:2010:MMC] that keeps all sub-module capacitor
voltages within fractions of a volt of their target through the
entire simulated run.

# Acknowledgements

The author thanks the broader open-source power-electronics
community — in particular the maintainers of `ngspice`, the
`scikit-build-core` developers who made the C++/Python build trivial,
and the early adopters who filed the bug reports that hardened the
PWL cache. Development was supported by personal time.

# References
