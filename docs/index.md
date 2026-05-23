# Pulsim Documentation

<div class="pulsim-hero">
  <h1>Pulsim</h1>
  <p>Power-electronics circuit simulator — C++23 header-only kernel with a Python-first API.</p>
  <p>Recommended surface: <code>import pulsim as p</code> + the <code>CircuitBuilder</code> + <code>p.simulate(...)</code> pipeline.</p>
  <div class="pulsim-hero-actions">
    <a class="md-button md-button--primary" href="v2/getting-started/">Get Started</a>
    <a class="md-button" href="v2/mental-model/">Mental Model</a>
    <a class="md-button" href="v2/api-reference/">API Reference</a>
    <a class="md-button" href="v2/helpers/">UX Helpers</a>
  </div>
</div>

## What's in the box

Pulsim 1.0.0 retired the legacy v1 kernel; the v2 kernel is the only
shipped surface.

- **PLECS-style PWL cache** — switched-converter steady-state in
  milliseconds instead of minutes via a state-space cache indexed by
  the switch combinatorics + Newton refresh on top of the cached
  linear factor for nonlinear devices.
- **Header-only C++23 kernel** — drop ``pulsim/v2/`` into your CMake
  target via ``pulsim::v2``; no static-library link step.
- **Python-first ergonomics** — ``CircuitBuilder`` takes string node
  names and SI-unit parameters; ``p.simulate(b, t_end=, dt=)``
  returns a ``SimulationResult`` whether you ran a transient, an AC
  sweep, or a parameter sweep.
- **MixedDomainBlockChain** — PI/PID, comparators, rate limiters,
  op-amps, FOC blocks, thermal Foster networks composed at kernel
  speed (no Python interpreter cost per step).
- **Frequency-domain** — small-signal MNA Bode + swept-sine FRA +
  closed-loop GM/PM measurement, all in the same surface.

## Where to start

If you're new: ``v2/getting-started.md`` walks you from ``cmake -S
. -B build`` to a closed-loop buck Bode plot in ten minutes.

If you're porting v1 code: ``migration-guide.md`` maps every v1
idiom to its v2 equivalent (and lists the handful of features that
didn't migrate).

If you're writing your own helpers: ``pulsim-v2/`` has the layer-by-layer
internal architecture (Layer 0 numeric primitives all the way up to
the high-level builder API and YAML loader).

## Quality gates

- ``troubleshooting.md`` — common build / import failures.
- ``v2/gotchas.md`` — every footgun we've hit so far.
- ``performance-tuning.md`` — SIMD / cache hygiene.
- ``build-system.md`` — what gets built and why.
- ``versioning-and-release.md`` — docs publishing setup (MkDocs Material + mike).
