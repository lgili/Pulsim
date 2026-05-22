"""Robust transient — automatic retry + Newton/linear-solver tuning.

`pulsim.run_transient(circuit, t_start, t_stop, dt, robust=True)` is the
ergonomic wrapper around the C++ Simulator. With ``robust=True`` (the
default) it:

  - tunes the Newton options (auto-damping, voltage/current step limits,
    trust-region, looser min damping) — the recipe documented in
    ``docs/robustness-policy.md``
  - tunes the linear-solver stack (KLU first, then EnhancedSparseLU,
    then GMRES + ILUT) with fallback enabled
  - retries the run with progressively smaller dt and (optionally)
    auto-injected high-value bleeder resistors when convergence fails

The classic failure mode without robustness: a switching converter where
one PWM step lands on a near-singular Jacobian. ``robust=True`` keeps
the simulation alive at the cost of some wallclock; ``robust=False``
returns the bare diagnostic so power users can write their own retry.

Run::

    python 09_robustness_wrapper.py            # robust=True (default)
    python 09_robustness_wrapper.py --strict   # robust=False side-by-side

See also: docs/robustness-policy.md
"""

from __future__ import annotations

import argparse

import pulsim


def build_diff_pair_oscillator() -> pulsim.Circuit:
    """A small RLC ringer driven by a step source. Stable but non-trivial
    — exercises the retry loop's Newton tuning without needing a hard-to-
    converge nonlinear circuit."""
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    n1  = ckt.add_node("n1")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 5.0)
    ckt.add_resistor("R1", in_, n1, 1.0)
    ckt.add_inductor("L1", n1, out, 100e-6, 0.0)
    ckt.add_capacitor("C1", out, ckt.ground(), 10e-6, 0.0)
    ckt.add_resistor("Rload", out, ckt.ground(), 50.0)
    return ckt


def run_one(robust: bool) -> None:
    label = "robust=True" if robust else "robust=False"
    ckt = build_diff_pair_oscillator()
    print(f"=== {label} ===")
    time, states, success, message = pulsim.run_transient(
        ckt, 0.0, 1e-3, 1e-7, robust=robust,
    )
    out_idx = ckt.get_node("out")
    print(f"  success: {success}")
    if message:
        print(f"  message: {message}")
    print(f"  samples: {len(states)}    final V_out: {states[-1][out_idx]:.4f} V")
    print(f"  end-time: {time[-1]*1e3:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="also run with robust=False to compare")
    args = parser.parse_args()

    run_one(robust=True)
    if args.strict:
        print()
        run_one(robust=False)


if __name__ == "__main__":
    main()
