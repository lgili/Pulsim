#!/usr/bin/env python3
"""The candidate "next defaults" profile of the number-changing release, as data.

    next_defaults.py --list        # print the table (option, value, engine, owner)

Nothing here changes a library default: the profile is applied by a harness
(``apply_profile`` to a ``simulate()`` call's keywords, ``builder_form`` to the
builder calls) and only to keywords the caller did not set. Each entry names the
OpenSpec change and the decision that owns it (CORE_ROADMAP decision 3: one
release flips them all).

``form='branch'`` is not a ``simulate()`` keyword: it is a builder-time choice.
The harness applies it by patching ``CircuitBuilder.add_switch`` /
``add_diode`` for the duration of a build so that a netlist that gives
``g_on`` / ``g_off`` builds the branch form with ``R_on = 1/g_on`` and the same
``g_off`` (design D6 of add-branch-form-ideal-switch; the same patch as
``ideal_switch_flip_evidence.branch_form``). MOSFET / IGBT helpers, Shockley
diodes and YAML-loaded circuits are not reached by that patch.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]


@dataclass(frozen=True)
class Entry:
    value: object
    change: str
    decision: str
    short: str            # the configuration suffix of this option alone


# simulate() keywords, per engine the run lands on ('dsed' = 'exact').
PROFILE = {
    "pwl": {
        "discontinuity_restart": Entry(
            True, "add-discontinuity-restart-rule",
            "DECISIONS 4 (default off until the release); CORE_ROADMAP decisions 2-3 "
            "(SDIRK(2,2)x3, gate mask sampled at the step start, same flag)", "restart"),
        "enable_substep_state_correction": Entry(
            True, "add-fixed-step-event-splitting",
            "DECISIONS 1 (flips with the restart rule, maintainer's); refused without the rule", "split"),
        "mode_selection": Entry(
            "auto", "add-complementarity-mode-selection",
            "DECISIONS F5 ('auto' is the value the release flips to)", "mode_auto"),
        "control_tick": Entry(
            "state", "add-compiled-control-graph",
            "DECISIONS G30 (default 'label' until the release; G32 queue ticks)", "tick_state"),
        "device_path": Entry(
            "interface", "add-unified-device-interface",
            "DECISIONS 23/25 (the release flips the default and deletes the hand stamps)",
            "dev_interface"),
    },
    "trbdf2": {
        "mode_selection": Entry(
            "auto", "add-complementarity-mode-selection",
            "DECISIONS F5 (on trbdf2 'auto' resolves to the settle: no 'auto' rule there yet)",
            "mode_auto"),
        "control_tick": Entry("state", "add-compiled-control-graph", "DECISIONS G30", "tick_state"),
        "device_path": Entry("interface", "add-unified-device-interface", "DECISIONS 23/25",
                             "dev_interface"),
    },
    "dsed": {
        "mode_selection": Entry("auto", "add-complementarity-mode-selection", "DECISIONS F5",
                                "mode_auto"),
        "control_tick": Entry("state", "add-compiled-control-graph", "DECISIONS G30", "tick_state"),
        # without dt only (an explicit dt / store_every / recorders keep today's samples)
        "output_rtol": Entry(
            1e-5, "update-number-changing-release",
            "DECISIONS 1 (auto route): without dt the exact engine returns every event "
            "instant plus an output grid chosen by rule so that linear interpolation "
            "is within output_rtol (orchestrator's decision; the harness gate is never "
            "loosened); 1e-5, not 1e-4: at 1e-4 24 auto cells grew past 2x the trbdf2 "
            "strict error while agreeing, at 1e-5 none does", "output_grid"),
        # device_path: the exact engine refuses the keyword (option table row
        # 'device_path': dsed False) and has no Newton device to stamp.
    },
}

# Builder-time choice, every engine (applied by builder_form()).
BUILDER = {"form": Entry("branch", "add-branch-form-ideal-switch",
                         "DECISIONS 23 (measurement gate: flip allowed on pwl), 39 (flip "
                         "evidence); CORE_ROADMAP decision 3 (the default switch model)",
                         "branch")}

# engine='auto' under the profile: the route the release gives (not a keyword).
ROUTE = Entry(
    "exact if every segment is exact (exact_route), else trbdf2; an explicit dt -> pwl",
    "add-exact-propagator-strategy",
    "D11 + 'Route report and the default-flip gate' (2026-11-18); task 19.1 (open)",
    "auto")

# Opt-in, not a default: an extra column only.
EXTRA = {"esdirk43": ("trbdf2", {"integrator": "esdirk43"},
                      Entry("esdirk43", "add-high-order-stiff-integrator",
                            "stays opt-in (CORE_ROADMAP corrections: helps resonant, "
                            "costs 2.5x per step on event-limited circuits)", "esdirk43"))}

# Number-changing items the release lists that a harness cannot apply on this tree.
NOT_APPLIED = (
    ("trbdf2", "step_grid_per_octave = 4", "update-trbdf2-factor-reuse",
     "tasks 4.2 / 6.1 open: the option does not exist on this tree"),
    ("trbdf2", "filter_error_estimate / sdirk_after_settle", "update-trbdf2-post-event-restart",
     "DECISIONS 9-10: C++-only TrBdf2Options, no simulate() keyword; both stay off"),
    ("pwl, trbdf2", "composite diode V_F as a conduction drop", "update-composite-diode-forward-drop",
     "nothing to flip: already the default on this tree (the add_mosfet / Level-1 docstrings "
     "read V_F as the body diode's CONDUCTION DROP); a builder default, not a run option"),
    ("all", "one undriven-switch policy and blocks 2-8", "add-kernel-simulation-object",
     "DECISIONS 23 (branch feat/core-p3-simulation-object): owner decisions, not options"),
)


def engine_key(engine: str) -> str:
    return "dsed" if engine == "exact" else engine


def profile_kwargs(engine: str, only=None) -> dict:
    """The profile's simulate() keywords for ``engine`` ('auto' has none: route first).
    ``only``: a set of option names to keep (one option alone)."""
    rows = PROFILE.get(engine_key(engine), {})
    return {k: e.value for k, e in rows.items() if only is None or k in only}


def apply_profile(engine: str, kwargs: dict, only=None) -> dict:
    """``kwargs`` with every profile option for ``engine`` that the caller did not set."""
    return {**profile_kwargs(engine, only), **kwargs}


@contextlib.contextmanager
def builder_form(on: bool = True):
    """add_switch / add_diode build the branch form with the same values (design D6)."""
    if not on:
        yield
        return
    from ideal_switch_flip_evidence import branch_form
    with branch_form():
        yield


def table() -> list:
    rows = [(eng, k, e) for eng, d in PROFILE.items() for k, e in d.items()]
    rows += [("all", k, e) for k, e in BUILDER.items()]
    rows += [("auto", "route", ROUTE)]
    rows += [(eng, "integrator (extra column)", e) for eng, _, e in EXTRA.values()]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the profile table")
    args = ap.parse_args(argv)
    if not args.list:
        ap.print_help()
        return 0
    print("| engine | option | value | change | decision |\n|---|---|---|---|---|")
    for eng, k, e in table():
        print(f"| {eng} | `{k}` | `{e.value!r}` | `{e.change}` | {e.decision} |")
    print("\nNot applied by the harness:\n")
    for eng, what, ch, why in NOT_APPLIED:
        print(f"- {eng}: {what} (`{ch}`): {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
