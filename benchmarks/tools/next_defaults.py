#!/usr/bin/env python3
"""The defaults of the number-changing release, as data: legacy -> current.

    next_defaults.py --list        # print the table (option, current, legacy, engine, owner)

Since the release (update-number-changing-release R1) ``PROFILE`` / ``PLAN`` /
``BUILDER`` / ``ROUTE`` are the LIBRARY DEFAULTS (``check_current()`` proves it:
an empty list); each entry names the OpenSpec change and decision that owns it
and the explicit value that restores the pre-release behaviour (``legacy``).

``LEGACY`` is the harness profile of the legacy period: the explicit options
that reproduce the pre-release numbers (``legacy_kwargs`` / ``install_legacy``:
every ``simulate()`` call gets them where it did not set the keyword,
``engine='auto'`` becomes the pre-release route, ``add_switch`` /
``add_diode`` build the conductance form). The old goldens stay checkable
with it. Before the release ``apply_profile`` / ``builder_form`` /
``plan_defaults`` injected the then-candidate profile; they still do (now
redundant with the defaults, kept for the rehearsal tool).

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
            "DECISIONS 2 (auto route): without dt the exact engine returns every event "
            "instant plus an output grid chosen by rule so that linear interpolation "
            "is within output_rtol (orchestrator's decision; the harness gate is never "
            "loosened); 1e-5, not 1e-4: at 1e-4 24 auto cells grew past 2x the trbdf2 "
            "strict error while agreeing, at 1e-5 none does", "output_grid"),
        # device_path: the exact engine refuses the keyword (option table row
        # 'device_path': dsed False) and has no Newton device to stamp.
    },
}

# Library defaults the run plan reads when the call leaves the keyword None
# (``pulsim._plan`` attributes, every engine and the 'auto' route): the harness
# sets the attribute, not a keyword (apply with plan_defaults()).
PLAN = {
    "DEFAULT_UNDRIVEN": Entry(
        "refuse", "add-kernel-simulation-object",
        "DECISIONS 26.1 (task 11.3; orchestrator decision 2026-11-19: an undriven "
        "controlled switch is refused by name on every engine; simulate(undriven=) "
        "keeps 'open' / 'closed' / None reachable)", "undriven_refuse"),
    "UNIFORM_ENGINE_RULES": Entry(
        True, "add-kernel-simulation-object",
        "DECISIONS 26.2 (items 6-7 of DECISIONS 23: one count check per option on every "
        "engine that reads it; on_impulse refused by name where the run cannot detect "
        "an impulse; the exact engine's impulse refusal names the device kind)",
        "uniform_rules"),
}

# Builder-time choice, every engine (applied by builder_form()).
BUILDER = {"form": Entry("branch", "add-branch-form-ideal-switch",
                         "DECISIONS 23 (measurement gate: flip allowed on pwl), 39 (flip "
                         "evidence); CORE_ROADMAP decision 3 (the default switch model)",
                         "branch")}

# engine='auto' under the profile: the route the release gives (not a keyword).
ROUTE = Entry(
    "without dt: exact if the kernel's exact capability table takes every device, no "
    "b_extra_fn and no keyword the exact engine refuses, else trbdf2 (and trbdf2 when the "
    "exact run refuses by name); an explicit dt -> pwl",
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
    ("pwl", "mmc_arms / on_cache, progress / live_stream", "add-kernel-simulation-object",
     "DECISIONS 26.3: kept per engine (they need the fixed engine's own cache / per-step "
     "ring); every other engine refuses them by name (option table)"),
)


# The legacy period (one release): per engine the explicit simulate() keywords
# that restore the pre-release behaviour of every flipped default. undriven:
# the old per-engine behaviour (pwl / exact ran undriven switches CLOSED with a
# warning, trbdf2 refused); output_rtol=0 is the rule's explicit off.
LEGACY = {
    "pwl": {"discontinuity_restart": False, "enable_substep_state_correction": False,
            "mode_selection": "legacy",
            "control_tick": "label", "device_path": "legacy", "undriven": "closed"},
    "trbdf2": {"mode_selection": "legacy", "control_tick": "label",
               "device_path": "legacy", "undriven": "refuse"},
    "dsed": {"mode_selection": "legacy", "control_tick": "label", "output_rtol": 0,
             "undriven": "closed"},
}
LEGACY_BUILDER = {"form": "conductance"}        # add_switch / add_diode (YAML: `form: conductance`)
LEGACY_YAML = {"switch_form": "conductance"}    # load_yaml_file / load_yaml_string: a netlist's default
# No keyword: the uniform rules change only refusal order and messages.
LEGACY_PLAN = {"UNIFORM_ENGINE_RULES": False}


def legacy_route(dt) -> str:
    """The pre-release ``engine='auto'``: pwl with a dt, else trbdf2."""
    return "pwl" if dt else "trbdf2"


def legacy_kwargs(engine: str) -> dict:
    return dict(LEGACY.get(engine_key(engine), {}))


@contextlib.contextmanager
def legacy():
    """The legacy profile for the duration: every ``pulsim.simulate`` call takes
    ``LEGACY`` for its engine where it did not set the keyword (``engine='auto'``
    first takes ``legacy_route``; a ``solver=`` bundle's set fields count as
    set), ``add_switch`` / ``add_diode`` without ``form`` build the conductance
    form, YAML netlists load with ``LEGACY_YAML``, ``LEGACY_PLAN`` is set.
    Restored after."""
    import pulsim
    from pulsim import _plan
    orig, cb = pulsim.simulate, pulsim.CircuitBuilder
    sw, di = cb.add_switch, cb.add_diode
    old_plan = {k: getattr(_plan, k) for k in LEGACY_PLAN}

    def simulate(*args, **kw):
        eng = kw.get("engine", "auto")
        if eng == "auto":
            dt = args[1] if len(args) > 1 else kw.get("dt")
            b = args[0] if args else kw.get("builder")
            if dt is None and isinstance(b, pulsim.LoadedCircuit):
                dt = b.options.dt or None
            eng = kw["engine"] = legacy_route(dt)
        solver = kw.get("solver")
        add = {k: v for k, v in legacy_kwargs(eng).items()
               if k not in kw and not (solver is not None and getattr(solver, k, None) is not None)}
        return orig(*args, **add, **kw)

    def add_switch(self, *a, **k):
        k.setdefault("form", LEGACY_BUILDER["form"])
        return sw(self, *a, **k)

    def add_diode(self, *a, **k):
        k.setdefault("form", LEGACY_BUILDER["form"])
        return di(self, *a, **k)
    yf, ys = pulsim.load_yaml_file, pulsim.load_yaml_string

    def load_yaml_file(*a, **k):
        return yf(*a, **{**LEGACY_YAML, **k})

    def load_yaml_string(*a, **k):
        return ys(*a, **{**LEGACY_YAML, **k})
    pulsim.simulate, cb.add_switch, cb.add_diode = simulate, add_switch, add_diode
    pulsim.load_yaml_file, pulsim.load_yaml_string = load_yaml_file, load_yaml_string
    for k, v in LEGACY_PLAN.items():
        setattr(_plan, k, v)
    try:
        yield
    finally:
        pulsim.simulate, cb.add_switch, cb.add_diode = orig, sw, di
        pulsim.load_yaml_file, pulsim.load_yaml_string = yf, ys
        for k, v in old_plan.items():
            setattr(_plan, k, v)


def check_current() -> list:
    """The release is the profile: every PROFILE / PLAN / BUILDER entry is the
    library's default. Returns the mismatches (empty = the flip is the profile)."""
    import inspect

    import pulsim
    from pulsim import _plan
    sig = inspect.signature(pulsim.simulate).parameters
    resolved = {  # keyword -> the value simulate() runs with when the call leaves it unset
        "discontinuity_restart": _plan.DEFAULT_DISCONTINUITY_RESTART,
        "enable_substep_state_correction": _plan.DEFAULT_SUBSTEP_STATE_CORRECTION,
        "mode_selection": _plan.DEFAULT_MODE_SELECTION,
        "device_path": _plan.DEFAULT_DEVICE_PATH,
        "output_rtol": _plan.DEFAULT_OUTPUT_RTOL,
        "control_tick": sig["control_tick"].default,
    }
    bad = [f"{eng}.{k}: library {resolved.get(k)!r} != profile {e.value!r}"
           for eng, rows in PROFILE.items() for k, e in rows.items() if resolved.get(k) != e.value]
    bad += [f"_plan.{k}: library {getattr(_plan, k)!r} != profile {e.value!r}"
            for k, e in PLAN.items() if getattr(_plan, k) != e.value]
    for meth in ("add_switch", "add_diode"):
        doc = getattr(pulsim.CircuitBuilder, meth).__doc__ or ""
        if f"form: str = '{BUILDER['form'].value}'" not in doc:
            bad.append(f"CircuitBuilder.{meth}: default form is not {BUILDER['form'].value!r}")
    return bad


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
def plan_defaults(on: bool = True):
    """``pulsim._plan`` takes the PLAN values for the duration (restored after)."""
    from pulsim import _plan
    old = {k: getattr(_plan, k) for k in PLAN}
    try:
        if on:
            for k, e in PLAN.items():
                setattr(_plan, k, e.value)
        yield
    finally:
        for k, v in old.items():
            setattr(_plan, k, v)


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
    rows += [("all", f"_plan.{k}", e) for k, e in PLAN.items()]
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
    print("| engine | option | current | legacy (explicit) | change | decision |\n"
          "|---|---|---|---|---|---|")
    for eng, k, e in table():
        old = (LEGACY.get(eng, {}).get(k) if eng in LEGACY else LEGACY_BUILDER.get(k)
               if k in LEGACY_BUILDER else "explicit engine=" if k == "route" else
               LEGACY_PLAN.get(k.removeprefix("_plan."), "-"))
        if k == "_plan.DEFAULT_UNDRIVEN":
            old = "undriven= per engine (LEGACY)"
        print(f"| {eng} | `{k}` | `{e.value!r}` | `{old!r}` | `{e.change}` | {e.decision} |")
    print("\nNot applied by the harness:\n")
    for eng, what, ch, why in NOT_APPLIED:
        print(f"- {eng}: {what} (`{ch}`): {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
