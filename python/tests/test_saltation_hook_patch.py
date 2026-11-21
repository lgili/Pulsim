"""The event hook comparator-gate saltation needs, kept applicable
(add-design-optimization-workflow DECISIONS 30).

`probes/saltation_hook.patch` adds `HasEventRegister` and one call in
`PEDSimulator::fire_event_` (core/include/pulsim/dsed/scheduler.hpp, not
editable in the slice that wrote it); `probes/saltation_hook_check.cpp`
is the check that passes with it (bit-identical trajectory, every event
named). This test keeps the patch's context in step with the header, so
whoever lands the hook applies it as measured."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBES = ROOT / "openspec/changes/add-design-optimization-workflow/probes"


def _hunks(text):
    hunk = None
    for line in text.splitlines():
        if line.startswith("@@"):
            if hunk:
                yield hunk
            hunk = []
        elif hunk is not None and line[:1] in (" ", "-"):
            hunk.append(line[1:])
    if hunk:
        yield hunk


def test_the_hook_patch_still_applies_to_the_scheduler():
    patch = (PROBES / "saltation_hook.patch").read_text(encoding="utf-8")
    target = patch.splitlines()[0].removeprefix("--- a/")
    header = (ROOT / target).read_text(encoding="utf-8")
    if "HasEventRegister" in header:              # landed: the call is there
        assert "switch_fn_.register_event(t_event, name, ptype, aux_index)" \
            in header
        return
    hunks = list(_hunks(patch))
    assert len(hunks) == 2
    for old in hunks:
        assert "\n".join(old) in header, old[0]
    check = (PROBES / "saltation_hook_check.cpp").read_text(encoding="utf-8")
    assert "register_event" in check and "saltation_hook.patch" in check
