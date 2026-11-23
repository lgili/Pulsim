"""The byte-compare tool's rule for result attributes added after a baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "tools"))
from pwl_identity import first_difference  # noqa: E402

BASE = {".result.times": "h1", ".result.x[0].y": "1", ".result.x#len": "1"}


def test_new_attribute_is_not_a_difference():
    new = {**BASE, ".result.right_limit_index": "None", ".result.inner_events.after": "z"}
    assert first_difference(BASE, new) is None


def test_changes_to_recorded_attributes_still_differ():
    assert first_difference(BASE, {**BASE, ".result.times": "h2"})
    assert first_difference(BASE, {**BASE, ".result.x[1].y": "2", ".result.x#len": "2"})
    assert first_difference(BASE, {k: v for k, v in BASE.items() if k != ".result.times"})
