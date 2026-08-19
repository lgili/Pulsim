"""Phase 1: SwitchStateMask beyond 64 switches (dynamic width).

The v1.x mask was a single uint64 whose constructor raised for more
than 64 switches, so >64-switch circuits (e.g. 120-switch MMC phases)
could not be represented at all.  Phase 1 replaces the backing store
with a dynamic word array; the Python-visible API (set/get/size) is
unchanged and now accepts any width.
"""

import pulsim


def test_mask_beyond_64_switches():
    m = pulsim.SwitchStateMask(70)
    assert m.size == 70
    m.set(0, True)
    m.set(63, True)
    m.set(64, True)   # first bit of the second word
    m.set(69, True)
    assert m.get(0) and m.get(63) and m.get(64) and m.get(69)
    assert not m.get(65)
    m.set(64, False)
    assert not m.get(64)


def test_mask_hundreds_of_switches():
    # Heap-spill regime (> 128 bits): a 200-SM MMC arm scale.
    m = pulsim.SwitchStateMask(400)
    assert m.size == 400
    m.set(399, True)
    assert m.get(399)
    assert not m.get(398)


def test_to_string_reports_width():
    m = pulsim.SwitchStateMask(70)
    m.set(69, True)
    s = m.to_string()
    assert "N=70" in s
    assert s.startswith("0b1")  # MSB-first: bit 69 leads
