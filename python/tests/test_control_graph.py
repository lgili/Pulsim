"""The compiled control graph: slots, schedule, algebraic loops
(add-compiled-control-graph, task groups 2 and 3)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import pulsim as p


def _spec():
    return {
        "err": (p.Subtract(), dict(a=1.0, b="channel:meas"), "e"),
        "gain": (p.Gain(k=2.0), dict(x="channel:e"), "u"),
        "half": (p.Gain(k=0.5), dict(x="channel:u"), "y"),
    }


def _build(order, spec=None, n_rec=()):
    spec = spec or _spec()
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    for name in order:
        blk, inputs, out = spec[name]
        ch.add(name, blk, inputs=inputs, output=out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = ch.make_step_observer(b, dt=1e-6)
    for name in n_rec:
        ch._cxx_chain.record_channel(name)
    return ch, obs


def _drive(order, n=200):
    ch, obs = _build(order, n_rec=("y",))
    cxx = ch._cxx_chain
    meas = 0.5 * np.sin(np.arange(n) * 0.05)
    x = np.zeros(1)
    for k in range(n):
        cxx.set_channel("meas", float(meas[k]))
        obs(k * 1e-6, x)
    return np.asarray(cxx.get_channel_history("y")), meas


def test_any_insertion_order_gives_bit_identical_results():
    ref, meas = _drive(["err", "gain", "half"])
    np.testing.assert_array_equal(ref, 0.5 * (2.0 * (1.0 - meas)))
    for order in itertools.permutations(["err", "gain", "half"]):
        y, _ = _drive(list(order))
        assert y.tobytes() == ref.tobytes(), order


def test_schedule_and_reorder_report():
    ch, _ = _build(["err", "gain", "half"])
    assert list(ch._cxx_chain.schedule()) == ["err", "gain", "half"]
    assert list(ch._cxx_chain.reorders()) == []
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    rev = p.MixedDomainBlockChain()
    for name in ["half", "gain", "err"]:
        blk, inputs, out = _spec()[name]
        rev.add(name, blk, inputs=inputs, output=out)
    with pytest.warns(UserWarning, match="differs from insertion") as rec:
        rev.make_step_observer(b, dt=1e-6)
    msg = str(rec[0].message)
    assert "'half' reads 'u' written by 'gain'" in msg
    assert "'gain' reads 'e' written by 'err'" in msg
    assert list(rev._cxx_chain.schedule()) == ["err", "gain", "half"]
    assert len(rev._cxx_chain.reorders()) == 2


def _loop_spec(with_delay):
    spec = {
        "g1": (p.Gain(k=0.5), dict(x="channel:fb"), "a"),
        "sum": (p.Sum(), dict(a="channel:a", b=1.0), "s"),
        "g2": (p.Gain(k=0.5), dict(x="channel:s"),
               "fb_raw" if with_delay else "fb"),
    }
    if with_delay:
        spec["z"] = (p.DelayBlock(samples=1), dict(x="channel:fb_raw"), "fb")
    return spec


def test_algebraic_loop_is_a_build_error_naming_its_blocks():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    for name, (blk, inputs, out) in _loop_spec(False).items():
        ch.add(name, blk, inputs=inputs, output=out)
    with pytest.raises(p.ControlGraphBuildError,
                       match=r"algebraic loop.*'g1', 'sum', 'g2'.*DelayBlock"):
        ch.make_step_observer(b, dt=1e-6)
    assert issubclass(p.ControlGraphBuildError, ValueError)


@pytest.mark.parametrize("order", list(itertools.permutations(
    ["g1", "sum", "g2", "z"])))
def test_a_delay_in_the_loop_makes_it_run(order):
    spec = _loop_spec(True)
    ch, obs = _build(order, spec, n_rec=("fb",))
    x = np.zeros(1)
    for k in range(60):
        obs(k * 1e-6, x)
    fb = np.asarray(ch._cxx_chain.get_channel_history("fb"))
    # fb[k] = fb_raw[k-1], fb_raw[k] = 0.5 * (0.5 * fb[k] + 1)
    expect = [0.0]
    for _ in range(59):
        expect.append(0.5 * (0.5 * expect[-1] + 1.0))
    assert fb.tolist() == expect
    assert fb[-1] == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_a_block_reading_its_own_output_is_a_loop():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=0.5), inputs=dict(x="channel:y"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="'g' reads its own"):
        ch.make_step_observer(b, dt=1e-6)


def test_a_signal_with_two_writers_is_a_build_error():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("g1", p.Gain(k=1.0), inputs=dict(x=1.0), output="y")
    ch.add("g2", p.Gain(k=2.0), inputs=dict(x=1.0), output="y")
    with pytest.raises(p.ControlGraphBuildError,
                       match="'y' is written by two blocks, 'g1' and 'g2'"):
        ch.make_step_observer(b, dt=1e-6)


def test_block_evaluation_counter():
    ch, obs = _build(["err", "gain", "half"])
    x = np.zeros(1)
    for k in range(25):
        obs(k * 1e-6, x)
    assert ch._cxx_chain.n_evals == 25
    assert ch._cxx_chain.n_block_evals == 25 * 3


@pytest.mark.parametrize("block,inputs", [
    (p.Gain(k=2.0), dict(x="out")),
    (p.Limiter(output_min=0.0, output_max=1.0), dict(x="out")),
])
def test_node_gain_and_limiter_compile_to_the_kernel(block, inputs):
    # Before: the C++ build failed (a node ref with no name; Limiter's
    # attributes read as .min/.max) and the chain ran on the Python
    # evaluator behind a warning.
    b = p.CircuitBuilder()
    b.add_resistor("R", "out", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("blk", block, inputs=inputs, output="u")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ch.make_step_observer(b, dt=1e-6)
    assert ch._cxx_chain is not None


# ---------------------------------------------------------------------------
# edge cases found by the step-3 review
# ---------------------------------------------------------------------------

def test_an_empty_channel_name_reads_zero_and_names_no_slot():
    k = p._pulsim
    ch = k.CxxBlockChain()
    ch.add_gain(2.0, k.CxxInputRef.channel(""), "y")
    ch.set_channel("", 5.0)          # no slot: a no-op, not a stray write
    ch.record_channel("")
    obs = ch.make_step_observer(1e-6)
    obs(0.0, np.zeros(1))
    assert ch.get_channel("y") == 0.0 and ch.get_channel("") == 0.0


def test_a_nameless_negative_node_index_is_ground():
    k = p._pulsim
    ch = k.CxxBlockChain()
    ch.add_gain(2.0, k.CxxInputRef.node_idx(-1), "y")   # node_id_of("gnd")
    ch.make_step_observer(1e-6)(0.0, np.ones(2))
    assert ch.get_channel("y") == 0.0


def test_a_failed_add_leaves_no_phantom_reads():
    k = p._pulsim
    ch = k.CxxBlockChain()
    with pytest.raises(p.ControlGraphBuildError, match="not resolved"):
        ch.add_subtract(k.CxxInputRef.channel("y"), k.CxxInputRef.node("n"),
                        "e")
    ch.add_gain(1.0, k.CxxInputRef.const_value(1.0), "y")  # would loop
    ch.compile()                     # on the dropped read of 'y'
    assert list(ch.schedule()) == ["block0"]


def test_delay_samples_below_one_is_a_typed_build_error():
    k = p._pulsim
    for n in (0, -2):
        with pytest.raises(p.ControlGraphBuildError, match=f"got {n}"):
            k.CxxBlockChain().add_delay(n, 0.0, k.CxxInputRef.const_value(1.0),
                                        "y")


def test_reset_zeroes_signals_and_refills_the_delay():
    k = p._pulsim
    ch = k.CxxBlockChain()
    ch.add_delay(2, 0.25, k.CxxInputRef.const_value(1.0), "y")
    obs = ch.make_step_observer(1e-6)
    seq = []
    for i in range(4):
        obs(i * 1e-6, np.zeros(1))
        seq.append(ch.get_channel("y"))
    ch.reset()
    assert ch.get_channel("y") == 0.0
    obs(0.0, np.zeros(1))
    assert seq == [0.25, 0.25, 1.0, 1.0] and ch.get_channel("y") == 0.25

