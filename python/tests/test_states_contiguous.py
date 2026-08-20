"""v2.0 Phase 1: `res.states` as a zero-copy 2-D numpy view + store_every.

Audit finding `waveform-storage-vector-of-vectors`. v1.x built a
fresh Python list of N 1-D ndarray objects on EVERY `res.states`
access (O(N) object construction per access, on top of one heap
block per sample in C++). It is now ONE contiguous buffer exposed
as a read-only (num_steps, state_size) view.
"""

import numpy as np
import pytest

import pulsim


def _rc_run(store_every=1, t_end=1e-4, dt=1e-6):
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 12.0)
    b.add_switch("S", "vin", "vout", 1e-3, 1e9)
    b.add_resistor("R", "vout", "gnd", 10.0)
    b.add_capacitor("C", "vout", "gnd", 1e-6)
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(dt)
    opts = pulsim.SimulationOptions(t_start=0.0, t_end=t_end, dt=dt)
    opts.store_every = store_every

    def sw(_t):
        m = pulsim.SwitchStateMask(1)
        m.set(0, True)
        return m

    return pulsim.run_transient(cache, b.graph, b.pool, opts, sw), opts


def test_states_is_2d_readonly_ndarray():
    res, _opts = _rc_run()
    s = res.states
    assert isinstance(s, np.ndarray)
    assert s.ndim == 2
    assert s.shape == (res.num_steps(), s.shape[1])
    assert s.dtype == np.float64
    # Aliases kernel memory → read-only, so an accidental write is a
    # loud error rather than silent corruption. (v1.x rows were
    # non-writeable views too, so this preserves that behaviour.)
    assert not s.flags.writeable
    with pytest.raises(ValueError):
        s[0, 0] = 1.0
    # ...and .copy() gives a mutable array, as documented.
    c = s.copy()
    c[0, 0] = 1.0
    assert c.flags.writeable


def test_indexing_matches_v1_list_semantics():
    res, _ = _rc_run()
    s = res.states
    assert len(s) == res.num_steps()
    assert len(res.times) == res.num_steps()
    # Row access, scalar access, negative index, slicing, iteration.
    assert s[-1].shape == (s.shape[1],)
    assert isinstance(float(s[0][0]), float)
    assert s[3][1] == s[3, 1]
    assert s[2:5].shape == (3, s.shape[1])
    assert sum(1 for _ in s) == res.num_steps()
    assert np.asarray(s).shape == s.shape


def test_states_view_survives_result_temporary():
    # The view's base keeps the C++ result alive: grabbing states
    # from a temporary result must not dangle.
    s = _rc_run()[0].states
    assert np.isfinite(s).all()
    assert abs(float(s[-1][0]) - 12.0) < 1e-6   # source node at 12 V


def test_states_access_is_O1_not_O_n():
    # v1.x rebuilt a list of N ndarray objects per access; the view
    # is a constant-cost wrapper. Compare a short and a long run:
    # per-access cost must not scale with the sample count.
    import time

    short, _ = _rc_run(t_end=1e-4)      # ~101 samples
    long_, _ = _rc_run(t_end=1e-2)      # ~10001 samples
    assert long_.num_steps() > 50 * short.num_steps()

    def cost(res, reps=200):
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = res.states
        return (time.perf_counter() - t0) / reps

    # Warm up, then measure. Generous 10x bound: the point is that
    # a 100x larger run does not cost 100x more to access.
    cost(short, 20)
    cost(long_, 20)
    assert cost(long_) < 10 * cost(short) + 1e-4


def test_states_bytes_matches_shape():
    res, _ = _rc_run()
    s = res.states
    assert res.states_bytes == s.shape[0] * s.shape[1] * 8


def test_store_every_decimates_on_a_uniform_grid():
    full, _opts_full = _rc_run(store_every=1, t_end=1e-4)
    dec, opts_dec = _rc_run(store_every=10, t_end=1e-4)

    assert opts_dec.expected_sample_count() == dec.num_steps()
    assert dec.num_steps() < full.num_steps()

    # Sample j of the decimated run == sample j*m of the full run,
    # exactly: decimation changes what is STORED, not the solve.
    m = 10
    for j in range(dec.num_steps()):
        assert dec.times[j] == pytest.approx(full.times[j * m], abs=1e-15)
        np.testing.assert_array_equal(dec.states[j], full.states[j * m])

    # Grid stays strictly uniform at m*dt (FFT/harmonic analysis on
    # the result stays valid, just at the coarser spacing).
    spacing = np.diff(np.asarray(dec.times))
    assert np.allclose(spacing, spacing[0], rtol=0, atol=1e-15)

    # Memory drops by the stride.
    assert dec.states_bytes * 5 < full.states_bytes


def test_store_every_default_is_one():
    opts = pulsim.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-6)
    assert opts.store_every == 1
    assert opts.expected_sample_count() == opts.expected_step_count()
    opts.store_every = 4
    assert opts.expected_sample_count() == (
        opts.expected_step_count() + 3) // 4
