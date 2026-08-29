"""MMC Thevenin arm (GGJ aggregation) — the crown parity tests.

v2.0 Phase 3, "obra №2". The C++ unit tests pin the aggregation
math against closed forms; THESE tests pin the claim that matters —
the aggregation is ALGEBRAIC, so an explicit chain of real switches
and capacitors, driven by the same gates at the same dt through the
same pwl engine, must produce the same arm current and the same
capacitor voltages. If the coupling (observer timing, b_extra sign,
parametric refactor) were off by one step or one sign anywhere, the
traces would diverge immediately.
"""

import math

import numpy as np
import pytest

import pulsim as p

R_ON = 1e-3          # per-SM conducting resistance [ohm]
G_ON = 1.0 / R_ON
G_OFF = 1e-9         # explicit-switch leak — the parity floor
C_SM = 2e-3
N_SM = 4
DT = 1e-6


def _thevenin_run(n_on_fn, t_end, vdc, rload, record_masks=False):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "p", "gnd", vdc)
    b.add_resistor("Rload", "p", "a", rload)
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=N_SM, c_sm=C_SM, r_on=R_ON, v_c_init=0.0,
        n_on=n_on_fn)

    masks = []
    observer = None
    if record_masks:
        # The composed observer runs the MMC driver FIRST, so by the
        # time this fires the arm's `inserted` is the selection the
        # COMING solve will be stamped with. Call #1 is the kernel's
        # priming call — the driver does nothing there (solve #1 is
        # stamped at call #2), so the recorder skips it too.
        calls = [0]

        def observer(t, x):
            calls[0] += 1
            if calls[0] == 1:
                return
            masks.append(tuple(arm._kernel.inserted))

    res = p.simulate(b, t_end=t_end, dt=DT, mmc_arms=[arm],
                      step_observer=observer)
    return res, arm, masks


def _explicit_run(mask_of_solve, t_end, vdc, rload):
    """The same arm as N_SM real half-bridge submodules.

    SM i spans chain node i -> i+1: a bypass switch straight
    across, and an insert path (switch -> capacitor). Gates are
    complementary, taken from `mask_of_solve(k)` for the solve at
    t = k*dt.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "p", "gnd", vdc)
    b.add_resistor("Rload", "p", "a", rload)
    chain = ["a", "n1", "n2", "n3", "gnd"]
    for i in range(N_SM):
        top, bot = chain[i], chain[i + 1]
        b.add_switch(f"Sb{i}", top, bot, G_ON, G_OFF)
        b.add_switch(f"Si{i}", top, f"m{i}", G_ON, G_OFF)
        b.add_capacitor(f"C{i}", f"m{i}", bot, C_SM)

    idx_b = [b.switch_index_of(f"Sb{i}") for i in range(N_SM)]
    idx_i = [b.switch_index_of(f"Si{i}") for i in range(N_SM)]
    n_sw = b.graph.num_switches

    def switch_fn(t):
        k = int(round(t / DT))
        ins = mask_of_solve(k)
        m = p.SwitchStateMask(n_sw)
        for i in range(N_SM):
            m.set(idx_i[i], bool(ins[i]))
            m.set(idx_b[i], not bool(ins[i]))
        return m

    res = p.simulate(b, t_end=t_end, dt=DT, switch_fn=switch_fn)
    v_c_final = [
        float(res.v(f"m{i}")[-1]
              - (res.v(chain[i + 1])[-1]
                 if chain[i + 1] != "gnd" else 0.0))
        for i in range(N_SM)
    ]
    return res, v_c_final


def test_parity_all_in_all_out_square():
    """0/4 square gating: aggregation vs explicit chain, whole run.

    All-in/all-out keeps sort-and-select trivial (the selection is
    unique), so both sides can be driven from the same closed-form
    schedule with no replay machinery — a pure end-to-end check of
    the Norton coupling, the observer timing, and the parametric
    refactor at every gating edge.
    """
    T_HALF = 100 * DT        # 5 kHz square
    t_end = 2000 * DT        # 10 full cycles

    def n_on(t):
        return 0 if int(t / T_HALF) % 2 == 0 else N_SM

    res_t, arm, _ = _thevenin_run(n_on, t_end, vdc=600.0, rload=10.0)

    def mask_of_solve(k):
        t = k * DT
        on = n_on(t)
        return [on == N_SM] * N_SM

    res_e, v_c_expl = _explicit_run(mask_of_solve, t_end,
                                     vdc=600.0, rload=10.0)

    va_t = np.asarray(res_t.v("a"))
    va_e = np.asarray(res_e.v("a"))
    assert va_t.shape == va_e.shape
    # The explicit chain leaks through g_off = 1e-9 S; that is the
    # honest parity floor, far below any coupling/timing bug (which
    # shows up in volts, not microvolts).
    assert np.max(np.abs(va_t - va_e)) < 1e-6

    v_c_thev = arm.v_c
    for i in range(N_SM):
        assert v_c_thev[i] == pytest.approx(v_c_expl[i], abs=1e-6)


def test_parity_partial_insertion_with_balancer_replay():
    """Partial insertion: the balancer's own selections, replayed.

    n_on(t) walks a staircase 0..4. The Thevenin run records which
    submodules its sort-and-select balancer inserted at every step;
    the explicit chain replays exactly those gate patterns. Any
    one-step lag between the recorded selection and the solve it
    was stamped for would break parity at the first n_on change.
    """
    t_end = 1500 * DT

    def n_on(t):
        return int(round(2.0 + 2.0 * math.sin(
            2.0 * math.pi * 1000.0 * t)))

    res_t, arm, masks = _thevenin_run(
        n_on, t_end, vdc=500.0, rload=8.0, record_masks=True)

    n_solves = len(res_t.times) - 1
    assert len(masks) == n_solves
    # The r_changed=False path (count constant, MEMBERSHIP swapped —
    # no refactor, but V_eq and b_extra must restamp) must actually
    # occur in this run, or the replay parity proves nothing about
    # it. Deliberate assert so an innocent edit to the schedule
    # can't silently evaporate the coverage.
    assert any(
        masks[k] != masks[k + 1]
        and sum(masks[k]) == sum(masks[k + 1])
        for k in range(len(masks) - 1)
    )

    def mask_of_solve(k):
        return masks[min(k, n_solves) - 1] if k >= 1 else masks[0]

    res_e, v_c_expl = _explicit_run(mask_of_solve, t_end,
                                     vdc=500.0, rload=8.0)

    va_t = np.asarray(res_t.v("a"))
    va_e = np.asarray(res_e.v("a"))
    assert np.max(np.abs(va_t - va_e)) < 1e-6

    v_c_thev = arm.v_c
    for i in range(N_SM):
        assert v_c_thev[i] == pytest.approx(v_c_expl[i], abs=1e-6)


def test_rc_discharge_matches_trapezoidal_closed_form():
    """One SM through the WHOLE engine vs the companion closed form.

    Same numbers as the C++ unit test, but the current now comes
    back from the network solve through b_extra and the observer's
    i = G·Δv − I_N recovery. N solves with the run's trailing
    finalize give exactly N companion finalizations:
    v_N = v0·(R_t/(R_t+R_c))·q^(N−1), q=(R_t−R_c)/(R_t+R_c).
    """
    b = p.CircuitBuilder()
    # Seed through v_c_preset (v_c_init deliberately 0) so this test
    # also pins that the preset actually reaches the kernel — with
    # init alone, deleting the preset wiring kept the suite green.
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=1, c_sm=2e-3, r_on=1e-3, v_c_init=0.0,
        v_c_preset=[48.0],
        n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 0.5)

    dt = 1e-5
    n = 2000
    p.simulate(b, t_end=n * dt, dt=dt, mmc_arms=[arm])

    r_t = 0.5 + 1e-3
    r_c = dt / (2 * 2e-3)
    q = (r_t - r_c) / (r_t + r_c)
    ref = 48.0 * (r_t / (r_t + r_c)) * q ** (n - 1)
    assert arm.v_c[0] == pytest.approx(ref, rel=1e-9)


def test_composes_with_user_hooks():
    """A user observer and b_extra_fn keep working alongside arms."""
    b = p.CircuitBuilder()
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=2, c_sm=1e-3, r_on=1e-3, v_c_init=10.0,
        n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 1.0)
    b.add_resistor("Rx", "x", "gnd", 2.0)

    seen = []

    def user_obs(t, x):
        seen.append(t)

    state_size = b.pool.state_size(b.graph)
    x_row = b.node_id_of("x")

    def user_b(t):
        v = np.zeros(state_size)
        v[x_row] = -1.0        # inject 1 A into x -> v(x) = 2 V
        return v

    res = p.simulate(b, t_end=20e-6, dt=1e-6, mmc_arms=[arm],
                      step_observer=user_obs, b_extra_fn=user_b)
    assert len(seen) > 0
    assert res.v("x")[-1] == pytest.approx(2.0, rel=1e-9)
    # And the arm still did its job on its own node.
    assert arm.v_c[0] != pytest.approx(10.0, abs=0.0)


def test_three_phase_600_sm_runs_and_makes_sense():
    """The scaling claim, as a test: 6 arms x 100 SM (600
    submodules — the explicit equivalent is 1200 switches with a
    2^1200 mask space) builds instantly, runs a mains cycle, and
    produces textbook waveforms: AC amplitude ~ m·Vdc/2, capacitor
    mean at Vdc/N, no NaN."""
    n = 100
    vdc = 10e3
    v_sm = vdc / n
    m_idx = 0.9
    f0 = 50.0
    dt = 10e-6

    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", vdc)
    arms = []
    for ph, phi in (("a", 0.0), ("b", -2 * math.pi / 3),
                     ("c", 2 * math.pi / 3)):
        def n_up(t, phi=phi):
            return int(round(0.5 * n * (1.0 - m_idx * math.sin(
                2 * math.pi * f0 * t + phi))))

        def n_lo(t, phi=phi):
            return n - int(round(0.5 * n * (1.0 - m_idx * math.sin(
                2 * math.pi * f0 * t + phi))))

        arms.append(p.add_mmc_thevenin_arm(
            b, f"arm_{ph}_u", "dcp", f"m{ph}u",
            n_sm=n, c_sm=10e-3, r_on=1e-3, v_c_init=v_sm,
            n_on=n_up))
        b.add_inductor(f"L{ph}u", f"m{ph}u", f"ac_{ph}", 5e-3)
        b.add_inductor(f"L{ph}l", f"ac_{ph}", f"m{ph}l", 5e-3)
        arms.append(p.add_mmc_thevenin_arm(
            b, f"arm_{ph}_l", f"m{ph}l", "gnd",
            n_sm=n, c_sm=10e-3, r_on=1e-3, v_c_init=v_sm,
            n_on=n_lo))
        b.add_resistor(f"Rload_{ph}", f"ac_{ph}", "n", 20.0)

    # Three mains cycles: the circulating-current transient
    # (tau ~ L_arm/R_arm ~ 14 ms) must die before the amplitude
    # check window.
    res = p.simulate(b, t_end=60e-3, dt=dt, mmc_arms=arms)

    va = np.asarray(res.v("ac_a")) - np.asarray(res.v("n"))
    assert not np.isnan(va).any()
    half = len(va) // 2
    amp = 0.5 * (va[half:].max() - va[half:].min())
    assert amp == pytest.approx(m_idx * vdc / 2, rel=0.05)
    vc_all = np.concatenate([a.v_c for a in arms])
    assert vc_all.mean() == pytest.approx(v_sm, rel=0.01)
    assert vc_all.max() - vc_all.min() < 0.25 * v_sm


def test_refuses_dsed_engine():
    b = p.CircuitBuilder()
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=2, c_sm=1e-3, r_on=1e-3, v_c_init=10.0,
        n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 1.0)
    with pytest.raises(ValueError, match="mmc_arms"):
        p.simulate(b, t_end=1e-4, engine="dsed", rtol=1e-6,
                    mmc_arms=[arm])


def test_preset_length_validated():
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="v_c_preset"):
        p.add_mmc_thevenin_arm(
            b, "arm", "a", "gnd",
            n_sm=4, c_sm=1e-3, r_on=1e-3,
            n_on=lambda t: 1, v_c_preset=[1.0, 2.0])


def test_parity_two_floating_arms():
    """A whole leg: two FLOATING arms (all four terminals are
    internal nodes) with complementary gating, aggregated vs
    explicit. Every other parity test grounds one terminal, which
    hides sign errors that cancel at ground; and this is the only
    parity-grade test where two arms interact through the network.
    """
    n_sm = 2
    t_half = 100 * DT
    t_end = 1000 * DT

    def n_u(t):
        return 0 if int(t / t_half) % 2 == 0 else n_sm

    def n_l(t):
        return n_sm - n_u(t)

    def build_common(b):
        b.add_voltage_source("Vdc", "p", "gnd", 600.0)
        b.add_resistor("Rtop", "p", "a", 5.0)
        b.add_resistor("Rbot", "b2", "gnd", 5.0)

    # --- aggregated ---
    b = p.CircuitBuilder()
    build_common(b)
    arm_u = p.add_mmc_thevenin_arm(
        b, "arm_u", "a", "mid", n_sm=n_sm, c_sm=C_SM,
        r_on=R_ON, v_c_init=0.0, n_on=n_u)
    arm_l = p.add_mmc_thevenin_arm(
        b, "arm_l", "mid", "b2", n_sm=n_sm, c_sm=C_SM,
        r_on=R_ON, v_c_init=0.0, n_on=n_l)
    res_t = p.simulate(b, t_end=t_end, dt=DT,
                        mmc_arms=[arm_u, arm_l])

    # --- explicit: two chains of real switches + caps ---
    be = p.CircuitBuilder()
    build_common(be)
    chains = {"u": ["a", "u1", "mid"], "l": ["mid", "l1", "b2"]}
    for tag, chain in chains.items():
        for i in range(n_sm):
            top, bot = chain[i], chain[i + 1]
            be.add_switch(f"Sb_{tag}{i}", top, bot, G_ON, G_OFF)
            be.add_switch(f"Si_{tag}{i}", top, f"m_{tag}{i}",
                           G_ON, G_OFF)
            be.add_capacitor(f"C_{tag}{i}", f"m_{tag}{i}", bot,
                              C_SM)
    n_sw = be.graph.num_switches
    idx = {(tag, i): (be.switch_index_of(f"Sb_{tag}{i}"),
                       be.switch_index_of(f"Si_{tag}{i}"))
           for tag in ("u", "l") for i in range(n_sm)}

    def switch_fn(t):
        k = int(round(t / DT))
        on_u = n_u(k * DT) == n_sm
        on_l = n_l(k * DT) == n_sm
        m = p.SwitchStateMask(n_sw)
        for tag, on in (("u", on_u), ("l", on_l)):
            for i in range(n_sm):
                sb, si = idx[(tag, i)]
                m.set(si, on)
                m.set(sb, not on)
        return m

    res_e = p.simulate(be, t_end=t_end, dt=DT, switch_fn=switch_fn)

    for node in ("a", "mid", "b2"):
        d = np.max(np.abs(np.asarray(res_t.v(node))
                           - np.asarray(res_e.v(node))))
        assert d < 1e-6, f"node {node}: {d}"
    for tag, arm in (("u", arm_u), ("l", arm_l)):
        chain = chains[tag]
        for i in range(n_sm):
            bot = chain[i + 1]
            v_expl = float(
                np.asarray(res_e.v(f"m_{tag}{i}"))[-1]
                - (np.asarray(res_e.v(bot))[-1]
                   if bot != "gnd" else 0.0))
            assert arm.v_c[i] == pytest.approx(v_expl, abs=1e-6)


def test_arm_from_ground_orientation():
    """Arm registered FROM ground: the Norton pair lands only on the
    `to` node and every sign flips. Same closed form as the RC
    discharge, with the node voltage mirrored negative."""
    b = p.CircuitBuilder()
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "gnd", "a",
        n_sm=1, c_sm=2e-3, r_on=1e-3, v_c_init=48.0,
        n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 0.5)

    dt = 1e-5
    n = 2000
    res = p.simulate(b, t_end=n * dt, dt=dt, mmc_arms=[arm])

    r_t = 0.5 + 1e-3
    r_c = dt / (2 * 2e-3)
    # va = -V_eq*Rl/(Rl+R_eq): mirrored sign of the from='a' case.
    assert res.v("a")[1] == pytest.approx(
        -48.0 * 0.5 / (0.5 + 1e-3 + r_c), rel=1e-9)
    q = (r_t - r_c) / (r_t + r_c)
    ref = 48.0 * (r_t / (r_t + r_c)) * q ** (n - 1)
    assert arm.v_c[0] == pytest.approx(ref, rel=1e-9)


def test_zero_solve_runs_leave_caps_untouched():
    """No solve ever ran => the capacitors must not move. Both
    zero-solve paths: a sub-dt horizon, and a should_continue that
    cancels before the first step. (The first version 'finalized' a
    phantom step here and discharged 48 V to 13.7 V.)"""
    def fresh():
        b = p.CircuitBuilder()
        arm = p.add_mmc_thevenin_arm(
            b, "arm", "a", "gnd",
            n_sm=1, c_sm=2e-3, r_on=1e-3, v_c_init=48.0,
            n_on=lambda t: 1)
        b.add_resistor("Rload", "a", "gnd", 0.5)
        return b, arm

    dt = 1e-5
    b, arm = fresh()
    p.simulate(b, t_end=0.5 * dt, dt=dt, mmc_arms=[arm])
    assert arm.v_c[0] == 48.0

    b, arm = fresh()
    p.simulate(b, t_end=10 * dt, dt=dt, mmc_arms=[arm],
                should_continue=lambda: False)
    assert arm.v_c[0] == 48.0


def test_solve_one_gating_samples_step_end():
    """n_on is sampled at each step's END — including the FIRST
    solve (the engine's switch_fn convention). A schedule that
    turns on inside (t_start, dt] must gate solve #1 ON."""
    b = p.CircuitBuilder()
    dt = 1e-5
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=1, c_sm=2e-3, r_on=1e-3, v_c_init=48.0,
        n_on=lambda t: 1 if t >= 0.999 * dt else 0)
    b.add_resistor("Rload", "a", "gnd", 0.5)
    res = p.simulate(b, t_end=2 * dt, dt=dt, mmc_arms=[arm])
    r_c = dt / (2 * 2e-3)
    # Inserted on solve 1: va(dt) = V_eq*Rl/(Rl+R_eq), not ~0.
    assert res.v("a")[1] == pytest.approx(
        48.0 * 0.5 / (0.5 + 1e-3 + r_c), rel=1e-9)


def test_unsupported_combinations_refuse():
    """Each of these silently corrupted the capacitor bookkeeping
    before it was refused (all four demonstrated by adversarial
    review): decimated finals, DC op without V_eq, sub-dt re-solves
    against a full-dt companion, and continuation with a silent
    capacitor reset."""
    def fresh():
        b = p.CircuitBuilder()
        arm = p.add_mmc_thevenin_arm(
            b, "arm", "a", "gnd",
            n_sm=2, c_sm=1e-3, r_on=1e-3, v_c_init=10.0,
            n_on=lambda t: 1)
        b.add_resistor("Rload", "a", "gnd", 1.0)
        return b, arm

    b, arm = fresh()
    with pytest.raises(ValueError, match="store_every"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm],
                    store_every=2)
    b, arm = fresh()
    with pytest.raises(ValueError, match="start_from_dc_op"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm],
                    start_from_dc_op=True)
    b, arm = fresh()
    with pytest.raises(ValueError, match="max_dt_halvings"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm],
                    max_dt_halvings=3)
    b, arm = fresh()
    with pytest.raises(ValueError, match="substep"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm],
                    enable_substep_state_correction=True)
    b, arm = fresh()
    with pytest.raises(ValueError, match="v_c_preset"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm],
                    initial_state=np.zeros(1))
    # Continuation WITH the documented recipe is accepted.
    b = p.CircuitBuilder()
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=1, c_sm=2e-3, r_on=1e-3, v_c_init=0.0,
        v_c_preset=[47.0], n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 0.5)
    state_size = b.pool.state_size(b.graph)
    res = p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm],
                      initial_state=np.zeros(state_size))
    assert np.isfinite(res.v("a")).all()


def test_handle_identity_refusals():
    """A duplicate handle doubles the Norton injection; a handle
    from another builder mutates whatever device holds that branch
    id. Both refuse by identity, not by accident."""
    b = p.CircuitBuilder()
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=1, c_sm=1e-3, r_on=1e-3, v_c_init=10.0,
        n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 1.0)
    with pytest.raises(ValueError, match="more than once"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm, arm])

    b2 = p.CircuitBuilder()
    b2.add_resistor("R1", "a", "gnd", 1.0)
    b2.add_resistor("R2", "a", "gnd", 1.0)
    with pytest.raises(ValueError, match="DIFFERENT builder"):
        p.simulate(b2, t_end=1e-4, dt=1e-6, mmc_arms=[arm])


def test_n_on_out_of_range_raises():
    for bad in (-1, 3):
        b = p.CircuitBuilder()
        arm = p.add_mmc_thevenin_arm(
            b, "arm", "a", "gnd",
            n_sm=2, c_sm=1e-3, r_on=1e-3, v_c_init=10.0,
            n_on=lambda t, bad=bad: bad)
        b.add_resistor("Rload", "a", "gnd", 1.0)
        with pytest.raises(ValueError, match="outside"):
            p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm])


def test_user_b_extra_wrong_length_raises():
    b = p.CircuitBuilder()
    arm = p.add_mmc_thevenin_arm(
        b, "arm", "a", "gnd",
        n_sm=1, c_sm=1e-3, r_on=1e-3, v_c_init=10.0,
        n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 1.0)
    # A second node makes state_size 2, so the length-1 return is a
    # genuine mismatch (it would broadcast into every row).
    b.add_resistor("Rx", "x", "gnd", 2.0)
    with pytest.raises(ValueError, match="shape"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm],
                    b_extra_fn=lambda t: [0.0])
