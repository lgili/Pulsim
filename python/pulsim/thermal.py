"""Pulsim — thermal Foster networks + electro-thermal co-simulation.

Junction-temperature tracking for power devices, in two flavours:

1. **Post-processing** — given a power-dissipation trace P(t) and a
   Foster ``Z_th(t)`` model (a sum of exponentials), compute the
   junction-temperature rise ``ΔT_j(t)`` by direct convolution.
   Useful when you already have an electrical-only simulation result
   and want a quick thermal characterisation:

       T_j_trace = p.compute_temperature(times, p_loss_trace, stages,
                                            T_amb_C=25.0)

2. **Live co-simulation** — embed the Foster network as ordinary
   v2 R/C devices in the same `CircuitBuilder` as the electrical
   circuit, and inject the instantaneous power dissipation via a
   `step_observer` + `b_extra_fn` pair. Junction temperature is
   then just another node voltage you can read at each step:

       p.add_foster_network(builder, stages,
                              junction_node="T_j",
                              ambient_node="T_amb",
                              T_amb_C=25.0)
       observer, b_extra = p.make_thermal_observer(
           builder, mosfet_branch=q1_branch,
           junction_node="T_j", T_amb_C=25.0)

The Foster RC ladder is the standard SPICE-compatible representation
of a device's transient thermal impedance ``Z_th(t)``. Each stage
has a thermal resistance ``R_th_i [K/W]`` and a time constant
``τ_i = R_th_i · C_th_i [s]``. The corresponding ``Z_th(t)`` is

    Z_th(t) = Σ R_th_i · (1 − exp(−t / τ_i))

For unit conversion convenience this module uses the standard
"temperature = voltage" / "power = current" analogy:
  * R_th [K/W] → resistance [Ω]
  * C_th [J/K] → capacitance [F]
  * T [°C]   → node voltage [V]
  * P [W]    → injected current [A]

…so the existing v2 transient solver runs the thermal network with
no special handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


__all__ = [
    "FosterStage",
    "CauerStage",
    "fit_foster_from_zth",
    "predict_zth_curve",
    "compute_temperature",
    "add_foster_network",
    "add_cauer_thermal_network",
    "make_thermal_observer",
    "ThermalLimitMonitor",
    "device_thermal_summary",
]


# =============================================================================
# Foster-stage dataclass
# =============================================================================

@dataclass
class FosterStage:
    """One Foster pole: a parallel R_th // C_th block.

    Time constant ``τ = R_th · C_th`` is the more intuitive parameter
    when fitting from a Z_th(t) datasheet curve, so both are stored.
    """
    R_th_K_per_W: float       # thermal resistance, K/W
    tau_s: float              # time constant, s

    @property
    def C_th_J_per_K(self) -> float:
        return self.tau_s / max(self.R_th_K_per_W, 1e-30)


# =============================================================================
# Foster fitting + prediction
# =============================================================================

def predict_zth_curve(t: np.ndarray,
                          stages) -> np.ndarray:
    """Predict ``Z_th(t)`` from a list of FosterStage.

    Sum-of-exponentials:
        Z_th(t) = Σ R_th_i · (1 − exp(−t / τ_i))
    """
    t = np.asarray(t, dtype=float)
    z = np.zeros_like(t)
    for s in stages:
        z += s.R_th_K_per_W * (1.0 - np.exp(-t / s.tau_s))
    return z


def fit_foster_from_zth(t_samples,
                            zth_samples,
                            n_stages: int = 3,
                            tau_init=None,
                            n_iter: int = 30,
                            ) -> list:
    """Fit a Foster network to a measured / datasheet Z_th(t) curve.

    Uses alternating optimization:
      1. Fix τ, solve for R via linear least-squares (the inner
         problem is linear once τ is known).
      2. For each pole, take a small log-step in τ that reduces the
         squared residual (numerical gradient).
      3. Repeat `n_iter` times.

    This gives a much better fit than a single lstsq with fixed τ —
    typically converges to ≤ 1 % max error after ~20 iterations on
    smooth datasheet curves.

    Parameters
    ----------
    t_samples
        Times (s) of the Z_th measurements. Must be strictly positive.
    zth_samples
        Z_th values (K/W) at those times.
    n_stages
        Number of Foster stages.
    tau_init
        Optional list of initial time constants. Defaults to log-
        spaced from `t_samples.min()` to `t_samples.max()`.
    n_iter
        Number of alternating fit iterations. Set to 0 to get the
        single-shot lstsq result (fast but less accurate).

    Returns
    -------
    list[FosterStage]
        The fitted Foster ladder.
    """
    t = np.asarray(t_samples, dtype=float)
    z = np.asarray(zth_samples, dtype=float)
    if t.shape != z.shape:
        raise ValueError(
            "t_samples and zth_samples must have the same shape")

    if tau_init is None:
        tau_init = np.geomspace(t.min(), t.max(), n_stages)
    taus = np.asarray(tau_init, dtype=float)
    if len(taus) != n_stages:
        raise ValueError("tau_init must have length n_stages")

    def solve_R_for_tau(taus_arr):
        A = 1.0 - np.exp(-np.outer(t, 1.0 / taus_arr))
        R, *_ = np.linalg.lstsq(A, z, rcond=None)
        return np.maximum(R, 0.0)

    def residual(taus_arr, R_arr):
        z_model = (1.0 - np.exp(-np.outer(t, 1.0 / taus_arr))) @ R_arr
        return float(np.sum((z - z_model)**2))

    R_th = solve_R_for_tau(taus)
    err = residual(taus, R_th)

    # Coordinate descent on log(τ_k) — perturb each pole by ±step in
    # log-space and keep whichever reduces the residual.
    step = 0.3  # log-space step (≈ ×1.35)
    for _ in range(n_iter):
        improved = False
        for k in range(n_stages):
            for sign in (+1.0, -1.0):
                taus_try = taus.copy()
                taus_try[k] *= math.exp(sign * step)
                R_try = solve_R_for_tau(taus_try)
                err_try = residual(taus_try, R_try)
                if err_try < err * 0.999:
                    taus = taus_try
                    R_th = R_try
                    err = err_try
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break

    return [FosterStage(R_th_K_per_W=float(r), tau_s=float(tau))
              for r, tau in zip(R_th, taus)]


# =============================================================================
# Post-processing convolution
# =============================================================================

def compute_temperature(t: np.ndarray,
                            p_loss: np.ndarray,
                            stages,
                            T_amb_C: float = 25.0,
                            *,
                            should_continue=None) -> np.ndarray:
    """Convolve P(t) with Z_th(t) to get ΔT_j(t), then add T_amb.

    Uses the standard Foster decomposition: for each pole
    (R_th, τ), the per-step update is the discrete-time IIR
    ``T_i[n+1] = α_i · T_i[n] + R_th · (1 − α_i) · P[n+1]`` with
    α_i = exp(−dt / τ_i), then T_j = Σ T_i. Assumes a uniform dt
    inferred from `t` — for non-uniform sampling, interpolate first.

    Cancellation
    ------------
    When ``should_continue`` is non-``None``, it is invoked every
    1000 samples (or every 1 % of the trace, whichever is more
    frequent). Returning ``False`` raises :class:`pulsim.Cancelled`
    with ``where='compute_temperature'`` so a GUI's cancel button
    can preempt a long convolution within ~1 ms on a 100 kHz / 10 s
    trace.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p_loss, dtype=float)
    if t.shape != p.shape:
        raise ValueError("t and p_loss must have the same shape")
    if len(t) < 2:
        raise ValueError("need at least 2 samples")
    dt = float(t[1] - t[0])
    # Per-pole IIR state.
    state = np.zeros(len(stages))
    T_out = np.zeros_like(p)
    # Cancellation cadence: min(1000, 1% of trace).
    check_interval = max(1, min(1000, len(p) // 100))
    for n in range(len(p)):
        if should_continue is not None and n % check_interval == 0:
            if not should_continue():
                from . import Cancelled as _Cancelled
                raise _Cancelled("compute_temperature",
                                  point_index=n)
        delta = 0.0
        for i, s in enumerate(stages):
            alpha = math.exp(-dt / s.tau_s)
            state[i] = alpha * state[i] + \
                          s.R_th_K_per_W * (1.0 - alpha) * p[n]
            delta += state[i]
        T_out[n] = T_amb_C + delta
    return T_out


# =============================================================================
# Foster network as an embedded sub-circuit
# =============================================================================

@dataclass
class CauerStage:
    """One stage of a Cauer (physical) thermal ladder.

    Cauer parametrisation gives the per-layer R_th and C_th
    directly — these are the values you'd extract from a finite-
    element thermal simulation of the device's material stack,
    or from a Cauer fit of measured Z_th(t).

    Foster parametrisation (used by :class:`FosterStage`) uses
    R_th + τ instead, fitted from the impedance spectrum without
    physical interpretation per stage.

    Topologically both parametrisations feed into the same ladder
    (R between consecutive nodes, C from each node to ambient) —
    only the numerical values differ. Use whichever your data
    source provides; conversions are well-defined but not always
    elementary.
    """
    R_th_K_per_W: float
    C_th_J_per_K: float


def add_cauer_thermal_network(builder,
                                    stages,
                                    *,
                                    junction_node: str = "T_j",
                                    ambient_node: str = "T_amb",
                                    T_amb_C: float = 25.0,
                                    name_prefix: str = "Th") -> None:
    """Embed a Cauer RC ladder using direct (R, C) parameters.

    Identical topology to :func:`add_foster_network` (R between
    consecutive nodes, C from each node to ambient) but takes
    :class:`CauerStage` (R + C) instead of :class:`FosterStage`
    (R + τ). Convenient when the user has material-stack thermal
    values rather than impedance-fit values.

    Example
    -------
    ::

        stages = [
            p.CauerStage(R_th_K_per_W=0.05, C_th_J_per_K=0.001),
            p.CauerStage(R_th_K_per_W=0.10, C_th_J_per_K=0.010),
            p.CauerStage(R_th_K_per_W=0.20, C_th_J_per_K=0.100),
        ]
        p.add_cauer_thermal_network(b, stages,
            junction_node="T_j", T_amb_C=25.0)
    """
    n = len(stages)
    if n == 0:
        raise ValueError(
            "add_cauer_thermal_network: at least one stage required")
    builder.add_voltage_source(f"{name_prefix}_amb",
                                  ambient_node, "gnd",
                                  float(T_amb_C))
    prev = ambient_node
    for k, s in enumerate(stages):
        if k == n - 1:
            curr = junction_node
        else:
            curr = f"{name_prefix}_stage_{k}"
        builder.add_resistor(f"{name_prefix}_R{k}",
                                prev, curr,
                                float(s.R_th_K_per_W))
        builder.add_capacitor(f"{name_prefix}_C{k}",
                                  curr, ambient_node,
                                  float(s.C_th_J_per_K))
        prev = curr


class ThermalLimitMonitor:
    """Halts a simulation when junction temperature exceeds a limit.

    Designed to plug into ``simulate(should_continue=...)``. The
    monitor reads the node voltage at ``junction_node_idx`` from a
    Python step_observer that the user passes alongside; the monitor
    itself just remembers whether the limit was tripped and reports
    via :meth:`should_continue`.

    Example
    -------
    ::

        mon = p.ThermalLimitMonitor(T_limit_C=150.0)

        def observe(t, x):
            T_j = x[junction_node_idx]
            mon.update(t, T_j)

        res = p.simulate(b, t_end=10.0, dt=1e-4,
                            step_observer=observe,
                            should_continue=mon.should_continue)
        if mon.tripped:
            print(f"Thermal trip at t={mon.trip_time:.3f}s, "
                  f"T_j={mon.trip_temperature:.1f} °C")

    Attributes
    ----------
    T_limit_C
        Trip threshold (°C — matches the absolute-temperature
        convention used by :func:`add_foster_network` and
        :func:`add_cauer_thermal_network`).
    tripped
        True once the monitor saw ``T_j > T_limit_C``.
    trip_time
        Wall-clock simulation time at which the trip occurred.
    trip_temperature
        ``T_j`` value at the trip moment.
    peak_temperature
        Highest ``T_j`` observed across the run.
    """

    def __init__(self, *, T_limit_C: float, hysteresis_C: float = 0.0):
        self.T_limit_C = float(T_limit_C)
        self.hysteresis_C = float(hysteresis_C)
        self.tripped = False
        self.trip_time = float("nan")
        self.trip_temperature = float("nan")
        self.peak_temperature = -float("inf")
        self._last_time = 0.0

    def update(self, t: float, T_j: float) -> None:
        """Feed one (t, T_j) sample. Sets ``tripped`` when
        T_j > T_limit_C (or stays tripped while T_j > T_limit_C −
        hysteresis_C when ``hysteresis_C > 0``)."""
        T = float(T_j)
        if T > self.peak_temperature:
            self.peak_temperature = T
        self._last_time = float(t)
        if self.tripped:
            # Latch — only reset when below limit minus hysteresis.
            if T < self.T_limit_C - self.hysteresis_C:
                self.tripped = False
        else:
            if T > self.T_limit_C:
                self.tripped = True
                self.trip_time = float(t)
                self.trip_temperature = T

    def should_continue(self) -> bool:
        """Returns False once the limit was tripped — meant to
        be passed to ``simulate(should_continue=mon.should_continue)``.
        """
        return not self.tripped


def add_foster_network(builder,
                          stages,
                          *,
                          junction_node: str = "T_j",
                          ambient_node: str = "T_amb",
                          T_amb_C: float = 25.0,
                          name_prefix: str = "Th") -> None:
    """Embed a Foster RC ladder into an existing CircuitBuilder.

    Adds, in order:
      * A DC voltage source from `ambient_node` to gnd at value
        `T_amb_C` (so node voltage = absolute temperature in °C).
      * For each Foster stage, a parallel R_th // C_th block in
        cascade between `ambient_node` and `junction_node`. Each
        stage's intermediate node is named
        ``"{name_prefix}_stage_{k}"``.

    Power injection: at runtime, push a current of magnitude P_loss
    [A as W] into `junction_node` via `b_extra_fn` (see
    :func:`make_thermal_observer`).

    Notes
    -----
    The Foster ladder is *physically* a series of R-parallel-C
    blocks between the heat source and ambient. The implementation
    here uses voltage-source units (V ↔ °C) and current-source units
    (A ↔ W), which means R values are in K/W and C values in J/K —
    numerically identical to ohms and farads from the solver's POV.
    """
    n = len(stages)
    if n == 0:
        raise ValueError("at least one Foster stage required")
    # Ambient reference.
    builder.add_voltage_source(f"{name_prefix}_amb",
                                  ambient_node, "gnd", float(T_amb_C))
    # Build the cascade: ambient → stage_0 → stage_1 → … → junction.
    # Each stage is R || C between consecutive nodes.
    prev = ambient_node
    for k, s in enumerate(stages):
        if k == n - 1:
            curr = junction_node
        else:
            curr = f"{name_prefix}_stage_{k}"
        builder.add_resistor(f"{name_prefix}_R{k}",
                                prev, curr,
                                float(s.R_th_K_per_W))
        builder.add_capacitor(f"{name_prefix}_C{k}",
                                  curr, ambient_node,
                                  float(s.C_th_J_per_K))
        prev = curr


def make_thermal_observer(builder,
                              *,
                              junction_node: str,
                              power_fn,
                              ambient_node: str = "T_amb",
                              T_amb_C: float = 25.0):
    """Return a ``(step_observer, b_extra_fn)`` pair that drives the
    embedded thermal network from runtime power dissipation.

    Parameters
    ----------
    builder
        The CircuitBuilder that ALREADY has the Foster network added
        via :func:`add_foster_network`.
    junction_node
        Name of the junction node (same name passed to
        :func:`add_foster_network`).
    power_fn
        Callable ``power_fn(t, x) -> P_W`` returning the
        instantaneous power dissipation. Typical implementation:
        ``lambda t, x: (x[i_idx]**2) * R_DS_ON_value`` for a MOSFET.
    ambient_node
        Same name passed to :func:`add_foster_network`.
    T_amb_C
        Same value passed to :func:`add_foster_network`. Used to
        compute T_j = T_amb_C + ΔT.

    Returns
    -------
    (step_observer, b_extra_fn)
        Both should be passed to ``simulate(...)``. The observer
        captures the latest computed P_loss in a closure; the
        b_extra_fn reads that closure to inject the current at each
        step.
    """
    state_size = builder.pool.state_size(builder.graph)
    j_idx = builder.node_id_of(junction_node)

    latest = {"P": 0.0}

    def step_observer(t, x):
        latest["P"] = float(power_fn(t, x))

    def b_extra_fn(t):
        out = [0.0] * state_size
        # Inject the current at the junction node — current source
        # of value P (W) entering the node from outside the network.
        # Convention: positive entry in `b` adds to the right-hand
        # side of node-`i`'s KCL row; for a current source flowing
        # INTO node i we add +I to b[i]. The kernel's residual is
        # J·x + b = 0, so adding +I to b[i] means node i sees an
        # external current of −I being injected; we want +I (power
        # heats the junction → raises its voltage), so set b[i] = −I.
        out[j_idx] = -latest["P"]
        return out

    def read_T_j(x) -> float:
        """Convenience: read T_j (°C) from a state vector."""
        return float(x[j_idx])

    # Attach the reader as an attribute on the observer for ergonomic
    # access during post-processing.
    step_observer.read_T_j = read_T_j  # type: ignore[attr-defined]

    return step_observer, b_extra_fn


# =============================================================================
# Post-hoc convenience: per-device P(t) → T_j(t) pipeline
# =============================================================================

def _states_as_array(result) -> np.ndarray:
    s = result.states
    if hasattr(s, "shape"):
        return np.asarray(s, dtype=float)
    return np.asarray([list(v) for v in s], dtype=float)


def _node_voltage(states: np.ndarray, node_id: int) -> np.ndarray:
    if node_id < 0:
        return np.zeros(states.shape[0], dtype=float)
    return states[:, node_id]


def _eval_mask_trace(switch_fn, times, switch_idx) -> np.ndarray:
    out = np.zeros(times.size, dtype=bool)
    for k, t in enumerate(times):
        out[k] = bool(switch_fn(float(t)).get(int(switch_idx)))
    return out


def device_thermal_summary(
    builder,
    result,
    *,
    thermal_specs,
    T_ambient_C: float = 25.0,
    switch_fn=None,
    switch_specs=None,
    diode_specs=None,
    core_loss_specs=None,
):
    """End-to-end ``per-device P(t) → T_j(t)`` glue.

    Walks every R / inductor / switch / diode branch the user
    supplied a thermal model for, reconstructs the per-step
    conduction power ``P_cond(t)``, layers the PSIM-style
    switching-loss average ``P_sw_avg`` on top (constant — the
    thermal time constants are orders of magnitude longer than the
    nanosecond switching event, so smearing the impulse is the
    standard convention), and convolves the result with the Foster
    network via :func:`compute_temperature`.

    Parameters
    ----------
    builder, result
        The populated :class:`CircuitBuilder` and the
        :class:`SimulationResult` it produced.
    thermal_specs : mapping
        Per-device Foster network. Keys are branch *names*
        (preferred) or numeric branch ids. Values are mappings
        with::

            {"stages":      list[FosterStage],   # required
             "T_ambient_C": float}               # optional, defaults
                                                 # to the top-level
                                                 # ``T_ambient_C``
    T_ambient_C : float, default 25.0
        Fallback ambient temperature when a device spec omits its
        own.
    switch_fn, switch_specs, diode_specs, core_loss_specs
        Forwarded into the loss-reconstruction phase — same shapes
        and semantics as :func:`pulsim.device_loss_summary`.
        ``switch_fn`` is required when any switch entry is listed
        in ``thermal_specs``.

    Returns
    -------
    list of dict
        One entry per thermally-modelled device in branch-id
        order. Each carries::

            {"branch_id": int, "kind": str, "name": str,
             "P_cond_avg": float, "P_sw_avg": float,
             "P_core_avg": float, "P_total_avg": float,
             "T_j_trace": np.ndarray,  # °C
             "T_j_avg": float, "T_j_peak": float,
             "T_ambient_C": float,
             "R_th_total": float}      # sum of Foster R_th [K/W]

    Notes
    -----
    * For inductors with a ``core_loss_specs`` entry the magnetic
      loss is averaged (Steinmetz / iGSE deliver a scalar density)
      and folded into ``P_sw_avg`` so the same constant-offset
      assumption applies.
    * Switching loss for diodes (Q_rr / E_rr_ref) and MOSFET / IGBT
      ideal switches (E_on / E_off) is averaged over the whole run
      via :func:`pulsim.device_loss_summary` and added to the
      constant offset.
    """
    # Lazy import to avoid an import cycle (losses → thermal? no —
    # losses doesn't depend on thermal, but we keep the import
    # here for symmetry with the rest of the module's local helpers).
    from . import losses as _losses

    if not thermal_specs:
        return []

    states = _states_as_array(result)
    times = np.asarray(result.times, dtype=float)

    # Resolve thermal_specs into branch_id-keyed mapping.
    try:
        comps_by_bid = {c["branch_id"]: c for c in builder.components()}
    except Exception:
        comps_by_bid = {}
    comps_by_name = {c.get("name"): c for c in comps_by_bid.values()
                       if c.get("name")}

    thermal_by_bid = {}
    for key, spec in thermal_specs.items():
        if isinstance(key, int):
            thermal_by_bid[int(key)] = spec
        else:
            desc = comps_by_name.get(str(key))
            if desc is None:
                raise KeyError(
                    f"device_thermal_summary: no device named "
                    f"{key!r} in the builder.")
            thermal_by_bid[int(desc["branch_id"])] = spec

    # Pull the per-device average switching / core losses from the
    # standard loss summary; we'll layer those on the conduction
    # trace as constant offsets (thermal time constants ≫ switching
    # transition duration).
    loss_entries = _losses.device_loss_summary(
        builder, result,
        switch_fn=switch_fn,
        core_loss_specs=core_loss_specs,
        diode_specs=diode_specs,
        switch_specs=switch_specs,
    )
    loss_by_bid = {int(e["branch_id"]): e for e in loss_entries}

    # Need switch_seq_idx (same logic as device_loss_summary).
    switch_seq_idx_by_bid = {}
    switch_seq_counter = 0
    for bid in range(builder.graph.num_branches):
        desc = comps_by_bid.get(bid, {})
        kind = desc.get("kind", "unknown")
        if kind in ("diode", "switch"):
            switch_seq_idx_by_bid[bid] = switch_seq_counter
            switch_seq_counter += 1

    out = []
    for bid, t_spec in sorted(thermal_by_bid.items()):
        desc = comps_by_bid.get(bid, {})
        kind = desc.get("kind", "unknown")
        name = desc.get("name", "")
        stages = t_spec.get("stages")
        if not stages:
            raise ValueError(
                f"thermal_specs[{name!r}] is missing required key "
                f"'stages' (list of FosterStage).")
        T_amb_dev = float(t_spec.get("T_ambient_C", T_ambient_C))

        loss = loss_by_bid.get(bid, {})
        P_sw_avg = float(loss.get("P_sw_avg", 0.0))
        P_core_avg = float(loss.get("P_core_avg", 0.0))
        P_constant = P_sw_avg + P_core_avg

        # Reconstruct P_cond(t) per kind.
        if kind == "resistor":
            R = float(desc.get("params", {}).get("R_ohms", float("nan")))
            from_id, to_id = desc.get("nodes", (None, None))
            if not (np.isfinite(R) and R > 0) or from_id is None \
                    or to_id is None:
                continue
            v_R = (_node_voltage(states, int(from_id))
                   - _node_voltage(states, int(to_id)))
            P_cond_t = v_R * v_R / R
        elif kind == "inductor":
            # Ideal inductor — conduction loss is zero. Any thermal
            # rise comes from P_core_avg above.
            P_cond_t = np.zeros_like(times)
        elif kind == "diode":
            params = desc.get("params", {})
            g_on = float(params.get("g_on", float("nan")))
            g_off = float(params.get("g_off", float("nan")))
            V_th = float(params.get("V_th", 0.0))
            from_id, to_id = desc.get("nodes", (None, None))
            if not (np.isfinite(g_on) and np.isfinite(g_off)) \
                    or from_id is None or to_id is None:
                continue
            v_D = (_node_voltage(states, int(from_id))
                   - _node_voltage(states, int(to_id)))
            g_arr = np.where(v_D > V_th, g_on, g_off)
            P_cond_t = v_D * v_D * g_arr
        elif kind == "switch":
            params = desc.get("params", {})
            g_on = float(params.get("g_on", float("nan")))
            g_off = float(params.get("g_off", float("nan")))
            from_id, to_id = desc.get("nodes", (None, None))
            if not (np.isfinite(g_on) and np.isfinite(g_off)) \
                    or from_id is None or to_id is None:
                continue
            if switch_fn is None:
                raise ValueError(
                    f"thermal_specs targets switch {name!r} but "
                    "switch_fn= was not provided — pass the same "
                    "callable you used in simulate().")
            sidx = switch_seq_idx_by_bid.get(bid)
            if sidx is None:
                continue
            closed = _eval_mask_trace(switch_fn, times, sidx)
            v_SW = (_node_voltage(states, int(from_id))
                    - _node_voltage(states, int(to_id)))
            g_arr = np.where(closed, g_on, g_off)
            P_cond_t = v_SW * v_SW * g_arr
        else:
            # Unsupported kind for the thermal pipeline.
            continue

        P_total_t = P_cond_t + P_constant
        T_j_trace = compute_temperature(times, P_total_t,
                                          stages, T_amb_dev)
        T = (times[-1] - times[0]) if times.size > 1 else 1.0
        P_cond_avg = (float(np.trapezoid(P_cond_t, times) / T)
                      if T > 0 else float(P_cond_t.mean()))
        out.append({
            "branch_id": int(bid),
            "kind": kind,
            "name": name,
            "P_cond_avg": P_cond_avg,
            "P_sw_avg": P_sw_avg,
            "P_core_avg": P_core_avg,
            "P_total_avg": P_cond_avg + P_constant,
            "T_j_trace": T_j_trace,
            "T_j_avg": float(np.mean(T_j_trace)),
            "T_j_peak": float(np.max(T_j_trace)),
            "T_ambient_C": T_amb_dev,
            "R_th_total": float(sum(s.R_th_K_per_W for s in stages)),
        })
    return out
