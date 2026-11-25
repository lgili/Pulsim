"""`pulsim.Simulation`: the kernel's `sim::Simulation` with the result
handles `simulate()` attaches (add-kernel-simulation-object)."""

from __future__ import annotations

from . import _pulsim  # type: ignore[import-not-found]


class Simulation(_pulsim.Simulation):
    """``Simulation(circuit, options)``: one simulation, planned by the kernel.

    The constructor plans the run: the preflight on a private copy of the
    circuit (``circuit`` is the one that runs, ``plan['ties']`` what was
    added), routing (``plan['engine']``, ``plan['route_reasons']``), the
    option x engine refusals (``pulsim.option_capabilities()``) and the
    device capability check, all before anything is solved. Attach a gate
    source (``set_switch_fn``), ticks (``add_tick(period, fn)``), a source
    term, a cancellation hook; then ``run()``, or ``run_until(t)`` and
    ``snapshot()`` / ``restore()`` to continue (exact on ``'pwl'``). Runs
    ``'pwl'`` and ``'trbdf2'``; the exact engine and C blocks are composed
    by :func:`pulsim.simulate`. ``diagnostics`` lists the run's warnings.
    """

    _builder = None

    def run(self):
        return self._decorate(super().run())

    def run_until(self, t):
        return self._decorate(super().run_until(t))

    def _decorate(self, res):
        if self._builder is None:
            self._builder = self.circuit.to_builder()
        plan = self.plan
        res._builder = self._builder
        res.engine_used = plan["engine"]
        res.engine_route_reason = "; ".join(plan["route_reasons"]) or None
        from . import _set_run_info  # res.grid / res.stats, as simulate() sets them
        eng = plan["engine"]
        _set_run_info(res, eng, self.trbdf2_stats if eng == "trbdf2" else None)
        return res
