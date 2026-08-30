"""Subsystems — define once, instantiate many, with scoped names.

v2.0 Phase 3, audit A.7. Today a 100-submodule arm is 100
hand-prefixed groups of ``add_*`` calls: 302 branches whose every
name is an f-string the author had to keep unique by discipline
(and a duplicate is accepted SILENTLY — the second device becomes
unreachable by name forever). Nothing says "inside SM17"; the
SPICE importer has no target to flatten a ``.subckt`` into.

A subsystem is a named body of ``add_*`` calls with declared
PORTS and PARAMS. Instantiating it under a path emits the same
devices into the same builder with every name scoped::

    sm = p.define_subsystem("HalfBridgeSM",
                             ports=("top", "bot"),
                             params={"C": 2e-3, "r_on": 1e-3})

    @sm.body
    def _(s):
        s.add_switch("Sb", "top", "bot", 1 / s.p.r_on, 1e-9)
        s.add_switch("Si", "top", "m",   1 / s.p.r_on, 1e-9)
        s.add_capacitor("C", "m", "bot", s.p.C)

    prev = "dc_p"
    for i in range(100):
        node = f"n{i}"
        sm.instantiate(b, f"leg_a/sm{i}", top=prev, bot=node)
        prev = node

``top`` and ``bot`` bind to the caller's nets — no node is
created. ``m`` is INTERNAL, so it becomes ``leg_a/sm17/m``, and
the capacitor becomes ``leg_a/sm17/C``. Those strings are the
real names in the graph, so ``res.v("leg_a/sm17/m")``,
``res.i("leg_a/sm17/C")`` and every named diagnostic already speak
the path — nothing downstream had to learn about hierarchy.

**Flattened at instantiate time. The kernel is untouched**: this
module only rewrites the strings on their way into the ordinary
builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

#: Separator between path segments. A slash reads as a path, and
#: no existing name uses one (the hand-rolled macros use ``__``,
#: which stays valid — a subsystem does not take the character
#: away from anyone).
SEP = "/"

_GROUND_ALIASES = frozenset({"gnd", "GND", "0", "ground"})


class _Params:
    """Attribute view over the resolved parameter dict."""

    __slots__ = ("_d",)

    def __init__(self, d: Dict[str, Any]) -> None:
        object.__setattr__(self, "_d", d)

    def __getattr__(self, k: str) -> Any:
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(
                f"subsystem parameter {k!r} is not declared; "
                f"declared: {sorted(self._d)}") from None

    def __getitem__(self, k: str) -> Any:
        return self.__getattr__(k)

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return f"_Params({self._d!r})"


class ScopedBuilder:
    """A builder proxy that rewrites names into a subsystem scope.

    Every ``add_*`` call is forwarded to the real builder with:

    * the FIRST positional string — the device name — prefixed by
      the instance path;
    * every OTHER string argument treated as a NET: a declared
      port resolves to the net the caller bound it to, a ground
      alias passes through untouched, and anything else is an
      internal net and gets the same prefix.

    That rule is what the whole feature rests on, so it is worth
    being explicit about why it is safe: every circuit-level
    ``add_*`` on CircuitBuilder has the shape
    ``(name, node..., numeric values...)`` — no member takes a
    string OPTION. A future one that did would need to be listed
    in `_NON_NET_ARGS` below, and the failure mode is loud (it
    would try to create a net named after the option).
    """

    #: (method, kwarg) pairs whose string value is NOT a net.
    _NON_NET_ARGS: frozenset = frozenset()

    def __init__(self, builder, path: str,
                 port_map: Dict[str, str],
                 params: Dict[str, Any]) -> None:
        self._b = builder
        self._path = path
        self._ports = dict(port_map)
        self.p = _Params(params)
        #: Names this instance emitted, in order.
        self.emitted: "list[str]" = []

    # -- name rewriting -------------------------------------------------
    @property
    def path(self) -> str:
        return self._path

    def scoped(self, local_name: str) -> str:
        """The full path of a name local to this instance."""
        return f"{self._path}{SEP}{local_name}" if self._path \
            else local_name

    def net(self, local_name: str) -> str:
        """Resolve a net name the way the body's `add_*` calls do.

        Useful when a body needs to hand a net to something that
        is not an ``add_*`` (a closure, a probe, a nested
        instantiate).
        """
        if local_name in self._ports:
            return self._ports[local_name]
        if local_name in _GROUND_ALIASES:
            return local_name
        return self.scoped(local_name)

    def _rewrite(self, method: str, args: tuple, kwargs: dict):
        new_args = []
        seen_name = False
        for a in args:
            if isinstance(a, str):
                if not seen_name:
                    seen_name = True
                    new_args.append(self.scoped(a))
                else:
                    new_args.append(self.net(a))
            else:
                new_args.append(a)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str) and (method, k) not in \
                    self._NON_NET_ARGS:
                if k == "name" and not seen_name:
                    seen_name = True
                    new_kwargs[k] = self.scoped(v)
                else:
                    new_kwargs[k] = self.net(v)
            else:
                new_kwargs[k] = v
        return tuple(new_args), new_kwargs

    def __getattr__(self, method: str):
        target = getattr(self._b, method)
        if not method.startswith("add_"):
            # Read-only surface (graph, pool, node_id_of, …) is
            # passed straight through — those take names the
            # caller already scoped, or none at all.
            return target

        def _call(*args, **kwargs):
            a, kw = self._rewrite(method, args, kwargs)
            if a and isinstance(a[0], str):
                self.emitted.append(a[0])
            elif "name" in kw and isinstance(kw["name"], str):
                self.emitted.append(kw["name"])
            return target(*a, **kw)

        return _call


@dataclass
class SubsystemInstance:
    """What `instantiate` hands back."""

    path: str
    subsystem: "Subsystem"
    ports: Dict[str, str]
    params: Dict[str, Any]
    #: Full names of every device this instance emitted.
    devices: "list[str]" = field(default_factory=list)

    def name_of(self, local: str) -> str:
        """Full name of a device or net local to this instance."""
        return f"{self.path}{SEP}{local}" if self.path else local

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return (f"<{self.subsystem.name} at {self.path!r}: "
                f"{len(self.devices)} devices>")


class Subsystem:
    """A reusable body of `add_*` calls with ports and params."""

    def __init__(self, name: str, ports: Iterable[str],
                 params: Optional[Dict[str, Any]] = None) -> None:
        self.name = str(name)
        self.ports: Tuple[str, ...] = tuple(ports)
        if not self.ports:
            raise ValueError(
                f"define_subsystem({name!r}): needs at least one "
                "port — a subsystem with no ports cannot be "
                "connected to anything.")
        dupes = {p for p in self.ports
                 if list(self.ports).count(p) > 1}
        if dupes:
            raise ValueError(
                f"define_subsystem({name!r}): duplicate port(s) "
                f"{sorted(dupes)}")
        bad = sorted(p for p in self.ports if SEP in p)
        if bad:
            raise ValueError(
                f"define_subsystem({name!r}): port name(s) {bad} "
                f"contain {SEP!r}, which separates path segments.")
        self.defaults: Dict[str, Any] = dict(params or {})
        self._body: Optional[Callable[[ScopedBuilder], None]] = None
        #: Every instance made from this definition.
        self.instances: "list[SubsystemInstance]" = []

    def body(self, fn: Callable[[ScopedBuilder], None]):
        """Register the body. Usable as a decorator."""
        self._body = fn
        return fn

    def instantiate(self, builder, path: str, **bindings
                     ) -> SubsystemInstance:
        """Emit one instance of this subsystem under `path`.

        Port bindings and parameter overrides are both keyword
        arguments; ports are matched first, and anything left over
        must be a declared parameter.
        """
        if self._body is None:
            raise ValueError(
                f"subsystem {self.name!r} has no body — decorate "
                "one with @<subsystem>.body before instantiating.")
        path = str(path)
        if not path:
            raise ValueError(
                f"instantiate({self.name!r}): path must be "
                "non-empty — it is what makes the instance's names "
                "unique.")

        ports: Dict[str, str] = {}
        missing = []
        for prt in self.ports:
            if prt in bindings:
                ports[prt] = str(bindings.pop(prt))
            else:
                missing.append(prt)
        if missing:
            raise ValueError(
                f"instantiate({self.name!r} at {path!r}): port(s) "
                f"{missing} not connected. Declared ports: "
                f"{list(self.ports)}.")

        params = dict(self.defaults)
        unknown = sorted(k for k in bindings if k not in params)
        if unknown:
            raise ValueError(
                f"instantiate({self.name!r} at {path!r}): "
                f"{unknown} is/are neither a port nor a declared "
                f"parameter. Ports: {list(self.ports)}; params: "
                f"{sorted(params)}.")
        params.update(bindings)

        # A subsystem instantiated INSIDE another body composes
        # paths, so `leg_a` + `sm17` reads as `leg_a/sm17`.
        if isinstance(builder, ScopedBuilder):
            full_path = builder.scoped(path)
            real_builder = builder._b
            ports = {k: builder.net(v) for k, v in ports.items()}
        else:
            full_path = path
            real_builder = builder

        scoped = ScopedBuilder(real_builder, full_path, ports,
                                params)
        self._body(scoped)
        inst = SubsystemInstance(path=full_path, subsystem=self,
                                  ports=ports, params=params,
                                  devices=list(scoped.emitted))
        self.instances.append(inst)
        return inst


def define_subsystem(name: str, *, ports: Iterable[str],
                      params: Optional[Dict[str, Any]] = None
                      ) -> Subsystem:
    """Declare a reusable subsystem. See the module docstring."""
    return Subsystem(name, ports, params)


__all__ = [
    "Subsystem",
    "SubsystemInstance",
    "ScopedBuilder",
    "define_subsystem",
    "SEP",
]
