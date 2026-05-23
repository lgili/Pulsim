"""LLM-augmented topology classifier — Tier 2 of the schematic V2 stack.

Invoked when the deterministic recognizer returns ``confidence < 0.7``
OR returns ``None``. Sends a textual fingerprint of the circuit to the
Anthropic API (default ``claude-haiku-4-5``) and parses a JSON
response naming one of the topologies in ``LLM_KNOWN_TOPOLOGIES``.

Always-on by default. The user opts out with the env var
``PULSIM_LLM_LAYOUT_HINTS=0``, and the classifier is **silently
disabled** (returns ``None`` without warnings) if:

  * ``anthropic`` is not installed;
  * ``ANTHROPIC_API_KEY`` is not in the environment;
  * the network call raises any exception;
  * the response cannot be parsed as JSON or names an unknown
    topology.

In every failure mode, the renderer falls through to the
force-directed layout. The classifier never crashes the render
pipeline.

Determinism: the same circuit must produce the same layout across
runs. The classifier hits the API at most once per distinct
fingerprint and caches the result in
``~/.cache/pulsim/topology-cache.json`` (override via
``PULSIM_TOPOLOGY_CACHE_DIR``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from .topology_recognizer import (
    KNOWN_TOPOLOGIES,
    RecognizedTopology,
    _CircuitView,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Superset of the deterministic recognizer's catalog. The LLM tier
#: can identify topologies the Tier-1 detectors do not cover.
LLM_KNOWN_TOPOLOGIES: frozenset[str] = KNOWN_TOPOLOGIES | frozenset({
    # Analog amplifiers + small-signal blocks Tier 1 doesn't cover.
    "common_source",
    "common_emitter",
    "op_amp_inverting",
    "op_amp_non_inverting",
    "voltage_divider",
})

#: Schema version stamped into every cache file. A mismatch causes the
#: file to be discarded and rebuilt (no automatic migration).
CACHE_SCHEMA_VERSION: str = "topology-cache-v1"

#: Default model used when ``PULSIM_LLM_MODEL`` is not set. Haiku-4-5
#: is plenty for the closed-vocabulary classification task at roughly
#: 1/8th the cost of Sonnet.
DEFAULT_MODEL: str = "claude-haiku-4-5"

#: Confidence below which the dispatcher prefers the force-directed
#: fallback over the LLM's verdict.
LLM_MIN_CONFIDENCE: float = 0.5

#: Maximum tokens the LLM may emit. The response is a small JSON
#: object — 256 is plenty even with a generous role map.
_MAX_OUTPUT_TOKENS: int = 512


_SYSTEM_PROMPT: str = (
    "You are a power-electronics topology classifier. Given the "
    "textual fingerprint of an electrical circuit, identify which "
    "of the following canonical topologies it most closely matches:\n\n"
    "    {topologies}\n\n"
    "Respond with ONE compact JSON object — no markdown, no prose, "
    "no code fences. The object MUST have these exact keys:\n\n"
    "    topology:    one of the names listed above, lowercase, "
    "underscored;\n"
    "    confidence:  float in [0.0, 1.0] — how certain you are;\n"
    "    role_map:    {{component_name: role_name}} mapping each "
    "component\n"
    "                 in the input to its functional role "
    "(e.g. \"switch_high\",\n"
    "                 \"freewheel_diode\", \"inductor_main\", "
    "\"output_capacitor\",\n"
    "                 \"load_resistor\"). Use empty {{}} if you "
    "cannot infer roles.\n\n"
    "If the circuit does NOT match any of the listed topologies, "
    "set topology to the closest match with a low confidence "
    "(< 0.5). Never emit a topology name that is not in the list."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(circuit: Any) -> Optional[RecognizedTopology]:
    """Run the LLM-augmented classifier against ``circuit``.

    Returns ``None`` (without raising) on any of: env-var opt-out,
    missing dependency, missing API key, network error, parse error,
    or unknown topology name. The renderer treats every ``None`` as
    "fall back to force-directed."
    """
    # Opt-out: the env var is the cheapest filter, check it first.
    if os.environ.get("PULSIM_LLM_LAYOUT_HINTS", "1").strip() == "0":
        return None

    # Fingerprint the circuit deterministically. Identical circuits
    # (modulo insertion order or component renames) hash to the same
    # cache key.
    try:
        fingerprint = circuit_fingerprint(circuit)
    except Exception as exc:
        logger.debug("circuit_fingerprint failed: %s", exc)
        return None
    key = _cache_key(fingerprint)

    # Cache lookup. A hit returns immediately without touching the
    # network.
    cached = _read_cache_entry(key)
    if cached is not None:
        return RecognizedTopology(
            name=cached["topology"],
            confidence=float(cached["confidence"]),
            role_map=dict(cached.get("role_map", {})),
            source="cache",
        )

    # Cache miss — invoke the LLM. Any failure returns None without
    # raising.
    response = _call_anthropic(fingerprint)
    if response is None:
        return None

    topology = response.get("topology")
    if not isinstance(topology, str):
        return None
    if topology not in LLM_KNOWN_TOPOLOGIES:
        logger.debug("LLM returned unknown topology %r", topology)
        return None
    confidence = float(response.get("confidence", 0.0))
    if confidence < LLM_MIN_CONFIDENCE:
        return None
    role_map_raw = response.get("role_map", {}) or {}
    if not isinstance(role_map_raw, dict):
        role_map_raw = {}
    role_map = {str(k): str(v) for k, v in role_map_raw.items()}

    # Persist BEFORE returning so a crash mid-render still warms the
    # cache for next time.
    _write_cache_entry(
        key=key,
        entry={
            "topology":   topology,
            "confidence": confidence,
            "role_map":   role_map,
            "model":      _model_name(),
            "cached_at":  _now_iso(),
        },
    )

    return RecognizedTopology(
        name=topology,
        confidence=confidence,
        role_map=role_map,
        source="llm",
    )


def circuit_fingerprint(circuit: Any) -> str:
    """Deterministic, whitespace-normalized textual representation of a
    circuit.

    The fingerprint is stable across:
      * insertion order of components (rows are sorted by name);
      * node renaming (node names are kept as-is — renaming a node
        DOES change the fingerprint, which is the desired behavior);
      * the ground sentinel value (always rendered as ``<gnd>``).

    Two circuits with the same fingerprint must produce the same
    layout. Two circuits with different fingerprints are allowed to
    differ.
    """
    view = _CircuitView(circuit)

    lines: list[str] = ["COMPONENTS:"]
    sorted_comps = sorted(view.components, key=lambda c: c.name)
    for comp in sorted_comps:
        # Map ground sentinel to a stable label; otherwise emit the
        # numeric id (node names aren't available without a Graph
        # round-trip, which we don't depend on here).
        node_labels = [
            "<gnd>" if int(n) == view.ground else f"n{int(n)}"
            for n in comp.nodes
        ]
        lines.append(
            f"  {comp.name}: {comp.kind}, terminals=[{', '.join(node_labels)}]"
        )

    # Sorted node id set.
    all_nodes: set[int] = set()
    for comp in sorted_comps:
        for n in comp.nodes:
            all_nodes.add(int(n))
    node_labels_sorted = sorted(
        ("<gnd>" if n == view.ground else f"n{n}") for n in all_nodes
    )
    lines.append("")
    lines.append(f"NODES: {', '.join(node_labels_sorted)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal — cache
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    env = os.environ.get("PULSIM_TOPOLOGY_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path("~/.cache/pulsim").expanduser().resolve()


def _cache_file() -> Path:
    return _cache_dir() / "topology-cache.json"


def _cache_key(fingerprint: str) -> str:
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, Any]:
    path = _cache_file()
    if not path.exists():
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("topology cache unreadable (%s), rebuilding", exc)
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    if not isinstance(data, dict):
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        logger.debug("topology cache schema mismatch (%r != %r), rebuilding",
                       data.get("schema_version"), CACHE_SCHEMA_VERSION)
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    if not isinstance(data.get("entries"), dict):
        data["entries"] = {}
    return data


def _save_cache(data: dict[str, Any]) -> None:
    """Atomic write — first to a temp file in the same directory, then
    ``os.replace`` over the destination. Safe under concurrent pytest
    runs as long as ``PULSIM_TOPOLOGY_CACHE_DIR`` is per-process."""
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file()
    fd, tmp = tempfile.mkstemp(
        prefix="topology-cache-", suffix=".json.tmp", dir=str(cache_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_cache_entry(key: str) -> Optional[dict[str, Any]]:
    cache = _load_cache()
    entries = cache.get("entries", {})
    entry = entries.get(key)
    if not isinstance(entry, dict):
        return None
    topology = entry.get("topology")
    if not isinstance(topology, str) or topology not in LLM_KNOWN_TOPOLOGIES:
        return None
    return entry


def _write_cache_entry(*, key: str, entry: dict[str, Any]) -> None:
    cache = _load_cache()
    cache.setdefault("entries", {})[key] = entry
    try:
        _save_cache(cache)
    except OSError as exc:
        # Cache write is best-effort. A read-only filesystem must not
        # crash a render.
        logger.debug("topology cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Internal — LLM call
# ---------------------------------------------------------------------------

def _model_name() -> str:
    return os.environ.get("PULSIM_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_user_prompt(fingerprint: str) -> str:
    return (
        "Classify the following circuit and respond with a single "
        "JSON object as instructed in the system prompt.\n\n"
        "CIRCUIT FINGERPRINT:\n"
        f"{fingerprint}\n"
    )


def _build_system_prompt() -> str:
    topo_list = "\n    ".join(sorted(LLM_KNOWN_TOPOLOGIES))
    return _SYSTEM_PROMPT.format(topologies=topo_list)


def _call_anthropic(fingerprint: str) -> Optional[dict[str, Any]]:
    """Invoke the Anthropic API. Returns the parsed JSON response or
    ``None`` on any failure mode (missing dep, missing key, network
    error, parse error)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("anthropic package not installed; LLM tier disabled")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=_model_name(),
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=_build_system_prompt(),
            messages=[
                {"role": "user", "content": _build_user_prompt(fingerprint)},
            ],
        )
    except Exception as exc:
        logger.debug("anthropic API call failed: %s", exc)
        return None

    # Concatenate every text block in the response (the SDK returns a
    # list of content blocks; for our prompt there is typically exactly
    # one TextBlock).
    try:
        text_parts: list[str] = []
        for block in response.content:
            block_text = getattr(block, "text", None)
            if block_text:
                text_parts.append(block_text)
        raw = "".join(text_parts).strip()
    except Exception as exc:
        logger.debug("anthropic response shape unexpected: %s", exc)
        return None

    if not raw:
        return None

    # Be tolerant of accidental code fences — strip them before
    # parsing.
    if raw.startswith("```"):
        # ``` or ```json …
        first_newline = raw.find("\n")
        raw = raw[first_newline + 1:] if first_newline != -1 else raw[3:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("anthropic response was not valid JSON: %r", raw[:200])
        return None

    if not isinstance(data, dict):
        return None
    return data


# Re-export for easier mocking in tests.
__all__ = [
    "classify",
    "circuit_fingerprint",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_MODEL",
    "LLM_KNOWN_TOPOLOGIES",
    "LLM_MIN_CONFIDENCE",
]
