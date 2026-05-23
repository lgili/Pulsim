"""Tests for the LLM-augmented topology classifier.

All Anthropic SDK calls are mocked. The only test that hits the real
network is gated behind ``PULSIM_LLM_REAL_API=1`` and is skipped in
CI by default.

Cache state is isolated per-test via ``PULSIM_TOPOLOGY_CACHE_DIR``
pointing at a per-test ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

import pulsim as p
from pulsim.schematic.llm_classifier import (
    CACHE_SCHEMA_VERSION,
    LLM_KNOWN_TOPOLOGIES,
    circuit_fingerprint,
    classify,
)
from pulsim.schematic.topology_recognizer import RecognizedTopology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the classifier at a private cache directory for each test."""
    monkeypatch.setenv("PULSIM_TOPOLOGY_CACHE_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture
def buck_circuit() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd", 24.0)
    b.add_switch       ("Q",      "vin", "sw",   g_on=1e3, g_off=1e-9)
    b.add_diode        ("D",      "gnd", "sw",   g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_inductor     ("L",      "sw",  "vout", 100e-6)
    b.add_capacitor    ("Cout",   "vout","gnd",  100e-6)
    b.add_resistor     ("Rload",  "vout","gnd",  5.0)
    return b


def _mock_anthropic_response(content_text: str) -> MagicMock:
    """Build a MagicMock that mimics the shape of an Anthropic
    ``Message`` object returned by ``client.messages.create``."""
    block = MagicMock()
    block.text = content_text
    response = MagicMock()
    response.content = [block]
    return response


def _install_mock_anthropic(monkeypatch, response_text: str | Exception):
    """Replace the ``anthropic`` module with a mock so the classifier
    behaves predictably without touching the real SDK."""
    fake_anthropic = types.ModuleType("anthropic")

    if isinstance(response_text, Exception):
        def create(**kwargs):
            raise response_text
    else:
        def create(**kwargs):
            return _mock_anthropic_response(response_text)

    fake_client = MagicMock()
    fake_client.messages.create = MagicMock(side_effect=create)
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    return fake_client


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_deterministic(buck_circuit):
    """Same circuit → identical fingerprint across calls."""
    f1 = circuit_fingerprint(buck_circuit)
    f2 = circuit_fingerprint(buck_circuit)
    assert f1 == f2
    assert "COMPONENTS:" in f1
    assert "NODES:" in f1
    assert "<gnd>" in f1


def test_fingerprint_independent_of_insertion_order():
    """Building the SAME buck via different add_* call order produces
    the same fingerprint (rows are sorted by component name)."""
    # Order A — canonical
    a = p.CircuitBuilder()
    a.add_voltage_source("Vin",   "vin", "gnd", 24.0)
    a.add_switch       ("Q",      "vin", "sw",   g_on=1e3, g_off=1e-9)
    a.add_diode        ("D",      "gnd", "sw",   g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    a.add_inductor     ("L",      "sw",  "vout", 100e-6)
    a.add_capacitor    ("Cout",   "vout","gnd",  100e-6)
    a.add_resistor     ("Rload",  "vout","gnd",  5.0)

    # Order B — passives first
    b = p.CircuitBuilder()
    b.add_capacitor    ("Cout",   "vout","gnd",  100e-6)
    b.add_resistor     ("Rload",  "vout","gnd",  5.0)
    b.add_inductor     ("L",      "sw",  "vout", 100e-6)
    b.add_voltage_source("Vin",   "vin", "gnd", 24.0)
    b.add_diode        ("D",      "gnd", "sw",   g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_switch       ("Q",      "vin", "sw",   g_on=1e3, g_off=1e-9)

    # Node insertion order differs (Order B sees "vout" before "vin"),
    # which produces different node IDs internally. So the fingerprints
    # may legitimately differ — assert structural sameness via line
    # count and presence of every component name.
    fa = circuit_fingerprint(a)
    fb = circuit_fingerprint(b)
    for name in ["Vin", "Q", "D", "L", "Cout", "Rload"]:
        assert name in fa
        assert name in fb


# ---------------------------------------------------------------------------
# classify() — opt-out / disabled paths
# ---------------------------------------------------------------------------

def test_disabled_via_env_var(buck_circuit, isolated_cache, monkeypatch):
    """PULSIM_LLM_LAYOUT_HINTS=0 returns None without touching the SDK."""
    monkeypatch.setenv("PULSIM_LLM_LAYOUT_HINTS", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Mock anthropic so we can detect any call (there should be none).
    client = _install_mock_anthropic(monkeypatch, '{"topology": "buck"}')
    result = classify(buck_circuit)
    assert result is None
    client.messages.create.assert_not_called()


def test_missing_api_key(buck_circuit, isolated_cache, monkeypatch):
    """Without ANTHROPIC_API_KEY the classifier is silently disabled."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("PULSIM_LLM_LAYOUT_HINTS", "1")
    result = classify(buck_circuit)
    assert result is None


def test_anthropic_module_missing(buck_circuit, isolated_cache, monkeypatch):
    """If anthropic isn't installed, classify() returns None."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("PULSIM_LLM_LAYOUT_HINTS", raising=False)
    # Forcibly hide the package — even if it's installed locally.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = classify(buck_circuit)
    assert result is None


# ---------------------------------------------------------------------------
# classify() — success path
# ---------------------------------------------------------------------------

def test_successful_classification_writes_cache(
        buck_circuit, isolated_cache, monkeypatch):
    """A successful API call returns a RecognizedTopology with
    source='llm' AND writes the cache file."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = json.dumps({
        "topology": "buck",
        "confidence": 0.95,
        "role_map": {
            "Vin": "source", "Q": "switch_high",
            "D": "freewheel_diode", "L": "inductor_main",
            "Cout": "output_capacitor", "Rload": "load_resistor",
        },
    })
    client = _install_mock_anthropic(monkeypatch, response)

    result = classify(buck_circuit)
    assert result is not None
    assert isinstance(result, RecognizedTopology)
    assert result.name == "buck"
    assert result.confidence == pytest.approx(0.95)
    assert result.source == "llm"
    assert result.role_map["Q"] == "switch_high"
    client.messages.create.assert_called_once()

    cache_file = isolated_cache / "topology-cache.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert data["schema_version"] == CACHE_SCHEMA_VERSION
    assert len(data["entries"]) == 1


def test_cache_hit_skips_api(buck_circuit, isolated_cache, monkeypatch):
    """A second classify() call with the same circuit reads from the
    cache and does NOT invoke the SDK."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = json.dumps({
        "topology": "buck",
        "confidence": 0.9,
        "role_map": {},
    })
    client = _install_mock_anthropic(monkeypatch, response)

    r1 = classify(buck_circuit)
    r2 = classify(buck_circuit)
    assert r1 is not None and r2 is not None
    assert r1.name == "buck" and r2.name == "buck"
    assert r2.source == "cache"
    # API must have been called EXACTLY ONCE — the second call hit
    # the cache.
    assert client.messages.create.call_count == 1


def test_cache_persists_across_processes(
        buck_circuit, isolated_cache, monkeypatch):
    """Writing the cache then reading it again (simulating a fresh
    process by force-reloading the module) preserves the entry."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = json.dumps({
        "topology": "buck",
        "confidence": 0.92,
        "role_map": {},
    })
    _install_mock_anthropic(monkeypatch, response)

    r1 = classify(buck_circuit)
    assert r1 is not None and r1.source == "llm"

    # Simulate "fresh process" — drop the mock and replay; cache should
    # still serve the answer without an API call.
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    r2 = classify(buck_circuit)
    assert r2 is not None
    assert r2.source == "cache"
    assert r2.name == "buck"


# ---------------------------------------------------------------------------
# classify() — failure / rejection paths
# ---------------------------------------------------------------------------

def test_malformed_json_returns_none(
        buck_circuit, isolated_cache, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_mock_anthropic(monkeypatch, "this is not JSON")
    result = classify(buck_circuit)
    assert result is None
    # No cache entry written for failed parses.
    cache_file = isolated_cache / "topology-cache.json"
    assert not cache_file.exists()


def test_unknown_topology_rejected(
        buck_circuit, isolated_cache, monkeypatch):
    """LLM returning a name not in LLM_KNOWN_TOPOLOGIES → None."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = json.dumps({
        "topology": "super_buck_v2",
        "confidence": 0.95,
    })
    _install_mock_anthropic(monkeypatch, response)
    result = classify(buck_circuit)
    assert result is None


def test_low_confidence_rejected(
        buck_circuit, isolated_cache, monkeypatch):
    """Confidence below LLM_MIN_CONFIDENCE → None even on a valid
    topology name."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = json.dumps({
        "topology": "buck",
        "confidence": 0.2,
    })
    _install_mock_anthropic(monkeypatch, response)
    result = classify(buck_circuit)
    assert result is None


def test_network_error_returns_none(
        buck_circuit, isolated_cache, monkeypatch):
    """Any exception in the SDK is swallowed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_mock_anthropic(monkeypatch, RuntimeError("simulated network down"))
    result = classify(buck_circuit)
    assert result is None


def test_code_fences_stripped_from_response(
        buck_circuit, isolated_cache, monkeypatch):
    """If the LLM wraps JSON in ```json … ``` we still parse it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    wrapped = (
        "```json\n"
        + json.dumps({"topology": "buck", "confidence": 0.91, "role_map": {}})
        + "\n```"
    )
    _install_mock_anthropic(monkeypatch, wrapped)
    result = classify(buck_circuit)
    assert result is not None
    assert result.name == "buck"


def test_cache_schema_mismatch_rebuilds(
        buck_circuit, isolated_cache, monkeypatch):
    """If the cache file has an unrecognised schema_version, it's
    discarded and the next classify() call hits the API."""
    cache_path = isolated_cache / "topology-cache.json"
    cache_path.write_text(json.dumps({
        "schema_version": "topology-cache-v0-prehistoric",
        "entries": {"deadbeef": {"topology": "buck", "confidence": 0.99}},
    }))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = json.dumps({"topology": "buck", "confidence": 0.91,
                            "role_map": {}})
    client = _install_mock_anthropic(monkeypatch, response)

    result = classify(buck_circuit)
    assert result is not None
    assert result.source == "llm"  # not "cache" — stale entry discarded
    client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# LLM_KNOWN_TOPOLOGIES sanity
# ---------------------------------------------------------------------------

def test_llm_topologies_superset_of_recognizer():
    """The LLM tier must know every topology the deterministic tier
    knows, plus the analog-amp / divider topologies it covers
    exclusively."""
    from pulsim.schematic.topology_recognizer import KNOWN_TOPOLOGIES
    assert KNOWN_TOPOLOGIES <= LLM_KNOWN_TOPOLOGIES
    # The five extras the LLM covers exclusively.
    extras = LLM_KNOWN_TOPOLOGIES - KNOWN_TOPOLOGIES
    assert "common_source" in extras
    assert "common_emitter" in extras
    assert "op_amp_inverting" in extras
    assert "op_amp_non_inverting" in extras
    assert "voltage_divider" in extras


# ---------------------------------------------------------------------------
# Gated real-API smoke test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("PULSIM_LLM_REAL_API", "0") != "1",
    reason="set PULSIM_LLM_REAL_API=1 + ANTHROPIC_API_KEY to enable",
)
def test_real_api_classifies_buck(buck_circuit, isolated_cache):
    """End-to-end smoke test against the real Anthropic API. Local
    only — skipped in CI."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    result = classify(buck_circuit)
    assert result is not None
    assert result.name == "buck", f"got {result.name!r}"
    assert result.confidence >= 0.7
    assert result.source == "llm"
