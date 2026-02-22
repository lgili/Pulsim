"""Regression checks for docs tooling and release deployment policy."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_on_section(workflow: dict) -> dict:
    # PyYAML can decode `on:` as boolean `True` under YAML 1.1 rules.
    if "on" in workflow:
        return workflow["on"]
    if True in workflow:
        return workflow[True]
    raise AssertionError("Workflow missing `on` section")


def test_docs_workflow_is_tag_only_and_version_scoped() -> None:
    workflow = _load_yaml(ROOT / ".github/workflows/docs.yml")
    on_section = _workflow_on_section(workflow)

    assert "workflow_dispatch" not in on_section
    assert "push" in on_section

    push_section = on_section["push"]
    assert "branches" not in push_section
    assert "tags" in push_section
    assert "v*.*.*" in push_section["tags"]


def test_mkdocs_uses_material_and_mike_versioning() -> None:
    config = _load_yaml(ROOT / "mkdocs.yml")

    assert config["theme"]["name"] == "material"
    assert config["extra"]["version"]["provider"] == "mike"

    mike_plugin = None
    for plugin in config.get("plugins", []):
        if isinstance(plugin, dict) and "mike" in plugin:
            mike_plugin = plugin["mike"]
            break

    assert mike_plugin is not None
    assert mike_plugin.get("version_selector") is True
    assert mike_plugin.get("canonical_version") == "latest"


def test_docs_requirements_pin_mkdocs_major_to_v1() -> None:
    requirements = (ROOT / "docs/requirements.txt").read_text(encoding="utf-8")

    assert "mkdocs>=1.6,<2.0" in requirements
    assert "mkdocs-material" in requirements
    assert "mike" in requirements
