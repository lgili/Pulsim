---
name: pulsim-library-expert
description: Expert guidance for using Pulsim (Python API and YAML netlists), including class/function discovery, full parameter/configuration explanation, solver and convergence tuning, and beginner-to-advanced usage patterns. Use when users ask how to use Pulsim, configure simulation options, understand netlist/component/waveform fields, tune performance/convergence, troubleshoot failures, or migrate APIs. Trigger for Portuguese asks such as como usar pulsim, parametros, configuracao, netlist, convergencia, simulacao, e exemplos.
---

# Pulsim Library Expert

## Overview

Use this skill to answer Pulsim usage questions with source-grounded precision, from onboarding to advanced tuning.
Prioritize exact parameter names, executable examples, and concrete next steps.

## Workflow

1. Classify request type and user level.
- Detect whether the question is about Python API, YAML netlist, convergence/performance tuning, troubleshooting, or migration.
- Infer level as beginner or advanced from user wording and requested depth.

2. Load only the required references.
- For complete API names/attributes/methods, read `references/python-api-inventory.md`.
- For YAML schema and simulation keys, read `references/yaml-netlist-and-solver.md`.
- For diagnosis and tuning, read `references/troubleshooting-and-tuning.md`.
- For source routing, read `references/source-map.md`.

3. Verify uncertain details in project sources.
- Confirm exported symbols in `python/pulsim/__init__.py`.
- Confirm typed parameters in `python/pulsim/__init__.pyi`.
- If symbol is exported but not typed, inspect `python/bindings.cpp`.

4. Build answer at the right depth.
- Beginner mode: explain minimum concepts, provide one working example, list only critical knobs.
- Advanced mode: include tradeoffs, fallback strategy, integrator/solver tuning, and validation checks.

5. End with a runnable validation step.
- Include an exact command or short script snippet for user verification.
- State expected success criteria (e.g., `result.success == True`, stable timestep progression).

## Output Rules

- Use exact Pulsim parameter names as defined in source.
- Prefer short, runnable examples over abstract descriptions.
- Call out product-surface boundary: supported path is Python package plus YAML; legacy CLI and gRPC are not the primary user-facing surface.
- When discussing configuration, group knobs by objective: correctness, convergence robustness, runtime performance, or model fidelity.
- If a requested feature appears missing, explicitly state whether it is absent in stubs, absent in exports, or present only in bindings.

## Reference Maintenance

Regenerate API inventory after API changes:

```bash
python3 skills/pulsim-library-expert/scripts/build_api_inventory.py
```

Use this refresh whenever `python/pulsim/__init__.py`, `python/pulsim/__init__.pyi`, or `python/bindings.cpp` changes.
