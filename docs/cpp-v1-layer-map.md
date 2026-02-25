# C++ v1 Layer Map

This document defines the Phase 1 architectural module map for the v1 core and
the dependency direction policy used by automated checks.

## Layer Order

1. `domain-model`
2. `equation-services`
3. `solve-services`
4. `runtime-orchestrator`
5. `adapters`

Rule: dependencies are one-way from higher-numbered layers to lower-numbered
layers. A lower layer must not include files owned by a higher layer.

## Source of Truth

- Layer ownership map: `core/v1_layer_map.json`
- Boundary checker: `scripts/check_v1_layer_boundaries.py`

## Running The Check

```bash
python3 scripts/check_v1_layer_boundaries.py \
  --map core/v1_layer_map.json \
  --strict
```

The checker is also wired into CI under the lint job.
