# Pulsim Source Map

## Table of Contents
- [Purpose](#purpose)
- [Primary Sources](#primary-sources)
- [Question-to-Source Routing](#question-to-source-routing)
- [Fast Verification Commands](#fast-verification-commands)

## Purpose

Use this map to decide which project files to read before answering user questions about the Pulsim library.

## Primary Sources

- API exports: `python/pulsim/__init__.py`
- Typed API surface: `python/pulsim/__init__.pyi`
- Binding implementation details: `python/bindings.cpp`
- User-facing runtime flow: `docs/user-guide.md`
- YAML schema and component keys: `docs/netlist-format.md`
- Convergence tuning: `docs/convergence-tuning-guide.md`
- Performance tuning: `docs/performance-tuning.md`
- Device model specifics: `docs/device-models.md`
- Migration and removed surfaces: `docs/migration-guide.md`

## Question-to-Source Routing

- "How do I start?" or "first simulation": read `README.md` + `docs/user-guide.md`.
- "Which parameters exist in class X?": read `references/python-api-inventory.md`.
- "Class exported but missing from stub": read `references/python-api-inventory.md` export gaps, then inspect `python/bindings.cpp`.
- "How to write YAML netlist?": read `references/yaml-netlist-and-solver.md` + `docs/netlist-format.md`.
- "Convergence failed / timestep stalls": read `references/troubleshooting-and-tuning.md` + `docs/convergence-tuning-guide.md`.
- "How to tune runtime speed": read `docs/performance-tuning.md`.

## Fast Verification Commands

```bash
# Rebuild API inventory used by this skill
python3 skills/pulsim-library-expert/scripts/build_api_inventory.py

# Locate class bindings quickly
rg "py::class_<.*>(v2, \"ClassName\"" python/bindings.cpp

# Locate option fields quickly
rg "def_readwrite\(|class .*Config|class .*Options" python/bindings.cpp python/pulsim/__init__.pyi
```
