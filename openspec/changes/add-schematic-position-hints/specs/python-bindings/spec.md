## ADDED Requirements

### Requirement: Position Hint Binding
Python bindings SHALL expose ``Circuit.set_position(name, layer=, slot=, x=, y=)`` and ``Circuit.position_hints()`` for programmatic schematic placement, parallel to the YAML ``position`` field.

#### Scenario: Set a semantic position hint from Python
- **WHEN** Python code calls ``circuit.set_position("Vdc", layer=0, slot=0)``
- **THEN** ``circuit.position_hints()`` returns a dict containing ``"Vdc"`` with ``layer == 0`` and ``slot == 0``

#### Scenario: YAML-loaded hints visible from Python
- **WHEN** a YAML netlist with ``position`` hints is loaded via ``YamlParser.load(...)``
- **THEN** ``circuit.position_hints()`` returns the same hint set the YAML declared

#### Scenario: render() applies hints automatically
- **WHEN** ``pulsim.schematic.render(circuit, "out.svg")`` runs on a circuit with hints
- **THEN** the rendered SVG places each hinted component at the user-specified grid coordinate (within rendering tolerance)
- **AND** components without a hint use the netlistsvg auto-layout
