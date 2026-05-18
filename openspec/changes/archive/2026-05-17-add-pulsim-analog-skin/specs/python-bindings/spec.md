## ADDED Requirements

### Requirement: Switching-Device Analog Symbols in Schematic Render
``pulsim.schematic.render`` SHALL produce SVG output where MOSFET, IGBT, and voltage-controlled switch components are drawn with dedicated analog schematic symbols instead of falling back to a labeled generic rectangle.

#### Scenario: Render a MOSFET in a boost converter
- **WHEN** a circuit containing a MOSFET (e.g. the `Q1` in `boost_pfc`) is rendered via the default netlistsvg backend
- **THEN** the SVG contains an element with the canonical N-channel MOSFET symbol (gate stub, channel rectangle, drain/source terminals)
- **AND** does NOT contain a `s:type="generic"` element for that component

#### Scenario: Render an IGBT
- **WHEN** a circuit containing an IGBT is rendered
- **THEN** the SVG contains an element with the IGBT symbol (MOSFET body + collector triangle)

#### Scenario: Render a voltage-controlled switch
- **WHEN** a circuit containing a vcswitch (e.g. the `S1` in `buck_converter`) is rendered
- **THEN** the SVG contains an element with the vcswitch symbol (pivoting bar with explicit control terminal)
