## ADDED Requirements

### Requirement: C-Block Netlist Node
The netlist YAML SHALL support a `c_block` node that mirrors the Python
`add_c_block` surface so a custom-code block can be declared
declaratively. The node SHALL specify its `inputs`, `outputs`, sample
time `dt`, language, and either inline `code` or a path to a compiled
`lib` (or a source `file`).

#### Scenario: Declare a c_block in YAML
- **WHEN** a netlist contains a `c_block` node with `inputs`, `outputs`, `dt`, `lang: c`, and inline `code`
- **THEN** loading the netlist creates the block, wires its inputs/outputs to the named circuit nodes, and runs it at `dt` during simulation

#### Scenario: Reference a precompiled library from YAML
- **WHEN** a `c_block` node sets `lib: path/to/block.so` instead of inline `code`
- **THEN** the loader binds the block to that shared library via the C-block ABI
