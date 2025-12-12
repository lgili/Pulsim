"""SPICE-like netlist parser for Pulsim.

This module provides functions to parse SPICE-format netlists and create
Pulsim Circuit objects for simulation.

Example:
    >>> import pulsim as ps
    >>> netlist = '''
    ... * RC Circuit
    ... V1 in 0 5
    ... R1 in out 1k
    ... C1 out 0 1u
    ... .end
    ... '''
    >>> ckt = ps.parse_netlist(netlist)
    >>> result = ps.dc_operating_point(ckt)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ._pulsim import Circuit, MOSFETParams, IGBTParams


# Engineering suffix multipliers (case-insensitive)
SUFFIX_MULTIPLIERS = {
    'f': 1e-15,    # femto
    'p': 1e-12,    # pico
    'n': 1e-9,     # nano
    'u': 1e-6,     # micro
    'µ': 1e-6,     # micro (unicode)
    'm': 1e-3,     # milli
    'k': 1e3,      # kilo
    'meg': 1e6,    # mega
    'g': 1e9,      # giga
    't': 1e12,     # tera
}


class NetlistParseError(Exception):
    """Error during netlist parsing with line context."""

    def __init__(
        self,
        message: str,
        line_number: Optional[int] = None,
        line_content: Optional[str] = None
    ):
        self.line_number = line_number
        self.line_content = line_content
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        if self.line_number is not None:
            msg = f"Line {self.line_number}: {message}"
            if self.line_content:
                msg += f"\n  >> {self.line_content}"
            return msg
        return message


@dataclass
class NetlistWarning:
    """Warning from netlist parsing."""
    line_number: int
    line_content: str
    message: str

    def __str__(self) -> str:
        return f"Line {self.line_number}: {self.message}"


@dataclass
class ParsedNetlist:
    """Result of parsing a SPICE netlist."""
    circuit: Circuit
    title: str = ""
    warnings: List[NetlistWarning] = field(default_factory=list)
    models: Dict[str, Any] = field(default_factory=dict)
    node_map: Dict[str, int] = field(default_factory=dict)


def parse_value(value_str: str, line_number: Optional[int] = None) -> float:
    """Parse a value string with optional engineering suffix.

    Supports:
        - Plain numbers: '100', '1.5', '-3.14'
        - Scientific notation: '1e-6', '1.5E3'
        - Engineering suffixes: '10k', '100u', '1.5meg', '4.7n'

    Args:
        value_str: Value string to parse
        line_number: Line number for error reporting

    Returns:
        Parsed float value

    Raises:
        NetlistParseError: If value cannot be parsed

    Examples:
        >>> parse_value('100')
        100.0
        >>> parse_value('10k')
        10000.0
        >>> parse_value('1.5meg')
        1500000.0
        >>> parse_value('100u')
        0.0001
        >>> parse_value('4.7n')
        4.7e-09
    """
    original = value_str
    value_str = value_str.strip().lower()

    if not value_str:
        raise NetlistParseError(f"Empty value", line_number=line_number)

    # Handle scientific notation first (before checking suffixes)
    # Pattern: optional sign, digits, optional decimal, 'e' or 'E', optional sign, digits
    sci_match = re.match(r'^([+-]?\d*\.?\d+)[eE]([+-]?\d+)$', value_str)
    if sci_match:
        try:
            return float(value_str)
        except ValueError:
            raise NetlistParseError(
                f"Invalid scientific notation: '{original}'",
                line_number=line_number
            )

    # Check for 'meg' suffix first (before 'm' for milli)
    if value_str.endswith('meg'):
        try:
            return float(value_str[:-3]) * 1e6
        except ValueError:
            raise NetlistParseError(
                f"Invalid value: '{original}'",
                line_number=line_number
            )

    # Check single-character suffixes
    for suffix, multiplier in SUFFIX_MULTIPLIERS.items():
        if suffix != 'meg' and value_str.endswith(suffix):
            try:
                return float(value_str[:-len(suffix)]) * multiplier
            except ValueError:
                raise NetlistParseError(
                    f"Invalid value: '{original}'",
                    line_number=line_number
                )

    # No suffix - try direct conversion
    try:
        return float(value_str)
    except ValueError:
        raise NetlistParseError(
            f"Cannot parse value: '{original}'",
            line_number=line_number
        )


class _NetlistParser:
    """Internal parser state machine."""

    def __init__(
        self,
        strict: bool = False,
        default_mosfet_params: Optional[MOSFETParams] = None,
        default_igbt_params: Optional[IGBTParams] = None,
    ):
        self.strict = strict
        self.default_mosfet_params = default_mosfet_params
        self.default_igbt_params = default_igbt_params

        self.circuit = Circuit()
        self.node_map: Dict[str, int] = {}
        self.models: Dict[str, Any] = {}
        self.warnings: List[NetlistWarning] = []
        self.title = ""
        self._first_line_processed = False

    def _get_node(self, name: str) -> int:
        """Get or create node index for a node name."""
        name_lower = name.lower()

        # Ground aliases
        if name_lower in ('0', 'gnd', 'ground'):
            return Circuit.ground()

        if name not in self.node_map:
            idx = self.circuit.add_node(name)
            self.node_map[name] = idx

        return self.node_map[name]

    def _add_warning(self, line_num: int, line: str, message: str):
        self.warnings.append(NetlistWarning(line_num, line, message))

    def _preprocess_line(self, line: str) -> str:
        """Remove comments and normalize whitespace."""
        # Remove inline comments (after ;)
        line = line.split(';')[0]
        line = line.strip()

        # Skip full-line comments
        if line.startswith('*'):
            return ''

        return line

    def parse(self, netlist: str) -> ParsedNetlist:
        """Parse the netlist and return result.

        Uses a two-pass approach:
        1. First pass: collect all node names and .MODEL definitions
        2. Second pass: create nodes (all at once) and add devices

        This is required because the C++ Circuit class expects all nodes
        to be created before devices are added.
        """
        lines = netlist.strip().split('\n')

        # Storage for parsed device info (to be added in second pass)
        parsed_devices: List[Tuple[str, List[str], int, str]] = []  # (type, tokens, line_num, raw_line)
        all_nodes: set = set()

        # First pass: collect nodes and models
        for line_num, raw_line in enumerate(lines, start=1):
            # First line is title (SPICE convention)
            if not self._first_line_processed:
                self._first_line_processed = True
                stripped = raw_line.strip()
                if stripped.startswith('*'):
                    self.title = stripped[1:].strip()
                    continue
                elif stripped:
                    # First non-empty line that's not a comment - could be title or device
                    if not stripped.startswith('.') and not stripped[0].upper() in 'RCLVIDSQM':
                        self.title = stripped
                        continue

            line = self._preprocess_line(raw_line)
            if not line:
                continue

            tokens = line.split()
            if not tokens:
                continue

            first = tokens[0]

            # Handle directives in first pass (for .MODEL)
            if first.startswith('.'):
                if first.upper() == '.MODEL':
                    self._parse_model(tokens, line_num, raw_line)
                elif first.upper() != '.END':
                    # Store unknown directives for warning in second pass
                    parsed_devices.append(('directive', tokens, line_num, raw_line))
                continue

            # Collect node names from device lines
            device_char = first[0].upper()

            if device_char in 'RCLVI':
                # 2-terminal: name n1 n2 value
                if len(tokens) >= 3:
                    all_nodes.add(tokens[1])
                    all_nodes.add(tokens[2])
            elif device_char == 'D':
                # Diode: name anode cathode
                if len(tokens) >= 3:
                    all_nodes.add(tokens[1])
                    all_nodes.add(tokens[2])
            elif device_char == 'S':
                # Switch: name n1 n2
                if len(tokens) >= 3:
                    all_nodes.add(tokens[1])
                    all_nodes.add(tokens[2])
            elif device_char == 'M':
                # MOSFET: name drain gate source
                if len(tokens) >= 4:
                    all_nodes.add(tokens[1])
                    all_nodes.add(tokens[2])
                    all_nodes.add(tokens[3])
            elif device_char == 'Q':
                # IGBT: name collector gate emitter
                if len(tokens) >= 4:
                    all_nodes.add(tokens[1])
                    all_nodes.add(tokens[2])
                    all_nodes.add(tokens[3])

            # Store for second pass
            parsed_devices.append((device_char, tokens, line_num, raw_line))

        # Remove ground aliases from node set (they don't need to be created)
        ground_aliases = {'0', 'gnd', 'ground'}
        all_nodes = {n for n in all_nodes if n.lower() not in ground_aliases}

        # Create all nodes first (sorted for deterministic ordering)
        for node_name in sorted(all_nodes):
            idx = self.circuit.add_node(node_name)
            self.node_map[node_name] = idx

        # Second pass: add devices
        for device_type, tokens, line_num, raw_line in parsed_devices:
            if device_type == 'directive':
                # Handle unknown directives
                directive = tokens[0].upper()
                msg = f"Unsupported directive: {directive}"
                if self.strict:
                    raise NetlistParseError(msg, line_num, raw_line)
                self._add_warning(line_num, raw_line, msg)
                continue

            # Route to device handler
            handlers = {
                'R': self._add_resistor,
                'C': self._add_capacitor,
                'L': self._add_inductor,
                'V': self._add_voltage_source,
                'I': self._add_current_source,
                'D': self._add_diode,
                'S': self._add_switch,
                'M': self._add_mosfet,
                'Q': self._add_igbt,
            }

            handler = handlers.get(device_type)
            if handler:
                handler(tokens, line_num, raw_line)
            else:
                msg = f"Unknown device type: '{tokens[0]}'"
                if self.strict:
                    raise NetlistParseError(msg, line_num, raw_line)
                self._add_warning(line_num, raw_line, msg)

        return ParsedNetlist(
            circuit=self.circuit,
            title=self.title,
            warnings=self.warnings,
            models=self.models,
            node_map=self.node_map,
        )

    def _parse_line(self, line: str, line_num: int, raw_line: str):
        """Parse a single netlist line."""
        tokens = line.split()
        if not tokens:
            return

        first = tokens[0]

        # Directives
        if first.startswith('.'):
            self._handle_directive(tokens, line_num, raw_line)
            return

        # Device by first character
        device_char = first[0].upper()

        handlers = {
            'R': self._add_resistor,
            'C': self._add_capacitor,
            'L': self._add_inductor,
            'V': self._add_voltage_source,
            'I': self._add_current_source,
            'D': self._add_diode,
            'S': self._add_switch,
            'M': self._add_mosfet,
            'Q': self._add_igbt,
        }

        handler = handlers.get(device_char)
        if handler:
            handler(tokens, line_num, raw_line)
        else:
            msg = f"Unknown device type: '{first}'"
            if self.strict:
                raise NetlistParseError(msg, line_num, raw_line)
            self._add_warning(line_num, raw_line, msg)

    def _handle_directive(self, tokens: List[str], line_num: int, raw_line: str):
        """Handle SPICE directives."""
        directive = tokens[0].upper()

        if directive == '.END':
            return

        if directive == '.MODEL':
            self._parse_model(tokens, line_num, raw_line)
            return

        # Unsupported directives
        msg = f"Unsupported directive: {directive}"
        if self.strict:
            raise NetlistParseError(msg, line_num, raw_line)
        self._add_warning(line_num, raw_line, msg)

    def _parse_model(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse .MODEL directive."""
        if len(tokens) < 3:
            raise NetlistParseError(
                ".MODEL requires: .MODEL name type [(params)]",
                line_num, raw_line
            )

        model_name = tokens[1]
        model_type = tokens[2].upper()

        # Parse parameters in parentheses
        params_str = ' '.join(tokens[3:])
        params = self._parse_model_params(params_str, line_num)

        if model_type in ('NMOS', 'PMOS'):
            mosfet_params = MOSFETParams()
            if 'VTH' in params:
                mosfet_params.vth = params['VTH']
            if 'KP' in params:
                mosfet_params.kp = params['KP']
            if 'LAMBDA' in params:
                mosfet_params.lambda_ = params['LAMBDA']
            mosfet_params.is_nmos = (model_type == 'NMOS')
            self.models[model_name] = mosfet_params

        elif model_type == 'IGBT':
            igbt_params = IGBTParams()
            if 'VTH' in params:
                igbt_params.vth = params['VTH']
            if 'GON' in params:
                igbt_params.g_on = params['GON']
            if 'GOFF' in params:
                igbt_params.g_off = params['GOFF']
            if 'VCESAT' in params:
                igbt_params.v_ce_sat = params['VCESAT']
            self.models[model_name] = igbt_params

        elif model_type in ('D', 'SW'):
            # Store raw params for future use
            self.models[model_name] = {'type': model_type, 'params': params}

        else:
            self._add_warning(line_num, raw_line,
                           f"Unsupported model type: {model_type}")

    def _parse_model_params(self, params_str: str, line_num: int) -> Dict[str, float]:
        """Parse model parameters from string like '(VTH=2 KP=0.1)'."""
        params = {}

        # Remove parentheses
        params_str = params_str.replace('(', ' ').replace(')', ' ')

        # Find all param=value pairs
        for match in re.finditer(r'(\w+)\s*=\s*([^\s,)]+)', params_str):
            name = match.group(1).upper()
            try:
                value = parse_value(match.group(2), line_num)
                params[name] = value
            except NetlistParseError:
                # Just skip invalid params with warning
                pass

        return params

    # Device handlers

    def _add_resistor(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Rname n1 n2 value"""
        if len(tokens) < 4:
            raise NetlistParseError(
                "Resistor requires: Rname n1 n2 value",
                line_num, raw_line
            )
        name, n1, n2 = tokens[0], tokens[1], tokens[2]
        value = parse_value(tokens[3], line_num)

        if value <= 0:
            raise NetlistParseError(
                f"Resistor value must be positive: {value}",
                line_num, raw_line
            )

        self.circuit.add_resistor(
            name,
            self._get_node(n1),
            self._get_node(n2),
            value
        )

    def _add_capacitor(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Cname n1 n2 value [IC=initial]"""
        if len(tokens) < 4:
            raise NetlistParseError(
                "Capacitor requires: Cname n1 n2 value [IC=v]",
                line_num, raw_line
            )
        name, n1, n2 = tokens[0], tokens[1], tokens[2]
        value = parse_value(tokens[3], line_num)

        # Parse optional IC=value
        ic = 0.0
        for token in tokens[4:]:
            if token.upper().startswith('IC='):
                ic = parse_value(token[3:], line_num)
                break

        self.circuit.add_capacitor(
            name,
            self._get_node(n1),
            self._get_node(n2),
            value,
            ic
        )

    def _add_inductor(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Lname n1 n2 value [IC=initial]"""
        if len(tokens) < 4:
            raise NetlistParseError(
                "Inductor requires: Lname n1 n2 value [IC=i]",
                line_num, raw_line
            )
        name, n1, n2 = tokens[0], tokens[1], tokens[2]
        value = parse_value(tokens[3], line_num)

        # Parse optional IC=value
        ic = 0.0
        for token in tokens[4:]:
            if token.upper().startswith('IC='):
                ic = parse_value(token[3:], line_num)
                break

        self.circuit.add_inductor(
            name,
            self._get_node(n1),
            self._get_node(n2),
            value,
            ic
        )

    def _add_voltage_source(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Vname n+ n- value [DC value]"""
        if len(tokens) < 4:
            raise NetlistParseError(
                "Voltage source requires: Vname n+ n- value",
                line_num, raw_line
            )
        name, npos, nneg = tokens[0], tokens[1], tokens[2]

        # Handle optional "DC" keyword
        value_idx = 3
        if tokens[3].upper() == 'DC' and len(tokens) > 4:
            value_idx = 4

        value = parse_value(tokens[value_idx], line_num)

        self.circuit.add_voltage_source(
            name,
            self._get_node(npos),
            self._get_node(nneg),
            value
        )

    def _add_current_source(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Iname n+ n- value [DC value]"""
        if len(tokens) < 4:
            raise NetlistParseError(
                "Current source requires: Iname n+ n- value",
                line_num, raw_line
            )
        name, npos, nneg = tokens[0], tokens[1], tokens[2]

        # Handle optional "DC" keyword
        value_idx = 3
        if tokens[3].upper() == 'DC' and len(tokens) > 4:
            value_idx = 4

        value = parse_value(tokens[value_idx], line_num)

        self.circuit.add_current_source(
            name,
            self._get_node(npos),
            self._get_node(nneg),
            value
        )

    def _add_diode(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Dname anode cathode [model]"""
        if len(tokens) < 3:
            raise NetlistParseError(
                "Diode requires: Dname anode cathode",
                line_num, raw_line
            )
        name, anode, cathode = tokens[0], tokens[1], tokens[2]

        # Get model params if specified
        g_on = 1e3
        g_off = 1e-9
        if len(tokens) > 3:
            model_name = tokens[3]
            if model_name in self.models:
                model = self.models[model_name]
                if isinstance(model, dict) and model.get('type') == 'D':
                    params = model.get('params', {})
                    g_on = params.get('GON', g_on)
                    g_off = params.get('GOFF', g_off)

        self.circuit.add_diode(
            name,
            self._get_node(anode),
            self._get_node(cathode),
            g_on,
            g_off
        )

    def _add_switch(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Sname n1 n2 [model] [ON|OFF]"""
        if len(tokens) < 3:
            raise NetlistParseError(
                "Switch requires: Sname n1 n2",
                line_num, raw_line
            )
        name, n1, n2 = tokens[0], tokens[1], tokens[2]

        # Check for ON/OFF state
        closed = False
        g_on = 1e6
        g_off = 1e-12

        for token in tokens[3:]:
            upper = token.upper()
            if upper == 'ON':
                closed = True
            elif upper == 'OFF':
                closed = False
            elif token in self.models:
                model = self.models[token]
                if isinstance(model, dict) and model.get('type') == 'SW':
                    params = model.get('params', {})
                    g_on = params.get('RON', 1.0) if 'RON' in params else g_on
                    g_off = 1.0 / params.get('ROFF', 1e12) if 'ROFF' in params else g_off

        self.circuit.add_switch(
            name,
            self._get_node(n1),
            self._get_node(n2),
            closed,
            g_on,
            g_off
        )

    def _add_mosfet(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Mname drain gate source [model]"""
        if len(tokens) < 4:
            raise NetlistParseError(
                "MOSFET requires: Mname drain gate source",
                line_num, raw_line
            )
        name = tokens[0]
        drain, gate, source = tokens[1], tokens[2], tokens[3]

        # Get params
        params = self.default_mosfet_params or MOSFETParams()
        if len(tokens) > 4:
            model_name = tokens[4]
            if model_name in self.models:
                model = self.models[model_name]
                if isinstance(model, MOSFETParams):
                    params = model

        self.circuit.add_mosfet(
            name,
            self._get_node(gate),
            self._get_node(drain),
            self._get_node(source),
            params
        )

    def _add_igbt(self, tokens: List[str], line_num: int, raw_line: str):
        """Parse: Qname collector gate emitter [model]"""
        if len(tokens) < 4:
            raise NetlistParseError(
                "IGBT requires: Qname collector gate emitter",
                line_num, raw_line
            )
        name = tokens[0]
        collector, gate, emitter = tokens[1], tokens[2], tokens[3]

        # Get params
        params = self.default_igbt_params or IGBTParams()
        if len(tokens) > 4:
            model_name = tokens[4]
            if model_name in self.models:
                model = self.models[model_name]
                if isinstance(model, IGBTParams):
                    params = model

        self.circuit.add_igbt(
            name,
            self._get_node(gate),
            self._get_node(collector),
            self._get_node(emitter),
            params
        )


def parse_netlist(
    netlist: str,
    *,
    strict: bool = False,
    default_mosfet_params: Optional[MOSFETParams] = None,
    default_igbt_params: Optional[IGBTParams] = None,
) -> Circuit:
    """Parse a SPICE-like netlist string and return a Pulsim Circuit.

    Args:
        netlist: SPICE netlist string
        strict: If True, raise error on unknown directives; if False, warn and skip
        default_mosfet_params: Default MOSFET parameters when model not specified
        default_igbt_params: Default IGBT parameters when model not specified

    Returns:
        Configured Circuit object ready for simulation

    Raises:
        NetlistParseError: On syntax errors or invalid device specifications

    Example:
        >>> netlist = '''
        ... * RC Circuit
        ... V1 in 0 5
        ... R1 in out 1k
        ... C1 out 0 1u IC=0
        ... .end
        ... '''
        >>> ckt = parse_netlist(netlist)
        >>> print(ckt.num_devices())
        3
    """
    parser = _NetlistParser(
        strict=strict,
        default_mosfet_params=default_mosfet_params,
        default_igbt_params=default_igbt_params,
    )
    result = parser.parse(netlist)
    return result.circuit


def parse_netlist_verbose(
    netlist: str,
    *,
    strict: bool = False,
    default_mosfet_params: Optional[MOSFETParams] = None,
    default_igbt_params: Optional[IGBTParams] = None,
) -> ParsedNetlist:
    """Parse netlist and return detailed result with warnings.

    Args:
        netlist: SPICE netlist string
        strict: If True, raise error on unknown directives
        default_mosfet_params: Default MOSFET parameters
        default_igbt_params: Default IGBT parameters

    Returns:
        ParsedNetlist object containing:
        - circuit: The built Circuit object
        - title: Circuit title (first comment line)
        - warnings: List of NetlistWarning objects
        - models: Dict of parsed .model definitions
        - node_map: Dict mapping node names to indices

    Example:
        >>> result = parse_netlist_verbose(netlist)
        >>> print(f"Title: {result.title}")
        >>> for w in result.warnings:
        ...     print(f"Warning: {w}")
    """
    parser = _NetlistParser(
        strict=strict,
        default_mosfet_params=default_mosfet_params,
        default_igbt_params=default_igbt_params,
    )
    return parser.parse(netlist)
