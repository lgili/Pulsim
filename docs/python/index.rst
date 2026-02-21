Pulsim Python API Documentation
===============================

Pulsim provides a Python-first interface to the unified v1 simulation kernel.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/circuit
   api/simulation
   api/devices
   api/results

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/rc_filter
   examples/buck_converter
   examples/thermal_modeling

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   contributing

Quick Start
-----------

.. code-block:: python

   import pulsim as ps

   parser = ps.YamlParser(ps.YamlParserOptions())
   circuit, options = parser.load("circuit.yaml")

   options.newton_options.num_nodes = int(circuit.num_nodes())
   options.newton_options.num_branches = int(circuit.num_branches())

   simulator = ps.Simulator(circuit, options)
   result = simulator.run_transient(circuit.initial_state())

   print(result.success, result.total_steps)

Features
--------

* Fast transient simulation with runtime linear-solver stack
* Power-electronics devices and converter-focused workflows
* Electro-thermal options and telemetry
* Deterministic fallback trace support for stiff cases
* YAML-first benchmark/parity/stress tooling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
