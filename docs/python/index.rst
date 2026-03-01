Pulsim Python API Documentation
==================================

Pulsim is a high-performance circuit simulator for power electronics applications.
This documentation covers the Python API for Pulsim.

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
   api/client

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

Install Pulsim:

.. code-block:: bash

   pip install pulsim

Run a simple simulation:

.. code-block:: python

   import pulsim

   # Simulate from JSON netlist
   result = pulsim.simulate("circuit.json")

   # Access results
   print(result.time)
   print(result.voltages["out"])

Or build circuits programmatically:

.. code-block:: python

   import pulsim

   # Create circuit
   circuit = pulsim.Circuit("RC Filter")
   circuit.add_voltage_source("V1", "in", "0", 5.0)
   circuit.add_resistor("R1", "in", "out", 1000)
   circuit.add_capacitor("C1", "out", "0", 1e-6)

   # Configure simulation
   options = pulsim.SimulationOptions(
       stop_time=0.01,
       timestep=1e-6
   )

   # Run simulation
   result = pulsim.simulate(circuit, options)


Features
--------

* **Fast Transient Simulation**: Optimized sparse matrix solvers
* **Power Electronics**: Ideal switches, MOSFETs, IGBTs, diodes
* **Thermal Modeling**: Foster networks with temperature coupling
* **Loss Calculation**: Conduction and switching losses
* **Parallel Execution**: Multi-threaded sweeps and batch runs
* **Remote API**: gRPC client for server-based simulation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
