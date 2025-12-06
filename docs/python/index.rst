SpiceLab Python API Documentation
==================================

SpiceLab is a high-performance circuit simulator for power electronics applications.
This documentation covers the Python API for SpiceLab.

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

Install SpiceLab:

.. code-block:: bash

   pip install spicelab

Run a simple simulation:

.. code-block:: python

   import spicelab

   # Simulate from JSON netlist
   result = spicelab.simulate("circuit.json")

   # Access results
   print(result.time)
   print(result.voltages["out"])

Or build circuits programmatically:

.. code-block:: python

   import spicelab

   # Create circuit
   circuit = spicelab.Circuit("RC Filter")
   circuit.add_voltage_source("V1", "in", "0", 5.0)
   circuit.add_resistor("R1", "in", "out", 1000)
   circuit.add_capacitor("C1", "out", "0", 1e-6)

   # Configure simulation
   options = spicelab.SimulationOptions(
       stop_time=0.01,
       timestep=1e-6
   )

   # Run simulation
   result = spicelab.simulate(circuit, options)


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
