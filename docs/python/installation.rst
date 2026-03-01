Installation
============

Requirements
------------

* Python 3.8 or later
* NumPy 1.20 or later
* A C++20 compatible compiler (for building from source)

Installing from PyPI
--------------------

The easiest way to install Pulsim is using pip:

.. code-block:: bash

   pip install pulsim

This will install the pre-built binary package if available for your platform.


Installing from Source
----------------------

To build Pulsim from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-org/pulsim-core.git
      cd pulsim-core

2. Install build dependencies:

   .. code-block:: bash

      pip install build pybind11 numpy

3. Build and install:

   .. code-block:: bash

      cd python
      pip install .

Or for development installation:

   .. code-block:: bash

      pip install -e .


Optional Dependencies
---------------------

For additional functionality, install optional dependencies:

.. code-block:: bash

   # For plotting
   pip install matplotlib

   # For data analysis
   pip install pandas xarray

   # For Jupyter notebooks
   pip install jupyter ipywidgets

   # For gRPC client
   pip install grpcio grpcio-tools

   # All optional dependencies
   pip install pulsim[all]


Verifying Installation
----------------------

Verify that Pulsim is installed correctly:

.. code-block:: python

   import pulsim
   print(pulsim.__version__)

   # Run a quick test
   circuit = pulsim.Circuit("Test")
   circuit.add_resistor("R1", "a", "b", 1000)
   print("Pulsim is working!")


Docker
------

Pulsim is also available as a Docker image:

.. code-block:: bash

   # Pull the image
   docker pull pulsim:latest

   # Run the gRPC server
   docker run -p 50051:50051 pulsim:latest

Connect from Python:

.. code-block:: python

   from pulsim.client import PulsimClient

   client = PulsimClient("localhost:50051")
   result = client.simulate("circuit.json")
