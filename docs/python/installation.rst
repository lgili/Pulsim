Installation
============

Requirements
------------

* Python 3.8 or later
* NumPy 1.20 or later
* A C++20 compatible compiler (for building from source)

Installing from PyPI
--------------------

The easiest way to install SpiceLab is using pip:

.. code-block:: bash

   pip install spicelab

This will install the pre-built binary package if available for your platform.


Installing from Source
----------------------

To build SpiceLab from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-org/spicelab-core.git
      cd spicelab-core

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
   pip install spicelab[all]


Verifying Installation
----------------------

Verify that SpiceLab is installed correctly:

.. code-block:: python

   import spicelab
   print(spicelab.__version__)

   # Run a quick test
   circuit = spicelab.Circuit("Test")
   circuit.add_resistor("R1", "a", "b", 1000)
   print("SpiceLab is working!")


Docker
------

SpiceLab is also available as a Docker image:

.. code-block:: bash

   # Pull the image
   docker pull spicelab:latest

   # Run the gRPC server
   docker run -p 50051:50051 spicelab:latest

Connect from Python:

.. code-block:: python

   from spicelab.client import SpiceLabClient

   client = SpiceLabClient("localhost:50051")
   result = client.simulate("circuit.json")
