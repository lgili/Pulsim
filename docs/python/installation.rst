Installation
============

Requirements
------------

* Python 3.10 or later
* NumPy 1.20 or later
* C++23-capable toolchain (when building from source)
* CMake 3.20+

Install from source (repository workflow)
----------------------------------------

.. code-block:: bash

   cmake -S . -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DPULSIM_BUILD_PYTHON=ON
   cmake --build build -j

Use local bindings directly:

.. code-block:: bash

   export PYTHONPATH=build/python

Install package with pip
------------------------

.. code-block:: bash

   pip install pulsim

Optional dependencies
---------------------

.. code-block:: bash

   # Plotting / notebooks
   pip install matplotlib jupyter ipywidgets

   # Data workflows
   pip install pandas xarray

Verify installation
-------------------

.. code-block:: python

   import pulsim as ps

   print(ps.__version__)
   parser = ps.YamlParser(ps.YamlParserOptions())

Notes
-----

Pulsim's supported user-facing runtime is Python + YAML netlists.
Legacy CLI/gRPC/JSON flows are not part of the supported surface.
