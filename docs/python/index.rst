Legacy docs/python entrypoint
=============================

This folder is kept only for backward compatibility in local setups.

The maintained documentation site now lives under ``docs/`` root and starts at:

- ``docs/index.md``

To build the maintained docs:

.. code-block:: bash

   python3 -m pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html
