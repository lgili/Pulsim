"""Allow `python -m pulsim_bench` invocation as an alternative to the
installed `pulsim-bench` console script. Useful when running from a
checkout without `pip install -e tools/bench`."""

from pulsim_bench.cli import app

if __name__ == "__main__":
    app()
