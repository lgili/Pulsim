"""``python -m pulsim.guide`` prints the usage guide for the installed pulsim.

``python -m pulsim.guide --path`` prints only the path of ``AI_GUIDE.md``.
The text is also available as ``pulsim.ai_guide()``.
"""

from __future__ import annotations

import sys


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from ._ai_guide import ai_guide, guide_path

    parser = argparse.ArgumentParser(
        prog="python -m pulsim.guide",
        description="Print the Pulsim usage guide for AI coding assistants.")
    parser.add_argument("--path", action="store_true",
                        help="print only the path of AI_GUIDE.md")
    args = parser.parse_args(argv)
    _write((guide_path() if args.path else ai_guide()) + "\n")
    return 0


def _write(text: str) -> None:
    # The guide is ASCII, but the path may not be; never fail on a
    # console whose encoding cannot represent it.
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.flush()
        sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    sys.exit(main())
