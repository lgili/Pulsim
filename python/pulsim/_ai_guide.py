"""Accessor for the usage guide shipped with the package (``AI_GUIDE.md``).

Kept in a private module so that ``import pulsim`` never imports
``pulsim.guide`` (the ``python -m pulsim.guide`` entry point): runpy warns
when the module it is asked to run is already in ``sys.modules``.
"""

from __future__ import annotations

from importlib import resources

GUIDE_FILE = "AI_GUIDE.md"


def guide_path() -> str:
    """Filesystem path of the shipped ``AI_GUIDE.md``."""
    ref = resources.files("pulsim").joinpath(GUIDE_FILE)
    if not ref.is_file():
        raise FileNotFoundError(
            f"{GUIDE_FILE} is missing from the installed pulsim package "
            f"({ref}); the wheel or build-tree copy is incomplete")
    return str(ref)


def ai_guide() -> str:
    """Return the Pulsim usage guide for AI coding assistants (and people).

    The first line names the installed ``pulsim.__version__`` and the
    guide's path; the rest is ``AI_GUIDE.md`` verbatim. The same text is
    printed by ``python -m pulsim.guide``.
    """
    from . import __version__

    ref = resources.files("pulsim").joinpath(GUIDE_FILE)
    text = ref.read_text(encoding="utf-8")
    return f"pulsim {__version__} - {guide_path()}\n{text}"
