"""The per-user compile cache for native code Pulsim builds from source.

C blocks (``add_c_block(code=...)``) compile here; the cache was a shared
directory under the system temp dir before 2.1, where another local user
could plant a ``cblk_<hash>.so`` that Pulsim would then load. Now:

* the directory is per user — ``~/Library/Caches/pulsim/<sub>`` (macOS),
  ``$XDG_CACHE_HOME`` or ``~/.cache/pulsim/<sub>`` (Linux),
  ``%LOCALAPPDATA%\\pulsim\\<sub>`` (Windows) — mode 0700, and both it and
  ``pulsim/`` must be owned by the current user and writable only by them
  (else refused: Pulsim cannot repair a directory someone else controls);
* a cached LIBRARY that another user owns, that the group or others can
  write, or that is a symlink is not loaded: it is deleted and rebuilt;
* Windows has no POSIX owner or mode: the cache must sit inside
  ``%USERPROFILE%`` (whose ACL is the user's).

A build writes private names and renames atomically, so a concurrent
process never loads a half-written library.
"""
from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Sequence


def cache_dir(sub: str) -> Path:
    """The per-user cache directory ``pulsim/<sub>``, created mode 0700."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA")
                    or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME")
                    or Path.home() / ".cache")
    d = base / "pulsim" / sub
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "posix":
        check_in_profile(d)
    for p in (d.parent, d):         # pulsim/ too: its owner could swap <sub>/
        if os.name == "posix" and p.lstat().st_uid == os.getuid() \
                and not p.is_symlink():
            os.chmod(p, 0o700)      # mkdir's mode is masked by the umask
        check_owned(p)
    return d


def check_in_profile(path: Path) -> None:
    """Windows: refuse a cache outside %USERPROFILE%."""
    home = Path(os.environ.get("USERPROFILE") or Path.home()).resolve()
    if not Path(path).resolve().is_relative_to(home):
        raise RuntimeError(
            f"native cache: refusing {path}: it is outside your profile "
            f"({home}), so pulsim cannot tell who may write it. Point "
            "LOCALAPPDATA inside your profile.")


def is_private(path: Path) -> bool:
    """POSIX: owned by the current user, not group/world-writable, not a
    symlink. Always True elsewhere."""
    if os.name != "posix":
        return True
    st = path.lstat()
    return st.st_uid == os.getuid() and not st.st_mode & 0o022 \
        and not path.is_symlink()


def check_owned(path: Path) -> None:
    """Refuse a cache directory another user owns or can write (POSIX)."""
    if not is_private(path):
        st = path.lstat()
        raise RuntimeError(
            f"native cache: refusing {path}: owned by uid {st.st_uid} with "
            f"mode {oct(st.st_mode & 0o777)}; the compile cache must belong "
            "to you and be writable only by you. Remove it.")


def compiled(src: str, ext: str, cmd: Sequence[str], key_parts: Sequence[str],
             *, link: Sequence[str] = (), sub: str, prefix: str, who: str) -> Path:
    """The shared library built from ``src`` by ``cmd`` (the compiler and
    its flags; the source, ``-o <out>`` and ``link`` are appended),
    cached under ``key_parts``. A cached file that is not private is
    deleted and rebuilt, never loaded."""
    key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]
    cache = cache_dir(sub)
    so_path = cache / f"{prefix}_{key}.so"
    if so_path.exists() or so_path.is_symlink():
        if is_private(so_path) and stat.S_ISREG(so_path.lstat().st_mode):
            return so_path
        so_path.unlink()            # our directory: we may remove it (a directory raises)
    tag = f"{os.getpid()}_{threading.get_ident()}"
    src_path = cache / f"{prefix}_{key}_{tag}{ext}"
    tmp_so = cache / f"{prefix}_{key}_{tag}.so.tmp"
    src_path.write_text(src, encoding="utf-8")
    full = [*cmd, str(src_path), "-o", str(tmp_so), *link]
    try:
        subprocess.run(full, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tmp_so.unlink(missing_ok=True)
        raise RuntimeError(
            f"{who}: compile failed:\n$ {' '.join(full)}\n{exc.stderr}") from None
    finally:
        src_path.unlink(missing_ok=True)
    if os.name == "posix":
        os.chmod(tmp_so, 0o700)     # the linker's mode follows the umask
    os.replace(tmp_so, so_path)
    return so_path
