"""Repo paths, and the one cross-drive-safe `relpath`.

WHY THIS MODULE EXISTS
----------------------
`os.path.relpath` RAISES on Windows when the two paths are on different drive
letters:

    >>> os.path.relpath(r"C:\\Temp\\ep.npz", r"D:\\Thesis Project")
    ValueError: path is on mount 'C:', start on mount 'D:'

It is pure string arithmetic - neither path has to exist - so the failure does
not need an exotic setup to reach. A temp directory, a scratch path, an `--out`
on an external disk holding a backup of the collection, or a machine whose repo
is on D: while `TEMP` is on C: all produce it. The crash then happens while
FORMATTING A MESSAGE, so the operation that failed is never the operation that
was being done, and the traceback points at a print statement.

THIS FIX HAD BEEN WRITTEN THREE TIMES, SEPARATELY
-------------------------------------------------
Once in the teleop episode writer, once (reported, still missing) in the scripted
recorder's ledger write, and once as a private `_rel` in the Phase 4 loader.
Three independent copies of one fix is this repo's characteristic bug: the next
call site starts from `os.path.relpath` again, because that is what the standard
library offers and nothing says otherwise. So there is now exactly one function,
every site imports it, and a new site that reaches for `os.path.relpath` is
visible in a grep.

`repo_relpath` is TOTAL: it never raises. A path it cannot express relatively
comes back absolute, which is correct for the thing every caller wanted - a
short, readable path for a human or a ledger row - and a ledger row recording an
absolute path on another drive is accurate rather than broken.
"""
from __future__ import annotations

import os

#: The repository root: the directory holding g1_data/, g1_model/, g1_teleop/.
ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_relpath(path: str, start: str = ROOT) -> str:
    """`path` relative to `start`, or absolute when that is not expressible.

    Never raises. The fallback is `abspath`, not the input, so the result is
    always a usable path rather than whatever relative fragment was passed in.
    """
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.abspath(path)
