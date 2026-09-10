"""Editable-install loader for the canonical :mod:`scripts.kscore` source."""

from ._editable import load_script

load_script(globals(), "kscore.py")
