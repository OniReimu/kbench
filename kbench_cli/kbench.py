"""Editable-install loader for the canonical :mod:`scripts.kbench` source."""

from ._editable import load_script

load_script(globals(), "kbench.py")
