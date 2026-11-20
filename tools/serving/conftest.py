"""Lets the CPU tests import individual sglang modules by path without executing sglang/__init__.py.

Registers bare namespace stubs for sglang, sglang.srt, sglang.srt.speculative and sglang.srt.observability so that
`from sglang.srt.speculative.copy_proposer import ...` (used by step_profile.py) resolves to the checkout's file.
"""

import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PKG = os.path.join(os.environ.get("SGLANG_TEST_CHECKOUT", os.path.join(ROOT, "serving", "sglang")), "python", "sglang")


def _stub(name, path):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


for _name, _rel in (("sglang", ""), ("sglang.srt", "srt"), ("sglang.srt.speculative", "srt/speculative"), ("sglang.srt.observability", "srt/observability")):
    _stub(_name, os.path.join(PKG, _rel) if _rel else PKG)
