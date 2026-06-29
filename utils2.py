"""Notebook helper kept for backwards-compatible imports.

The LLM/embedding helpers that used to live here have been consolidated into
``utils.py`` (and, in cleaned form, ``src/rag/``). Only the subprocess-silencing
context manager remains, since the notebook imports it from this module.
"""
import subprocess
from contextlib import contextmanager


@contextmanager
def suppress_subprocess_output():
    """Silence stdout/stderr of any subprocess.Popen calls made within the block."""
    original_popen = subprocess.Popen

    def patched_popen(*args, **kwargs):
        kwargs['stdout'] = subprocess.DEVNULL
        kwargs['stderr'] = subprocess.DEVNULL
        return original_popen(*args, **kwargs)

    try:
        subprocess.Popen = patched_popen
        yield
    finally:
        subprocess.Popen = original_popen
