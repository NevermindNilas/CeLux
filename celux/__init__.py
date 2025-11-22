# celux/__init__.py

import os
import sys

# On Windows, we must add the package directory to the DLL search path
# so that the bundled DLLs (ffmpeg, libyuv, etc.) can be found.
if os.name == "nt":
    package_dir = os.path.dirname(os.path.abspath(__file__))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(package_dir)
    else:
        os.environ["PATH"] = package_dir + ";" + os.environ["PATH"]

import torch
from ._celux import (
    __version__,
    VideoReader,
    VideoEncoder,
    Audio,
    set_log_level,
    LogLevel,
)

__all__ = [
    "__version__",
    "VideoReader",
    "VideoEncoder",
    "Audio",
    "set_log_level",
    "LogLevel",
]
