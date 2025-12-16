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
    __cuda_support__,
    VideoReader as _VideoReaderBase,
    VideoEncoder,
    Audio,
    set_log_level,
    LogLevel,
)
from .batch import BatchMixin


# Create enhanced VideoReader with batch support
class VideoReader(BatchMixin, _VideoReaderBase):
    """
    VideoReader with batch frame reading support.
    
    Inherits from BatchMixin to provide efficient batch decoding capabilities
    while maintaining all original VideoReader functionality.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store decoder reference for batch operations
        self._decoder = self


__all__ = [
    "__version__",
    "__cuda_support__",
    "VideoReader",
    "VideoEncoder",
    "Audio",
    "set_log_level",
    "LogLevel",
]
