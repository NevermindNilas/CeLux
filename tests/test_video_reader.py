# We test the VideoReader class in this file. We test that it raises exceptions for invalid inputs.
import pytest
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import celux
from tests.utils.video_downloader import get_video

@pytest.fixture(scope="session")
def test_video():
    """Fetches and returns the path to a test video."""
    return get_video(mode="lite")

def test_invalid_video_path():
    """Test that VideoReader raises an exception for non-existent files."""
    with pytest.raises(RuntimeError, match=r"Failure Opening Input:: .*"):
        celux.VideoReader("invalid_path.mp4")

def test_invalid_format():
    """Test that VideoReader fails with an unsupported format."""
    with pytest.raises(RuntimeError, match=r"Failure Opening Input:: .*"):
        celux.VideoReader("tests/data/invalid_format.txt")

def test_empty_video():
    """Test that VideoReader fails with an empty file."""
    empty_file = "tests/data/empty.mp4"
    open(empty_file, "w").close()  # Create an empty file

    with pytest.raises(RuntimeError, match=r"Failure Opening Input:: .*"):
        celux.VideoReader(empty_file)

if __name__ == "__main__":
    pytest.main([__file__])
