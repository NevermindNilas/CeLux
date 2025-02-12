# test_color_fmts.py
# ----------------------------------------------------------------------------
# Script used to test the color formats supported by CeLux.
# Prints a summary of supported and unsupported formats after running the tests.
# ----------------------------------------------------------------------------

import pytest
import os
import sys
import logging
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import celux  # your library
from tests.utils.generate_test_videos import generate_test_videos

# Set up logging with a cleaner format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Directory holding test videos
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "color_fmts")

# Ensure test videos exist before running tests
@pytest.fixture(scope="session", autouse=True)
def setup_videos():
    """Generate test videos, but suppress duplicate logs if already generated."""
    generate_test_videos()


# Dynamically gather all files in DATA_DIR
VIDEO_FILES = sorted(
    f for f in os.listdir(DATA_DIR)
    if os.path.isfile(os.path.join(DATA_DIR, f))
    # you can also filter by extension if desired, e.g.:
    # and (f.endswith(".mp4") or f.endswith(".mov") or f.endswith(".mkv"))
)

# Track results for a final summary
SUPPORTED_FORMATS = []
UNSUPPORTED_FORMATS = []

@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_video_format_loading(video_file):
    """Test whether CeLux can load different video color formats."""
    video_path = os.path.join(DATA_DIR, video_file)

    # Ensure file exists
    assert os.path.exists(video_path), f"❌ Missing test video: {video_file}"

    try:
        reader = celux.VideoReader(video_path)
        logging.info(f"🔍 Loading video: {video_file}")
        # pixel_format is a property of the VideoReader class
        logging.info(f"    - Pixel Format: {reader.pixel_format}")

        first_frame = next(iter(reader))
        assert first_frame is not None, "❌ Failed to decode first frame"

        # If successful, mark it as supported
        SUPPORTED_FORMATS.append(video_file)
        cv2.imshow(F"Video: {video_file}", first_frame.numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"❌ Error reading {video_file}: {e}")
        UNSUPPORTED_FORMATS.append((video_file, str(e)))


@pytest.fixture(scope="session", autouse=True)
def print_summary(request):
    """Prints a summary of supported and unsupported formats after tests run."""
    def summary():
        logging.info("\n" + "=" * 60)
        logging.info("\033[92m✅ Supported Formats:\033[0m")  # Green color
        for fmt in SUPPORTED_FORMATS:
            # remove the "output_" prefix and file extension if you like
            shortname = fmt
            if shortname.startswith("output_"):
                shortname = shortname[7:]
            shortname = os.path.splitext(shortname)[0]
            logging.info(f"    - {shortname}")

        if UNSUPPORTED_FORMATS:
            logging.info("\n\033[91m❌ Unsupported Formats:\033[0m")  # Red color
            for fmt, reason in UNSUPPORTED_FORMATS:
                shortname = fmt
                if shortname.startswith("output_"):
                    shortname = shortname[7:]
                shortname = os.path.splitext(shortname)[0]
                logging.info(f"    - {shortname}  (Reason: {reason})")

        logging.info("=" * 60 + "\n")

    request.addfinalizer(summary)


if __name__ == "__main__":
    pytest.main([__file__])
