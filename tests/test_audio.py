# test_audio.py
# ----------------------------------------------------------------------------
# Script to test audio extraction and tensor conversion in CeLux.
# Optional playback of the extracted audio tensor using --playback.
# ----------------------------------------------------------------------------

import pytest
import os
import sys
import logging
import torch
import numpy as np
import sounddevice as sd  # For audio playback
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import celux  # your library

# Set up logging with a cleaner format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Directory holding test videos
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "default")

# Parse command-line arguments for playback
parser = argparse.ArgumentParser(description="Test CeLux audio functions with optional playback.")
parser.add_argument("--playback", action="store_true", help="Play back the extracted audio tensor")
args, unknown = parser.parse_known_args()

# Ensure test videos exist before running tests
@pytest.fixture(scope="session", autouse=True)
def setup_videos():
    """Ensure test videos are available before running tests."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.warning("⚠️  Test video directory did not exist. Created empty folder.")

# Dynamically gather all files in DATA_DIR
VIDEO_FILES = sorted(
    f for f in os.listdir(DATA_DIR)
    if os.path.isfile(os.path.join(DATA_DIR, f))
)

# Track results for final summary
PASSED_TESTS = []
FAILED_TESTS = []

@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_audio_extraction(video_file):
    """Test extracting audio from video to a file."""
    video_path = os.path.join(DATA_DIR, video_file)
    output_audio_path = os.path.join(DATA_DIR, f"{os.path.splitext(video_file)[0]}.wav")

    assert os.path.exists(video_path), f"❌ Missing test video: {video_file}"

    try:
        reader = celux.VideoReader(video_path)

        if not reader.has_audio:
            logging.warning(f"⚠️  No audio stream found in {video_file}")
            FAILED_TESTS.append((video_file, "No audio stream"))
            return

        audio = reader.audio
        logging.info(f"🔍 Extracting audio from: {video_file} → {output_audio_path}")
        success = audio.file(output_audio_path)

        assert success and os.path.exists(output_audio_path), "❌ Audio extraction failed"
        PASSED_TESTS.append(video_file)
    except Exception as e:
        logging.error(f"❌ Error extracting audio from {video_file}: {e}")
        FAILED_TESTS.append((video_file, str(e)))


@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_audio_tensor(video_file):
    """Test extracting audio as a PyTorch tensor."""
    video_path = os.path.join(DATA_DIR, video_file)

    assert os.path.exists(video_path), f"❌ Missing test video: {video_file}"

    try:
        reader = celux.VideoReader(video_path)

        if not reader.has_audio:
            logging.warning(f"⚠️  No audio stream found in {video_file}")
            FAILED_TESTS.append((video_file, "No audio stream"))
            return

        audio = reader.audio
        logging.info(f"🔍 Extracting audio tensor from: {video_file}")
        tensor = audio.tensor()

        assert tensor is not None and tensor.numel() > 0, "❌ Audio tensor extraction failed"
        logging.info(f"    - Audio Tensor Shape: {tensor.shape}")

        num_channels = audio.channels
        reshaped_tensor = tensor.view(num_channels, -1)
        logging.info(f"    - Reshaped Tensor Shape: {reshaped_tensor.shape}")

        assert reshaped_tensor.shape[0] == num_channels, "❌ Audio tensor shape mismatch"
        PASSED_TESTS.append(video_file)

        # Play the audio if the --playback flag is used
        if args.playback:
            play_audio(reshaped_tensor, audio.sample_rate, num_channels)

    except Exception as e:
        logging.error(f"❌ Error extracting audio tensor from {video_file}: {e}")
        FAILED_TESTS.append((video_file, str(e)))


def play_audio(tensor, sample_rate, num_channels):
    """Plays back the audio from a PyTorch tensor using sounddevice."""
    logging.info("🔊 Playing extracted audio...")

    # Convert tensor to NumPy array
    audio_np = tensor.numpy().astype(np.float32)

    # Normalize audio to [-1, 1] range
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np /= max_val

    # Transpose if stereo (shape: [channels, samples] → [samples, channels])
    if num_channels > 1:
        audio_np = np.transpose(audio_np)

    try:
        sd.play(audio_np, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        logging.error(f"❌ Audio playback failed: {e}")


@pytest.fixture(scope="session", autouse=True)
def print_summary(request):
    """Prints a summary of successful and failed audio tests after execution."""
    def summary():
        logging.info("\n" + "=" * 60)
        logging.info("\033[92m✅ Passed Audio Tests:\033[0m")  # Green color
        for video in PASSED_TESTS:
            logging.info(f"    - {video}")

        if FAILED_TESTS:
            logging.info("\n\033[91m❌ Failed Audio Tests:\033[0m")  # Red color
            for video, reason in FAILED_TESTS:
                logging.info(f"    - {video}  (Reason: {reason})")

        logging.info("=" * 60 + "\n")

    request.addfinalizer(summary)


if __name__ == "__main__":
    pytest.main([__file__])
