import os
import sys
import time
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Append the parent directory to system path to allow package import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Configure logging for clean output
logging.basicConfig(level=logging.INFO, format="%(message)s")

import celux
celux.set_log_level(celux.LogLevel.off)
def get_opencv_frame(video_path, seek_value, seek_by="time"):
    """
    Extract a frame using OpenCV, either by time (seconds) or frame index.

    Args:
        video_path (str): Path to the video file.
        seek_value (float or int): Timestamp (seconds) or frame index.
        seek_by (str): "time" for timestamp, "frame" for frame index.

    Returns:
        frame (np.array): Extracted frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("❌ OpenCV failed to open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if seek_by == "time":
        frame_number = int(seek_value * fps)  # Convert time to frame number
    else:
        frame_number = seek_value

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logging.error(f"❌ OpenCV failed to seek to {seek_value} ({seek_by}).")
        return None

    return frame




def get_celux_frame(video_path, seek_value):
    """
    Extract a frame using CeLux by timestamp or frame index.

    Args:
        video_path (str): Path to the video file.
        seek_value (float or int): Timestamp (seconds) or frame index.

    Returns:
        frame (torch.Tensor): Extracted frame.
    """
    reader = celux.VideoReader(video_path)
    
    try:
        frame = reader[seek_value]  # CeLux supports both float (time) & int (frame index)
        return frame.numpy()  # Convert to NumPy for visualization
    except Exception as e:
        logging.error(f"❌ CeLux failed to seek to {seek_value}: {e}")
        return None


def compare_seeking(video_path, seek_values, seek_by="time"):
    """
    Compare frame extraction speed and accuracy between CeLux and OpenCV.

    Args:
        video_path (str): Path to the video file.
        seek_values (list): List of timestamps (float) or frame indexes (int).
        seek_by (str): "time" or "frame".
    """
    logging.info(f"🔍 Comparing CeLux vs OpenCV ({seek_by})...")

    for seek_value in seek_values:
        logging.info(f"⏩ Seeking to {seek_value} ({seek_by})...")

        # Benchmark OpenCV
        start_time = time.time()
        frame_opencv = get_opencv_frame(video_path, seek_value, seek_by)  # Pass correct seek mode
        time_opencv = time.time() - start_time

        # Benchmark CeLux
        start_time = time.time()
        frame_celux = get_celux_frame(video_path, seek_value)
        time_celux = time.time() - start_time

        if frame_opencv is None or frame_celux is None:
            logging.warning(f"⚠️ Skipping {seek_value} ({seek_by}) due to missing frame.")
            continue

        # Resize both frames to 256x256 for easier side-by-side comparison
        frame_opencv_resized = cv2.resize(frame_opencv, (256, 256))
        frame_celux_resized = cv2.resize(frame_celux, (256, 256))

        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(cv2.cvtColor(frame_opencv_resized, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"OpenCV ({time_opencv:.3f}s)")

        axes[1].imshow(frame_celux_resized)
        axes[1].set_title(f"CeLux ({time_celux:.3f}s)")

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"Comparison at {seek_value} ({seek_by})")
        plt.show()

        logging.info(f"✅ Comparison at {seek_value}: OpenCV {time_opencv:.3f}s | CeLux {time_celux:.3f}s")


if __name__ == "__main__":
    # Default test video
    default_video = os.path.join(os.path.dirname(__file__), "..", "data", "default", "BigBuckBunny.mp4")
    input_video = sys.argv[1] if len(sys.argv) > 1 else default_video

    if not os.path.exists(input_video):
        logging.error(f"❌ Missing test video: {input_video}")
        sys.exit(1)

    # Test seeking
    seek_times = [1.0, 2.5, 5.0, 10.0, 60.0]   # Seconds
    seek_frames = [10, 50, 100, 200]  # Frame numbers

    compare_seeking(input_video, seek_times, "time")
    compare_seeking(input_video, seek_frames, "frame")


