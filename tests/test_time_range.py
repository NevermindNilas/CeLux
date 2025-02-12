# test_time_range.py
# ----------------------------------------------------------------------------
# Script used to extract a time range from a video using FFmpeg, OpenCV, and Celux, and compare the results.
# ----------------------------------------------------------------------------

import cv2
import subprocess
import os
import sys
import matplotlib.pyplot as plt

# Adjust if needed to locate your Celux package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from celux import VideoReader
import celux
celux.set_log_level(celux.LogLevel.off)  # trace, debug, info, error, warn, critical, off
# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
video_path = os.path.join(os.path.dirname(__file__), "data", "default", "BigBuckBunny.mp4")

# Time range you want to extract, in seconds
start_time = 5.8391
end_time   = 6.256 # non-inclusive

output_clip_ffmpeg_times = "tests/data/tests/clip_ffmpeg_times_2.5_5.5.mp4"
output_clip_opencv_times = "tests/data/tests/clip_opencv_times_2.5_5.5.mp4"
output_clip_celux_times  = "tests/data/tests/clip_celux_times_2.5_5.5.mp4"


# ----------------------------------------------------------------------------
# FFmpeg: Extract [start_time, end_time) in seconds
# ----------------------------------------------------------------------------
def extract_clip_ffmpeg_times(video_path, start_time, end_time, output_video):
    """
    Extract [start_time, end_time) using FFmpeg's -ss and -to options in seconds.
    Note: -c copy can be problematic if you want partial-GOP segments, so we re-encode.
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_time),     # Seek to start_time
        "-to", str(end_time),       # End time
        "-c:v", "libx264", "-crf", "18",  # Re-encode video
        "-an",                           # Drop audio (optional)
        output_video,
        "-y",             # overwrite output
        "-hide_banner",
        "-loglevel", "error"
    ]
    subprocess.run(cmd)
    print(f"[FFmpeg] Extracted clip [{start_time}, {end_time}) → {output_video}")


# ----------------------------------------------------------------------------
# OpenCV: Extract [start_time, end_time) in seconds
# ----------------------------------------------------------------------------
def extract_clip_opencv_times(video_path, start_time, end_time, output_video):
    """
    Extract [start_time, end_time) by converting to frames using OpenCV,
    i.e. frame range = [start_sec * fps, end_sec * fps).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_time * fps)
    end_frame   = int(end_time * fps)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Jump to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[OpenCV] Extracted clip [{start_time}, {end_time}) → {output_video}")


# ----------------------------------------------------------------------------
# Celux: Extract [start_time, end_time) in seconds
# ----------------------------------------------------------------------------
def extract_clip_celux_times(video_path, start_time, end_time, output_video):
    """
    Extract [start_time, end_time) using Celux VideoReader's bracket notation with floats,
    which interprets floats as seconds.
    """
    reader = VideoReader(video_path)

    # Passing floats => time-based range
    # That means decode from start_time s up to end_time s
    reader([start_time, end_time])

    fps    = reader.fps
    width  = reader.width
    height = reader.height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame in reader:
        if frame is None:
            break
        out.write(frame.numpy())

    out.release()
    print(f"[Celux] Extracted clip [{start_time}, {end_time}) → {output_video}")


# ----------------------------------------------------------------------------
# Optional: Count Frames in Output Videos
# ----------------------------------------------------------------------------
def count_frames_in_video(filepath):
    cap = cv2.VideoCapture(filepath)
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # FFmpeg
    extract_clip_ffmpeg_times(
        video_path, start_time, end_time, output_clip_ffmpeg_times
    )

    # OpenCV
    extract_clip_opencv_times(
        video_path, start_time, end_time, output_clip_opencv_times
    )

    # Celux
    extract_clip_celux_times(
        video_path, start_time, end_time, output_clip_celux_times
    )

    # Optional: Check how many frames each output has
    print()
    ffmpeg_count = count_frames_in_video(output_clip_ffmpeg_times)
    opencv_count = count_frames_in_video(output_clip_opencv_times)
    celux_count  = count_frames_in_video(output_clip_celux_times)
    print("[INFO] FFmpeg clip frame count  :", ffmpeg_count)
    print("[INFO] OpenCV clip frame count  :", opencv_count)
    print("[INFO] Celux clip frame count   :", celux_count)
