#test_frame_range.py
# ----------------------------------------------------------------------------
# Script used to extract a frame range from a video using FFmpeg, OpenCV, and Celux, and compare the results.
# ----------------------------------------------------------------------------

import cv2
import subprocess
import os
import sys

import matplotlib.pyplot as plt

# Adjust if needed to locate your Celux package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from celux import VideoReader

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
video_path = os.path.join(os.path.dirname(__file__), "data", "default", "BigBuckBunny.mp4")

# Frame range you want to extract
start_frame = 60
end_frame   = 120  # non-inclusive

output_clip_ffmpeg_frames = "tests/data/tests/clip_ffmpeg_frames_60_120.mp4"
output_clip_opencv_frames = "tests/data/tests/clip_opencv_frames_60_120.mp4"
output_clip_celux_frames  = "tests/data/tests/clip_celux_frames_60_120.mp4"


# ----------------------------------------------------------------------------
# FFmpeg: Extract frames [start_frame, end_frame) by frame count
# ----------------------------------------------------------------------------
def extract_clip_ffmpeg_frames(video_path, start_frame, end_frame, output_video):
    """
    Extract frames [start_frame, end_frame) using FFmpeg's select filter.
    NOTE: This re-encodes the video. If you want to copy streams,
    frame-accurate copying can be more complicated.
    """
    # We use the select filter:  select='between(n,start_frame,end_frame-1)'
    # Then 'setpts=N/FRAME_RATE/TB' ensures proper timestamps in the output.
    # If you prefer a different codec, tweak '-c:v libx264 -crf 18'.
    # '-an' is used if you want to drop audio, or you can keep it if you like.
    filter_str = f"select='between(n,{start_frame},{end_frame-1})',setpts=N/FRAME_RATE/TB"

    cmd = [
        "ffmpeg", 
        "-i", video_path,
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "18",   # re-encode video
        "-an",                              # drop audio (optional)
        output_video,
        "-y",            # overwrite output
        "-hide_banner",
        "-loglevel", "error"
    ]
    subprocess.run(cmd)
    print(f"[FFmpeg] Extracted frames [{start_frame}, {end_frame}) → {output_video}")


# ----------------------------------------------------------------------------
# OpenCV: Extract frames [start_frame, end_frame) by reading the file
# ----------------------------------------------------------------------------
def extract_clip_opencv_frames(video_path, start_frame, end_frame, output_video):
    """
    Extract frames [start_frame, end_frame) by manually reading frames using OpenCV.
    We'll write them to an MP4 file (H.264) at the input's original FPS.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Frame width/height
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
    print(f"[OpenCV] Extracted frames [{start_frame}, {end_frame}) → {output_video}")


# ----------------------------------------------------------------------------
# Celux: Extract frames [start_frame, end_frame) with VideoReader slicing
# ----------------------------------------------------------------------------
def extract_clip_celux_frames(video_path, start_frame, end_frame, output_video):
    """
    Extract frames [start_frame, end_frame) using Celux VideoReader's bracket notation,
    which interprets integer arguments as frame indices.
    """
    reader = VideoReader(video_path)

    # The bracket operator [start, end] sets a frame-based range
    # because we're passing integers. It decodes that exact slice.
    reader([start_frame, end_frame])

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
    print(f"[Celux] Extracted frames [{start_frame}, {end_frame}) → {output_video}")


# ----------------------------------------------------------------------------
# Optional: Simple Demo to Show Frame Count in Results
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
    extract_clip_ffmpeg_frames(
        video_path, start_frame, end_frame, output_clip_ffmpeg_frames
    )

    # OpenCV
    extract_clip_opencv_frames(
        video_path, start_frame, end_frame, output_clip_opencv_frames
    )

    # Celux
    extract_clip_celux_frames(
        video_path, start_frame, end_frame, output_clip_celux_frames
    )

    # Optional: Check how many frames each output has
    print()
    print("[INFO] FFmpeg clip frame count  :", count_frames_in_video(output_clip_ffmpeg_frames))
    print("[INFO] OpenCV clip frame count  :", count_frames_in_video(output_clip_opencv_frames))
    print("[INFO] Celux clip frame count   :", count_frames_in_video(output_clip_celux_frames))
