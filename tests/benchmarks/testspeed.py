import time
import cv2
import av
import torch
import sys
import os
# Adjust the path to include CeLux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from celux import VideoReader

VIDEO_PATH = r"C:\Users\tjerf\Downloads\TAS_fpqg.mp4"

def benchmark_opencv():
    cap = cv2.VideoCapture(VIDEO_PATH)
    start = time.time()
    frame_count = 0
    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()
    return frame_count / (time.time() - start)

def benchmark_pyav():
    container = av.open(VIDEO_PATH)
    stream = container.streams.video[0]
    start = time.time()
    frame_count = 0
    for frame in container.decode(stream):
        frame_count += 1
    return frame_count / (time.time() - start)

def benchmark_celux():
    reader = VideoReader(VIDEO_PATH)
    start = time.time()
    frame_count = 0
    for _ in reader:
        frame_count += 1
    return frame_count / (time.time() - start)

if __name__ == "__main__":
    print(f"OpenCV FPS: {benchmark_opencv():.2f}")
    print(f"PyAV FPS: {benchmark_pyav():.2f}")
    print(f"CeLux FPS: {benchmark_celux():.2f}")
