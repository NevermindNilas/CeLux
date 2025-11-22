import time
import celux
import subprocess
import os

SOURCE_FILE = "benchmark_source.mp4"
WIDTH = 1920
HEIGHT = 1080
FRAMES = 300


def generate_source():
    if os.path.exists(SOURCE_FILE):
        return
    print(f"Generating source video: {SOURCE_FILE}...")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration=10:size={WIDTH}x{HEIGHT}:rate=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(FRAMES),
        SOURCE_FILE,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def benchmark(use_libyuv):
    reader = celux.VideoReader(SOURCE_FILE)
    # Check if set_libyuv_enabled exists (it should if build is updated)
    if hasattr(reader, "set_libyuv_enabled"):
        reader.set_libyuv_enabled(use_libyuv)
    else:
        print("Warning: set_libyuv_enabled not found on VideoReader")
        return

    start = time.time()
    count = 0
    for frame in reader:
        count += 1
    end = time.time()

    duration = end - start
    fps = count / duration
    print(
        f"libyuv={'ON ' if use_libyuv else 'OFF'}: {count} frames in {duration:.4f}s ({fps:.2f} fps)"
    )


if __name__ == "__main__":
    generate_source()
    print("Warming up...")
    benchmark(True)

    print("\nRunning Benchmark...")
    benchmark(False)
    benchmark(True)
