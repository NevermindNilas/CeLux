import os
import subprocess
import logging

# Define the test video formats
TEST_VIDEOS = [
    ("output_yuv420p8le.mp4", "yuv420p", "libx264"),
    ("output_yuv420p10le.mp4", "yuv420p10le", "libx265"),
    ("output_yuv420p12le.mp4", "yuv420p12le", "libx265"),
    ("output_yuv422p8le.mp4", "yuv422p", "libx264"),
    ("output_yuv422p10le.mp4", "yuv422p10le", "libx265"),
    ("output_yuv422p12le.mp4", "yuv422p12le", "libx265"),
    ("output_yuv444p8le.mp4", "yuv444p", "libx264"),
    ("output_yuv444p10le.mp4", "yuv444p10le", "libx265"),
    ("output_yuv444p12le.mp4", "yuv444p12le", "libx265"),
    ("output_rawvideo.yuv", "yuv420p", "rawvideo"),
    ("output_prores422.mov", "yuv422p10le", "prores_ks"),
    ("output_prores4444.mov", "yuv444p10le", "prores_ks"),
    ("output_rgb24.mp4", "rgb24", "libx264"),
    ("output_nv12.mp4", "nv12", "libx264"),
]

# Define the test video path
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_test_videos():
    """Generate missing test videos using FFmpeg."""
    for filename, pix_fmt, codec in TEST_VIDEOS:
        output_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(output_path):
            logging.info(f"Generating {filename}...")
            ffmpeg_cmd = [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"testsrc=duration=3:size=1920x1080:rate=24",
                "-pix_fmt", pix_fmt,
                "-c:v", codec,
                "-y", output_path  # Overwrite if exists
            ]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    logging.info("Test videos generated successfully.")
            #logging.info(f"{filename} already exists, skipping.")

if __name__ == "__main__":
    generate_test_videos()
