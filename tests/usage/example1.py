import os
import sys
import time
import logging

# Append the parent directory to system path to allow package import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Configure logging for clean output
logging.basicConfig(level=logging.INFO, format="%(message)s")

import celux

def test_video(input_video):
    """
    Test the VideoReader class with a sample video.
    
    Args:
        input_video (str): Path to the input video file.
    """
    if not os.path.exists(input_video):
        logging.error(f"❌ Missing test video: {input_video}")
        return

    try:
        logging.info(f"🔍 Loading video: {input_video}")

        # Initialize the video reader
        reader = celux.VideoReader(input_video)
        print(reader.pixel_format)
        print(reader.properties)
        # Decode the first frame
        first_frame = next(iter(reader))
        if first_frame is None:
            logging.error("❌ Failed to decode first frame")
            return
        
        logging.info("✅ Video loaded successfully.")

        # Benchmark frame decoding speed
        num_frames = 100  # Adjust this for more accuracy
        start_time = time.time()
        
        for _ in range(num_frames):
            _ = next(iter(reader), None)
        
        elapsed_time = time.time() - start_time
        fps = num_frames / elapsed_time if elapsed_time > 0 else 0

        logging.info(f"⚡ Benchmark: {fps:.2f} FPS for {num_frames} frames.")

    except Exception as e:
        logging.error(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    # Allow passing a video file as an argument or use a default test video
    default_video = os.path.join(os.path.dirname(__file__), "..", "data", "default", "BigBuckBunny.mp4")
    input_video = sys.argv[1] if len(sys.argv) > 1 else default_video

    test_video(input_video)
