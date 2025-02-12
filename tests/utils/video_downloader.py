import os
import logging
import requests
from requests.exceptions import RequestException

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define test video URLs
TEST_VIDEOS = {
    "lite": {
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "filename": "ForBiggerBlazes.mp4",
    },
    "full": {
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "filename": "BigBuckBunny.mp4",
    },
}

# Define storage path for test videos
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure data directory exists

def get_video(mode="full"):
    """
    Fetches the test video based on the specified mode.
    
    Args:
        mode (str): "lite" for smaller test video, "full" for full-size video.
    
    Returns:
        str: Path to the downloaded video file.
    """
    if mode not in TEST_VIDEOS:
        raise ValueError("Invalid mode. Choose 'lite' or 'full'.")

    video_info = TEST_VIDEOS[mode]
    video_path = os.path.join(DATA_DIR, video_info["filename"])

    if not os.path.exists(video_path):
        download_video(video_info["url"], video_path)
   # else:
       # logging.info(f"Video already exists: {video_path}")

    return video_path

def download_video(url, output_path):
    """
    Downloads a video from the given URL to the specified output path.

    Args:
        url (str): The URL of the video to download.
        output_path (str): The path where the video will be saved.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as file:
            logging.info(f"Downloading {url} to {output_path}")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        logging.info(f"Download completed: {output_path}")

    except RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download test videos for CeLux.")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["lite", "full"],
        help="Choose 'lite' for a small test video or 'full' for a larger test video."
    )
    args = parser.parse_args()

    video_path = get_video(args.mode)
    logging.info(f"Test video available at: {video_path}")
