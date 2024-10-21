"""
This test is to check the performance of the VideoReader class with torch frames.
Should be useful inside Github Actions to keep track of the performance of the VideoReader class.
"""

import time
import argparse
import logging
import requests
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import celux_cuda as cx
cx.set_log_level(cx.LogLevel.trace)
STREAM = torch.cuda.Stream("cuda")

from requests.exceptions import RequestException

# Nicer prints
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def getVideo(mode):
    if mode == "lite":
        videoUrl = r"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
        videoPath = os.path.join(os.getcwd(), "ForBiggerBlazes.mp4")
    else:
        videoUrl = r"https://commondatastorage.googleapis.com/gtv-videos-buckett/sample/BigBuckBunny.mp4"
        videoPath = os.path.join(os.getcwd(), "BigBuckBunny.mp4")
        
    if not os.path.exists(videoPath):
        downloadVideo(videoUrl, videoPath)
    else:
        logging.info(f"Video already exists at {videoPath}")
        

def downloadVideo(url, outputPath):
    """
    Downloads a video from the given URL to the specified output path.

    Args:
        url (str): The URL of the video to download.
        outputPath (str): The path where the video will be saved.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(outputPath, "wb") as file:
            logging.info(f"Downloading {url} to {outputPath}")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise


def processVideoCuda(videoPath):
    """
    Processes the video to count frames and measure performance.

    Args:
        videoPath (str): The path to the video file.
    """
    try:
        frameCount = 0
        start = time.time()
        with cx.VideoReader(videoPath, device = "cuda" , d_type="uint8", stream = STREAM) as reader:
            writer = cx.VideoWriter("./output_cuda.mp4", reader.get_properties()["width"],
                                reader.get_properties()["height"], reader.get_properties()["fps"],
                                device = "cuda")
            for frame in reader:
                if frameCount == 0:
                    logging.info(
                        f"Frame data: {frame.shape, frame.dtype, frame.device}"
                    )
                writer(frame)
                frameCount += 1
        end = time.time()
        logging.info(f"Time taken: {end-start} seconds")
        logging.info(f"Total Frames: {frameCount}")
        logging.info(f"FPS: {frameCount/(end-start)}")
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise


def processVideoCPU(videoPath):
    """
    Processes the video to count frames and measure performance.

    Args:
        videoPath (str): The path to the video file.
    """
    try:
        frameCount = 0
        start = time.time()
        with cx.VideoReader(videoPath, device = "cpu", d_type="uint8") as reader:
            writer = cx.VideoWriter("./output_cpu.mp4", reader.get_properties()["width"],
                                reader.get_properties()["height"], reader.get_properties()["fps"],
                                device = "cpu")
            for frame in reader:
                if frameCount == 0:
                    logging.info(
                        f"Frame data: {frame.shape, frame.dtype, frame.device}"
                    )
                writer(frame.clone().cpu())
                frameCount += 1
        
        end = time.time()
        logging.info(f"Time taken: {end-start} seconds")
        logging.info(f"Total Frames: {frameCount}")
        logging.info(f"FPS: {frameCount/(end-start)}")
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise


def main(args):
    if args.mode == "lite":
        videoUrl = r"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
        videoPath = os.path.join(os.getcwd(), "ForBiggerBlazes.mp4")
    else:
        videoUrl = r"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        videoPath = os.path.join(os.getcwd(), "BigBuckBunny.mp4")

    if not os.path.exists(videoPath):
        downloadVideo(videoUrl, videoPath)
    else:
        logging.info(f"Video already exists at {videoPath}")


    
    logging.info("Processing video with CUDA")
    processVideoCuda(videoPath)
    
    
    print("")
    
    logging.info("Processing video with CPU")
    processVideoCPU(videoPath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add 'lite' for github actions and 'full' for local testing
    parser.add_argument(
        "--mode",
        type=str,
        default="lite",
        help="The mode to run the test in. 'lite' for github actions and 'full' for local testing.",
        choices=["lite", "full"],
    )
    args = parser.parse_args()

    main(args)


#new function designed to get the video based on arg passed
