import celux
import subprocess
import numpy as np
import argparse
import os
import sys

def process_video(input_path, output_path):
    # 1. Initialize Reader (Decodes to RGB Full Range)
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    try:
        reader = celux.VideoReader(input_path)
    except Exception as e:
        print(f"Error opening video with Celux: {e}")
        return

    width = reader.width
    height = reader.height
    fps = reader.fps

    print(f"Processing {input_path} -> {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    # 2. Configure FFmpeg Command
    # Key Filter: scale=in_range=pc:out_range=tv:out_color_matrix=bt709
    # - in_range=pc: Tells FFmpeg input is 0-255 (Full)
    # - out_range=tv: Tells FFmpeg to compress to 16-235 (Limited)
    # - out_color_matrix=bt709: Uses HD standard coefficients
    command = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-vf', 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',     # Tag output metadata as TV Range
        '-colorspace', '1',      # Tag as BT.709
        '-color_primaries', '1', # Tag as BT.709
        '-color_trc', '1',       # Tag as BT.709
        output_path
    ]

    print(f"Running FFmpeg command: {' '.join(command)}")

    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # 3. Process Frames
    try:
        for i, frame in enumerate(reader):
            # Ensure frame is numpy uint8
            if hasattr(frame, "cpu"): frame = frame.cpu()
            if hasattr(frame, "numpy"): frame = frame.numpy()
            else: frame = np.array(frame)
            
            if frame.dtype != np.uint8:
                 frame = (frame * 255).astype(np.uint8)
            
            # Write raw bytes
            if process.stdin:
                process.stdin.write(frame.tobytes())
            
            if i % 30 == 0:
                print(f"Processed frame {i}...", end='\r')
                
    except BrokenPipeError:
        print("\nError: FFmpeg process disconnected.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if process.stdin:
            try:
                process.stdin.close()
            except:
                pass
        process.wait()
        print(f"\nFinished processing. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video using Celux -> FFmpeg correct color pipeline")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    
    args = parser.parse_args()
    process_video(args.input, args.output)
