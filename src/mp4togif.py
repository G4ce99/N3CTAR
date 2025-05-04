from moviepy.editor import VideoFileClip
import sys
import os
import argparse

def mp4_to_gif(input_path, output_path):
    try:
        # Load the video file
        clip = VideoFileClip(input_path)
        
        # Write the GIF file
        clip.write_gif(output_path, fps=10)
        print(f"GIF saved to {output_path}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 to GIF.")
    parser.add_argument("--input", "-i", type=str, help="Path to the input MP4 file.")
    parser.add_argument("--output", "-o", type=str, help="Path to save the output GIF file.")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist.")
        sys.exit(1)

    mp4_to_gif(input_path, output_path)