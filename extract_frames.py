from datetime import datetime
import numpy as np
import subprocess
import pickle
import h5py
import cv2
import sys
import os

def get_args():
    """
    Parse 'name=value' command line arguments.
    """

    args = {}

    for arg in sys.argv[1:]:

        key, value = arg.split('=')
        args[key] = value

    return args

def video_to_frames(
    video_path: str,
    frames_folder: str
):
    """
    Cut a video in frames.
    """

    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    command = f"ffmpeg -i {video_path} -vf \"scale=456:256\" -q:v 2 {frames_folder}/frame_%010d.jpg"
    subprocess.run(command, shell = True)

def main():

    args = get_args()
    video_path = args.get('video_path', None)
    frames_folder = args.get('frames_folder', None)

    if video_path and frames_folder:
        video_to_frames(video_path, frames_folder)
    else:
        print("Missing one argument")

if __name__ == '__main__':
    main()