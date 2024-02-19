import pickle
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

def main():

    args = get_args()

    frames_dir = args.get('frames_dir', './S04/frames/PG1')
    video = args.get('video', '/PG1')
    timestamps_file = frames_dir + '/' + args.get('timestamps_file', 'timestamps.txt')
    output_file = frames_dir + '/' + args.get('output_file', 'PG1_frames_timestamps.pkl')

    # Step 1: Read the timestamps and clean them
    with open(timestamps_file, "r") as f:
        timestamps = [float(".".join(line.strip().split())) for line in f]

    # Step 2: Generate all frames path (assuming there is exactly one per timestamp)
    frames_paths = [os.path.join(frames_dir + video, f"frame_{i:010d}.jpg") for i in range(1, len(timestamps) + 1)]

    # Step 3: Create all dictionaries
    frames_timestamps = [{'timestamp': ts, 'frame': path} for ts, path in zip(timestamps, frames_paths)]

    # Step 4: Save the dictionaries in a '.pkl' file
    with open(output_file, "wb") as f:
        pickle.dump(frames_timestamps, f)

    print(f"Data saved in {output_file}.")

if __name__ == '__main__':
    main()