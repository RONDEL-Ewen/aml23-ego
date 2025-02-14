from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import torch
import gc
import os

with open('./S04/emg_data/S04_left.pkl', 'rb') as f:
    left_data = pickle.load(f)
with open('./S04/emg_data/S04_right.pkl', 'rb') as f:
    right_data = pickle.load(f)

def find_matching_dict(
    target_dict,
    data_list
):
    """
    Find the corresponding pair
    """

    for d in data_list:
        if (d['label'] == target_dict['label'] and
                round(d['start_time']) == round(target_dict['start_time'])):
            return d
    return None

combined_data = []

for left_dict in left_data:

    matching_right_dict = find_matching_dict(left_dict, right_data)

    if matching_right_dict is not None:
        combined_dict = {
            'label': left_dict['label'],
            'emg_left': left_dict['data'],
            'emg_right': matching_right_dict['data'],
            'start_time': max(left_dict['start_time'], matching_right_dict['start_time']),
            'end_time': min(left_dict['end_time'], matching_right_dict['end_time'])
        }
        combined_data.append(combined_dict)

del left_data
del right_data
gc.collect()

filenames = [
    './S04/frames/PG1_frames_timestamps.pkl',
    './S04/frames/PG2_frames_timestamps.pkl',
    './S04/frames/PG3_frames_timestamps.pkl',
    './S04/frames/PG4_frames_timestamps.pkl',
    './S04/frames/PG5_frames_timestamps.pkl'
]

all_frames_data = []

for filename in filenames:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        all_frames_data.extend(data)

del data
gc.collect()

def select_uniform_frames(
    frames_data,
    start_time,
    end_time,
    frames_per_5sec = 5
):
    
    duration = end_time - start_time
    num_frames = int(np.ceil(duration / 5.0) * frames_per_5sec)
    
    relevant_frames = [frame for frame in frames_data if start_time <= frame['timestamp'] <= end_time]
    
    indices = np.linspace(0, len(relevant_frames) - 1, num_frames, dtype=int)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    frame_vectors = []
    for i in indices:
        frame = relevant_frames[i]
        with Image.open(frame['frame']).convert('RGB') as image:
            tensor_image = transform(image)
            frame_vectors.append(tensor_image)
    
    return frame_vectors

for item in combined_data:
    item['RGB_frames'] = select_uniform_frames(all_frames_data, item['start_time'], item['end_time'])

del all_frames_data
gc.collect()

output_filename = "./S04/aggregated_data.pkl"
with open(output_filename, 'wb') as output_file:
    pickle.dump(combined_data, output_file)

print(f"Data saved in: {output_filename}")