import numpy as np
import random
import pickle
import gc

def divide_actions(
    action
):

    nb_frames = len(action['RGB_frames'])
    nb_segments = nb_frames // 5
    
    total_duration = action['end_time'] - action['start_time']
    segment_duration = total_duration / nb_segments
    
    total_emg_values_per_segment = 100
    total_emg_values = total_emg_values_per_segment * nb_segments

    segments = []

    for i in range(nb_segments):

        segment = {
            'label': action['label'],
            'start_time': action['start_time'] + i * segment_duration,
            'end_time': action['start_time'] + (i + 1) * segment_duration,
            'RGB_frames': action['RGB_frames'][i*5:(i+1)*5]
        }
        
        indices_left = np.linspace(0, len(action['emg_left']), num=total_emg_values, endpoint=False, dtype=int)
        indices_right = np.linspace(0, len(action['emg_right']), num=total_emg_values, endpoint=False, dtype=int)
        
        segment_emg_indices = np.linspace(i * total_emg_values_per_segment, (i + 1) * total_emg_values_per_segment, num=total_emg_values_per_segment, endpoint=False, dtype=int)
        segment['emg_left'] = action['emg_left'][indices_left[segment_emg_indices]]
        segment['emg_right'] = action['emg_right'][indices_right[segment_emg_indices]]

        segments.append(segment)
    
    return segments

def main():

    with open('./S04/aggregated_data.pkl', 'rb') as f:
        data = pickle.load(f)

    divided_data = []
    for action in data:
        divided_data.extend(divide_actions(action))

    del data
    gc.collect()

    random.shuffle(divided_data)
    split_index = int(len(divided_data)*0.8)
    train_data = divided_data[:split_index]
    test_data = divided_data[split_index:]

    print(f"All data: {len(divided_data)}")
    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    del divided_data
    gc.collect()

    with open('./S04/train_data.pkl', 'wb') as output_file:
        pickle.dump(train_data, output_file)
    with open('./S04/test_data.pkl', 'wb') as output_file:
        pickle.dump(test_data, output_file)

if __name__ == '__main__':
    main()