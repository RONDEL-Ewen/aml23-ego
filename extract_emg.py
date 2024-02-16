import numpy as np
import pickle
import h5py
import sys

def get_args():
    """
    Parse 'name=value' command line arguments.
    """

    args = {}

    for arg in sys.argv[1:]:

        key, value = arg.split('=')
        args[key] = value

    return args

def read_hdf5_data(
    filepath,
    device_name,
    stream_name
):
    """
    Read and return data from an HDF5 file for a specified device and stream.
    """

    with h5py.File(filepath, 'r') as h5_file:

        data = np.array(h5_file[device_name][stream_name]['data'])
        times = np.squeeze(np.array(h5_file[device_name][stream_name]['time_s']))
    
    return data, times

def read_activity_data(
    filepath,
    device_name,
    stream_name
):
    """
    Extract activity data including labels, timestamps, and metadata.
    """

    with h5py.File(filepath, 'r') as h5_file:

        activities = [[x.decode('utf-8') for x in row] for row in h5_file[device_name][stream_name]['data']]
        times = np.squeeze(np.array(h5_file[device_name][stream_name]['time_s']))
    
    return activities, times

def process_activities(
    activities,
    times,
    exclude_bad = True
):
    """
    Process activity data to combine start/stop entries and exclude bad labels if specified.
    """

    processed_activities = []

    for i, (label, start_stop, validity, notes) in enumerate(activities):

        if exclude_bad and validity in ['Bad', 'Maybe']:
            continue

        if start_stop == 'Start':

            processed_activities.append({
                'label': label,
                'start_time': times[i],
                'notes': notes
            })
        
        elif start_stop == 'Stop' and processed_activities:

            processed_activities[-1]['end_time'] = times[i]

    return processed_activities

def segment_emg_data(
    emg_data,
    emg_times,
    activities
):
    """
    Segment EMG data based on activity start and end times.
    """

    segments = []

    for activity in activities:

        start_time, end_time = activity.get('start_time', None), activity.get('end_time', None)
        if start_time and end_time:

            idx = np.where((emg_times >= start_time) & (emg_times <= end_time))[0]
            segments.append({
                'label': activity['label'],
                'data': emg_data[idx],
                'start_time': start_time,
                'end_time': end_time
            })

    return segments

def save_data(
    data,
    filepath
):
    """
    Save data to a pickle file.
    """

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def main():

    args = get_args()
    source_filepath = args.get('source_file', './emg_data/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5')
    left_filepath = args.get('left_filepath', './emg_data/S04_left.pkl')
    right_filepath = args.get('right_filepath', './emg_data/S04_right.pkl')
    
    # Left
    emg_data, emg_times = read_hdf5_data(source_filepath, 'myo-left', 'emg')
    activities, activities_times = read_activity_data(source_filepath, 'experiment-activities', 'activities')
    processed_activities = process_activities(activities, activities_times)
    segments = segment_emg_data(emg_data, emg_times, processed_activities)
    save_data(segments, left_filepath)
    print(f'Left data successfully saved to {left_filepath}')

    # Right
    emg_data, emg_times = read_hdf5_data(source_filepath, 'myo-right', 'emg')
    activities, activities_times = read_activity_data(source_filepath, 'experiment-activities', 'activities')
    processed_activities = process_activities(activities, activities_times)
    segments = segment_emg_data(emg_data, emg_times, processed_activities)
    save_data(segments, right_filepath)
    print(f'Right data successfully saved to {right_filepath}')

if __name__ == '__main__':
    main()