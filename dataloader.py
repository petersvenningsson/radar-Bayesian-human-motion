import os
import pickle

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import IncrementalPCA as PCA

import utils


decode_lable = { 1: 'Walking', 2: 'Standing', 3: 'Sitting down', 4:'Sitting', 5: 'Standing up',
     6: 'Bending (Sitting)', 7: 'Falling', 8: 'Lying down', 9:'Bending (Standing)', 0: 'N/A',
}


def redefine_classes(df):
    reduce_class_set = {0:0, 1: 1, 2: 2, 3: 3, 4:5, 5: 6, 6:9, 7:7, 8:5, 9:7}

    df['lable'] = df.apply(lambda row: reduce_class_set[row['lable']], axis=1)
    # Drop zero (clutter) class
    df.drop(df.loc[df['lable'] == 0].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # unpack stationary class into standing, sitting, lying
    lable_to_dynamic_mode = {1:2, 5:2, 3:4, 7:8} # 2: standing, 3: Sitting, 8: Lying on ground
    for j, file in enumerate(df.file_index.unique()):
        seq_df = df.loc[df['file_index'] == file]
        
        dynamic_mode = 2
        for time in np.sort(seq_df.time.unique()):
            df_step = seq_df[seq_df['time'] == time]
            lable = df_step.head(n=1)['lable'].to_numpy()[0]
            try:
                dynamic_mode = lable_to_dynamic_mode[lable]
            except KeyError:
                pass
            if lable == 2:
                df.loc[df_step.index, 'lable'] = dynamic_mode

    return df


class DataloaderRangeDoppler():
    def __init__(self, files):
        self.index = -1
        self.files = files
        self.len = len(files)

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index < self.len:
            file_name = self.files[self.index]
            return self.read_file(file_name)
        else:
            raise StopIteration

        self.index += 1
        file = files


    def __len__(self):
        return len(self.files)


    def read_file(self, path):
        
        data = scipy.io.loadmat(path)
        data = self.downsample(data, step = 3)

        RD_collection = data['RD_maps']

        n_range_bins, n_doppler_bins, n_time_bins = RD_collection[0].shape       

        intervals = {'time_interval': data['time_interval'][0],
            'range_interval': data['range_interval'][0],
            'velocity_interval': [2.17, -2.17]
        }

        sensor_measurements = []
        for j, sensor_RD in enumerate(RD_collection):
            measurements = []
            for time_bin in range(n_time_bins):
                RD = sensor_RD[:,:, time_bin]
                measurements.append(self.extract_centroid(RD, intervals))

            sensor_measurements.append( measurements )
        time_step = (intervals['time_interval'][1] - intervals['time_interval'][0])/n_time_bins

        lables = data['lables'].squeeze()
        return {'sensor_measurements': sensor_measurements,
            'time_step': time_step,
            'n_time_steps': n_time_bins,
            'lables': lables,
            'name':  os.path.basename(path).split('.')[0],
            'path': path
        }


    def extract_centroid(self, RD, intervals):
        if RD.max() == 0.0:
            return None

        centroid = []
        quantities = ['range', 'velocity']
        for q, quantity in enumerate(quantities):
            bins = np.linspace(intervals[f'{quantity}_interval'][0], intervals[f'{quantity}_interval'][1], num = RD.shape[q])
            centroid.append(np.sum( RD.sum(axis = int(not q)) * bins ) / RD.sum())
        return np.array(centroid)


    def downsample(self, data, step):
        downsampled_collection = []
        RD_collection = data['RD_maps'].squeeze()
        for RD in RD_collection:
            RD = RD[:,:,::step]
            downsampled_collection.append(RD)
        
        data['RD_maps'] = downsampled_collection
        data['lables'] = data['lables'][:,::step]
        return data


class DataLoaderSpectrogram():
    def __init__(self, render=False, debug=False):
        self.file_index = 0 
        self.debug = debug
        self.render = render


    def build(self, dataset_path, information, downsample_step=5, window = 1):
        """
            files (list of str): File_names of experiment sequences
            sample_time (list of list of float): Sampling times
        """
        self.downsample_step = downsample_step
        self.dataset_path = dataset_path
        self.window = window

        self.synchronize_RD_spec()
        self.build_dataframe(information)


    def load(self, path_df):
        self.df = pickle.load( open( path_df, 'rb' ) )


    def synchronize_RD_spec(self):
        """ Sets sampling_times, self.files, lables, angle
        """
        self.files = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.dataset_path, 'spectrograms')):
            self.files += [os.path.join(dirpath, file) for file in filenames]

        self.sampling_times = []
        self.lables = []
        self.state_histories = []

        for file in self.files:
            
            file = os.path.normpath(file)
            RD_path = file.split(os.sep)
            RD_path[-3] = 'serialized'
            RD_path[0] = '/' 
            RD_path = os.path.join(*RD_path[:-1])
            file_name = os.path.split(file)[1].split('.')[0]

            rd_file = os.path.join(RD_path, file_name + '.pickle')

            with open(rd_file, 'rb') as f:
                RD = pickle.load(f)

            time_step = RD[0]['time_step']
            n_steps = RD[0]['n_time_steps']
            lables = RD[0]['lables']
            state_history = RD[1]
            self.sensor_positions = [ sensor.sensor_bias for sensor in RD[3]]

            time_steps = [i*time_step for i in range(n_steps)]
            lables, time_steps, state_history, time_step = self.downsample(lables, time_steps, state_history, self.downsample_step)

            self.sampling_times.append(time_steps)
            self.lables.append(lables)
            self.state_histories.append(state_history)


    def downsample(self, lables, time_steps, state_history, step):
        lables = lables[::step]
        time_steps = time_steps[::step]
        state_history = state_history[::step]
        time_step = time_steps[1] - time_steps[0]
        return lables, time_steps, state_history, time_step


    def build_dataframe(self, information):
        if information == 'spectrograms':
            samples = []
            for file, times, lable, state_history in zip(self.files, self.sampling_times, self.lables, self.state_histories):
                samples.extend(self.unpack_spectrograms(file, times, lable, state_history, flatten=False))
            if self.render:
                print("done!")
                raise RuntimeError
            self.df = pd.DataFrame(samples)

        if information == 'PCA':
            samples = []
            for i_file, (file, times, lable, state_history) in enumerate(zip(self.files, self.sampling_times, self.lables, self.state_histories)):
                print(f'at file {i_file}/{len(self.files)}')
                samples.extend(self.unpack_spectrograms(file, times, lable, state_history, flatten = True))

            df = pd.DataFrame(samples)
            df = redefine_classes(df)
            pca = PCA(n_components=20, batch_size=2000)

            _df = df.groupby('lable', group_keys=False).apply(lambda x: x.sample(len(df), replace=True))

            pca.fit(np.vstack(_df['spectrogram'].to_numpy()))
            components = pca.transform(np.vstack(df['spectrogram'].to_numpy()))
            for i, component in enumerate(components.T):
                df[f"PC_{i}"] = component

            df.drop(['spectrogram'], axis = 1, inplace=True)
            self.df = df


    def unpack_spectrograms(self, file, sampling_times, lables, state_history, flatten=False):
        radars = [f'sensor_{i}' for i in range(5)]
        data = scipy.io.loadmat(file)
        self.frequency_axis = data['frequencies']
        time_axis = data['time']
        spectrograms = data['spectrograms']
        n_bins = 8
        samples = []
        
        for time, lable, state in zip(sampling_times, lables, state_history):
            
            predicted_angles = utils.get_angle_estimates(state['predicted_state'], self.sensor_positions, n_bins=n_bins)
            updated_angles = utils.get_angle_estimates(state['updated_state'], self.sensor_positions, n_bins=n_bins)
            
            for r, (radar, updated_angle, predicted_angle) in enumerate(zip(radars, updated_angles, predicted_angles)):
                spectrogram = Spectrogram(spectrograms[:,:,r], self.frequency_axis, time_axis)
                try:
                    sample, _ = spectrogram.get_sample(time, window_time=self.window)
                    sample = spectrogram.mean_shift(sample)
                except IndexError as E:
                    print(E)
                    continue

                if flatten:
                    sample = sample.flatten()
                samples.append({'radar': radar, 'spectrogram': sample, 'lable': lable, 'angle': updated_angle, 'predicted_angle': predicted_angle,
                    'person': file.split('.')[0].split('_')[-1], 'sequence_type': file.split('.')[0].split('_')[-2],
                    'file_index': self.file_index, 'time': time,
                })

        self.file_index += 1
        return samples


class Spectrogram():
    def __init__(self, spectrogram, freq_axis, time_axis):
        
        self.spectrogram = spectrogram
        self.freq_axis = freq_axis.squeeze()
        self.time_axis = time_axis.squeeze()
        self.dt = self.time_axis[1] - self.time_axis[0]
        self.threshold = -80

    
    def _get_sample(self, time, window_time=2):
        start_time = (time - window_time/2)

        if start_time < 0:
            raise IndexError(f'Sample start time before start of sequence, start time {start_time}')

        start_index = np.abs(self.time_axis - start_time).argmin()
        end_index = start_index + int(window_time/self.dt)
        sample, axis = self.spectrogram[:, range(start_index, end_index)], self.time_axis[start_index:end_index]
        return sample, axis


    def get_sample(self, time, window_time=2):

        sample, axis = self._get_sample(time, window_time)

        # Normalize on a windows of 4 seconds
        try:
            larger_sample, _ = self._get_sample(time, window_time=4)
        except IndexError:
            try:
                larger_sample, _ = self._get_sample(time+3, window_time=4)
            except IndexError:
                try:
                    larger_sample, _ = self._get_sample(time-3, window_time=4)
                except IndexError:
                    raise IndexError('Could not sample the spectrogram for normalization')
        
        sample = sample - larger_sample.max()
        sample[sample < self.threshold] = self.threshold

        return sample, axis


    def mean_shift(self, sample):
        n_freq_bins, n_time_bins = sample.shape
        index_axis = np.array(range(n_freq_bins))

        doppler_centroid = utils.get_centriod(sample, index_axis)
        mean_centroid = np.mean(doppler_centroid).astype(np.int)

        center_bin = np.abs(self.freq_axis).argmin()
        shift = center_bin - mean_centroid

        # Roll the array
        sample = np.roll(sample, shift, axis=0)

        if shift > 0: 
            sample[:shift,:] = self.threshold
        elif shift < 0:
            sample[shift:,:] = self.threshold

        return sample