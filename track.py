import pickle
import os

import numpy as np

from pytrack import measmodels, processmodels
from pytrack.density import Gaussian
from pytrack import radar_configuration
from dataloader import DataloaderRangeDoppler


def state_estimation():

    # Load experiment
    load_directory = './data/pulseON/RD'

    files = []
    for (dirpath, dirnames, filenames) in os.walk(load_directory):
        files += [os.path.join(dirpath, file) for file in filenames]
    
    # files = [os.path.join(load_directory, f) for f in os.listdir(load_directory)]
    dataloader = DataloaderRangeDoppler(files)

    for experiment in dataloader:

        time_step = experiment['time_step']
        n_time_steps = experiment['n_time_steps']
        sensor_measurements = experiment['sensor_measurements']

        # Define sensors and process model
        sensors = []
        sensor_positions = radar_configuration.pulseON_positions

        for position in sensor_positions:
            sensors.append(measmodels.RangeDopplerMeasurmentModel(sensor_bias = position, sigma_position=0.45, sigma_rangerate=0.45, state_space = 'CT'))
        process_model = processmodels.CoordinatedTurnModel(T = time_step, sigma_velocity = 1, sigma_turnrate = 15*np.pi/180)

        # Run tracker
        predicted_state = Gaussian(mean = np.array([[0],[0],[0],[0],[0]]), covariance = np.diag([50,50,50,2,2]))
        state_history = []
        for t in range(0,n_time_steps):
            print(f'Time step: {t}/{n_time_steps}')

            state = predicted_state
            for j, (sensor, measurements) in enumerate(zip(sensors, sensor_measurements)):
                measurement = Gaussian.ellipsoidalGating(measurements[t], predicted_state, sensor)
                if measurement is not None:
                    state = Gaussian.update(state, measurement, sensor, linearization=predicted_state.x)
                else:
                    print('Missed detection')
            
            state_history.append({'updated_state': state, 'predicted_state': predicted_state})
            predicted_state = Gaussian.predict(state, process_model)
    
        path = os.path.normpath(experiment['path'])
        save_path = path.split(os.sep)
        save_path[-3] = 'serialized'
        save_path[0] = '/' 
        save_path = os.path.join(*save_path[:-1])

        with open(os.path.join(save_path, experiment['name']+'.pickle'), 'wb') as f:
            pickle.dump((experiment, state_history, experiment['lables'], sensors, process_model), f)

if __name__ == '__main__':
    state_estimation()
