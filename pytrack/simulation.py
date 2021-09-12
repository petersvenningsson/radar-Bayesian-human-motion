import numpy as np
from matplotlib import pyplot as plt

import processmodels
import utils
import measmodels
import render
from density import Gaussian
import radar_configuration


def simulate_ct_radar():
    time_steps = 50
    process_model = processmodels.CoordinatedTurnModel( T = 0.2, sigma_velocity = 1, sigma_turnrate = 30*np.pi/180)

    # Define ground truth movement
    left_turn = utils.generate_ground_truth(process_model, initial_state = np.array([[-2],[0],[1.5],[np.pi/3],[-0.9]]), n_steps=int(time_steps - 20))
    right_turn_initial = left_turn[-1]
    right_turn_initial[-1] = -2
    right_turn = utils.generate_ground_truth(process_model, initial_state = right_turn_initial, n_steps=int(10))
    straight_initial = right_turn[-1]
    straight_initial[-1] = 0
    straight = utils.generate_ground_truth(process_model, initial_state = straight_initial, n_steps=int(10))
    ground_truth = np.vstack((left_turn, right_turn, straight))

    sensors = []
    sensor_measurements = []

    # Define sensors
    sensor_positions = radar_configuration.pulseON_positions

    for position in sensor_positions:
        sensors.append(measmodels.RangeDopplerMeasurmentModel(sensor_bias = position, sigma_position=0.3, sigma_rangerate=0.1, state_space = 'CT'))

    for sensor in sensors:
        sensor_measurements.append(utils.measurement_sequence(sensor, ground_truth))

    state = Gaussian(mean = np.array([[0],[0],[0],[0],[0]]), covariance = np.diag([50,50,50,2,2]))
    state_history = []
    for t in range(time_steps):
        for sensor, measurements in zip(sensors,sensor_measurements):
            state = Gaussian.update(state, measurements[t], sensor)
        state_history.append({'updated_state': state})
        state = Gaussian.predict(state, process_model)

    figure = render.base_figure()
    figure = render.render_ground_truth(figure, ground_truth, process_model)
    figure = render.render_estimate(figure, state_history, render_cov=True)
    figure = render.render_estimates_quiver(figure, state_history, process_model)

    for position in sensor_positions:
        plt.plot(position[0], position[1], marker = 'H', color = 'black', figure = figure)
    
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    simulate_ct_radar()