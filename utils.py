from copy import deepcopy

import numpy as np
from scipy.stats import multivariate_normal

def get_angle_estimates(state, sensor_positions, n=50000, n_bins = 4):
    state = deepcopy(state)
    if state.x[2] < 0:
        state.x[3] = state.x[3] + np.pi

    state_samples = multivariate_normal.rvs(state.x.squeeze(), state.P, size=n)
    estimates = []
    for sensor in sensor_positions:
        estimates.append(estimate_angle(state_samples, sensor, n_bins))
    return estimates


def estimate_angle(state_samples, sensor, n_bins):

    sensor_pos = sensor - state_samples[:, 0:2]
    state_samples[:, 3] = np.mod(state_samples[:, 3], 2*np.pi)

    # get angle from target to sensor.
    sensor_angle = np.mod(np.arctan2(sensor_pos[:,1], sensor_pos[:,0]), 2*np.pi) # Note that first argument is y axis

    # Angle between sensor and heading angle
    viewing_angle = np.mod(sensor_angle - state_samples[:, 3], 2*np.pi)

    # Discretization
    bin_size = np.pi/n_bins # bin resolution 2 times higher to enable wrapping bins around 0
    bins = [ bin_size * i for i in range(n_bins * 2 + 1) ]

    binned_values = list(np.digitize(viewing_angle, bins))
    counts = np.array([ binned_values.count(i+1) for i in range(n_bins*2) ])

    estimates = counts/np.sum(counts)
    estimates = np.roll(estimates, shift=1)
    estimates = np.array([ estimates[i] + estimates[i+1] for i in range(0, estimates.shape[0], 2) ])

    return estimates

def get_centriod(spectrogram, doppler_bins):
    spectrogram = np.power(10, spectrogram/20)
    n_time_bins = spectrogram.shape[1]
    index_axes = np.repeat(doppler_bins[:,np.newaxis], n_time_bins, axis=1)
    centroid = np.sum(spectrogram * index_axes, axis=0)/np.sum(spectrogram, axis=0)
    return centroid
